package agent

import (
	"errors"
	"sync"

	"github.com/dpoage/llmkit"
)

// ErrSteeringInUse is returned by [Runner.Run] and [Runner.RunJSON] when the
// [Steering] given via [WithSteering] is already bound to another active
// run.
var ErrSteeringInUse = errors.New("agent: steering handle already bound to an active run")

// steerKind marks which kind of turn a queued turn is. Both kinds share one
// FIFO so a would-be-finish drain delivers turns in enqueue order across
// them; the kinds differ only in where the loop is willing to deliver.
type steerKind int

const (
	kindSteer steerKind = iota
	kindFollowUp
)

// deliveredTurn is one steering turn leaving the queue: the message the loop
// appends, and whether it was queued as a follow-up — the followUp fact the
// Runner stamps on the run's Steer event.
type deliveredTurn struct {
	msg      llmkit.Message
	followUp bool
}

// queuedTurn is one pending user turn: its kind and the message the loop
// appends when it drains.
type queuedTurn struct {
	kind    steerKind
	message llmkit.Message
}

// Steering queues user turns for injection into a running [Runner] loop.
//
// [Steering.Steer] delivers before the next model call, after the current
// turn's tool results; [Steering.FollowUp] delivers when the run would
// otherwise finish. Each call queues ONE user turn built like the seeded
// task turn of [Attach] with an empty task: the blocks in the order given,
// no text block added; a call with no blocks queues a single empty text
// turn. Both methods are safe for concurrent use; the run drains the queue
// on its own goroutine. The would-be-finish drain delivers steers and
// follow-ups together in enqueue order; the pre-completion boundary drains
// only steers, so a follow-up queued before a later steer delivers at the
// finish, after that steer. Any drain that delivers at least one turn
// continues the loop instead of ending it.
//
// Queued turns are ordinary user messages: the transcript records them as
// request messages and [Outcome.Messages] includes them. Limits apply
// unchanged, so a run stopped by [Limits.MaxIterations],
// [Limits.TokenBudget], or a [BudgetPool] leaves undelivered turns queued
// — [Steering.Pending] reports them, and [Continue] with the SAME Steering
// delivers pending steers before the continued run's first completion and
// pending follow-ups at its first would-be finish. A refusal stop
// ([StopReasonError]) returns before the would-be-finish drain: queued
// turns stay pending and the refusal is never papered over. A steer already
// queued delivers at the pre-completion boundary, before the refusing
// completion. At an empty turn the queued content replaces the synthetic
// nudge and does not consume a nudge attempt (see [maxEmptyTurnNudges]).
//
// A Steering serves one run at a time. Passing it to a second concurrent
// run fails that run with [ErrSteeringInUse]; sequential runs —
// including [Continue] chains — rebind cleanly. Steer and FollowUp queue
// while no run is bound; those turns deliver on the next run bound to the
// handle. [Runner.RunJSON] honors both drain points with no special case:
// the JSON parse applies to the last completion, so a turn queued after a
// parseable answer simply continues the run.
type Steering struct {
	mu    sync.Mutex
	queue []queuedTurn
	bound bool
}

// NewSteering returns an empty steering handle.
func NewSteering() *Steering { return &Steering{} }

// Steer queues blocks as one user turn delivered before the next model
// call, after the current turn's tool results. See [Steering].
func (s *Steering) Steer(blocks ...llmkit.Block) { s.enqueue(kindSteer, blocks) }

// FollowUp queues blocks as one user turn delivered when the run would
// otherwise finish. See [Steering].
func (s *Steering) FollowUp(blocks ...llmkit.Block) { s.enqueue(kindFollowUp, blocks) }

// Pending returns the number of queued turns not yet delivered to the model,
// steers and follow-ups combined.
func (s *Steering) Pending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.queue)
}

// WithSteering binds s to a run so [Runner.Run], [Runner.RunJSON], and
// [Runner.RunJSONAs] drain it at the turn boundaries in [Steering]. A nil
// s runs without steering.
func WithSteering(s *Steering) RunOption {
	return func(c *runConfig) { c.steering = s }
}

// enqueue appends one turn of the given kind, built with the [Attach]
// task turn's shape rules so the empty-text rule stays in [taskTurn].
func (s *Steering) enqueue(kind steerKind, blocks []llmkit.Block) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.queue = append(s.queue, queuedTurn{kind: kind, message: taskTurn("", blocks)})
}

// bind claims the handle for one run. A second concurrent claim fails with
// [ErrSteeringInUse]; the caller defers [Steering.unbind].
func (s *Steering) bind() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.bound {
		return ErrSteeringInUse
	}
	s.bound = true
	return nil
}

// unbind releases the handle so a later run can claim it.
func (s *Steering) unbind() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bound = false
}

// drainSteers removes and returns the queued steer turns in queue order,
// leaving follow-ups queued for the would-be-finish drain.
func (s *Steering) drainSteers() []deliveredTurn {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []deliveredTurn
	var kept []queuedTurn
	for _, t := range s.queue {
		if t.kind == kindSteer {
			out = append(out, deliveredTurn{msg: t.message})
		} else {
			kept = append(kept, t)
		}
	}
	s.queue = kept
	return out
}

// drainAll removes and returns every queued turn in enqueue order across
// both kinds — the would-be-finish drain.
func (s *Steering) drainAll() []deliveredTurn {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.queue) == 0 {
		return nil
	}
	out := make([]deliveredTurn, len(s.queue))
	for i, t := range s.queue {
		out[i] = deliveredTurn{msg: t.message, followUp: t.kind == kindFollowUp}
	}
	s.queue = nil
	return out
}

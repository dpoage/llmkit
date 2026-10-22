package decide

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"sync"

	"github.com/dpoage/llmkit"
)

// ErrReplayDiverged is wrapped by a [Replay] Decider's error when an Ask has
// no recorded decision to match, or the matched record does not answer the
// asked questions. Match it with errors.Is.
var ErrReplayDiverged = errors.New("decide: replay diverged")

// ErrReplayedFailure is wrapped by a [Replay] Decider's error when the
// matched recorded decision is a failed Ask. The error text carries the
// recorded failure's text; the live error's type and chain are not
// recorded, so a replayed failure has no root llmkit sentinel. Match it
// with errors.Is.
var ErrReplayedFailure = errors.New("decide: replayed failed decision")

// Replay returns a Decider that answers from the [llmkit.KindDecision]
// events src holds for run, with no backend call: the policy half of a
// deterministic replay, the counterpart of the agent package's replay
// client. It returns src's error (wrapping [llmkit.ErrUnknownRun] when src
// has no record of run). A run that recorded no decisions is valid; every
// Ask against it diverges.
//
// Each Ask follows these rules:
//
//  1. [Normalize] runs first. A refusal is returned unchanged and consumes
//     no record.
//  2. The Ask matches the earliest unconsumed recorded decision with the
//     same Step ([llmkit.StepFromContext] of the Ask's context equals the
//     event's Step) and the same set of question ids. The match consumes
//     the record, so two Asks of one step with different id sets match
//     whichever order they arrive in, and two Asks with the same id set
//     match in recorded order.
//  3. With no match, Ask returns the zero Response and an error wrapping
//     [ErrReplayDiverged] that names the step.
//  4. A matched record with Err set returns the zero Response and an error
//     wrapping [ErrReplayedFailure] with the recorded text.
//  5. Otherwise Ask rebuilds the Response (Model, Usage, and the Nouls,
//     Choices, and Scores of the recorded answers, by the kind of the
//     asked question) and returns it if it passes [Check] against the
//     questions; a Check failure wraps [ErrReplayDiverged].
//
// Replay compares neither the state nor the Backend: a replayed Ask may
// judge a different state than the recorded one and still receives the
// recorded answer. A caller that needs to know the state changed must
// compare it itself. Recorded decisions the replay never asks for are not
// reported; a replay that stops early is not an error here. A Decider from
// Replay is safe for concurrent use.
func Replay(src llmkit.Source, run llmkit.RunID) (Decider, error) {
	events, err := src.Events(context.Background(), run)
	if err != nil {
		return nil, fmt.Errorf("decide: replay run %q: %w", run, err)
	}
	r := &replayer{}
	for i := range events {
		ev := &events[i]
		if ev.Kind != llmkit.KindDecision || ev.Decision == nil {
			continue
		}
		ids := make([]string, len(ev.Decision.Questions))
		for j, q := range ev.Decision.Questions {
			ids[j] = q.ID
		}
		sort.Strings(ids)
		r.records = append(r.records, &replayRecord{step: ev.Step, ids: ids, dec: ev.Decision})
	}
	return r, nil
}

type replayRecord struct {
	step int
	ids  []string // sorted
	dec  *llmkit.DecisionEvent
	used bool
}

type replayer struct {
	mu      sync.Mutex
	records []*replayRecord
}

func (r *replayer) Ask(ctx context.Context, state any, qs Questions) (Response, error) {
	if _, _, err := Normalize(state, qs); err != nil {
		return Response{}, err
	}
	step := llmkit.StepFromContext(ctx)
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	rec := r.take(step, ids)
	if rec == nil {
		return Response{}, fmt.Errorf("%w: step %d: no unconsumed recorded decision for questions %q", ErrReplayDiverged, step, ids)
	}
	if rec.dec.Err != "" {
		return Response{}, fmt.Errorf("%w: step %d: %s", ErrReplayedFailure, step, rec.dec.Err)
	}
	resp, err := rebuild(rec.dec, qs)
	if err != nil {
		return Response{}, fmt.Errorf("%w: step %d: %v", ErrReplayDiverged, step, err)
	}
	return resp, nil
}

// take consumes and returns the earliest unconsumed record at step with
// exactly the question ids ids (sorted), or nil.
func (r *replayer) take(step int, ids []string) *replayRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, rec := range r.records {
		if !rec.used && rec.step == step && slices.Equal(rec.ids, ids) {
			rec.used = true
			return rec
		}
	}
	return nil
}

// rebuild converts a recorded decision's answers back into a Response by
// the kind of each asked question, then checks it against qs.
func rebuild(dec *llmkit.DecisionEvent, qs Questions) (Response, error) {
	resp := Response{Model: dec.Model, Usage: dec.Usage}
	for _, a := range dec.Answers {
		switch qs[a.ID].(type) {
		case Noul:
			if a.Belief != nil {
				if resp.Nouls == nil {
					resp.Nouls = map[string]float64{}
				}
				resp.Nouls[a.ID] = *a.Belief
			}
		case Choice:
			if resp.Choices == nil {
				resp.Choices = map[string]ChoiceAnswer{}
			}
			resp.Choices[a.ID] = ChoiceAnswer{Choice: a.Choice, Probabilities: a.Probabilities, Confidence: a.Confidence}
		case Score:
			if a.Score != nil {
				if resp.Scores == nil {
					resp.Scores = map[string]ScoreAnswer{}
				}
				resp.Scores[a.ID] = ScoreAnswer{Score: *a.Score, Levels: a.Levels, Probabilities: a.LevelProbabilities, Confidence: a.Confidence}
			}
		default:
			return Response{}, fmt.Errorf("answers[%q]: answer for an unknown question id", a.ID)
		}
	}
	if err := Check(qs, resp); err != nil {
		return Response{}, err
	}
	return resp, nil
}

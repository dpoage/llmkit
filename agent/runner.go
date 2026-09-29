package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/dpoage/llmkit"
	"slices"
	"strings"
	"time"
)

// Runner drives an [llmkit.Client] through a tool-call loop. Construct one per
// agent role with that role's system prompt and tool set, then call
// [Runner.Run], [Runner.RunJSON], or [Runner.RunJSONAs] per task.
//
// A Runner is safe for concurrent use: every field is fixed at construction,
// and all mutable state (conversation, transcript, budget accounting,
// compaction thresholds, nudge counters) is local to a single Run call.
// Tools inherit an obligation from this: a [Tool] passed to one Runner may be
// invoked concurrently (see [Tool.Run] and [WithParallelTools]), so it must
// be safe for concurrent calls. Observability hooks (WithHooks) fire from
// whichever goroutine reaches the event — concurrently under
// WithParallelTools or concurrent Run calls — so hook functions must
// synchronize their own state.
type Runner struct {
	client       llmkit.Client
	tools        toolSet
	systemPrompt string
	limits       Limits

	// observers are the durable event sinks, in registration order (see
	// [WithObserver]); every run's chain fans out to the in-memory transcript
	// first and then to each of these.
	observers []llmkit.Observer
	// maxTokens caps output tokens per completion (see [WithMaxTokens]); zero
	// lets the adapter apply its own default.
	maxTokens int
	// hooks holds every [WithHooks] registration, in registration order (see
	// [Hooks]); the fire helpers in record.go walk it.
	hooks []Hooks
	// requestPolicy, when non-nil, shapes every completion request (see
	// [RequestPolicy] and [WithRequestPolicy]).
	requestPolicy RequestPolicy
	// toolTimeout, when positive, is the per-call deadline applied to every
	// Tool.Run (see WithToolTimeout).
	toolTimeout time.Duration
	// parallelTools, when true, dispatches one turn's tool calls concurrently
	// (see WithParallelTools).
	parallelTools bool
	// budgetPool, when non-nil, caps spend across runs (see WithBudgetPool);
	// nil means unlimited.
	budgetPool *BudgetPool
	// toolPolicy, when non-nil, gates every model-requested tool call (see
	// WithToolPolicy); nil allows all calls.
	toolPolicy ToolPolicy
}

// Option configures a [Runner] at construction.
type Option func(*Runner)

// WithLimits sets the per-run iteration and token limits. Zero fields resolve
// to package defaults.
func WithLimits(l Limits) Option {
	return func(r *Runner) { r.limits = l }
}

// WithObserver appends obs to the Runner's event sinks: every [llmkit.Event]
// the Runner emits — start, completion, tool runs, compaction, steering,
// finalize — fans out to the in-memory [Transcript] first and then to each
// WithObserver sink in registration order (see [llmkit.Observers], whose
// semantics apply: a nil Observer interface value is skipped, and a panic in
// one sink propagates at once, so later sinks do not see that event). The
// transcript always backs [Outcome.Transcript]; the sinks are the durable
// record and side channels (spend ledgers, metrics, traces).
//
// Emission happens on the loop goroutine, so obs must be safe for concurrent
// use only because concurrent Run calls on one Runner are allowed — each run
// emits from its own goroutine ([JSONL] is safe). A sink reports its own
// failures through the callback it was constructed with ([JSONL]) and never
// fails the run. The Runner does not enforce one durable history: register at
// most one sink that records the run's history. A second [JSONL] on a
// different directory writes a second copy of every run; one on the same
// directory records nothing, because its exclusive create of each run's file
// fails, and it reports that refusal through its onErr.
func WithObserver(obs llmkit.Observer) Option {
	return func(r *Runner) { r.observers = append(r.observers, obs) }
}

// WithMaxTokens caps output tokens per completion. Zero uses the adapter
// default.
func WithMaxTokens(n int) Option {
	return func(r *Runner) { r.maxTokens = n }
}

// WithBudgetPool makes the Runner share pool across its runs: the Runner
// checks the pool once per main-loop turn (when the check returns
// ErrBudgetExhausted, Run stops the run with TruncBudgetPool and reports it
// as an [*IncompleteError]) and
// charges it after every successful completion with that completion's
// llmkit.Usage.ChargeableTokens
// (CacheReadWeight-discounted). Continuation, finalization, and repair
// completions are charged without a fresh check. A nil pool — the zero
// default — is unlimited: no check, no charge. The pool may be shared by
// any number of concurrently running Runners; spend external to the loop
// (other processes, other clients) can still be recorded with BudgetPool.Add.
func WithBudgetPool(pool *BudgetPool) Option {
	return func(r *Runner) { r.budgetPool = pool }
}

// NewRunner builds a Runner bound to client, the given tools, and a system
// prompt. Options such as [WithLimits], [WithMaxTokens], [WithHooks], and
// [WithObserver] tune limits, output token caps, callbacks, and event sinks.
//
// NewRunner panics if tools contains a nil entry or two tools whose
// Def().Name is the same: the model addresses a tool by name alone, so a
// duplicate would advertise one tool's schema and dispatch another. The panic
// value is a string naming the tool index (nil) or the duplicated name and
// both indexes.
func NewRunner(client llmkit.Client, tools []Tool, systemPrompt string, opts ...Option) *Runner {
	r := &Runner{
		client:       client,
		tools:        newToolSet(tools),
		systemPrompt: systemPrompt,
	}
	for _, opt := range opts {
		opt(r)
	}
	r.limits = r.limits.resolve()
	return r
}

// Run executes the tool loop for a single task: it seeds the conversation
// with the task as a user message, then repeatedly calls the model and
// executes any requested tools until the run ends.
//
// Pass [Continue] to run the task inside a prior conversation instead of a
// fresh one.
//
// Pass [Attach] to carry image or document blocks on the task turn.
//
// Pass [WithSteering] to inject queued user turns while the run is in
// flight ([Steering]).
//
// A truncated run returns an [*IncompleteError] next to the Outcome, whose
// [Outcome.TruncationReason] is non-empty; the error's Outcome is that same
// pointer, and [Continue] accepts it. A failure returns its own non-nil
// error next to the Outcome, such as a failed completion, a [RequestPolicy]
// error, [ErrSteeringInUse], context cancellation, or [StopReasonError]. The
// returned Outcome's Transcript is always non-nil, even on error, capturing
// whatever happened before the failure.
//
// Max-tokens continuation: when a turn stops at the output token cap
// (StopMaxTokens) with no tool calls, Run makes ONE extra continuation
// completion that turn. It appends a user turn that asks the model to
// continue from exactly where it stopped, then appends the continuation as
// its own assistant turn. The stitched text reaches the caller in
// [Outcome.FinalText] and the returned response; the conversation history
// keeps both assistant turns, separated by that continuation nudge. This
// applies to plain Run, not only RunJSON. It costs at most one
// additional completion per truncated turn and is reflected in the
// Outcome's Iterations and Usage.
//
// When the continuation also stops at the cap without a tool call and the
// loop has no queued [Steering] turn to deliver after it, the run ends with
// [TruncOutputCap].
func (r *Runner) Run(ctx context.Context, task string, opts ...RunOption) (outcome *Outcome, err error) {
	var cfg runConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	ctx, em := r.begin(ctx, cfg, task)
	// The deferred emitFinalize emits the run's Finalize event when Run
	// returns, and with panicked set when a panic unwinds it (see
	// [emitFinalize]).
	returned := false
	defer func() { r.emitFinalize(ctx, em, outcome, err, !returned) }()
	outcome, err = r.run(ctx, em, cfg.seed, task, cfg.attach, "", nil, cfg.steering)
	if err == nil && outcome.TruncationReason != "" {
		err = &IncompleteError{Reason: outcome.TruncationReason, Outcome: outcome}
	}
	returned = true
	return outcome, err
}

// RunOption is a per-call option for [Runner.Run], [Runner.RunJSON], and
// [Runner.RunJSONAs]. Options apply in order; a later option wins, except
// [Attach], which accumulates.
type RunOption func(*runConfig)

// runConfig carries the resolved per-call options. It is unexported so new
// options never widen the exported surface.
type runConfig struct {
	// seed, when non-empty, continues a prior conversation (see [Continue]).
	seed []llmkit.Message
	// attach rides on the seeded task turn (see [Attach] and [taskTurn]).
	attach []llmkit.Block
	// steering, when non-nil, drains queued user turns at the loop's turn boundaries (see [Steering]).
	steering *Steering
	// runID, when non-empty, pins the run's [llmkit.RunID] (see [WithRunID]).
	runID llmkit.RunID
	// parentRunID is the id of the run this one continues (see [Continue]);
	// empty on a fresh run, rides on every event as ParentRunID.
	parentRunID llmkit.RunID
}

// WithRunID pins the run's [llmkit.RunID] to id: every event of the run
// carries it, a [JSONL] sink names its file after it, and [Continue] chains
// record it as the next run's ParentRunID. Use it when a caller generates
// stable identifiers up front (a store row's key) and must recover the run's
// records by that identifier later. An empty id is a no-op: the Runner mints
// one with [llmkit.NewRunID].
//
// The id must be a safe filename component — non-empty, not "." or "..",
// no path separators, no NUL — because the durable sink names the run's file
// exactly "<RunID>.jsonl". The task text is not part of the name: it rides
// the run's Start event.
func WithRunID(id llmkit.RunID) RunOption {
	return func(c *runConfig) { c.runID = id }
}

// Continue makes the run CONTINUE a prior conversation instead of reseeding
// one: prev.Messages (the Messages field of an earlier Run/RunJSON call's
// Outcome on this same Runner) becomes the starting history, and task is
// appended as a NEW user turn rather than becoming the conversation's sole
// seed message. A nil prev, or a prev with no Messages, degrades to plain
// reseeding — identical to a run without Continue — so a first-turn caller
// can pass it unconditionally without a nil check.
//
// The continued run is linked to its predecessor: prev.RunID rides on every
// event of the new run as [llmkit.Event].ParentRunID (and on
// Transcript.ParentRunID), so a sink can reconstruct the whole Continue
// lineage. A prev whose RunID is empty simply starts an unlinked run.
//
// The same-Runner contract: prev must come from a run of THIS Runner — the
// system prompt and the tool set must match the seed — and Limits apply per
// call: iteration and token budgets are re-armed fresh for each continued
// run (a [BudgetPool] installed with [WithBudgetPool] already spans runs).
//
// If the seed's final assistant turn carries tool calls that were never
// answered (the history a context-cancelled run returns), that trailing
// turn is dropped before the run starts — only the seed's last assistant
// turn is checked, and everything from it to the end of the seed is
// dropped — so the wire request is always well-formed. Trimming can leave
// two consecutive user turns on the wire (the cancelled tool turn is
// dropped whole, so nothing answers between them); the in-tree adapters
// send consecutive user messages as-is and the providers accept or coalesce
// them.
//
// Like Run, there is no context-window management: a caller driving a long
// conversation must bound the history itself, using the client's
// [llmkit.Capabilities].ContextWindow and [EstimateHistoryTokens].
//
// When an earlier call returned [StopReasonError] (model refusal/safety
// stop), the attached err.Outcome may be threaded back in here: the refusal
// turn stays in the history and the conversation proceeds from it.
func Continue(prev *Outcome) RunOption {
	return func(c *runConfig) {
		if prev != nil {
			c.seed = prev.Messages
			c.parentRunID = prev.RunID
		}
	}
}

// run is the shared loop body. seed, when non-empty, is a prior conversation
// to continue: task is appended as a NEW user turn onto seed instead of
// becoming the conversation's sole seed message. Run and RunJSON pass seed ==
// nil (reseed every call); [Continue] seeds a prior Outcome's Messages so the
// next round lands in the SAME conversation as the one that produced it.
// task and attach together form that seeded task turn (see [taskTurn]):
// [Attach]'s blocks ride on it and nowhere else. Before the task is appended,
// a seed whose trailing assistant turn carries unanswered tool calls is
// trimmed — see [trimDanglingToolTurn].
//
// finalizePrompt, when non-empty, enables forced finalization: when the
// iteration cap, the per-run token budget, or the shared budget pool stops
// the run, the loop injects this user-role message and takes a single final
// tool-less completion so the model can emit its answer instead of dangling
// exploration prose or a silently empty output. RunJSON passes a
// JSON-demanding prompt; the public Run passes "" and therefore never pays
// the extra turn.
//
// responseSchema, when non-nil, is the JSON Schema for the final answer,
// attached to every completion in the run (capability-gated; see
// [complete]). The public Run passes nil; RunJSON passes its schema.
//
// steering, when non-nil, delivers queued user turns at two drain points
// ([Steering]): steers before every completion — below the limit and
// context checks, so a run stopped by a limit leaves them queued;
// steers and follow-ups together, in enqueue order, whenever a completion
// returns no tool calls, continuing the loop instead of breaking. The
// bind at entry refuses a second concurrent run on the same handle
// with [ErrSteeringInUse].
//
// maxEmptyTurnNudges bounds how many times run() will nudge a model that
// produced neither a tool call nor visible text (after stripping reasoning
// <think> blocks) back into the loop before giving up. Some reasoning models
// emit an assistant turn that is ONLY an inline think block — stop=end_turn,
// zero tool calls — which would otherwise hand RunJSON unparseable empty text
// and burn its single repair. The cap stops a persistently silent model from
// looping forever. The count is per run and is never reset; see
// [TruncNoAnswer].
const maxEmptyTurnNudges = 2

// emptyTurnNudge is the text of the user turn run() appends as an empty-turn
// nudge: another chance for the model to call a tool or emit its final
// answer. See [maxEmptyTurnNudges].
const emptyTurnNudge = "You made no tool call and produced no final answer. Continue: call a tool or emit your final answer now."

func (r *Runner) run(ctx context.Context, em runEmitter, seed []llmkit.Message, task string, attach []llmkit.Block, finalizePrompt string, responseSchema json.RawMessage, steering *Steering) (*Outcome, error) {
	s := &runState{
		outcome:          &Outcome{Transcript: em.tr, RunID: em.tr.RunID},
		finalizePrompt:   finalizePrompt,
		responseSchema:   responseSchema,
		compactThreshold: r.limits.HistoryTokenBudget,
	}
	outcome := s.outcome
	// Bind before any work so a second concurrent run on the same handle
	// cannot interleave its turns into this run's queue.
	if steering != nil {
		if err := steering.bind(); err != nil {
			return outcome, err
		}
		defer steering.unbind()
	}

	taskMsg := taskTurn(task, attach)
	if len(seed) > 0 {
		seed = trimDanglingToolTurn(seed)
		s.messages = make([]llmkit.Message, 0, len(seed)+1)
		s.messages = append(s.messages, seed...)
		s.messages = append(s.messages, taskMsg)
	} else {
		s.messages = []llmkit.Message{taskMsg}
	}

	// Snapshot the conversation into the Outcome on every return path so a caller
	// that wants to continue this conversation ([Continue]) always has the latest
	// history available, even from a truncated or erroring run. s.messages is
	// reassigned throughout the loop; the deferred closure reads it by reference
	// at return time.
	defer func() { outcome.Messages = s.messages }()

	// History-compaction state. toolNameByID lets a tool-result stub name the
	// tool it answered; on a continued run the map starts from the seed's
	// assistant tool calls, so a PRIOR run's results also stub with their
	// real tool names instead of the generic fallback (see [compactStub]).
	s.toolNameByID = map[string]string{}
	for _, m := range s.messages {
		if m.Role == llmkit.RoleAssistant {
			for _, call := range m.ToolCalls {
				s.toolNameByID[call.ID] = call.Name
			}
		}
	}

	for {
		// Stop before the next turn if we've hit the iteration cap. The
		// finalizeAndTruncate helper below gives RunJSON its one reserved
		// finalization turn so a near-cap model can still emit its answer; the
		// public Run (finalizePrompt == "") is a no-op and proceeds straight
		// to the truncation mark.
		if r.limits.MaxIterations >= 0 && outcome.Iterations >= r.limits.MaxIterations {
			if err := r.finalizeAndTruncate(ctx, em, s, TruncMaxIterations); err != nil {
				return outcome, err
			}
			s.stop(TruncMaxIterations)
			break
		}
		// Stop before the next turn if we're already over budget. The budget
		// stop gets the same one reserved finalization turn the iteration cap
		// gets (RunJSON only), so a near-budget agent can emit its answer
		// instead of returning a silently empty result to the caller.
		if r.overBudget(outcome.Usage) {
			if err := r.finalizeAndTruncate(ctx, em, s, TruncTokenBudget); err != nil {
				return outcome, err
			}
			s.stop(TruncTokenBudget)
			break
		}
		// Consult the shared budget pool (if any) before issuing the next model
		// call, so a run already in flight stops at this turn boundary once the
		// run-spanning ceiling is hit rather than running to completion. This is a
		// read-only check: it does not touch System/Tools/Messages, so request
		// prefix stability is preserved.
		if r.budgetPool != nil && r.budgetPool.Check() != nil {
			// Shared pool exhausted: give the model one reserved finalization
			// turn (RunJSON only) so a near-budget agent can still emit its
			// answer before we classify the stop as TruncBudgetPool.
			if err := r.finalizeAndTruncate(ctx, em, s, TruncBudgetPool); err != nil {
				return outcome, err
			}
			s.stop(TruncBudgetPool)
			break
		}

		if err := ctx.Err(); err != nil {
			return outcome, err
		}

		// Pre-completion drain: deliver queued steers below the limit and
		// context checks, so a run stopped by a limit leaves its queue
		// intact (see [Steering]); follow-ups are never drained here.
		if steering != nil {
			if steered := steering.drainSteers(); len(steered) > 0 {
				for _, d := range steered {
					em.emit(ctx, steerEvent(ctx, d.msg, false, outcome.Iterations+1))
					s.messages = append(s.messages, d.msg)
				}
			}
		}

		// Threshold-triggered, one-shot-per-crossing history compaction. Done at
		// the turn boundary BEFORE the completion so the smaller history is what
		// gets billed this turn. Re-arming the threshold upward after a firing
		// keeps compaction bounded and avoids re-paying a prefix cache miss every
		// subsequent turn (see compactRearmFactor).
		s.messages, s.compactThreshold = r.maybeCompact(ctx, em, s.messages, s.compactThreshold, s.toolNameByID, outcome.Iterations+1)

		resp, err := r.completeOnce(ctx, em, &s.messages, outcome, responseSchema, false)
		if err != nil {
			return outcome, err
		}

		if len(resp.ToolCalls) == 0 {
			// StopError/StopRefusal/StopContentFilter mean the model stopped
			// for a provider-specific error reason (refusal, safety filter,
			// recitation). Breaking cleanly here would record refusal prose —
			// or stale FinalText from an earlier turn — as the answer.
			// Surface a typed error instead.
			if resp.StopReason == llmkit.StopError ||
				resp.StopReason == llmkit.StopRefusal ||
				resp.StopReason == llmkit.StopContentFilter {
				return outcome, &StopReasonError{StopReason: resp.StopReason, Text: resp.Text, Outcome: outcome}
			}

			// Would-be-finish drain: any queued turn — steer or follow-up,
			// in enqueue order — continues the loop instead of ending
			// it, and at an empty turn the queued content replaces the
			// synthetic nudge without consuming a nudge attempt. The
			// refusal check above returns before this drain, so a queued
			// turn never papers over a refusal stop (see [Steering]).
			if steering != nil {
				if drained := steering.drainAll(); len(drained) > 0 {
					for _, d := range drained {
						em.emit(ctx, steerEvent(ctx, d.msg, d.followUp, outcome.Iterations+1))
						s.messages = append(s.messages, d.msg)
					}
					continue
				}
			}
			if resp.StopReason == llmkit.StopMaxTokens {
				s.stop(TruncOutputCap)
				break
			}
			// A turn with no tool call and no visible text (after stripping
			// reasoning <think> blocks) is an empty/think-only turn, not a real
			// answer. Nudge the model to continue instead of treating the turn
			// as finished, up to maxEmptyTurnNudges times; the nudge turn goes
			// through the normal loop top so it bills and counts like any
			// other turn. A truncated, unclosed think block also strips to
			// empty and gets the same nudge.
			if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
				if s.emptyTurnNudges >= maxEmptyTurnNudges {
					s.stop(TruncNoAnswer)
					break
				}
				s.emptyTurnNudges++
				s.messages = append(s.messages, llmkit.TextMessage(llmkit.RoleUser, emptyTurnNudge))
				continue
			}
			break
		}

		// Execute the requested tool calls and feed results back. By default
		// the calls run sequentially in the model's order; with
		// WithParallelTools they run concurrently, but the results below are
		// appended in the MODEL's original call order either way, so the
		// wire-visible history is identical.
		executed := r.executeTools(ctx, outcome, resp.ToolCalls)
		for i, call := range resp.ToolCalls {
			if i >= len(executed) {
				// Context cancelled before this call was dispatched; the
				// ctx check below returns with the executed prefix kept.
				break
			}
			em.emit(ctx, toolRunEvent(ctx, call, executed[i], outcome.Iterations))
			s.toolNameByID[call.ID] = call.Name
			s.messages = append(s.messages, executed[i].message(call.ID))
		}
		if err := ctx.Err(); err != nil {
			return outcome, err
		}

		// After executing tools, check the budget again before looping so we
		// truncate promptly rather than issuing one more expensive completion.
		// A budget hit post-tool still gets the one reserved finalization turn
		// (RunJSON only) so a near-budget agent can emit its answer.
		if r.overBudget(outcome.Usage) {
			if err := r.finalizeAndTruncate(ctx, em, s, TruncTokenBudget); err != nil {
				return outcome, err
			}
			s.stop(TruncTokenBudget)
			break
		}
	}

	return outcome, nil
}

// runState is one run's mutable loop state, owned by [Runner.run] and passed
// to the helpers that need it. Everything the Runner is shared for lives on
// the Runner; everything that changes per run lives here.
type runState struct {
	// messages is the conversation history, reassigned as the loop appends and
	// compacts; run snapshots it into Outcome.Messages on every return path.
	messages []llmkit.Message
	// compactThreshold is the history-token threshold currently in force,
	// re-armed upward after each real prune (see [Runner.maybeCompact]).
	compactThreshold int64
	// toolNameByID names the tool each tool-result answered, for compaction
	// stubs.
	toolNameByID map[string]string
	// emptyTurnNudges counts empty/think-only turns already nudged this run
	// (see [maxEmptyTurnNudges]).
	emptyTurnNudges int
	// finalizePrompt, when non-empty, enables the one reserved finalization
	// turn at a stop condition (see [Runner.finalizeAndTruncate]).
	finalizePrompt string
	// responseSchema, when non-nil, rides every completion of the run
	// (capability-gated; see [Runner.complete]).
	responseSchema json.RawMessage
	// outcome accumulates the run's result and is what run returns.
	outcome *Outcome
}

// stop marks the run as truncated: reason, the stop condition that fired (not
// "finalized"), lands in Outcome.TruncationReason. It is the single place a
// run's truncation is recorded; [Runner.Run] converts it into an
// [IncompleteError] at the run's end.
func (s *runState) stop(reason TruncationReason) {
	s.outcome.TruncationReason = reason
}

// trimDanglingToolTurn returns seed with a dangling trailing assistant
// tool-call turn removed, so a conversation continued from a context-cancelled
// run never ends a wire request with an unanswered tool call (Anthropic
// rejects it; other providers misbehave). Only the seed's LAST assistant turn
// is checked — the loop answers every earlier turn's calls before moving on,
// so only a cancelled turn can dangle, and only at the tail. If that turn's
// ToolCalls are not all answered by the RoleToolResult messages after it,
// everything from that assistant turn to the end of the seed is dropped: a
// half-executed turn has no model-visible meaning to salvage. The input is
// never mutated; run() copies the returned subslice into its own conversation
// before appending anything.
func trimDanglingToolTurn(seed []llmkit.Message) []llmkit.Message {
	last := -1
	for i := len(seed) - 1; i >= 0; i-- {
		if seed[i].Role == llmkit.RoleAssistant {
			last = i
			break
		}
	}
	if last < 0 || len(seed[last].ToolCalls) == 0 {
		return seed
	}
	answered := make(map[string]bool, len(seed[last].ToolCalls))
	for _, m := range seed[last+1:] {
		if m.Role == llmkit.RoleToolResult && m.ToolCallID != "" {
			answered[m.ToolCallID] = true
		}
	}
	for _, call := range seed[last].ToolCalls {
		if !answered[call.ID] {
			return seed[:last]
		}
	}
	return seed
}

// finalizeAndTruncate is the single reserved finalization turn taken at the
// iteration-cap, per-run token budget, and shared budget pool stops. It is a
// no-op unless finalizePrompt is non-empty AND the run
// has not already taken a finalization turn this run, so the public Run path
// (finalizePrompt == "") never pays an extra model call. When it does fire it:
//
//   - appends finalizePrompt as a user-role message;
//   - sets outcome.Finalized = true;
//   - compacts once so the prompt that is sent is the smallest it can be;
//   - takes ONE tool-less completion via completeOnce (which itself handles
//     the StopMaxTokens continuation retry), giving the model a cheap final
//     shot at emitting its answer instead of leaving the caller with a
//     silently empty output.
//
// s.responseSchema, when non-nil, is attached to the finalization completion so
// a schema-aware adapter applies grammar-constrained decoding on the final turn
// too. The public Run path passes nil so no schema is attached.
//
// Returns a non-nil err only when the underlying completion failed — in which
// case the caller should return early with the error. Otherwise the caller is
// responsible for s.stop(reason) + break: the reason is the STOP condition
// (budget/iteration), not "finalized".
func (r *Runner) finalizeAndTruncate(ctx context.Context, em runEmitter, s *runState, reason TruncationReason) error {
	if s.finalizePrompt == "" || s.outcome.Finalized {
		return nil
	}
	s.messages = append(s.messages, llmkit.TextMessage(llmkit.RoleUser, s.finalizePrompt))
	s.outcome.Finalized = true
	// Compact before the finalization turn: it is often the largest history of
	// the run, and the model needs only its own reasoning chain (preserved) to
	// emit the answer, not every earlier file dump. The re-armed threshold is
	// discarded: this is the run's final turn, so no later compaction can fire.
	s.messages, _ = r.maybeCompact(ctx, em, s.messages, s.compactThreshold, s.toolNameByID, s.outcome.Iterations+1)
	if _, cerr := r.completeOnce(ctx, em, &s.messages, s.outcome, s.responseSchema, true); cerr != nil {
		return cerr
	}
	return nil
}

// maybeCompact applies threshold-triggered history compaction. When compaction
// is enabled (threshold > 0) and the estimated history size exceeds threshold,
// it prunes tool-result content older than the most recent few turns to short
// stubs and returns the compacted history together with a re-armed (higher)
// threshold so the next firing only happens once history has grown materially
// again. When disabled or under threshold — or when there is nothing left to
// prune — it returns the messages and threshold unchanged, so a normal run pays
// no allocation and the append-only prefix (and its cache) is preserved.
//
// step is the completion that will consume the (possibly) compacted history;
// it is stamped on the Compaction event, which is emitted only when this call
// actually pruned — a threshold crossing with nothing to reclaim is silent.
func (r *Runner) maybeCompact(ctx context.Context, em runEmitter, messages []llmkit.Message, threshold int64, toolNameByID map[string]string, step int) ([]llmkit.Message, int64) {
	if threshold <= 0 {
		return messages, threshold
	}
	if estimateTokens(messages) <= threshold {
		return messages, threshold
	}
	compacted, pruned := compactHistory(messages, compactRecentToolResults, toolNameByID)
	if pruned == 0 {
		// Over threshold but nothing prunable yet: either history is dominated by
		// assistant reasoning, or every prunable result is already a stub, or there
		// simply aren't more than compactRecentToolResults tool results so far. Leave the
		// threshold UNCHANGED so the check fires again next turn once a result ages out of
		// the recent window — re-arming here would starve real compaction by ratcheting
		// the threshold above the history before anything was ever reclaimed.
		return messages, threshold
	}
	before := estimateTokens(messages)
	after := estimateTokens(compacted)
	// The pruned history's consuming turn rides the context the event
	// sees, matching the Step it reports.
	ctx = llmkit.WithStep(ctx, step)
	r.recordCompaction(ctx, em, step, before, after, pruned)
	// Real pruning happened (one prefix cache miss paid). Re-arm upward so the
	// next firing only comes after history has grown materially again, bounding
	// total firings and avoiding turn-over-turn cache thrash.
	return compacted, threshold * compactRearmFactor
}

// repair issues a SINGLE tools-less, schema-bearing completion against the
// repair prompt. Tools are dropped so Google and Anthropic (which refuse to
// combine tool use with native structured output) also get the schema honored.
//
// responseSchema, when non-nil, is attached capability-gated (same contract as
// the main run path: dropped silently when the adapter lacks structured output).
//
// The repair uses a fresh outcome seeded with baseIter but shares the caller's
// transcript so the assistant turn is recorded there for parity with the main
// run path. baseIter is the parent run's completed iteration count, so a parent
// that recorded steps 1..N records its repair at N+1 — step numbering stays
// monotonic across the boundary. No tool loop runs. The single completion is
// issued via completeOnce, so a stop at the output token cap pays the ONE
// max-tokens continuation completion on top — the pass is bounded to at most
// two schema-bearing, tool-less completions.
func (r *Runner) repair(ctx context.Context, em runEmitter, prompt string, responseSchema json.RawMessage, baseIter int) (*Outcome, error) {
	outcome := &Outcome{Transcript: em.tr, Iterations: baseIter, RunID: llmkit.RunFromContext(ctx)}
	messages := []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, prompt)}
	if _, err := r.completeOnce(ctx, em, &messages, outcome, responseSchema, true); err != nil {
		return outcome, err
	}
	return outcome, nil
}

// completeOnce issues one model completion against the current conversation,
// records it, folds its usage into the outcome, and appends the assistant turn
// to *messages. When final is true the request carries no tools (the
// finalization turn forbids further investigation).
//
// responseSchema, when non-nil and the client's StructuredOutput capability is
// on, is attached to the request so adapters can apply native
// schema-constrained output. When the capability is off, the schema is
// dropped silently — the prompt-embedded schema instruction is the only
// enforcement, matching the no-cap passthrough path's contract.
//
// If the completion stops at the token cap (StopMaxTokens) with no tool
// calls, it makes ONE continuation completion — appending a short user
// nudge — so a JSON answer cut off mid-object has a chance to be completed
// rather than failing to parse. The outcome's LastStopReason reflects the
// final completion served here.
func (r *Runner) completeOnce(ctx context.Context, em runEmitter, messages *[]llmkit.Message, outcome *Outcome, responseSchema json.RawMessage, final bool) (llmkit.Response, error) {
	resp, err := r.complete(ctx, em, *messages, outcome, responseSchema, final)
	if err != nil {
		return llmkit.Response{}, err
	}
	*messages = append(*messages, resp.Message())

	// One continuation retry when output was truncated mid-generation: ask the
	// model to continue and emit ONLY the remaining answer, then concatenate.
	// Guarded so it fires at most once per completeOnce call.
	if resp.StopReason == llmkit.StopMaxTokens && len(resp.ToolCalls) == 0 {
		*messages = append(*messages, llmkit.TextMessage(llmkit.RoleUser,
			"Your previous message was cut off at the output token limit. Continue from exactly where you stopped and output ONLY the remaining text needed to complete the answer — no preamble, no repetition."))
		cont, cerr := r.complete(ctx, em, *messages, outcome, responseSchema, final)
		if cerr != nil {
			return llmkit.Response{}, cerr
		}
		*messages = append(*messages, cont.Message())
		// Stitch the two halves so the caller (and FinalText) sees one answer.
		// Models frequently ignore "continue from where you stopped" and instead
		// restart, repeating some head of the first half. A naive resp.Text+cont.Text
		// would then double that prefix and corrupt the JSON. Trim the longest
		// suffix of resp.Text that the continuation re-emits as its prefix before
		// concatenating, so a clean cut and a repeated-prefix restart both stitch
		// into one well-formed answer.
		joined := stitchContinuation(resp.Text, cont.Text)
		outcome.FinalText = joined
		resp.Text = joined
		resp.ToolCalls = cont.ToolCalls
		resp.StopReason = cont.StopReason
		resp.Blocks = stitchBlocks(resp.Blocks, cont.Blocks, joined)
	}
	return resp, nil
}

// maxStitchOverlap bounds how far back stitchContinuation scans for a repeated
// prefix. The overlap is the chunk a restarting model re-emits; a few KB covers
// realistic restarts while keeping the scan O(n) and immune to a pathological
// continuation that happens to share a huge prefix with the head.
const maxStitchOverlap = 4096

// stitchContinuation joins a truncated first half with its continuation, undoing
// the common case where the model restarts and repeats some head of head as the
// prefix of cont. It finds the LONGEST suffix of head (bounded by
// maxStitchOverlap) that is a prefix of cont and drops that overlap from cont
// before concatenating. With no overlap it degrades to head+cont (the clean-cut
// case). The scan is longest-first so a model that repeats more text wins over a
// coincidental short match.
func stitchContinuation(head, cont string) string {
	max := len(head)
	if max > maxStitchOverlap {
		max = maxStitchOverlap
	}
	if max > len(cont) {
		max = len(cont)
	}
	for n := max; n > 0; n-- {
		if head[len(head)-n:] == cont[:n] {
			return head + cont[n:]
		}
	}
	return head + cont
}

// stitchBlocks shapes the Block list of the stitched llmkit.Response that
// completeOnce returns to ITS caller. It does not touch the conversation
// history: completeOnce already appended each half verbatim via
// [llmkit.Response.Message], so both halves' thinking blocks are carried in history
// in order exactly as the provider issued them. Only text blocks are
// stitched (see [stitchContinuation]); thinking blocks from BOTH halves are
// carried through untouched — never split, reordered, or re-emitted
// differently — because the next turn must return them to the provider
// exactly as they arrived. Assistant turns carry no other block kinds.
func stitchBlocks(head, cont []llmkit.Block, joinedText string) []llmkit.Block {
	var out []llmkit.Block
	for _, b := range head {
		if b.Kind == llmkit.BlockThinking {
			out = append(out, b)
		}
	}
	for _, b := range cont {
		if b.Kind == llmkit.BlockThinking {
			out = append(out, b)
		}
	}
	return append(out, llmkit.Block{Kind: llmkit.BlockText, Text: joinedText})
}

// complete issues a single completion, records it, and accumulates its usage and
// stop reason onto the outcome. It does NOT mutate the conversation; callers
// append the assistant turn so they control conversation shape (e.g. the
// continuation retry).
//
// responseSchema, when non-nil AND the client's StructuredOutput capability is
// on, is attached to the request so adapters that support native structured
// output can apply grammar-constrained decoding. When the capability is off
// (a conservative openai-compatible endpoint, etc.), the schema is silently
// dropped on the wire (per [llmkit.Request.ResponseSchema] docs) and only the
// prompt-embedded schema instruction is in effect. This is the agent-layer
// gate between the no-cap passthrough path and the with-cap native shape
// guarantee.
func (r *Runner) complete(ctx context.Context, em runEmitter, messages []llmkit.Message, outcome *Outcome, responseSchema json.RawMessage, final bool) (llmkit.Response, error) {
	req := llmkit.Request{
		System:    r.systemPrompt,
		Messages:  messages,
		Tools:     r.tools.defs,
		MaxTokens: r.maxTokens,
	}
	// The finalization turn forbids further investigation: drop the tools so the
	// model can only answer.
	if final {
		req.Tools = nil
	}
	// Native schema-constrained output is capability-gated at the agent layer:
	// when the client cannot honor a schema, do not put one on the wire. The
	// schema name is left empty so each adapter picks its own default
	// ("response" on OpenAI, "emit_answer" on Anthropic).
	if len(responseSchema) > 0 && r.client.Capabilities().StructuredOutput {
		req.ResponseSchema = responseSchema
	}
	// This is the single fire point for [RequestPolicy.PrepareRequest],
	// [Hooks.BeforeCompletion], and [Hooks.AfterCompletion]: every
	// client.Complete in the loop (main turn, max-tokens continuation,
	// forced finalization, repair) goes through here. step is the 1-based
	// turn this completion belongs to (outcome.Iterations+1 at fire time):
	// the SAME number the policy and every other hook report for this turn —
	// ToolEvent.Step — and the Event.Step of the
	// Completion event below, so consumers can join all hook families on
	// Step. Captured before Complete because outcome.Iterations is
	// incremented only after the call returns, keeping the hook pair's step
	// identical.
	step := outcome.Iterations + 1
	// The turn's Step rides the context for everything this completion
	// touches — the request policy, the hooks, and the client, whose retry
	// stage reads it into its Attempt events — so decorator-emitted events
	// join the Runner's own on Step, not just on SpanID. Placed before the
	// span mint below because the policy and BeforeCompletion fire first.
	ctx = llmkit.WithStep(ctx, step)
	// The clone isolates the loop's history from slice-level edits by the
	// policy; the post-policy slice is what the wire and the transcript see.
	if r.requestPolicy != nil {
		req.Messages = slices.Clone(messages)
		if err := r.requestPolicy.PrepareRequest(ctx, step, &req); err != nil {
			return llmkit.Response{}, fmt.Errorf("agent: request policy at iteration %d: %w", step, err)
		}
	}
	r.fireBeforeCompletion(ctx, step, &req)
	// The Runner is the emitter for this turn: it mints the completion's
	// span with BeginCompletion on the ctx handed to the client and emits
	// the Completion event itself, per the emission rule documented on
	// [llmkit.Observer]. clientCtx carries that span; ctx (Step set, no
	// span) is what RequestPolicy, BeforeCompletion, Delta, and
	// AfterCompletion receive.
	clientCtx := llmkit.BeginCompletion(ctx)
	start := time.Now()

	var resp llmkit.Response
	var err error
	if sink := r.deltaSink(ctx, step); sink != nil {
		// Delta opts the turn into incremental delivery: llmkit.Stream uses
		// the client's native stream when it implements StreamingClient and
		// synthesizes deltas from the finished Response otherwise. The
		// returned Response — and therefore everything below — is identical
		// to the Complete path either way. The hook itself fires on the
		// pre-claim ctx, not clientCtx.
		resp, err = llmkit.Stream(clientCtx, r.client, req, sink)
	} else {
		resp, err = r.client.Complete(clientCtx, req)
	}
	r.fireAfterCompletion(ctx, step, &req, &resp, err)
	// The Completion event: one per logical completion, success or failure,
	// emitted after the AfterCompletion hook so the hook family stays
	// closest to the wire call. Built on clientCtx so its SpanID matches
	// the one the client (and any Attempt events it emitted) saw.
	r.emitCompletion(clientCtx, em, step, start, req, resp, err)
	if err != nil {
		return llmkit.Response{}, fmt.Errorf("agent: completion failed at iteration %d: %w", step, err)
	}

	outcome.Iterations++
	outcome.Usage = outcome.Usage.Add(resp.Usage)
	outcome.LastStopReason = resp.StopReason
	// FinalText is assigned after each successful completion, so an empty
	// completion empties it; a failed completion returned above and leaves it
	// unchanged. completeOnce overwrites it with the stitched text after a
	// max-tokens continuation.
	outcome.FinalText = resp.Text
	// Charge the shared budget pool (WithBudgetPool) for this completion:
	// every successful completion of the run — main turn, continuation,
	// forced finalization, repair — pays here, right where its usage is
	// folded, so the pre-turn pool check above always sees complete spend.
	r.budgetPool.Add(resp.Usage.ChargeableTokens(r.limits.CacheReadWeight))
	return resp, nil
}

// overBudget reports whether cumulative usage has exceeded the token budget. A
// negative budget means unlimited. Cache reads are discounted by
// CacheReadWeight (resolved to 1.0 by Limits.resolve() when unset) so a
// cache-heavy run is bounded by its real cost, not raw prompt size.
func (r *Runner) overBudget(u llmkit.Usage) bool {
	if r.limits.TokenBudget < 0 {
		return false
	}
	return u.ChargeableTokens(r.limits.CacheReadWeight) > r.limits.TokenBudget
}

package llmkit

import (
	"context"
	"encoding/json"
	"time"
)

// Observer receives typed events from the harness's nondeterministic
// boundaries: LLM completions, provider attempts, tool runs, compaction,
// steering, run finalization, decisions, embeddings, and sandbox
// executions. One method, one event, no return value: an Observer is a data
// sink. It MUST NOT affect the caller's result — never mutate the event or
// its payload, never block for long, never write back into harness state. A
// panicking Observer is a harness bug and propagates to the caller; it is
// never recovered by llmkit.
//
// # Emission rule
//
// A Completion event is emitted exactly once per logical completion by the
// OUTERMOST harness layer — either the agent Runner (with Event.Step set) or
// the llmkit.Observe client decorator for bare clients — never both.
// Attempt events come only from the provider retry stage, inside the same
// logical completion; deterministic replay consumes Completion events only
// and ignores Attempts. Join rule: the Completion emitter mints a fresh
// Event.SpanID per logical completion and places it in the context it passes
// to the client ([WithSpan]), so every Attempt event carries the same
// SpanID as its Completion — Attempts join on SpanID, not on time windows,
// and nested completions (a tool calling the model) get their own span.
//
// Run identity: events carry the run's [RunID] (from the context, see
// [WithRun]); only the agent Runner stamps ParentRunID and Step.
type Observer interface {
	Observe(ctx context.Context, ev Event)
}

// ObserverFunc adapts a plain function to the [Observer] interface, the way
// [RecorderFunc] adapts to Recorder.
type ObserverFunc func(ctx context.Context, ev Event)

// Observe calls f(ctx, ev). It satisfies [Observer].
func (f ObserverFunc) Observe(ctx context.Context, ev Event) { f(ctx, ev) }

// Observers fans an event out to every observer, in argument order. A nil
// Observer interface value is skipped, so a caller can chain an optional
// durable sink after a mandatory one without a nil check; a typed-nil
// pointer or ObserverFunc(nil) is NOT skipped — it is called and will
// panic. An empty (or all-nil-interface) chain is a valid no-op observer,
// defined so callers never branch on nil. Observers are invoked
// synchronously on the emitting goroutine and a panic in one propagates
// immediately — later observers are not run and nothing is recovered.
func Observers(observers ...Observer) Observer {
	live := make([]Observer, 0, len(observers))
	for _, o := range observers {
		if o != nil {
			live = append(live, o)
		}
	}
	return ObserverFunc(func(ctx context.Context, ev Event) {
		for _, o := range live {
			o.Observe(ctx, ev)
		}
	})
}

// EventKind discriminates [Event]. Each kind populates its own payload
// pointer on Event; exactly one payload is set on an emitted event.
type EventKind string

const (
	// KindStart opens a run. The agent Runner emits it as the first event of
	// every run so a sink can open its run record (a task slug for a JSONL
	// filename, a row for a runs table) before the first Completion. Payload:
	// [StartEvent]. Step 0; ParentRunID rides here on continued runs.
	KindStart EventKind = "start"
	// KindCompletion records one logical completion — the request, the final
	// response or the error, provider and model. Emitted once per completion
	// by the outermost layer (Runner with Step, or the llmkit.Observe
	// decorator without), never by the retry stage. The emitter mints a
	// fresh Event.SpanID per logical completion ([WithSpan]) and passes it
	// to the client, so the retry stage's Attempts join this event. Payload:
	// [CompletionEvent].
	KindCompletion EventKind = "completion"
	// KindAttempt records one provider attempt inside a completion, including
	// failed ones. Emitted only from the provider retry stage; replay
	// consumes Completion, not Attempt. Carries the Completion's Event.SpanID.
	// Payload: [AttemptEvent]. Step 0.
	KindAttempt EventKind = "attempt"
	// KindToolRun records one tool call: the model's call, the textual
	// result, and whether it errored. A policy denial is the same kind with
	// Denied set — there is no separate policy-denied kind. Payload:
	// [ToolRunEvent]. Step is set on Runner-emitted events.
	KindToolRun EventKind = "tool_run"
	// KindCompaction records one context-window compaction pass: token
	// totals before and after, and how many messages were pruned. Payload:
	// [CompactionEvent]. Step is set on Runner-emitted events.
	KindCompaction EventKind = "compaction"
	// KindSteer records one user steering message injected into a running
	// agent turn, and whether the runner queued a follow-up turn for it.
	// Payload: [SteerEvent]. Step is set on Runner-emitted events.
	KindSteer EventKind = "steer"
	// KindFinalize closes a run: why it stopped (truncation reason, if the
	// step budget or context limit ended it), how many model iterations it
	// took, the run's total usage, and whether a forced-finalization turn
	// fired. Payload: [FinalizeEvent]. Step is set.
	KindFinalize EventKind = "finalize"
	// KindDecision records one decision-model call: the judged state, the
	// questions asked and the answers returned (or the error). The payload
	// mirrors the decide package's vocabulary in root-owned structs so the
	// root package never imports decide. Payload: [DecisionEvent]. Step 0.
	KindDecision EventKind = "decision"
	// KindEmbed records one embedding call: model, input count, resulting
	// dimensions, cache hits, usage, or the error. Payload: [EmbedEvent].
	// Step 0.
	KindEmbed EventKind = "embed"
	// KindExec records one sandbox execution: backend, command, exit code,
	// captured byte counts, whether output was truncated, or the error.
	// Payload: [ExecEvent]. Step 0.
	KindExec EventKind = "exec"
)

// Event is one observation at a nondeterministic boundary. One struct,
// Kind-discriminated: exactly one payload pointer is non-nil on an event
// emitted by the harness; the rest stay zero and are omitted from JSON, so
// encoding/json round-trips an event with no registry and no untyped bag.
//
// Header fields every event carries: Kind, RunID (empty outside a run),
// SpanID (the logical completion a Completion or Attempt belongs to), Step
// (populated ONLY by the agent Runner; 0 elsewhere), Time (stamped by the
// emitter when the observed operation ended), Duration, and
// SchemaVersion ([EventSchemaVersion]). ParentRunID is a run-level fact only
// the Runner stamps, on continued runs.
//
// Two wire facts for sink and replay authors: omitempty collapses empty
// slices and maps to absent (decoding yields nil, not []), and a
// json.RawMessage payload is carried compacted on the wire. Compare decoded
// payloads semantically (as parsed JSON), not with reflect.DeepEqual.
type Event struct {
	Kind EventKind `json:"kind"`
	// RunID correlates every event of one run ([WithRun]).
	RunID RunID `json:"run_id,omitempty"`
	// ParentRunID is set by the agent Runner on every event of a run
	// continued from a previous one; decorators leave it empty.
	ParentRunID RunID `json:"parent_run_id,omitempty"`
	// SpanID is stamped by [NewEvent] on every kind from the context
	// ([WithSpan]); it is load-bearing on Completion and Attempt: the
	// Completion emitter mints one per logical completion and passes it to
	// the client in the context, the retry stage inherits it, so an Attempt
	// joins its Completion on SpanID. A nested completion gets its own span.
	SpanID SpanID `json:"span_id,omitempty"`
	// Step is the 1-based model turn within an agent run, set only on events
	// the Runner emits; 0 on decorator-emitted events.
	Step int `json:"step,omitempty"`
	// Time is when the observed operation ended, set by the emitter. Sinks
	// never re-stamp it.
	Time time.Time `json:"time"`
	// Duration is the wall time of the operation the event closes: the full
	// logical completion for [CompletionEvent] (retries and backoff
	// included), one attempt for [AttemptEvent], the execution for
	// [ToolRunEvent]/[EmbedEvent]/[ExecEvent], the Ask for [DecisionEvent],
	// the compaction pass for [CompactionEvent], and the whole run for
	// [FinalizeEvent]. Zero for instantaneous kinds ([StartEvent],
	// [SteerEvent]) and for failed boundaries that never measured one. The
	// wire unit is nanoseconds (time.Duration's JSON form).
	Duration time.Duration `json:"duration,omitempty"`
	// SchemaVersion is [EventSchemaVersion], set by [NewEvent]; a hand-built
	// Event carries 0. Present on every encoded event.
	SchemaVersion int `json:"schema_version"`

	// Start is set on KindStart.
	Start *StartEvent `json:"start,omitempty"`
	// Completion is set on KindCompletion.
	Completion *CompletionEvent `json:"completion,omitempty"`
	// Attempt is set on KindAttempt.
	Attempt *AttemptEvent `json:"attempt,omitempty"`
	// ToolRun is set on KindToolRun.
	ToolRun *ToolRunEvent `json:"tool_run,omitempty"`
	// Compaction is set on KindCompaction.
	Compaction *CompactionEvent `json:"compaction,omitempty"`
	// Steer is set on KindSteer.
	Steer *SteerEvent `json:"steer,omitempty"`
	// Finalize is set on KindFinalize.
	Finalize *FinalizeEvent `json:"finalize,omitempty"`
	// Decision is set on KindDecision.
	Decision *DecisionEvent `json:"decision,omitempty"`
	// Embed is set on KindEmbed.
	Embed *EmbedEvent `json:"embed,omitempty"`
	// Exec is set on KindExec.
	Exec *ExecEvent `json:"exec,omitempty"`
}

// StartEvent opens a run ([KindStart]). Task is the caller's task text the
// run is working on; Tools lists the tool names offered to the model, in
// offer order. Whether the run continued from a previous one is
// Event.ParentRunID on the same event, not a separate fact.
type StartEvent struct {
	Task  string   `json:"task,omitempty"`
	Tools []string `json:"tools,omitempty"`
}

// CompletionEvent records one logical completion ([KindCompletion]): the
// request sent, the final response or the error, and the provider/model that
// served it. Err is non-empty exactly when the completion failed; a failed
// completion carries the zero Response. Request and Response embed the full
// message round-trip, so a sink can replay the turn from this event alone.
type CompletionEvent struct {
	Request  Request  `json:"request"`
	Response Response `json:"response"`
	// Err is the failure, as text; empty on success.
	Err string `json:"err,omitempty"`
	// Provider is the provider tag ("anthropic", "openai", ...) or the
	// caller's own config name.
	Provider string `json:"provider,omitempty"`
	// Model is the model identifier.
	Model string `json:"model,omitempty"`
}

// AttemptEvent records one provider attempt ([KindAttempt]) from inside the
// retry stage: the same request, the attempt's own response or error, and
// its 1-based position. Attempt 1 is the first wire call. The response is
// the raw adapter response — attempts are observed below the tool-call
// serializer — so the CompletionEvent's Response may differ from it where
// the serializer truncated. A failed attempt carries the error text and
// the zero Response; when that error is an [*APIError], StatusCode and
// RetryAfter carry its HTTP status and the server-suggested delay the
// retry classifier read (both zero on success and for unclassified
// transport failures). Event.SpanID joins it to its logical completion.
type AttemptEvent struct {
	// Attempt is the 1-based attempt number within the logical completion.
	Attempt  int      `json:"attempt"`
	Request  Request  `json:"request"`
	Response Response `json:"response"`
	Err      string   `json:"err,omitempty"`
	// StatusCode is the HTTP status of a failed attempt whose error is an
	// [*APIError]; 0 on success and for unclassified transport failures.
	StatusCode int `json:"status_code,omitempty"`
	// RetryAfter is the *APIError's RetryAfter as the server sent it,
	// before the stage caps it at MaxDelay; 0 when the server supplied
	// none.
	RetryAfter time.Duration `json:"retry_after,omitempty"`
	Provider   string        `json:"provider,omitempty"`
	Model      string        `json:"model,omitempty"`
}

// ToolRunEvent records one tool call ([KindToolRun]). Call is the model's
// request; Result is the model-visible textual result. IsError marks a
// failed execution (Result carries the "ERROR:"-prefixed message). Denied
// marks a policy denial — then Result is empty and DenyReason says why;
// there is no separate policy-denied event kind.
type ToolRunEvent struct {
	Call       ToolCall `json:"call"`
	Result     string   `json:"result,omitempty"`
	IsError    bool     `json:"is_error,omitempty"`
	Denied     bool     `json:"denied,omitempty"`
	DenyReason string   `json:"deny_reason,omitempty"`
}

// CompactionEvent records one compaction pass ([KindCompaction]): context
// size before and after, in tokens, and how many messages the pass pruned.
type CompactionEvent struct {
	BeforeTokens int64 `json:"before_tokens"`
	AfterTokens  int64 `json:"after_tokens"`
	Pruned       int   `json:"pruned"`
}

// SteerEvent records one steering message injected into a running turn
// ([KindSteer]). Message is what the user sent; FollowUp marks that the
// runner will run a follow-up model turn with it after the current one.
type SteerEvent struct {
	Message  Message `json:"message"`
	FollowUp bool    `json:"follow_up,omitempty"`
}

// FinalizeEvent closes a run ([KindFinalize]). TruncationReason is set when
// the run ended by hitting a limit rather than finishing its task;
// Iterations counts the model turns the run took; Usage totals the run's
// completions; Finalized marks that a forced-finalization turn fired (a
// fact about the run's shape that is not derivable from the event stream).
type FinalizeEvent struct {
	TruncationReason string `json:"truncation_reason,omitempty"`
	Finalized        bool   `json:"finalized,omitempty"`
	Iterations       int    `json:"iterations"`
	Usage            Usage  `json:"usage"`
}

// DecisionEvent records one decision-model call ([KindDecision]). The
// struct mirrors the decide package's question/answer vocabulary in
// root-owned types — the root package cannot import decide (decide imports
// root), so decide converts to and from these types at its boundary.
//
// On a failed Ask, Err is non-empty (State and Questions may still be
// recorded — what was judged and asked is known even when the call
// failed); on success Answers is populated and Err is empty. Question and
// answer kinds are the decide wire discriminators: "noul" (binary
// belief), "choice" (one option picked), "score" (levels rated).
// Questions and Answers are each sorted by ID (decide.Questions is a map;
// the conversion layer sorts), and an answer matches its question by ID.
type DecisionEvent struct {
	// Backend names the decision backend (e.g. "typesafe"); Model its model
	// identifier.
	Backend string `json:"backend,omitempty"`
	Model   string `json:"model,omitempty"`
	// State is the state the Ask judged, as raw JSON (string, object, or
	// array) — the input half of the call, so a sink can interpret or replay
	// the decision from this event alone. State is emitted verbatim: an
	// invalid RawMessage fails Marshal for the whole event, so converters
	// must pre-validate.
	State json.RawMessage `json:"state,omitempty"`
	// Questions is what was asked, sorted by ID.
	Questions []DecisionQuestion `json:"questions,omitempty"`
	// Answers is what came back, sorted by question ID.
	Answers []DecisionAnswer `json:"answers,omitempty"`
	// Usage is the decision call's token consumption.
	Usage Usage `json:"usage"`
	// Err is the failure, as text; empty on success.
	Err string `json:"err,omitempty"`
}

// DecisionQuestion is one question inside a [DecisionEvent]. ID matches its
// answer's ID. Kind is "noul", "choice", or "score". Instructions is the
// question's free-text payload as raw JSON. True and False are a noul
// question's belief criteria as raw JSON (noul only). Options maps
// choice-option names to their raw-JSON descriptions (choice only); Levels
// lists the score legend entries as raw JSON (score only).
type DecisionQuestion struct {
	ID           string                     `json:"id"`
	Kind         string                     `json:"kind"`
	Instructions json.RawMessage            `json:"instructions,omitempty"`
	True         json.RawMessage            `json:"true,omitempty"`
	False        json.RawMessage            `json:"false,omitempty"`
	Options      map[string]json.RawMessage `json:"options,omitempty"`
	Levels       []json.RawMessage          `json:"levels,omitempty"`
}

// DecisionAnswer is one answer inside a [DecisionEvent], matched to its
// question by ID. By question kind, exactly one group is populated:
//
//   - noul: Belief (0..1);
//   - choice: Choice plus Probabilities (per option) and Confidence;
//   - score: Score (the server's numeric score) plus Levels, the full
//     dense legend, index-aligned with LevelProbabilities.
//
// Belief and Score are pointers so a 0 value survives the round-trip
// distinct from "absent".
type DecisionAnswer struct {
	ID                 string             `json:"id"`
	Belief             *float64           `json:"belief,omitempty"`
	Score              *float64           `json:"score,omitempty"`
	Choice             string             `json:"choice,omitempty"`
	Probabilities      map[string]float64 `json:"probabilities,omitempty"`
	Confidence         float64            `json:"confidence,omitempty"`
	Levels             []string           `json:"levels,omitempty"`
	LevelProbabilities []float64          `json:"level_probabilities,omitempty"`
}

// EmbedEvent records one embedding call ([KindEmbed]): the model, how many
// inputs were embedded, the per-vector dimensions, how many inputs were
// served from the caller's cache, the usage, or the error. Err is non-empty
// exactly when the call failed. v1 records a summary, not the inputs or
// vectors; replaying this boundary is out of Round A scope and additive
// later.
type EmbedEvent struct {
	Model      string `json:"model,omitempty"`
	Inputs     int    `json:"inputs"`
	Dimensions int    `json:"dimensions"`
	CacheHits  int    `json:"cache_hits"`
	Usage      Usage  `json:"usage"`
	Err        string `json:"err,omitempty"`
}

// ExecEvent records one sandbox execution ([KindExec]): the backend, the
// command and argv, the exit code, how many bytes of output each stream
// produced, whether capture was truncated, or the error that prevented
// execution. ExitCode is -1 when the process never ran to an exit (Err
// non-empty). v1 records a summary, not the captured output; replaying this
// boundary is out of Round A scope and additive later.
type ExecEvent struct {
	Backend     string   `json:"backend,omitempty"`
	Command     []string `json:"command,omitempty"`
	ExitCode    int      `json:"exit_code"`
	StdoutBytes int64    `json:"stdout_bytes"`
	StderrBytes int64    `json:"stderr_bytes"`
	Truncated   bool     `json:"truncated,omitempty"`
	Err         string   `json:"err,omitempty"`
}

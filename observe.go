package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Observer receives typed events from the harness's nondeterministic
// boundaries: LLM completions, provider attempts, tool runs, compaction,
// steering, run finalization, decisions, embeddings, and sandbox
// executions. One method, one event, no return value: an Observer is a data
// sink. It MUST NOT affect the caller's result — never mutate the event or
// its payload, never block for long, never write back into harness state. A
// panicking Observer is a harness bug.
//
// # Emission rule
//
// A logical completion is owned by the span in its ctx, and
// [BeginCompletion] is the only minter. The rule has three parts:
//
//   - An emitter ([Observe], or a [github.com/dpoage/llmkit/provider]
//     New/Wrap stack above its retry stage) mints and emits only when
//     the ctx it receives carries no span; a ctx that already carries
//     one belongs to an enclosing emitter, and the call passes through
//     without a new span or Completion event. A stack claims even when it
//     has no Options.Observer; it then emits nothing.
//   - The agent Runner mints on the ctx it hands its client, on every
//     turn, and emits that turn's Completion.
//   - Hooks and tools receive a ctx without the turn's span.
//
// A client can make a model call inside its own Complete, on the ctx it
// received, through one of those emitters. If that ctx carries a span (it
// does when an emitter or the agent Runner called the client), the nested
// call emits no Completion, and its spend reaches the ledger only through
// the Response.Usage the enclosing client returns; a provider stack's
// Attempt events still join the enclosing span. If that ctx carries no
// span, the nested emitter mints its own. An agent Runner run inside a
// client's Complete mints its own span and emits its own Completions.
//
// Attempt events come only from the provider retry stage inside the same
// logical completion, joined to it by Event.SpanID; deterministic
// replay consumes Completion events only. A [CompletionEvent]'s
// Response.Usage (and [DecisionEvent.Usage]) is the spend ledger;
// Attempt and Finalize are views over it. A failed completion carries
// the zero Response and reports no usage.
//
// Run identity: events carry the run's [RunID] from the context (see
// [WithRun]); ParentRunID is stamped only by the agent Runner, and Step
// follows the context ([WithStep]) wherever a Runner turn is in scope.
type Observer interface {
	Observe(ctx context.Context, ev Event)
}

// ObserverFunc adapts a plain function to the [Observer] interface.
type ObserverFunc func(ctx context.Context, ev Event)

// Observe calls f(ctx, ev). It satisfies [Observer].
func (f ObserverFunc) Observe(ctx context.Context, ev Event) { f(ctx, ev) }

// Observers fans an event out to every observer, in argument order. A nil
// Observer interface value is skipped, so a caller can chain an optional
// durable sink after a mandatory one without a nil check; a typed-nil
// pointer or ObserverFunc(nil) is NOT skipped — it is called and will
// panic. An empty (or all-nil-interface) chain is a valid no-op observer,
// defined so callers never branch on nil. Observers are invoked
// synchronously on the emitting goroutine; a panic in one leaves the later
// observers uncalled for that event, and Observers does not recover it.
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

// Source is the read side of recording: the ordered events of one run,
// however and wherever they were recorded — an in-memory transcript, a JSONL
// directory, a database. Callers do not know which sink holds a run, only
// that its events come back in emission order or the run is unknown.
//
// Events returns the run's events in emission order; the caller owns the
// returned slice. run is the id the events carry ([RunID]). A Source with no
// record of run returns an error wrapping [ErrUnknownRun] — never an empty
// success.
type Source interface {
	Events(ctx context.Context, run RunID) ([]Event, error)
}

// ErrUnknownRun is the error a [Source] wraps when it has no record of the
// requested run. Match it with errors.Is.
var ErrUnknownRun = errors.New("llmkit: unknown run")

// EventKind discriminates [Event]. Each kind populates its own payload
// pointer on Event; exactly one payload is set on an emitted event.
type EventKind string

const (
	// KindStart opens a run. The agent Runner emits it as the first event of
	// every run so a sink can open its run record (a JSONL filename, a row
	// for a runs table) before the first Completion. Payload:
	// [StartEvent]. Step 0; ParentRunID rides here on continued runs.
	KindStart EventKind = "start"
	// KindCompletion records one logical completion — the request, the final
	// response or the error, provider and model. Emitted once per completion
	// by the outermost layer — the agent Runner for agent runs, the
	// llmkit.Observe decorator for bare clients — never by the retry stage.
	// The emitter mints a fresh Event.SpanID per logical completion
	// ([BeginCompletion]) and passes it to the client, so the retry stage's
	// Attempts join this event: (run_id, span_id) identifies a completion,
	// (run_id, step) groups a turn's events. Step is the enclosing Runner
	// turn when the completion fires inside one — a tool's own
	// decorator-wrapped client included. Payload: [CompletionEvent].
	KindCompletion EventKind = "completion"
	// KindAttempt records one provider attempt inside a completion, including
	// failed ones. Emitted only from the provider retry stage; replay
	// consumes Completion, not Attempt. Carries the Completion's Event.SpanID.
	// Payload: [AttemptEvent]. Step is the enclosing turn when emitted inside
	// a Runner turn, else 0.
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
	// KindFinalize closes a run: how it ended (status and error). Unless the
	// status is [RunPanicked], it also carries the run's truncation reason
	// when it has one, its usage, and whether a forced-finalization turn
	// fired, and Step is the completed-turn count. Payload: [FinalizeEvent].
	KindFinalize EventKind = "finalize"
	// KindDecision records one decision-model call: the judged state, the
	// questions asked and the answers returned (or the error). The payload
	// mirrors the decide package's vocabulary in root-owned structs so the
	// root package never imports decide. Payload: [DecisionEvent]. Step is the
	// enclosing turn when emitted inside a Runner turn, else 0.
	KindDecision EventKind = "decision"
	// KindEmbed records one embedding call: model, input count, resulting
	// dimensions, cache hits, usage, or the error. Payload: [EmbedEvent]. Step
	// is the enclosing turn when emitted inside a Runner turn, else 0.
	KindEmbed EventKind = "embed"
	// KindExec records one sandbox execution: backend, command, exit code,
	// captured byte counts, whether output was truncated, or the error.
	// Payload: [ExecEvent]. Step is the enclosing turn when emitted inside a
	// Runner turn, else 0.
	KindExec EventKind = "exec"
)

// Event is one observation at a nondeterministic boundary. One struct,
// Kind-discriminated: exactly one payload pointer is non-nil on an event
// emitted by the harness; the rest stay zero and are omitted from JSON, so
// encoding/json round-trips an event with no registry and no untyped bag.
//
// Header fields every event carries: Kind, RunID (empty outside a run),
// SpanID (the logical completion a Completion or Attempt belongs to), Step
// (see Event.Step), Time (stamped by the emitter when the
// observed operation ended), Duration, and SchemaVersion
// ([EventSchemaVersion]). ParentRunID is a run-level fact only the Runner
// stamps, on continued runs.
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
	// ([BeginCompletion]); it is load-bearing on Completion and Attempt: the
	// Completion emitter mints one per logical completion and passes it to
	// the client in the context, the retry stage inherits it, so an Attempt
	// joins its Completion on SpanID. Which call gets a new span follows
	// the emission rule on [Observer].
	SpanID SpanID `json:"span_id,omitempty"`
	// Step is the 1-based model turn within an agent run. [NewEvent] stamps
	// it from the context ([WithStep]), so decorator-emitted events inside a
	// Runner turn — the retry stage's Attempt, a decision, a sandbox Exec or
	// embedding from a tool — carry that turn. On a [KindFinalize] event,
	// Step is the count of COMPLETED turns, not a turn number: it is 0 when
	// no turn completed, so a finalize can carry Step 0 inside a run whose
	// first completion failed (beside that completion's own Step 1). The
	// Runner emits every RunPanicked finalize with Step 0 (omitted in
	// JSON), whatever step the context carries; a hand-built event or an
	// older record may carry another step.
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
// request sent, the final response or error, and the identity of the
// client that served it. Err is non-empty exactly when the completion
// failed; a failed completion carries the zero Response and reports no
// usage (Response.Usage is the spend ledger — see [Observer] above).
// Request and Response embed the full message round-trip, so a sink can
// replay the turn from this event alone. Request aliases the caller's
// Messages and Blocks slices until the call returns — Observe and the
// provider stages do not clone them — so a sink that retains this event
// past the call must copy the top-level slices itself.
type CompletionEvent struct {
	Request  Request  `json:"request"`
	Response Response `json:"response"`
	// Err is the failure, as text; empty on success.
	Err string `json:"err,omitempty"`
	// Provider is the provider tag ("anthropic", "openai", ...) or the
	// caller's own config name — the client's [IdentityOf].Provider.
	Provider string `json:"provider,omitempty"`
	// Model is the model identifier — the client's [IdentityOf].Model.
	Model string `json:"model,omitempty"`
}

// AttemptEvent records one provider attempt ([KindAttempt]) from inside the
// retry stage: the same request, the attempt's own response or error, and
// its 1-based position. Attempt 1 is the first wire call. The response is
// the raw adapter response — attempts are observed below the tool-call
// serializer — so the CompletionEvent's Response may differ from it where
// the serializer truncated. A failed attempt carries the error text and
// the zero Response; when that error is an [*APIError], StatusCode carries
// its HTTP status (0 on success and for unclassified transport failures).
// HasRetryAfter reports whether that *APIError carried a Retry-After
// header; RetryAfter is meaningful only when HasRetryAfter is true and is
// zero otherwise, even when the underlying APIError set a nonzero
// RetryAfter without the presence bit. The presence bit records the
// header; it does not by itself say the stage retried after the attempt
// — the final attempt, a non-retryable status (a 400 with a present
// zero, for instance), and a cancelled parent all report
// (RetryAfter, HasRetryAfter) without a retry following. Event.SpanID
// joins it to its logical completion.
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
	// before the stage caps it at MaxDelay. Meaningful only when
	// HasRetryAfter is true.
	RetryAfter time.Duration `json:"retry_after,omitempty"`
	// HasRetryAfter reports that the failed attempt's *APIError carried a
	// Retry-After header. The bit records the header's presence; it does
	// not by itself say the stage retried after the attempt — the final
	// attempt, a non-retryable status, and a cancelled parent all carry
	// (RetryAfter, HasRetryAfter) without a retry following.
	HasRetryAfter bool   `json:"has_retry_after,omitempty"`
	Provider      string `json:"provider,omitempty"`
	Model         string `json:"model,omitempty"`
}

// ToolRunEvent records one tool call ([KindToolRun]). Call is the model's
// request, exactly as the model made it; Result is the model-visible textual
// result. IsError marks a failed execution (Result carries the
// "ERROR:"-prefixed message). Denied marks a policy denial — then Result is
// empty and DenyReason says why; there is no separate policy-denied event
// kind.
//
// DispatchedArguments holds the arguments Tool.Run received when a tool
// policy rewrote them to valid JSON or cleared them; a cleared payload is
// the JSON literal null. It is empty when the call ran with arguments that
// decode to the same JSON value as Call.Arguments, for a denied call or a
// call to an unregistered tool (neither reaches Tool.Run), and when the
// rewrite is not valid JSON: a replay under the same policy then hands its
// tool the rewritten bytes, compares them with Call.Arguments, and
// diverges. A record written before the field existed reads as never
// rewritten.
type ToolRunEvent struct {
	Call                ToolCall        `json:"call"`
	DispatchedArguments json.RawMessage `json:"dispatched_arguments,omitempty"`
	Result              string          `json:"result,omitempty"`
	IsError             bool            `json:"is_error,omitempty"`
	Denied              bool            `json:"denied,omitempty"`
	DenyReason          string          `json:"deny_reason,omitempty"`
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

// RunStatus is the vocabulary for what a run's end was, recorded on
// [FinalizeEvent.Status]. It is an open string: a reader must tolerate a
// value it does not know (a newer writer's), and the empty string marks a
// record from before the field existed.
type RunStatus string

const (
	// RunCompleted: the run finished with an answer.
	RunCompleted RunStatus = "completed"
	// RunIncomplete: the run stopped without an answer, on a limit or another
	// non-answer end.
	RunIncomplete RunStatus = "incomplete"
	// RunRefused: the model stopped without an answer for a provider stop
	// reason: a refusal, a safety or content filter, or another provider
	// error stop.
	RunRefused RunStatus = "refused"
	// RunFailed: the run failed with an error.
	RunFailed RunStatus = "failed"
	// RunCanceled: the run's context ended (canceled or past its deadline).
	RunCanceled RunStatus = "canceled"
	// RunPanicked: the run panicked.
	RunPanicked RunStatus = "panicked"
)

// FinalizeEvent closes a run ([KindFinalize]). Unless Status is
// [RunPanicked], TruncationReason is set when the Runner truncated the run,
// Usage is the run's cumulative token usage, Finalized marks that a
// forced-finalization turn fired (a fact about the run's shape that is not
// derivable from the event stream), and the completed-turn count is
// Event.Step, not a payload field.
//
// When a panic in an agent hook, request policy, tool policy, or RunJSON
// out-value unmarshal unwinds a run after its Start event, the Runner emits
// a Finalize with Status [RunPanicked], Step 0 (whatever step the context
// carries, a nested Runner's parent tool phase included), an empty
// TruncationReason, FinalText, and Err, and zero Usage.
type FinalizeEvent struct {
	TruncationReason string `json:"truncation_reason,omitempty"`
	Finalized        bool   `json:"finalized,omitempty"`
	Usage            Usage  `json:"usage"`
	// FinalText is the run's text as the Runner stitched it across a
	// max-tokens continuation, so a store can persist it without re-deriving
	// the stitch. For Runner.Run it is the text of the last completion that
	// succeeded: after a failed later completion it is still the earlier
	// completion's text, so it is not necessarily an answer. For
	// Runner.RunJSON, when a repair completion succeeded, it holds text from
	// the repair turn. It is empty when no completion succeeded, and
	// when the run panicked. A refusal or safety stop (the Runner's
	// StopReasonError) records the model's refusal prose here, with Status
	// [RunRefused] and Err set.
	FinalText string `json:"final_text,omitempty"`
	// Status is how the run ended, in the [RunStatus] vocabulary. Empty marks
	// a record written before this field existed; a reader must not read
	// empty as [RunCompleted].
	Status RunStatus `json:"status,omitempty"`
	// Err is the text of the error the run returned; empty when the run
	// completed, and on records from before this field existed.
	Err string `json:"err,omitempty"`
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
// inputs were embedded, the per-vector dimensions, or the error. Err is
// non-empty exactly when the call failed. v1 records a summary, not the
// inputs or vectors.
type EmbedEvent struct {
	Model      string `json:"model,omitempty"`
	Inputs     int    `json:"inputs"`
	Dimensions int    `json:"dimensions"`
	Err        string `json:"err,omitempty"`
}

// ExecEvent records one sandbox execution ([KindExec]): the backend, the
// command and argv, the exit code, how many bytes of output each stream
// produced, whether capture was truncated, or the error that prevented
// execution. ExitCode is -1 when the process never ran to an exit (Err
// non-empty). v1 records a summary, not the captured output; replaying
// this boundary is out of scope today and additive later.
type ExecEvent struct {
	Backend     string   `json:"backend,omitempty"`
	Command     []string `json:"command,omitempty"`
	ExitCode    int      `json:"exit_code"`
	StdoutBytes int64    `json:"stdout_bytes"`
	StderrBytes int64    `json:"stderr_bytes"`
	Truncated   bool     `json:"truncated,omitempty"`
	Err         string   `json:"err,omitempty"`
}

// Validate checks an event's shape, for sinks, stores, and tests that must
// reject a malformed event before trusting it. The harness NEVER calls it —
// not on [NewEvent], not in the [Observer] emission path, not in any
// decorator — so it adds nothing to emission. Three rules:
//
//   - Kind must be one of the ten declared constants; empty or unknown
//     fails.
//   - Exactly one payload pointer must be non-nil, and it must be the one
//     the Kind names. Zero payloads fails, a foreign payload fails (the
//     error names the expected field and the offending one), and two set
//     payloads fail.
//   - SchemaVersion must be non-zero — a hand-built Event that skipped
//     [NewEvent] carries 0. Any non-zero value passes: a newer schema
//     version is a sink's business to interpret, not Validate's to refuse.
//
// Validate inspects only this shape. It does not check payload field
// values, and it does not gate encoding — Marshal emits whatever it is
// given.
func (ev Event) Validate() error {
	want, ok := kindPayloadField[ev.Kind]
	if !ok {
		return fmt.Errorf("llmkit: unknown event kind %q", string(ev.Kind))
	}
	if ev.SchemaVersion == 0 {
		return fmt.Errorf("llmkit: %s event has schema_version 0 (set Event.SchemaVersion from EventSchemaVersion, or build events with NewEvent)", ev.Kind)
	}
	// Count the non-nil payloads and remember the first two names; no
	// []string is built, so a well-formed event validates without
	// allocating (the names feed only the error text).
	n, first, second := ev.scanPayloads()
	switch n {
	case 1:
		if first != want {
			return fmt.Errorf("llmkit: %s event carries payload field %s, want %s", ev.Kind, first, want)
		}
		return nil
	case 0:
		return fmt.Errorf("llmkit: %s event carries no payload, want %s", ev.Kind, want)
	default:
		fields := fmt.Sprintf("%s and %s", first, second)
		if n > 2 {
			fields += ", ..."
		}
		return fmt.Errorf("llmkit: %s event carries %d payload fields (%s), want exactly %s", ev.Kind, n, fields, want)
	}
}

// kindPayloadField maps each declared [EventKind] to the Event field that
// carries its payload, so [Event.Validate] and its errors cannot drift from
// the constants.
var kindPayloadField = map[EventKind]string{
	KindStart:      "Start",
	KindCompletion: "Completion",
	KindAttempt:    "Attempt",
	KindToolRun:    "ToolRun",
	KindCompaction: "Compaction",
	KindSteer:      "Steer",
	KindFinalize:   "Finalize",
	KindDecision:   "Decision",
	KindEmbed:      "Embed",
	KindExec:       "Exec",
}

// scanPayloads counts the non-nil payload pointers, in struct order, and
// names the first two. It allocates nothing: the names feed only Validate's
// error text, and the closure stays on the stack.
func (ev Event) scanPayloads() (n int, first, second string) {
	seen := func(field string) {
		switch n {
		case 0:
			first = field
		case 1:
			second = field
		}
		n++
	}
	if ev.Start != nil {
		seen("Start")
	}
	if ev.Completion != nil {
		seen("Completion")
	}
	if ev.Attempt != nil {
		seen("Attempt")
	}
	if ev.ToolRun != nil {
		seen("ToolRun")
	}
	if ev.Compaction != nil {
		seen("Compaction")
	}
	if ev.Steer != nil {
		seen("Steer")
	}
	if ev.Finalize != nil {
		seen("Finalize")
	}
	if ev.Decision != nil {
		seen("Decision")
	}
	if ev.Embed != nil {
		seen("Embed")
	}
	if ev.Exec != nil {
		seen("Exec")
	}
	return n, first, second
}

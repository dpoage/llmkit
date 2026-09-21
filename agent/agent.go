// Package agent is a tool-call execution harness: a reusable loop that drives
// an [llmkit.Client] through a bounded set of tools until the model produces
// a final answer, runs out of iterations, or exhausts a token budget.
//
// Callers with different roles (a coder, a reviewer, a researcher) construct
// the same [Runner] with different system prompts and tool sets. The harness
// is provider-agnostic: it speaks only the normalized [llmkit] vocabulary.
// The narrative guide, with a turn diagram and option tables, lives in
// docs/agent-loop.md.
//
// # Tools
//
// A [Tool] declares its schema with [Tool.Def] and executes with [Tool.Run].
// [Func] builds a Tool from a plain function; the struct argument is the
// schema.
//
// An ordinary tool error never ends the run. The harness feeds it back to
// the model as a tool result prefixed "ERROR:", and the model retries, tries
// another tool, or gives up. A panic inside [Tool.Run] is recovered and
// rendered the same way, in both dispatch modes. Two error types are
// special: a *ToolHealthError marks the failure as infrastructure, so it
// also reaches [Hooks.ToolHealth]; a *StopReasonError ends the run, because
// the model itself stopped for a provider error reason.
//
// Four failures end a run with a non-nil error: a failed
// [llmkit.Client.Complete], a cancelled context, a [RequestPolicy] error,
// and [StopReasonError].
//
// Tools run one call at a time by default. [WithParallelTools] runs a turn's
// calls concurrently, and concurrent [Runner.Run] calls on one Runner are
// always allowed, so a Tool must be safe for concurrent calls.
//
// # Limits and outcomes
//
// The loop enforces two limits: [Limits.MaxIterations] (model turns) and
// [Limits.TokenBudget] (cumulative input+output tokens from [llmkit.Usage]).
// Zero selects the package default; [Unlimited] removes the cap;
// [Limits.CacheReadWeight] discounts cache-read tokens in the check.
// Exceeding a limit stops the run cleanly: [Runner.Run] returns an [Outcome]
// with a non-empty [Outcome.TruncationReason], and [Outcome.FinalText] holds
// the text of the last completion. Partial results are data, not errors.
// [Limits.HistoryTokenBudget] enables threshold-triggered history
// compaction, which replaces old tool results with short stubs and re-arms
// its threshold so the cache cost stays bounded.
//
// # Policies
//
// Two single-method interfaces sit between the model and the world.
//
//   - [RequestPolicy] ([WithRequestPolicy]) edits the outgoing request. The
//     Runner calls it before every completion: the main turn, the
//     max-tokens continuation turn, the forced-finalization turn, and the
//     RunJSON repair turn. It runs before [Hooks.BeforeCompletion], and the
//     completion event records the post-policy request. req.Messages is a
//     shallow clone of the loop's history, so a policy reshapes this turn
//     without touching the loop's history.
//   - [ToolPolicy] ([WithToolPolicy]) authorizes every tool call of a turn,
//     in model order, on the loop goroutine, before any [Tool.Run] starts —
//     in both dispatch modes. A denial feeds the model
//     "ERROR: tool <name> denied: ..." and the run continues. Only
//     call.Arguments may be rewritten; the history keeps the model's
//     original arguments.
//
// # Hooks
//
// [WithHooks] registers the observer struct [Hooks]: one optional callback
// per loop event, covering completions, deltas, tool calls, compaction,
// repair, and finalization. Hooks run synchronously, inline on the goroutine
// that reaches the fire point. Every hook family reports the same 1-based
// step for a turn — [ToolEvent.Step], [CompactionEvent.Step], and the
// Completion event's [llmkit.Event].Step — so consumers join on it, and a
// RunJSON repair turn continues the numbering. When [Hooks.Delta] is set,
// the Runner streams each completion through [llmkit.Stream]. Durable event
// recording is a separate seam: [WithObserver] installs the single sink
// ([JSONL]), whose failures surface through the callback it was constructed
// with, never through Hooks or the run. A panicking hook is a harness bug:
// it propagates out of [Runner.Run] and is never rendered to the model.
//
// # Steering
//
// [Steering] ([NewSteering], [WithSteering]) queues user turns into a
// running loop from any goroutine. [Steering.Steer] delivers before the next
// model call, after the current turn's tool results. [Steering.FollowUp]
// delivers when the run would otherwise finish and continues the loop
// instead. Undelivered turns stay queued when a limit stops the run;
// [Steering.Pending] reports them, and [Continue] with the same handle
// delivers them on the continued run. A refusal stop skips the
// would-be-finish drain, so queued turns stay pending; a steer queued
// before the refusing completion delivers at the pre-completion boundary.
// A second concurrent run on one handle fails with [ErrSteeringInUse].
//
// # Structured output
//
// [Runner.RunJSON] asks the model for a JSON answer that matches a schema,
// deep-validates the answer, and unmarshals it into a caller pointer. On a
// parse or schema violation it makes one repair round-trip before failing
// with [ErrUnparseableOutput]. A run already stopped by the token budget or
// the budget pool skips the repair: it fails immediately, and the Outcome
// keeps the budget TruncationReason. RunJSON reserves the last iteration
// for a forced-finalization turn, so a capped run still emits its answer.
// [RunJSONAs] derives the schema from a Go type with [SchemaOf] and returns
// the decoded value, so the type the model must satisfy and the type that
// decodes its answer cannot drift apart.
//
// # Budgets
//
// [BudgetPool] ([WithBudgetPool]) shares one token budget across concurrent
// Runner runs. The Runner checks the pool once per loop turn and charges it
// after every successful completion; continuation, finalization, and repair
// completions are charged without a fresh check. An exhausted pool stops a
// run cleanly with [TruncBudgetPool]; [ErrBudgetExhausted] is the check
// failure. A nil pool is the default and means unlimited.
//
// # Transcripts, observers, and replay
//
// Every run emits [llmkit.Event] values through one observer chain: the
// in-memory [Transcript] first (it backs [Outcome.Transcript] and always
// exists), then the single durable sink installed with [WithObserver] —
// [JSONL] streams one JSON line per event to a file per run. Events carry
// the run's identity ([WithRunID], [llmkit.RunID]; [Continue] chains record
// ParentRunID) and the 1-based turn as Step. Transcripts serialize to JSONL
// with [Transcript.SaveJSONL], load back with [LoadJSONL], and replay
// offline through [NewReplayClient] — the read-side [Source] interface, so
// replay works from any sink; the client's [ReplayClient.Tools] serve the
// recorded tool results and [ReplayClient.Err] reports a diverged replay
// (assert it after every replayed run). Do not wrap a Runner's client with
// [llmkit.Observe]: the Runner already emits the Completion event for every
// completion it makes, and a decorator would double it.
package agent

import (
	"fmt"

	"github.com/dpoage/llmkit"
)

// Default limits applied when a Runner is constructed with zero-value limits.
const (
	// DefaultMaxIterations bounds the number of model turns in a single run.
	DefaultMaxIterations = 20
	// DefaultTokenBudget bounds cumulative input+output tokens across a run.
	// Zero in Limits means "use this default"; Unlimited (-1) means no limit.
	DefaultTokenBudget int64 = 1_000_000
	// Unlimited signals that a Limits field has no cap. Distinct from zero,
	// which means "use the package default". Applies to MaxIterations
	// (use int(Unlimited)) and TokenBudget.
	Unlimited int64 = -1
)

// Limits bounds a single [Runner.Run]. The zero value is valid and resolves to
// the package defaults.
type Limits struct {
	// MaxIterations caps the number of model turns. Zero uses
	// DefaultMaxIterations. A negative value disables the iteration cap.
	MaxIterations int
	// TokenBudget caps cumulative input+output tokens (summed from llmkit.Usage
	// across every completion in the run). Zero uses DefaultTokenBudget. A
	// negative value disables the budget.
	TokenBudget int64
	// CacheReadWeight discounts cache-read input tokens in the per-run budget
	// check and in the shared budget pool's charge (see [WithBudgetPool]).
	// Valid range is (0,1]; a zero or negative value resolves to 1.0 (no
	// discount).
	CacheReadWeight float64
	// HistoryTokenBudget enables threshold-triggered history compaction. When
	// the estimated size of the message history (bytes/4 over message content
	// and tool-call arguments) exceeds this many tokens, the Runner compacts
	// once: tool-result content older than the most recent few turns is
	// replaced with short stubs. Compaction preserves the task message, every
	// assistant turn (the reasoning chain), and tool_call/tool_result ID
	// pairing. The threshold then re-arms at a higher level, so compaction
	// fires at most a few bounded times per run.
	//
	// Compaction trades cost against the provider's prompt cache. Each
	// firing mutates the message prefix and costs one full-price cache miss
	// that turn; cheaper append-only cache hits resume afterwards. Compaction
	// pays off only when history is large enough that the bytes reclaimed
	// over the remaining turns exceed that one-time miss. Size the budget
	// well above a typical short run; the re-arm stops repeated firings.
	//
	// Zero disables compaction (pure append-only history). A negative value also
	// disables it.
	HistoryTokenBudget int64
}

// resolve returns the effective limits, substituting defaults for zero values.
func (l Limits) resolve() Limits {
	out := l
	if out.MaxIterations == 0 {
		out.MaxIterations = DefaultMaxIterations
	}
	if out.TokenBudget == 0 {
		out.TokenBudget = DefaultTokenBudget
	}
	if out.CacheReadWeight <= 0 {
		out.CacheReadWeight = 1.0
	}
	return out
}

// TruncationReason explains why a run stopped before the model finished its
// turn. The zero value ("") means the run was not truncated; see
// [Outcome.Truncated].
type TruncationReason string

// Truncation reasons recorded in [Outcome.TruncationReason].
const (
	// TruncMaxIterations means the run hit the iteration cap before the model
	// produced a final (non-tool) answer.
	TruncMaxIterations TruncationReason = "max_iterations"
	// TruncTokenBudget means cumulative token usage exceeded the budget.
	TruncTokenBudget TruncationReason = "token_budget"
	// TruncBudgetPool means a shared budget pool (installed with
	// [WithBudgetPool]) was exhausted before this run's own per-run budget,
	// so the run stopped pre-turn. It is distinct from TruncTokenBudget so
	// the caller can tell a run that was stopped by the run-spanning ceiling
	// ("budget-stopped") from one that merely exhausted its own allowance.
	TruncBudgetPool TruncationReason = "budget_pool"
)

// Outcome is the result of a [Runner.Run]. A truncated run is still a valid
// outcome: FinalText holds the last completion's text, and the Transcript
// captures the full interaction.
//
// A truncated outcome is exactly one whose TruncationReason is non-empty;
// see [Outcome.Truncated].
type Outcome struct {
	// FinalText is the text of the last completion of the run. That completion
	// is the main-loop turn, a stitched max-tokens continuation, the forced
	// finalization turn, or the RunJSON repair completion, whichever ran last.
	// It is empty when that completion produced no text, and it never holds
	// text from an earlier turn. Present it as the run's answer verbatim, with
	// two exceptions: after a [StopReasonError] it holds the refusing turn's
	// text, not an answer; after a RunJSON repair it is the repair
	// completion's text (see [Runner.RunJSON]).
	FinalText string
	// TruncationReason is set when the run stopped because it hit a limit
	// rather than the model finishing its turn: one of the Trunc* constants.
	// Empty means the model finished cleanly.
	TruncationReason TruncationReason
	// Iterations is the number of completed model turns.
	Iterations int
	// Usage is cumulative token consumption across the run.
	Usage llmkit.Usage
	// Finalized reports whether forced finalization fired: the loop reserved
	// its last turn, injected the finalization prompt, and took one final
	// completion instead of returning dangling exploration prose. Only
	// [Runner.RunJSON] reserves that turn (the public [Runner.Run] passes no
	// finalize prompt).
	Finalized bool
	// LastStopReason is the stop reason of the final completion in the run. It is
	// StopMaxTokens when the model's last output was truncated at the token cap,
	// which JSON-expecting callers use to distinguish "truncated mid-answer" from
	// a genuine parse failure.
	LastStopReason llmkit.StopReason
	// RunID is the id of this run: minted with [llmkit.NewRunID], or the
	// caller's id via [WithRunID]. Every event of the run carries it, and
	// [Continue] records it as the next run's ParentRunID.
	RunID llmkit.RunID
	// Transcript is the full ordered record of the run. Never nil.
	Transcript *Transcript
	// Messages is the conversation state at the point the run returned: the
	// seed task, every tool call and result, and the final assistant turn,
	// with no system message. It is plumbing for [Continue]: a caller
	// driving a multi-round revision loop, or a chat REPL, passes a round's
	// Outcome back in as the next round's starting history. The model keeps
	// its prior turns instead of re-orienting from scratch. Callers that do
	// not continue a conversation can ignore this field.
	Messages []llmkit.Message
}

// Truncated reports whether the run stopped because it hit a limit rather
// than the model finishing its turn. It reports whether TruncationReason
// is set.
func (o *Outcome) Truncated() bool { return o.TruncationReason != "" }

// StopReasonError is returned by [Runner.Run] when the model's final turn
// ended with a provider error stop reason ([llmkit.StopError],
// [llmkit.StopRefusal], or [llmkit.StopContentFilter] — refusal, safety
// filter, recitation) and no tool calls. The refusal prose is carried in
// Text and is never presented as the run's answer. The partial Outcome is
// attached so callers can still inspect usage and the transcript, and can
// be threaded into [Continue] to keep the conversation going.
type StopReasonError struct {
	// StopReason is the provider stop reason that ended the run.
	StopReason llmkit.StopReason
	// Text is whatever text the refusing turn carried (often refusal prose).
	Text string
	// Outcome is the partial outcome at the point the run stopped. Never nil.
	Outcome *Outcome
}

func (e *StopReasonError) Error() string {
	return fmt.Sprintf("agent: model stopped without tool calls: stop reason %q", e.StopReason)
}

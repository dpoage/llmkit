// Package agent is a tool-call execution harness: a reusable loop that
// drives an [llmkit.Client] through a bounded set of tools until the model
// produces a final answer, runs out of iterations, or exhausts a token budget.
//
// Callers with different roles (e.g. a coder, a reviewer, a researcher) all
// instantiate the same [Runner] with different system prompts and tool
// sets. The harness itself is provider-agnostic: it speaks only the
// normalized [llmkit] vocabulary.
//
// # Tools
//
// A [Tool] declares its schema via [Tool.Def] and executes via [Tool.Run].
// Tool errors are *not* loop failures: they are fed back to the model as
// tool-result content prefixed with "ERROR:" so the model can recover (retry
// with different arguments, try another tool, or give up gracefully). Only
// infrastructure-level failures (a failed [llmkit.Client.Complete], context
// cancellation) abort the loop.
//
// # Limits and partial results
//
// The loop enforces two limits: [Limits.MaxIterations] (model turns) and
// [Limits.TokenBudget] (cumulative input+output tokens from [llmkit.Usage]).
// Exceeding either stops the loop cleanly, returning an [Outcome] with a
// non-empty [Outcome.TruncationReason]; Outcome.FinalText holds the text of
// the LAST completion of the run (empty when that completion produced no
// text) — partial results are data, not errors. Only context
// cancellation, infra failures, and [StopReasonError] return a non-nil error
// from [Runner.Run].
//
// # Transcripts
//
// Every run can be recorded as an ordered [Transcript] of events
// (requests, assistant turns, tool calls, tool results, usage). Transcripts
// serialize to JSONL and can be replayed offline through a [ReplayClient],
// which is the building block for the eval harness.
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
	// HistoryTokenBudget enables threshold-triggered history compaction. When the
	// estimated size of the growing message history (bytes/4 over message content
	// and tool-call arguments) exceeds this many tokens, the Runner compacts ONCE:
	// tool-result content older than the most recent few turns is replaced with
	// short stubs, preserving the task message, every assistant turn (the
	// reasoning chain), and tool_call/tool_result ID pairing. The threshold
	// then re-arms at a higher level so compaction fires at most a few
	// bounded times per run.
	//
	// Compaction trades cost against the provider's prompt cache: each firing
	// mutates the message prefix and therefore costs one full-price cache miss
	// that turn, after which cheaper append-only cache hits resume. It is a net
	// win only once history is large enough that the bytes reclaimed over the
	// remaining turns exceed that one-time miss — hence a budget sized well above
	// a typical short run, and the re-arm that stops it thrashing.
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
// A truncated outcome is exactly one whose TruncationReason is non-empty (see
// [Outcome.Truncated]); the type makes any other encoding unrepresentable.
// finishTruncated is the sole point that sets it.
type Outcome struct {
	// FinalText is the text of the LAST completion of the run — the
	// main-loop turn, a stitched max-tokens continuation, the forced
	// finalization turn, or the RunJSON repair completion, whichever ran
	// last. Empty when that completion produced no text; never text from an
	// earlier turn. Callers can present it as the run's answer verbatim.
	// On a run that ended in [StopReasonError], it holds the refusing
	// turn's text (refusal prose) — never present it as the answer.
	// After a RunJSON repair, it is the repair completion's text (see
	// [Runner.RunJSON]).
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
	// Transcript is the full ordered record of the run. Never nil.
	Transcript *Transcript
	// Messages is the full conversation state (system-less: user/assistant/
	// tool-result turns only) at the point the run returned, including the
	// seed task, every tool call/result, and the final assistant turn. It is
	// opaque plumbing for [Continue]: a caller driving a multi-round revision
	// loop — or a chat REPL — threads a round's Outcome back in as the next
	// round's starting history so the model keeps its prior turns instead of
	// re-orienting from scratch.
	// Callers that don't continue a conversation (the common case) can ignore
	// this field entirely.
	Messages []llmkit.Message
}

// Truncated reports whether the run stopped because it hit a limit rather
// than the model finishing its turn — i.e. whether TruncationReason is set.
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

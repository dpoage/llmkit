package agent

import (
	"context"
	"time"

	"github.com/dpoage/llmkit"
)

// Hooks is the Runner's observer surface: a struct of optional callback funcs,
// one per loop event. Every field is independent; a nil func is a no-op with
// zero overhead, and a zero Hooks value is valid.
//
// Fire points, in the order a typical run hits them:
//
//   - BeforeCompletion / AfterCompletion around EVERY client.Complete — the
//     main loop turn, a max-tokens continuation turn, a forced-finalization
//     turn, and a RunJSON repair turn. step is the runner's iteration counter
//     at fire time (0 for the first completion of a run). req is the exact
//     wire request; observe it only — mutating it is undefined. AfterCompletion
//     receives resp == nil together with a non-nil err when the completion
//     failed.
//   - ToolStart / ToolEnd around each Tool.Run. ToolEnd carries the final
//     Result, IsError, and measured Duration; ToolStart leaves those zero.
//     A model naming an unregistered tool never reaches Tool.Run, so neither
//     hook fires for it. Step matches the transcript's tool-result event.
//   - ToolHealth when a tool returns a *ToolHealthError (a genuine
//     harness/infra failure) — but not for ordinary model-recoverable tool
//     errors, and never for a failure caused by an already-cancelled context.
//   - Compaction when history compaction actually pruned (not on a
//     threshold crossing with nothing to reclaim). Step is the completion
//     that will consume the compacted history.
//   - Repair at the start of RunJSON's single repair pass.
//   - Finalize when the reserved forced-finalization turn is taken; reason is
//     the stop condition (a Trunc* constant) that triggered it.
//   - TranscriptError on transcript streaming write/encode/open failures.
//     Streaming stays best-effort — the failure never fails the run.
//
// Concurrency: with WithParallelTools set, ToolStart/ToolEnd fire
// concurrently from the per-call goroutines; with concurrent Run calls on one
// Runner (which is safe), every hook can fire concurrently across runs. Hook
// functions must therefore be safe for concurrent use — synchronize their own
// state.
type Hooks struct {
	// BeforeCompletion fires immediately before each client.Complete.
	BeforeCompletion func(ctx context.Context, step int, req *llmkit.Request)
	// AfterCompletion fires immediately after each client.Complete returns.
	AfterCompletion func(ctx context.Context, step int, req *llmkit.Request, resp *llmkit.Response, err error)
	// ToolStart fires immediately before each Tool.Run.
	ToolStart func(ctx context.Context, ev ToolEvent)
	// ToolEnd fires immediately after each Tool.Run, with Result, IsError,
	// and Duration set.
	ToolEnd func(ctx context.Context, ev ToolEvent)
	// ToolHealth fires when a tool returns a *ToolHealthError.
	ToolHealth func(ctx context.Context, tool string, he *ToolHealthError)
	// Compaction fires when a compaction pass actually pruned history.
	Compaction func(ctx context.Context, ev CompactionEvent)
	// Repair fires at the start of a RunJSON repair pass.
	Repair func(ctx context.Context)
	// Finalize fires when the reserved forced-finalization turn is taken.
	Finalize func(ctx context.Context, reason string)
	// TranscriptError fires on transcript streaming open/encode/write failures.
	TranscriptError func(err error)
}

// ToolEvent is one tool call's lifecycle, as delivered to [Hooks.ToolStart]
// and [Hooks.ToolEnd]. ToolStart sets Step and Call only; ToolEnd additionally
// sets Result, IsError, and Duration. The raw [llmkit.ToolCall] is carried
// verbatim: consumers do their own tool-name to structured-activity mapping.
type ToolEvent struct {
	// Step is the runner iteration whose completion requested the call.
	Step int
	// Call is the model's tool call, unmodified.
	Call llmkit.ToolCall
	// Result is the tool-result text fed back to the model (including the
	// "ERROR: " prefix on failure).
	Result string
	// IsError reports whether Result is an error rendering.
	IsError bool
	// Duration is the measured wall time of Tool.Run.
	Duration time.Duration
}

// CompactionEvent describes one pruning compaction pass, as delivered to
// [Hooks.Compaction]. BeforeTokens and AfterTokens come from the same
// bytes/4 estimate the compaction trigger uses, so Before - After is the
// reclaimed estimate; Pruned is the number of tool-result messages stubbed.
type CompactionEvent struct {
	// Step is the completion that will consume the compacted history.
	Step int
	// BeforeTokens is the estimated history size before pruning.
	BeforeTokens int64
	// AfterTokens is the estimated history size after pruning.
	AfterTokens int64
	// Pruned is the number of tool-result messages replaced with stubs.
	Pruned int
}

// WithHooks registers the Runner's observer callbacks. A nil func field is a
// no-op with zero overhead. See the [Hooks] documentation for fire points and
// the concurrency contract.
func WithHooks(h Hooks) Option {
	return func(r *Runner) { r.hooks = h }
}

// WithToolTimeout applies a per-call deadline to every Tool.Run. On expiry the
// model receives an "ERROR: tool <name> timed out after <d>" tool result and
// the loop continues — the run's own context is unaffected. Zero (the
// default) means no per-tool deadline.
func WithToolTimeout(d time.Duration) Option {
	return func(r *Runner) { r.toolTimeout = d }
}

// WithParallelTools opts in to concurrent dispatch of the tool calls a single
// completion requests: one goroutine per call, bounded by the number of
// calls. Results are appended to the conversation history and the transcript
// in the model's original call order, so the wire-visible history is
// identical to sequential dispatch. Per-call failures are isolated: one tool's
// error never affects its siblings. The default (sequential) dispatch is
// unchanged.
//
// Tools that run under this option must be safe for concurrent calls, and so
// must the [Hooks] callbacks. Without this option, calls within a turn run
// sequentially exactly as before.
func WithParallelTools() Option {
	return func(r *Runner) { r.parallelTools = true }
}

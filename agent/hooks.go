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
//     turn, and a RunJSON repair turn. step is the 1-based transcript step
//     the completion is recorded under — the SAME base every hook family
//     uses: ToolEvent.Step, CompactionEvent.Step, and the transcript's
//     Event.Step all carry this number for the same turn, so consumers can
//     join on Step. req is the FINAL wire request — the exact request
//     client.Complete receives and the transcript's request event records —
//     already passed through [RequestPolicy.PrepareRequest] when one is
//     registered. Observe it only: mutating req here is undefined, and
//     request shaping (messages, sampling, tool choice) belongs to
//     [RequestPolicy]. AfterCompletion receives resp == nil together with
//     a non-nil err when the completion failed.
//   - ToolStart / ToolEnd around each Tool.Run. ToolEnd carries the final
//     Result, IsError, and measured Duration; ToolStart leaves those zero.
//     A model naming an unregistered tool never reaches Tool.Run, so neither
//     hook fires for it. A panicking Tool.Run is recovered by the harness in
//     BOTH dispatch modes and ToolEnd fires with the rendered panic result
//     ("ERROR: tool <name> panicked: …", IsError=true). Step matches the
//     transcript's tool-result event.
//
// Hook functions are part of the harness, not the model conversation: a
// panic raised INSIDE any Hooks callback is a harness bug. It is never
// rendered to the model as a tool result. Sequential dispatch propagates it
// inline; under WithParallelTools the per-call goroutine recovers it and the
// loop re-panics with the original value after all sibling calls finish —
// the run aborts with that panic in both modes.
//
//   - ToolHealth when a tool returns a *ToolHealthError (a genuine
//     harness/infra failure) — but not for ordinary model-recoverable tool
//     errors, and never for a failure caused by an already-cancelled context.
//   - Compaction when history compaction actually pruned (not on a
//     threshold crossing with nothing to reclaim). Step is the transcript
//     step of the completion that will consume the compacted history — the
//     same number that completion's Before/AfterCompletion report.
//   - Repair at the start of RunJSON's single repair pass.
//   - Finalize when the reserved forced-finalization turn is taken; reason is
//     the stop condition (a Trunc* constant) that triggered it.
//   - TranscriptError on transcript streaming write/encode/open failures.
//     Streaming stays best-effort — the failure never fails the run.
//
// Invocation is SYNCHRONOUS: each hook runs inline on the goroutine that
// reaches the fire point (the loop goroutine, or the per-call goroutine for
// ToolStart/ToolEnd under WithParallelTools). A slow hook stalls the run —
// and, under WithParallelTools, the tool call it wraps.
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
	Finalize func(ctx context.Context, reason TruncationReason)
	// TranscriptError fires on transcript streaming open/encode/write failures.
	TranscriptError func(err error)
}

// ToolEvent is one tool call's lifecycle, as delivered to [Hooks.ToolStart]
// and [Hooks.ToolEnd]. ToolStart sets Step and Call only; ToolEnd additionally
// sets Result, IsError, and Duration. The raw [llmkit.ToolCall] is carried
// verbatim: consumers do their own tool-name to structured-activity mapping.
type ToolEvent struct {
	// Step is the 1-based transcript step (Event.Step) of the tool-result
	// event this call produced — the same number the completion hooks
	// (Before/AfterCompletion) reported for the turn that requested the call.
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
	// Step is the 1-based transcript step (Event.Step) of the completion that
	// will consume the compacted history — the same number that completion's
	// Before/AfterCompletion report.
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

// WithToolTimeout applies a per-call deadline to every Tool.Run. The deadline
// derives from the run's own context, so cancelling the run also cancels
// in-flight tools. On expiry the model receives an
// "ERROR: tool <name> timed out after <d>" tool result and the loop
// continues — the run's own context is unaffected. Zero (the default) means
// no per-tool deadline.
//
// The deadline is advisory for tools that ignore their context: Go cannot
// abandon a running Tool.Run, so a ctx-ignoring tool runs to completion — a
// success arriving after the deadline is passed through verbatim, and a
// failure arriving after the deadline is reported as the timeout.
func WithToolTimeout(d time.Duration) Option {
	return func(r *Runner) { r.toolTimeout = d }
}

// WithParallelTools opts in to concurrent dispatch of the tool calls a
// single completion requests: one goroutine per call, bounded by the number
// of calls. Results are appended to the conversation history and the
// transcript in the model's original call order, so the wire-visible history
// is identical to sequential dispatch. Per-call failures are isolated: one
// tool's error never affects its siblings.
//
// Failure semantics are IDENTICAL to sequential dispatch — toggling this
// option never changes who sees a panic. A panicking Tool.Run is recovered
// in the call's goroutine and rendered as that call's error result
// ("ERROR: tool <name> panicked: …"), so siblings complete normally and the
// run continues — the same rendering the sequential path produces. A
// panicking [Hooks] callback is a harness bug, not tool data: it is
// recovered in the per-call goroutine and re-panicked with the original
// value after all sibling calls finish, aborting the run exactly as a hook
// panic does sequentially — no goroutine leaks, no tool_result recorded for
// the interrupted turn. See [Tool.Run] and [Hooks].
//
// Tools that run under this option must be safe for concurrent calls (see
// [Tool.Run]), and so must the [Hooks] callbacks. Without this option, calls
// within a turn run sequentially exactly as before.
func WithParallelTools() Option {
	return func(r *Runner) { r.parallelTools = true }
}

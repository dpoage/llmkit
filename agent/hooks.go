package agent

import (
	"context"
	"time"

	"github.com/dpoage/llmkit"
)

// Hooks is the Runner's callback surface: a struct of optional callback funcs,
// one per live loop event. Every field is independent; a nil func is a no-op
// with zero overhead, and a zero Hooks value is valid. Hooks cover only what
// the loop does around a completion or a tool call; facts about the run as a
// whole — compaction, steering, finalization — are events on the run's
// observer chain ([WithObserver]) and fields of [Outcome].
//
// Fire points, in the order a typical run hits them:
//
//   - BeforeCompletion / AfterCompletion around EVERY client.Complete — the
//     main loop turn, a max-tokens continuation turn, a forced-finalization
//     turn, and a RunJSON repair turn. step is the 1-based turn number and
//     the SAME base every hook family uses: ToolEvent.Step and the step the
//     Runner stamps on the run's llmkit.Event values all carry this number
//     for the same turn, so consumers can join on Step. req is the FINAL
//     wire request — the exact request client.Complete receives and the
//     completion event's request records — already passed through
//     [RequestPolicy.PrepareRequest] when one is registered. Observe it
//     only: mutating req here is undefined,
//     and request shaping (messages, sampling, tool choice) belongs to
//     [RequestPolicy]. AfterCompletion receives resp == nil together with
//     a non-nil err when the completion failed.
//   - Delta, when set, fires once per incremental fragment of every
//     completion: the Runner issues the completion via [llmkit.Stream] —
//     the client's native stream when it implements
//     [llmkit.StreamingClient], deltas synthesized from the finished
//     [llmkit.Response] otherwise — so a non-streaming client still fires
//     the hook and the completion the loop records is byte-identical to
//     the un-streamed path. step matches BeforeCompletion and
//     AfterCompletion for the same turn. Invocation is synchronous on the
//     loop goroutine, between BeforeCompletion and AfterCompletion; a
//     slow hook stalls the stream.
//   - ToolStart / ToolEnd around each Tool.Run. ToolEnd carries the final
//     Result, IsError, and measured Duration; ToolStart leaves those zero.
//     A model naming an unregistered tool never reaches Tool.Run, so neither
//     hook fires for it; the same goes for a call a [ToolPolicy] denied. The
//     run's ToolRun events also record unknown-tool calls and policy denials
//     (a denial carries Denied and DenyReason, not a result). A run that
//     ends mid-turn, for any reason including a cancelled context or a
//     panic, can leave requested calls without a ToolRun event. A panicking
//     Tool.Run is recovered by the harness in BOTH dispatch modes and
//     ToolEnd fires with the rendered panic result ("ERROR: tool <name>
//     panicked: …", IsError=true). Step matches the tool_run event.
//   - ToolHealth when a tool returns a *ToolHealthError (a genuine
//     harness/infra failure) — but not for ordinary model-recoverable tool
//     errors, and never for a failure caused by an already-cancelled context.
//
// Composition: [WithHooks] appends. With several registrations, callbacks
// fire in registration order; a nil field in one registration does not
// suppress another registration's callback.
//
// Invocation is synchronous: each hook runs inline on the goroutine that
// reaches the fire point (the loop goroutine, or the per-call goroutine for
// ToolStart, ToolEnd, and ToolHealth in a turn of two or more calls under
// WithParallelTools). A slow hook stalls the run — and, under
// WithParallelTools, the tool call it wraps.
//
// Hook panics are harness bugs, never tool data: a panic inside any callback
// is never rendered as a tool result. The remaining callbacks registered for
// that event do not fire. In a turn of two or more calls under
// WithParallelTools, a panic in ToolStart, ToolEnd, or ToolHealth is
// recovered in the per-call goroutine, and the loop re-panics with the
// original value after all sibling calls finish.
//
// Concurrency: in a turn of two or more calls under WithParallelTools,
// ToolStart, ToolEnd, and ToolHealth fire concurrently from the per-call
// goroutines; with concurrent Run calls on one Runner (which is safe), every
// hook can fire concurrently across runs. Hook functions must therefore be
// safe for concurrent use — synchronize their own state.
//
// Every hook receives a context without the turn's completion span
// (Step set): the Runner claims the span only on the context it hands
// the client, and Delta and AfterCompletion, which fire after that
// claim, still receive the pre-claim context. Tools receive the loop
// context, likewise without the span. Which model calls made from a
// hook or a tool emit their own Completion follows the emission rule on
// [llmkit.Observer].
type Hooks struct {
	// BeforeCompletion fires immediately before each client.Complete.
	BeforeCompletion func(ctx context.Context, step int, req *llmkit.Request)
	// AfterCompletion fires immediately after each client.Complete returns.
	AfterCompletion func(ctx context.Context, step int, req *llmkit.Request, resp *llmkit.Response, err error)
	// Delta fires once per incremental fragment of every completion; the
	// Runner streams via [llmkit.Stream] only when this is set.
	Delta func(ctx context.Context, step int, d llmkit.Delta)
	// ToolStart fires immediately before each Tool.Run.
	ToolStart func(ctx context.Context, ev ToolEvent)
	// ToolEnd fires immediately after each Tool.Run, with Result, IsError,
	// and Duration set.
	ToolEnd func(ctx context.Context, ev ToolEvent)
	// ToolHealth fires when a tool returns a *ToolHealthError.
	ToolHealth func(ctx context.Context, tool string, he *ToolHealthError)
}

// ToolEvent is one tool call's lifecycle, as delivered to [Hooks.ToolStart]
// and [Hooks.ToolEnd]. ToolStart sets Step and Call only; ToolEnd additionally
// sets Result, IsError, and Duration. Consumers do their own tool-name to
// structured-activity mapping; see [ToolEvent.Call] for how the dispatched
// call relates to the model's.
type ToolEvent struct {
	// Step is the 1-based transcript step (Event.Step) of the tool-result
	// event this call produced — the same number the completion hooks
	// (Before/AfterCompletion) reported for the turn that requested the call.
	Step int
	// Call is the tool call as dispatched: rewritten by a [ToolPolicy]
	// when it modified Arguments; otherwise the model's call.
	Call llmkit.ToolCall
	// Result is the tool-result text fed back to the model (including the
	// "ERROR: " prefix on failure).
	Result string
	// IsError reports whether Result is an error rendering.
	IsError bool
	// Duration is the measured wall time of Tool.Run.
	Duration time.Duration
}

// WithHooks appends h to the Runner's callback registrations; it does not
// replace an earlier WithHooks. Callbacks fire in registration order, so
// independent concerns (a progress display, a health counter) each register
// their own Hooks. A nil func field is a no-op with zero overhead. See the
// [Hooks] documentation for fire points, the panic contract, and the
// concurrency contract.
func WithHooks(h Hooks) Option {
	return func(r *Runner) { r.hooks = append(r.hooks, h) }
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
// panicking [Hooks] callback is a harness bug, not tool data: a panic in
// [Hooks.ToolStart], [Hooks.ToolEnd], or [Hooks.ToolHealth] during a turn of
// two or more calls is recovered in the per-call goroutine and re-panicked
// with the original value after all sibling calls finish, aborting the run as
// a panic in the same hook does sequentially. No tool_result is recorded for
// the interrupted turn. See [Tool.Run] and [Hooks].
//
// Tools that run under this option must be safe for concurrent calls (see
// [Tool.Run]), and so must the [Hooks] callbacks. Without this option, calls
// within a turn run sequentially exactly as before.
func WithParallelTools() Option {
	return func(r *Runner) { r.parallelTools = true }
}

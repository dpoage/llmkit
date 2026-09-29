package agent

import (
	"context"

	"github.com/dpoage/llmkit"
)

// ToolPolicy decides whether a model-requested tool call runs. A policy
// allows a call, denies it, or rewrites its arguments. [Hooks] observe tool
// calls; ToolPolicy decides them.
//
// The Runner calls Authorize once per registered tool call, in the model's
// call order, on the loop goroutine. All calls of a turn are authorized
// before the first [Tool.Run] of that turn starts, in both dispatch modes.
// Under [WithParallelTools] an interactive policy (an "allow this?" prompt)
// therefore never overlaps tool execution. A call that names an
// unregistered tool never reaches the policy; it keeps the "unknown tool"
// error path. A nil policy allows every call and copies nothing.
//
// ctx is the run's context. If ctx is cancelled, the run aborts: calls not
// yet authorized are never dispatched; each renders
// "ERROR: tool <name> not run: <context error>" so the record never mistakes
// a cancelled run for a policy denial. [WithToolTimeout] applies to Tool.Run
// only, not to Authorize; a policy that blocks must honor ctx itself.
//
// Authorize may assign a new value to call.Arguments. Tool.Run receives the
// new arguments, and [ToolEvent.Call] carries them in [Hooks.ToolStart] and
// [Hooks.ToolEnd]. The assistant turn in the conversation history and the
// transcript keep the model's original arguments. Changes to call.Name or
// call.ID are ignored.
//
// A non-nil error denies the call. Tool.Run does not run, the model
// receives the tool result "ERROR: tool <name> denied: <err>" with IsError
// set, and the run continues. The error text is sent to the model
// verbatim. ToolStart, ToolEnd, and ToolHealth do not fire for a denied
// call; the run still records the denial as a ToolRun event (Denied with
// the reason, no result).
//
// A Runner is safe for concurrent Run calls, so a policy shared across
// runs is called concurrently and must synchronize its own state. The
// model-order serialization holds within one run. A panic inside Authorize
// aborts the run and is never rendered to the model.
//
// ToolPolicy does not rewrite tool results. Wrap the [Tool] to do that.
type ToolPolicy interface {
	// Authorize allows call, denies it with a non-nil error, or rewrites
	// call.Arguments before allowing it.
	Authorize(ctx context.Context, call *llmkit.ToolCall) error
}

// ToolPolicyFunc adapts a function to [ToolPolicy].
type ToolPolicyFunc func(ctx context.Context, call *llmkit.ToolCall) error

// Authorize calls f(ctx, call).
func (f ToolPolicyFunc) Authorize(ctx context.Context, call *llmkit.ToolCall) error {
	return f(ctx, call)
}

// WithToolPolicy installs p as the Runner's [ToolPolicy]. A nil p allows
// every call.
func WithToolPolicy(p ToolPolicy) Option {
	return func(r *Runner) { r.toolPolicy = p }
}

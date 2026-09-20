package agent

import (
	"bytes"
	"context"
	"fmt"

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
// error path.
//
// ctx is the run's context. If ctx is cancelled, the run aborts: calls not
// yet authorized are denied with the context error and never dispatched.
// [WithToolTimeout] applies to Tool.Run only, not to Authorize; a policy
// that blocks must honor ctx itself.
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
// call; the transcript still records the tool result.
//
// A Runner is safe for concurrent Run calls, so a policy shared across
// runs is called concurrently and must synchronize its own state. The
// model-order serialization holds within one run. A panic inside Authorize
// propagates out of [Runner.Run] with its original value in both dispatch
// modes; the model never sees it.
//
// ToolPolicy does not rewrite tool results. Wrap the [Tool] to do that.
//
// A nil policy allows every call and copies nothing.
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

// authorizeCalls consults the policy once per call in model order on the
// caller's goroutine, returning the calls to dispatch (a private copy when
// a policy is installed) and the denied indexes. Denied results are written
// straight into results[i] so dispatch treats them like executed results.
// Cancellation denies every remaining registered call with the context
// error so neither mode can run a never-authorized call.
func (r *Runner) authorizeCalls(ctx context.Context, calls []llmkit.ToolCall, results []toolResult) ([]llmkit.ToolCall, []bool) {
	if r.toolPolicy == nil {
		return calls, nil
	}
	// Clone the argument bytes: the policy rewrites Arguments in place, so
	// neither the copy nor the RawMessage backing may alias the history.
	dispatch := make([]llmkit.ToolCall, len(calls))
	copy(dispatch, calls)
	denied := make([]bool, len(calls))
	for i := range dispatch {
		dispatch[i].Arguments = bytes.Clone(dispatch[i].Arguments)
		// Unregistered names never reach the policy; runTool owns that path.
		if _, ok := r.tools.lookup(dispatch[i].Name); !ok {
			continue
		}
		if ctx.Err() != nil {
			// Deny with the context error so the slice stays full-length and
			// no call past this point can be authorized or dispatched.
			denied[i] = true
			results[i] = toolResult{
				result:     toolError(fmt.Errorf("tool %s denied: %w", calls[i].Name, ctx.Err())),
				isErr:      true,
				denied:     true,
				denyReason: ctx.Err().Error(),
			}
			continue
		}
		if err := r.toolPolicy.Authorize(ctx, &dispatch[i]); err != nil {
			denied[i] = true
			results[i] = toolResult{
				result:     toolError(fmt.Errorf("tool %s denied: %w", calls[i].Name, err)),
				isErr:      true,
				denied:     true,
				denyReason: err.Error(),
			}
			continue
		}
		// Only Arguments is part of the rewrite contract: revert Name/ID.
		dispatch[i].Name = calls[i].Name
		dispatch[i].ID = calls[i].ID
	}
	return dispatch, denied
}

package agent

import (
	"bytes"
	"context"
	"fmt"

	"github.com/dpoage/llmkit"
)

// ToolPolicy is the Runner's permission gate for model-requested tool calls:
// the decision seam behind [Hooks], which only observes. A policy may ALLOW
// a call, DENY it, or REWRITE its arguments before any [Tool.Run] of the
// turn dispatches.
//
// Authorize is consulted once per model-requested call, in the model's call
// order, on the loop goroutine, and for ALL calls of a turn BEFORE the first
// Tool.Run is dispatched — so under [WithParallelTools] an interactive
// Authorize (an "allow this?" prompt) never races the fan-out. ctx is the
// run's own context: cancelling it aborts the run instead of answering the
// model. [WithToolTimeout] does NOT cover Authorize — the per-call deadline
// applies to Tool.Run only — so a policy that can block must honor ctx
// itself.
//
// Authorize may rewrite call.Arguments in place. The rewrite is what
// Tool.Run receives and what [ToolEvent.Call] carries in
// [Hooks.ToolStart]/[Hooks.ToolEnd]; the assistant turn already appended to
// the conversation history and the transcript's tool_result event keep the
// model's original arguments — the wire history must equal what the model
// said. Only Arguments is part of the rewrite contract: Name or ID changes
// are ignored and reverted to the model's values.
//
// A non-nil error DENIES the call: Tool.Run never runs for it, the model
// receives the tool result "ERROR: tool <name> denied: <err>" with IsError
// set, and the run continues — a denial is model-recoverable data, not a
// loop failure. ToolStart, ToolEnd, and ToolHealth do not fire for a denied
// call (the same rule as the unregistered-tool case); the transcript still
// records the tool result, and toolNameByID bookkeeping is unchanged. The
// error's text is model-facing data — it is rendered verbatim into the tool
// result — not a log line.
//
// Only REGISTERED tools reach the policy: a call naming an unregistered tool
// keeps the "unknown tool" error path and is never authorized. A policy
// value shared across concurrent Run calls on one Runner (a Runner is safe
// for concurrent runs) is consulted CONCURRENTLY — once per run, on that
// run's loop goroutine; the model-order serialization above is per-run — so
// a stateful policy must synchronize its own state, exactly like [Hooks]. A
// panic inside Authorize is a harness bug (like a hook panic): it propagates
// out of [Runner.Run] with its original value in BOTH dispatch modes and is
// never rendered to the model.
//
// Rewriting a call's RESULT (a mutable tool_result) is deliberately not part
// of this seam: a Tool decorator already composes for that.
//
// A nil policy — the zero default — allows every call with zero overhead: no
// copy of the turn's calls, no consultation.
type ToolPolicy interface {
	// Authorize decides whether call may run, possibly rewriting
	// call.Arguments in place first. A non-nil error denies the call.
	Authorize(ctx context.Context, call *llmkit.ToolCall) error
}

// ToolPolicyFunc adapts an ordinary function to the [ToolPolicy] interface:
// if f is a function with the appropriate signature, ToolPolicyFunc(f) is a
// ToolPolicy that calls f.
type ToolPolicyFunc func(ctx context.Context, call *llmkit.ToolCall) error

// Authorize calls f(ctx, call).
func (f ToolPolicyFunc) Authorize(ctx context.Context, call *llmkit.ToolCall) error {
	return f(ctx, call)
}

// WithToolPolicy installs p as the Runner's permission gate for
// model-requested tool calls. Nil (the default) allows every call. See
// [ToolPolicy] for the full contract: once per call, model order, loop
// goroutine, before the turn's first Tool.Run; a rewrite is visible to
// Tool.Run and [ToolEvent.Call] only.
func WithToolPolicy(p ToolPolicy) Option {
	return func(r *Runner) { r.toolPolicy = p }
}

// authorizeCalls is the [ToolPolicy] pre-pass at the top of
// [Runner.executeTools]: it consults the policy once per call, in the
// model's order, on the caller's goroutine, and returns the calls to
// dispatch — the input slice unchanged when no policy is installed,
// otherwise a private copy the policy may have rewritten — together with
// the denied indexes.
//
// A denied call's rendered result is written straight into results[i], so
// the dispatch loops treat it exactly like an executed result and run()'s
// append loop records the transcript tool_result and the IsError history
// message unchanged. Cancellation is checked before each Authorize: from
// the first observation of a cancelled ctx, every remaining registered
// call is denied with the context error instead of authorized or dispatched
// — in EITHER mode, so the parallel fan-out can never run a
// never-authorized call.
func (r *Runner) authorizeCalls(ctx context.Context, calls []llmkit.ToolCall, results []toolResult) ([]llmkit.ToolCall, []bool) {
	if r.toolPolicy == nil {
		return calls, nil
	}
	// Copy before consulting — and clone the argument bytes: the policy
	// rewrites call.Arguments in place, so neither the copied structs nor
	// their RawMessage backing arrays may alias resp.ToolCalls, whose bytes
	// the assistant turn already appended to the history shares.
	dispatch := make([]llmkit.ToolCall, len(calls))
	copy(dispatch, calls)
	denied := make([]bool, len(calls))
	for i := range dispatch {
		dispatch[i].Arguments = bytes.Clone(dispatch[i].Arguments)
		// Unregistered names never reach the policy: runTool keeps the
		// "unknown tool" error path (the same rule as the hooks).
		if _, ok := r.tools.lookup(dispatch[i].Name); !ok {
			continue
		}
		if ctx.Err() != nil {
			// Cancelled mid-turn: no call past this point may be authorized
			// or dispatched in EITHER mode — a never-authorized call that
			// dispatched would bypass the permission gate. Deny it with the
			// context error so the slice stays full-length and the run's
			// own ctx check returns cleanly.
			denied[i] = true
			results[i] = toolResult{
				result: toolError(fmt.Errorf("tool %s denied: %w", calls[i].Name, ctx.Err())),
				isErr:  true,
			}
			continue
		}
		if err := r.toolPolicy.Authorize(ctx, &dispatch[i]); err != nil {
			denied[i] = true
			results[i] = toolResult{
				result: toolError(fmt.Errorf("tool %s denied: %w", calls[i].Name, err)),
				isErr:  true,
			}
			continue
		}
		// Only Arguments is part of the rewrite contract: a policy
		// reassigning Name or ID would dispatch a different tool under the
		// requested name or corrupt the call/result pairing — revert both
		// to the model's values.
		dispatch[i].Name = calls[i].Name
		dispatch[i].ID = calls[i].ID
	}
	return dispatch, denied
}

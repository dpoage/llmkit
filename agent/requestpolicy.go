package agent

import (
	"context"

	"github.com/dpoage/llmkit"
)

// RequestPolicy edits the request the Runner is about to send. The Runner
// calls PrepareRequest before every client.Complete: the main loop turn,
// the max-tokens continuation turn, the forced-finalization turn, and the
// RunJSON repair turn. [Hooks.BeforeCompletion] fires after the policy and
// observes the result.
//
// The Runner builds the request first, then hands it to the policy. step is
// the 1-based transcript step of the completion, the same number every hook
// reports. System, Messages, Tools, MaxTokens, and ResponseSchema (when the
// client's StructuredOutput capability is on) are already set. Tools is nil
// on the finalization and repair turns of a RunJSON run and on every turn
// of a Runner constructed with no tools.
//
// req.Messages is a shallow clone of the loop's history. A policy may
// filter, append, or reorder messages; the change applies to this turn's
// wire request only, and the loop's history stays unchanged. The Block and
// ToolCalls slices inside each message are shared with the history: replace
// a message, do not edit its blocks or tool calls in place. Every other
// field of req is the policy's to set: Thinking, Temperature, ToolChoice,
// StopSequences, TopP, TopK, Seed. The adapters gate each field by
// capability at the wire (see [llmkit.Request]).
//
// A non-nil error aborts the run before the wire call. [Runner.Run] and
// [Runner.RunJSON] return the error wrapped as
// "agent: request policy at iteration <step>: …". No completion is issued
// and nothing is recorded for the step.
//
// A Runner is safe for concurrent Run calls, so a policy shared across runs
// is called concurrently and must synchronize its own state. Implement the
// interface directly or wrap a function in [RequestPolicyFunc].
type RequestPolicy interface {
	// PrepareRequest edits the request for step or returns an error to
	// abort the run before the wire call.
	PrepareRequest(ctx context.Context, step int, req *llmkit.Request) error
}

// RequestPolicyFunc adapts a function to [RequestPolicy]:
//
//	agent.WithRequestPolicy(agent.RequestPolicyFunc(
//		func(_ context.Context, _ int, req *llmkit.Request) error {
//			req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
//			return nil
//		}))
type RequestPolicyFunc func(ctx context.Context, step int, req *llmkit.Request) error

// PrepareRequest calls f(ctx, step, req).
func (f RequestPolicyFunc) PrepareRequest(ctx context.Context, step int, req *llmkit.Request) error {
	return f(ctx, step, req)
}

// WithRequestPolicy installs p as the Runner's [RequestPolicy]. A nil p
// installs no policy: the request goes to the client as the Runner built
// it, and the message slice is not cloned.
func WithRequestPolicy(p RequestPolicy) Option {
	return func(r *Runner) { r.requestPolicy = p }
}

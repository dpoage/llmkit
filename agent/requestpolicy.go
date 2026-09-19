package agent

import (
	"context"

	"github.com/dpoage/llmkit"
)

// RequestPolicy is the Runner's mutation seam for the wire request: run
// before EVERY client.Complete — the main loop turn, a max-tokens
// continuation turn, a forced-finalization turn, and a RunJSON repair turn
// all pass through it (the same single fire point as
// [Hooks.BeforeCompletion], which fires after it and is observe-only).
//
// PrepareRequest receives the fully built request for the completion
// recorded under step — the same 1-based transcript step every hook family
// reports — with the runner-owned fields already set: System, Messages,
// Tools, MaxTokens, and ResponseSchema when the client's StructuredOutput
// capability is on. On the finalization and repair turns Tools is nil
// (those turns forbid further investigation); a policy that needs to
// distinguish them checks Tools rather than a separate flag.
//
// Message aliasing: req.Messages is a shallow clone of the loop's history,
// so a policy may filter, append, and reorder messages freely — the edits
// shape THIS turn's wire view only, and the loop's own history is never
// changed. The Block slices inside the messages are shared with the
// history: replace a message wholesale instead of mutating its blocks in
// place. Every other field of *req is likewise the policy's to set —
// Thinking, Temperature, ToolChoice, StopSequences, TopP, TopK, Seed — each
// capability-gated by the adapters at the wire boundary (see
// [llmkit.Request]).
//
// A non-nil error aborts the run BEFORE any wire call: [Runner.Run] and
// [Runner.RunJSON] return it wrapped as
// "agent: request policy at iteration <step>: …", with no completion issued
// and nothing recorded for the step. This is the counterpart of the hook
// panic rule: a policy error is data the harness author chose to surface,
// so it comes back as an error, not a panic.
//
// A Runner is safe for concurrent Run calls, so a policy may fire
// concurrently across runs; a stateful policy must synchronize its own
// state. A policy written for one harness drops into another unchanged:
// implement the single method directly, or wrap a function in
// [RequestPolicyFunc].
type RequestPolicy interface {
	// PrepareRequest may rewrite any field of the request about to be sent
	// for step, or return an error to abort the completion before the wire
	// call. See [RequestPolicy] for the full contract.
	PrepareRequest(ctx context.Context, step int, req *llmkit.Request) error
}

// RequestPolicyFunc adapts a plain function to [RequestPolicy], so a static
// per-turn tweak needs no named type:
//
//	agent.WithRequestPolicy(agent.RequestPolicyFunc(
//		func(_ context.Context, _ int, req *llmkit.Request) error {
//			req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
//			return nil
//		}))
type RequestPolicyFunc func(ctx context.Context, step int, req *llmkit.Request) error

// PrepareRequest calls f — see [RequestPolicy] for the contract.
func (f RequestPolicyFunc) PrepareRequest(ctx context.Context, step int, req *llmkit.Request) error {
	return f(ctx, step, req)
}

// WithRequestPolicy registers p as the Runner's [RequestPolicy]. A nil p —
// like leaving the option unset — runs no policy: the request goes to the
// client exactly as the runner built it, with no clone of the message slice
// and zero overhead.
func WithRequestPolicy(p RequestPolicy) Option {
	return func(r *Runner) { r.requestPolicy = p }
}

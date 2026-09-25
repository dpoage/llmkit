package provider

import (
	"context"

	llmkit "github.com/dpoage/llmkit"
)

// serializingClient adapts a parallel-tool-call agent loop to a model
// that does not support parallel tool calls. When the underlying model
// returns more than one tool call in a single response, only the first
// is surfaced to the caller; the rest are dropped so the caller never
// has to satisfy parallel calls. The model re-requests any further
// tools on the next turn once it sees the first result, which
// serializes the round-trips.
type serializingClient struct {
	inner llmkit.Client
}

// serializeStage wraps c so each response carries at most one tool call.
// If c already supports parallel tool calls it is returned unchanged —
// [Wrap] decorates unconditionally and lets capability drive behavior.
func serializeStage(c llmkit.Client) llmkit.Client {
	if c.Capabilities().ParallelToolCalls {
		return c
	}
	return &serializingClient{inner: c}
}

// Capabilities reports ParallelToolCalls=false, since this wrapper
// guarantees at-most-one tool call regardless of the underlying model;
// other capabilities pass through.
func (s *serializingClient) Capabilities() llmkit.Capabilities {
	caps := s.inner.Capabilities()
	caps.ParallelToolCalls = false
	return caps
}

// Identity forwards the wrapped client's Identity, like every provider
// stage forwards [llmkit.Identity] (and [llmkit.StreamingClient]) from
// the client it wraps.
func (s *serializingClient) Identity() llmkit.Identity { return llmkit.IdentityOf(s.inner) }

// truncateToolCalls caps resp at its first tool call, keeping the
// normalized tool_use stop reason so the agent loop continues feeding
// results.
func truncateToolCalls(resp *llmkit.Response) {
	if len(resp.ToolCalls) > 1 {
		resp.ToolCalls = resp.ToolCalls[:1]
		resp.StopReason = llmkit.StopToolUse
	}
}

func (s *serializingClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	resp, err := s.inner.Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	truncateToolCalls(&resp)
	return resp, nil
}

// Stream streams from the wrapped client, forwarding text and thinking
// deltas, dropping tool-call deltas for any Index beyond the first, and
// truncating the final Response the same way Complete does — the delta
// stream never announces a call the caller will not see. A nil fn is
// replaced with a no-op so the inner Stream never sees a nil callback.
func (s *serializingClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	if fn == nil {
		fn = func(llmkit.Delta) error { return nil }
	}
	resp, err := llmkit.Stream(ctx, s.inner, req, func(d llmkit.Delta) error {
		if d.Kind == llmkit.DeltaToolCall && d.Index != 0 {
			return nil
		}
		return fn(d)
	})
	if err != nil {
		return resp, err
	}
	truncateToolCalls(&resp)
	return resp, nil
}

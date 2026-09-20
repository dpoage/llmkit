package llmkit

import (
	"context"
	"time"
)

// Observe wraps c so that every logical completion — one Complete call or
// one Stream call, success or failure — emits exactly one Completion event
// to obs. It mirrors [WithRecorder]'s shape and nil handling: a nil obs
// returns c unchanged.
//
// Each event carries the request as received, the final response or the
// error text, and the provider and model arguments as tags. On the Stream
// path the response is the one the inner client assembles from its own
// deltas ([Stream]'s synthesis for a client without a native stream), so
// both paths observe the same shape. Before calling the client, Observe
// mints a fresh [SpanID] and places it in the context ([WithSpan]) —
// unconditionally, never reusing one from the incoming context — so a
// wrapping retry stage's Attempt events join this Completion on SpanID and
// a nested completion (a tool calling the model) gets its own span.
// Event.Duration covers the whole call, inner retries and backoff included.
//
// The emission rule lives on [Observer]: the OUTERMOST harness layer emits
// the Completion event. Observe is that layer for bare clients; the agent
// Runner emits its own, with Step set — do not wrap a Runner-run client in
// Observe, or every completion is recorded twice.
func Observe(c Client, obs Observer, provider, model string) Client {
	if obs == nil {
		return c
	}
	return &observingClient{inner: c, obs: obs, provider: provider, model: model}
}

// observingClient is the Completion-emitting decorator. Like recordingClient
// and retryClient, it implements Stream by delegating to the package-level
// Stream, so it streams natively exactly when the wrapped client does and
// synthesizes deltas otherwise.
type observingClient struct {
	inner    Client
	obs      Observer
	provider string
	model    string
}

func (o *observingClient) Capabilities() Capabilities { return o.inner.Capabilities() }

func (o *observingClient) Complete(ctx context.Context, req Request) (Response, error) {
	ctx = WithSpan(ctx, NewSpanID())
	start := time.Now()
	resp, err := o.inner.Complete(ctx, req)
	o.emit(ctx, req, resp, err, time.Since(start))
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

func (o *observingClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	ctx = WithSpan(ctx, NewSpanID())
	start := time.Now()
	resp, err := Stream(ctx, o.inner, req, fn)
	o.emit(ctx, req, resp, err, time.Since(start))
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

// emit closes one logical completion with exactly one Completion event. A
// failed completion carries the error text and the zero Response, per the
// [CompletionEvent] contract.
func (o *observingClient) emit(ctx context.Context, req Request, resp Response, err error, d time.Duration) {
	ce := &CompletionEvent{
		Request:  req,
		Provider: o.provider,
		Model:    o.model,
	}
	if err != nil {
		ce.Err = err.Error()
	} else {
		ce.Response = resp
	}
	ev := NewEvent(ctx, KindCompletion)
	ev.Duration = d
	ev.Completion = ce
	o.obs.Observe(ctx, ev)
}

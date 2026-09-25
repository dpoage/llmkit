package llmkit

import (
	"context"
	"time"
)

// Observe wraps c so each logical completion — one Complete call or one
// Stream call, success or failure — whose ctx does not already carry a
// span emits exactly one Completion event to obs, tagged with
// [IdentityOf](c). A nil obs returns c unchanged.
//
// Ownership: Observe reads the ctx's span with [SpanFromContext] before
// calling the client. A non-empty span means an enclosing emitter (an
// outer Observe, a provider stack, or the agent Runner) already claimed
// this call, so Observe passes the client through UNCHANGED — no event,
// no new span — and any wrapped retry stage's Attempt events still
// join the enclosing Completion on that span. An empty span means this
// call is unclaimed: Observe calls [BeginCompletion] to mint a fresh
// span, places it in the ctx it hands the inner client, and emits the
// Completion event on that same ctx when the call returns — so a
// wrapping retry stage's Attempt events join this Completion.
//
// Each event carries the request as received, the final response or
// error text, and Duration covering the whole call (inner retries and
// backoff included). On the Stream path the response is the one the
// inner client assembles from its own deltas ([Stream]'s synthesis for a
// Complete-only client), so both paths observe the same shape. On
// failure the caller receives the inner client's (response, error) pair
// unchanged; only the event's Response is zeroed, per the
// [CompletionEvent] contract.
func Observe(c Client, obs Observer) Client {
	if obs == nil {
		return c
	}
	return &observingClient{inner: c, obs: obs}
}

// observingClient is the ownership-aware Completion-emitting decorator.
// It implements Stream by delegating to the package-level Stream, so it
// streams natively when the wrapped client does and synthesizes deltas
// otherwise. It forwards [IdentifiedClient] and [StreamingClient] from
// the client it wraps.
type observingClient struct {
	inner Client
	obs   Observer
}

func (o *observingClient) Capabilities() Capabilities { return o.inner.Capabilities() }

// Identity forwards the wrapped client's Identity via [IdentityOf], so
// Observe never hides the identity of a client [github.com/dpoage/llmkit/provider.New] or
// [github.com/dpoage/llmkit/provider.Wrap] built.
func (o *observingClient) Identity() Identity { return IdentityOf(o.inner) }

func (o *observingClient) Complete(ctx context.Context, req Request) (Response, error) {
	if SpanFromContext(ctx) != "" {
		return o.inner.Complete(ctx, req)
	}
	claimed := BeginCompletion(ctx)
	start := time.Now()
	resp, err := o.inner.Complete(claimed, req)
	o.emit(claimed, req, resp, err, time.Since(start))
	return resp, err
}

func (o *observingClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	if SpanFromContext(ctx) != "" {
		return Stream(ctx, o.inner, req, fn)
	}
	claimed := BeginCompletion(ctx)
	start := time.Now()
	resp, err := Stream(claimed, o.inner, req, fn)
	o.emit(claimed, req, resp, err, time.Since(start))
	return resp, err
}

// emit closes one logical completion with exactly one [CompletionEvent],
// tagged with IdentityOf(o.inner). A failed completion carries the error
// text and the zero Response, per the [CompletionEvent] contract — even
// when the inner client returned a partial response alongside the error;
// the caller still receives that partial response, unchanged.
func (o *observingClient) emit(ctx context.Context, req Request, resp Response, err error, d time.Duration) {
	identity := IdentityOf(o.inner)
	ce := &CompletionEvent{
		Request:  req,
		Provider: identity.Provider,
		Model:    identity.Model,
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

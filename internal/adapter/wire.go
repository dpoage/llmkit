package adapter

import (
	"context"
	"net/http"
	"sync/atomic"
)

// The zero-request rule. An adapter call that returns an error while zero
// requests reached the transport, with the caller's context still live, is a
// refusal: the SDK rejected the call locally (a marshal failure, a
// "streaming is required" size guard, a malformed URL) and nothing was sent,
// so the failure is the request's, not the transport's, and retrying it
// would only repeat it. It is classified ErrInvalidRequest, which
// [llmkit.Classify] treats as terminal. The rule lives here, once; the
// adapters wire it in with [WithWire], [WireTransport] and [NoResponseError]
// and never re-encode it.

// Wire records what one adapter call put on the transport: how many
// requests reached RoundTrip and the status of the last response. A Wire
// belongs to exactly one call — the adapters are shared across goroutines,
// so the recorder rides the call's context, never the adapter.
type Wire struct {
	requests atomic.Int64
	status   atomic.Int64
}

type wireKey struct{}

// WithWire returns ctx carrying a fresh Wire for one adapter call. Pass the
// returned context to the SDK so [WireTransport] finds the recorder on each
// request, and to [NoResponseError] so the rule finds it on failure.
func WithWire(ctx context.Context) (context.Context, *Wire) {
	w := &Wire{}
	return context.WithValue(ctx, wireKey{}, w), w
}

// Requests reports how many requests reached the transport, counting one
// whose RoundTrip returned an error (a dial failure was still sent).
func (w *Wire) Requests() int { return int(w.requests.Load()) }

// Status reports the HTTP status of the most recent response the transport
// returned, or 0 when none returned one. Last write wins, so a redirect
// chain leaves the final status the SDK itself saw.
func (w *Wire) Status() int { return int(w.status.Load()) }

func wireFrom(ctx context.Context) *Wire {
	w, _ := ctx.Value(wireKey{}).(*Wire)
	return w
}

// WireStatus returns the status recorded on ctx's Wire, or 0 when ctx
// carries none.
func WireStatus(ctx context.Context) int {
	if w := wireFrom(ctx); w != nil {
		return w.Status()
	}
	return 0
}

// WireTransport wraps the transport an SDK is built on so every request is
// counted, and every response status recorded, on the Wire in the request's
// context. A request whose context carries no Wire passes through
// unrecorded. A nil rt means http.DefaultTransport, resolved per request,
// exactly as an http.Client with a nil Transport resolves it.
func WireTransport(rt http.RoundTripper) http.RoundTripper {
	return &wireTransport{rt: rt}
}

type wireTransport struct {
	rt http.RoundTripper
}

func (t *wireTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	w := wireFrom(req.Context())
	if w != nil {
		// Count on entry: a request that dies dialing was still sent.
		w.requests.Add(1)
	}
	rt := t.rt
	if rt == nil {
		rt = http.DefaultTransport
	}
	resp, err := rt.RoundTrip(req)
	if resp != nil && w != nil {
		w.status.Store(int64(resp.StatusCode))
	}
	return resp, err
}

// WireClient returns a copy of base whose transport is [WireTransport]
// around base's own; a nil base is an empty http.Client. base is not
// modified.
func WireClient(base *http.Client) *http.Client {
	if base == nil {
		base = &http.Client{}
	}
	wrapped := *base
	wrapped.Transport = WireTransport(wrapped.Transport)
	return &wrapped
}

// NoResponseError classifies a failure that produced no HTTP response. When
// ctx carries a Wire that saw zero requests and the caller's context is
// still live, nothing left the process and the caller's context did not
// cause the failure: it is a refusal, a non-retryable
// *llmkit.APIError{Kind: ErrInvalidRequest, StatusCode: 0} with err
// chained. Its Message is err's text, which can quote caller input at any
// length (a malformed BaseURL), so [NormalizeSDKError] builds it and caps
// it. Every other case is [TransportError]: a failure after at least one
// request (including a dial error) keeps its transport classification, and
// a context that ended keeps the cancellation semantics TransportError
// documents. A ctx without a Wire cannot prove zero requests and takes the
// TransportError path.
func NoResponseError(provider string, ctx context.Context, err error) error {
	if w := wireFrom(ctx); w != nil && w.Requests() == 0 && ctx.Err() == nil {
		return NormalizeSDKError(provider, VendorError{
			Type:    "invalid_request_error",
			Message: "request not sent: " + err.Error(),
			Err:     err,
		})
	}
	return TransportError(provider, ctx, err)
}

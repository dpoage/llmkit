package llmkit

import (
	"context"
	"errors"
	"time"

	"github.com/dpoage/llmkit/retry"
)

// retryClient wraps a Client with exponential-backoff-with-jitter retries on
// transient failures (429/5xx, transport failures the inner client reports
// as retryable *APIErrors), honoring Retry-After when present.
// When obs is non-nil it also emits one Attempt event per attempt, failures
// included — the retry stage is the only layer that sees attempt boundaries.
type retryClient struct {
	inner    Client
	cfg      retry.Config
	obs      Observer
	provider string
	model    string
}

// WithRetry wraps c so that Complete and Stream retry transient failures per
// cfg. Capabilities is delegated unchanged. Wrapping is composable with
// WithRecorder and WithSerializedToolCalls.
//
// Failures are classified by [Classify] and run through [retry.Do], so when
// the caller's context is done as the stage returns, the error is never
// retryable: a retryable last error (a transport failure cut short by the
// caller's deadline included) is replaced by a plain error chaining
// ctx.Err() that carries its text, and a terminal one — an auth
// [*APIError], a stream callback's sentinel — is returned as-is.
func WithRetry(c Client, cfg retry.Config) Client {
	return WithRetryObserver(c, cfg, nil, "", "")
}

// WithRetryObserver wraps c exactly as [WithRetry] does and, when obs is
// non-nil, makes the retry stage emit one [KindAttempt] event per attempt —
// the wire call, or the stream plus its delta delivery — tagged with the
// provider and model arguments and numbered 1..N by the loop itself.
// Failures are events too: a failed attempt carries the error text and the
// zero Response; when that error is an [*APIError] the event also carries
// its StatusCode and RetryAfter. Events are emitted on the retry caller's
// context — not the per-attempt timeout context — so a ctx-honouring sink
// never drops the timed-out attempts and never sees a leftover per-attempt
// deadline. SpanID is inherited from the context, so the attempts join the
// Completion event their emitter minted ([Observe] or the agent Runner) on
// SpanID; Step follows the context ([WithStep]) the same way, so an attempt
// inside a Runner turn carries the enclosing turn. On final failure the
// stage returns the zero Response to the caller — the [WithRetry]
// behaviour, unchanged; [Observe], by contrast, passes the inner
// (response, error) pair through. The stage never emits a Completion
// event; the outermost harness layer owns that.
func WithRetryObserver(c Client, cfg retry.Config, obs Observer, provider, model string) Client {
	return &retryClient{inner: c, cfg: cfg, obs: obs, provider: provider, model: model}
}

func (r *retryClient) Capabilities() Capabilities { return r.inner.Capabilities() }

// Complete retries transient failures per cfg. Each attempt runs the inner
// Complete under a per-attempt RequestTimeout child context, so a stalled
// round-trip aborts as context.DeadlineExceeded. That stall is retried
// only when the inner client reports it as a retryable *APIError — what
// the adapters' TransportError produces; a bare context error (say a
// third-party Client returning ctx.Err()) is terminal under Classify.
// retry.Do invokes the action exactly once per attempt, so the local
// counter is the 1-based attempt number.
func (r *retryClient) Complete(ctx context.Context, req Request) (Response, error) {
	var resp Response
	var err error
	attempt := 0
	err = retry.Do(ctx, r.cfg, Classify, func(actx context.Context) error {
		attempt++
		start := time.Now()
		resp, err = r.inner.Complete(actx, req)
		r.observe(ctx, attempt, req, resp, err, time.Since(start))
		return err
	})
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

// Stream streams from the wrapped client under the same retry policy as
// Complete, the per-attempt RequestTimeout bounding the whole attempt. An
// error that follows an already-delivered delta is terminal and returned
// as-is while the caller's context is live: the caller holds partial
// output, and a retry would replay or diverge from it. Once the caller's
// context is done, [Classify] decides it like any other failure, so a
// retryable one is replaced by a plain error chaining ctx.Err().
func (r *retryClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	// delivered marks a delta that reached the caller inside the current
	// attempt; while the caller's ctx is live the classifier turns it into
	// a terminal error.
	delivered := false
	classify := func(err error) (time.Duration, bool, bool) {
		if delivered && ctx.Err() == nil {
			return 0, false, false
		}
		return Classify(err)
	}
	// A nil fn is replaced with a no-op so the inner Stream never sees a nil
	// callback; the wrapper still marks progress.
	if fn == nil {
		fn = func(Delta) error { return nil }
	}
	var resp Response
	var err error
	attempt := 0
	err = retry.Do(ctx, r.cfg, classify, func(actx context.Context) error {
		delivered = false // a delta from an EARLIER attempt never leaks in
		attempt++
		start := time.Now()
		resp, err = Stream(actx, r.inner, req, func(d Delta) error {
			delivered = true
			return fn(d)
		})
		r.observe(ctx, attempt, req, resp, err, time.Since(start))
		return err
	})
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

// observe reports one finished attempt to the stage's observer, when wired,
// on the retry caller's context: the per-attempt RequestTimeout context is
// frequently done by emission time, and a ctx-honouring sink must not drop
// the very attempts that timed out. The attempt's Duration is the wall time
// of the inner call — backoff between attempts belongs to no attempt. A
// failed attempt carries the error text and the zero Response; the
// successful attempt's Response is the raw adapter response, before any
// outer decorator (the tool-call serializer) adjusts it.
func (r *retryClient) observe(ctx context.Context, n int, req Request, resp Response, err error, d time.Duration) {
	if r.obs == nil {
		return
	}
	ae := &AttemptEvent{
		Attempt:  n,
		Request:  req,
		Provider: r.provider,
		Model:    r.model,
	}
	if err != nil {
		ae.Err = err.Error()
		var apiErr *APIError
		if errors.As(err, &apiErr) {
			ae.StatusCode = apiErr.StatusCode
			ae.RetryAfter = apiErr.RetryAfter
		}
	} else {
		ae.Response = resp
	}
	ev := NewEvent(ctx, KindAttempt)
	ev.Duration = d
	ev.Attempt = ae
	r.obs.Observe(ctx, ev)
}

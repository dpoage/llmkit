package provider

import (
	"context"
	"errors"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// retryClient wraps a Client with exponential-backoff-with-jitter retries
// on transient failures (429/5xx, retryable *APIErrors), honoring
// Retry-After when the failure's HasRetryAfter bit is set. When obs is
// non-nil it emits one Attempt event per attempt, failures included —
// the retry stage is the only layer that sees attempt boundaries.
// Identity is fixed at construction: [Wrap] and [New] compute it once
// and hand it to this stage, the innermost provider stage.
type retryClient struct {
	inner    llmkit.Client
	cfg      retry.Config
	obs      llmkit.Observer
	identity llmkit.Identity
}

// retryStage wraps c so Complete and Stream retry transient failures per
// cfg, and, when obs is non-nil, emits one llmkit.KindAttempt event per
// attempt — the wire call, or the stream plus its delta delivery —
// tagged with identity and numbered 1..N by the loop itself. It always
// wraps (retry policy applies whether or not an observer is attached).
func retryStage(c llmkit.Client, cfg retry.Config, obs llmkit.Observer, identity llmkit.Identity) llmkit.Client {
	return &retryClient{inner: c, cfg: cfg, obs: obs, identity: identity}
}

func (r *retryClient) Capabilities() llmkit.Capabilities { return r.inner.Capabilities() }

// Identity implements llmkit.IdentifiedClient: the retry stage is the
// innermost provider stage, so it carries the Identity [Wrap] or [New]
// computed.
func (r *retryClient) Identity() llmkit.Identity { return r.identity }

// Complete retries transient failures per cfg. Each attempt runs the
// inner Complete under a per-attempt RequestTimeout child context, so a
// stalled round-trip aborts as context.DeadlineExceeded. That stall is
// retried only when the inner client reports it as a retryable
// *APIError; a bare context error (a third-party Client returning
// ctx.Err()) is terminal under [llmkit.Classify]. retry.Do invokes the
// action exactly once per attempt, so the local counter is the 1-based
// attempt number.
func (r *retryClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	var resp llmkit.Response
	var err error
	attempt := 0
	err = retry.Do(ctx, r.cfg, llmkit.Classify, func(actx context.Context) error {
		attempt++
		start := time.Now()
		resp, err = r.inner.Complete(actx, req)
		r.observe(ctx, attempt, req, resp, err, time.Since(start))
		return err
	})
	if err != nil {
		return llmkit.Response{}, err
	}
	return resp, nil
}

// Stream streams from the wrapped client under the same retry policy as
// Complete, the per-attempt RequestTimeout bounding the whole attempt.
// An error that follows an already-delivered delta is terminal and
// returned as-is while the caller's context is live: the caller holds
// partial output, and a retry would replay or diverge from it. Once the
// caller's context is done, [llmkit.Classify] decides it like any other
// failure, so a retryable one is replaced by a plain error chaining
// ctx.Err().
func (r *retryClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	// delivered marks a delta that reached the caller inside the current
	// attempt; while the caller's ctx is live the classifier turns it into
	// a terminal error.
	delivered := false
	classify := func(err error) (time.Duration, bool, bool) {
		if delivered && ctx.Err() == nil {
			return 0, false, false
		}
		return llmkit.Classify(err)
	}
	// A nil fn is replaced with a no-op so the inner Stream never sees a
	// nil callback; the wrapper still marks progress.
	if fn == nil {
		fn = func(llmkit.Delta) error { return nil }
	}
	var resp llmkit.Response
	var err error
	attempt := 0
	err = retry.Do(ctx, r.cfg, classify, func(actx context.Context) error {
		delivered = false // a delta from an EARLIER attempt never leaks in
		attempt++
		start := time.Now()
		resp, err = llmkit.Stream(actx, r.inner, req, func(d llmkit.Delta) error {
			delivered = true
			return fn(d)
		})
		r.observe(ctx, attempt, req, resp, err, time.Since(start))
		return err
	})
	if err != nil {
		return llmkit.Response{}, err
	}
	return resp, nil
}

// observe reports one finished attempt to the stage's observer, when
// wired, on the retry caller's context: the per-attempt RequestTimeout
// context is frequently done by emission time, and a ctx-honouring sink
// must not drop the very attempts that timed out. Duration is the wall
// time of the inner call — backoff between attempts belongs to no
// attempt. A failed attempt carries the error text and the zero
// Response; the successful attempt's Response is the raw adapter
// response, before any outer decorator (the tool-call serializer)
// adjusts it. RetryAfter and HasRetryAfter carry the failed attempt's
// *APIError exactly: RetryAfter is copied only when HasRetryAfter is
// true.
func (r *retryClient) observe(ctx context.Context, n int, req llmkit.Request, resp llmkit.Response, err error, d time.Duration) {
	if r.obs == nil {
		return
	}
	ae := &llmkit.AttemptEvent{
		Attempt:  n,
		Request:  req,
		Provider: r.identity.Provider,
		Model:    r.identity.Model,
	}
	if err != nil {
		ae.Err = err.Error()
		var apiErr *llmkit.APIError
		if errors.As(err, &apiErr) {
			ae.StatusCode = apiErr.StatusCode
			if apiErr.HasRetryAfter {
				ae.RetryAfter = apiErr.RetryAfter
				ae.HasRetryAfter = true
			}
		}
	} else {
		ae.Response = resp
	}
	ev := llmkit.NewEvent(ctx, llmkit.KindAttempt)
	ev.Duration = d
	ev.Attempt = ae
	r.obs.Observe(ctx, ev)
}

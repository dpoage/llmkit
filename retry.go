package llmkit

import (
	"context"
	"time"

	"github.com/dpoage/llmkit/retry"
)

// retryClient wraps a Client with exponential-backoff-with-jitter retries on
// transient failures (429/5xx/timeouts), honoring Retry-After when present.
type retryClient struct {
	inner Client
	cfg   retry.Config
}

// WithRetry wraps c so that Complete and Stream retry transient failures per
// cfg. Capabilities is delegated unchanged. Wrapping is composable with
// WithRecorder and WithSerializedToolCalls.
func WithRetry(c Client, cfg retry.Config) Client {
	return &retryClient{inner: c, cfg: cfg}
}

func (r *retryClient) Capabilities() Capabilities { return r.inner.Capabilities() }

// Complete retries transient failures per cfg. Each attempt runs the inner
// Complete under a per-attempt RequestTimeout child context, so a stalled
// round-trip aborts as context.DeadlineExceeded and is retried like any
// other transient transport failure.
func (r *retryClient) Complete(ctx context.Context, req Request) (Response, error) {
	var resp Response
	var err error
	err = retry.Do(ctx, r.cfg, retryableClass, func(actx context.Context) error {
		resp, err = r.inner.Complete(actx, req)
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
// as-is: the caller holds partial output, and a retry would replay or
// diverge from it.
func (r *retryClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	// delivered marks a delta that reached the caller inside the current
	// attempt; the classifier turns it into a terminal error.
	delivered := false
	classify := func(err error) (time.Duration, bool, bool) {
		if delivered {
			return 0, false, false
		}
		return retryableClass(err)
	}
	// A nil fn is replaced with a no-op so the inner Stream never sees a nil
	// callback; the wrapper still marks progress.
	if fn == nil {
		fn = func(Delta) error { return nil }
	}
	var resp Response
	var err error
	err = retry.Do(ctx, r.cfg, classify, func(actx context.Context) error {
		delivered = false // a delta from an EARLIER attempt never leaks in
		resp, err = Stream(actx, r.inner, req, func(d Delta) error {
			delivered = true
			return fn(d)
		})
		return err
	})
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

// retryableClass classifies err per the shared retry policy: retryable errors
// carry any server Retry-After the error chains; everything else is terminal.
func retryableClass(err error) (time.Duration, bool, bool) {
	if !isRetryable(err) {
		return 0, false, false
	}
	after, hasAfter := retryAfter(err)
	return after, hasAfter, true
}

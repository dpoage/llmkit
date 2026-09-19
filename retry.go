package llmkit

import (
	"context"
	"math/rand/v2"
	"time"
)

// RetryConfig tunes the shared retry wrapper. The zero value is not usable;
// callers should start from DefaultRetryConfig.
type RetryConfig struct {
	// MaxAttempts is the total number of attempts (initial try + retries). Must
	// be >= 1.
	MaxAttempts int
	// BaseDelay is the first backoff interval; subsequent delays grow
	// exponentially.
	BaseDelay time.Duration
	// MaxDelay caps any single backoff interval.
	MaxDelay time.Duration
	// Jitter, in [0,1], is the fraction of each delay randomized to avoid
	// thundering herds. 0.2 means the delay is multiplied by a random factor in
	// [0.8, 1.2].
	Jitter float64

	// RequestTimeout is the per-attempt wall-clock deadline applied to
	// each inner Complete call. It bounds a single provider request so a
	// stalled HTTP round-trip that never returns does not block a
	// goroutine forever. Zero or negative means DefaultRequestTimeout.
	//
	// Applied to a child context derived from the caller's ctx, so it
	// expires as context.DeadlineExceeded on the ATTEMPT only — never
	// as parent cancellation. A timed-out attempt is treated as a
	// transient transport failure and is retried (see Complete).
	RequestTimeout time.Duration

	// sleep is an injection point for tests; nil uses a real timer.
	sleep func(ctx context.Context, d time.Duration) error
	// rng is an injection point for deterministic jitter in tests; nil uses a
	// package-level source.
	rng func() float64
}

// DefaultRequestTimeout bounds a single LLM request attempt. It is the
// backstop against a provider that accepts a request and then never responds:
// the in-flight HTTP round-trip is aborted via context, the attempt fails with
// context.DeadlineExceeded, and the retry loop tries again rather than blocking
// indefinitely.
//
// 5 minutes is chosen to comfortably exceed a legitimate slow completion
// (large reasoning-model responses with big output budgets can take a
// couple of minutes) while staying well below multi-minute, no-progress
// stalls. It is per-attempt, so the worst case before giving up is
// roughly MaxAttempts * (RequestTimeout + backoff).
const DefaultRequestTimeout = 5 * time.Minute

// DefaultRetryConfig returns sensible defaults: 4 attempts, 500ms base,
// 30s cap, 20% jitter, and a 5m per-attempt request timeout.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    4,
		BaseDelay:      500 * time.Millisecond,
		MaxDelay:       30 * time.Second,
		Jitter:         0.2,
		RequestTimeout: DefaultRequestTimeout,
	}
}

// retryClient wraps a Client with exponential-backoff-with-jitter retries on
// transient failures (429/5xx/timeouts), honoring Retry-After when present.
type retryClient struct {
	inner Client
	cfg   RetryConfig
}

// WithRetry wraps c so that Complete and Stream retry transient failures per
// cfg. Capabilities is delegated unchanged. Wrapping is composable with
// WithRecorder and WithSerializedToolCalls.
func WithRetry(c Client, cfg RetryConfig) Client {
	if cfg.MaxAttempts < 1 {
		cfg.MaxAttempts = 1
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = DefaultRequestTimeout
	}
	// Clamp Jitter to [0,1]: values outside this range produce nonsensical
	// backoff factors (negative delay or unbounded amplification).
	if cfg.Jitter < 0 {
		cfg.Jitter = 0
	} else if cfg.Jitter > 1 {
		cfg.Jitter = 1
	}
	return &retryClient{inner: c, cfg: cfg}
}

func (r *retryClient) Capabilities() Capabilities { return r.inner.Capabilities() }

func (r *retryClient) Complete(ctx context.Context, req Request) (Response, error) {
	return r.retryLoop(ctx, func() (Response, error, bool) {
		resp, err := r.attempt(ctx, req)
		return resp, err, false
	})
}

// Stream streams from the wrapped client under the same retry policy as
// Complete, the per-attempt RequestTimeout bounding the whole attempt. An
// error that follows an already-delivered delta is terminal and is returned
// as-is, the way Complete returns a non-retryable error: the caller holds
// partial output, and a retry would replay or diverge from it.
func (r *retryClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	return r.retryLoop(ctx, func() (Response, error, bool) {
		return r.streamAttempt(ctx, req, fn)
	})
}

// retryLoop runs run until it succeeds, retrying transient failures with
// backoff and honoring Retry-After. run's delivered result marks a terminal
// error: output already reached the caller, so no further attempt may run.
func (r *retryClient) retryLoop(ctx context.Context, run func() (resp Response, err error, delivered bool)) (Response, error) {
	var lastErr error
	for attempt := range r.cfg.MaxAttempts {
		if attempt > 0 {
			delay := r.backoff(attempt, lastErr)
			if err := r.doSleep(ctx, delay); err != nil {
				// Context cancelled while waiting; surface the last real error if
				// we have one, otherwise the context error.
				if lastErr != nil {
					return Response{}, lastErr
				}
				return Response{}, err
			}
		}

		resp, err, delivered := run()
		if err == nil {
			return resp, nil
		}
		if delivered {
			return Response{}, err
		}
		lastErr = err

		// Never retry once the CALLER'S context is done. We check the parent ctx
		// (not the per-attempt child) so a per-attempt RequestTimeout — which
		// expires only the child as context.DeadlineExceeded — is NOT mistaken
		// for parent cancellation. Parent cancellation/deadline aborts the loop
		// here; a stalled-request timeout falls through to the retryable check
		// below and is retried like any other transient transport failure.
		if ctx.Err() != nil {
			return Response{}, err
		}
		if !isRetryable(err) {
			return Response{}, err
		}
	}
	return Response{}, lastErr
}

// attempt runs one inner Complete under a per-attempt wall-clock deadline
// derived from ctx. The child bounds a single provider request so a
// stalled round-trip aborts as context.DeadlineExceeded instead of
// blocking forever. cancel is called explicitly before returning — never
// via defer-in-loop — so the timer is released on every path without
// leaking across iterations.
//
// Because the deadline lives on the child, parent cancellation
// semantics are untouched: the caller's loop distinguishes them by
// inspecting the parent ctx.Err() directly.
func (r *retryClient) attempt(ctx context.Context, req Request) (Response, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, r.cfg.RequestTimeout)
	defer cancel()
	return r.inner.Complete(attemptCtx, req)
}

// streamAttempt runs one inner stream attempt under a per-attempt wall-clock
// deadline derived from ctx (see attempt). delivered reports whether at
// least one delta reached fn before the attempt ended; Stream treats such
// an attempt's error as terminal. A nil fn is replaced with a no-op so the
// inner Stream never sees a nil callback; the wrapper still marks progress.
func (r *retryClient) streamAttempt(ctx context.Context, req Request, fn func(Delta) error) (resp Response, err error, delivered bool) {
	if fn == nil {
		fn = func(Delta) error { return nil }
	}
	attemptCtx, cancel := context.WithTimeout(ctx, r.cfg.RequestTimeout)
	defer cancel()
	resp, err = Stream(attemptCtx, r.inner, req, func(d Delta) error {
		delivered = true
		return fn(d)
	})
	return resp, err, delivered
}

// backoff computes the delay before the given attempt (1-indexed for the first
// retry). It prefers a server-supplied Retry-After when present, otherwise uses
// exponential backoff with jitter.
func (r *retryClient) backoff(attempt int, lastErr error) time.Duration {
	if d, ok := retryAfter(lastErr); ok {
		if d > r.cfg.MaxDelay && r.cfg.MaxDelay > 0 {
			return r.cfg.MaxDelay
		}
		return d
	}

	// Exponential: base * 2^(attempt-1).
	delay := r.cfg.BaseDelay
	for range attempt - 1 {
		delay *= 2
		if r.cfg.MaxDelay > 0 && delay >= r.cfg.MaxDelay {
			delay = r.cfg.MaxDelay
			break
		}
	}
	if r.cfg.MaxDelay > 0 && delay > r.cfg.MaxDelay {
		delay = r.cfg.MaxDelay
	}

	if r.cfg.Jitter > 0 {
		// factor in [1-jitter, 1+jitter].
		factor := 1 + r.cfg.Jitter*(2*r.random()-1)
		delay = time.Duration(float64(delay) * factor)
	}
	return delay
}

func (r *retryClient) random() float64 {
	if r.cfg.rng != nil {
		return r.cfg.rng()
	}
	return rand.Float64()
}

func (r *retryClient) doSleep(ctx context.Context, d time.Duration) error {
	if r.cfg.sleep != nil {
		return r.cfg.sleep(ctx, d)
	}
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

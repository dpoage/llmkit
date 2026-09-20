package llmkit

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RetryConfig tunes a retry loop — [WithRetry]'s wrapper and [Retry] alike.
// Both clamp a zero or negative MaxAttempts to 1, so the zero value means
// "one attempt, no retries" — not usable as-is: start from
// [DefaultRetryConfig].
type RetryConfig struct {
	// MaxAttempts is the total number of attempts (initial try + retries).
	// WithRetry and Retry clamp values below 1 to 1.
	MaxAttempts int
	// BaseDelay is the first backoff interval; subsequent delays grow
	// exponentially.
	BaseDelay time.Duration
	// MaxDelay caps any single backoff interval.
	MaxDelay time.Duration
	// Jitter, in [0,1], is the fraction of each delay randomized to avoid
	// thundering herds. 0.2 means the delay is multiplied by a random factor in
	// [0.8, 1.2]. WithRetry and Retry clamp values outside [0,1].
	Jitter float64

	// RequestTimeout is the per-attempt wall-clock deadline applied to
	// each inner call. It bounds a single provider request so a
	// stalled HTTP round-trip that never returns does not block a
	// goroutine forever. Zero or negative means DefaultRequestTimeout.
	//
	// Applied to a child context derived from the caller's ctx, so it
	// expires as context.DeadlineExceeded on the ATTEMPT only — never
	// as parent cancellation. A timed-out attempt is treated as a
	// transient failure and is retried (see Complete and Retry).
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

// Retry runs fn up to cfg.MaxAttempts times and returns nil on the first
// success. It retries the failures classify marks retryable.
//
// Each attempt runs under a RequestTimeout child context derived from ctx, so
// a stalled round-trip aborts as context.DeadlineExceeded and is classified
// like any other failure. The caller's ctx is never modified; once it is
// done, Retry returns the last error immediately and never retries.
//
// classify decides which errors are worth retrying. hasRetryAfter carries a
// server-supplied delay that replaces the computed backoff for that sleep.
// Presence with a zero delay means an immediate retry; absence means the
// exponential schedule. A delay above cfg.MaxDelay is capped at MaxDelay, and
// a negative one is treated as zero.
//
// Retry applies the same normalization as [WithRetry]: MaxAttempts below 1
// becomes 1, RequestTimeout at or below 0 becomes [DefaultRequestTimeout], and
// Jitter is clamped to [0,1]. Both entry points accept the same configs.
//
// If the wait between attempts is cut short by a cancelled ctx, Retry returns
// fn's last error. Cancellation is never retried and never swallowed.
func Retry(ctx context.Context, cfg RetryConfig, classify func(err error) (retryAfter time.Duration, hasRetryAfter, retryable bool), fn func(ctx context.Context) error) error {
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
	var err error
	for attempt := 1; ; attempt++ {
		actx, cancel := context.WithTimeout(ctx, cfg.RequestTimeout)
		err = fn(actx)
		cancel()
		if err == nil {
			return nil
		}
		// Check the PARENT ctx, not the per-attempt child: a per-attempt
		// RequestTimeout expires only the child as context.DeadlineExceeded
		// and must not be mistaken for parent cancellation.
		if ctx.Err() != nil {
			return err
		}
		after, hasAfter, retryable := classify(err)
		if !retryable || attempt >= cfg.MaxAttempts {
			return err
		}
		if serr := sleepBeforeRetry(ctx, cfg, backoffDelay(cfg, attempt, after, hasAfter)); serr != nil {
			return err
		}
	}
}

// ParseRetryAfter parses a Retry-After header value: an integer delay in
// seconds or an HTTP-date. A past date, or a negative delay, clamps to 0 with
// ok true — the server supplied the header, so the wait is immediate. ok is
// false for empty or malformed values: no delay was supplied, and the caller
// falls back to its own backoff schedule.
func ParseRetryAfter(v string, now time.Time) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(v); err == nil {
		d := time.Duration(secs) * time.Second
		if d < 0 {
			d = 0
		}
		return d, true
	}
	if t, err := http.ParseTime(v); err == nil {
		d := t.Sub(now)
		if d < 0 {
			d = 0
		}
		return d, true
	}
	return 0, false
}

// backoffDelay picks the wait before retrying after the given attempt
// (1-indexed): a server Retry-After when supplied (capped at MaxDelay),
// otherwise exponential backoff with jitter.
func backoffDelay(cfg RetryConfig, attempt int, after time.Duration, hasAfter bool) time.Duration {
	if hasAfter {
		if after < 0 {
			after = 0
		}
		if cfg.MaxDelay > 0 && after > cfg.MaxDelay {
			return cfg.MaxDelay
		}
		return after
	}

	delay := cfg.BaseDelay
	for range attempt - 1 {
		delay *= 2
		// Overflow clamp: a doubled delay past int64 means an absurd wait,
		// not a negative one; zero BaseDelay stays zero.
		if delay < 0 {
			delay = 1 << 62
			break
		}
		if cfg.MaxDelay > 0 && delay >= cfg.MaxDelay {
			delay = cfg.MaxDelay
			break
		}
	}
	if cfg.MaxDelay > 0 && delay > cfg.MaxDelay {
		delay = cfg.MaxDelay
	}

	if cfg.Jitter > 0 {
		// factor in [1-jitter, 1+jitter].
		factor := 1 + cfg.Jitter*(2*jitterSource(cfg)-1)
		delay = time.Duration(float64(delay) * factor)
	}
	return delay
}

// jitterSource returns the configured deterministic source for tests, or the
// package-level random source when unset.
func jitterSource(cfg RetryConfig) float64 {
	if cfg.rng != nil {
		return cfg.rng()
	}
	return rand.Float64()
}

// sleepBeforeRetry waits d before the next attempt. cfg.sleep replaces the
// timer in tests and observes every wait, including a zero one. Without the
// hook, a wait of zero or less returns immediately; the wait returns early
// with ctx.Err() when ctx is done.
func sleepBeforeRetry(ctx context.Context, cfg RetryConfig, d time.Duration) error {
	if cfg.sleep != nil {
		return cfg.sleep(ctx, d)
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

// Complete retries transient failures per cfg. Each attempt runs the inner
// Complete under a per-attempt RequestTimeout child context, so a stalled
// round-trip aborts as context.DeadlineExceeded and is retried like any
// other transient transport failure.
func (r *retryClient) Complete(ctx context.Context, req Request) (Response, error) {
	var resp Response
	var err error
	err = Retry(ctx, r.cfg, retryableClass, func(actx context.Context) error {
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
	err = Retry(ctx, r.cfg, classify, func(actx context.Context) error {
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

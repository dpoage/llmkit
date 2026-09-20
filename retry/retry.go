// Package retry runs an operation repeatedly under one backoff schedule:
// exponential backoff with jitter, a server-supplied Retry-After delay that
// replaces the schedule and is capped at MaxDelay, and a per-attempt
// RequestTimeout deadline.
//
// The package imports only the standard library and knows nothing about
// llmkit. A caller supplies a classify function that decides, for each
// error, whether the loop retries and which delay applies. llmkit's
// WithRetry wraps [Do] around the APIError classifier; call Do directly to
// retry any other error source under the same policy.
//
// The loop runs in this order:
//
//  1. Each attempt runs fn under a RequestTimeout child context derived
//     from ctx. A stalled attempt aborts as context.DeadlineExceeded and
//     is classified like any other failure.
//  2. After a failed attempt, Do checks the parent ctx. When it is done,
//     the loop returns fn's last error immediately. Do never retries and
//     never swallows a cancellation.
//  3. classify decides whether the error is retryable. Its hasRetryAfter
//     value is a presence bit, not a nonzero check: the loop applies
//     retryAfter only when hasRetryAfter is true. A present zero delay
//     retries immediately; an absent delay falls back to the exponential
//     schedule.
//  4. The chosen delay is capped at MaxDelay (MaxDelay 0 or less is
//     uncapped); a negative Retry-After is treated as zero.
//  5. The loop stops when fn succeeds, when classify marks the error
//     terminal, or after MaxAttempts attempts, and it returns fn's last
//     error.
//
// Do normalizes cfg itself: MaxAttempts below 1 becomes 1; RequestTimeout
// at or below 0 becomes [DefaultRequestTimeout]; Jitter is clamped to
// [0, 1]. Every entry point accepts the same configs as a result.
//
// [Config.Sleep] and [Config.Rand] exist for deterministic tests. nil
// means a real timer and the package-level random source; Sleep observes
// every wait including zero ones, and Rand pins the jitter draw.
package retry

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Config tunes a retry loop — the schedule [Do] applies to fn's failures.
// Do clamps a zero or negative MaxAttempts to 1, so the zero value means
// "one attempt, no retries" — not usable as-is: start from [Default].
type Config struct {
	// MaxAttempts is the total number of attempts (initial try + retries).
	// Do clamps values below 1 to 1.
	MaxAttempts int
	// BaseDelay is the first backoff interval; subsequent delays grow
	// exponentially.
	BaseDelay time.Duration
	// MaxDelay caps any single backoff interval.
	MaxDelay time.Duration
	// Jitter, in [0,1], is the fraction of each delay randomized to avoid
	// thundering herds. 0.2 means the delay is multiplied by a random factor in
	// [0.8, 1.2]. Do clamps values outside [0,1].
	Jitter float64

	// RequestTimeout is the per-attempt wall-clock deadline applied to
	// each call of fn. It bounds a single operation so a stalled
	// round-trip that never returns does not block a goroutine forever.
	// Zero or negative means DefaultRequestTimeout.
	//
	// Applied to a child context derived from the caller's ctx, so it
	// expires as context.DeadlineExceeded on the ATTEMPT only — never
	// as parent cancellation. A timed-out attempt is classified like
	// any other failure and may be retried.
	RequestTimeout time.Duration

	// Sleep, if non-nil, replaces the timer between attempts: it observes
	// every wait, including a zero one, and returns ctx.Err() to cut the
	// wait short. It exists for deterministic tests; nil uses a real timer.
	Sleep func(ctx context.Context, d time.Duration) error
	// Rand, if non-nil, replaces the jitter source for deterministic
	// delays in tests; nil uses the package-level random source.
	Rand func() float64
}

// DefaultRequestTimeout bounds a single request attempt. It is the
// backstop against a server that accepts a request and then never responds:
// the round-trip is aborted via context, the attempt fails with
// context.DeadlineExceeded, and the retry loop retries rather than
// blocking indefinitely.
//
// 5 minutes comfortably exceeds a legitimate slow completion
// (reasoning models with large output budgets take a couple of
// minutes) while staying well below multi-minute, no-progress
// stalls. It is per-attempt, so the worst case before giving up is
// roughly MaxAttempts * (RequestTimeout + backoff).
const DefaultRequestTimeout = 5 * time.Minute

// Default returns sensible defaults: 4 attempts, 500ms base,
// 30s cap, 20% jitter, and a 5m per-attempt request timeout.
func Default() Config {
	return Config{
		MaxAttempts:    4,
		BaseDelay:      500 * time.Millisecond,
		MaxDelay:       30 * time.Second,
		Jitter:         0.2,
		RequestTimeout: DefaultRequestTimeout,
	}
}

// Do runs fn up to cfg.MaxAttempts times and returns nil on the first
// success. It retries the failures classify marks retryable.
//
// Each attempt runs under a RequestTimeout child context derived from ctx, so
// a stalled round-trip aborts as context.DeadlineExceeded and is classified
// like any other failure. The caller's ctx is never modified; once it is
// done, Do returns the last error immediately and never retries.
//
// classify decides which errors are worth retrying. hasRetryAfter carries a
// server-supplied delay that replaces the computed backoff for that sleep.
// Presence with a zero delay means an immediate retry; absence means the
// exponential schedule. A delay above cfg.MaxDelay is capped at MaxDelay, and
// a negative one is treated as zero.
//
// Do applies the normalization described on [Config]: MaxAttempts below 1
// becomes 1, RequestTimeout at or below 0 becomes [DefaultRequestTimeout],
// and Jitter is clamped to [0,1].
//
// If the wait between attempts is cut short by a cancelled ctx, Do returns
// fn's last error. Cancellation is never retried and never swallowed.
func Do(ctx context.Context, cfg Config, classify func(err error) (retryAfter time.Duration, hasRetryAfter, retryable bool), fn func(ctx context.Context) error) error {
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
		// The attempt's context is scoped to this closure so its cancel runs
		// via defer — even when fn panics — without accumulating defers in
		// the loop.
		err = func() error {
			actx, cancel := context.WithTimeout(ctx, cfg.RequestTimeout)
			defer cancel()
			return fn(actx)
		}()
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
func backoffDelay(cfg Config, attempt int, after time.Duration, hasAfter bool) time.Duration {
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
func jitterSource(cfg Config) float64 {
	if cfg.Rand != nil {
		return cfg.Rand()
	}
	return rand.Float64()
}

// sleepBeforeRetry waits d before the next attempt. cfg.Sleep replaces the
// timer in tests and observes every wait, including a zero one. Without the
// hook, a wait of zero or less returns immediately; the wait returns early
// with ctx.Err() when ctx is done.
func sleepBeforeRetry(ctx context.Context, cfg Config, d time.Duration) error {
	if cfg.Sleep != nil {
		return cfg.Sleep(ctx, d)
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

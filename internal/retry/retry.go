// Package retry holds the backoff loop shared by llmkit's non-chat
// packages. Exponential backoff with jitter, server Retry-After precedence
// capped at MaxDelay, a per-attempt RequestTimeout deadline, and
// parent-cancellation-is-terminal semantics. Each caller supplies the
// classifier that decides which of its own errors are transient.
package retry

import (
	"context"
	"github.com/dpoage/llmkit"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Do runs fn up to cfg.MaxAttempts times, retrying transient failures
// under a per-attempt RequestTimeout child context — a stalled round-trip
// aborts as a timeout-classified error and is retried. The parent ctx is
// never modified: once it is done, Do returns the last error immediately.
//
// classify reports whether err is worth retrying. When ok is true, after
// and hasAfter carry a server-supplied Retry-After delay that replaces
// the computed backoff for that sleep. cfg must be normalized by the
// caller: MaxAttempts and RequestTimeout must be positive.
func Do(ctx context.Context, cfg llmkit.RetryConfig, classify func(error) (after time.Duration, hasAfter, ok bool), fn func(context.Context) error) error {
	var err error
	for attempt := 1; ; attempt++ {
		actx, cancel := context.WithTimeout(ctx, cfg.RequestTimeout)
		err = fn(actx)
		cancel()
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return err
		}
		after, hasAfter, ok := classify(err)
		if !ok || attempt >= cfg.MaxAttempts {
			return err
		}
		if serr := sleepCtx(ctx, backoffDelay(cfg, attempt, after, hasAfter)); serr != nil {
			return err
		}
	}
}

// ParseRetryAfter parses a Retry-After header value: either delay seconds or
// an HTTP-date. A past date clamps to 0. ok is false for empty or malformed
// values.
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
func backoffDelay(cfg llmkit.RetryConfig, attempt int, after time.Duration, hasAfter bool) time.Duration {
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
		// Overflow clamp: zero BaseDelay means no wait, not a runaway sleep.
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
		factor := 1 + cfg.Jitter*(2*rand.Float64()-1)
		delay = time.Duration(float64(delay) * factor)
	}
	return delay
}

// sleepCtx waits d, returning early with ctx.Err() if ctx is cancelled.
func sleepCtx(ctx context.Context, d time.Duration) error {
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

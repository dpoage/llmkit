package embed

import (
	"context"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// statusError reports a non-200 HTTP response from a backend call. It carries
// the parsed Retry-After header so the retry loop can honor it; the backend
// prefix is part of Error so messages keep their "ollama: HTTP 429: ..." shape.
type statusError struct {
	backend       string
	status        int
	body          string
	retryAfter    time.Duration // valid only when hasRetryAfter
	hasRetryAfter bool
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s: HTTP %d: %s", e.backend, e.status, e.body)
}

// retryAfterDuration returns the server-supplied delay and whether one was
// present.
func (e *statusError) retryAfterDuration() (time.Duration, bool) {
	return e.retryAfter, e.hasRetryAfter
}

// newStatusError builds a statusError with a truncated body and the parsed
// Retry-After header value, if any.
func newStatusError(backend string, status int, retryAfterHeader, body string) *statusError {
	se := &statusError{backend: backend, status: status, body: truncate(body, 200)}
	if d, ok := parseRetryAfter(retryAfterHeader, time.Now()); ok {
		se.retryAfter, se.hasRetryAfter = d, true
	}
	return se
}

// parseRetryAfter parses a Retry-After header value: either delay seconds or
// an HTTP-date. A past date clamps to 0. ok is false for empty or malformed
// values.
func parseRetryAfter(v string, now time.Time) (time.Duration, bool) {
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

// retryable classifies an error from a backend call as worth retrying.
// Transient: HTTP 429 and 5xx (honoring Retry-After when supplied), and
// network errors that are timeouts or marked temporary. Everything else —
// other 4xx, decode failures, context cancellation — is terminal.
// hasDelay reports whether the server supplied a Retry-After delay that
// should replace the computed backoff.
func retryable(err error) (delay time.Duration, hasDelay bool, ok bool) {
	var se *statusError
	if errors.As(err, &se) {
		if se.status == http.StatusTooManyRequests || se.status >= 500 {
			if d, present := se.retryAfterDuration(); present {
				return d, true, true
			}
			return 0, false, true
		}
		return 0, false, false
	}
	var ne net.Error
	if errors.As(err, &ne) && (ne.Timeout() || ne.Temporary()) {
		return 0, false, true
	}
	return 0, false, false
}

// retryDo runs fn up to cfg.MaxAttempts times, retrying transient failures
// with Retry-After-aware exponential backoff. Each attempt runs under a child
// context carrying the cfg.RequestTimeout deadline — the same per-attempt
// bound llmkit.WithRetry applies to provider calls — so a stalled round-trip
// aborts as a timeout-classified error and is retried instead of blocking
// forever. The parent ctx is never modified: once it is done, retryDo returns
// the last error immediately — cancellation is never retried. cfg must come
// from Config.retryPolicy, which normalizes MaxAttempts and RequestTimeout.
func retryDo(ctx context.Context, cfg llmkit.RetryConfig, fn func(context.Context) error) error {
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
		after, hasAfter, ok := retryable(err)
		if !ok || attempt >= cfg.MaxAttempts {
			return err
		}
		if serr := sleepCtx(ctx, backoffDelay(cfg, attempt, after, hasAfter)); serr != nil {
			return err
		}
	}
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

	// Exponential: base * 2^(attempt-1).
	delay := cfg.BaseDelay
	for range attempt - 1 {
		delay *= 2
		if delay < 0 { // int64 overflow; zero BaseDelay means no wait
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

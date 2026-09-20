package retry

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// retryAny classifies every non-nil error as retryable, with no Retry-After.
func retryAny(err error) (time.Duration, bool, bool) {
	if err == nil {
		return 0, false, false
	}
	return 0, false, true
}

// TestBackoffDelay_RetryAfterCappedAtMaxDelay pins the delay policy: a
// server Retry-After wins over the exponential schedule but is capped
// at MaxDelay; exponential backoff stays in (0, MaxDelay]; zero
// BaseDelay stays zero (immediate retry), never mistaken for int64
// overflow.
func TestBackoffDelay_RetryAfterCappedAtMaxDelay(t *testing.T) {
	def := llmkit.DefaultRetryConfig()
	// Normalized retry policy: explicit MaxAttempts, kit-default delays,
	// embed's 60s per-attempt timeout.
	p := llmkit.RetryConfig{MaxAttempts: 5, BaseDelay: def.BaseDelay, MaxDelay: def.MaxDelay, RequestTimeout: 60 * time.Second}

	after := backoffDelay(p, 1, time.Hour, true)
	if after != def.MaxDelay {
		t.Errorf("Retry-After backoff = %v, want capped at %v", after, def.MaxDelay)
	}
	d := backoffDelay(p, 3, 0, false)
	if d <= 0 || d > def.MaxDelay {
		t.Errorf("exponential backoff = %v, want in (0, %v]", d, def.MaxDelay)
	}

	if d := backoffDelay(llmkit.RetryConfig{BaseDelay: 0}, 3, 0, false); d != 0 {
		t.Errorf("backoffDelay with zero BaseDelay = %v, want 0", d)
	}
}

// TestBackoffDelay_JitterZeroDeterministic pins that explicit Jitter 0
// means no jitter: attempt 2 doubles BaseDelay exactly.
func TestBackoffDelay_JitterZeroDeterministic(t *testing.T) {
	if d := backoffDelay(llmkit.RetryConfig{BaseDelay: 100 * time.Millisecond, Jitter: 0}, 2, 0, false); d != 200*time.Millisecond {
		t.Errorf("backoffDelay with Jitter 0 = %v, want exact 200ms (deterministic)", d)
	}
}

// TestDo_RetriesTransientThenSucceeds pins the loop: a transient failure is
// retried, and success on a later attempt returns nil.
func TestDo_RetriesTransientThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	err := Do(context.Background(), llmkit.RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}, retryAny, func(ctx context.Context) error {
		if calls.Add(1) < 3 {
			return errors.New("transient")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("calls = %d, want 3", got)
	}
}

// TestDo_GivesUpAfterMaxAttempts pins that Do returns the last error after
// MaxAttempts attempts, not one more.
func TestDo_GivesUpAfterMaxAttempts(t *testing.T) {
	var calls atomic.Int32
	want := errors.New("always transient")
	err := Do(context.Background(), llmkit.RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}, retryAny, func(ctx context.Context) error {
		calls.Add(1)
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want the last attempt's error", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("calls = %d, want 3 (MaxAttempts)", got)
	}
}

// TestDo_UnclassifiedTerminal pins that a classify miss returns immediately
// without a second attempt.
func TestDo_UnclassifiedTerminal(t *testing.T) {
	var calls atomic.Int32
	err := Do(context.Background(), llmkit.RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond}, func(error) (time.Duration, bool, bool) {
		return 0, false, false
	}, func(ctx context.Context) error {
		calls.Add(1)
		return errors.New("terminal")
	})
	if err == nil {
		t.Fatal("expected the terminal error")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("calls = %d, want 1 (terminal errors are never retried)", got)
	}
}

// TestDo_RetryAfterReplacesBackoff pins that a classified Retry-After delay
// replaces the exponential schedule: with a 1h BaseDelay and no cap, only the
// server's 1ms delay can let this test finish.
func TestDo_RetryAfterReplacesBackoff(t *testing.T) {
	var calls atomic.Int32
	start := time.Now()
	err := Do(context.Background(), llmkit.RetryConfig{MaxAttempts: 2, BaseDelay: time.Hour, Jitter: 0},
		func(error) (time.Duration, bool, bool) { return time.Millisecond, true, true },
		func(ctx context.Context) error {
			calls.Add(1)
			if calls.Load() == 1 {
				return errors.New("transient")
			}
			return nil
		})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("Do took %v; Retry-After delay was not used instead of backoff", elapsed)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("calls = %d, want 2", got)
	}
}

// TestDo_ParentCancelTerminal pins that cancellation of the parent context
// ends the loop immediately with the last error — cancellation is never
// retried and never swallowed.
func TestDo_ParentCancelTerminal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32
	want := errors.New("boom")
	err := Do(ctx, llmkit.RetryConfig{MaxAttempts: 5, BaseDelay: time.Hour, Jitter: 0}, retryAny, func(ctx context.Context) error {
		calls.Add(1)
		cancel() // cancel the parent mid-attempt
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want the in-flight attempt's error", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("calls = %d, want 1 (cancelled parents never retry)", got)
	}
}

// TestDo_PerAttemptTimeout pins the per-attempt deadline: a fn that outlives
// cfg.RequestTimeout sees its context expire, and the timeout-classified
// failure is retried rather than blocking forever.
func TestDo_PerAttemptTimeout(t *testing.T) {
	var calls atomic.Int32
	err := Do(context.Background(), llmkit.RetryConfig{MaxAttempts: 2, BaseDelay: time.Millisecond, Jitter: 0, RequestTimeout: 20 * time.Millisecond}, retryAny, func(ctx context.Context) error {
		calls.Add(1)
		select {
		case <-time.After(200 * time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("calls = %d, want 2 (per-attempt timeouts are retried)", got)
	}
}

// TestParseRetryAfter pins the header grammar: delay seconds (negative clamps
// to 0), HTTP-date (past clamps to 0), and ok=false for empty or malformed
// values.
func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)

	if d, ok := ParseRetryAfter("30", now); !ok || d != 30*time.Second {
		t.Errorf("ParseRetryAfter(\"30\") = %v, %v; want 30s, true", d, ok)
	}
	if d, ok := ParseRetryAfter(" 5 ", now); !ok || d != 5*time.Second {
		t.Errorf("ParseRetryAfter(\" 5 \") = %v, %v; want 5s, true (trimmed)", d, ok)
	}
	if d, ok := ParseRetryAfter("-3", now); !ok || d != 0 {
		t.Errorf("ParseRetryAfter(\"-3\") = %v, %v; want 0, true (negative clamps)", d, ok)
	}
	if d, ok := ParseRetryAfter(now.Add(90*time.Second).UTC().Format("Mon, 02 Jan 2006 15:04:05 GMT"), now); !ok || d != 90*time.Second {
		t.Errorf("ParseRetryAfter(future HTTP-date) = %v, %v; want 90s, true", d, ok)
	}
	if d, ok := ParseRetryAfter(now.Add(-time.Hour).UTC().Format("Mon, 02 Jan 2006 15:04:05 GMT"), now); !ok || d != 0 {
		t.Errorf("ParseRetryAfter(past HTTP-date) = %v, %v; want 0, true (past clamps)", d, ok)
	}
	if _, ok := ParseRetryAfter("", now); ok {
		t.Error("ParseRetryAfter(\"\") ok = true, want false")
	}
	if _, ok := ParseRetryAfter("soon", now); ok {
		t.Error("ParseRetryAfter(malformed) ok = true, want false")
	}
}

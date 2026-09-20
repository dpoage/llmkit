package llmkit

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
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
	def := DefaultRetryConfig()
	// Normalized retry policy: explicit MaxAttempts, kit-default delays,
	// embed's 60s per-attempt timeout.
	p := RetryConfig{MaxAttempts: 5, BaseDelay: def.BaseDelay, MaxDelay: def.MaxDelay, RequestTimeout: 60 * time.Second}

	after := backoffDelay(p, 1, time.Hour, true)
	if after != def.MaxDelay {
		t.Errorf("Retry-After backoff = %v, want capped at %v", after, def.MaxDelay)
	}
	d := backoffDelay(p, 3, 0, false)
	if d <= 0 || d > def.MaxDelay {
		t.Errorf("exponential backoff = %v, want in (0, %v]", d, def.MaxDelay)
	}

	if d := backoffDelay(RetryConfig{BaseDelay: 0}, 3, 0, false); d != 0 {
		t.Errorf("backoffDelay with zero BaseDelay = %v, want 0", d)
	}
}

// TestBackoffDelay_JitterZeroDeterministic pins that explicit Jitter 0
// means no jitter: attempt 2 doubles BaseDelay exactly.
func TestBackoffDelay_JitterZeroDeterministic(t *testing.T) {
	if d := backoffDelay(RetryConfig{BaseDelay: 100 * time.Millisecond, Jitter: 0}, 2, 0, false); d != 200*time.Millisecond {
		t.Errorf("backoffDelay with Jitter 0 = %v, want exact 200ms (deterministic)", d)
	}
}

// TestRetry_RetriesTransientThenSucceeds pins the loop: a transient failure is
// retried, and success on a later attempt returns nil.
func TestRetry_RetriesTransientThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	err := Retry(context.Background(), RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}, retryAny, func(ctx context.Context) error {
		if calls.Add(1) < 3 {
			return errors.New("transient")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Retry: %v", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("calls = %d, want 3", got)
	}
}

// TestRetry_GivesUpAfterMaxAttempts pins that Retry returns the last error
// after MaxAttempts attempts, not one more.
func TestRetry_GivesUpAfterMaxAttempts(t *testing.T) {
	var calls atomic.Int32
	want := errors.New("always transient")
	err := Retry(context.Background(), RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}, retryAny, func(ctx context.Context) error {
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

// TestRetry_UnclassifiedTerminal pins that a classify miss returns immediately
// without a second attempt.
func TestRetry_UnclassifiedTerminal(t *testing.T) {
	var calls atomic.Int32
	err := Retry(context.Background(), RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond}, func(error) (time.Duration, bool, bool) {
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

// TestRetry_RetryAfterReplacesBackoff pins that a classified Retry-After delay
// replaces the exponential schedule: with a 1h BaseDelay and no cap, only the
// server's 1ms delay can let this test finish.
func TestRetry_RetryAfterReplacesBackoff(t *testing.T) {
	var calls atomic.Int32
	start := time.Now()
	err := Retry(context.Background(), RetryConfig{MaxAttempts: 2, BaseDelay: time.Hour, Jitter: 0},
		func(error) (time.Duration, bool, bool) { return time.Millisecond, true, true },
		func(ctx context.Context) error {
			calls.Add(1)
			if calls.Load() == 1 {
				return errors.New("transient")
			}
			return nil
		})
	if err != nil {
		t.Fatalf("Retry: %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("Retry took %v; Retry-After delay was not used instead of backoff", elapsed)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("calls = %d, want 2", got)
	}
}

// TestRetry_ParentCancelTerminal pins that cancellation of the parent context
// ends the loop immediately with the last error — cancellation is never
// retried and never swallowed.
func TestRetry_ParentCancelTerminal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32
	want := errors.New("boom")
	err := Retry(ctx, RetryConfig{MaxAttempts: 5, BaseDelay: time.Hour, Jitter: 0}, retryAny, func(ctx context.Context) error {
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

// TestRetry_PerAttemptTimeout pins the per-attempt deadline: a fn that
// outlives cfg.RequestTimeout sees its context expire, and the
// timeout-classified failure is retried rather than blocking forever.
func TestRetry_PerAttemptTimeout(t *testing.T) {
	var calls atomic.Int32
	err := Retry(context.Background(), RetryConfig{MaxAttempts: 2, BaseDelay: time.Millisecond, Jitter: 0, RequestTimeout: 20 * time.Millisecond}, retryAny, func(ctx context.Context) error {
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

// TestRetry_ZeroConfigIsOneAttempt pins the MaxAttempts clamp: the zero
// RetryConfig means exactly one attempt and no retries, never a zero-value
// panic or an unbounded loop.
func TestRetry_ZeroConfigIsOneAttempt(t *testing.T) {
	var calls atomic.Int32
	err := Retry(context.Background(), RetryConfig{}, retryAny, func(ctx context.Context) error {
		calls.Add(1)
		return errors.New("transient")
	})
	if err == nil {
		t.Fatal("expected the last error")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("calls = %d, want 1 (zero MaxAttempts clamps to 1)", got)
	}
}

// TestRetry_RetryDefaultsRequestTimeout pins the RequestTimeout clamp: a zero
// or negative RequestTimeout gives every attempt the DefaultRequestTimeout
// deadline.
func TestRetry_RetryDefaultsRequestTimeout(t *testing.T) {
	for _, cfg := range []RetryConfig{{}, {MaxAttempts: 2, RequestTimeout: -time.Second}} {
		var ok bool
		var deadline time.Time
		_ = Retry(context.Background(), cfg, retryAny, func(ctx context.Context) error {
			deadline, ok = ctx.Deadline()
			return errors.New("transient")
		})
		if !ok {
			t.Fatal("attempt context has no deadline; want DefaultRequestTimeout")
		}
		if remaining := time.Until(deadline); remaining > DefaultRequestTimeout || remaining < DefaultRequestTimeout-time.Minute {
			t.Errorf("attempt deadline in %v, want ≈ %v (RequestTimeout clamped to DefaultRequestTimeout)", remaining, DefaultRequestTimeout)
		}
	}
}

// TestRetry_JitterClampedInRetry pins the Jitter clamp on the Retry entry
// point: negative Jitter behaves as 0 (exact doubling), and Jitter above 1
// behaves as 1 (with rng pinned to 0 the wait is 0, not the negative delay
// unclamped jitter would compute).
func TestRetry_JitterClampedInRetry(t *testing.T) {
	var slept []time.Duration
	cfg := RetryConfig{MaxAttempts: 3, BaseDelay: 100 * time.Millisecond, Jitter: -1}
	cfg.sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	_ = Retry(context.Background(), cfg, retryAny, func(ctx context.Context) error {
		return errors.New("transient")
	})
	if len(slept) != 2 || slept[0] != 100*time.Millisecond || slept[1] != 200*time.Millisecond {
		t.Errorf("negative Jitter sleeps = %v, want [100ms 200ms] (clamped to 0: deterministic doubling)", slept)
	}

	slept = nil
	cfg = RetryConfig{MaxAttempts: 2, BaseDelay: 100 * time.Millisecond, Jitter: 3}
	cfg.rng = func() float64 { return 0 }
	cfg.sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	_ = Retry(context.Background(), cfg, retryAny, func(ctx context.Context) error {
		return errors.New("transient")
	})
	if len(slept) != 1 || slept[0] != 0 {
		t.Errorf("Jitter 3 sleeps = %v, want [0] (clamped to 1: factor 0 at rng 0, never negative)", slept)
	}
}

// TestRetry_ZeroRetryAfterMeansImmediateRetry pins the presence-bit semantics:
// hasRetryAfter with a zero delay is a server-supplied "retry now", distinct
// from absence, which would schedule the 1h exponential backoff.
func TestRetry_ZeroRetryAfterMeansImmediateRetry(t *testing.T) {
	var calls atomic.Int32
	var slept []time.Duration
	cfg := RetryConfig{MaxAttempts: 2, BaseDelay: time.Hour, Jitter: 0}
	cfg.sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	err := Retry(context.Background(), cfg, func(error) (time.Duration, bool, bool) {
		return 0, true, true
	}, func(ctx context.Context) error {
		calls.Add(1)
		if calls.Load() == 1 {
			return errors.New("transient")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Retry: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("calls = %d, want 2", got)
	}
	if len(slept) != 1 || slept[0] != 0 {
		t.Errorf("slept = %v, want [0] (presence with zero delay waits nothing, not the backoff)", slept)
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

// TestRetryLoop_DelayAgreement pins the one delay schedule every retry entry
// point now shares: a table of (cfg, attempt, after, hasAfter) cases run
// through Retry's own sleep hook with a deterministic rng, asserting exact
// delays including the clamp edges — negative Retry-After, BaseDelay*2^k
// overflow, and MaxDelay 0 meaning uncapped.
func TestRetryLoop_DelayAgreement(t *testing.T) {
	base := 500 * time.Millisecond
	cases := []struct {
		name     string
		cfg      RetryConfig
		attempt  int
		after    time.Duration
		hasAfter bool
		want     time.Duration
	}{
		{"retry-after capped at MaxDelay", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, time.Hour, true, 30 * time.Second},
		{"retry-after under cap wins over backoff", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 2 * time.Second, true, 2 * time.Second},
		{"negative retry-after clamps to zero", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, -5 * time.Second, true, 0},
		{"zero retry-after with presence waits nothing", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 0, true, 0},
		{"MaxDelay zero leaves retry-after uncapped", RetryConfig{BaseDelay: base, MaxDelay: 0, Jitter: 0}, 1, time.Hour, true, time.Hour},
		{"first exponential delay is BaseDelay", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 0, false, 500 * time.Millisecond},
		{"third exponential delay is four times BaseDelay", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 3, 0, false, 2 * time.Second},
		{"schedule caps at MaxDelay", RetryConfig{BaseDelay: 20 * time.Second, MaxDelay: 30 * time.Second, Jitter: 0}, 2, 0, false, 30 * time.Second},
		{"MaxDelay zero leaves the schedule uncapped", RetryConfig{BaseDelay: base, MaxDelay: 0, Jitter: 0}, 10, 0, false, 256 * time.Second},
		{"doubling past int64 clamps instead of overflowing", RetryConfig{BaseDelay: time.Duration(1) << 62, MaxDelay: 0, Jitter: 0}, 2, 0, false, time.Duration(1) << 62},
		{"zero BaseDelay stays zero", RetryConfig{BaseDelay: 0, MaxDelay: 30 * time.Second, Jitter: 0}, 3, 0, false, 0},
		{"jitter scales with the rng source", RetryConfig{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0.2}, 1, 0, false, 550 * time.Millisecond},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var slept []time.Duration
			cfg := tc.cfg
			cfg.MaxAttempts = tc.attempt + 1
			cfg.rng = func() float64 { return 0.75 } // jitter factor 1.1 when Jitter > 0
			cfg.sleep = func(ctx context.Context, d time.Duration) error {
				slept = append(slept, d)
				return nil
			}
			err := Retry(context.Background(), cfg, func(error) (time.Duration, bool, bool) {
				return tc.after, tc.hasAfter, true
			}, func(ctx context.Context) error {
				return errors.New("always transient")
			})
			if err == nil {
				t.Fatal("expected the last error from an always-failing fn")
			}
			if len(slept) == 0 {
				t.Fatal("no sleeps recorded; the loop never backed off")
			}
			if got := slept[len(slept)-1]; got != tc.want {
				t.Errorf("delay after attempt %d = %v, want %v (all sleeps: %v)", tc.attempt, got, tc.want, slept)
			}
		})
	}
}

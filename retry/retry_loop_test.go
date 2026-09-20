package retry

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
	def := Default()
	// Explicit MaxAttempts and 60s RequestTimeout; default delays.
	p := Config{MaxAttempts: 5, BaseDelay: def.BaseDelay, MaxDelay: def.MaxDelay, RequestTimeout: 60 * time.Second}

	after := backoffDelay(p, 1, time.Hour, true)
	if after != def.MaxDelay {
		t.Errorf("Retry-After backoff = %v, want capped at %v", after, def.MaxDelay)
	}
	d := backoffDelay(p, 3, 0, false)
	if d <= 0 || d > def.MaxDelay {
		t.Errorf("exponential backoff = %v, want in (0, %v]", d, def.MaxDelay)
	}

	if d := backoffDelay(Config{BaseDelay: 0}, 3, 0, false); d != 0 {
		t.Errorf("backoffDelay with zero BaseDelay = %v, want 0", d)
	}
}

// TestBackoffDelay_JitterZeroDeterministic pins that explicit Jitter 0
// means no jitter: attempt 2 doubles BaseDelay exactly.
func TestBackoffDelay_JitterZeroDeterministic(t *testing.T) {
	if d := backoffDelay(Config{BaseDelay: 100 * time.Millisecond, Jitter: 0}, 2, 0, false); d != 200*time.Millisecond {
		t.Errorf("backoffDelay with Jitter 0 = %v, want exact 200ms (deterministic)", d)
	}
}

// TestDo_RetriesTransientThenSucceeds pins the loop: a transient failure is
// retried, and success on a later attempt returns nil.
func TestDo_RetriesTransientThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	err := Do(context.Background(), Config{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}, retryAny, func(ctx context.Context) error {
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

// TestDo_GivesUpAfterMaxAttempts pins that Do returns the last error
// after MaxAttempts attempts, not one more.
func TestDo_GivesUpAfterMaxAttempts(t *testing.T) {
	var calls atomic.Int32
	want := errors.New("always transient")
	err := Do(context.Background(), Config{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}, retryAny, func(ctx context.Context) error {
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
	err := Do(context.Background(), Config{MaxAttempts: 3, BaseDelay: time.Millisecond}, func(error) (time.Duration, bool, bool) {
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
	err := Do(context.Background(), Config{MaxAttempts: 2, BaseDelay: time.Hour, Jitter: 0},
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
	err := Do(ctx, Config{MaxAttempts: 5, BaseDelay: time.Hour, Jitter: 0}, retryAny, func(ctx context.Context) error {
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

// TestDo_PerAttemptTimeout pins the per-attempt deadline: a fn that
// outlives cfg.RequestTimeout sees its context expire, and the
// timeout-classified failure is retried rather than blocking forever.
func TestDo_PerAttemptTimeout(t *testing.T) {
	var calls atomic.Int32
	err := Do(context.Background(), Config{MaxAttempts: 2, BaseDelay: time.Millisecond, Jitter: 0, RequestTimeout: 20 * time.Millisecond}, retryAny, func(ctx context.Context) error {
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

// TestDo_ZeroConfigIsOneAttempt pins the MaxAttempts clamp: the zero
// Config means exactly one attempt and no retries, never a zero-value
// panic or an unbounded loop.
func TestDo_ZeroConfigIsOneAttempt(t *testing.T) {
	var calls atomic.Int32
	err := Do(context.Background(), Config{}, retryAny, func(ctx context.Context) error {
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

// TestDo_DefaultsRequestTimeout pins the RequestTimeout clamp: a zero
// or negative RequestTimeout gives every attempt the DefaultRequestTimeout
// deadline.
func TestDo_DefaultsRequestTimeout(t *testing.T) {
	for _, cfg := range []Config{{}, {MaxAttempts: 2, RequestTimeout: -time.Second}} {
		var ok bool
		var deadline time.Time
		_ = Do(context.Background(), cfg, retryAny, func(ctx context.Context) error {
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

// TestDo_JitterClamped pins the Jitter clamp on the Do entry
// point: negative Jitter behaves as 0 (exact doubling), and Jitter above 1
// behaves as 1 (with Rand pinned to 0 the wait is 0, not the negative delay
// unclamped jitter would compute).
func TestDo_JitterClamped(t *testing.T) {
	var slept []time.Duration
	cfg := Config{MaxAttempts: 3, BaseDelay: 100 * time.Millisecond, Jitter: -1}
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	_ = Do(context.Background(), cfg, retryAny, func(ctx context.Context) error {
		return errors.New("transient")
	})
	if len(slept) != 2 || slept[0] != 100*time.Millisecond || slept[1] != 200*time.Millisecond {
		t.Errorf("negative Jitter sleeps = %v, want [100ms 200ms] (clamped to 0: deterministic doubling)", slept)
	}

	slept = nil
	cfg = Config{MaxAttempts: 2, BaseDelay: 100 * time.Millisecond, Jitter: 3}
	cfg.Rand = func() float64 { return 0 }
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	_ = Do(context.Background(), cfg, retryAny, func(ctx context.Context) error {
		return errors.New("transient")
	})
	if len(slept) != 1 || slept[0] != 0 {
		t.Errorf("Jitter 3 sleeps = %v, want [0] (clamped to 1: factor 0 at Rand 0, never negative)", slept)
	}
}

// TestDo_ZeroRetryAfterMeansImmediateRetry pins the presence-bit semantics:
// hasRetryAfter with a zero delay is a server-supplied "retry now", distinct
// from absence, which would schedule the 1h exponential backoff.
func TestDo_ZeroRetryAfterMeansImmediateRetry(t *testing.T) {
	var calls atomic.Int32
	var slept []time.Duration
	cfg := Config{MaxAttempts: 2, BaseDelay: time.Hour, Jitter: 0}
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	err := Do(context.Background(), cfg, func(error) (time.Duration, bool, bool) {
		return 0, true, true
	}, func(ctx context.Context) error {
		calls.Add(1)
		if calls.Load() == 1 {
			return errors.New("transient")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
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
// through Do's own sleep hook with a deterministic Rand, asserting exact
// delays including the clamp edges — negative Retry-After, BaseDelay*2^k
// overflow, MaxDelay 0 meaning uncapped, and BaseDelay above MaxDelay on the
// first retry (the post-loop cap is the only clamp that reaches it).
func TestRetryLoop_DelayAgreement(t *testing.T) {
	base := 500 * time.Millisecond
	cases := []struct {
		name     string
		cfg      Config
		attempt  int
		after    time.Duration
		hasAfter bool
		want     time.Duration
		wantAll  []time.Duration // set: assert the full sleep sequence
	}{
		{"retry-after capped at MaxDelay", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, time.Hour, true, 30 * time.Second, nil},
		{"retry-after under cap wins over backoff", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 2 * time.Second, true, 2 * time.Second, nil},
		{"negative retry-after clamps to zero", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, -5 * time.Second, true, 0, nil},
		{"zero retry-after with presence waits nothing", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 0, true, 0, nil},
		{"MaxDelay zero leaves retry-after uncapped", Config{BaseDelay: base, MaxDelay: 0, Jitter: 0}, 1, time.Hour, true, time.Hour, nil},
		{"retry-after without presence is ignored", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 5 * time.Second, false, 500 * time.Millisecond, nil},
		{"first delay caps at MaxDelay", Config{BaseDelay: time.Minute, MaxDelay: 30 * time.Second, Jitter: 0}, 1, 0, false, 30 * time.Second, nil},
		{"third exponential delay is four times BaseDelay", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0}, 3, 0, false, 2 * time.Second, nil},
		{"schedule caps at MaxDelay", Config{BaseDelay: 20 * time.Second, MaxDelay: 30 * time.Second, Jitter: 0}, 2, 0, false, 30 * time.Second, nil},
		{"MaxDelay zero leaves the schedule uncapped", Config{BaseDelay: base, MaxDelay: 0, Jitter: 0}, 10, 0, false, 256 * time.Second, nil},
		{"sequence grows then caps", Config{BaseDelay: 20 * time.Second, MaxDelay: 30 * time.Second, Jitter: 0}, 3, 0, false, 30 * time.Second,
			[]time.Duration{20 * time.Second, 30 * time.Second, 30 * time.Second}},
		{"doubling past int64 clamps instead of overflowing", Config{BaseDelay: time.Duration(1) << 62, MaxDelay: 0, Jitter: 0}, 2, 0, false, time.Duration(1) << 62, nil},
		{"zero BaseDelay stays zero", Config{BaseDelay: 0, MaxDelay: 30 * time.Second, Jitter: 0}, 3, 0, false, 0, nil},
		{"jitter scales with the Rand source", Config{BaseDelay: base, MaxDelay: 30 * time.Second, Jitter: 0.2}, 1, 0, false, 550 * time.Millisecond, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var slept []time.Duration
			cfg := tc.cfg
			cfg.MaxAttempts = tc.attempt + 1
			cfg.Rand = func() float64 { return 0.75 } // jitter factor 1.1 when Jitter > 0
			cfg.Sleep = func(ctx context.Context, d time.Duration) error {
				slept = append(slept, d)
				return nil
			}
			err := Do(context.Background(), cfg, func(error) (time.Duration, bool, bool) {
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
			if tc.wantAll != nil {
				if len(slept) != len(tc.wantAll) {
					t.Fatalf("sleeps = %v, want %v", slept, tc.wantAll)
				}
				for i := range tc.wantAll {
					if slept[i] != tc.wantAll[i] {
						t.Fatalf("sleeps = %v, want %v (sleep %d = %v, want %v)", slept, tc.wantAll, i, slept[i], tc.wantAll[i])
					}
				}
				return
			}
			if got := slept[len(slept)-1]; got != tc.want {
				t.Errorf("delay after attempt %d = %v, want %v (all sleeps: %v)", tc.attempt, got, tc.want, slept)
			}
		})
	}
}

// recovered runs fn, swallows a panic, and reports whether one happened.
func recovered(fn func()) (panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
		}
	}()
	fn()
	return false
}

// TestDo_PanicReleasesAttemptContext pins the per-attempt cleanup: fn's
// panic propagates, and the attempt context is still cancelled — its cancel
// runs via defer during the unwind, never an inline call a panic could skip.
func TestDo_PanicReleasesAttemptContext(t *testing.T) {
	var attemptCtx context.Context
	panicked := recovered(func() {
		_ = Do(context.Background(), Config{RequestTimeout: time.Hour}, retryAny, func(ctx context.Context) error {
			attemptCtx = ctx
			panic("fn panic")
		})
	})
	if !panicked {
		t.Fatal("expected fn's panic to propagate out of Do")
	}
	if attemptCtx == nil {
		t.Fatal("fn never saw an attempt context")
	}
	if attemptCtx.Err() != context.Canceled {
		t.Errorf("attempt ctx after panic: err = %v, want context.Canceled (cancel ran during unwind)", attemptCtx.Err())
	}
}

// TestDo_CancelsEachAttemptBeforeTheNext pins that the loop holds no
// accumulated defers: by the time attempt N runs, attempt N-1's context
// already reads context.Canceled — cancel runs at the end of each attempt,
// not when Do returns.
func TestDo_CancelsEachAttemptBeforeTheNext(t *testing.T) {
	var ctxs []context.Context
	var prevErrs []error
	cfg := Config{MaxAttempts: 3, BaseDelay: time.Millisecond, Jitter: 0}
	err := Do(context.Background(), cfg, retryAny, func(ctx context.Context) error {
		if prev := len(ctxs) - 1; prev >= 0 {
			prevErrs = append(prevErrs, ctxs[prev].Err())
		}
		ctxs = append(ctxs, ctx)
		if len(ctxs) < 3 {
			return errors.New("transient")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if len(ctxs) != 3 {
		t.Fatalf("attempts = %d, want 3", len(ctxs))
	}
	if len(prevErrs) != 2 {
		t.Fatalf("observed %d predecessor contexts, want 2", len(prevErrs))
	}
	for i, e := range prevErrs {
		if e != context.Canceled {
			t.Errorf("attempt %d's ctx while attempt %d ran: err = %v, want context.Canceled (cancel must not accumulate until Do returns)", i+1, i+2, e)
		}
	}
}

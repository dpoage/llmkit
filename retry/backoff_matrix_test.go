package retry

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"testing"
	"time"
)

// doubledSchedule returns the jitter-free delay for base > 0 after n
// doublings: base * 2^n, or 1<<62 once the product passes int64 (the
// documented overflow landing point). It is computed with big integers so
// it shares no arithmetic with backoffDelay.
func doubledSchedule(base time.Duration, n int) time.Duration {
	v := new(big.Int).Lsh(big.NewInt(int64(base)), uint(n))
	if v.Cmp(big.NewInt(math.MaxInt64)) > 0 {
		return 1 << 62
	}
	return time.Duration(v.Int64())
}

// TestBackoffDelay_RangeMatrix pins the backoffDelay range properties over
// the out-of-range configuration space — every BaseDelay/MaxDelay pair from
// int64 extremes to the ordinary, every Jitter, Rand at both ends and just
// under 1, attempts 1..100, with and without a Retry-After hint:
//
//   - never negative;
//   - never above MaxDelay when MaxDelay > 0;
//   - zero when BaseDelay <= 0 and there is no hint (a negative BaseDelay is
//     not overflow);
//   - MaxDelay <= 0 is uncapped: Jitter 0 gives exactly the doubled
//     schedule (landing at 1<<62 once it passes int64), and jitter that
//     scales a delay past int64 saturates at MaxInt64, never wrapping to a
//     negative sleep.
func TestBackoffDelay_RangeMatrix(t *testing.T) {
	delays := []time.Duration{math.MinInt64, -1, 0, 1, time.Second, 1 << 62, math.MaxInt64}
	jitters := []float64{0, 0.5, 1}
	rands := []float64{0, 0.999999, 1.0}
	hints := []struct {
		has bool
		d   time.Duration
	}{{false, 0}, {true, math.MinInt64}, {true, 0}, {true, time.Second}, {true, math.MaxInt64}}

	for _, base := range delays {
		for _, maxD := range delays {
			for _, jit := range jitters {
				for _, r := range rands {
					for _, h := range hints {
						for attempt := 1; attempt <= 100; attempt++ {
							cfg := Config{BaseDelay: base, MaxDelay: maxD, Jitter: jit, Rand: func() float64 { return r }}
							got := backoffDelay(cfg, attempt, h.d, h.has)
							name := fmt.Sprintf("base=%d max=%d jitter=%v rand=%v hint=%v/%d attempt=%d -> %d",
								base, maxD, jit, r, h.has, h.d, attempt, got)

							if got < 0 {
								t.Fatalf("negative delay: %s", name)
							}
							if maxD > 0 && got > maxD {
								t.Fatalf("delay above MaxDelay: %s", name)
							}
							if !h.has && base <= 0 && got != 0 {
								t.Fatalf("BaseDelay <= 0 without a hint must be zero: %s", name)
							}
							if !h.has && base > 0 && maxD <= 0 {
								want := doubledSchedule(base, attempt-1)
								if jit == 0 && got != want {
									t.Fatalf("uncapped jitter-free delay != doubled schedule %d: %s", want, name)
								}
								// Jitter that multiplies a delay of 2^62 or more by a
								// factor of at least 2 (rand 1.0, jitter 1) overflows
								// int64 and must saturate.
								if want >= 1<<62 && jit == 1 && r == 1.0 && got != math.MaxInt64 {
									t.Fatalf("uncapped jittered overflow must saturate at MaxInt64: %s", name)
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestBackoffDelay_PinnedOverflowRows names the two repros from the defect
// so a regression reads as a specific row, not a matrix coordinate.
func TestBackoffDelay_PinnedOverflowRows(t *testing.T) {
	one := func() float64 { return 1.0 }

	// The jittered product 1<<62 * 2 overflows int64; it saturates, then the
	// cap (MaxInt64) leaves it.
	cfg := Config{BaseDelay: 1 << 62, MaxDelay: math.MaxInt64, Jitter: 1, Rand: one}
	if got := backoffDelay(cfg, 1, 0, false); got != math.MaxInt64 {
		t.Errorf("{1<<62, MaxInt64, jitter 1, rand 1.0} attempt 1 = %v, want MaxInt64", got)
	}

	// Uncapped: 1s doubles past int64 near attempt 34; jitter must not wrap it.
	unc := Config{BaseDelay: time.Second, Jitter: 1, Rand: one}
	for _, attempt := range []int{33, 34, 35, 64, 100} {
		if got := backoffDelay(unc, attempt, 0, false); got <= 0 {
			t.Errorf("uncapped attempt %d = %v, want a positive delay", attempt, got)
		}
	}
}

// TestDo_NegativeBaseDelaySleepsNonNegative runs the reported end-to-end
// case: a negative BaseDelay with no positive MaxDelay reached the sleep as
// a negative wait before the first retry and as the 1<<62 (about 146-year)
// overflow clamp before later retries. Every sleep Do records is >= 0.
func TestDo_NegativeBaseDelaySleepsNonNegative(t *testing.T) {
	var sleeps []time.Duration
	cfg := Config{
		MaxAttempts: 3,
		BaseDelay:   -1,
		MaxDelay:    -1,
		Sleep: func(_ context.Context, d time.Duration) error {
			sleeps = append(sleeps, d)
			return nil
		},
	}
	err := Do(context.Background(), cfg, retryAny, func(context.Context) error { return errors.New("transient") })
	if err == nil {
		t.Fatal("Do returned nil, want the last attempt's error")
	}
	if len(sleeps) != 2 {
		t.Fatalf("sleeps = %v, want 2 (one between each of 3 attempts)", sleeps)
	}
	for i, d := range sleeps {
		if d < 0 {
			t.Errorf("sleep[%d] = %v, want >= 0", i, d)
		}
	}
}

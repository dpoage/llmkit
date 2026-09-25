package llmkit

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/dpoage/llmkit/retry"
)

// TestClassify_Table pins the kit's retryability rule row by row. Row
// order is precedence: anything chaining context.Canceled is terminal
// before Kind is consulted, so an *APIError wrapping a cancelled context
// is terminal too. A bare context.DeadlineExceeded (the per-attempt
// timeout seen from outside an adapter) is terminal; the retryable shape
// is the adapter-wrapped *APIError{Kind: ErrServer}. Retry-After rides
// the HasRetryAfter presence bit, not a nonzero check.
func TestClassify_Table(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		after     time.Duration
		has       bool
		retryable bool
	}{
		{"nil", nil, 0, false, false},
		{"bare canceled", context.Canceled, 0, false, false},
		{"wrapped canceled", fmt.Errorf("call: %w", context.Canceled), 0, false, false},
		{"bare deadline exceeded", context.DeadlineExceeded, 0, false, false},
		{"plain error", errors.New("boom"), 0, false, false},
		{"rate limited with delay", &APIError{Kind: ErrRateLimited, RetryAfter: 7 * time.Second, HasRetryAfter: true}, 7 * time.Second, true, true},
		{"rate limited present zero", &APIError{Kind: ErrRateLimited, HasRetryAfter: true}, 0, true, true},
		{"rate limited absent", &APIError{Kind: ErrRateLimited}, 0, false, true},
		{"server error", &APIError{Kind: ErrServer}, 0, false, true},
		{"overloaded", &APIError{Kind: ErrOverloaded}, 0, false, true},
		{"wrapped server error", fmt.Errorf("call: %w", &APIError{Kind: ErrServer}), 0, false, true},
		{"auth", &APIError{Kind: ErrAuth}, 0, false, false},
		{"invalid request", &APIError{Kind: ErrInvalidRequest}, 0, false, false},
		{"context too long", &APIError{Kind: ErrContextTooLong}, 0, false, false},
		{"api error wrapping canceled", &APIError{Kind: ErrServer, Err: context.Canceled}, 0, false, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			after, has, retryable := Classify(tc.err)
			if after != tc.after || has != tc.has || retryable != tc.retryable {
				t.Errorf("Classify(%v) = (%v, %v, %v), want (%v, %v, %v)",
					tc.err, after, has, retryable, tc.after, tc.has, tc.retryable)
			}
		})
	}
}

// TestClassify_DoAttemptCounts pins the classifier through the loop it
// feeds: retry.Do makes exactly MaxAttempts calls for a retryable row
// and exactly 1 for each terminal row. The Sleep hook stands in for the
// timer.
func TestClassify_DoAttemptCounts(t *testing.T) {
	tests := []struct {
		name  string
		err   error
		calls int
	}{
		{"retryable server error", &APIError{Kind: ErrServer}, 3},
		{"retryable rate limit", &APIError{Kind: ErrRateLimited, HasRetryAfter: true}, 3},
		{"retryable overloaded", &APIError{Kind: ErrOverloaded}, 3},
		{"terminal canceled", context.Canceled, 1},
		{"terminal api error wrapping canceled", &APIError{Kind: ErrServer, Err: context.Canceled}, 1},
		{"terminal bare deadline exceeded", context.DeadlineExceeded, 1},
		{"terminal plain error", errors.New("boom"), 1},
		{"terminal auth", &APIError{Kind: ErrAuth}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			cfg := retry.Config{MaxAttempts: 3, Sleep: func(context.Context, time.Duration) error { return nil }}
			err := retry.Do(context.Background(), cfg, Classify, func(context.Context) error {
				calls++
				return tc.err
			})
			if !errors.Is(err, tc.err) {
				t.Errorf("Do err = %v, want the row error %v", err, tc.err)
			}
			if calls != tc.calls {
				t.Errorf("calls = %d, want %d", calls, tc.calls)
			}
		})
	}
}

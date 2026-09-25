package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit/retry"
)

type fakeClient struct {
	caps      Capabilities
	responses []Response
	errs      []error
	calls     int
}

func (f *fakeClient) Capabilities() Capabilities { return f.caps }

func (f *fakeClient) Complete(ctx context.Context, req Request) (Response, error) {
	i := f.calls
	f.calls++
	var err error
	if i < len(f.errs) {
		err = f.errs[i]
	}
	var resp Response
	if i < len(f.responses) {
		resp = f.responses[i]
	}
	return resp, err
}

func rateLimitErr(retryAfter time.Duration) error {
	return &APIError{Kind: ErrRateLimited, StatusCode: 429, RetryAfter: retryAfter, Provider: "fake", Message: "429"}
}

func TestRetry_SucceedsAfterRateLimit(t *testing.T) {
	fc := &fakeClient{
		errs:      []error{rateLimitErr(0), rateLimitErr(0), nil},
		responses: []Response{{}, {}, {Text: "ok", StopReason: StopEndTurn}},
	}
	var slept []time.Duration
	cfg := retry.Default()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	cfg.Rand = func() float64 { return 0.5 } // deterministic: jitter factor = 1.0

	client := WithRetry(fc, cfg)
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
	if fc.calls != 3 {
		t.Errorf("calls = %d, want 3", fc.calls)
	}
	if len(slept) != 2 {
		t.Fatalf("slept %d times, want 2", len(slept))
	}
	// Exponential backoff: 500ms, then 1s (jitter factor 1.0).
	if slept[0] != 500*time.Millisecond {
		t.Errorf("slept[0] = %v, want 500ms", slept[0])
	}
	if slept[1] != 1*time.Second {
		t.Errorf("slept[1] = %v, want 1s", slept[1])
	}
}

func TestRetry_HonorsRetryAfter(t *testing.T) {
	// A carried Retry-After is a presence bit (APIError.HasRetryAfter), not
	// a nonzero check: when the header is present, its delay replaces the
	// backoff schedule.
	fc := &fakeClient{
		errs: []error{&APIError{
			Kind:          ErrRateLimited,
			StatusCode:    429,
			RetryAfter:    3 * time.Second,
			HasRetryAfter: true,
			Provider:      "fake",
			Message:       "429",
		}},
		responses: []Response{{}, {Text: "ok"}},
	}
	var slept []time.Duration
	cfg := retry.Default()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }

	client := WithRetry(fc, cfg)
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(slept) != 1 || slept[0] != 3*time.Second {
		t.Errorf("slept = %v, want [3s] (carried Retry-After honored over backoff)", slept)
	}
}

func TestRetry_DoesNotRetryAuth(t *testing.T) {
	authErr := &APIError{Kind: ErrAuth, StatusCode: 401, Provider: "fake", Message: "401"}
	fc := &fakeClient{errs: []error{authErr}}
	cfg := retry.Default()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := WithRetry(fc, cfg)
	_, err := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(err, ErrAuth) {
		t.Errorf("err = %v, want ErrAuth", err)
	}
	if fc.calls != 1 {
		t.Errorf("calls = %d, want 1 (auth is not retryable)", fc.calls)
	}
}

func TestRetry_ExhaustsAttempts(t *testing.T) {
	fc := &fakeClient{errs: []error{rateLimitErr(0), rateLimitErr(0), rateLimitErr(0), rateLimitErr(0)}}
	cfg := retry.Default() // 4 attempts
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := WithRetry(fc, cfg)
	_, err := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(err, ErrRateLimited) {
		t.Errorf("err = %v, want ErrRateLimited", err)
	}
	if fc.calls != 4 {
		t.Errorf("calls = %d, want 4", fc.calls)
	}
}

// TestRetry_StopsOnContextCancel pins a caller's ctx that ends during the backoff sleep after a retryable 429 yields a plain error chaining context.Canceled carrying the 429 as text, never the retryable *APIError, terminal under Classify.
func TestRetry_StopsOnContextCancel(t *testing.T) {
	fc := &fakeClient{errs: []error{rateLimitErr(0), rateLimitErr(0)}}
	cfg := retry.Default()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error {
		cancel() // the caller's ctx ends during the backoff sleep
		return ctx.Err()
	}

	client := WithRetry(fc, cfg)
	_, err := client.Complete(ctx, simpleRequest())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled in the chain", err)
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		t.Errorf("err = %v (%T), want no *APIError on a cancelled path", err, err)
	}
	if errors.Is(err, ErrRateLimited) {
		t.Errorf("err = %v, want the last attempt's error NOT chained (text only)", err)
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("err = %v, want the last attempt's message in the text", err)
	}
	if _, _, retryable := Classify(err); retryable {
		t.Errorf("Classify(err) = retryable, want terminal for a cancelled path")
	}
	if fc.calls != 1 {
		t.Errorf("calls = %d, want 1", fc.calls)
	}
}

// TestRetry_StreamCallbackSentinelSurvivesCancel pins a callback that cancels the caller's ctx then returns its own sentinel gets that sentinel back through WithRetry — the failure is terminal under Classify (plain error, no *APIError), so Do returns it as-is.
func TestRetry_StreamCallbackSentinelSurvivesCancel(t *testing.T) {
	sentinel := errors.New("caller budget exceeded")
	inner := &scriptedStreamClient{deltas: []Delta{{Kind: DeltaText, Text: "hi"}}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var sleeps int
	client := WithRetry(inner, countedPolicy(3, time.Minute, &sleeps))
	_, err := Stream(ctx, client, simpleRequest(), func(Delta) error {
		cancel()
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want the callback's sentinel returned as-is", err)
	}
	if inner.calls != 1 || sleeps != 0 {
		t.Errorf("calls = %d, sleeps = %d; want 1 and 0", inner.calls, sleeps)
	}
}

// cancelAfterClient answers every Complete with err after cancelling the
// caller's ctx, so the ctx is done when the retry loop inspects the failure.
type cancelAfterClient struct {
	err    error
	cancel context.CancelFunc
	calls  int
}

func (c *cancelAfterClient) Capabilities() Capabilities { return Capabilities{} }

func (c *cancelAfterClient) Complete(context.Context, Request) (Response, error) {
	c.calls++
	c.cancel()
	return Response{}, c.err
}

// TestRetry_AuthErrorSurvivesCancel pins a 401 followed by the caller's ctx ending before the loop returns comes back as that *APIError (Kind ErrAuth), not replaced by a cancellation error — through retry.Do with Classify and through WithRetry.
func TestRetry_AuthErrorSurvivesCancel(t *testing.T) {
	authErr := &APIError{Kind: ErrAuth, StatusCode: 401, Provider: "fake", Message: "401"}
	cfg := retry.Config{MaxAttempts: 3, BaseDelay: time.Hour}
	for _, tc := range []struct {
		name string
		run  func(ctx context.Context, c *cancelAfterClient) error
	}{
		{"retry.Do", func(ctx context.Context, c *cancelAfterClient) error {
			return retry.Do(ctx, cfg, Classify, func(actx context.Context) error {
				_, err := c.Complete(actx, simpleRequest())
				return err
			})
		}},
		{"WithRetry", func(ctx context.Context, c *cancelAfterClient) error {
			_, err := WithRetry(c, cfg).Complete(ctx, simpleRequest())
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			c := &cancelAfterClient{err: authErr, cancel: cancel}
			err := tc.run(ctx, c)
			if ctx.Err() == nil {
				t.Fatal("probe broken: ctx still live when the loop returned")
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Kind != ErrAuth {
				t.Fatalf("err = %v (%T), want the 401 *APIError (Kind ErrAuth) returned as-is", err, err)
			}
			if _, _, retryable := Classify(err); retryable {
				t.Errorf("Classify(err) = retryable, want terminal")
			}
			if c.calls != 1 {
				t.Errorf("calls = %d, want 1", c.calls)
			}
		})
	}
}

// TestRetry_SaturatedRetryAfterCappedAtMaxDelay closes the ParseRetryAfter
// range at the retry-stage level: a parsed Retry-After that saturates to the
// maximum Duration — including a value strconv cannot parse — sleeps exactly
// MaxDelay, and a negative one sleeps exactly 0; neither wraps.
func TestRetry_SaturatedRetryAfterCappedAtMaxDelay(t *testing.T) {
	for _, tc := range []struct {
		v    string
		want time.Duration
	}{
		{"-5", 0},
		{"-9223372037", 0},
		{"9223372037", 30 * time.Second},
		{"9223372036854775807", 30 * time.Second},
		{"9223372036854775808", 30 * time.Second},
		{"99999999999999999999", 30 * time.Second},
	} {
		t.Run(tc.v, func(t *testing.T) {
			d, ok := retry.ParseRetryAfter(tc.v, time.Now())
			rl := &APIError{
				Kind:          ErrRateLimited,
				StatusCode:    429,
				RetryAfter:    d,
				HasRetryAfter: ok,
				Provider:      "fake",
				Message:       "429",
			}
			fc := &fakeClient{errs: []error{rl, rl}}
			var slept []time.Duration
			cfg := retry.Default()
			cfg.MaxAttempts = 2
			cfg.MaxDelay = 30 * time.Second
			cfg.Rand = func() float64 { return 0.5 } // no jitter on the schedule fallback
			cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }

			client := WithRetry(fc, cfg)
			if _, err := client.Complete(context.Background(), simpleRequest()); !errors.Is(err, ErrRateLimited) {
				t.Fatalf("Complete err = %v, want ErrRateLimited", err)
			}
			if len(slept) != 1 || slept[0] != tc.want {
				t.Errorf("Retry-After %q (parsed %v, ok %v): slept = %v, want [%v]", tc.v, d, ok, slept, tc.want)
			}
		})
	}
}

func TestRecorder_RecordsUsageOnSuccess(t *testing.T) {
	fc := &fakeClient{responses: []Response{{Text: "x", Usage: Usage{InputTokens: 100, OutputTokens: 50}}}}
	var mu sync.Mutex
	var events []UsageEvent
	rec := RecorderFunc(func(ev UsageEvent) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, ev)
	})

	client := WithRecorder(fc, rec, "anthropic", "claude-test")
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	ev := events[0]
	if ev.Provider != "anthropic" || ev.Model != "claude-test" {
		t.Errorf("event tags = %+v", ev)
	}
	if ev.Usage.InputTokens != 100 || ev.Usage.OutputTokens != 50 {
		t.Errorf("event usage = %+v, want {100 50}", ev.Usage)
	}
}

func TestRecorder_NoRecordOnError(t *testing.T) {
	fc := &fakeClient{errs: []error{errors.New("boom")}}
	var count int
	rec := RecorderFunc(func(ev UsageEvent) { count++ })

	client := WithRecorder(fc, rec, "p", "m")
	if _, err := client.Complete(context.Background(), simpleRequest()); err == nil {
		t.Fatal("expected error")
	}
	if count != 0 {
		t.Errorf("recorded %d events on error, want 0", count)
	}
}

func TestSerialize_TruncatesParallelToolCalls(t *testing.T) {
	fc := &fakeClient{
		caps: Capabilities{ParallelToolCalls: false},
		responses: []Response{{
			ToolCalls: []ToolCall{
				{ID: "1", Name: "a", Arguments: json.RawMessage(`{}`)},
				{ID: "2", Name: "b", Arguments: json.RawMessage(`{}`)},
				{ID: "3", Name: "c", Arguments: json.RawMessage(`{}`)},
			},
			StopReason: StopToolUse,
		}},
	}
	client := WithSerializedToolCalls(fc)
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "1" {
		t.Errorf("kept tool call ID = %q, want first (1)", resp.ToolCalls[0].ID)
	}
	if resp.StopReason != StopToolUse {
		t.Errorf("StopReason = %q, want tool_use", resp.StopReason)
	}
	if client.Capabilities().ParallelToolCalls {
		t.Error("wrapped Capabilities should report ParallelToolCalls=false")
	}
}

func TestSerialize_NoOpForParallelCapableClient(t *testing.T) {
	fc := &fakeClient{caps: Capabilities{ParallelToolCalls: true}}
	client := WithSerializedToolCalls(fc)
	// Must return the same client unchanged (no wrapping).
	if _, ok := client.(*serializingClient); ok {
		t.Error("parallel-capable client was wrapped; want passthrough")
	}
}

// TestClassify_Table pins S2 row by row: Classify is the kit's retryability
// rule. Row order is precedence — anything chaining context.Canceled is
// terminal before Kind is consulted, so an *APIError wrapping a cancelled
// context is terminal too. A bare context.DeadlineExceeded (the per-attempt
// timeout seen from outside an adapter) is terminal; the retryable shape is
// the adapter-wrapped *APIError{Kind: ErrServer}. Retry-After rides the
// HasRetryAfter presence bit, not a nonzero check.
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
		// Cancellation precedes Kind: the APIError carries Canceled in Err.
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
// feeds: retry.Do makes exactly MaxAttempts calls for a retryable row and
// exactly 1 for each terminal row. The Sleep hook stands in for the timer.
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

// TestWithRetry_RetryAfterPresenceBit pins the presence bit end to end
// through WithRetry (BaseDelay 100ms, Jitter 0): a carried Retry-After of 0
// sleeps exactly 0 and retries immediately; a set RetryAfter with
// HasRetryAfter false is absent — the loop sleeps the 100ms schedule
// instead of the 5s the nonzero check would pick.
func TestWithRetry_RetryAfterPresenceBit(t *testing.T) {
	tests := []struct {
		name string
		err  *APIError
		want time.Duration
	}{
		{"present zero retries immediately", &APIError{Kind: ErrRateLimited, HasRetryAfter: true}, 0},
		{"absent falls back to schedule", &APIError{Kind: ErrRateLimited, RetryAfter: 5 * time.Second}, 100 * time.Millisecond},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fc := &fakeClient{errs: []error{tc.err}, responses: []Response{{}, {Text: "ok"}}}
			var slept []time.Duration
			cfg := retry.Config{
				MaxAttempts: 2,
				BaseDelay:   100 * time.Millisecond,
				Sleep:       func(_ context.Context, d time.Duration) error { slept = append(slept, d); return nil },
			}
			if _, err := WithRetry(fc, cfg).Complete(context.Background(), simpleRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if len(slept) != 1 || slept[0] != tc.want {
				t.Errorf("slept = %v, want [%v]", slept, tc.want)
			}
		})
	}
}

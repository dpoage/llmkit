package provider

// Retry-stage behavior tests exercise Wrap's retry decorator, asserting:
// MaxAttempts N on persistent failure = N wire hits; per-attempt
// RequestTimeout; panic releases the attempt ctx; auth error survives
// cancel; jitter clamp and field-wise Or completion.

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// fakeClient is a scripted Client for retry/serialize stage tests.
type fakeClient struct {
	caps      llmkit.Capabilities
	responses []llmkit.Response
	errs      []error
	calls     int
}

func (f *fakeClient) Capabilities() llmkit.Capabilities { return f.caps }

func (f *fakeClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	i := f.calls
	f.calls++
	var err error
	if i < len(f.errs) {
		err = f.errs[i]
	}
	var resp llmkit.Response
	if i < len(f.responses) {
		resp = f.responses[i]
	}
	return resp, err
}

func rateLimitErr(retryAfter time.Duration) error {
	return &llmkit.APIError{Kind: llmkit.ErrRateLimited, StatusCode: 429, RetryAfter: retryAfter, Provider: "fake", Message: "429"}
}

// blockingClient blocks each Complete call until its context is done, then
// fails the attempt the way an adapter's transport normalization does.
type blockingClient struct {
	blockAttempts int
	calls         int
}

func (b *blockingClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (b *blockingClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	i := b.calls
	b.calls++
	if i < b.blockAttempts {
		<-ctx.Done()
		return llmkit.Response{}, &llmkit.APIError{
			Kind: llmkit.ErrServer, Provider: "fake", Message: ctx.Err().Error(), Err: ctx.Err(),
		}
	}
	return llmkit.Response{Text: "ok", StopReason: llmkit.StopEndTurn}, nil
}

// probingClient records each attempt's context deadline, fails transiently
// for the first probeAttempts attempts, then succeeds.
type probingClient struct {
	probeAttempts int
	calls         int
	deadline      time.Time
	hasDeadline   bool
}

func (p *probingClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (p *probingClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	i := p.calls
	p.calls++
	p.deadline, p.hasDeadline = ctx.Deadline()
	if i < p.probeAttempts {
		return llmkit.Response{}, &llmkit.APIError{Kind: llmkit.ErrServer, Provider: "fake", Message: "transient"}
	}
	return llmkit.Response{Text: "ok", StopReason: llmkit.StopEndTurn}, nil
}

// panicClient captures the attempt context, then panics.
type panicClient struct {
	completeCtx context.Context
}

func (p *panicClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (p *panicClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	p.completeCtx = ctx
	panic("inner complete panic")
}

func recovered(fn func()) (panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
		}
	}()
	fn()
	return false
}

// cancelAfterClient answers every Complete with err after cancelling the
// caller's ctx, so the ctx is done when the retry loop inspects the failure.
type cancelAfterClient struct {
	err    error
	cancel context.CancelFunc
	calls  int
}

func (c *cancelAfterClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *cancelAfterClient) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	c.calls++
	c.cancel()
	return llmkit.Response{}, c.err
}

func TestWrap_RetrySucceedsAfterRateLimit(t *testing.T) {
	fc := &fakeClient{
		errs:      []error{rateLimitErr(0), rateLimitErr(0), nil},
		responses: []llmkit.Response{{}, {}, {Text: "ok", StopReason: llmkit.StopEndTurn}},
	}
	var slept []time.Duration
	cfg := retry.Default()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	cfg.Rand = func() float64 { return 0.5 } // deterministic: jitter factor = 1.0

	client := Wrap(fc, Options{Retry: cfg})
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
	if slept[0] != 500*time.Millisecond {
		t.Errorf("slept[0] = %v, want 500ms", slept[0])
	}
	if slept[1] != 1*time.Second {
		t.Errorf("slept[1] = %v, want 1s", slept[1])
	}
}

func TestWrap_RetryHonorsRetryAfter(t *testing.T) {
	fc := &fakeClient{
		errs: []error{&llmkit.APIError{
			Kind: llmkit.ErrRateLimited, StatusCode: 429, RetryAfter: 3 * time.Second,
			HasRetryAfter: true, Provider: "fake", Message: "429",
		}},
		responses: []llmkit.Response{{}, {Text: "ok"}},
	}
	var slept []time.Duration
	cfg := retry.Default()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }

	client := Wrap(fc, Options{Retry: cfg})
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(slept) != 1 || slept[0] != 3*time.Second {
		t.Errorf("slept = %v, want [3s] (carried Retry-After honored over backoff)", slept)
	}
}

func TestWrap_RetryDoesNotRetryAuth(t *testing.T) {
	authErr := &llmkit.APIError{Kind: llmkit.ErrAuth, StatusCode: 401, Provider: "fake", Message: "401"}
	fc := &fakeClient{errs: []error{authErr}}
	cfg := retry.Default()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := Wrap(fc, Options{Retry: cfg})
	_, err := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(err, llmkit.ErrAuth) {
		t.Errorf("err = %v, want ErrAuth", err)
	}
	if fc.calls != 1 {
		t.Errorf("calls = %d, want 1 (auth is not retryable)", fc.calls)
	}
}

func TestWrap_RetryExhaustsAttempts(t *testing.T) {
	fc := &fakeClient{errs: []error{rateLimitErr(0), rateLimitErr(0), rateLimitErr(0), rateLimitErr(0)}}
	cfg := retry.Default() // 4 attempts
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := Wrap(fc, Options{Retry: cfg})
	_, err := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(err, llmkit.ErrRateLimited) {
		t.Errorf("err = %v, want ErrRateLimited", err)
	}
	if fc.calls != 4 {
		t.Errorf("calls = %d, want 4", fc.calls)
	}
}

// TestWrap_RetryStopsOnContextCancel pins that a caller's ctx ending
// during the backoff sleep after a retryable 429 yields a plain error
// chaining context.Canceled carrying the 429 as text — never the
// retryable *APIError — and the loop is terminal under Classify.
func TestWrap_RetryStopsOnContextCancel(t *testing.T) {
	fc := &fakeClient{errs: []error{rateLimitErr(0), rateLimitErr(0)}}
	cfg := retry.Default()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cfg.Sleep = func(ctx context.Context, d time.Duration) error {
		cancel()
		return ctx.Err()
	}

	client := Wrap(fc, Options{Retry: cfg})
	_, err := client.Complete(ctx, simpleRequest())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled in the chain", err)
	}
	var apiErr *llmkit.APIError
	if errors.As(err, &apiErr) {
		t.Errorf("err = %v (%T), want no *APIError on a cancelled path", err, err)
	}
	if errors.Is(err, llmkit.ErrRateLimited) {
		t.Errorf("err = %v, want the last attempt's error NOT chained (text only)", err)
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("err = %v, want the last attempt's message in the text", err)
	}
	if _, _, retryable := llmkit.Classify(err); retryable {
		t.Errorf("Classify(err) = retryable, want terminal for a cancelled path")
	}
	if fc.calls != 1 {
		t.Errorf("calls = %d, want 1", fc.calls)
	}
}

// TestWrap_RetryAuthErrorSurvivesCancel pins that a 401 followed by the
// caller's ctx ending before the loop returns comes back as that *APIError
// (Kind ErrAuth), not replaced by a cancellation error.
func TestWrap_RetryAuthErrorSurvivesCancel(t *testing.T) {
	authErr := &llmkit.APIError{Kind: llmkit.ErrAuth, StatusCode: 401, Provider: "fake", Message: "401"}
	cfg := retry.Config{MaxAttempts: 3, BaseDelay: time.Hour}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := &cancelAfterClient{err: authErr, cancel: cancel}
	_, err := Wrap(c, Options{Retry: cfg}).Complete(ctx, simpleRequest())
	if ctx.Err() == nil {
		t.Fatal("probe broken: ctx still live when the loop returned")
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || apiErr.Kind != llmkit.ErrAuth {
		t.Fatalf("err = %v (%T), want the 401 *APIError (Kind ErrAuth) returned as-is", err, err)
	}
	if _, _, retryable := llmkit.Classify(err); retryable {
		t.Errorf("Classify(err) = retryable, want terminal")
	}
	if c.calls != 1 {
		t.Errorf("calls = %d, want 1", c.calls)
	}
}

// TestWrap_RetrySaturatedRetryAfterCappedAtMaxDelay closes the
// ParseRetryAfter range at the retry-stage level: a parsed Retry-After
// that saturates to the maximum Duration sleeps exactly MaxDelay, and a
// negative one sleeps exactly 0; neither wraps.
func TestWrap_RetrySaturatedRetryAfterCappedAtMaxDelay(t *testing.T) {
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
			rl := &llmkit.APIError{
				Kind: llmkit.ErrRateLimited, StatusCode: 429, RetryAfter: d, HasRetryAfter: ok,
				Provider: "fake", Message: "429",
			}
			fc := &fakeClient{errs: []error{rl, rl}}
			var slept []time.Duration
			cfg := retry.Default()
			cfg.MaxAttempts = 2
			cfg.MaxDelay = 30 * time.Second
			cfg.Rand = func() float64 { return 0.5 }
			cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }

			client := Wrap(fc, Options{Retry: cfg})
			if _, err := client.Complete(context.Background(), simpleRequest()); !errors.Is(err, llmkit.ErrRateLimited) {
				t.Fatalf("Complete err = %v, want ErrRateLimited", err)
			}
			if len(slept) != 1 || slept[0] != tc.want {
				t.Errorf("Retry-After %q (parsed %v, ok %v): slept = %v, want [%v]", tc.v, d, ok, slept, tc.want)
			}
		})
	}
}

// TestWrap_RetryAfterPresenceBitDrivesSchedule pins the presence bit end
// to end through Wrap (BaseDelay 100ms, Jitter 0 via Rand pin): a carried
// Retry-After of 0 sleeps exactly 0 and retries immediately; a set
// RetryAfter with HasRetryAfter false is absent — the loop sleeps the
// 100ms schedule instead of the 5s the nonzero check would pick.
func TestWrap_RetryAfterPresenceBitDrivesSchedule(t *testing.T) {
	tests := []struct {
		name string
		err  *llmkit.APIError
		want time.Duration
	}{
		{"present zero retries immediately", &llmkit.APIError{Kind: llmkit.ErrRateLimited, HasRetryAfter: true}, 0},
		{"absent falls back to schedule", &llmkit.APIError{Kind: llmkit.ErrRateLimited, RetryAfter: 5 * time.Second}, 100 * time.Millisecond},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fc := &fakeClient{errs: []error{tc.err}, responses: []llmkit.Response{{}, {Text: "ok"}}}
			var slept []time.Duration
			cfg := retry.Config{
				MaxAttempts: 2,
				BaseDelay:   100 * time.Millisecond,
				Rand:        func() float64 { return 0.5 },
				Sleep:       func(_ context.Context, d time.Duration) error { slept = append(slept, d); return nil },
			}
			if _, err := Wrap(fc, Options{Retry: cfg}).Complete(context.Background(), simpleRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if len(slept) != 1 || slept[0] != tc.want {
				t.Errorf("slept = %v, want [%v]", slept, tc.want)
			}
		})
	}
}

// TestWrap_RetryJitterClamped pins Wrap's [0,1] Jitter clamp: negative
// behaves as 0, above 1 behaves as 1, in-range values are preserved.
// An explicit Jitter of 0 resolves through retry.Config.Or to
// retry.Default's 20%, since zero is treated as unset.
func TestWrap_RetryJitterClamped(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  time.Duration
	}{
		{"negative clamped to 0", -0.5, 100 * time.Millisecond},
		{"above 1 clamped to 1", 1.5, 150 * time.Millisecond},
		{"one preserved", 1, 150 * time.Millisecond},
		{"valid mid-range preserved", 0.2, 110 * time.Millisecond},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fc := &fakeClient{errs: []error{rateLimitErr(0)}, responses: []llmkit.Response{{}, {Text: "ok"}}}
			cfg := retry.Config{MaxAttempts: 2, BaseDelay: 100 * time.Millisecond, Jitter: tc.input}
			cfg.Rand = func() float64 { return 0.75 }
			var slept []time.Duration
			cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
			client := Wrap(fc, Options{Retry: cfg})
			if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if len(slept) != 1 {
				t.Fatalf("slept = %v, want exactly one backoff sleep", slept)
			}
			if slept[0] != tc.want {
				t.Errorf("Jitter %v slept %v, want %v", tc.input, slept[0], tc.want)
			}
		})
	}
}

// TestWrap_PerAttemptTimeoutReturnsFast asserts that a single attempt
// whose inner Complete never returns is bounded by RequestTimeout.
func TestWrap_PerAttemptTimeoutReturnsFast(t *testing.T) {
	inner := &blockingClient{blockAttempts: 100}
	cfg := retry.Default()
	cfg.MaxAttempts = 2
	cfg.RequestTimeout = 50 * time.Millisecond
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := Wrap(inner, Options{Retry: cfg})

	done := make(chan struct{})
	var err error
	go func() {
		_, err = client.Complete(context.Background(), simpleRequest())
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Complete did not return; per-attempt timeout failed to bound a blocked request")
	}

	if err == nil {
		t.Fatal("expected an error from a perpetually-blocking client")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
	if inner.calls != 2 {
		t.Errorf("calls = %d, want 2 (timed-out attempt is retried up to MaxAttempts)", inner.calls)
	}
}

// TestWrap_PerAttemptTimeoutIsRetryable pins that a per-attempt timeout
// does NOT abort the retry loop: attempt 1 blocks (times out, surfacing
// the *APIError{Kind: ErrServer} shape an adapter's transport
// normalization produces for a stalled call) and attempt 2 succeeds.
func TestWrap_PerAttemptTimeoutIsRetryable(t *testing.T) {
	inner := &blockingClient{blockAttempts: 1}
	cfg := retry.Default()
	cfg.MaxAttempts = 3
	cfg.RequestTimeout = 50 * time.Millisecond
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := Wrap(inner, Options{Retry: cfg})
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v, want success after a timed-out first attempt", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
	if inner.calls != 2 {
		t.Errorf("calls = %d, want 2 (timeout retried, second attempt succeeded)", inner.calls)
	}
}

// TestWrap_ParentCancelStillAbortsImmediately asserts that when the
// caller's context is cancelled mid-attempt, Complete aborts without
// retrying.
func TestWrap_ParentCancelStillAbortsImmediately(t *testing.T) {
	inner := &blockingClient{blockAttempts: 100}
	cfg := retry.Default()
	cfg.MaxAttempts = 4
	cfg.RequestTimeout = 10 * time.Second
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	ctx, cancel := context.WithCancel(context.Background())
	client := Wrap(inner, Options{Retry: cfg})

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	done := make(chan struct{})
	var err error
	go func() {
		_, err = client.Complete(ctx, simpleRequest())
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Complete did not return after parent cancellation")
	}

	if err == nil {
		t.Fatal("expected an error after parent cancellation")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled (parent cancellation)", err)
	}
	if inner.calls != 1 {
		t.Errorf("calls = %d, want 1 (parent cancellation must abort, not retry)", inner.calls)
	}
}

// TestWrap_DefaultsRequestTimeout asserts the backstop observably: a
// caller-supplied retry.Config that leaves RequestTimeout zero still
// gives every attempt the retry.DefaultRequestTimeout deadline.
func TestWrap_DefaultsRequestTimeout(t *testing.T) {
	inner := &probingClient{probeAttempts: 1}
	cfg := retry.Config{MaxAttempts: 2}
	client := Wrap(inner, Options{Retry: cfg})
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !inner.hasDeadline {
		t.Fatal("attempt context has no deadline; want retry.DefaultRequestTimeout")
	}
	if remaining := time.Until(inner.deadline); remaining > retry.DefaultRequestTimeout || remaining < retry.DefaultRequestTimeout-time.Minute {
		t.Errorf("attempt deadline in %v, want ≈ %v (zero RequestTimeout resolves to DefaultRequestTimeout)", remaining, retry.DefaultRequestTimeout)
	}
}

// TestWrap_PreservesExplicitRequestTimeout asserts that a retry.Config the
// caller supplies survives Wrap unchanged.
func TestWrap_PreservesExplicitRequestTimeout(t *testing.T) {
	want := 42 * time.Second
	inner := &probingClient{probeAttempts: 3}
	cfg := retry.Default()
	cfg.RequestTimeout = want
	cfg.Sleep = func(context.Context, time.Duration) error { return nil }
	client := Wrap(inner, Options{Retry: cfg})
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
	if inner.calls != retry.Default().MaxAttempts {
		t.Errorf("calls = %d, want %d (MaxAttempts preserved)", inner.calls, retry.Default().MaxAttempts)
	}
	if !inner.hasDeadline {
		t.Fatal("attempt context has no deadline")
	}
	if remaining := time.Until(inner.deadline); remaining > want || remaining < want-time.Second {
		t.Errorf("attempt deadline in %v, want ≈ %v (explicit RequestTimeout preserved)", remaining, want)
	}
}

// TestWrap_PanicReleasesAttemptContext pins the per-attempt cleanup
// through the wrapper: the inner panic propagates, and the attempt
// context is still cancelled.
func TestWrap_PanicReleasesAttemptContext(t *testing.T) {
	cfg := retry.Config{MaxAttempts: 3, RequestTimeout: time.Hour}
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }
	pc := &panicClient{}
	if !recovered(func() { _, _ = Wrap(pc, Options{Retry: cfg}).Complete(context.Background(), simpleRequest()) }) {
		t.Error("expected the inner panic to propagate through Wrap Complete")
	}
	if pc.completeCtx == nil {
		t.Fatal("inner Complete never saw an attempt context")
	}
	if pc.completeCtx.Err() != context.Canceled {
		t.Errorf("Complete attempt ctx after panic: err = %v, want context.Canceled", pc.completeCtx.Err())
	}
}

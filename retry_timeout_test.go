package llmkit

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/dpoage/llmkit/retry"
)

// simpleRequest is the minimal completion request used across the root
// package tests.
func simpleRequest() Request {
	return Request{
		System:    "you are a test",
		Messages:  []Message{TextMessage(RoleUser, "hi")},
		MaxTokens: 64,
	}
}

// blockingClient blocks each Complete call until its context is done, then
// returns ctx.Err(). The blockAttempts field configures a fixed run of
// leading blocking attempts before later calls succeed.
type blockingClient struct {
	// blockAttempts is the number of leading attempts that block until their
	// (per-attempt) context expires. Attempts beyond this succeed.
	blockAttempts int
	calls         int
}

func (b *blockingClient) Capabilities() Capabilities { return Capabilities{} }

func (b *blockingClient) Complete(ctx context.Context, req Request) (Response, error) {
	i := b.calls
	b.calls++
	if i < b.blockAttempts {
		<-ctx.Done()
		// Mimic an adapter that received a transport/timeout error it could not
		// classify: it returns the raw context error (bare context.DeadlineExceeded).
		return Response{}, ctx.Err()
	}
	return Response{Text: "ok", StopReason: StopEndTurn}, nil
}

// probingClient records each attempt's context deadline, fails transiently
// for the first probeAttempts attempts, then succeeds.
type probingClient struct {
	probeAttempts int
	calls         int
	deadline      time.Time
	hasDeadline   bool
}

func (p *probingClient) Capabilities() Capabilities { return Capabilities{} }

func (p *probingClient) Complete(ctx context.Context, req Request) (Response, error) {
	i := p.calls
	p.calls++
	p.deadline, p.hasDeadline = ctx.Deadline()
	if i < p.probeAttempts {
		return Response{}, errors.New("transient")
	}
	return Response{Text: "ok", StopReason: StopEndTurn}, nil
}

// TestRetry_PerAttemptTimeout_ReturnsFast asserts that a single attempt whose
// inner Complete never returns is bounded by RequestTimeout: Complete returns
// within roughly the timeout (times MaxAttempts) rather than blocking forever,
// and surfaces a deadline-exceeded-flavored error.
func TestRetry_PerAttemptTimeout_ReturnsFast(t *testing.T) {
	inner := &blockingClient{blockAttempts: 100} // always blocks
	cfg := retry.Default()
	cfg.MaxAttempts = 2
	cfg.RequestTimeout = 50 * time.Millisecond
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil } // skip backoff

	client := WithRetry(inner, cfg)

	done := make(chan struct{})
	var err error
	go func() {
		_, err = client.Complete(context.Background(), simpleRequest())
		close(done)
	}()

	// 2 attempts * 50ms timeout, generous slack for scheduling.
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

// TestRetry_PerAttemptTimeout_IsRetryable asserts that a per-attempt timeout
// does NOT abort the retry loop: attempt 1 blocks (times out), attempt 2
// succeeds, and Complete returns the success.
func TestRetry_PerAttemptTimeout_IsRetryable(t *testing.T) {
	inner := &blockingClient{blockAttempts: 1} // first attempt blocks, then succeeds
	cfg := retry.Default()
	cfg.MaxAttempts = 3
	cfg.RequestTimeout = 50 * time.Millisecond
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := WithRetry(inner, cfg)
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

// TestRetry_PerAttemptTimeout_WrappedAsServerError covers the production
// path where an adapter wraps an unclassified context-deadline error
// into an *APIError{Kind: ErrServer} (as openaiAdapter.normalizeErr does)
// — retryable, so the loop recovers on a later success.
func TestRetry_PerAttemptTimeout_WrappedAsServerError(t *testing.T) {
	inner := &wrappingBlockClient{blockAttempts: 1}
	cfg := retry.Default()
	cfg.MaxAttempts = 3
	cfg.RequestTimeout = 50 * time.Millisecond
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	client := WithRetry(inner, cfg)
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v, want success after a wrapped timeout", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
	if inner.calls != 2 {
		t.Errorf("calls = %d, want 2", inner.calls)
	}
}

// wrappingBlockClient blocks then returns the timeout wrapped as an *APIError,
// matching how the openai adapter normalizes an unrecognized transport error.
type wrappingBlockClient struct {
	blockAttempts int
	calls         int
}

func (w *wrappingBlockClient) Capabilities() Capabilities { return Capabilities{} }

func (w *wrappingBlockClient) Complete(ctx context.Context, req Request) (Response, error) {
	i := w.calls
	w.calls++
	if i < w.blockAttempts {
		<-ctx.Done()
		return Response{}, &APIError{
			Kind:     ErrServer,
			Provider: "fake",
			Message:  ctx.Err().Error(),
			Err:      ctx.Err(),
		}
	}
	return Response{Text: "ok", StopReason: StopEndTurn}, nil
}

// TestRetry_ParentCancelStillAbortsImmediately asserts that when the
// caller's context is cancelled mid-attempt, Complete aborts without
// retrying — a cancelled parent must not be confused with a
// per-attempt deadline.
func TestRetry_ParentCancelStillAbortsImmediately(t *testing.T) {
	inner := &blockingClient{blockAttempts: 100} // always blocks until ctx done
	cfg := retry.Default()
	cfg.MaxAttempts = 4
	cfg.RequestTimeout = 10 * time.Second // long: the parent cancel must win, not the timeout
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }

	ctx, cancel := context.WithCancel(context.Background())
	client := WithRetry(inner, cfg)

	// Cancel the parent shortly after the (blocking) first attempt begins.
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

// TestWithRetry_DefaultsRequestTimeout asserts the backstop observably: a
// caller-supplied retry.Config that leaves RequestTimeout zero still gives
// every attempt the retry.DefaultRequestTimeout deadline, so no construction path
// can produce a client with an unbounded request.
func TestWithRetry_DefaultsRequestTimeout(t *testing.T) {
	inner := &probingClient{probeAttempts: 1}
	cfg := retry.Config{MaxAttempts: 2} // RequestTimeout zero
	client := WithRetry(inner, cfg)
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

// TestWithRetry_JitterClamped verifies observably that WithRetry applies the
// [0,1] Jitter clamp: negative Jitter behaves as 0 (no jitter), Jitter above
// 1 behaves as 1, and in-range values are preserved. Every case runs two
// attempts with Rand pinned to 0.75 (jitter factor 1.5 at Jitter 1) and the
// recorded sleeps assert the effective schedule.
func TestWithRetry_JitterClamped(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  time.Duration
	}{
		{"negative clamped to 0", -0.5, 100 * time.Millisecond},
		{"above 1 clamped to 1", 1.5, 150 * time.Millisecond},
		{"zero preserved", 0, 100 * time.Millisecond},
		{"one preserved", 1, 150 * time.Millisecond},
		{"valid mid-range preserved", 0.2, 110 * time.Millisecond},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			inner := &probingClient{probeAttempts: 1}
			cfg := retry.Config{MaxAttempts: 2, BaseDelay: 100 * time.Millisecond, Jitter: tc.input}
			cfg.Rand = func() float64 { return 0.75 }
			var slept []time.Duration
			cfg.Sleep = func(ctx context.Context, d time.Duration) error { slept = append(slept, d); return nil }
			client := WithRetry(inner, cfg)
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

// TestWithRetry_PreservesExplicitRequestTimeout asserts that a retry.Config
// the caller supplies (as provider.New does after applying its
// zero-MaxAttempts default) survives WithRetry unchanged: the explicit
// RequestTimeout bounds the attempt and the default MaxAttempts still runs
// all four attempts.
func TestWithRetry_PreservesExplicitRequestTimeout(t *testing.T) {
	want := 42 * time.Second
	inner := &probingClient{probeAttempts: 3} // fails 3 times, succeeds on the 4th
	cfg := retry.Default()
	cfg.RequestTimeout = want
	client := WithRetry(inner, cfg)
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

// panicClient captures the attempt context, then panics: the loop must
// release that context even on the panic path.
type panicClient struct {
	completeCtx context.Context
	streamCtx   context.Context
}

func (p *panicClient) Capabilities() Capabilities { return Capabilities{} }

func (p *panicClient) Complete(ctx context.Context, req Request) (Response, error) {
	p.completeCtx = ctx
	panic("inner complete panic")
}

func (p *panicClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	p.streamCtx = ctx
	panic("inner stream panic")
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

// TestWithRetry_PanicReleasesAttemptContext pins the per-attempt cleanup
// through the wrapper: the inner panic propagates, and the attempt context
// is still cancelled — its cancel runs via defer during the unwind, never an
// inline call a panic could skip. (The same guarantee for a bare retry.Do
// call is pinned in the retry package.)
func TestWithRetry_PanicReleasesAttemptContext(t *testing.T) {
	// Complete path. RequestTimeout is an hour, so an un-released context
	// would still report Err() == nil here.
	cfg := retry.Config{MaxAttempts: 3, RequestTimeout: time.Hour}
	cfg.Sleep = func(ctx context.Context, d time.Duration) error { return nil }
	pc := &panicClient{}
	if !recovered(func() { _, _ = WithRetry(pc, cfg).Complete(context.Background(), simpleRequest()) }) {
		t.Error("expected the inner panic to propagate through WithRetry Complete")
	}
	if pc.completeCtx == nil {
		t.Fatal("inner Complete never saw an attempt context")
	}
	if pc.completeCtx.Err() != context.Canceled {
		t.Errorf("Complete attempt ctx after panic: err = %v, want context.Canceled", pc.completeCtx.Err())
	}

	// Stream path.
	ps := &panicClient{}
	if !recovered(func() {
		_, _ = Stream(context.Background(), WithRetry(ps, cfg), simpleRequest(), func(Delta) error { return nil })
	}) {
		t.Error("expected the inner panic to propagate through WithRetry Stream")
	}
	if ps.streamCtx == nil {
		t.Fatal("inner Stream never saw an attempt context")
	}
	if ps.streamCtx.Err() != context.Canceled {
		t.Errorf("Stream attempt ctx after panic: err = %v, want context.Canceled", ps.streamCtx.Err())
	}
}

package llmkit

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit/retry"
)

// captureObserver collects every event it receives — with the context it
// was delivered on — for asserting emission counts, ordering, payloads, and
// which context emitters hand their sinks.
type captureObserver struct {
	mu     sync.Mutex
	events []Event
	ctxs   []context.Context
}

func (c *captureObserver) Observe(ctx context.Context, ev Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
	c.ctxs = append(c.ctxs, ctx)
}

func (c *captureObserver) snapshot() []Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Event(nil), c.events...)
}

func (c *captureObserver) ctxAt(i int) context.Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ctxs[i]
}

// byKind returns the captured events of one kind, in emission order.
func (c *captureObserver) byKind(k EventKind) []Event {
	var out []Event
	for _, ev := range c.snapshot() {
		if ev.Kind == k {
			out = append(out, ev)
		}
	}
	return out
}

// retrySleeps is a retry.Config whose backoff sleeps are recorded, so tests
// can pin that the Completion Duration covers the attempts plus the backoff.
type retrySleeps struct {
	cfg   retry.Config
	slept []time.Duration
}

// newRetrySleeps builds a retry.Config whose backoff sleeps are recorded
// AND actually taken (microsecond-scale, capped at MaxDelay), so tests can
// pin that the Completion Duration covers the attempts plus real backoff
// wall time.
func newRetrySleeps(maxAttempts int) *retrySleeps {
	r := &retrySleeps{cfg: retry.Config{
		MaxAttempts:    maxAttempts,
		BaseDelay:      200 * time.Microsecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: time.Second,
	}}
	r.cfg.Sleep = func(_ context.Context, d time.Duration) error {
		r.slept = append(r.slept, d)
		time.Sleep(d)
		return nil
	}
	return r
}

func okResponse() Response {
	return Response{
		Text:       "hello",
		Blocks:     []Block{Text("hello")},
		Usage:      Usage{InputTokens: 3, OutputTokens: 2},
		StopReason: StopEndTurn,
	}
}

// TestObserve_NilObserverPassthrough: a nil observer returns the client
// unchanged, mirroring WithRecorder.
func TestObserve_NilObserverPassthrough(t *testing.T) {
	fc := &fakeClient{}
	if got := Observe(fc, nil, "p", "m"); got != Client(fc) {
		t.Fatalf("Observe(nil observer) returned a different client: %T", got)
	}
}

// TestObserve_CompleteEmitsExactlyOneEvent pins the full shape of the
// Completion event on the success path.
func TestObserve_CompleteEmitsExactlyOneEvent(t *testing.T) {
	fc := &fakeClient{responses: []Response{okResponse()}}
	obs := &captureObserver{}
	runID := NewRunID()
	client := Observe(WithRetry(fc, newRetrySleeps(4).cfg), obs, "anthropic", "claude-test")

	req := simpleRequest()
	resp, err := client.Complete(WithRun(context.Background(), runID), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	events := obs.snapshot()
	if len(events) != 1 {
		t.Fatalf("emitted %d events, want exactly 1 Completion: %+v", len(events), events)
	}
	ev := events[0]
	if ev.Kind != KindCompletion || ev.Completion == nil {
		t.Fatalf("Kind=%q payload=%+v, want KindCompletion with a payload", ev.Kind, ev.Completion)
	}
	ce := ev.Completion
	if !reflect.DeepEqual(ce.Request, req) {
		t.Errorf("Request = %+v, want the request as received %+v", ce.Request, req)
	}
	if !reflect.DeepEqual(ce.Response, resp) {
		t.Errorf("Response = %+v, want the returned response %+v", ce.Response, resp)
	}
	if ce.Err != "" {
		t.Errorf("Err = %q, want empty on success", ce.Err)
	}
	if ce.Provider != "anthropic" || ce.Model != "claude-test" {
		t.Errorf("tags = %q/%q, want anthropic/claude-test", ce.Provider, ce.Model)
	}
	if ev.RunID != runID {
		t.Errorf("RunID = %q, want %q from the context", ev.RunID, runID)
	}
	if ev.SpanID == "" {
		t.Error("SpanID empty: the emitter must mint a fresh span per completion")
	}
	if ev.ParentRunID != "" || ev.Step != 0 {
		t.Errorf("ParentRunID=%q Step=%d, want both zero: only the Runner stamps them", ev.ParentRunID, ev.Step)
	}
	if ev.SchemaVersion != EventSchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", ev.SchemaVersion, EventSchemaVersion)
	}
	if ev.Duration <= 0 {
		t.Errorf("Duration = %v, want the whole call including inner retries", ev.Duration)
	}
	if ev.Start != nil || ev.Attempt != nil || ev.ToolRun != nil || ev.Compaction != nil ||
		ev.Steer != nil || ev.Finalize != nil || ev.Decision != nil || ev.Embed != nil || ev.Exec != nil {
		t.Error("a payload other than Completion is set on a Completion event")
	}
}

// TestObserve_CompleteError: a failed completion's event carries the error
// text and the zero Response — even when the inner client returned a partial
// response alongside the error — while the caller still receives that
// partial response unchanged.
func TestObserve_CompleteError(t *testing.T) {
	fc := &fakeClient{
		errs:      []error{&APIError{Kind: ErrAuth, StatusCode: 401, Provider: "fake", Message: "bad key"}},
		responses: []Response{{Text: "partial"}},
	}
	obs := &captureObserver{}
	client := Observe(fc, obs, "p", "m")

	resp, err := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(err, ErrAuth) {
		t.Fatalf("err = %v, want ErrAuth", err)
	}
	if resp.Text != "partial" {
		t.Errorf("caller Response.Text = %q, want the inner partial response passed through", resp.Text)
	}
	events := obs.snapshot()
	if len(events) != 1 || events[0].Completion == nil {
		t.Fatalf("events = %+v, want exactly one Completion", events)
	}
	ce := events[0].Completion
	if ce.Err != err.Error() {
		t.Errorf("Err = %q, want the error text %q", ce.Err, err.Error())
	}
	if !reflect.DeepEqual(ce.Response, Response{}) {
		t.Errorf("event Response = %+v, want the zero Response on failure (not the partial the caller got)", ce.Response)
	}
}

// TestObserve_StreamEmitsExactlyOneEvent: the native streaming path delivers
// deltas and closes with one Completion whose Response is the final one.
func TestObserve_StreamEmitsExactlyOneEvent(t *testing.T) {
	inner := &scriptedStreamClient{
		deltas: []Delta{{Kind: DeltaText, Text: "he"}, {Kind: DeltaText, Text: "llo"}},
		resp:   okResponse(),
	}
	obs := &captureObserver{}
	client := Observe(inner, obs, "openai", "gpt-test")

	var got []Delta
	resp, err := Stream(context.Background(), client, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(got) != 2 || strings.Join([]string{got[0].Text, got[1].Text}, "") != "hello" {
		t.Errorf("deltas = %+v, want the scripted fragments", got)
	}
	if inner.calls != 1 || inner.completes != 0 {
		t.Errorf("inner calls=%d completes=%d, want the native stream taken once", inner.calls, inner.completes)
	}
	events := obs.snapshot()
	if len(events) != 1 || events[0].Completion == nil {
		t.Fatalf("events = %+v, want exactly one Completion", events)
	}
	ce := events[0].Completion
	if !reflect.DeepEqual(ce.Response, resp) {
		t.Errorf("Response = %+v, want the assembled final response %+v", ce.Response, resp)
	}
	if ce.Err != "" {
		t.Errorf("Err = %q, want empty on success", ce.Err)
	}
}

// TestObserve_StreamError: a failed stream still closes with exactly one
// Completion carrying the error text and the zero Response.
func TestObserve_StreamError(t *testing.T) {
	inner := &scriptedStreamClient{err: &APIError{Kind: ErrServer, StatusCode: 500, Provider: "openai", Message: "boom"}}
	obs := &captureObserver{}
	client := Observe(inner, obs, "openai", "gpt-test")

	_, err := Stream(context.Background(), client, simpleRequest(), func(Delta) error { return nil })
	if !errors.Is(err, ErrServer) {
		t.Fatalf("err = %v, want ErrServer", err)
	}
	events := obs.snapshot()
	if len(events) != 1 || events[0].Completion == nil {
		t.Fatalf("events = %+v, want exactly one Completion", events)
	}
	ce := events[0].Completion
	if ce.Err != err.Error() {
		t.Errorf("Err = %q, want %q", ce.Err, err.Error())
	}
	if !reflect.DeepEqual(ce.Response, Response{}) {
		t.Errorf("Response = %+v, want the zero Response on failure", ce.Response)
	}
}

// TestObserve_StreamAndCompleteDeepEqual: the stream path and the complete
// path observe the same event modulo the per-call facts (Time, Duration, and
// the freshly minted SpanID).
func TestObserve_StreamAndCompleteDeepEqual(t *testing.T) {
	resp := okResponse()
	runID := NewRunID()

	obsC := &captureObserver{}
	c := Observe(&fakeClient{responses: []Response{resp}}, obsC, "anthropic", "m")
	if _, err := c.Complete(WithRun(context.Background(), runID), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	obsS := &captureObserver{}
	s := Observe(&fakeClient{responses: []Response{resp}}, obsS, "anthropic", "m")
	if _, err := Stream(WithRun(context.Background(), runID), s, simpleRequest(), nil); err != nil {
		t.Fatalf("Stream: %v", err)
	}

	strip := func(ev Event) Event {
		ev.Time = time.Time{}
		ev.Duration = 0
		ev.SpanID = ""
		return ev
	}
	gotC, gotS := obsC.snapshot(), obsS.snapshot()
	if len(gotC) != 1 || len(gotS) != 1 {
		t.Fatalf("events: complete=%d stream=%d, want 1 each", len(gotC), len(gotS))
	}
	if !reflect.DeepEqual(strip(gotC[0]), strip(gotS[0])) {
		t.Errorf("stream event differs from complete event modulo Time/Duration/SpanID:\n%+v\n%+v",
			strip(gotC[0]), strip(gotS[0]))
	}
}

// TestObserve_AttemptsJoinCompletionSpan: N retried attempts produce Attempt
// events numbered 1..N with the completion's SpanID. Failed ones carry the
// error, the zero Response (even when the inner client returned a partial
// one), and the APIError's status and Retry-After; the completion's Duration
// covers every attempt plus the backoff between them.
func TestObserve_AttemptsJoinCompletionSpan(t *testing.T) {
	fc := &fakeClient{
		// The inner client returns a partial Response alongside each
		// failure, so the zero-Response assertion below discriminates:
		// the event is zeroed by the emitter, not by the fixture.
		errs:      []error{rateLimitErr(2 * time.Second), rateLimitErr(2 * time.Second)},
		responses: []Response{{Text: "partial"}, {Text: "partial"}, okResponse()},
	}
	obs := &captureObserver{}
	runID := NewRunID()
	rc := newRetrySleeps(4)
	client := Observe(WithRetryObserver(fc, rc.cfg, obs, "anthropic", "claude-test"), obs, "anthropic", "claude-test")

	resp, err := client.Complete(WithRun(context.Background(), runID), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if fc.calls != 3 {
		t.Fatalf("wire calls = %d, want 3", fc.calls)
	}
	completions := obs.byKind(KindCompletion)
	attempts := obs.byKind(KindAttempt)
	if len(completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(completions))
	}
	if len(attempts) != 3 {
		t.Fatalf("attempts = %d, want 3 (one per wire call)", len(attempts))
	}
	span := completions[0].SpanID
	if span == "" {
		t.Fatal("completion SpanID empty")
	}
	var attemptDurations time.Duration
	for i, ev := range attempts {
		ae := ev.Attempt
		if ae == nil {
			t.Fatalf("attempt %d has no payload", i)
		}
		attemptDurations += ev.Duration
		if ae.Attempt != i+1 {
			t.Errorf("attempt %d numbered %d, want %d", i, ae.Attempt, i+1)
		}
		if ev.SpanID != span {
			t.Errorf("attempt %d SpanID = %q, want the completion's %q", i, ev.SpanID, span)
		}
		if ev.RunID != runID || ev.Step != 0 {
			t.Errorf("attempt %d RunID=%q Step=%d, want the run id and Step 0", i, ev.RunID, ev.Step)
		}
		if ae.Provider != "anthropic" || ae.Model != "claude-test" {
			t.Errorf("attempt %d tags = %q/%q", i, ae.Provider, ae.Model)
		}
		if i < 2 {
			if ae.Err == "" {
				t.Errorf("attempt %d Err empty, want the rate-limit failure", i)
			}
			if !reflect.DeepEqual(ae.Response, Response{}) {
				t.Errorf("attempt %d Response = %+v, want zero on failure (not the partial the inner returned)", i, ae.Response)
			}
			if ae.StatusCode != 429 {
				t.Errorf("attempt %d StatusCode = %d, want 429 from the APIError", i, ae.StatusCode)
			}
			if ae.RetryAfter != 2*time.Second {
				t.Errorf("attempt %d RetryAfter = %v, want 2s as read by the retry classifier", i, ae.RetryAfter)
			}
		} else {
			if ae.Err != "" {
				t.Errorf("successful attempt Err = %q", ae.Err)
			}
			if ae.StatusCode != 0 || ae.RetryAfter != 0 {
				t.Errorf("successful attempt StatusCode/RetryAfter = %d/%v, want zeros", ae.StatusCode, ae.RetryAfter)
			}
			if !reflect.DeepEqual(ae.Response, resp) {
				t.Errorf("successful attempt Response = %+v, want the completion's %+v", ae.Response, resp)
			}
		}
	}
	var backoff time.Duration
	for _, d := range rc.slept {
		backoff += d
	}
	if completions[0].Duration < attemptDurations+backoff {
		t.Errorf("completion Duration %v < attempts %v + backoff %v: the whole call is not covered",
			completions[0].Duration, attemptDurations, backoff)
	}
}

// TestObserve_AttemptEventsUseRetryCallerContext: attempt events are
// delivered on the retry caller's context. With a non-zero RequestTimeout
// the per-attempt context is done by emission time — a ctx-honouring sink
// must still see ok == false from Deadline and a nil Err, or it would drop
// exactly the timed-out attempts.
func TestObserve_AttemptEventsUseRetryCallerContext(t *testing.T) {
	stalls := &stallClient{}
	obs := &captureObserver{}
	cfg := retry.Config{
		MaxAttempts:    2,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: 50 * time.Millisecond,
		Sleep:          func(context.Context, time.Duration) error { return nil },
	}
	client := WithRetryObserver(stalls, cfg, obs, "p", "m")

	// Plain Background: any deadline the observer sees would have to come
	// from the retry stage's per-attempt context, which must not leak.
	_, err := client.Complete(context.Background(), simpleRequest())
	if err == nil {
		t.Fatal("Complete: want the final timeout error")
	}
	if stalls.calls != 2 {
		t.Fatalf("wire calls = %d, want 2", stalls.calls)
	}
	attempts := obs.byKind(KindAttempt)
	if len(attempts) != 2 {
		t.Fatalf("attempt events = %d, want 2 (both timed-out attempts delivered)", len(attempts))
	}
	for i, ev := range attempts {
		ctx := obs.ctxAt(i)
		if _, ok := ctx.Deadline(); ok {
			t.Errorf("attempt %d delivered on a context with a deadline", i)
		}
		if ctx.Err() != nil {
			t.Errorf("attempt %d delivered with ctx.Err() = %v, want nil", i, ctx.Err())
		}
		if ev.Attempt == nil || ev.Attempt.Err == "" {
			t.Errorf("attempt %d carries no error; want the timeout failure", i)
		}
	}
}

// stallClient blocks every Complete until its context is done, simulating a
// wire call the per-attempt RequestTimeout has to abort.
type stallClient struct {
	calls int
}

func (s *stallClient) Capabilities() Capabilities { return Capabilities{} }

func (s *stallClient) Complete(ctx context.Context, req Request) (Response, error) {
	s.calls++
	<-ctx.Done()
	return Response{}, ctx.Err()
}

// TestObserve_FailedFinalAttempt: a terminal failure still emits its Attempt
// and its Completion — both with the error text and zero responses.
func TestObserve_FailedFinalAttempt(t *testing.T) {
	fc := &fakeClient{errs: []error{&APIError{Kind: ErrAuth, StatusCode: 401, Provider: "fake", Message: "denied"}}}
	obs := &captureObserver{}
	client := Observe(WithRetryObserver(fc, newRetrySleeps(4).cfg, obs, "p", "m"), obs, "p", "m")

	_, err := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(err, ErrAuth) {
		t.Fatalf("err = %v, want ErrAuth", err)
	}
	attempts, completions := obs.byKind(KindAttempt), obs.byKind(KindCompletion)
	if len(attempts) != 1 || len(completions) != 1 {
		t.Fatalf("attempts=%d completions=%d, want 1 and 1", len(attempts), len(completions))
	}
	if attempts[0].Attempt.Err != err.Error() {
		t.Errorf("attempt Err = %q, want %q", attempts[0].Attempt.Err, err.Error())
	}
	if attempts[0].Attempt.StatusCode != 401 {
		t.Errorf("attempt StatusCode = %d, want 401 from the APIError", attempts[0].Attempt.StatusCode)
	}
	if completions[0].Completion.Err != err.Error() {
		t.Errorf("completion Err = %q, want %q", completions[0].Completion.Err, err.Error())
	}
	if !reflect.DeepEqual(attempts[0].Attempt.Response, Response{}) ||
		!reflect.DeepEqual(completions[0].Completion.Response, Response{}) {
		t.Error("failed attempt/completion must carry the zero Response")
	}
}

// TestObserve_MintsFreshSpanPerCompletion: the span is minted
// unconditionally — an inherited span is never reused, and two completions
// never share one (the nested-completion rule).
func TestObserve_MintsFreshSpanPerCompletion(t *testing.T) {
	fc := &fakeClient{responses: []Response{okResponse(), okResponse(), okResponse()}}
	obs := &captureObserver{}
	client := Observe(fc, obs, "p", "m")

	// A context that already carries a span, as a harness would hand one to
	// a nested caller: the emitter must mint its own anyway.
	inherited := NewSpanID()
	ctx := WithSpan(WithRun(context.Background(), NewRunID()), inherited)
	if _, err := client.Complete(ctx, simpleRequest()); err != nil {
		t.Fatalf("first Complete: %v", err)
	}
	if _, err := client.Complete(ctx, simpleRequest()); err != nil {
		t.Fatalf("second Complete: %v", err)
	}
	events := obs.snapshot()
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	if events[0].SpanID == inherited || events[1].SpanID == inherited {
		t.Error("emitter reused the inherited span; spans must be minted per completion")
	}
	if events[0].SpanID == "" || events[0].SpanID == events[1].SpanID {
		t.Errorf("spans %q and %q: each completion needs its own fresh span",
			events[0].SpanID, events[1].SpanID)
	}
}

// TestObserve_ObserverPanicPropagates: a panicking observer is a harness bug
// and surfaces to the caller, never swallowed.
func TestObserve_ObserverPanicPropagates(t *testing.T) {
	panicking := ObserverFunc(func(context.Context, Event) { panic("sink bug") })
	client := Observe(&fakeClient{responses: []Response{okResponse()}}, panicking, "p", "m")
	if !recovered(func() { _, _ = client.Complete(context.Background(), simpleRequest()) }) {
		t.Error("expected the observer panic to propagate out of Complete")
	}
	streaming := Observe(&scriptedStreamClient{resp: okResponse()}, panicking, "p", "m")
	if !recovered(func() { _, _ = Stream(context.Background(), streaming, simpleRequest(), nil) }) {
		t.Error("expected the observer panic to propagate out of Stream")
	}
}

// TestObserve_SynthesizedStreamFromPlainClient: wrapping a Complete-only
// client still observes one Completion per Stream call — synthesis happens
// on the same response Complete returned.
func TestObserve_SynthesizedStreamFromPlainClient(t *testing.T) {
	resp := okResponse()
	inner := &fakeClient{responses: []Response{resp}}
	obs := &captureObserver{}
	client := Observe(inner, obs, "p", "m")

	var texts []string
	if _, err := Stream(context.Background(), client, simpleRequest(), func(d Delta) error {
		if d.Kind == DeltaText {
			texts = append(texts, d.Text)
		}
		return nil
	}); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if strings.Join(texts, "") != "hello" {
		t.Errorf("synthesized text = %q, want %q", strings.Join(texts, ""), "hello")
	}
	if inner.calls != 1 {
		t.Errorf("inner completes = %d, want 1", inner.calls)
	}
	events := obs.snapshot()
	if len(events) != 1 || events[0].Completion == nil {
		t.Fatalf("events = %+v, want exactly one Completion", events)
	}
	if !reflect.DeepEqual(events[0].Completion.Response, resp) {
		t.Errorf("Response = %+v, want the synthesized response %+v", events[0].Completion.Response, resp)
	}
}

// TestObserve_StreamAttemptsExhausted: retrying a native stream emits one
// Attempt event per wire stream call — numbered 1..N, each joined to the
// (failed) completion's SpanID, each carrying its failure.
func TestObserve_StreamAttemptsExhausted(t *testing.T) {
	inner := &scriptedStreamClient{err: rateLimitErr(0)}
	obs := &captureObserver{}
	client := Observe(WithRetryObserver(inner, newRetrySleeps(3).cfg, obs, "openai", "gpt-test"), obs, "openai", "gpt-test")

	runID := NewRunID()
	_, err := Stream(WithRun(context.Background(), runID), client, simpleRequest(), func(Delta) error { return nil })
	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
	if inner.calls != 3 {
		t.Fatalf("wire stream calls = %d, want 3", inner.calls)
	}
	completions := obs.byKind(KindCompletion)
	attempts := obs.byKind(KindAttempt)
	if len(completions) != 1 || len(attempts) != 3 {
		t.Fatalf("completions=%d attempts=%d, want 1 and 3", len(completions), len(attempts))
	}
	span := completions[0].SpanID
	for i, ev := range attempts {
		ae := ev.Attempt
		if ae == nil {
			t.Fatalf("attempt %d has no payload", i)
		}
		if ae.Attempt != i+1 {
			t.Errorf("attempt %d numbered %d, want %d", i, ae.Attempt, i+1)
		}
		if ev.SpanID != span {
			t.Errorf("attempt %d SpanID = %q, want the completion's %q", i, ev.SpanID, span)
		}
		if ae.Err == "" || !reflect.DeepEqual(ae.Response, Response{}) {
			t.Errorf("attempt %d = %q/%+v, want the failure with a zero Response", i, ae.Err, ae.Response)
		}
	}
	if completions[0].Completion.Err == "" {
		t.Error("failed stream completion carries no error")
	}
}

// TestObserve_StreamAttemptSucceedsAfterRetry: a stream that succeeds on a
// later attempt numbers its attempts across the wire calls, joins them to
// the successful completion's span, and the successful attempt's Response
// matches the completion's.
func TestObserve_StreamAttemptSucceedsAfterRetry(t *testing.T) {
	inner := &scriptedAttempts{
		errs:  []error{rateLimitErr(0)},
		resps: []Response{{}, okResponse()},
	}
	obs := &captureObserver{}
	client := Observe(WithRetryObserver(inner, newRetrySleeps(4).cfg, obs, "openai", "gpt-test"), obs, "openai", "gpt-test")

	resp, err := Stream(context.Background(), client, simpleRequest(), func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if inner.calls != 2 {
		t.Fatalf("wire stream calls = %d, want 2", inner.calls)
	}
	completions := obs.byKind(KindCompletion)
	attempts := obs.byKind(KindAttempt)
	if len(completions) != 1 || len(attempts) != 2 {
		t.Fatalf("completions=%d attempts=%d, want 1 and 2", len(completions), len(attempts))
	}
	for i, ev := range attempts {
		if ev.Attempt == nil || ev.Attempt.Attempt != i+1 {
			t.Errorf("attempt %d numbered %+v, want %d", i, ev.Attempt, i+1)
		}
		if ev.SpanID != completions[0].SpanID {
			t.Errorf("attempt %d SpanID = %q, want the completion's %q", i, ev.SpanID, completions[0].SpanID)
		}
	}
	if attempts[0].Attempt.Err == "" || !reflect.DeepEqual(attempts[0].Attempt.Response, Response{}) {
		t.Errorf("failed attempt = %q/%+v, want the failure with a zero Response",
			attempts[0].Attempt.Err, attempts[0].Attempt.Response)
	}
	if !reflect.DeepEqual(attempts[1].Attempt.Response, resp) {
		t.Errorf("successful attempt Response = %+v, want the completion's %+v", attempts[1].Attempt.Response, resp)
	}
}

// TestObserve_StreamPartialResponseDiscriminated: an inner stream that
// returns a partial Response alongside a terminal error yields that partial
// to the CALLER unchanged, while the Completion event carries the zero
// Response — the event zeroing must not depend on the fixture returning zero.
func TestObserve_StreamPartialResponseDiscriminated(t *testing.T) {
	inner := &scriptedAttempts{
		errs:  []error{&APIError{Kind: ErrServer, StatusCode: 500, Provider: "openai", Message: "boom"}},
		resps: []Response{{Text: "partial"}},
	}
	obs := &captureObserver{}
	client := Observe(inner, obs, "openai", "gpt-test")

	resp, err := Stream(context.Background(), client, simpleRequest(), func(Delta) error { return nil })
	if !errors.Is(err, ErrServer) {
		t.Fatalf("err = %v, want ErrServer", err)
	}
	if resp.Text != "partial" {
		t.Errorf("caller Response.Text = %q, want the inner partial response passed through", resp.Text)
	}
	completions := obs.byKind(KindCompletion)
	if len(completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(completions))
	}
	if completions[0].Completion.Err != err.Error() {
		t.Errorf("Err = %q, want %q", completions[0].Completion.Err, err.Error())
	}
	if !reflect.DeepEqual(completions[0].Completion.Response, Response{}) {
		t.Errorf("event Response = %+v, want the zero Response on failure (not the partial the caller got)",
			completions[0].Completion.Response)
	}
}

// TestWithRetry_NilObserverEmitsNothing: WithRetry keeps its exact old
// behavior — no events — when a stack is wrapped only by an outer Observe.
func TestWithRetry_NilObserverEmitsNothing(t *testing.T) {
	fc := &fakeClient{responses: []Response{okResponse()}}
	obs := &captureObserver{}
	client := Observe(WithRetry(fc, newRetrySleeps(4).cfg), obs, "p", "m")
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for _, ev := range obs.snapshot() {
		if ev.Kind == KindAttempt {
			t.Error("WithRetry emitted an Attempt event; it must stay silent")
		}
	}
	if len(obs.snapshot()) != 1 {
		t.Errorf("events = %d, want only the Completion", len(obs.snapshot()))
	}
}

// scriptedAttempts is a StreamingClient with per-call error and response
// scripts, so retry tests can script fail-then-succeed sequences and
// partial-response failures the shared scriptedStreamClient cannot express.
type scriptedAttempts struct {
	caps  Capabilities
	errs  []error // per Stream call; nil means success with the scripted resp
	resps []Response
	calls int
}

func (s *scriptedAttempts) Capabilities() Capabilities { return s.caps }

func (s *scriptedAttempts) Complete(ctx context.Context, req Request) (Response, error) {
	return Response{}, errors.New("scriptedAttempts: unexpected Complete call")
}

func (s *scriptedAttempts) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	i := s.calls
	s.calls++
	if i < len(s.errs) && s.errs[i] != nil {
		var resp Response
		if i < len(s.resps) {
			resp = s.resps[i]
		}
		return resp, s.errs[i]
	}
	var resp Response
	if i < len(s.resps) {
		resp = s.resps[i]
	} else if len(s.resps) > 0 {
		resp = s.resps[len(s.resps)-1]
	}
	return resp, nil
}

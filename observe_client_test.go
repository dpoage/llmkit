package llmkit

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

// fakeClient is a scripted Client for root-package Observe tests: per-call
// responses and errors in order.
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

// scriptedStreamClient implements StreamingClient with a canned delta
// sequence, final response, and optional terminal error.
type scriptedStreamClient struct {
	caps      Capabilities
	deltas    []Delta
	resp      Response
	err       error
	calls     int
	completes int
}

func (s *scriptedStreamClient) Capabilities() Capabilities { return s.caps }

func (s *scriptedStreamClient) Complete(ctx context.Context, req Request) (Response, error) {
	s.completes++
	return Response{}, errors.New("scriptedStreamClient: unexpected Complete call")
}

func (s *scriptedStreamClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	s.calls++
	if fn != nil {
		for _, d := range s.deltas {
			if err := fn(d); err != nil {
				return Response{}, err
			}
		}
	}
	if s.err != nil {
		return s.resp, s.err
	}
	return s.resp, nil
}

// safeClient is a concurrency-safe scripted Client, for the concurrent-spans
// test (fakeClient's calls counter races under concurrent use).
type safeClient struct {
	mu    sync.Mutex
	calls int
	resp  Response
}

func (s *safeClient) Capabilities() Capabilities { return Capabilities{} }

func (s *safeClient) Complete(ctx context.Context, req Request) (Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	return s.resp, nil
}

// withIdentity wraps a Client to add a fixed Identity. Client is embedded
// as an interface field, so when the wrapped value also implements
// StreamingClient, Go promotes Stream onto withIdentity too — it satisfies
// StreamingClient automatically, the same forwarding every provider stage
// does explicitly.
type withIdentity struct {
	Client
	id Identity
}

func (w withIdentity) Identity() Identity { return w.id }

// Stream forwards to the package-level Stream over the wrapped client, so
// withIdentity satisfies StreamingClient (matching every provider stage,
// which always implements Stream this way) whether or not the wrapped
// client streams natively.
func (w withIdentity) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	return Stream(ctx, w.Client, req, fn)
}

func simpleRequest() Request {
	return Request{
		System:    "you are a test",
		Messages:  []Message{TextMessage(RoleUser, "hi")},
		MaxTokens: 64,
	}
}

func okResponse() Response {
	return Response{
		Text:       "hello",
		Blocks:     []Block{Text("hello")},
		Usage:      Usage{InputTokens: 3, OutputTokens: 2},
		StopReason: StopEndTurn,
	}
}

// captureObserver collects every event it receives — with the delivery
// context — for asserting counts, ordering, and payloads.
type captureObserver struct {
	mu     sync.Mutex
	events []Event
}

func (c *captureObserver) Observe(ctx context.Context, ev Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
}

func (c *captureObserver) snapshot() []Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Event(nil), c.events...)
}

// recovered runs fn and reports whether it panicked.
func recovered(fn func()) (panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
		}
	}()
	fn()
	return false
}

// TestObserve_NilObserverPassthrough pins a nil observer returns the client unchanged.
func TestObserve_NilObserverPassthrough(t *testing.T) {
	fc := &fakeClient{}
	if got := Observe(fc, nil); got != Client(fc) {
		t.Fatalf("Observe(nil observer) returned a different client: %T", got)
	}
}

// TestObserve_CompleteEmitsExactlyOneEvent pins the full shape of the
// Completion event on the success path, tagged with IdentityOf(c).
func TestObserve_CompleteEmitsExactlyOneEvent(t *testing.T) {
	fc := withIdentity{Client: &fakeClient{responses: []Response{okResponse()}}, id: Identity{Provider: "anthropic", Model: "claude-test"}}
	obs := &captureObserver{}
	runID := NewRunID()
	client := Observe(fc, obs)

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
		t.Errorf("tags = %q/%q, want anthropic/claude-test (IdentityOf(c))", ce.Provider, ce.Model)
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
		t.Errorf("Duration = %v, want the whole call covered", ev.Duration)
	}
	if ev.Start != nil || ev.Attempt != nil || ev.ToolRun != nil || ev.Compaction != nil ||
		ev.Steer != nil || ev.Finalize != nil || ev.Decision != nil || ev.Embed != nil || ev.Exec != nil {
		t.Error("a payload other than Completion is set on a Completion event")
	}
}

// TestObserve_CompleteError pins a failed completion carries the error text
// and zero Response while the caller still receives the inner partial
// unchanged; a plain (unidentified) client tags the event with zero Identity.
func TestObserve_CompleteError(t *testing.T) {
	fc := &fakeClient{
		errs:      []error{&APIError{Kind: ErrAuth, StatusCode: 401, Provider: "fake", Message: "bad key"}},
		responses: []Response{{Text: "partial"}},
	}
	obs := &captureObserver{}
	client := Observe(fc, obs)

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
	if ce.Provider != "" || ce.Model != "" {
		t.Errorf("tags = %q/%q, want zero: a plain fake is not an IdentifiedClient", ce.Provider, ce.Model)
	}
}

// TestObserve_StreamEmitsExactlyOneEvent pins the native streaming path
// delivers deltas and closes with one Completion whose Response is the final one.
func TestObserve_StreamEmitsExactlyOneEvent(t *testing.T) {
	inner := withIdentity{
		Client: &scriptedStreamClient{deltas: []Delta{{Kind: DeltaText, Text: "he"}, {Kind: DeltaText, Text: "llo"}}, resp: okResponse()},
		id:     Identity{Provider: "openai", Model: "gpt-test"},
	}
	obs := &captureObserver{}
	client := Observe(inner, obs)

	var got []Delta
	resp, err := Stream(context.Background(), client, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("deltas = %+v, want the scripted fragments", got)
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
	if ce.Provider != "openai" || ce.Model != "gpt-test" {
		t.Errorf("tags = %q/%q, want openai/gpt-test", ce.Provider, ce.Model)
	}
}

// TestObserve_StreamError pins a failed stream still closes with exactly
// one Completion carrying the error text and zero Response.
func TestObserve_StreamError(t *testing.T) {
	inner := &scriptedStreamClient{err: &APIError{Kind: ErrServer, StatusCode: 500, Provider: "openai", Message: "boom"}}
	obs := &captureObserver{}
	client := Observe(inner, obs)

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

// TestObserve_StreamPartialResponseDiscriminated defends: when the inner
// client fails after assembling a partial Response (deltas already
// delivered), the caller still receives that partial Response unchanged,
// but the Completion event's Response is zeroed.
func TestObserve_StreamPartialResponseDiscriminated(t *testing.T) {
	inner := &scriptedStreamClient{
		resp: Response{Text: "partial"},
		err:  &APIError{Kind: ErrServer, StatusCode: 500, Provider: "openai", Message: "boom"},
	}
	obs := &captureObserver{}
	client := Observe(inner, obs)

	resp, err := Stream(context.Background(), client, simpleRequest(), func(Delta) error { return nil })
	if !errors.Is(err, ErrServer) {
		t.Fatalf("err = %v, want ErrServer", err)
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
		t.Errorf("Err = %q, want %q", ce.Err, err.Error())
	}
	if !reflect.DeepEqual(ce.Response, Response{}) {
		t.Errorf("event Response = %+v, want the zero Response on failure (not the partial the caller got)", ce.Response)
	}
}

// TestObserve_StreamAndCompleteDeepEqual pins the stream and complete paths
// observe the same event modulo per-call facts (Time, Duration, SpanID).
func TestObserve_StreamAndCompleteDeepEqual(t *testing.T) {
	resp := okResponse()
	runID := NewRunID()

	obsC := &captureObserver{}
	c := Observe(&fakeClient{responses: []Response{resp}}, obsC)
	if _, err := c.Complete(WithRun(context.Background(), runID), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	obsS := &captureObserver{}
	s := Observe(&fakeClient{responses: []Response{resp}}, obsS)
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

// TestObserve_PassthroughWhenAlreadyClaimed defends: a ctx whose span is
// already claimed (an enclosing emitter such as a Runner or an outer
// Observe) makes Observe pass the client straight through — no new span
// minted, no event emitted.
func TestObserve_PassthroughWhenAlreadyClaimed(t *testing.T) {
	t.Run("complete", func(t *testing.T) {
		fc := &fakeClient{responses: []Response{okResponse()}}
		obs := &captureObserver{}
		client := Observe(fc, obs)

		claimed := BeginCompletion(WithRun(context.Background(), NewRunID()))
		if _, err := client.Complete(claimed, simpleRequest()); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if got := len(obs.snapshot()); got != 0 {
			t.Fatalf("events = %d, want 0: an already-claimed ctx must pass through with no emission", got)
		}
		if fc.calls != 1 {
			t.Fatalf("inner calls = %d, want 1", fc.calls)
		}
	})
	t.Run("stream", func(t *testing.T) {
		inner := &scriptedStreamClient{resp: okResponse()}
		obs := &captureObserver{}
		client := Observe(inner, obs)

		claimed := BeginCompletion(WithRun(context.Background(), NewRunID()))
		if _, err := Stream(claimed, client, simpleRequest(), nil); err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if got := len(obs.snapshot()); got != 0 {
			t.Fatalf("events = %d, want 0: an already-claimed ctx must pass through with no emission", got)
		}
		if inner.calls != 1 {
			t.Fatalf("inner calls = %d, want 1", inner.calls)
		}
	})
}

// TestObserve_NestedObserveEmitsOnce defends: wrapping an already-Observed
// client in another Observe still emits exactly one Completion — the outer
// Observe's mint claims the ctx before the inner Observe ever sees it.
func TestObserve_NestedObserveEmitsOnce(t *testing.T) {
	fc := &fakeClient{responses: []Response{okResponse()}}
	obs := &captureObserver{}
	client := Observe(Observe(fc, obs), obs)
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := len(obs.snapshot()); got != 1 {
		t.Fatalf("nested Observe emitted %d Completion events, want exactly 1", got)
	}
}

// TestObserve_ObserverPanicPropagates pins a panicking observer surfaces to
// the caller, never swallowed.
func TestObserve_ObserverPanicPropagates(t *testing.T) {
	panicking := ObserverFunc(func(context.Context, Event) { panic("sink bug") })
	client := Observe(&fakeClient{responses: []Response{okResponse()}}, panicking)
	if !recovered(func() { _, _ = client.Complete(context.Background(), simpleRequest()) }) {
		t.Error("expected the observer panic to propagate out of Complete")
	}
	streaming := Observe(&scriptedStreamClient{resp: okResponse()}, panicking)
	if !recovered(func() { _, _ = Stream(context.Background(), streaming, simpleRequest(), nil) }) {
		t.Error("expected the observer panic to propagate out of Stream")
	}
}

// TestObserve_SynthesizedStreamFromPlainClient pins wrapping a Complete-only
// client still observes one Completion per Stream call.
func TestObserve_SynthesizedStreamFromPlainClient(t *testing.T) {
	resp := okResponse()
	inner := &fakeClient{responses: []Response{resp}}
	obs := &captureObserver{}
	client := Observe(inner, obs)

	var texts []string
	if _, err := Stream(context.Background(), client, simpleRequest(), func(d Delta) error {
		if d.Kind == DeltaText {
			texts = append(texts, d.Text)
		}
		return nil
	}); err != nil {
		t.Fatalf("Stream: %v", err)
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

// TestObserve_IdentityOfZeroCases defends: a plain fake (no Identity
// method) reports the zero Identity, and so does Observe wrapping one —
// Observe forwards, never invents.
func TestObserve_IdentityOfZeroCases(t *testing.T) {
	fc := &fakeClient{}
	if got := IdentityOf(fc); got != (Identity{}) {
		t.Fatalf("IdentityOf(plain fake) = %+v, want zero", got)
	}
	obs := ObserverFunc(func(context.Context, Event) {})
	if got := IdentityOf(Observe(fc, obs)); got != (Identity{}) {
		t.Fatalf("IdentityOf(Observe(plain fake, obs)) = %+v, want zero", got)
	}
}

// TestObserve_ForwardsIdentity pins that Observe's returned client
// implements IdentifiedClient by forwarding IdentityOf(c), and that every
// Completion event it emits carries the same tags.
func TestObserve_ForwardsIdentity(t *testing.T) {
	fc := withIdentity{Client: &fakeClient{responses: []Response{okResponse()}}, id: Identity{Provider: "fake", Model: "m-1"}}
	obs := &captureObserver{}
	client := Observe(fc, obs)

	if got := IdentityOf(client); got != (Identity{Provider: "fake", Model: "m-1"}) {
		t.Fatalf("IdentityOf(Observe(identified, obs)) = %+v, want the forwarded identity", got)
	}
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	ce := obs.snapshot()[0].Completion
	if ce.Provider != "fake" || ce.Model != "m-1" {
		t.Errorf("event tags = %q/%q, want fake/m-1", ce.Provider, ce.Model)
	}
}

// TestObserve_RequestAliasing defends: a sink that retains
// ev.Completion.Request sees the caller's later mutation of its own
// Messages slice — Observe never clones the Request.
func TestObserve_RequestAliasing(t *testing.T) {
	fc := &fakeClient{responses: []Response{okResponse()}}
	var retained *CompletionEvent
	obs := ObserverFunc(func(_ context.Context, ev Event) { retained = ev.Completion })
	client := Observe(fc, obs)

	messages := []Message{TextMessage(RoleUser, "original")}
	if _, err := client.Complete(context.Background(), Request{Messages: messages}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := retained.Request.Messages[0].Text(); got != "original" {
		t.Fatalf("retained event before mutation = %q, want original", got)
	}
	messages[0] = TextMessage(RoleUser, "mutated")
	if got := retained.Request.Messages[0].Text(); got != "mutated" {
		t.Fatalf("retained event did not alias the caller's slice: got %q, want mutated (aliasing is documented, not a bug)", got)
	}
}

// TestObserve_ConcurrentCompletionsGetDistinctSpans pins every concurrent
// completion mints its own span under -race, with no cross-talk.
func TestObserve_ConcurrentCompletionsGetDistinctSpans(t *testing.T) {
	obs := &captureObserver{}
	inner := &safeClient{resp: okResponse()}
	client := Observe(inner, obs)

	const runs = 64
	var wg sync.WaitGroup
	for range runs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
				t.Errorf("Complete: %v", err)
			}
		}()
	}
	wg.Wait()

	if inner.calls != runs {
		t.Fatalf("inner calls = %d, want %d", inner.calls, runs)
	}
	events := obs.snapshot()
	if len(events) != runs {
		t.Fatalf("events = %d, want %d", len(events), runs)
	}
	seen := make(map[SpanID]bool, runs)
	for _, ev := range events {
		if ev.SpanID == "" {
			t.Error("completion carries no SpanID")
		}
		if seen[ev.SpanID] {
			t.Errorf("span %q reused across concurrent completions", ev.SpanID)
		}
		seen[ev.SpanID] = true
	}
}

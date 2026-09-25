package provider

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// eventCapture collects llmkit events emitted by a Wrap/New stack.
type eventCapture struct {
	mu     sync.Mutex
	events []llmkit.Event
}

func (c *eventCapture) Observe(_ context.Context, ev llmkit.Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
}

func (c *eventCapture) snapshot() []llmkit.Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]llmkit.Event(nil), c.events...)
}

func (c *eventCapture) byKind(k llmkit.EventKind) []llmkit.Event {
	var out []llmkit.Event
	for _, ev := range c.snapshot() {
		if ev.Kind == k {
			out = append(out, ev)
		}
	}
	return out
}

// fastRetry keeps retry-driven provider tests hermetic: no real backoff.
func fastRetry() retry.Config {
	return retry.Config{
		MaxAttempts:    3,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: time.Second,
		Sleep:          func(context.Context, time.Duration) error { return nil },
	}
}

// observeSpec builds an openai-compatible spec aimed at the test server. The
// default profile of this type reports ParallelToolCalls=false, so the full
// production stack — serializer included — is installed.
func observeSpec(baseURL string) Spec {
	return Spec{
		Type:    TypeOpenAICompatible,
		Model:   "llama-test",
		BaseURL: baseURL,
		Secret:  "test-key",
	}
}

// erroringClient fails Complete with err (Stream inherits the same).
type erroringClient struct{ err error }

func (e *erroringClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }
func (e *erroringClient) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	return llmkit.Response{}, e.err
}

// TestNew_ObservesOneCompletionAndItsAttempts asserts that a bare New
// client over httptest with Options.Observer emits exactly one
// Completion per Complete call, with a non-empty SpanID, and every
// Attempt shares that span.
func TestNew_ObservesOneCompletionAndItsAttempts(t *testing.T) {
	hits := 0
	var mu sync.Mutex
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		i := hits
		hits++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if i == 0 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(errorBody("openai-compatible", 429, "slow down")))
			return
		}
		_, _ = w.Write([]byte(mockTextBody("openai-compatible", "hello world", 12, 5)))
	})

	obs := &eventCapture{}
	client, err := New(context.Background(), observeSpec(base), Options{
		Retry:    fastRetry(),
		Observer: obs,
		Provider: "compat-lane",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	runID := llmkit.NewRunID()
	resp, err := client.Complete(llmkit.WithRun(context.Background(), runID), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "hello world" {
		t.Errorf("Text = %q, want hello world", resp.Text)
	}
	if hits != 2 {
		t.Fatalf("wire calls = %d, want 2", hits)
	}

	completions := obs.byKind(llmkit.KindCompletion)
	attempts := obs.byKind(llmkit.KindAttempt)
	if len(completions) != 1 {
		t.Fatalf("completions = %d, want exactly 1 (Q1: New's own emitter)", len(completions))
	}
	if len(attempts) != 2 {
		t.Fatalf("attempts = %d, want 2", len(attempts))
	}
	ce := completions[0]
	if ce.SpanID == "" {
		t.Fatal("completion SpanID empty")
	}
	if ce.RunID != runID {
		t.Errorf("completion RunID = %q, want %q", ce.RunID, runID)
	}
	if ce.Completion.Provider != "compat-lane" || ce.Completion.Model != "llama-test" {
		t.Errorf("completion tags = %q/%q, want compat-lane/llama-test", ce.Completion.Provider, ce.Completion.Model)
	}
	if ce.Completion.Response.Text != "hello world" {
		t.Errorf("completion Response.Text = %q, want hello world", ce.Completion.Response.Text)
	}
	for i, ev := range attempts {
		ae := ev.Attempt
		if ae == nil {
			t.Fatalf("attempt %d has no payload", i)
		}
		if ev.SpanID != ce.SpanID {
			t.Errorf("attempt %d SpanID = %q, want the completion's %q", i, ev.SpanID, ce.SpanID)
		}
		if ae.Attempt != i+1 {
			t.Errorf("attempt %d numbered %d, want %d", i, ae.Attempt, i+1)
		}
		if ae.Provider != "compat-lane" || ae.Model != "llama-test" {
			t.Errorf("attempt %d tags = %q/%q, want compat-lane/llama-test", i, ae.Provider, ae.Model)
		}
		if i == 0 {
			if ae.Err == "" || ae.StatusCode != 429 {
				t.Errorf("failed attempt %d = %q (status %d), want the 429 failure", i, ae.Err, ae.StatusCode)
			}
		} else if ae.Err != "" || ae.Response.Text != "hello world" {
			t.Errorf("successful attempt %d = %q/%+v, want clean hello world", i, ae.Err, ae.Response)
		}
	}
}

// TestNew_StreamObservesOneCompletionAndItsAttempts asserts the Stream
// counterpart of C3(a): a 503 attempt followed by a successful SSE
// attempt yields one Completion and two Attempt events, all sharing one
// span, with deltas delivered once.
func TestNew_StreamObservesOneCompletionAndItsAttempts(t *testing.T) {
	hits := 0
	var mu sync.Mutex
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		i := hits
		hits++
		mu.Unlock()
		if i == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(errorBody("openai-compatible", 503, "overloaded")))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseHello))
	})

	obs := &eventCapture{}
	client, err := New(context.Background(), observeSpec(base), Options{
		Retry:    fastRetry(),
		Observer: obs,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var texts []string
	resp, err := llmkit.Stream(context.Background(), client, simpleRequest(), func(d llmkit.Delta) error {
		if d.Kind == llmkit.DeltaText {
			texts = append(texts, d.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if hits != 2 {
		t.Fatalf("wire calls = %d, want 2", hits)
	}
	if strings.Join(texts, "") != "hello" {
		t.Errorf("deltas = %q, want hello delivered once from the successful attempt", strings.Join(texts, ""))
	}
	if resp.Text != "hello" || resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 2 {
		t.Errorf("final response = %+v, want the assembled stream response", resp)
	}

	completions := obs.byKind(llmkit.KindCompletion)
	attempts := obs.byKind(llmkit.KindAttempt)
	if len(completions) != 1 || len(attempts) != 2 {
		t.Fatalf("completions=%d attempts=%d, want 1 and 2", len(completions), len(attempts))
	}
	span := completions[0].SpanID
	if span == "" {
		t.Fatal("completion SpanID empty")
	}
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
		if i == 0 {
			if ae.Err == "" || ae.StatusCode != 503 {
				t.Errorf("failed attempt = %q (status %d), want the 503 failure", ae.Err, ae.StatusCode)
			}
			if !reflect.DeepEqual(ae.Response, llmkit.Response{}) {
				t.Errorf("failed attempt Response = %+v, want zero", ae.Response)
			}
		} else if ae.Err != "" || ae.Response.Text != "hello" {
			t.Errorf("successful attempt = %q/%+v, want clean hello", ae.Err, ae.Response)
		}
	}
	if completions[0].Completion.Err != "" || completions[0].Completion.Response.Text != "hello" {
		t.Errorf("completion = %q/%+v, want clean hello", completions[0].Completion.Err, completions[0].Completion.Response)
	}
}

// TestNew_TruncationVisibleInCompletion asserts the serializer sits
// inside the completion emitter: the Completion event shows the
// caller-visible truncated response while the retry stage's Attempt
// event still shows the raw adapter response, both joined on one span.
func TestNew_TruncationVisibleInCompletion(t *testing.T) {
	twoToolCalls := `{"id":"chatcmpl-3","object":"chat.completion","created":1,"model":"llama-test",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[` +
		`{"id":"call_a","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"utc\"}"}},` +
		`{"id":"call_b","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"oslo\"}"}}` +
		`]},"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":9,"completion_tokens":7,"total_tokens":16}}`
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(twoToolCalls))
	})

	obs := &eventCapture{}
	client, err := New(context.Background(), observeSpec(base), Options{
		Retry:    retry.Config{MaxAttempts: 1, RequestTimeout: time.Second},
		Observer: obs,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	completions := obs.byKind(llmkit.KindCompletion)
	attempts := obs.byKind(llmkit.KindAttempt)
	if len(completions) != 1 || len(attempts) != 1 {
		t.Fatalf("completions=%d attempts=%d, want 1 and 1", len(completions), len(attempts))
	}
	seen := completions[0].Completion.Response
	raw := attempts[0].Attempt.Response
	if len(seen.ToolCalls) != 1 {
		t.Errorf("Completion.Response carries %d tool calls, want the truncated 1 the caller sees", len(seen.ToolCalls))
	} else if seen.ToolCalls[0].ID != "call_a" {
		t.Errorf("kept call = %q, want call_a", seen.ToolCalls[0].ID)
	}
	if len(raw.ToolCalls) != 2 {
		t.Errorf("Attempt.Response carries %d tool calls, want the raw 2 from the wire", len(raw.ToolCalls))
	}
	if completions[0].SpanID == "" || completions[0].SpanID != attempts[0].SpanID {
		t.Errorf("span join broken: completion %q attempt %q, want one non-empty shared span",
			completions[0].SpanID, attempts[0].SpanID)
	}
}

// TestNew_FailedFinalAttemptMatchesCompletion asserts that on a
// non-retryable failure the retry stage's failed-attempt event and the
// completion emitter's failed-completion event agree: exactly one of
// each, both carrying the same error text and the zero Response, joined
// on one span.
func TestNew_FailedFinalAttemptMatchesCompletion(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(errorBody("openai-compatible", http.StatusUnauthorized, "denied")))
	})
	obs := &eventCapture{}
	client, err := New(context.Background(), observeSpec(base), Options{
		Retry:    retry.Config{MaxAttempts: 4, RequestTimeout: time.Second, Sleep: func(context.Context, time.Duration) error { return nil }},
		Observer: obs,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, completeErr := client.Complete(context.Background(), simpleRequest())
	if !errors.Is(completeErr, llmkit.ErrAuth) {
		t.Fatalf("Complete err = %v, want ErrAuth", completeErr)
	}
	attempts := obs.byKind(llmkit.KindAttempt)
	completions := obs.byKind(llmkit.KindCompletion)
	if len(attempts) != 1 || len(completions) != 1 {
		t.Fatalf("attempts=%d completions=%d, want 1 and 1 (auth is not retryable)", len(attempts), len(completions))
	}
	ae, ce := attempts[0].Attempt, completions[0].Completion
	if ae.Err != completeErr.Error() {
		t.Errorf("attempt Err = %q, want %q", ae.Err, completeErr.Error())
	}
	if ae.StatusCode != http.StatusUnauthorized {
		t.Errorf("attempt StatusCode = %d, want 401", ae.StatusCode)
	}
	if ce.Err != completeErr.Error() {
		t.Errorf("completion Err = %q, want %q", ce.Err, completeErr.Error())
	}
	if !reflect.DeepEqual(ae.Response, llmkit.Response{}) || !reflect.DeepEqual(ce.Response, llmkit.Response{}) {
		t.Error("failed attempt/completion must both carry the zero Response")
	}
	if attempts[0].SpanID == "" || attempts[0].SpanID != completions[0].SpanID {
		t.Errorf("span join broken: attempt %q completion %q, want one non-empty shared span",
			attempts[0].SpanID, completions[0].SpanID)
	}
}

// sseHello is a minimal successful Chat Completions stream: one text
// delta, the finish chunk, the include_usage chunk, and [DONE].
const sseHello = "data: " +
	`{"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"llama-test",` +
	`"system_fingerprint":"fp_1","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},` +
	`"finish_reason":null}],"usage":null}` + "\n\n" + "data: " +
	`{"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"llama-test",` +
	`"system_fingerprint":"fp_1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
	`"usage":null}` + "\n\n" + "data: " +
	`{"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"llama-test",` +
	`"system_fingerprint":"fp_1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,` +
	`"total_tokens":7}}` + "\n\ndata: [DONE]\n\n"

// TestNew_Identity asserts IdentityOf(New(...)) resolves Options.Provider
// when set, else string(spec.Type), and that every Completion and
// Attempt event the client emits carries those same tags — identity must
// survive every stage it passes through.
func TestNew_Identity(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("openai-compatible", "hi", 1, 1)))
	})
	spec := Spec{Type: TypeOpenAICompatible, Model: "m", BaseURL: base, Secret: "k"}

	t.Run("Options.Provider overrides", func(t *testing.T) {
		client, err := New(context.Background(), spec, Options{Provider: "cfg"})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := llmkit.IdentityOf(client); got != (llmkit.Identity{Provider: "cfg", Model: "m"}) {
			t.Fatalf("IdentityOf = %+v, want {cfg m}", got)
		}
	})
	t.Run("default is spec.Type", func(t *testing.T) {
		client, err := New(context.Background(), spec, Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := llmkit.IdentityOf(client); got != (llmkit.Identity{Provider: "openai-compatible", Model: "m"}) {
			t.Fatalf("IdentityOf = %+v, want {openai-compatible m}", got)
		}
	})

	obs := &eventCapture{}
	client, err := New(context.Background(), spec, Options{Provider: "cfg", Observer: obs, Retry: fastRetry()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	events := obs.snapshot()
	if len(events) == 0 {
		t.Fatal("no events observed")
	}
	for _, ev := range events {
		switch ev.Kind {
		case llmkit.KindAttempt:
			if ev.Attempt.Provider != "cfg" || ev.Attempt.Model != "m" {
				t.Errorf("attempt tags = %q/%q, want cfg/m", ev.Attempt.Provider, ev.Attempt.Model)
			}
		case llmkit.KindCompletion:
			if ev.Completion.Provider != "cfg" || ev.Completion.Model != "m" {
				t.Errorf("completion tags = %q/%q, want cfg/m", ev.Completion.Provider, ev.Completion.Model)
			}
		}
	}
}

// TestWrap_IdentityFromForeignClient asserts Wrap's identity rule over a
// client with no Identity of its own: with no Options.Provider the
// Identity is zero; with Options.Provider set, only Provider is tagged
// (Model stays empty because only an IdentifiedClient can supply it).
func TestWrap_IdentityFromForeignClient(t *testing.T) {
	fc := &erroringClient{err: errors.New("unused")}
	if got := llmkit.IdentityOf(Wrap(fc, Options{})); got != (llmkit.Identity{}) {
		t.Fatalf("IdentityOf(Wrap(foreign, no Provider)) = %+v, want zero", got)
	}
	if got := llmkit.IdentityOf(Wrap(fc, Options{Provider: "cfg"})); got != (llmkit.Identity{Provider: "cfg"}) {
		t.Fatalf("IdentityOf(Wrap(foreign, Provider set)) = %+v, want {cfg }", got)
	}
}

// TestNew_RetryAfterPresence asserts the Retry-After header is parsed
// exactly: an absent header is (0,false); "0" is (0,true) — present-zero
// means retry immediately; "2" is (2s,true).
func TestNew_RetryAfterPresence(t *testing.T) {
	tests := []struct {
		name      string
		header    string // "" means the header is omitted
		wantAfter time.Duration
		wantHas   bool
	}{
		{"absent", "", 0, false},
		{"present zero", "0", 0, true},
		{"present two", "2", 2 * time.Second, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if tc.header != "" {
					w.Header().Set("Retry-After", tc.header)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(errorBody("openai-compatible", 429, "slow down")))
			})
			obs := &eventCapture{}
			client, err := New(context.Background(), observeSpec(base), Options{
				Retry:    retry.Config{MaxAttempts: 1, RequestTimeout: time.Second},
				Observer: obs,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := client.Complete(context.Background(), simpleRequest()); err == nil {
				t.Fatal("Complete: want an error from the 429")
			}
			attempts := obs.byKind(llmkit.KindAttempt)
			if len(attempts) != 1 {
				t.Fatalf("attempts = %d, want 1", len(attempts))
			}
			ae := attempts[0].Attempt
			if ae.RetryAfter != tc.wantAfter || ae.HasRetryAfter != tc.wantHas {
				t.Errorf("RetryAfter/HasRetryAfter = %v/%v, want %v/%v", ae.RetryAfter, ae.HasRetryAfter, tc.wantAfter, tc.wantHas)
			}
		})
	}
}

// TestWrap_RetryAfterCopiedOnlyWhenPresent asserts over a fake returning
// APIError{ErrRateLimited, RetryAfter 5s, HasRetryAfter false} that the
// emitted Attempt reports (0,false): RetryAfter is copied only when the
// presence bit is set, never on a nonzero value alone.
func TestWrap_RetryAfterCopiedOnlyWhenPresent(t *testing.T) {
	fc := &erroringClient{err: &llmkit.APIError{Kind: llmkit.ErrRateLimited, RetryAfter: 5 * time.Second, HasRetryAfter: false}}
	obs := &eventCapture{}
	client := Wrap(fc, Options{
		Retry:    retry.Config{MaxAttempts: 1, RequestTimeout: time.Second, Sleep: func(context.Context, time.Duration) error { return nil }},
		Observer: obs,
	})
	if _, err := client.Complete(context.Background(), simpleRequest()); err == nil {
		t.Fatal("Complete: want an error")
	}
	attempts := obs.byKind(llmkit.KindAttempt)
	if len(attempts) != 1 {
		t.Fatalf("attempts = %d, want 1", len(attempts))
	}
	ae := attempts[0].Attempt
	if ae.RetryAfter != 0 || ae.HasRetryAfter {
		t.Errorf("RetryAfter/HasRetryAfter = %v/%v, want 0/false: a nonzero RetryAfter without HasRetryAfter must not copy", ae.RetryAfter, ae.HasRetryAfter)
	}
}

// TestNew_CompletesPartialRetryConfigFieldWise asserts that a partial
// Options.Retry that sets only MaxAttempts and Sleep keeps both and
// takes retry.Default's BaseDelay/Jitter field-wise — the hook observes
// the ~500ms jittered backoff, never a zero-delay hot loop.
func TestNew_CompletesPartialRetryConfigFieldWise(t *testing.T) {
	hits := 0
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(errorBody("anthropic", http.StatusServiceUnavailable, "overloaded")))
	})

	var sleeps []time.Duration
	hook := func(_ context.Context, d time.Duration) error {
		sleeps = append(sleeps, d)
		return nil
	}
	client, err := New(t.Context(), Spec{
		Type:    TypeAnthropic,
		Model:   "claude-test",
		BaseURL: base,
		Secret:  "test-key",
	}, Options{Retry: retry.Config{MaxAttempts: 2, Sleep: hook}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Complete(context.Background(), simpleRequest())
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("err = %v, want *llmkit.APIError ErrServer from the final 503", err)
	}
	if hits != 2 {
		t.Errorf("wire hits = %d, want 2 (MaxAttempts kept from the partial config)", hits)
	}
	if len(sleeps) != 1 {
		t.Fatalf("sleeps = %v, want exactly one backoff sleep", sleeps)
	}
	// Default BaseDelay 500ms with Default Jitter 0.2: [400ms, 600ms]. A
	// whole-config swap (or no fill at all) sleeps 0.
	if sleeps[0] < 400*time.Millisecond || sleeps[0] > 600*time.Millisecond {
		t.Errorf("slept %v, want the completed 500ms±20%% jittered default — not 0", sleeps[0])
	}
}

// deliveryCapture records the delivery ctx alongside each event, so a
// probe can detect a per-attempt deadline leaking into the sink.
type deliveryCapture struct {
	eventCapture
	mu   sync.Mutex
	ctxs []context.Context
}

func (c *deliveryCapture) Observe(ctx context.Context, ev llmkit.Event) {
	c.eventCapture.Observe(ctx, ev)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ctxs = append(c.ctxs, ctx)
}

func (c *deliveryCapture) deliveryCtxs() []context.Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]context.Context(nil), c.ctxs...)
}

// TestWrap_AttemptEventsUseRetryCallerContext asserts attempt events
// are delivered on the retry caller's context, never the per-attempt
// timeout context.
func TestWrap_AttemptEventsUseRetryCallerContext(t *testing.T) {
	stalls := &stallClient{}
	obs := &deliveryCapture{}
	cfg := retry.Config{
		MaxAttempts:    2,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: 50 * time.Millisecond,
		Sleep:          func(context.Context, time.Duration) error { return nil },
	}
	client := Wrap(stalls, Options{Retry: cfg, Observer: obs})

	// Plain Background: any deadline seen would have to come from the
	// retry stage's per-attempt context, which must not leak.
	_, err := client.Complete(context.Background(), simpleRequest())
	if err == nil {
		t.Fatal("Complete: want the final timeout error")
	}
	if stalls.calls != 2 {
		t.Fatalf("wire calls = %d, want 2", stalls.calls)
	}
	events := obs.snapshot()
	deliveryCtxs := obs.deliveryCtxs()
	if len(events) != len(deliveryCtxs) {
		t.Fatalf("events = %d ctxs = %d, want equal counts", len(events), len(deliveryCtxs))
	}
	var attemptIdx []int
	for i, ev := range events {
		if ev.Kind == llmkit.KindAttempt {
			attemptIdx = append(attemptIdx, i)
		}
	}
	if len(attemptIdx) != 2 {
		t.Fatalf("attempt events = %d, want 2 (both timed-out attempts delivered)", len(attemptIdx))
	}
	for _, i := range attemptIdx {
		ctx := deliveryCtxs[i]
		if _, ok := ctx.Deadline(); ok {
			t.Errorf("attempt at index %d delivered on a context with a deadline", i)
		}
		if ctx.Err() != nil {
			t.Errorf("attempt at index %d delivered with ctx.Err() = %v, want nil", i, ctx.Err())
		}
		if events[i].Attempt == nil || events[i].Attempt.Err == "" {
			t.Errorf("attempt at index %d carries no error; want the timeout failure", i)
		}
	}
}

// stallClient blocks every Complete until its context is done, so the
// per-attempt RequestTimeout must abort it.
type stallClient struct {
	calls int
}

func (s *stallClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (s *stallClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	s.calls++
	<-ctx.Done()
	return llmkit.Response{}, &llmkit.APIError{
		Kind: llmkit.ErrServer, Provider: "fake", Message: ctx.Err().Error(), Err: ctx.Err(),
	}
}

// TestWrap_AttemptCounterIsPerCall asserts the 1..N attempt numbering
// restarts on every logical completion through the same client.
func TestWrap_AttemptCounterIsPerCall(t *testing.T) {
	fc := &fakeClient{errs: []error{
		rateLimitErr(0), rateLimitErr(0), rateLimitErr(0), rateLimitErr(0),
	}}
	obs := &eventCapture{}
	client := Wrap(fc, Options{Retry: retry.Config{
		MaxAttempts: 2, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond,
		Sleep: func(context.Context, time.Duration) error { return nil },
	}, Observer: obs})
	for round := 0; round < 2; round++ {
		if _, err := client.Complete(context.Background(), simpleRequest()); !errors.Is(err, llmkit.ErrRateLimited) {
			t.Fatalf("Complete %d: err = %v, want ErrRateLimited", round, err)
		}
	}
	var got []int
	for _, ev := range obs.byKind(llmkit.KindAttempt) {
		got = append(got, ev.Attempt.Attempt)
	}
	if !reflect.DeepEqual(got, []int{1, 2, 1, 2}) {
		t.Fatalf("attempt numbers across two completions = %v, want [1 2 1 2]", got)
	}
}

// safeClient is a concurrency-safe scripted Client — the concurrency
// test shares one across goroutines, and fakeClient's counter would race.
type safeClient struct {
	mu    sync.Mutex
	calls int
	resp  llmkit.Response
}

func (s *safeClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (s *safeClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	return s.resp, nil
}

// TestWrap_ConcurrentSpanJoin asserts under 64 concurrent completions
// (runs under -race) that every Attempt joins exactly one Completion
// and no span is shared across completions.
func TestWrap_ConcurrentSpanJoin(t *testing.T) {
	obs := &eventCapture{}
	inner := &safeClient{resp: llmkit.Response{Text: "hello", StopReason: llmkit.StopEndTurn}}
	client := Wrap(inner, Options{Retry: retry.Config{
		MaxAttempts: 2, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond,
		Sleep: func(context.Context, time.Duration) error { return nil },
	}, Observer: obs})

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
		t.Fatalf("inner wire calls = %d, want %d", inner.calls, runs)
	}
	completions := obs.byKind(llmkit.KindCompletion)
	attempts := obs.byKind(llmkit.KindAttempt)
	if len(completions) != runs || len(attempts) != runs {
		t.Fatalf("completions=%d attempts=%d, want %d each", len(completions), len(attempts), runs)
	}
	attemptsBySpan := map[llmkit.SpanID]int{}
	for _, ev := range attempts {
		attemptsBySpan[ev.SpanID]++
		if ev.Attempt.Attempt != 1 {
			t.Errorf("attempt %d on span %q: failure-free completions have exactly one attempt numbered 1",
				ev.Attempt.Attempt, ev.SpanID)
		}
	}
	for _, ev := range completions {
		if attemptsBySpan[ev.SpanID] != 1 {
			t.Errorf("completion span %q joined by %d attempts, want exactly 1", ev.SpanID, attemptsBySpan[ev.SpanID])
		}
	}
	if len(attemptsBySpan) != runs {
		t.Errorf("%d distinct spans across %d completions; spans must be minted per completion",
			len(attemptsBySpan), runs)
	}
}

// TestWrap_NilObserverEmitsNothing asserts the retry stage stays silent
// when a stack is wrapped only by an outer llmkit.Observe with no
// Options.Observer of its own.
func TestWrap_NilObserverEmitsNothing(t *testing.T) {
	fc := &fakeClient{responses: []llmkit.Response{{Text: "ok", StopReason: llmkit.StopEndTurn}}}
	obs := &eventCapture{}
	client := llmkit.Observe(Wrap(fc, Options{Retry: fastRetry()}), obs)
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for _, ev := range obs.snapshot() {
		if ev.Kind == llmkit.KindAttempt {
			t.Error("the unobserved retry stage emitted an Attempt event; it must stay silent")
		}
	}
	if len(obs.snapshot()) != 1 {
		t.Errorf("events = %d, want only the Completion", len(obs.snapshot()))
	}
}

// noSleep replaces retry.Config.Sleep in tests that count hits, not time.
func noSleep(context.Context, time.Duration) error { return nil }

// callOnce runs one Complete or Stream through client and returns its
// error.
func callOnce(ctx context.Context, client llmkit.Client, method string) error {
	if method == "Stream" {
		_, err := llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error { return nil })
		return err
	}
	_, err := client.Complete(ctx, simpleRequest())
	return err
}

// wireClient is a foreign Client making one HTTP POST per Complete to
// url, mapping a 429 to a retryable *APIError — so Wrap over it counts
// wire hits at an httptest server.
type wireClient struct{ url string }

func (w *wireClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (w *wireClient) Complete(ctx context.Context, _ llmkit.Request) (llmkit.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.url, nil)
	if err != nil {
		return llmkit.Response{}, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return llmkit.Response{}, err
	}
	_ = resp.Body.Close()
	if resp.StatusCode == http.StatusTooManyRequests {
		return llmkit.Response{}, &llmkit.APIError{Kind: llmkit.ErrRateLimited, StatusCode: 429, Provider: "wire", Message: "429"}
	}
	return llmkit.Response{Text: "ok", StopReason: llmkit.StopEndTurn}, nil
}

// TestWrap_IdempotentOverStack asserts that Wrap over a client New or
// Wrap returned rebuilds ONE stack from the client below its stages
// with the new Options: against a persistent 429 one call makes exactly
// MaxAttempts wire hits, numbered 1..MaxAttempts once each on one span
// with one Completion; the inner stack's Options — its Observer
// included — no longer apply; its Identity is kept.
func TestWrap_IdempotentOverStack(t *testing.T) {
	var hits atomic.Int64
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(errorBody("openai-compatible", 429, "slow down")))
	})
	four := retry.Config{
		MaxAttempts: 4, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond,
		RequestTimeout: time.Second, Sleep: noSleep,
	}
	newStack := func(t *testing.T, obs llmkit.Observer) llmkit.Client {
		t.Helper()
		c, err := New(context.Background(), observeSpec(base), Options{Retry: four, Observer: obs, Provider: "cfg"})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return c
	}
	rows := []struct {
		name  string
		build func(t *testing.T, innerObs, obs llmkit.Observer) llmkit.Client
	}{
		{"New then Wrap", func(t *testing.T, innerObs, obs llmkit.Observer) llmkit.Client {
			return Wrap(newStack(t, innerObs), Options{Retry: four, Observer: obs})
		}},
		{"Wrap then Wrap", func(t *testing.T, innerObs, obs llmkit.Observer) llmkit.Client {
			inner := Wrap(&wireClient{url: base}, Options{Retry: four, Observer: innerObs, Provider: "cfg"})
			return Wrap(inner, Options{Retry: four, Observer: obs})
		}},
		{"New then Wrap twice", func(t *testing.T, innerObs, obs llmkit.Observer) llmkit.Client {
			mid := Wrap(newStack(t, innerObs), Options{Retry: four, Observer: innerObs})
			return Wrap(mid, Options{Retry: four, Observer: obs})
		}},
	}
	for _, row := range rows {
		for _, method := range []string{"Complete", "Stream"} {
			t.Run(row.name+"/"+method, func(t *testing.T) {
				innerObs, obs := &eventCapture{}, &eventCapture{}
				client := row.build(t, innerObs, obs)
				hits.Store(0)
				if err := callOnce(context.Background(), client, method); !errors.Is(err, llmkit.ErrRateLimited) {
					t.Fatalf("%s: err = %v, want ErrRateLimited", method, err)
				}
				if got := hits.Load(); got != 4 {
					t.Fatalf("wire hits = %d, want 4 (one retry loop of MaxAttempts 4)", got)
				}
				if n := len(innerObs.snapshot()); n != 0 {
					t.Errorf("inner stack's Observer got %d events, want 0: the rebuilt stack takes the new Options", n)
				}
				if got := llmkit.IdentityOf(client).Provider; got != "cfg" {
					t.Errorf("IdentityOf(rebuilt).Provider = %q, want the inner stack's cfg", got)
				}
				completions := obs.byKind(llmkit.KindCompletion)
				if len(completions) != 1 {
					t.Fatalf("completions = %d, want 1", len(completions))
				}
				span := completions[0].SpanID
				if span == "" || completions[0].Completion.Err == "" {
					t.Errorf("completion span %q err %q, want a span and the 429 failure", span, completions[0].Completion.Err)
				}
				var numbers []int
				for _, ev := range obs.byKind(llmkit.KindAttempt) {
					numbers = append(numbers, ev.Attempt.Attempt)
					if ev.SpanID != span {
						t.Errorf("attempt %d span = %q, want the completion's %q", ev.Attempt.Attempt, ev.SpanID, span)
					}
					if ev.Attempt.Provider != "cfg" {
						t.Errorf("attempt %d Provider = %q, want cfg", ev.Attempt.Attempt, ev.Attempt.Provider)
					}
				}
				if !reflect.DeepEqual(numbers, []int{1, 2, 3, 4}) {
					t.Errorf("attempt numbers = %v, want [1 2 3 4]", numbers)
				}
			})
		}
	}
}

// TestWrap_RebuildKeepsIdentityUnlessProviderSet asserts the Identity
// half of stack rebuild: Wrap over a New stack keeps that stack's
// Identity when opts.Provider is empty and replaces only Provider when
// it is set.
func TestWrap_RebuildKeepsIdentityUnlessProviderSet(t *testing.T) {
	inner, err := New(context.Background(), observeSpec("http://127.0.0.1:1"), Options{Provider: "cfg"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got, want := llmkit.IdentityOf(Wrap(inner, Options{})), (llmkit.Identity{Provider: "cfg", Model: "llama-test"}); got != want {
		t.Errorf("IdentityOf(Wrap(stack, no Provider)) = %+v, want %+v", got, want)
	}
	if got, want := llmkit.IdentityOf(Wrap(inner, Options{Provider: "new"})), (llmkit.Identity{Provider: "new", Model: "llama-test"}); got != want {
		t.Errorf("IdentityOf(Wrap(stack, Provider new)) = %+v, want %+v", got, want)
	}
}

// TestWrap_ClaimsAboveRetryWithoutObserver asserts a Wrap stack claims
// the completion above its retry stage whether or not Options.Observer
// is set: an llmkit.Observe nested inside it is silenced on every
// attempt (0 Completions), and the same observer passed as
// Options.Observer sees exactly one Completion joined by the three
// Attempts.
func TestWrap_ClaimsAboveRetryWithoutObserver(t *testing.T) {
	ok := llmkit.Response{Text: "ok", StopReason: llmkit.StopEndTurn}
	script := func() *fakeClient {
		return &fakeClient{errs: []error{rateLimitErr(0), rateLimitErr(0)}, responses: []llmkit.Response{{}, {}, ok}}
	}
	for _, method := range []string{"Complete", "Stream"} {
		t.Run(method+"/no Options.Observer", func(t *testing.T) {
			fc, obs := script(), &eventCapture{}
			client := Wrap(llmkit.Observe(fc, obs), Options{Retry: fastRetry()})
			if err := callOnce(context.Background(), client, method); err != nil {
				t.Fatalf("%s: %v", method, err)
			}
			if fc.calls != 3 {
				t.Fatalf("wire calls = %d, want 3", fc.calls)
			}
			if n := len(obs.byKind(llmkit.KindCompletion)); n != 0 {
				t.Errorf("nested Observe emitted %d Completions, want 0: the stack claims above its retry stage", n)
			}
		})
		t.Run(method+"/same obs as Options.Observer", func(t *testing.T) {
			fc, obs := script(), &eventCapture{}
			client := Wrap(llmkit.Observe(fc, obs), Options{Retry: fastRetry(), Observer: obs})
			if err := callOnce(context.Background(), client, method); err != nil {
				t.Fatalf("%s: %v", method, err)
			}
			completions := obs.byKind(llmkit.KindCompletion)
			if len(completions) != 1 {
				t.Fatalf("completions = %d, want exactly 1", len(completions))
			}
			attempts := obs.byKind(llmkit.KindAttempt)
			if len(attempts) != 3 {
				t.Fatalf("attempts = %d, want 3", len(attempts))
			}
			for i, ev := range attempts {
				if ev.SpanID != completions[0].SpanID {
					t.Errorf("attempt %d span = %q, want the completion's %q", i, ev.SpanID, completions[0].SpanID)
				}
			}
		})
	}
}

// spanProbeClient records the SpanID of the ctx each attempt receives
// and fails its leading attempts with a retryable 429.
type spanProbeClient struct {
	fail  int
	spans []llmkit.SpanID
}

func (s *spanProbeClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (s *spanProbeClient) Complete(ctx context.Context, _ llmkit.Request) (llmkit.Response, error) {
	s.spans = append(s.spans, llmkit.SpanFromContext(ctx))
	if len(s.spans) <= s.fail {
		return llmkit.Response{}, rateLimitErr(0)
	}
	return llmkit.Response{Text: "ok", StopReason: llmkit.StopEndTurn}, nil
}

// TestWrap_AttemptsCarrySpanWithoutObserver asserts a stack with no
// Options.Observer still hands every attempt of one call the same
// non-empty span, and a ctx that already carries a span keeps it.
func TestWrap_AttemptsCarrySpanWithoutObserver(t *testing.T) {
	t.Run("unclaimed ctx", func(t *testing.T) {
		probe := &spanProbeClient{fail: 1}
		if _, err := Wrap(probe, Options{Retry: fastRetry()}).Complete(context.Background(), simpleRequest()); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if len(probe.spans) != 2 || probe.spans[0] == "" || probe.spans[0] != probe.spans[1] {
			t.Errorf("attempt spans = %q, want two equal non-empty spans", probe.spans)
		}
	})
	t.Run("claimed ctx", func(t *testing.T) {
		probe := &spanProbeClient{fail: 1}
		ctx := llmkit.BeginCompletion(context.Background())
		want := llmkit.SpanFromContext(ctx)
		if _, err := Wrap(probe, Options{Retry: fastRetry()}).Complete(ctx, simpleRequest()); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if !reflect.DeepEqual(probe.spans, []llmkit.SpanID{want, want}) {
			t.Errorf("attempt spans = %q, want the enclosing claim %q on both", probe.spans, want)
		}
	})
}

// TestWrap_StreamAttemptsExhausted asserts the two Stream-path retry
// properties through Wrap: every Attempt event is delivered on the
// retry caller's context (no per-attempt deadline leaks into the sink),
// and the one Completion's Duration covers every attempt plus the
// backoff between them.
func TestWrap_StreamAttemptsExhausted(t *testing.T) {
	inner := &scriptedStreamClient{err: rateLimitErr(0)}
	obs := &deliveryCapture{}
	var slept []time.Duration
	cfg := retry.Config{
		MaxAttempts:    3,
		BaseDelay:      5 * time.Millisecond,
		MaxDelay:       10 * time.Millisecond,
		RequestTimeout: time.Second,
		Sleep: func(_ context.Context, d time.Duration) error {
			slept = append(slept, d)
			time.Sleep(d)
			return nil
		},
	}
	client := Wrap(inner, Options{Retry: cfg, Observer: obs})

	runID := llmkit.NewRunID()
	_, err := llmkit.Stream(llmkit.WithRun(context.Background(), runID), client, simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, llmkit.ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
	if inner.calls != 3 {
		t.Fatalf("wire stream calls = %d, want 3", inner.calls)
	}
	events, ctxs := obs.snapshot(), obs.deliveryCtxs()
	if len(events) != len(ctxs) {
		t.Fatalf("events = %d ctxs = %d, want equal counts", len(events), len(ctxs))
	}
	var completion llmkit.Event
	var attemptDurations time.Duration
	attempts := 0
	for i, ev := range events {
		switch ev.Kind {
		case llmkit.KindCompletion:
			completion = ev
		case llmkit.KindAttempt:
			attempts++
			attemptDurations += ev.Duration
			if ev.Attempt.Attempt != attempts {
				t.Errorf("attempt numbered %d, want %d", ev.Attempt.Attempt, attempts)
			}
			if ev.RunID != runID {
				t.Errorf("attempt %d RunID = %q, want %q", attempts, ev.RunID, runID)
			}
			if _, ok := ctxs[i].Deadline(); ok {
				t.Errorf("attempt %d delivered on a context with a deadline: the per-attempt ctx leaked", attempts)
			}
			if ctxErr := ctxs[i].Err(); ctxErr != nil {
				t.Errorf("attempt %d delivered with ctx.Err() = %v, want nil", attempts, ctxErr)
			}
			if ev.Attempt.Err == "" || !reflect.DeepEqual(ev.Attempt.Response, llmkit.Response{}) {
				t.Errorf("attempt %d = %q/%+v, want the failure with a zero Response", attempts, ev.Attempt.Err, ev.Attempt.Response)
			}
		}
	}
	if attempts != 3 || completion.Completion == nil {
		t.Fatalf("attempts = %d completion = %v, want 3 and one Completion", attempts, completion.Completion != nil)
	}
	var backoff time.Duration
	for _, d := range slept {
		backoff += d
	}
	if len(slept) != 2 {
		t.Fatalf("backoff sleeps = %v, want 2", slept)
	}
	if completion.Duration < attemptDurations+backoff {
		t.Errorf("completion Duration %v < attempts %v + backoff %v: the stream call is not fully covered",
			completion.Duration, attemptDurations, backoff)
	}
	if completion.Completion.Err == "" {
		t.Error("failed stream completion carries no error")
	}
}

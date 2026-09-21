package provider

import (
	"context"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// eventCapture collects llmkit events emitted by the retry stage and by an
// outer llmkit.Observe wrapper, so the provider tests can pin who emits what.
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

// TestNew_ObserverEmitsAttemptsOnly: Options.Observer sits in the retry
// stage — one Attempt event per wire call (the failed one included), no
// Completion event ever, tags from Options.Provider and the spec model, and
// the caller's span inherited into every attempt.
func TestNew_ObserverEmitsAttemptsOnly(t *testing.T) {
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
	span := llmkit.NewSpanID()
	ctx := llmkit.WithSpan(llmkit.WithRun(context.Background(), runID), span)
	resp, err := client.Complete(ctx, simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "hello world" {
		t.Errorf("Text = %q, want hello world", resp.Text)
	}
	if hits != 2 {
		t.Fatalf("wire calls = %d, want 2", hits)
	}

	for _, ev := range obs.snapshot() {
		if ev.Kind != llmkit.KindAttempt {
			t.Errorf("New emitted a %s event; only Attempts may come from New", ev.Kind)
		}
	}
	attempts := obs.byKind(llmkit.KindAttempt)
	if len(attempts) != 2 {
		t.Fatalf("attempts = %d, want 2", len(attempts))
	}
	for i, ev := range attempts {
		ae := ev.Attempt
		if ae == nil {
			t.Fatalf("attempt %d has no payload", i)
		}
		if ae.Attempt != i+1 {
			t.Errorf("attempt %d numbered %d, want %d", i, ae.Attempt, i+1)
		}
		if ev.SpanID != span || ev.RunID != runID {
			t.Errorf("attempt %d span/run = %q/%q, want the caller's %q/%q", i, ev.SpanID, ev.RunID, span, runID)
		}
		if ae.Provider != "compat-lane" || ae.Model != "llama-test" {
			t.Errorf("attempt %d tags = %q/%q, want compat-lane/llama-test", i, ae.Provider, ae.Model)
		}
		if i == 0 {
			if ae.Err == "" {
				t.Error("failed attempt carries no error text")
			}
			if ae.StatusCode != 429 {
				t.Errorf("failed attempt StatusCode = %d, want 429 from the APIError", ae.StatusCode)
			}
			if !reflect.DeepEqual(ae.Response, llmkit.Response{}) {
				t.Errorf("failed attempt Response = %+v, want zero", ae.Response)
			}
		} else if ae.Err != "" || ae.Response.Text != "hello world" {
			t.Errorf("successful attempt = %q/%+v, want clean hello world", ae.Err, ae.Response)
		}
	}
}

// TestNew_TruncationVisibleInCompletion: Observe wraps the full provider
// stack from the outside, so its Completion event shows the serializer-
// truncated response the caller sees, while the retry stage's Attempt event
// still shows the raw adapter response — both joined on one span.
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
	stacked, err := New(context.Background(), observeSpec(base), Options{
		Retry:    retry.Config{MaxAttempts: 1, RequestTimeout: time.Second},
		Observer: obs,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	observed := llmkit.Observe(stacked, obs, "openai-compatible", "llama-test")

	// Observe mints the span itself, so the join to assert is completion <-> attempt,
	// not the caller's preset span.
	if _, err := observed.Complete(context.Background(), simpleRequest()); err != nil {
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

// sseHello is a minimal successful Chat Completions stream: one text delta,
// the finish chunk, and the include_usage chunk, terminated by [DONE].
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

// TestNew_StreamAttemptEventsAfterRetry: the retry stage emits Attempt
// events on the STREAM path too — a 503 attempt followed by a successful
// SSE attempt yields two events (the failed one carrying the APIError's
// status), joined to the completion's span, with deltas delivered once.
func TestNew_StreamAttemptEventsAfterRetry(t *testing.T) {
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
	stacked, err := New(context.Background(), observeSpec(base), Options{
		Retry:    fastRetry(),
		Observer: obs,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	observed := llmkit.Observe(stacked, obs, "openai-compatible", "llama-test")

	var texts []string
	resp, err := llmkit.Stream(context.Background(), observed, simpleRequest(), func(d llmkit.Delta) error {
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

// TestTag pins the resolution rule on the exported function: the caller's
// Options.Provider when set, string(spec.Type) otherwise.
func TestTag(t *testing.T) {
	spec := Spec{Type: TypeOpenAICompatible}
	if got := Tag(spec, Options{}); got != "openai-compatible" {
		t.Errorf("Tag(zero Options) = %q, want openai-compatible", got)
	}
	if got := Tag(spec, Options{Provider: "ledger-name"}); got != "ledger-name" {
		t.Errorf("Tag(Options.Provider set) = %q, want ledger-name", got)
	}
}

// TestNew_AttemptTagMatchesTag pins the seam by construction: New's emitted
// Attempt tag equals Tag(spec, opts) on both branches, so the exported
// resolver and the emission path cannot drift. Mutating Tag ALONE (leaving
// New untouched) breaks this test — the subtest also asserts the branch is
// actually exercised, so a mutation to either branch cannot hide.
func TestNew_AttemptTagMatchesTag(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("openai-compatible", "hello", 1, 1)))
	})
	attemptTag := func(t *testing.T, spec Spec, opts Options) string {
		t.Helper()
		capture := &eventCapture{}
		opts.Observer = capture
		opts.Retry = fastRetry()
		client, err := New(context.Background(), spec, opts)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		attempts := capture.byKind(llmkit.KindAttempt)
		if len(attempts) != 1 {
			t.Fatalf("attempts = %d, want 1", len(attempts))
		}
		return attempts[0].Attempt.Provider
	}

	t.Run("default branch", func(t *testing.T) {
		spec := observeSpec(base)
		var opts Options
		got := attemptTag(t, spec, opts)
		if got != Tag(spec, opts) {
			t.Errorf("New attempt tag = %q, want Tag(spec, opts) = %q", got, Tag(spec, opts))
		}
		if got != string(spec.Type) {
			t.Errorf("New attempt tag = %q, want the default branch %q", got, string(spec.Type))
		}
	})
	t.Run("override branch", func(t *testing.T) {
		spec := observeSpec(base)
		opts := Options{Provider: "compat-lane"}
		got := attemptTag(t, spec, opts)
		if got != Tag(spec, opts) {
			t.Errorf("New attempt tag = %q, want Tag(spec, opts) = %q", got, Tag(spec, opts))
		}
		if got != "compat-lane" {
			t.Errorf("New attempt tag = %q, want the override branch %q", got, "compat-lane")
		}
	})
}

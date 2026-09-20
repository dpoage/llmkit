package provider

import (
	"context"
	"net/http"
	"reflect"
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

	// Observe mints the span itself (contract C2), so the join to assert is
	// completion <-> attempt, not the caller's preset span.
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

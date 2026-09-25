package provider

// These exercise Wrap's retry stage over each vendor's real Stream
// adapter. Adapter packages must not depend on the provider stack.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/internal/anthropic"
	"github.com/dpoage/llmkit/provider/internal/google"
	"github.com/dpoage/llmkit/provider/internal/openai"
	"github.com/dpoage/llmkit/retry"
)

// fastConformanceRetry is a two-attempt retry.Config with no backoff
// sleep, so hit counts are deterministic.
func fastConformanceRetry() retry.Config {
	return retry.Config{
		MaxAttempts: 2,
		BaseDelay:   time.Millisecond,
		Sleep:       func(context.Context, time.Duration) error { return nil },
	}
}

// anthropicSSEEvent is one "event: NAME\ndata: JSON\n\n" frame in the
// Anthropic Messages SSE wire format.
type anthropicSSEEvent struct{ name, data string }

func anthropicSSEHandler(events []anthropicSSEEvent) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, ev := range events {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.name, ev.data)
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}
}

// TestConformance_AnthropicStream_DeliveredDeltaForbidsRetry asserts that
// an in-band SSE error on a committed 200 stream that has already
// delivered a delta is not retried: the stream is already partial.
func TestConformance_AnthropicStream_DeliveredDeltaForbidsRetry(t *testing.T) {
	events := []anthropicSSEEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`},
		{"error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
	}
	hits := 0
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits++
		anthropicSSEHandler(events)(w, r)
	})
	ad, err := anthropic.New("claude-test", anthropic.Options{APIKey: "test-key", BaseURL: base})
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}
	client := Wrap(ad, Options{Retry: fastConformanceRetry()})

	var got []llmkit.Delta
	_, err = llmkit.Stream(t.Context(), client, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if !errors.Is(err, llmkit.ErrOverloaded) {
		t.Errorf("err = %v, want ErrOverloaded", err)
	}
	if hits != 1 {
		t.Errorf("wire hits = %d, want 1 (a delivered delta forbids retry)", hits)
	}
	if len(got) != 1 || got[0].Text != "partial" {
		t.Errorf("deltas = %#v, want the single pre-error fragment", got)
	}
}

// TestConformance_AnthropicStream_FirstEventClassifiedByType asserts that
// an in-band SSE error event before any delivered delta is classified by
// vendor type: overloaded_error retries, invalid_request_error is
// terminal.
func TestConformance_AnthropicStream_FirstEventClassifiedByType(t *testing.T) {
	start := `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`
	tests := []struct {
		name     string
		events   []anthropicSSEEvent
		wantKind error
		wantHits int
	}{
		{
			name: "overloaded_error",
			events: []anthropicSSEEvent{
				{"message_start", start},
				{"error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
			},
			wantKind: llmkit.ErrOverloaded,
			wantHits: 2,
		},
		{
			name: "invalid_request_error",
			events: []anthropicSSEEvent{
				{"message_start", start},
				{"error", `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`},
			},
			wantKind: llmkit.ErrInvalidRequest,
			wantHits: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hits := 0
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits++
				anthropicSSEHandler(tc.events)(w, r)
			})
			ad, err := anthropic.New("claude-test", anthropic.Options{APIKey: "test-key", BaseURL: base})
			if err != nil {
				t.Fatalf("anthropic.New: %v", err)
			}
			client := Wrap(ad, Options{Retry: fastConformanceRetry()})

			resp, err := llmkit.Stream(context.Background(), client, simpleRequest(), func(llmkit.Delta) error { return nil })
			if err == nil {
				t.Fatal("Stream: want error, got nil")
			}
			if !errors.Is(err, tc.wantKind) {
				t.Errorf("err = %v, want %v (in-band type classification)", err, tc.wantKind)
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %T (%v), want *llmkit.APIError", err, err)
			}
			if apiErr.StatusCode != http.StatusOK {
				t.Errorf("StatusCode = %d, want 200 (the committed stream's status)", apiErr.StatusCode)
			}
			if !reflect.DeepEqual(resp, llmkit.Response{}) {
				t.Errorf("resp = %#v, want zero Response", resp)
			}
			if hits != tc.wantHits {
				t.Errorf("wire hits = %d, want %d", hits, tc.wantHits)
			}
		})
	}
}

// openaiSSEHandler serves "data: JSON\n\n" frames terminated by
// "data: [DONE]\n\n" — the OpenAI Chat Completions streaming wire format.
func openaiSSEHandler(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, c := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// TestConformance_OpenAIStream_InBandSSEErrorClassifiedByType asserts that
// an SSE data line carrying the body's error object on a committed 200
// stream is classified by vendor type: invalid_request_error is terminal,
// server_error is retryable.
func TestConformance_OpenAIStream_InBandSSEErrorClassifiedByType(t *testing.T) {
	tests := []struct {
		name     string
		event    string
		wantKind error
		wantHits int
		wantMsg  string
	}{
		{"invalid_request_error", `{"error":{"type":"invalid_request_error","message":"x"}}`, llmkit.ErrInvalidRequest, 1, "x"},
		{"server_error", `{"error":{"type":"server_error","message":"boom"}}`, llmkit.ErrServer, 2, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hits := 0
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits++
				openaiSSEHandler(tc.event)(w, r)
			})
			ad, err := openai.New("gpt-test", openai.Options{APIKey: "k", BaseURL: base})
			if err != nil {
				t.Fatalf("openai.New: %v", err)
			}
			client := Wrap(ad, Options{Retry: fastConformanceRetry()})

			resp, err := llmkit.Stream(context.Background(), client, simpleRequest(), func(llmkit.Delta) error { return nil })
			if err == nil {
				t.Fatal("Stream: want error, got nil")
			}
			if !errors.Is(err, tc.wantKind) {
				t.Errorf("err = %v, want %v (in-band type classification)", err, tc.wantKind)
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %T (%v), want *llmkit.APIError", err, err)
			}
			if apiErr.StatusCode != http.StatusOK {
				t.Errorf("StatusCode = %d, want 200 (the committed stream's status)", apiErr.StatusCode)
			}
			if tc.wantMsg != "" && apiErr.Message != tc.wantMsg {
				t.Errorf("Message = %q, want the event's error message %q", apiErr.Message, tc.wantMsg)
			}
			if !reflect.DeepEqual(resp, llmkit.Response{}) {
				t.Errorf("resp = %#v, want zero Response", resp)
			}
			if hits != tc.wantHits {
				t.Errorf("wire hits = %d, want %d", hits, tc.wantHits)
			}
		})
	}
}

// TestConformance_GoogleStream_RetryTimeoutStalledStream asserts that the
// retry stage's per-attempt RequestTimeout aborts a stream that stalls
// after delivering a chunk: a silent partial would look like a
// successful attempt and bypass the timeout.
func TestConformance_GoogleStream_RetryTimeoutStalledStream(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}`+"\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done() // stall until the per-attempt deadline reaps us
	})
	ad, err := google.New(context.Background(), "gemini-test", google.Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("google.New: %v", err)
	}
	client := Wrap(ad, Options{Retry: retry.Config{
		MaxAttempts:    1,
		RequestTimeout: 150 * time.Millisecond,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
	}})

	deltas := 0
	resp, err := llmkit.Stream(context.Background(), client, simpleRequest(), func(llmkit.Delta) error {
		deltas++
		return nil
	})
	if err == nil {
		t.Fatal("expected timeout error, got partial success")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want context.DeadlineExceeded in the chain", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response", resp)
	}
	if deltas != 1 {
		t.Fatalf("deltas = %d, want 1 (first chunk before the stall)", deltas)
	}
}

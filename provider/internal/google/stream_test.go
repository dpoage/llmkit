package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// SSE event shapes follow the Gemini API's streamGenerateContent?alt=sse
// format: one `data: {json}` event per GenerateContentResponse chunk,
// blank-line separated. The genai SDK parses exactly this shape
// (google.golang.org/genai v1.64.0 api_client.go: scan + iterateResponseStream);
// chunk layouts mirror the streaming and function-calling examples at
// https://ai.google.dev/gemini-api/docs (text parts, thought parts with
// thoughtSignature, functionCall parts, usageMetadata on the final chunk).

// testSig is a thoughtSignature in its wire form (base64 of "signature-1");
// testSig2 distinguishes the signed function call's signature.
var (
	testSig  = base64.StdEncoding.EncodeToString([]byte("signature-1"))
	testSig2 = base64.StdEncoding.EncodeToString([]byte("signature-2"))
)

// streamServer starts an httptest server answering the SSE streaming
// endpoint with the given chunks, one `data:` event each, and fails the
// test if the SDK hits any other endpoint shape.
func streamServer(t *testing.T, chunks ...map[string]any) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":streamGenerateContent") || r.URL.Query().Get("alt") != "sse" {
			t.Errorf("unexpected streaming request %s", r.URL)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			b, _ := json.Marshal(c)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
		}
	})
}

// completeServer starts an httptest server answering the non-streaming
// generateContent endpoint with the given body.
func completeServer(t *testing.T, body map[string]any) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		b, _ := json.Marshal(body)
		_, _ = w.Write(b)
	})
}

func textChunk(text string) map[string]any {
	return map[string]any{"candidates": []any{map[string]any{
		"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": text}}},
	}}}
}

func finalChunk(finish string, usage map[string]any) map[string]any {
	cand := map[string]any{}
	if finish != "" {
		cand["finishReason"] = finish
	}
	c := map[string]any{"candidates": []any{cand}}
	if usage != nil {
		c["usageMetadata"] = usage
	}
	return c
}

func newStreamClient(t *testing.T, base string) *googleAdapter {
	t.Helper()
	cl, err := New(context.Background(), "gemini-test", Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return cl.(*googleAdapter)
}

// TestStream_TextAcrossChunks streams one reply's text across two chunks and
// usage/finishReason on the last: deltas arrive per chunk, the aggregate
// merges the consecutive text parts into one block, and the usage is the
// LAST chunk's (Gemini repeats cumulative usageMetadata on every chunk).
func TestStream_TextAcrossChunks(t *testing.T) {
	base := streamServer(t,
		textChunk("Hello"),
		map[string]any{
			"candidates": []any{map[string]any{
				"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": " world"}}},
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 2, "totalTokenCount": 13},
		},
		finalChunk("STOP", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}),
	)
	cl := newStreamClient(t, base)

	var deltas []llmkit.Delta
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(d llmkit.Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []llmkit.Delta{
		{Kind: llmkit.DeltaText, Text: "Hello"},
		{Kind: llmkit.DeltaText, Text: " world"},
	}
	if !reflect.DeepEqual(deltas, want) {
		t.Fatalf("deltas = %+v, want %+v", deltas, want)
	}
	if resp.Text != "Hello world" {
		t.Errorf("Text = %q, want %q", resp.Text, "Hello world")
	}
	if len(resp.Blocks) != 1 || resp.Blocks[0].Kind != llmkit.BlockText || resp.Blocks[0].Text != "Hello world" {
		t.Errorf("Blocks = %+v, want one merged text block", resp.Blocks)
	}
	if resp.Usage != (llmkit.Usage{InputTokens: 11, OutputTokens: 7}) {
		t.Errorf("Usage = %+v, want last chunk's {11 7 0}", resp.Usage)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llmkit.StopEndTurn)
	}
}

// TestStream_ThoughtThenText streams a thought part carrying a
// thoughtSignature followed by plain text: the thought delta precedes the
// text delta, and the aggregated thinking block's Raw keeps the signature
// bytes verbatim so googleAssistantParts can re-emit them next turn.
func TestStream_ThoughtThenText(t *testing.T) {
	base := streamServer(t,
		map[string]any{"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": []any{map[string]any{
				"text":             " pondering",
				"thought":          true,
				"thoughtSignature": testSig,
			}}},
		}}},
		textChunk("Answer"),
		finalChunk("STOP", map[string]any{"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9}),
	)
	cl := newStreamClient(t, base)

	var deltas []llmkit.Delta
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(d llmkit.Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []llmkit.Delta{
		{Kind: llmkit.DeltaThinking, Text: " pondering"},
		{Kind: llmkit.DeltaText, Text: "Answer"},
	}
	if !reflect.DeepEqual(deltas, want) {
		t.Fatalf("deltas = %+v, want %+v", deltas, want)
	}
	if len(resp.Blocks) != 2 {
		t.Fatalf("Blocks = %+v, want [thinking text]", resp.Blocks)
	}
	thought := resp.Blocks[0]
	if thought.Kind != llmkit.BlockThinking || thought.Text != " pondering" || thought.Provider != "google" {
		t.Errorf("Blocks[0] = %+v, want google thinking block", thought)
	}
	var part struct {
		Text             string `json:"text"`
		Thought          bool   `json:"thought"`
		ThoughtSignature []byte `json:"thoughtSignature"`
	}
	if err := json.Unmarshal(thought.Raw, &part); err != nil {
		t.Fatalf("Blocks[0].Raw: %v", err)
	}
	if !part.Thought || string(part.ThoughtSignature) != "signature-1" {
		t.Errorf("Raw = %s, want thought:true with signature-1 bytes", thought.Raw)
	}
	if !reflect.DeepEqual(resp.Blocks[1], llmkit.Block{Kind: llmkit.BlockText, Text: "Answer"}) {
		t.Errorf("Blocks[1] = %+v, want text Answer", resp.Blocks[1])
	}
	if resp.Text != "Answer" {
		t.Errorf("Text = %q, want %q (thinking must not leak into Text)", resp.Text, "Answer")
	}
}

// TestStream_FunctionCalls streams one chunk carrying two functionCall
// parts: one delta per call in arrival order with the full arguments —
// Arguments render exactly the final ToolCall.Arguments, and a call the
// server sent without an id keeps the empty id the Complete normalizer
// produces (no synthetic scheme).
func TestStream_FunctionCalls(t *testing.T) {
	base := streamServer(t,
		map[string]any{"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": []any{
				map[string]any{"functionCall": map[string]any{
					"id":   "call-1",
					"name": "get_weather",
					"args": map[string]any{"city": "Paris"},
				}},
				map[string]any{"functionCall": map[string]any{"name": "get_time"}},
			}},
		}}},
		finalChunk("STOP", map[string]any{"promptTokenCount": 9, "candidatesTokenCount": 3, "totalTokenCount": 12}),
	)
	cl := newStreamClient(t, base)

	var deltas []llmkit.Delta
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(d llmkit.Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []llmkit.Delta{
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call-1", Name: "get_weather", Arguments: `{"city":"Paris"}`},
		{Kind: llmkit.DeltaToolCall, Index: 1, Name: "get_time", Arguments: "{}"},
	}
	if !reflect.DeepEqual(deltas, want) {
		t.Fatalf("deltas = %+v, want %+v", deltas, want)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("ToolCalls = %+v, want 2", resp.ToolCalls)
	}
	for i, d := range deltas {
		tc := resp.ToolCalls[i]
		if tc.ID != d.ID || tc.Name != d.Name || string(tc.Arguments) != d.Arguments {
			t.Errorf("ToolCalls[%d] = %+v, want delta's id/name/arguments", i, tc)
		}
	}
	if resp.StopReason != llmkit.StopToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llmkit.StopToolUse)
	}
}

// TestStream_IdentityWithComplete feeds the same exchange to Complete
// (single-shot JSON) and Stream (the same content split across SSE chunks,
// text re-merged) and requires byte-identical normalized Responses. There
// is no vendor Raw object on the Response to except: llmkit.Response is
// compared with reflect.DeepEqual as-is.
func TestStream_IdentityWithComplete(t *testing.T) {
	parts := []any{
		map[string]any{"text": "Hello world"},
		map[string]any{"text": " pondering", "thought": true, "thoughtSignature": testSig},
		map[string]any{
			"functionCall":     map[string]any{"id": "call-9", "name": "get_weather", "args": map[string]any{"city": "Paris"}},
			"thoughtSignature": testSig2,
		},
	}
	usage := map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}

	req := simpleRequest()

	cl := newStreamClient(t, completeServer(t, map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": "STOP",
		}},
		"usageMetadata":  usage,
		"promptFeedback": map[string]any{"safetyRatings": []any{map[string]any{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}}},
		"modelVersion":   "gemini-test",
		"responseId":     "resp-1",
	}))
	want, err := cl.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	cl = newStreamClient(t, streamServer(t,
		map[string]any{
			"candidates":     []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": "Hello"}}}}},
			"promptFeedback": map[string]any{"safetyRatings": []any{map[string]any{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}}},
		},
		textChunk(" world"),
		map[string]any{"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": []any{
				parts[1].(map[string]any), parts[2].(map[string]any),
			}},
		}}},
		finalChunk("STOP", usage),
		map[string]any{"modelVersion": "gemini-test", "responseId": "resp-1"},
	))
	var deltas []llmkit.Delta
	got, err := cl.Stream(context.Background(), req, func(d llmkit.Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Stream response != Complete response:\nstream:  %#v\ncomplete: %#v", got, want)
	}
	wantDeltas := []llmkit.Delta{
		{Kind: llmkit.DeltaText, Text: "Hello"},
		{Kind: llmkit.DeltaText, Text: " world"},
		{Kind: llmkit.DeltaThinking, Text: " pondering"},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call-9", Name: "get_weather", Arguments: `{"city":"Paris"}`},
	}
	if !reflect.DeepEqual(deltas, wantDeltas) {
		t.Fatalf("deltas = %+v, want %+v", deltas, wantDeltas)
	}
}

// TestStream_NilFn pins the contract that a nil fn is allowed and makes
// Stream equivalent to Complete's response.
func TestStream_NilFn(t *testing.T) {
	base := streamServer(t, textChunk("Hello"), finalChunk("STOP", map[string]any{"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4}))
	cl := newStreamClient(t, base)
	resp, err := cl.Stream(context.Background(), simpleRequest(), nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Text != "Hello" || resp.StopReason != llmkit.StopEndTurn || resp.Usage.OutputTokens != 1 {
		t.Fatalf("resp = %+v, want Complete-shaped response", resp)
	}
}

// TestStream_FnErrorCancels pins the contract that a fn error cancels the
// stream, is returned wrapped, and no Response is produced.
func TestStream_FnErrorCancels(t *testing.T) {
	base := streamServer(t, textChunk("Hello"), textChunk(" world"), finalChunk("STOP", nil))
	cl := newStreamClient(t, base)
	sentinel := errors.New("subscriber gone")
	calls := 0
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(d llmkit.Delta) error {
		calls++
		if calls == 2 {
			return sentinel
		}
		return nil
	})
	if err == nil {
		t.Fatal("expected fn error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("error = %v, want wrapped sentinel", err)
	}
	if !strings.HasPrefix(err.Error(), "llmkit: stream fn: ") {
		t.Fatalf("error = %v, want llmkit: stream fn: prefix", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response", resp)
	}
	if calls != 2 {
		t.Fatalf("fn called %d times, want 2 (cancel on second)", calls)
	}
}

// TestStream_RateLimited pins that a 429 before the stream starts
// normalizes through the same normalizeErr Complete uses.
func TestStream_RateLimited(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`))
	})
	cl := newStreamClient(t, base)
	_, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmkit.ErrRateLimited) {
		t.Fatalf("error = %v, want ErrRateLimited", err)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("error = %v, want *llmkit.APIError with StatusCode 429", err)
	}
}

// TestStream_ErrorMidStream pins that an in-band error event after good
// chunks normalizes the same way — genai reports it as its only error type,
// genai.APIError, and normalizeErr classifies it from the error code.
func TestStream_ErrorMidStream(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: "+mustJSON(t, textChunk("Hello"))+"\n\n")
		// In-band terminal error: a bare JSON object (no "data:" prefix) in
		// Google's error shape, which the SDK parses into genai.APIError
		// (api_client.go iterateResponseStream default branch).
		_, _ = fmt.Fprint(w, `{"error":{"code":500,"message":"generation failed","status":"INTERNAL"}}`+"\n\n")
	})
	cl := newStreamClient(t, base)
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response on stream error", resp)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

// TestStream_ImplementsStreamingClient pins the adapter to the streaming
// contract: llmkit.Stream must take this path, not the Complete fallback.
func TestStream_ImplementsStreamingClient(t *testing.T) {
	var _ llmkit.StreamingClient = (*googleAdapter)(nil)
}

// --- truncation and cancellation --------------------------------------------

// genai v1.64.0's iterateResponseStream never yields transport errors that
// surface after scanning starts — context cancellation, deadlines, and
// connection resets only reach its log — and a clean EOF is indistinguishable
// from a stream cut short. The tests below pin that Stream fails such
// exchanges loudly instead of returning a truncated partial as StopEndTurn.

// TestStream_ContextCanceledMidStream: the caller cancels while the stream
// is open. No Response, and context.Canceled reachable through the error
// chain — the same normalization Complete applies to a canceled request.
func TestStream_ContextCanceledMidStream(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: "+mustJSON(t, textChunk("Hello"))+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done() // client goes away when the test cancels
	})
	cl := newStreamClient(t, base)
	ctx, cancel := context.WithCancel(context.Background())
	deltas := 0
	resp, err := cl.Stream(ctx, simpleRequest(), func(llmkit.Delta) error {
		deltas++
		cancel()
		return nil
	})
	if err == nil {
		t.Fatal("expected error after cancellation, got partial success")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled in the chain", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response", resp)
	}
	if deltas != 1 {
		t.Fatalf("deltas = %d, want 1", deltas)
	}
}

// TestStream_CleanEOFWithoutFinishReason: the server closes the stream after
// content chunks without ever sending a finishReason. That is a truncated
// exchange, not a completion — mapGoogleStop would dress it up as StopEndTurn.
func TestStream_CleanEOFWithoutFinishReason(t *testing.T) {
	base := streamServer(t, textChunk("Hello"))
	cl := newStreamClient(t, base)
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if err == nil {
		t.Fatal("expected error for stream without finishReason, got partial success")
	}
	if !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
	if !strings.Contains(err.Error(), "finishReason") {
		t.Fatalf("error = %v, want finishReason in the message", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response", resp)
	}
}

// TestStream_ConnectionResetMidStream: the server resets the connection
// (RST via SO_LINGER 0) after the first chunk was consumed. The SDK's
// iterator logs the read error and stops silently; Stream must still fail.
func TestStream_ConnectionResetMidStream(t *testing.T) {
	firstSeen := make(chan struct{})
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Error("response is not a hijacker")
			return
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		// Hijacked conn: write the response head ourselves, then one SSE event.
		head := "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
		if _, err := conn.Write([]byte(head + "data: " + mustJSON(t, textChunk("Hello")) + "\n\n")); err != nil {
			t.Errorf("write: %v", err)
			return
		}
		select {
		case <-firstSeen: // client consumed the chunk; reset now
		case <-time.After(2 * time.Second):
			t.Error("client never consumed the first chunk")
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetLinger(0) // next Close sends RST, discarding the stream
		}
	})
	cl := newStreamClient(t, base)
	deltas := 0
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error {
		deltas++
		close(firstSeen)
		return nil
	})
	if err == nil {
		t.Fatal("expected error after connection reset, got partial success")
	}
	if !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response", resp)
	}
	if deltas != 1 {
		t.Fatalf("deltas = %d, want 1", deltas)
	}
}

// TestStream_RetryTimeoutStalledStream: WithRetry's per-attempt
// RequestTimeout must abort a stream that stalls after delivering a chunk —
// a silent partial would make the attempt look successful and defeat the
// timeout entirely. The attempt fails with context.DeadlineExceeded in the
// chain and no Response.
func TestStream_RetryTimeoutStalledStream(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: "+mustJSON(t, textChunk("Hello"))+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done() // stall until the attempt deadline reaps us
	})
	cl := llmkit.WithRetry(newStreamClient(t, base), llmkit.RetryConfig{
		MaxAttempts:    1,
		RequestTimeout: 150 * time.Millisecond,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
	})
	deltas := 0
	resp, err := llmkit.Stream(context.Background(), cl, simpleRequest(), func(llmkit.Delta) error {
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

// TestStream_FnErrorStopsIteration pins the cancellation: after fn errors,
// no further part may reach fn, even with more chunks queued on the wire.
func TestStream_FnErrorStopsIteration(t *testing.T) {
	base := streamServer(t, textChunk("A"), textChunk("B"), textChunk("C"))
	cl := newStreamClient(t, base)
	sentinel := errors.New("stop")
	calls := 0
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) || !strings.HasPrefix(err.Error(), "llmkit: stream fn: ") {
		t.Fatalf("error = %v, want wrapped sentinel", err)
	}
	if calls != 1 {
		t.Fatalf("fn called %d times, want 1 — iteration must stop on the fn error", calls)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want zero Response", resp)
	}
}

// TestStream_FinishReasonLastCarrierWins pins that the finishReason of the
// last chunk carrying one decides the stop reason.
func TestStream_FinishReasonLastCarrierWins(t *testing.T) {
	base := streamServer(t,
		map[string]any{"candidates": []any{map[string]any{"finishReason": "MAX_TOKENS"}}},
		map[string]any{"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": []any{map[string]any{"text": "hi"}}},
			"finishReason": "STOP",
		}}},
	)
	cl := newStreamClient(t, base)
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("StopReason = %q, want STOP's %q (last carrier wins)", resp.StopReason, llmkit.StopEndTurn)
	}
}

// TestStream_UnspecifiedFinishReasonStillFinishes pins that a chunk carrying
// only FINISH_REASON_UNSPECIFIED counts as having seen a finish reason (so a
// stream ending after it is not an error) while the aggregate keeps none —
// the same EndTurn Complete produces for an unspecified stop.
func TestStream_UnspecifiedFinishReasonStillFinishes(t *testing.T) {
	base := streamServer(t, map[string]any{"candidates": []any{map[string]any{
		"finishReason": "FINISH_REASON_UNSPECIFIED",
	}}})
	cl := newStreamClient(t, base)
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("StopReason = %q, want %q", resp.StopReason, llmkit.StopEndTurn)
	}
	if len(resp.Blocks) != 0 {
		t.Fatalf("Blocks = %+v, want none", resp.Blocks)
	}
}

// TestStream_EmptyTextPartsSuppressed pins that empty text parts emit no
// delta and do not break the merge of the text around them.
func TestStream_EmptyTextPartsSuppressed(t *testing.T) {
	base := streamServer(t,
		map[string]any{"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": []any{
				map[string]any{"text": ""},
				map[string]any{"text": "A"},
				map[string]any{"text": ""},
				map[string]any{"text": "B"},
			}},
		}}},
		finalChunk("STOP", map[string]any{"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}),
	)
	cl := newStreamClient(t, base)
	var deltas []llmkit.Delta
	resp, err := cl.Stream(context.Background(), simpleRequest(), func(d llmkit.Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "A"}, {Kind: llmkit.DeltaText, Text: "B"}}
	if !reflect.DeepEqual(deltas, want) {
		t.Fatalf("deltas = %+v, want %+v", deltas, want)
	}
	if len(resp.Blocks) != 1 || resp.Blocks[0].Text != "AB" || resp.Text != "AB" {
		t.Fatalf("resp = %+v, want one merged AB block", resp)
	}
}

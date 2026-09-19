package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/openai/openai-go/v3"
)

// Streaming fixtures use the vendor's real chunk shapes, copied from the
// openai-go SDK's own streaming test (streamaccumulator_test.go,
// mockResponseBody in openai-go v3.43.0): chat.completion.chunk objects with
// a choices[0].delta, a final usage-only chunk with empty choices sent under
// stream_options.include_usage, and a "data: [DONE]" terminator. Argument
// fragments are JSON-escaped mechanically so the wire bytes stay exact.

const (
	streamID      = "chatcmpl-A3Tguz3LSXTHBTY2NAPBCSyfBltxF"
	streamModel   = "gpt-test"
	streamCreated = 1725392480
	streamUsage   = `{"prompt_tokens":10,"completion_tokens":100,"total_tokens":110,` +
		`"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":null}`
)

// chunkJSON renders one delta-carrying chunk. finish and usage are the bare
// JSON values ("stop" / an object string); "" means the wire's null.
func chunkJSON(delta, finish, usage string) string {
	if delta == "" {
		delta = "{}"
	}
	if finish == "" {
		finish = "null"
	} else {
		finish = `"` + finish + `"`
	}
	if usage == "" {
		usage = "null"
	}
	return fmt.Sprintf(`{"id":%q,"object":"chat.completion.chunk","created":%d,"model":%q,`+
		`"system_fingerprint":"fp_157b3831f5","choices":[{"index":0,"delta":%s,`+
		`"finish_reason":%s}],"usage":%s}`, streamID, streamCreated, streamModel, delta, finish, usage)
}

// argFragChunk renders a tool-call argument fragment for vendor tool call
// index, with the fragment JSON-escaped exactly as the wire carries it.
func argFragChunk(index int64, fragment string) string {
	return fmt.Sprintf(`{"id":%q,"object":"chat.completion.chunk","created":%d,"model":%q,`+
		`"system_fingerprint":"fp_157b3831f5","choices":[{"index":0,"delta":{"tool_calls":[`+
		`{"index":%d,"function":{"arguments":%s}}]},"finish_reason":null}],"usage":null}`,
		streamID, streamCreated, streamModel, index, strconv.Quote(fragment))
}

// usageChunkJSON renders the final usage-only chunk (empty choices) that
// include_usage asks the endpoint for.
func usageChunkJSON(usage string) string {
	return fmt.Sprintf(`{"id":%q,"object":"chat.completion.chunk","created":%d,"model":%q,`+
		`"system_fingerprint":"fp_157b3831f5","choices":[],"usage":%s}`,
		streamID, streamCreated, streamModel, usage)
}

// sseHandler serves the chunks as a text/event-stream body terminated by
// [DONE], the wire format the OpenAI Chat Completions stream endpoint
// speaks. The request body is handed to onReq when non-nil.
func sseHandler(onReq func(body []byte), chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if onReq != nil {
			body, _ := io.ReadAll(r.Body)
			onReq(body)
		}
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

// sseTruncated serves the chunks and ends the response body without a
// finish_reason chunk and without [DONE] — the wire a client sees when a
// connection dies mid-generation: a clean EOF at the decoder.
func sseTruncated(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
		}
	}
}

// stream collects the deltas fn sees plus the final Response. It goes
// through the llmkit.StreamingClient interface, the same dispatch
// llmkit.Stream uses, so the tests prove real streaming rather than the
// synthesized path.
func stream(t *testing.T, baseURL string, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, []llmkit.Delta, error) {
	t.Helper()
	client := New(streamModel, Options{APIKey: "k", BaseURL: baseURL})
	sc, ok := client.(llmkit.StreamingClient)
	if !ok {
		t.Fatalf("openai adapter does not implement llmkit.StreamingClient")
	}
	var got []llmkit.Delta
	wrap := fn
	if fn != nil {
		wrap = func(d llmkit.Delta) error {
			got = append(got, d)
			return fn(d)
		}
	}
	resp, err := sc.Stream(context.Background(), req, wrap)
	return resp, got, err
}

// identityExchange is one wire exchange served two ways: the non-stream
// chat.completion body for Complete and an equivalent SSE chunk sequence for
// Stream (text, one tool call with fragmented arguments, finish_reason, and
// the final usage chunk).
const identityCompletionBody = `{"id":"` + streamID + `","object":"chat.completion","created":` +
	`1725392480,"model":"` + streamModel + `","choices":[{"index":0,"message":{"role":"assistant",` +
	`"content":"Let me check","refusal":null,"tool_calls":[{"id":"call_FXoAjBUMcVv1k40fficJ9cSs",` +
	`"type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Santorini, Greece\"}"}}]},` +
	`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":100,` +
	`"total_tokens":110,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":null}}`

var identityChunks = []string{
	chunkJSON(`{"role":"assistant","content":"","refusal":null}`, "", ""),
	chunkJSON(`{"content":"Let me"}`, "", ""),
	chunkJSON(`{"content":" check"}`, "", ""),
	chunkJSON(`{"tool_calls":[{"index":0,"id":"call_FXoAjBUMcVv1k40fficJ9cSs","type":"function",`+
		`"function":{"name":"get_weather","arguments":""}}]}`, "", ""),
	argFragChunk(0, `{"location`),
	argFragChunk(0, `":"Santorini, Greece"}`),
	chunkJSON(`{}`, "tool_calls", ""),
	usageChunkJSON(streamUsage),
}

// TestOpenAIStream_ResponseIdentityIsComplete feeds the same exchange to
// Complete (plain JSON) and Stream (SSE) and requires identical normalized
// Responses: Stream must normalize through the same toResponse Complete
// uses, with usage arriving on the final include_usage chunk.
func TestOpenAIStream_ResponseIdentityIsComplete(t *testing.T) {
	plain := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(identityCompletionBody))
	})
	sse := newServer(t, sseHandler(func(body []byte) {
		var wire struct {
			Stream        *bool `json:"stream"`
			StreamOptions struct {
				IncludeUsage *bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			t.Errorf("request body does not parse: %v", err)
			return
		}
		if wire.Stream == nil || !*wire.Stream {
			t.Errorf("stream request does not set stream:true")
		}
		if wire.StreamOptions.IncludeUsage == nil || !*wire.StreamOptions.IncludeUsage {
			t.Errorf("stream request does not set stream_options.include_usage")
		}
	}, identityChunks...))

	completeResp, err := New(streamModel, Options{APIKey: "k", BaseURL: plain}).
		Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	streamResp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if !reflect.DeepEqual(completeResp, streamResp) {
		t.Fatalf("Stream response differs from Complete:\ncomplete: %+v\nstream:   %+v", completeResp, streamResp)
	}
	if completeResp.Usage.InputTokens != 10 || completeResp.Usage.OutputTokens != 100 ||
		completeResp.Usage.CacheReadInputTokens != 20 {
		t.Fatalf("usage not taken from the final chunk: %+v", completeResp.Usage)
	}

	want := []llmkit.Delta{
		{Kind: llmkit.DeltaText, Text: "Let me"},
		{Kind: llmkit.DeltaText, Text: " check"},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_FXoAjBUMcVv1k40fficJ9cSs", Name: "get_weather", Arguments: ""},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_FXoAjBUMcVv1k40fficJ9cSs", Name: "get_weather", Arguments: `{"location`},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_FXoAjBUMcVv1k40fficJ9cSs", Name: "get_weather", Arguments: `":"Santorini, Greece"}`},
	}
	if !reflect.DeepEqual(want, deltas) {
		t.Fatalf("deltas =\n%+v\nwant\n%+v", deltas, want)
	}
}

// TestOpenAIStream_TwoToolCallsFragmented covers two calls whose argument
// fragments interleave on the wire: each delta repeats its call's ID and
// name (first fragment or remembered), and the final Response concatenates
// each call's arguments.
func TestOpenAIStream_TwoToolCallsFragmented(t *testing.T) {
	startChunk := func(index int64, id, name string) string {
		return chunkJSON(
			fmt.Sprintf(`{"tool_calls":[{"index":%d,"id":%q,"type":"function",`+
				`"function":{"name":%q,"arguments":""}}]}`, index, id, name), "", "")
	}
	chunks := []string{
		startChunk(0, "call_aaa", "get_time"),
		startChunk(1, "call_bbb", "get_weather"),
		argFragChunk(0, `{"now":"12:00"}`),
		argFragChunk(1, `{"city":"Santorini"}`),
		chunkJSON(`{}`, "tool_calls", ""),
		usageChunkJSON(streamUsage),
	}
	sse := newServer(t, sseHandler(nil, chunks...))
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	want := []llmkit.Delta{
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_aaa", Name: "get_time", Arguments: ""},
		{Kind: llmkit.DeltaToolCall, Index: 1, ID: "call_bbb", Name: "get_weather", Arguments: ""},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_aaa", Name: "get_time", Arguments: `{"now":"12:00"}`},
		{Kind: llmkit.DeltaToolCall, Index: 1, ID: "call_bbb", Name: "get_weather", Arguments: `{"city":"Santorini"}`},
	}
	if !reflect.DeepEqual(want, deltas) {
		t.Fatalf("deltas =\n%+v\nwant\n%+v", deltas, want)
	}
	wantCalls := []llmkit.ToolCall{
		{ID: "call_aaa", Name: "get_time", Arguments: []byte(`{"now":"12:00"}`)},
		{ID: "call_bbb", Name: "get_weather", Arguments: []byte(`{"city":"Santorini"}`)},
	}
	if !reflect.DeepEqual(wantCalls, resp.ToolCalls) {
		t.Fatalf("ToolCalls = %+v, want %+v", resp.ToolCalls, wantCalls)
	}
	if resp.StopReason != llmkit.StopToolUse {
		t.Fatalf("StopReason = %q, want %q", resp.StopReason, llmkit.StopToolUse)
	}
}

// TestToolCallTracker_RenumbersByFirstAppearance pins the vendor-index ->
// Delta.Index mapping: first-appearance order among the distinct calls seen,
// independent of the wire's numbers (which normally coincide with it).
func TestToolCallTracker_RenumbersByFirstAppearance(t *testing.T) {
	tr := newToolCallTracker()
	d1 := tr.delta(openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: 5, ID: "call_x", Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{Name: "a"},
	})
	d2 := tr.delta(openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: 9, Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{Arguments: `{`},
	})
	d3 := tr.delta(openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: 5, Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{Arguments: `}`},
	})
	want := []llmkit.Delta{
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_x", Name: "a"},
		{Kind: llmkit.DeltaToolCall, Index: 1, Arguments: `{`},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_x", Name: "a", Arguments: `}`},
	}
	// Index 5 -> position 0 (first appearance), index 9 -> position 1; a
	// call's later fragments repeat its remembered ID and name, and a call
	// whose fragments never carried an ID stays unnamed.
	if !reflect.DeepEqual(want, []llmkit.Delta{d1, d2, d3}) {
		t.Fatalf("tracker deltas =\n%+v\n%+v\n%+v\nwant\n%+v", d1, d2, d3, want)
	}
	ft := newToolCallTracker()
	n1 := ft.delta(openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: 0, ID: "call_n", Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{Name: "get_"},
	})
	n2 := ft.delta(openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: 0, Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{Name: "weather"},
	})
	// A fragmented name concatenates and every fragment repeats the
	// accumulated name the final Response will carry.
	if n1.Name != "get_" || n2.Name != "get_weather" {
		t.Fatalf("fragmented names = %q, %q; want get_, get_weather", n1.Name, n2.Name)
	}
	if n1.ID != "call_n" || n2.ID != "call_n" {
		t.Fatalf("IDs = %q, %q; want the remembered call_n on both fragments", n1.ID, n2.ID)
	}
}

// TestOpenAIStream_ReasoningContent covers the MiniMax/DeepSeek-style compat
// shape: reasoning arrives as a reasoning_content property on the delta, not
// a typed SDK field. It must stream as DeltaThinking, and JSON null values
// must not emit; the final Response carries only the content text, exactly
// as Complete normalizes the same endpoint's non-stream reply.
func TestOpenAIStream_ReasoningContent(t *testing.T) {
	chunks := []string{
		chunkJSON(`{"role":"assistant","content":null,"reasoning_content":"I should greet the user."}`, "", ""),
		chunkJSON(`{"content":null,"reasoning_content":"Then I answer."}`, "", ""),
		chunkJSON(`{"content":"Hi!","reasoning_content":null}`, "", ""),
		chunkJSON(`{}`, "stop", ""),
		usageChunkJSON(streamUsage),
	}
	sse := newServer(t, sseHandler(nil, chunks...))
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	want := []llmkit.Delta{
		{Kind: llmkit.DeltaThinking, Text: "I should greet the user."},
		{Kind: llmkit.DeltaThinking, Text: "Then I answer."},
		{Kind: llmkit.DeltaText, Text: "Hi!"},
	}
	if !reflect.DeepEqual(want, deltas) {
		t.Fatalf("deltas =\n%+v\nwant\n%+v", deltas, want)
	}
	if resp.Text != "Hi!" || len(resp.ToolCalls) != 0 || resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("final response = %+v, want plain text with StopEndTurn", resp)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 100 {
		t.Fatalf("usage = %+v, want the final chunk's totals", resp.Usage)
	}
}

// TestOpenAIStream_RefusalStreamsNothing covers refusal deltas: nothing is
// emitted while streaming, and the final Response normalizes exactly as
// Complete does (refusal text as Response.Text with StopRefusal).
func TestOpenAIStream_RefusalStreamsNothing(t *testing.T) {
	refusal := "I cannot help with that request."
	chunks := []string{
		chunkJSON(`{"role":"assistant","content":"","refusal":null}`, "", ""),
		chunkJSON(`{"refusal":"`+refusal+`"}`, "", ""),
		chunkJSON(`{}`, "stop", ""),
		usageChunkJSON(`{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}`),
	}
	sse := newServer(t, sseHandler(nil, chunks...))
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(deltas) != 0 {
		t.Fatalf("refusal emitted deltas: %+v", deltas)
	}
	if resp.Text != refusal || resp.StopReason != llmkit.StopRefusal {
		t.Fatalf("final response = %+v, want refusal text with StopRefusal", resp)
	}
}

// TestOpenAIStream_NilFnEqualsComplete: a nil fn makes Stream equivalent to
// Complete for the same exchange.
func TestOpenAIStream_NilFnEqualsComplete(t *testing.T) {
	plain := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(identityCompletionBody))
	})
	sse := newServer(t, sseHandler(nil, identityChunks...))

	completeResp, err := New(streamModel, Options{APIKey: "k", BaseURL: plain}).
		Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	streamResp, deltas, err := stream(t, sse, simpleRequest(), nil)
	if err != nil {
		t.Fatalf("Stream with nil fn: %v", err)
	}
	if len(deltas) != 0 {
		t.Fatalf("nil fn collected deltas: %+v", deltas)
	}
	if !reflect.DeepEqual(completeResp, streamResp) {
		t.Fatalf("Stream(nil fn) differs from Complete:\ncomplete: %+v\nstream:   %+v", completeResp, streamResp)
	}
}

// TestOpenAIStream_FnErrorCancels: a non-nil fn error cancels the stream and
// is returned wrapped as "llmkit: stream fn: ..." with a zero Response.
func TestOpenAIStream_FnErrorCancels(t *testing.T) {
	chunks := []string{
		chunkJSON(`{"content":"one"}`, "", ""),
		chunkJSON(`{"content":"two"}`, "", ""),
		chunkJSON(`{}`, "stop", ""),
		usageChunkJSON(streamUsage),
	}
	sse := newServer(t, sseHandler(nil, chunks...))
	boom := errors.New("boom")
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error {
		return boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the fn error wrapped", err)
	}
	if got := "llmkit: stream fn: boom"; err.Error() != got {
		t.Fatalf("err = %q, want %q", err.Error(), got)
	}
	if len(deltas) != 1 {
		t.Fatalf("fn saw %d deltas, want exactly 1 before the cancel", len(deltas))
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want the zero Response", resp)
	}
}

// TestOpenAIStream_RateLimitOnOpen covers a 429 when the stream opens: the
// error goes through the same normalization as Complete and surfaces as
// ErrRateLimited.
func TestOpenAIStream_RateLimitOnOpen(t *testing.T) {
	sse := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"Rate limit reached for requests","type":"requests","code":"rate_limit_exceeded"}}`))
	})
	resp, _, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, llmkit.ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("err = %v, want an APIError with status 429", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want the zero Response", resp)
	}
}

// TestOpenAIStream_GhostToolCallCompacted covers non-contiguous vendor
// indices (2 after 0): the accumulator places fragments by the raw wire
// index and would leave an empty ghost call between the two real ones. The
// ghost must be compacted away, so the deltas' Delta.Index numbers equal the
// calls' positions in Response.ToolCalls.
func TestOpenAIStream_GhostToolCallCompacted(t *testing.T) {
	startChunk := func(index int64, id, name string) string {
		return chunkJSON(
			fmt.Sprintf(`{"tool_calls":[{"index":%d,"id":%q,"type":"function",`+
				`"function":{"name":%q,"arguments":""}}]}`, index, id, name), "", "")
	}
	chunks := []string{
		startChunk(0, "call_aaa", "get_time"),
		startChunk(2, "call_bbb", "get_weather"),
		argFragChunk(0, `{"now":"12:00"}`),
		argFragChunk(2, `{"city":"Santorini"}`),
		chunkJSON(`{}`, "tool_calls", ""),
		usageChunkJSON(streamUsage),
	}
	sse := newServer(t, sseHandler(nil, chunks...))
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for _, d := range deltas {
		if d.Kind != llmkit.DeltaToolCall {
			t.Fatalf("unexpected delta %+v", d)
		}
		if d.Index != 0 && d.Index != 1 {
			t.Fatalf("delta index %d follows the wire's gap; want the compacted position", d.Index)
		}
	}
	wantCalls := []llmkit.ToolCall{
		{ID: "call_aaa", Name: "get_time", Arguments: []byte(`{"now":"12:00"}`)},
		{ID: "call_bbb", Name: "get_weather", Arguments: []byte(`{"city":"Santorini"}`)},
	}
	if !reflect.DeepEqual(wantCalls, resp.ToolCalls) {
		t.Fatalf("ToolCalls = %+v, want exactly %+v (no ghost entry)", resp.ToolCalls, wantCalls)
	}
	if resp.StopReason != llmkit.StopToolUse {
		t.Fatalf("StopReason = %q, want %q", resp.StopReason, llmkit.StopToolUse)
	}
}

// TestOpenAIStream_TruncatedMidText covers a stream that ends cleanly before
// any finish_reason: the partial text must be an ErrServer-class error with
// a zero Response, not a success — Complete over the same wire fails.
func TestOpenAIStream_TruncatedMidText(t *testing.T) {
	sse := newServer(t, sseTruncated(
		chunkJSON(`{"role":"assistant","content":""}`, "", ""),
		chunkJSON(`{"content":"Hel"}`, "", ""),
	))
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("err = %v, want an ErrServer-class error", err)
	}
	if !strings.Contains(err.Error(), "stream ended before finish_reason") {
		t.Fatalf("err = %q, want the truncation cause", err.Error())
	}
	if len(deltas) == 0 {
		t.Fatalf("fn saw no deltas; the truncated text should still stream before the error")
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want the zero Response", resp)
	}
}

// TestOpenAIStream_TruncatedMidToolArgs covers the same truncation while a
// tool call's arguments are still fragmenting: the partial call must error,
// not surface an unparseable half-call as success.
func TestOpenAIStream_TruncatedMidToolArgs(t *testing.T) {
	sse := newServer(t, sseTruncated(
		chunkJSON(`{"tool_calls":[{"index":0,"id":"call_aaa","type":"function",`+
			`"function":{"name":"get_time","arguments":""}}]}`, "", ""),
		argFragChunk(0, `{"now":`),
	))
	resp, deltas, err := stream(t, sse, simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("err = %v, want an ErrServer-class error", err)
	}
	if !strings.Contains(err.Error(), "stream ended before finish_reason") {
		t.Fatalf("err = %q, want the truncation cause", err.Error())
	}
	if len(deltas) == 0 {
		t.Fatalf("fn saw no deltas; the fragmenting call should still stream before the error")
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Fatalf("resp = %+v, want the zero Response", resp)
	}
}

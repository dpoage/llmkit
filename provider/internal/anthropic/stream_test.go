package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
)

// The SSE event shapes below mirror the vendor SDK's own streaming fixtures
// (anthropic-sdk-go v1.58.0 message_test.go TestAccumulate: message_start,
// content_block_start with an empty typed block, text/thinking/signature/
// input_json deltas, content_block_stop, message_delta with usage,
// message_stop) plus the usage fields documented on MessageStartEvent and
// MessageDeltaEvent. Complete-side bodies are handcrafted so byte-sensitive
// fields (thinking signatures, tool-call argument JSON) match the
// concatenation of the streamed fragments exactly.

// sseEvent is one server-sent event.
type sseEvent struct {
	name string // the SSE event: name
	data string // the JSON payload
}

// sseHandler writes the events as a text/event-stream response.
func sseHandler(events []sseEvent) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, ev := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.name, ev.data)
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}
}

// jsonHandler serves a non-streaming /v1/messages body.
func jsonHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

// newStreamAdapter builds an adapter pointed at base.
func newStreamAdapter(t *testing.T, base string) *anthropicAdapter {
	t.Helper()
	cl := New("claude-test", Options{APIKey: "test-key", BaseURL: base})
	ad, ok := cl.(*anthropicAdapter)
	if !ok {
		t.Fatalf("New returned %T, want *anthropicAdapter", cl)
	}
	return ad
}

// streamReasoningEvents is a thinking block (with signature) followed by a
// text block: the shape extended-thinking replies with.
func streamReasoningEvents() []sseEvent {
	return []sseEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Adding 2+2"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":", so 4."}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-abc"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 4."}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42,"input_tokens":11,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

// reasoningBody is the non-streaming equivalent of streamReasoningEvents.
const reasoningBody = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test",` +
	`"content":[{"type":"thinking","thinking":"Adding 2+2, so 4.","signature":"sig-abc"},` +
	`{"type":"text","text":"The answer is 4."}],` +
	`"stop_reason":"end_turn","stop_sequence":null,` +
	`"usage":{"input_tokens":11,"output_tokens":42,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}`

// streamToolEvents streams a text block and two tool_use blocks whose
// arguments arrive as multiple input_json_delta fragments.
func streamToolEvents() []sseEvent {
	return []sseEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":9,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking "}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather","input":{}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Par"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"is\"}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_time","input":{}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"tz\":\"UTC\"}"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":""}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30,"input_tokens":9}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

// toolBody is the non-streaming equivalent of streamToolEvents; each input
// is byte-identical to the concatenation of that call's partial_json
// fragments (the adapter re-marshals accumulated blocks, so the raw
// arguments must round-trip through the SDK unchanged).
const toolBody = `{"id":"msg_2","type":"message","role":"assistant","model":"claude-test",` +
	`"content":[{"type":"text","text":"Checking "},` +
	`{"type":"tool_use","id":"toolu_a","name":"get_weather","input": {"city":"Paris"}},` +
	`{"type":"tool_use","id":"toolu_b","name":"get_time","input": {"tz":"UTC"}}],` +
	`"stop_reason":"tool_use","stop_sequence":null,` +
	`"usage":{"input_tokens":9,"output_tokens":30}}`

// structuredOutputEvents streams the synthetic structured-output tool's
// call; its argument fragments must surface as text.
func structuredOutputEvents() []sseEvent {
	return []sseEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_s","name":"emit_answer","input":{}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"answer\":"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"42\"}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5,"input_tokens":7}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

const structuredOutputBody = `{"id":"msg_3","type":"message","role":"assistant","model":"claude-test",` +
	`"content":[{"type":"tool_use","id":"toolu_s","name":"emit_answer","input": {"answer":"42"}}],` +
	`"stop_reason":"tool_use","stop_sequence":null,` +
	`"usage":{"input_tokens":7,"output_tokens":5}}`

// schemaRequest asks for schema-constrained output without user tools, the
// combination that activates the synthetic structured-output tool.
func schemaRequest() llmkit.Request {
	return llmkit.Request{
		Messages:       []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
		MaxTokens:      64,
		ResponseSchema: []byte(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`),
	}
}

// TestAnthropicStreamTextThinking covers text and thinking fragments: the
// signature delta emits nothing, and usage lands from message_start plus
// message_delta under the inclusive InputTokens convention.
func TestAnthropicStreamTextThinking(t *testing.T) {
	base := newServer(t, sseHandler(streamReasoningEvents()))
	ad := newStreamAdapter(t, base)

	var got []llmkit.Delta
	resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	want := []llmkit.Delta{
		{Kind: llmkit.DeltaThinking, Text: "Adding 2+2"},
		{Kind: llmkit.DeltaThinking, Text: ", so 4."},
		{Kind: llmkit.DeltaText, Text: "The answer is 4."},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas mismatch:\n got: %#v\nwant: %#v", got, want)
	}
	if resp.Text != "The answer is 4." {
		t.Errorf("Text = %q", resp.Text)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Errorf("StopReason = %v, want StopEndTurn", resp.StopReason)
	}
	// message_start reports input 11 with 3 created + 4 read from cache;
	// message_delta finalizes output 42. InputTokens is the inclusive sum.
	wantUsage := llmkit.Usage{
		InputTokens:              18,
		OutputTokens:             42,
		CacheReadInputTokens:     4,
		CacheCreationInputTokens: 3,
	}
	if resp.Usage != wantUsage {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, wantUsage)
	}
}

// TestAnthropicStreamTwoToolCalls pins tool-call fragmenting: identity
// (Index/ID/Name) goes out at content_block_start, argument fragments
// concatenate to the final Arguments, wire block indices are renumbered to
// ToolCalls positions, and an empty partial_json emits nothing.
func TestAnthropicStreamTwoToolCalls(t *testing.T) {
	base := newServer(t, sseHandler(streamToolEvents()))
	ad := newStreamAdapter(t, base)

	var got []llmkit.Delta
	resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	want := []llmkit.Delta{
		{Kind: llmkit.DeltaText, Text: "Checking "},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "toolu_a", Name: "get_weather"},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "toolu_a", Name: "get_weather", Arguments: `{"city":"Par`},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "toolu_a", Name: "get_weather", Arguments: `is"}`},
		{Kind: llmkit.DeltaToolCall, Index: 1, ID: "toolu_b", Name: "get_time"},
		{Kind: llmkit.DeltaToolCall, Index: 1, ID: "toolu_b", Name: "get_time", Arguments: `{"tz":"UTC"}`},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas mismatch:\n got: %#v\nwant: %#v", got, want)
	}
	if resp.StopReason != llmkit.StopToolUse {
		t.Errorf("StopReason = %v, want StopToolUse", resp.StopReason)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

// TestAnthropicStreamMatchesComplete proves the contract's identity clause:
// the same wire exchange fed to Complete (non-stream JSON) and to Stream
// (SSE) yields equal Responses — equal everywhere except Block.Raw, which
// holds the provider's wire block: the stream path's accumulator
// re-serializes each accumulated block through the SDK's content-block
// union, compacting JSON and materializing absent variant fields, so Raw
// bytes legitimately differ while carrying the same data. The payload Raw
// exists to protect — verbatim thinking text and signature — is asserted
// by decoding both sides.
func TestAnthropicStreamMatchesComplete(t *testing.T) {
	tests := []struct {
		name   string
		body   string
		events []sseEvent
		req    llmkit.Request
	}{
		{
			name:   "thinking_and_text",
			body:   reasoningBody,
			events: streamReasoningEvents(),
			req:    simpleRequest(),
		},
		{
			name:   "two_tool_calls",
			body:   toolBody,
			events: streamToolEvents(),
			req:    simpleRequest(),
		},
		{
			name:   "structured_output",
			body:   structuredOutputBody,
			events: structuredOutputEvents(),
			req:    schemaRequest(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := newServer(t, jsonHandler(tt.body))
			completeResp, err := newStreamAdapter(t, base).Complete(t.Context(), tt.req)
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}

			sse := newServer(t, sseHandler(tt.events))
			var got []llmkit.Delta
			streamResp, err := newStreamAdapter(t, sse).Stream(t.Context(), tt.req, func(d llmkit.Delta) error {
				got = append(got, d)
				return nil
			})
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}

			if !reflect.DeepEqual(stripRaw(completeResp), stripRaw(streamResp)) {
				t.Fatalf("Response identity broken:\ncomplete: %#v\nstream:   %#v", completeResp, streamResp)
			}
			assertThinkingRawDecodedEqual(t, completeResp, streamResp)

			// For the structured-output case the synthetic tool's fragments
			// must surface as text, never as tool-call fragments, and must
			// join to the Response.Text finalize produces.
			if tt.name == "structured_output" {
				want := []llmkit.Delta{
					{Kind: llmkit.DeltaText, Text: `{"answer":`},
					{Kind: llmkit.DeltaText, Text: `"42"}`},
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("synthetic fragments = %#v, want %#v", got, want)
				}
			}
		})
	}
}

// TestAnthropicStreamViaHelper checks the root llmkit.Stream dispatch: the
// adapter satisfies StreamingClient, so the helper must call it (observed by
// the fragments reaching fn) rather than synthesize from Complete.
func TestAnthropicStreamViaHelper(t *testing.T) {
	base := newServer(t, sseHandler(streamReasoningEvents()))
	cl := New("claude-test", Options{APIKey: "test-key", BaseURL: base})
	if _, ok := cl.(llmkit.StreamingClient); !ok {
		t.Fatalf("adapter does not satisfy llmkit.StreamingClient")
	}

	calls := 0
	resp, err := llmkit.Stream(t.Context(), cl, simpleRequest(), func(d llmkit.Delta) error {
		calls++
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if calls != 3 {
		t.Errorf("fn called %d times, want 3 (2 thinking + 1 text)", calls)
	}
	if resp.Text != "The answer is 4." {
		t.Errorf("Text = %q", resp.Text)
	}
}

// TestAnthropicStreamNilFn pins the nil-fn clause: Stream behaves like
// Complete and returns its Response untouched.
func TestAnthropicStreamNilFn(t *testing.T) {
	base := newServer(t, sseHandler(streamReasoningEvents()))
	ad := newStreamAdapter(t, base)
	streamResp, err := ad.Stream(t.Context(), simpleRequest(), nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	cbase := newServer(t, jsonHandler(reasoningBody))
	completeResp, err := newStreamAdapter(t, cbase).Complete(t.Context(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !reflect.DeepEqual(stripRaw(completeResp), stripRaw(streamResp)) {
		t.Fatalf("nil fn broke identity:\ncomplete: %#v\nstream:   %#v", completeResp, streamResp)
	}
	assertThinkingRawDecodedEqual(t, completeResp, streamResp)
}

// stripRaw returns resp with every Block.Raw cleared: the only field the
// Complete and Stream paths cannot share byte-for-byte (see
// TestAnthropicStreamMatchesComplete).
func stripRaw(resp llmkit.Response) llmkit.Response {
	for i := range resp.Blocks {
		resp.Blocks[i].Raw = nil
	}
	return resp
}

// assertThinkingRawDecodedEqual proves the stream path's thinking Raw still
// decodes to the verbatim thinking text and signature that
// anthropicThinkingBlock re-emits on the next request — the property Raw
// exists to guarantee.
func assertThinkingRawDecodedEqual(t *testing.T, complete, stream llmkit.Response) {
	t.Helper()
	type thinkingRaw struct {
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
	}
	decoded := func(resp llmkit.Response) (thinkingRaw, bool) {
		for _, b := range resp.Blocks {
			if b.Kind != llmkit.BlockThinking || len(b.Raw) == 0 {
				continue
			}
			var tr thinkingRaw
			if err := json.Unmarshal(b.Raw, &tr); err != nil {
				t.Fatalf("decoding thinking Raw: %v", err)
			}
			return tr, true
		}
		return thinkingRaw{}, false
	}
	cRaw, cOK := decoded(complete)
	sRaw, sOK := decoded(stream)
	if cOK != sOK {
		t.Fatalf("thinking block presence mismatch: complete=%v stream=%v", cOK, sOK)
	}
	if cOK && cRaw != sRaw {
		t.Errorf("thinking Raw payload diverged:\ncomplete: %#v\nstream:   %#v", cRaw, sRaw)
	}
}

// TestAnthropicStreamFnErrorCancels pins the cancellation clause: a non-nil
// fn error stops delivery, cancels the underlying stream, and is returned
// wrapped with no Response.
func TestAnthropicStreamFnErrorCancels(t *testing.T) {
	handlerDone := make(chan struct{})
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		sseHandler([]sseEvent{
			{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"second"}}`},
		})(w, r)
		// Hold the response open: only closing the stream (not draining it)
		// can unblock this handler before the fallback timer fires.
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
		close(handlerDone)
	})
	ad := newStreamAdapter(t, base)

	sentinel := errors.New("consumer done")
	var calls int
	resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want wrapped sentinel", err)
	}
	if !strings.Contains(err.Error(), "llmkit: stream fn:") {
		t.Errorf("err = %v, want llmkit: stream fn: wrap", err)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Errorf("resp = %#v, want zero Response", resp)
	}
	if calls != 1 {
		t.Errorf("fn called %d times, want 1 (delivery stops at the error)", calls)
	}
	// The deferred Close must have canceled the request: the handler's
	// context fires well before its 5s fallback.
	select {
	case <-handlerDone:
	case <-time.After(2 * time.Second):
		t.Fatal("stream was not canceled: handler still writing after Stream returned")
	}
}

// TestAnthropicStreamErrorEvent pins mid-stream failure normalization: an
// error event surfaced by the SDK goes through the same normalizeErr path as
// Complete. The event rides the already-committed HTTP 200 response, so
// ClassifyStatus's default branch classifies it ErrInvalidRequest — the same
// outcome a 200-carrying body would produce on the non-streaming path.
func TestAnthropicStreamErrorEvent(t *testing.T) {
	events := []sseEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`},
		{"error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
	}
	base := newServer(t, sseHandler(events))
	ad := newStreamAdapter(t, base)

	var got []llmkit.Delta
	resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err == nil {
		t.Fatal("Stream: want error, got nil")
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T (%v), want *llmkit.APIError", err, err)
	}
	if apiErr.Kind != llmkit.ErrInvalidRequest {
		t.Errorf("Kind = %v, want ErrInvalidRequest (200-carrying error event)", apiErr.Kind)
	}
	if apiErr.Provider != "anthropic" {
		t.Errorf("Provider = %q", apiErr.Provider)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Errorf("resp = %#v, want zero Response", resp)
	}
	// The text delta that arrived before the error was still delivered.
	if len(got) != 1 || got[0].Text != "partial" {
		t.Errorf("deltas = %#v, want the single pre-error fragment", got)
	}
}

// TestAnthropicStreamProtocolViolation routes Accumulate failures through
// the adapter's error normalization: a delta for a block that never started
// is a server-class error, not a panic.
func TestAnthropicStreamProtocolViolation(t *testing.T) {
	events := []sseEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":4,"delta":{"type":"text_delta","text":"orphan"}}`},
	}
	base := newServer(t, sseHandler(events))
	ad := newStreamAdapter(t, base)

	resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error { return nil })
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *llmkit.APIError", err)
	}
	if apiErr.Kind != llmkit.ErrServer {
		t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Errorf("resp = %#v, want zero Response", resp)
	}
}

package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
)

// SSE event shapes mirror the vendor SDK's streaming fixtures
// (anthropic-sdk-go v1.58.0 message_test.go TestAccumulate: message_start,
// content_block_start with an empty typed block, text/thinking/signature/
// input_json deltas, content_block_stop, message_delta with usage,
// message_stop) plus the usage fields documented on MessageStartEvent and
// MessageDeltaEvent. Complete-side bodies are handcrafted so byte-sensitive
// fields (thinking signatures, tool-call argument JSON) match the
// concatenation of the streamed fragments exactly.

type sseEvent struct {
	name string
	data string
}

func sseHandler(events []sseEvent) http.HandlerFunc {
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

func jsonHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

func newStreamAdapter(t *testing.T, base string) *anthropicAdapter {
	t.Helper()
	cl, err := New("claude-test", Options{APIKey: "test-key", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
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
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

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
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

// toolBody is the non-streaming equivalent of streamToolEvents; each input
// is byte-identical to the concatenation of that call's partial_json
// fragments — the adapter re-marshals accumulated blocks, so the raw
// arguments must round-trip through the SDK unchanged.
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
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`},
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

// TestAnthropicStreamTextThinking: signature_delta emits nothing; usage
// merges message_start and message_delta under the inclusive InputTokens
// convention.
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

// TestAnthropicStreamTwoToolCalls: identity (Index/ID/Name) goes out at
// content_block_start, argument fragments concatenate to the final
// Arguments, wire block indices are renumbered to ToolCalls positions, and
// an empty partial_json emits nothing.
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

// TestAnthropicStreamMatchesComplete proves the identity clause: the same
// wire exchange fed to Complete (non-stream JSON) and Stream (SSE) yields
// equal Responses everywhere except Block.Raw. The stream path's
// accumulator re-serializes each accumulated block through the SDK's
// content-block union, so Raw bytes legitimately differ while carrying the
// same data; assertThinkingRawDecodedEqual pins the thinking Raw payloads
// the property exists to protect.
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

			// The synthetic tool's fragments must surface as text, never
			// as tool-call fragments, and join to the Response.Text finalize produces.
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

// TestAnthropicStreamViaHelper: the adapter satisfies StreamingClient, so
// llmkit.Stream must call it (observed by fragments reaching fn) rather
// than synthesize from Complete.
func TestAnthropicStreamViaHelper(t *testing.T) {
	base := newServer(t, sseHandler(streamReasoningEvents()))
	cl, err := New("claude-test", Options{APIKey: "test-key", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
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

// TestAnthropicStreamNilFn: Stream behaves like Complete and returns its
// Response untouched.
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

// stripRaw clears every Block.Raw: the only field Complete and Stream
// paths cannot share byte-for-byte. Blocks is cloned so the caller's
// responses keep their Raw for the decoded-payload assertion.
func stripRaw(resp llmkit.Response) llmkit.Response {
	resp.Blocks = slices.Clone(resp.Blocks)
	for i := range resp.Blocks {
		resp.Blocks[i].Raw = nil
	}
	return resp
}

// assertThinkingRawDecodedEqual proves the stream path's thinking Raws
// still decode to the verbatim thinking text and signatures that
// anthropicThinkingBlock re-emits on the next request. EVERY thinking
// block on both sides is compared, so a corrupted streamed signature
// fails here even when no other field shows it.
func assertThinkingRawDecodedEqual(t *testing.T, complete, stream llmkit.Response) {
	t.Helper()
	type thinkingRaw struct {
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
	}
	decoded := func(resp llmkit.Response) []thinkingRaw {
		var out []thinkingRaw
		for _, b := range resp.Blocks {
			if b.Kind != llmkit.BlockThinking || len(b.Raw) == 0 {
				continue
			}
			var tr thinkingRaw
			if err := json.Unmarshal(b.Raw, &tr); err != nil {
				t.Fatalf("decoding thinking Raw: %v", err)
			}
			out = append(out, tr)
		}
		return out
	}
	cRaw := decoded(complete)
	sRaw := decoded(stream)
	if !reflect.DeepEqual(cRaw, sRaw) {
		t.Errorf("thinking Raw payloads diverged:\ncomplete: %#v\nstream:   %#v", cRaw, sRaw)
	}
}

// TestAnthropicStreamFnErrorCancels: a non-nil fn error stops delivery,
// cancels the underlying stream, and is returned wrapped with no Response.
func TestAnthropicStreamFnErrorCancels(t *testing.T) {
	handlerDone := make(chan struct{})
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		sseHandler([]sseEvent{
			{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"second"}}`},
		})(w, r)
		// Hold the response open: only closing the stream (not draining
		// it) can unblock this handler before the fallback timer fires.
		select {
		case <-r.Context().Done():
		case <-time.After(1 * time.Second):
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

// TestAnthropicStreamTruncated: a connection that closes cleanly before
// message_stop is an error, never a partial success — Complete fails on
// the equivalent truncated body.
func TestAnthropicStreamTruncated(t *testing.T) {
	tests := []struct {
		name   string
		events []sseEvent
	}{
		{
			name: "mid_block",
			events: []sseEvent{
				{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
				{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
				{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}`},
			},
		},
		{
			name: "after_content_block_stop",
			events: []sseEvent{
				{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
				{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
				{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"all of it"}}`},
				{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := newServer(t, sseHandler(tt.events))
			ad := newStreamAdapter(t, base)

			resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error { return nil })
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v, want *llmkit.APIError", err)
			}
			if apiErr.Kind != llmkit.ErrServer {
				t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
			}
			if !strings.Contains(err.Error(), "message_stop") {
				t.Errorf("err = %v, want a message_stop mention", err)
			}
			if !reflect.DeepEqual(resp, llmkit.Response{}) {
				t.Errorf("resp = %#v, want zero Response", resp)
			}
		})
	}
}

// TestAnthropicStreamProtocolViolation: wire-sequence violations route
// through the adapter's error normalization — both the out-of-order index
// the SDK's Accumulate rejects and an input_json_delta aimed at a started
// non-input block (a text block), which Accumulate tolerates silently.
func TestAnthropicStreamProtocolViolation(t *testing.T) {
	tests := []struct {
		name   string
		events []sseEvent
	}{
		{
			name: "delta_for_never_started_block",
			events: []sseEvent{
				{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
				{"content_block_delta", `{"type":"content_block_delta","index":4,"delta":{"type":"text_delta","text":"orphan"}}`},
			},
		},
		{
			name: "input_json_delta_for_text_block",
			events: []sseEvent{
				{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
				{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
				{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}`},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := newServer(t, sseHandler(tt.events))
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
		})
	}
}

// TestAnthropicStreamServerToolDropped: server-side tools (e.g. web_search)
// stream as input_json_delta too, but toResponse drops that block, so no
// fragment is surfaced and the final Response carries no tool call.
func TestAnthropicStreamServerToolDropped(t *testing.T) {
	events := []sseEvent{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"weather\"}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Sunny."}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	base := newServer(t, sseHandler(events))
	ad := newStreamAdapter(t, base)

	var got []llmkit.Delta
	resp, err := ad.Stream(t.Context(), simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "Sunny."}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %#v, want only the text fragment (server tool dropped)", got)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %#v, want none", resp.ToolCalls)
	}
}

// TestAnthropicThinkingRawPadding pins the thinking replay guard's Raw
// handling. Raw that carries nothing the API accepts on replay — nil,
// plain "null" or whitespace-only (padded with the bytes JSON permits
// as space), or a JSON object that DECODES to an empty payload ({},
// {"type":"thinking"}) — must fail with a local ErrInvalidRequest BEFORE
// anything reaches the wire: a padded form would otherwise bypass the
// guard, no-op through json.Unmarshal, and emit
// {"signature":"","thinking":"","type":"thinking"} which the API rejects
// remotely. A payload that decodes to thinking text without a signature
// ({"type":"thinking","thinking":"why"}, with or without an explicit
// empty signature) is rejected too: the API verifies the signature on
// replay (https://platform.claude.com/docs/en/build-with-claude/thinking,
// "Thinking encryption"). Padding JSON does not permit (U+00A0) stays
// malformed, and valid thinking/redacted payloads — padded or not —
// still replay verbatim, including a signature-only block (the display
// "omitted" wire shape: empty thinking, live signature).
func TestAnthropicThinkingRawPadding(t *testing.T) {
	valid := `{"type":"thinking","thinking":"why","signature":"sig-1"}`
	// wsOnly is a payload of exactly the bytes JSON permits as space:
	// space, tab, LF, CR (written via runes so the source carries no
	// escape sequences).
	wsOnly := " " + string(rune(0x09)) + string(rune(0x0A)) + string(rune(0x0D))
	// paddedEmptyObject is {} padded with JSON whitespace.
	paddedEmptyObject := " {} " + string(rune(0x0A))
	tests := []struct {
		name    string
		raw     string // the Raw payload; empty means leave Raw nil
		wantErr bool
		errMsg  string // "empty": missing-Raw; "empty_decode": decoded-to-empty; "malformed": malformed-JSON
	}{
		{name: "nil_raw", wantErr: true, errMsg: "empty"},
		{name: "plain_null", raw: `null`, wantErr: true, errMsg: "empty"},
		{name: "padded_null", raw: " null", wantErr: true, errMsg: "empty"},
		{name: "padded_whitespace_only", raw: wsOnly, wantErr: true, errMsg: "empty"},
		{name: "nbsp_null", raw: "\u00a0null", wantErr: true, errMsg: "malformed"},
		{name: "valid_padded_json", raw: " " + valid + "\n"},
		{name: "empty_object", raw: `{}`, wantErr: true, errMsg: "empty_decode"},
		{name: "padded_empty_object", raw: paddedEmptyObject, wantErr: true, errMsg: "empty_decode"},
		{name: "spaces_in_object", raw: `{  }`, wantErr: true, errMsg: "empty_decode"},
		{name: "typed_empty_thinking", raw: `{"type":"thinking"}`, wantErr: true, errMsg: "empty_decode"},
		{name: "unsigned_thinking", raw: `{"type":"thinking","thinking":"why"}`, wantErr: true, errMsg: "unsigned"},
		{name: "empty_signature", raw: `{"type":"thinking","thinking":"why","signature":""}`, wantErr: true, errMsg: "unsigned"},
		{name: "omitted_mode_signature_only", raw: `{"type":"thinking","thinking":"","signature":"sig-1"}`},
		{name: "redacted_empty_data", raw: `{"type":"redacted_thinking"}`, wantErr: true, errMsg: "empty_decode"},
		{name: "valid_redacted", raw: `{"type":"redacted_thinking","data":"cGF5bG9hZA=="}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hits := 0
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits++
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, mockTextBody("ok", 1, 1))
			})
			ad := newStreamAdapter(t, base)

			thinking := llmkit.Block{Kind: llmkit.BlockThinking, Provider: "anthropic"}
			if tt.raw != "" {
				thinking.Raw = json.RawMessage(tt.raw)
			}
			req := llmkit.Request{
				Messages: []llmkit.Message{
					llmkit.TextMessage(llmkit.RoleUser, "hi"),
					{Role: llmkit.RoleAssistant, Content: []llmkit.Block{thinking}},
				},
				MaxTokens: 64,
			}
			_, err := ad.Complete(t.Context(), req)

			if !tt.wantErr {
				if err != nil {
					t.Fatalf("Complete: %v", err)
				}
				if hits != 1 {
					t.Fatalf("wire hits = %d, want 1 (valid padded thinking must replay)", hits)
				}
				return
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v, want *llmkit.APIError", err)
			}
			if apiErr.Kind != llmkit.ErrInvalidRequest {
				t.Errorf("Kind = %v, want ErrInvalidRequest", apiErr.Kind)
			}
			switch tt.errMsg {
			case "empty":
				if !strings.Contains(apiErr.Message, "Raw is empty") {
					t.Errorf("Message = %q, want the missing-Raw message", apiErr.Message)
				}
			case "empty_decode":
				if !strings.Contains(apiErr.Message, "decodes to an empty payload") {
					t.Errorf("Message = %q, want the decoded-to-empty message", apiErr.Message)
				}
			case "unsigned":
				if !strings.Contains(apiErr.Message, "unsigned") {
					t.Errorf("Message = %q, want the unsigned-thinking message", apiErr.Message)
				}
			case "malformed":
				if !strings.Contains(apiErr.Message, "malformed Raw JSON") {
					t.Errorf("Message = %q, want the malformed-Raw message", apiErr.Message)
				}
			}
			if hits != 0 {
				t.Errorf("wire hits = %d, want 0 — the guard must fire before any wire call", hits)
			}
		})
	}
}

// TestAnthropicStreamErrorEvent asserts that a mid-stream SSE error event
// goes through the same normalizeErr path as Complete: the vendor type in
// the event body classifies in band, so a committed 200 stream no longer
// dumps every such event into ErrInvalidRequest.
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
	if !errors.Is(err, llmkit.ErrOverloaded) {
		t.Errorf("Kind = %v, want ErrOverloaded (in-band overloaded_error on a 200 stream)", apiErr.Kind)
	}
	if apiErr.StatusCode != http.StatusOK {
		t.Errorf("StatusCode = %d, want 200 (the committed stream's status)", apiErr.StatusCode)
	}
	if apiErr.Provider != "anthropic" {
		t.Errorf("Provider = %q", apiErr.Provider)
	}
	if !reflect.DeepEqual(resp, llmkit.Response{}) {
		t.Errorf("resp = %#v, want zero Response", resp)
	}
	if len(got) != 1 || got[0].Text != "partial" {
		t.Errorf("deltas = %#v, want the single pre-error fragment", got)
	}

	// Retry-stage behaviour over this same in-band SSE error
	// (delivered-delta guard; vendor-type classification) is covered by
	// provider.TestConformance_AnthropicStream_DeliveredDeltaForbidsRetry
	// and provider.TestConformance_AnthropicStream_FirstEventClassifiedByType.
}

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
)

func hostOf(t *testing.T, base string) string {
	t.Helper()
	u, err := url.Parse(base)
	if err != nil {
		t.Fatal(err)
	}
	return u.Host
}

func newCompat(t *testing.T, base string) llmkit.Client {
	t.Helper()
	c, err := New(streamModel, Options{APIKey: "k", BaseURL: base, Compatible: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

// plainCompletion serves a chat.completion whose assistant message is msg
// (a JSON object string).
func plainCompletion(msg string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","created":1,"model":%q,`+
			`"choices":[{"index":0,"message":%s,"finish_reason":"stop"}],"usage":%s}`,
			streamID, streamModel, msg, streamUsage)
	}
}

// legs runs one wire exchange through Complete, Stream with a non-nil fn,
// and Stream with a nil fn against a single server (so the compat tag, which
// embeds the host, is the same on all three).
func legs(t *testing.T, compat bool, msg string, chunks []string) (complete, streamFn, streamNil llmkit.Response) {
	t.Helper()
	plain, sse := plainCompletion(msg), sseHandler(nil, chunks...)
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(string(raw), `"stream":true`) {
			sse(w, r)
			return
		}
		plain(w, r)
	})
	c, err := New(streamModel, Options{APIKey: "k", BaseURL: base, Compatible: compat})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	complete, err = c.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	sc := c.(llmkit.StreamingClient)
	streamFn, err = sc.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream(fn): %v", err)
	}
	streamNil, err = sc.Stream(context.Background(), simpleRequest(), nil)
	if err != nil {
		t.Fatalf("Stream(nil): %v", err)
	}
	return complete, streamFn, streamNil
}

func thinkingBlocks(r llmkit.Response) []llmkit.Block {
	var out []llmkit.Block
	for _, b := range r.Blocks {
		if b.Kind == llmkit.BlockThinking {
			out = append(out, b)
		}
	}
	return out
}

func TestReasoningBlock_AllLegs(t *testing.T) {
	const details = `[{"index":0,"text":"hmm","type":"reasoning.text"}]`
	cases := []struct {
		name        string
		extra       string // extra properties of the assistant message/delta, with a leading comma
		chunkExtras []string
		wantRaw     string // "" = no thinking block
		wantText    string
	}{
		{name: "absent"},
		{name: "empty string", extra: `,"reasoning_content":""`, chunkExtras: []string{`,"reasoning_content":""`}},
		{name: "null", extra: `,"reasoning_content":null,"reasoning_details":null`,
			chunkExtras: []string{`,"reasoning_content":null,"reasoning_details":null`}},
		{name: "empty array", extra: `,"reasoning_details":[]`, chunkExtras: []string{`,"reasoning_details":[]`}},
		{name: "both empty", extra: `,"reasoning_content":"","reasoning_details":[]`,
			chunkExtras: []string{`,"reasoning_content":"","reasoning_details":[]`}},
		{name: "content only", extra: `,"reasoning_content":"abc"`,
			chunkExtras: []string{`,"reasoning_content":"a"`, `,"reasoning_content":"bc"`},
			wantRaw:     `{"reasoning_content":"abc"}`, wantText: "abc"},
		{name: "details only", extra: `,"reasoning_details":` + details,
			chunkExtras: []string{`,"reasoning_details":` + details},
			wantRaw:     `{"reasoning_details":` + details + `}`},
		{name: "both", extra: `,"reasoning_content":"abc","reasoning_details":` + details,
			chunkExtras: []string{`,"reasoning_content":"ab","reasoning_details":` + details, `,"reasoning_content":"c"`},
			wantRaw:     `{"reasoning_content":"abc","reasoning_details":` + details + `}`, wantText: "abc"},
	}
	for _, compat := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("compat=%v/%s", compat, tc.name), func(t *testing.T) {
				msg := `{"role":"assistant","content":"answer"` + tc.extra + `}`
				var chunks []string
				for i, e := range tc.chunkExtras {
					d := `{"content":null` + e + `}`
					if i == 0 {
						d = `{"role":"assistant","content":null` + e + `}`
					}
					chunks = append(chunks, chunkJSON(d, "", ""))
				}
				chunks = append(chunks, chunkJSON(`{"content":"answer"}`, "", ""), chunkJSON(`{}`, "stop", ""), usageChunkJSON(streamUsage))
				c, sf, sn := legs(t, compat, msg, chunks)

				for name, r := range map[string]llmkit.Response{"complete": c, "stream-fn": sf, "stream-nil": sn} {
					tb := thinkingBlocks(r)
					if tc.wantRaw == "" {
						if len(tb) != 0 {
							t.Errorf("%s: thinking blocks = %+v, want none", name, tb)
						}
						continue
					}
					if len(tb) != 1 {
						t.Fatalf("%s: thinking blocks = %d, want exactly 1", name, len(tb))
					}
					if r.Blocks[0].Kind != llmkit.BlockThinking || r.Blocks[1].Kind != llmkit.BlockText {
						t.Errorf("%s: block order = %v, want thinking before text", name, r.Blocks)
					}
					b := tb[0]
					if (!compat && b.Provider != "openai") ||
						(compat && !strings.HasPrefix(b.Provider, "openai-compatible:127.0.0.1:")) {
						t.Errorf("%s: Provider = %q", name, b.Provider)
					}
					if string(b.Raw) != tc.wantRaw || b.Text != tc.wantText {
						t.Errorf("%s: block = {Text:%q Raw:%s}, want {Text:%q Raw:%s}", name, b.Text, b.Raw, tc.wantText, tc.wantRaw)
					}
					if r.Text != "answer" {
						t.Errorf("%s: Text = %q", name, r.Text)
					}
				}
				if !reflect.DeepEqual(c, sf) || !reflect.DeepEqual(c, sn) {
					t.Errorf("responses differ:\ncomplete:   %+v\nstream-fn:  %+v\nstream-nil: %+v", c, sf, sn)
				}
			})
		}
	}
}

func TestThinkingTag(t *testing.T) {
	for _, tc := range []struct {
		compat  bool
		baseURL string
		want    string
	}{
		{false, "", "openai"},
		{false, "http://localhost:9/v1", "openai"},
		{true, "https://api.deepseek.com/v1", "openai-compatible:api.deepseek.com"},
		{true, "http://127.0.0.1:8080/v1", "openai-compatible:127.0.0.1:8080"},
	} {
		if got := thinkingTag(tc.compat, tc.baseURL); got != tc.want {
			t.Errorf("thinkingTag(%v, %q) = %q, want %q", tc.compat, tc.baseURL, got, tc.want)
		}
	}
}

// TestOpenAIStream_ResponseIdentityIsCompleteWithReasoning extends the
// Complete/Stream identity to a turn carrying reasoning, text and a tool call.
func TestOpenAIStream_ResponseIdentityIsCompleteWithReasoning(t *testing.T) {
	msg := `{"role":"assistant","content":"Let me check","refusal":null,` +
		`"reasoning_content":"think one. think two.","reasoning_details":[{"type":"reasoning.text","text":"think one. think two.","index":0}],` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"q\":1}"}}]}`
	chunks := []string{
		chunkJSON(`{"role":"assistant","content":"","refusal":null,"reasoning_content":"think one. ","reasoning_details":[{"type":"reasoning.text","text":"think one. ","index":0}]}`, "", ""),
		chunkJSON(`{"reasoning_content":"think two.","reasoning_details":[{"type":"reasoning.text","text":"think two.","index":0}]}`, "", ""),
		chunkJSON(`{"content":"Let me check"}`, "", ""),
		chunkJSON(`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}`, "", ""),
		argFragChunk(0, `{"q":1}`),
		chunkJSON(`{}`, "tool_calls", ""),
		usageChunkJSON(streamUsage),
	}
	c, sf, sn := legs(t, true, msg, chunks)
	if len(thinkingBlocks(c)) != 1 {
		t.Fatalf("complete has no thinking block: %+v", c)
	}
	if !reflect.DeepEqual(c, sf) || !reflect.DeepEqual(c, sn) {
		t.Fatalf("responses differ:\ncomplete:   %+v\nstream-fn:  %+v\nstream-nil: %+v", c, sf, sn)
	}
}

// capture records the decoded body of every request and answers with a
// canned text reply (or an SSE stream when the request asks for one).
type capture struct {
	hits   atomic.Int32
	bodies []map[string]any
}

func (c *capture) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c.hits.Add(1)
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body: %v", err)
		}
		c.bodies = append(c.bodies, body)
		if body["stream"] == true {
			sseHandler(nil, chunkJSON(`{"role":"assistant","content":"ok"}`, "", ""), chunkJSON(`{}`, "stop", ""),
				usageChunkJSON(streamUsage))(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	}
}

func assistantMessages(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	var out []map[string]any
	msgs, _ := body["messages"].([]any)
	for _, m := range msgs {
		mm := m.(map[string]any)
		if mm["role"] == "assistant" {
			out = append(out, mm)
		}
	}
	return out
}

// compatHistory sends hist to a compat adapter whose tag is derived from the
// server host; mk builds the history given the adapter's own tag.
func compatHistory(t *testing.T, mk func(tag string) []llmkit.Message) (bodies []map[string]any, errs [2]error, hits int32) {
	t.Helper()
	var cp capture
	base := newServer(t, cp.handler(t))
	c := newCompat(t, base)
	tag := "openai-compatible:" + hostOf(t, base)
	req := simpleRequest()
	req.Messages = append(mk(tag), llmkit.TextMessage(llmkit.RoleUser, "next"))
	_, errs[0] = c.Complete(context.Background(), req)
	_, errs[1] = c.(llmkit.StreamingClient).Stream(context.Background(), req, func(llmkit.Delta) error { return nil })
	return cp.bodies, errs, cp.hits.Load()
}

func thinkingMsg(tag, raw string, calls ...llmkit.ToolCall) llmkit.Message {
	return llmkit.Message{
		Role:      llmkit.RoleAssistant,
		Content:   []llmkit.Block{{Kind: llmkit.BlockThinking, Provider: tag, Raw: json.RawMessage(raw)}, {Kind: llmkit.BlockText, Text: "t"}},
		ToolCalls: calls,
	}
}

// TestReasoningEmission: an own-tag thinking block's keys ride on its
// assistant message with or without tool calls; the later block wins.
func TestReasoningEmission(t *testing.T) {
	call := llmkit.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{}`)}
	bodies, errs, hits := compatHistory(t, func(tag string) []llmkit.Message {
		two := thinkingMsg(tag, `{"reasoning_content":"first","reasoning_details":[{"a":1}]}`)
		two.Content = append(two.Content[:1:1], llmkit.Block{Kind: llmkit.BlockThinking, Provider: tag, Raw: json.RawMessage(`{"reasoning_content":"second"}`)})
		return []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "q"),
			thinkingMsg(tag, `{"reasoning_content":"with tools"}`, call),
			llmkit.Message{Role: llmkit.RoleToolResult, ToolCallID: "c1", Content: []llmkit.Block{llmkit.Text("r")}},
			thinkingMsg(tag, `{"reasoning_content":"no tools","reasoning_details":[{"x":"y"}]}`),
			two,
		}
	})
	if errs[0] != nil || errs[1] != nil || hits != 2 {
		t.Fatalf("errs=%v hits=%d", errs, hits)
	}
	want := []map[string]any{
		{"reasoning_content": "with tools"},
		{"reasoning_content": "no tools", "reasoning_details": []any{map[string]any{"x": "y"}}},
		{"reasoning_content": "second", "reasoning_details": []any{map[string]any{"a": json.Number("1")}}},
	}
	for leg, body := range bodies {
		am := assistantMessages(t, body)
		if len(am) != len(want) {
			t.Fatalf("leg %d: %d assistant messages", leg, len(am))
		}
		for i, w := range want {
			for _, k := range []string{"reasoning_content", "reasoning_details"} {
				gotV, gotOK := am[i][k]
				wantV, wantOK := w[k]
				if gotOK != wantOK || (wantOK && !jsonEqual(t, gotV, wantV)) {
					t.Errorf("leg %d message %d key %s = %v (present=%v), want %v (present=%v)", leg, i, k, gotV, gotOK, wantV, wantOK)
				}
			}
		}
	}
}

func jsonEqual(t *testing.T, a, b any) bool {
	t.Helper()
	ab, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	var av, bv any
	_ = json.Unmarshal(ab, &av)
	_ = json.Unmarshal(bb, &bv)
	return reflect.DeepEqual(av, bv)
}

// TestReasoningRefusal: an own-tag thinking block whose Raw is not a JSON
// object with an included key is refused before any wire call.
func TestReasoningRefusal(t *testing.T) {
	for _, raw := range []string{``, `{}`, `[]`, `"x"`, `null`, `{bad`, `{"reasoning_content":""}`, `{"reasoning_details":[]}`,
		`{"reasoning_content":null}`, `{"other":"k"}`} {
		t.Run(raw, func(t *testing.T) {
			_, errs, hits := compatHistory(t, func(tag string) []llmkit.Message {
				return []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "q"), thinkingMsg(tag, raw)}
			})
			for leg, err := range errs {
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Errorf("leg %d: err = %v, want ErrInvalidRequest", leg, err)
				}
			}
			if hits != 0 {
				t.Errorf("wire hits = %d, want 0", hits)
			}
		})
	}
}

// TestReasoningForeignDropped: a foreign-tag block, and a compat block
// tagged with another host, emit no reasoning key.
func TestReasoningForeignDropped(t *testing.T) {
	raw := `{"reasoning_content":"leak","reasoning_details":[{"a":1}]}`
	bodies, errs, hits := compatHistory(t, func(tag string) []llmkit.Message {
		return []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "q"),
			thinkingMsg("anthropic", raw),
			thinkingMsg("openai", raw),
			thinkingMsg("openai-compatible:other.example.com", raw),
			thinkingMsg("openai-compatible:other.example.com", `not json`),
		}
	})
	if errs[0] != nil || errs[1] != nil || hits != 2 {
		t.Fatalf("errs=%v hits=%d", errs, hits)
	}
	for leg, body := range bodies {
		am := assistantMessages(t, body)
		if len(am) != 4 {
			t.Fatalf("leg %d: %d assistant messages", leg, len(am))
		}
		for i, m := range am {
			for _, k := range []string{"reasoning_content", "reasoning_details"} {
				if _, ok := m[k]; ok {
					t.Errorf("leg %d message %d carries %s", leg, i, k)
				}
			}
		}
	}
}

// openRouterDetails is the OpenRouter reasoning_details shape
// (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) plus
// one key the adapter has never heard of.
const openRouterDetails = `[` +
	`{"type":"reasoning.summary","summary":"Plan the answer.","id":"rs-1","format":"anthropic-claude-v1","index":0},` +
	`{"type":"reasoning.text","text":"Let me think carefully.","signature":"sig-abc==","id":"rt-1","format":"anthropic-claude-v1","index":1,"x_vendor":{"n":[1,2.50],"s":"<&>"}},` +
	`{"type":"reasoning.encrypted","data":"ZW5jcnlwdGVk","id":"re-1","format":"openai-responses-v1","index":2}]`

func decodeJSON(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("decode %q: %v", s, err)
	}
	return v
}

func detailsOf(t *testing.T, r llmkit.Response) any {
	t.Helper()
	tb := thinkingBlocks(r)
	if len(tb) != 1 {
		t.Fatalf("thinking blocks = %d, want 1", len(tb))
	}
	var raw struct {
		Details json.RawMessage `json:"reasoning_details"`
	}
	if err := json.Unmarshal(tb[0].Raw, &raw); err != nil {
		t.Fatalf("Raw %s: %v", tb[0].Raw, err)
	}
	return decodeJSON(t, string(raw.Details))
}

// TestReasoningDetails_OpenRouterRoundTrip: reasoning_details from Complete
// reaches the next request meaning-equal, and the same content streamed as
// fragments merges to the identical array.
func TestReasoningDetails_OpenRouterRoundTrip(t *testing.T) {
	want := decodeJSON(t, openRouterDetails)
	msg := `{"role":"assistant","content":"done","reasoning_details":` + openRouterDetails + `}`
	frag := func(s string) string { return chunkJSON(s, "", "") }
	chunks := []string{
		frag(`{"role":"assistant","content":null,"reasoning_details":[{"type":"reasoning.summary","summary":"Plan ","id":"rs-1","format":"anthropic-claude-v1","index":0}]}`),
		frag(`{"reasoning_details":[{"type":"reasoning.summary","summary":"the answer.","id":"rs-1","format":"anthropic-claude-v1","index":0},` +
			`{"type":"reasoning.text","text":"Let me ","signature":"","id":"rt-1","format":"anthropic-claude-v1","index":1}]}`),
		frag(`{"reasoning_details":[{"type":"reasoning.text","text":"think ","index":1,"id":"rt-1","format":"anthropic-claude-v1"},` +
			`{"type":"reasoning.encrypted","data":"ZW5j","id":"re-1","format":"openai-responses-v1","index":2}]}`),
		frag(`{"reasoning_details":[{"type":"reasoning.text","text":"carefully.","signature":"sig-abc==","index":1,"id":"rt-1","format":"anthropic-claude-v1","x_vendor":{"n":[1,2.50],"s":"<&>"}},` +
			`{"type":"reasoning.encrypted","data":"cnlwdGVk","id":"re-1","format":"openai-responses-v1","index":2}]}`),
		frag(`{"content":"done"}`),
		chunkJSON(`{}`, "stop", ""),
		usageChunkJSON(streamUsage),
	}
	c, sf, sn := legs(t, true, msg, chunks)
	for name, r := range map[string]llmkit.Response{"complete": c, "stream-fn": sf, "stream-nil": sn} {
		if got := detailsOf(t, r); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: details = %v\nwant %v", name, got, want)
		}
	}

	var cp capture
	base := newServer(t, cp.handler(t))
	cl := newCompat(t, base)
	req := simpleRequest()
	req.Messages = []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "q")}
	// Re-tag the response blocks for this server's host: the tag embeds it.
	hist := c.Message()
	hist.Content = append([]llmkit.Block(nil), hist.Content...)
	hist.Content[0].Provider = "openai-compatible:" + hostOf(t, base)
	req.Messages = append(req.Messages, hist, llmkit.TextMessage(llmkit.RoleUser, "next"))
	if _, err := cl.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	am := assistantMessages(t, cp.bodies[0])
	if len(am) != 1 || !reflect.DeepEqual(am[0]["reasoning_details"], want) {
		t.Fatalf("request details = %v\nwant %v", am, want)
	}
}

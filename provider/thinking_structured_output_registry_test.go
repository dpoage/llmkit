package provider_test

// The Anthropic vendor constraint under test: "tool use with manual extended
// thinking ... only supports tool_choice auto or none" and any forced
// tool_choice "results in an error"
// (https://platform.claude.com/docs/en/build-with-claude/thinking, section
// "Thinking with tool use"; the page the brief cites as
// docs.claude.com/en/docs/build-with-claude/extended-thinking). The adapter's
// structured-output path forces tool_choice to the synthetic tool, so
// Thinking + ResponseSchema is a guaranteed remote 400 — llmkit refuses it
// pre-wire with ErrInvalidRequest instead. This test drives the production
// construction chain (provider.New) over an httptest stub that counts wire
// hits, for both Complete and the native Stream path, and pins the two legal
// neighbors (ResponseSchema alone still forces tool_choice; Thinking alone
// still forwards).

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

// thinkingSchema is a small, valid JSON Schema for the structured-output
// cases.
var thinkingSchema = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

// anthropicTextBody is a successful text-only non-streaming response.
const anthropicTextBody = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`

// anthropicStreamEvents is a successful text-only SSE stream: the minimal
// event sequence the Anthropic SDK accumulates into a message.
var anthropicStreamEvents = []struct{ name, data string }{
	{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`},
	{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`},
	{"content_block_stop", `{"type":"content_block_stop","index":0}`},
	{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`},
	{"message_stop", `{"type":"message_stop"}`},
}

// wireStub counts every request the client makes and records each decoded
// request body.
type wireStub struct {
	mu     sync.Mutex
	hits   int
	bodies []map[string]any
}

func (s *wireStub) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.hits
}

func (s *wireStub) lastBody() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.bodies) == 0 {
		return nil
	}
	return s.bodies[len(s.bodies)-1]
}

// newAnthropicStub starts an httptest server that counts requests and replies
// with a successful Anthropic response — SSE when sse is true (the Stream
// path), JSON otherwise (Complete).
func newAnthropicStub(t *testing.T, sse bool) (*wireStub, string) {
	t.Helper()
	st := &wireStub{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		st.mu.Lock()
		st.hits++
		body, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		st.bodies = append(st.bodies, m)
		st.mu.Unlock()

		if sse {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			for _, ev := range anthropicStreamEvents {
				_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.name, ev.data)
			}
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, anthropicTextBody)
	}))
	t.Cleanup(srv.Close)
	return st, srv.URL
}

// newAnthropicClient builds the production wrapped client against the stub.
func newAnthropicClient(t *testing.T, base string) llmkit.Client {
	t.Helper()
	cl, err := provider.New(context.Background(), provider.Spec{
		Type:    provider.TypeAnthropic,
		Model:   "claude-test",
		BaseURL: base,
		Secret:  "test-key",
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	return cl
}

// TestAnthropicThinkingStructuredOutputRejectedPreWire pins the pre-wire
// refusal of Thinking + ResponseSchema on the Anthropic adapter — through
// provider.New, over both Complete and the native Stream path — and the two
// legal single-feature requests. The refusal case must make ZERO wire hits;
// each legal case forwards exactly once with the expected wire shape.
func TestAnthropicThinkingStructuredOutputRejectedPreWire(t *testing.T) {
	thinkingOnly := func() llmkit.Request {
		return llmkit.Request{
			System:    "you are a test",
			Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
			MaxTokens: 64,
			Thinking:  &llmkit.ThinkingConfig{BudgetTokens: 1024},
		}
	}
	schemaOnly := func() llmkit.Request {
		req := llmkit.Request{
			System:    "you are a test",
			Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
			MaxTokens: 64,
		}
		req.ResponseSchema = thinkingSchema
		return req
	}

	t.Run("refused_thinking_plus_schema", func(t *testing.T) {
		for _, mode := range []struct {
			name  string
			stub  func(t *testing.T) (*wireStub, llmkit.Client)
			call  func(cl llmkit.Client, req llmkit.Request) error
			depth string
		}{
			{name: "complete", stub: func(t *testing.T) (*wireStub, llmkit.Client) {
				st, base := newAnthropicStub(t, false)
				return st, newAnthropicClient(t, base)
			}, call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := cl.Complete(context.Background(), req)
				return err
			}, depth: "Complete"},
			{name: "stream", stub: func(t *testing.T) (*wireStub, llmkit.Client) {
				st, base := newAnthropicStub(t, true)
				return st, newAnthropicClient(t, base)
			}, call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := llmkit.Stream(context.Background(), cl, req, func(llmkit.Delta) error { return nil })
				return err
			}, depth: "Stream"},
		} {
			t.Run(mode.name, func(t *testing.T) {
				st, cl := mode.stub(t)
				req := thinkingOnly()
				req.ResponseSchema = thinkingSchema
				err := mode.call(cl, req)

				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Fatalf("%s err = %v, want ErrInvalidRequest", mode.depth, err)
				}
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("%s err = %v, want *llmkit.APIError", mode.depth, err)
				}
				for _, field := range []string{"Thinking", "ResponseSchema"} {
					if !strings.Contains(apiErr.Message, field) {
						t.Errorf("%s message = %q, want it to name %q and the vendor tool_choice constraint", mode.depth, apiErr.Message, field)
					}
				}
				if !strings.Contains(apiErr.Message, "tool_choice") {
					t.Errorf("%s message = %q, want it to name the vendor tool_choice constraint", mode.depth, apiErr.Message)
				}
				if n := st.count(); n != 0 {
					t.Errorf("%s wire hits = %d, want 0 — the refusal must fire before any wire call", mode.depth, n)
				}
			})
		}
	})

	t.Run("schema_alone_still_forces", func(t *testing.T) {
		for _, mode := range []struct {
			name string
			sse  bool
			call func(cl llmkit.Client, req llmkit.Request) error
		}{
			{name: "complete", call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := cl.Complete(context.Background(), req)
				return err
			}},
			{name: "stream", sse: true, call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := llmkit.Stream(context.Background(), cl, req, func(llmkit.Delta) error { return nil })
				return err
			}},
		} {
			t.Run(mode.name, func(t *testing.T) {
				st, base := newAnthropicStub(t, mode.sse)
				cl := newAnthropicClient(t, base)
				if mode.sse {
					if _, ok := cl.(llmkit.StreamingClient); !ok {
						t.Fatalf("provider.New client does not implement llmkit.StreamingClient; the Stream subtests would not exercise the native stream")
					}
				}
				if err := mode.call(cl, schemaOnly()); err != nil {
					t.Fatalf("%s: %v", mode.name, err)
				}
				if n := st.count(); n != 1 {
					t.Fatalf("wire hits = %d, want 1", n)
				}
				body := st.lastBody()
				tc, ok := body["tool_choice"].(map[string]any)
				if !ok {
					t.Fatalf("tool_choice missing on the wire: %v", body["tool_choice"])
				}
				if tc["type"] != "tool" || tc["name"] != "emit_answer" {
					t.Errorf("tool_choice = %v, want forced emit_answer", tc)
				}
				if _, present := body["thinking"]; present {
					t.Errorf("thinking = %v, want absent for a schema-only request", body["thinking"])
				}
			})
		}
	})

	t.Run("thinking_alone_still_forwards", func(t *testing.T) {
		for _, mode := range []struct {
			name string
			sse  bool
			call func(cl llmkit.Client, req llmkit.Request) error
		}{
			{name: "complete", call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := cl.Complete(context.Background(), req)
				return err
			}},
			{name: "stream", sse: true, call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := llmkit.Stream(context.Background(), cl, req, func(llmkit.Delta) error { return nil })
				return err
			}},
		} {
			t.Run(mode.name, func(t *testing.T) {
				st, base := newAnthropicStub(t, mode.sse)
				cl := newAnthropicClient(t, base)
				if mode.sse {
					if _, ok := cl.(llmkit.StreamingClient); !ok {
						t.Fatalf("provider.New client does not implement llmkit.StreamingClient; the Stream subtests would not exercise the native stream")
					}
				}
				if err := mode.call(cl, thinkingOnly()); err != nil {
					t.Fatalf("%s: %v", mode.name, err)
				}
				if n := st.count(); n != 1 {
					t.Fatalf("wire hits = %d, want 1", n)
				}
				body := st.lastBody()
				th, ok := body["thinking"].(map[string]any)
				if !ok {
					t.Fatalf("thinking missing on the wire: %v", body["thinking"])
				}
				if th["type"] != "enabled" || th["budget_tokens"] != float64(1024) {
					t.Errorf("thinking = %v, want enabled with budget_tokens 1024", th)
				}
				if _, present := body["tool_choice"]; present {
					t.Errorf("tool_choice = %v, want absent (auto is never serialized)", body["tool_choice"])
				}
			})
		}
	})
}

// newAnthropicClientWithCaps is newAnthropicClient with a Spec.Capabilities
// override, for profiles the model table would not produce.
func newAnthropicClientWithCaps(t *testing.T, base string, caps func(llmkit.Capabilities) llmkit.Capabilities) llmkit.Client {
	t.Helper()
	cl, err := provider.New(context.Background(), provider.Spec{
		Type:         provider.TypeAnthropic,
		Model:        "claude-test",
		BaseURL:      base,
		Secret:       "test-key",
		Capabilities: caps,
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	return cl
}

// TestAnthropicThinkingForcedToolChoiceRejectedPreWire pins the general
// refusal: ANY forced tool_choice combined with Thinking is refused pre-wire
// — whether the force comes from a caller-supplied ToolChoice (required or a
// named tool) or from the synthetic structured-output tool — while auto,
// none, and single-feature requests still forward. The caps-off case pins
// that a profile with Thinking=false drops the config, keeping the
// combination legal.
func TestAnthropicThinkingForcedToolChoiceRejectedPreWire(t *testing.T) {
	baseReq := func() llmkit.Request {
		return llmkit.Request{
			System:    "you are a test",
			Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
			MaxTokens: 64,
			Thinking:  &llmkit.ThinkingConfig{BudgetTokens: 1024},
		}
	}

	scenarios := []struct {
		name      string
		req       func() llmkit.Request
		caps      func(llmkit.Capabilities) llmkit.Capabilities // nil = table default
		wantErr   bool
		errFields []string // substrings the refusal message must carry
		verify    func(t *testing.T, body map[string]any)
	}{
		{
			name: "toolchoice_required_plus_thinking",
			req: func() llmkit.Request {
				r := baseReq()
				r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}
				return r
			},
			wantErr:   true,
			errFields: []string{"Thinking", "ToolChoice", "tool_choice"},
		},
		{
			name: "toolchoice_named_tool_plus_thinking",
			req: func() llmkit.Request {
				r := baseReq()
				r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceTool, Name: "lookup"}
				return r
			},
			wantErr:   true,
			errFields: []string{"Thinking", "ToolChoice", "tool_choice"},
		},
		{
			name: "toolchoice_auto_plus_thinking_forwards",
			req: func() llmkit.Request {
				r := baseReq()
				r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceAuto}
				return r
			},
			verify: func(t *testing.T, body map[string]any) {
				if _, present := body["tool_choice"]; present {
					t.Errorf("tool_choice = %v, want absent (auto is never serialized)", body["tool_choice"])
				}
			},
		},
		{
			name: "toolchoice_none_plus_thinking_forwards",
			req: func() llmkit.Request {
				r := baseReq()
				r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceNone}
				return r
			},
			verify: func(t *testing.T, body map[string]any) {
				tc, ok := body["tool_choice"].(map[string]any)
				if !ok || tc["type"] != "none" {
					t.Errorf("tool_choice = %v, want none", body["tool_choice"])
				}
			},
		},
		{
			name: "toolchoice_required_without_thinking_forwards",
			req: func() llmkit.Request {
				r := baseReq()
				r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}
				r.Thinking = nil
				return r
			},
			verify: func(t *testing.T, body map[string]any) {
				tc, ok := body["tool_choice"].(map[string]any)
				if !ok || tc["type"] != "any" {
					t.Errorf("tool_choice = %v, want any", body["tool_choice"])
				}
				if _, present := body["thinking"]; present {
					t.Errorf("thinking = %v, want absent", body["thinking"])
				}
			},
		},
		{
			name: "thinking_cap_off_still_forces",
			req: func() llmkit.Request {
				r := baseReq()
				r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}
				return r
			},
			caps: func(c llmkit.Capabilities) llmkit.Capabilities {
				c.Thinking = false
				return c
			},
			verify: func(t *testing.T, body map[string]any) {
				tc, ok := body["tool_choice"].(map[string]any)
				if !ok || tc["type"] != "any" {
					t.Errorf("tool_choice = %v, want any", body["tool_choice"])
				}
				if _, present := body["thinking"]; present {
					t.Errorf("thinking = %v, want absent (cap off drops the config)", body["thinking"])
				}
			},
		},
	}

	for _, sc := range scenarios {
		for _, mode := range []struct {
			name string
			sse  bool
			call func(cl llmkit.Client, req llmkit.Request) error
		}{
			{name: "complete", call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := cl.Complete(context.Background(), req)
				return err
			}},
			{name: "stream", sse: true, call: func(cl llmkit.Client, req llmkit.Request) error {
				_, err := llmkit.Stream(context.Background(), cl, req, func(llmkit.Delta) error { return nil })
				return err
			}},
		} {
			t.Run(sc.name+"/"+mode.name, func(t *testing.T) {
				st, base := newAnthropicStub(t, mode.sse)
				var cl llmkit.Client
				if sc.caps != nil {
					cl = newAnthropicClientWithCaps(t, base, sc.caps)
				} else {
					cl = newAnthropicClient(t, base)
				}
				err := mode.call(cl, sc.req())

				if sc.wantErr {
					if !errors.Is(err, llmkit.ErrInvalidRequest) {
						t.Fatalf("%s err = %v, want ErrInvalidRequest", mode.name, err)
					}
					var apiErr *llmkit.APIError
					if !errors.As(err, &apiErr) {
						t.Fatalf("%s err = %v, want *llmkit.APIError", mode.name, err)
					}
					for _, field := range sc.errFields {
						if !strings.Contains(apiErr.Message, field) {
							t.Errorf("message = %q, want it to name %q", apiErr.Message, field)
						}
					}
					if n := st.count(); n != 0 {
						t.Errorf("wire hits = %d, want 0 — the refusal must fire before any wire call", n)
					}
					return
				}
				if err != nil {
					t.Fatalf("%s: %v", mode.name, err)
				}
				if n := st.count(); n != 1 {
					t.Fatalf("wire hits = %d, want 1", n)
				}
				if sc.verify != nil {
					sc.verify(t, st.lastBody())
				}
			})
		}
	}
}

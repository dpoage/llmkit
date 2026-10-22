package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

// reasoningTurn is one scripted Chat Completions assistant turn.
type reasoningTurn struct {
	reasoning string // reasoning_content, delivered in two fragments when streamed
	details   string // reasoning_details JSON array
	content   string
	toolArgs  string // non-empty: one echo tool call
	finish    string
}

func (turn reasoningTurn) complete() string {
	msg := map[string]any{"role": "assistant", "reasoning_content": turn.reasoning,
		"reasoning_details": json.RawMessage(turn.details)}
	if turn.content != "" {
		msg["content"] = turn.content
	}
	if turn.toolArgs != "" {
		msg["tool_calls"] = []any{map[string]any{"id": "call_1", "type": "function",
			"function": map[string]any{"name": "echo", "arguments": turn.toolArgs}}}
	}
	b, _ := json.Marshal(map[string]any{"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": turn.finish}},
		"usage":   map[string]any{"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}})
	return string(b)
}

func (turn reasoningTurn) chunks() []string {
	chunk := func(delta any, finish any) string {
		b, _ := json.Marshal(map[string]any{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}, "usage": nil})
		return string(b)
	}
	half := len(turn.reasoning) / 2
	out := []string{
		chunk(map[string]any{"role": "assistant", "reasoning_content": turn.reasoning[:half],
			"reasoning_details": json.RawMessage(turn.details)}, nil),
		chunk(map[string]any{"reasoning_content": turn.reasoning[half:]}, nil),
	}
	if turn.content != "" {
		out = append(out, chunk(map[string]any{"content": turn.content}, nil))
	}
	if turn.toolArgs != "" {
		out = append(out, chunk(map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": "call_1", "type": "function",
			"function": map[string]any{"name": "echo", "arguments": turn.toolArgs}}}}, nil))
	}
	out = append(out, chunk(map[string]any{}, turn.finish),
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],`+
			`"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}`)
	return out
}

// TestRunner_ReasoningRoundTripsOnEveryLaterRequest: through agent.Runner
// against a Chat Completions server, every request after a thinking turn
// carries tools and every earlier assistant message's reasoning_content and
// reasoning_details — including the turn that stopped at MaxTokens and the
// continuation request it triggers — via both Complete and Stream.
func TestRunner_ReasoningRoundTripsOnEveryLaterRequest(t *testing.T) {
	turns := []reasoningTurn{
		{reasoning: "plan the lookup", details: `[{"type":"reasoning.text","text":"plan the lookup","index":0,"x":{"k":1}}]`,
			toolArgs: `{"v":"a"}`, finish: "tool_calls"},
		{reasoning: "partial answer thoughts", details: `[{"type":"reasoning.summary","summary":"s","index":0}]`,
			content: "The answer is", finish: "length"},
		{reasoning: "wrap up", details: `[{"type":"reasoning.text","text":"wrap up","index":0}]`,
			content: " forty-two.", finish: "stop"},
	}
	for _, streamed := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", streamed), func(t *testing.T) {
			var mu sync.Mutex
			var bodies []map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				var body map[string]any
				if err := json.Unmarshal(raw, &body); err != nil {
					t.Errorf("request body: %v", err)
				}
				mu.Lock()
				bodies = append(bodies, body)
				n := len(bodies)
				mu.Unlock()
				if n > len(turns) {
					t.Errorf("unexpected request %d", n)
					http.Error(w, "too many requests", http.StatusInternalServerError)
					return
				}
				turn := turns[n-1]
				if body["stream"] == true {
					w.Header().Set("Content-Type", "text/event-stream")
					for _, c := range turn.chunks() {
						_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
					}
					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(turn.complete()))
			}))
			t.Cleanup(srv.Close)

			client, err := provider.New(context.Background(), provider.Spec{
				Type: provider.TypeOpenAICompatible, Model: "m", BaseURL: srv.URL, Secret: "k",
			}, provider.Options{})
			if err != nil {
				t.Fatalf("provider.New: %v", err)
			}
			opts := []Option{WithMaxTokens(64)}
			if streamed {
				opts = append(opts, WithHooks(Hooks{Delta: func(context.Context, int, llmkit.Delta) {}}))
			}
			r := NewRunner(client, []Tool{echoTool{name: "echo"}}, "sys", opts...)
			if _, err := r.Run(context.Background(), "task"); err != nil {
				t.Fatalf("Run: %v", err)
			}

			if len(bodies) != 3 {
				t.Fatalf("requests = %d, want 3 (tool turn, truncated turn, continuation)", len(bodies))
			}
			for i, b := range bodies {
				if (b["stream"] == true) != streamed {
					t.Errorf("request %d stream flag = %v, want %v", i+1, b["stream"], streamed)
				}
			}
			for i := 1; i < len(bodies); i++ {
				if tools, _ := bodies[i]["tools"].([]any); len(tools) == 0 {
					t.Errorf("request %d carries no tools", i+1)
				}
				var asst []map[string]any
				for _, m := range bodies[i]["messages"].([]any) {
					if mm := m.(map[string]any); mm["role"] == "assistant" {
						asst = append(asst, mm)
					}
				}
				if len(asst) != i {
					t.Fatalf("request %d has %d assistant messages, want %d", i+1, len(asst), i)
				}
				for j, am := range asst {
					if am["reasoning_content"] != turns[j].reasoning {
						t.Errorf("request %d assistant %d reasoning_content = %v, want %q", i+1, j, am["reasoning_content"], turns[j].reasoning)
					}
					var want any
					_ = json.Unmarshal([]byte(turns[j].details), &want)
					if !reflect.DeepEqual(am["reasoning_details"], want) {
						t.Errorf("request %d assistant %d reasoning_details = %v, want %v", i+1, j, am["reasoning_details"], want)
					}
				}
			}
			if first := strings.TrimSpace(fmt.Sprint(bodies[0]["messages"])); strings.Contains(first, "reasoning_content") {
				t.Errorf("first request carries reasoning before any turn: %s", first)
			}
		})
	}
}

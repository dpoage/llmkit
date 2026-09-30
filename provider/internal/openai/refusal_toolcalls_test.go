package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestRefusalWithToolCalls_StopRefusalKeepsCalls pins the precedence for a
// choice carrying both message.refusal and message.tool_calls: the refusal
// wins the StopReason (StopRefusal) and its text is Response.Text, while
// the tool calls stay populated on the Response — the adapter neither
// drops them nor lets them turn the stop into StopToolUse. Complete and
// Stream, first-party and openai-compatible, all agree.
func TestRefusalWithToolCalls_StopRefusalKeepsCalls(t *testing.T) {
	const refusal = "I cannot help with that request."
	const callID = "call_refused_1"
	const args = `{"location":"Oslo"}`

	completeBody := `{"id":"chatcmpl-r","object":"chat.completion","created":1,"model":"gpt-test",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":` + jsonString(refusal) + `,` +
		`"tool_calls":[{"id":"` + callID + `","type":"function","function":{"name":"get_weather","arguments":` + jsonString(args) + `}}]},` +
		`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}`
	chunks := []string{
		chunkJSON(`{"role":"assistant","content":"","refusal":null}`, "", ""),
		chunkJSON(`{"refusal":`+jsonString(refusal)+`}`, "", ""),
		chunkJSON(`{"tool_calls":[{"index":0,"id":"`+callID+`","type":"function","function":{"name":"get_weather","arguments":""}}]}`, "", ""),
		argFragChunk(0, args),
		chunkJSON(`{}`, "tool_calls", ""),
		usageChunkJSON(`{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}`),
	}

	for _, compat := range []bool{false, true} {
		name := "openai"
		if compat {
			name = "openai-compatible"
		}
		client := func(t *testing.T, base string) llmkit.StreamingClient {
			t.Helper()
			c, err := New("gpt-test", Options{APIKey: "k", BaseURL: base, Compatible: compat})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			return c.(llmkit.StreamingClient)
		}
		check := func(t *testing.T, resp llmkit.Response, err error) {
			t.Helper()
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if resp.StopReason != llmkit.StopRefusal {
				t.Errorf("StopReason = %q, want %q (refusal outranks tool calls)", resp.StopReason, llmkit.StopRefusal)
			}
			if resp.Text != refusal {
				t.Errorf("Text = %q, want the refusal %q", resp.Text, refusal)
			}
			if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != callID || resp.ToolCalls[0].Name != "get_weather" ||
				string(resp.ToolCalls[0].Arguments) != args {
				t.Errorf("ToolCalls = %+v, want the one refused call kept with its arguments", resp.ToolCalls)
			}
		}

		t.Run(name+"/complete", func(t *testing.T) {
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(completeBody))
			})
			resp, err := client(t, base).Complete(context.Background(), simpleRequest())
			check(t, resp, err)
		})
		t.Run(name+"/stream", func(t *testing.T) {
			base := newServer(t, sseHandler(nil, chunks...))
			resp, err := client(t, base).Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
			check(t, resp, err)
		})
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return strings.TrimSpace(string(b))
}

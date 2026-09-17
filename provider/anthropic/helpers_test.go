package anthropic

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dpoage/llmkit"
)

// newServer starts an httptest server whose handler is invoked for every
// request and returns its base URL.
func newServer(t *testing.T, handler http.HandlerFunc) string {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv.URL
}

// simpleRequest is the minimal completion request shared by the adapter tests.
func simpleRequest() llmkit.Request {
	return llmkit.Request{
		System:    "you are a test",
		Messages:  []llmkit.Message{{Role: llmkit.RoleUser, Content: "hi"}},
		MaxTokens: 64,
	}
}

// mockTextBody returns a successful text-only response body in the Anthropic
// wire format.
func mockTextBody(text string, inTok, outTok int64) string {
	b, _ := json.Marshal(map[string]any{
		"id":            "msg_1",
		"type":          "message",
		"role":          "assistant",
		"model":         "claude-test",
		"content":       []any{map[string]any{"type": "text", "text": text}},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage":         map[string]any{"input_tokens": inTok, "output_tokens": outTok},
	})
	return string(b)
}

// mockToolCallBody returns a response in which the model requests a single
// tool call named toolName with the given raw-JSON arguments.
func mockToolCallBody(callID, toolName, argsJSON string, inTok, outTok int64) string {
	var args any
	_ = json.Unmarshal([]byte(argsJSON), &args)
	b, _ := json.Marshal(map[string]any{
		"id":    "msg_2",
		"type":  "message",
		"role":  "assistant",
		"model": "claude-test",
		"content": []any{map[string]any{
			"type":  "tool_use",
			"id":    callID,
			"name":  toolName,
			"input": args,
		}},
		"stop_reason": "tool_use",
		"usage":       map[string]any{"input_tokens": inTok, "output_tokens": outTok},
	})
	return string(b)
}

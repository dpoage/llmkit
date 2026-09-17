package openai

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

// mockTextBody returns a successful text-only response body in the OpenAI
// (chat.completion) wire format.
func mockTextBody(text string, inTok, outTok int64) string {
	b, _ := json.Marshal(map[string]any{
		"id":      "chatcmpl-1",
		"object":  "chat.completion",
		"created": 1,
		"model":   "gpt-test",
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": text},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{"prompt_tokens": inTok, "completion_tokens": outTok, "total_tokens": inTok + outTok},
	})
	return string(b)
}

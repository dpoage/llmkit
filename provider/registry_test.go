package provider

import (
	"context"
	"net/http"
	"testing"
)

func TestNewClient_OpenAICompatibleSerializesToolCalls(t *testing.T) {
	// New wires the serialize stage as the outermost decorator so an
	// openai-compatible endpoint (ParallelToolCalls=false) sees a
	// single tool call by the time Complete returns, with no caller-side
	// wrapping needed.
	multiBody := `{
		"id": "chatcmpl-multi",
		"object": "chat.completion",
		"created": 1,
		"model": "llama3",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{"id": "call_1", "type": "function",
					 "function": {"name": "read_file", "arguments": "{}"}},
					{"id": "call_2", "type": "function",
					 "function": {"name": "list_dir", "arguments": "{}"}}
				]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
	}`
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(multiBody))
	})
	spec := Spec{Type: TypeOpenAICompatible, BaseURL: base, Model: "llama3", Secret: "key"}
	client, err := New(context.Background(), spec, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if client.Capabilities().ParallelToolCalls {
		t.Error("openai-compatible client should report ParallelToolCalls=false (wrapper applied)")
	}
	// The wrapper must truncate the parallel tool calls down to one.
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1 (truncated by the serialize stage)", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_1" {
		t.Errorf("kept tool call ID = %q, want first (call_1)", resp.ToolCalls[0].ID)
	}
}

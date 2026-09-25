package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dpoage/llmkit"
)

// mockStopBody returns a handler serving a minimal message whose
// stop_reason is the given raw wire string.
func mockStopBody(stopReason string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, _ := json.Marshal(map[string]any{
			"id": "msg_stop", "type": "message", "role": "assistant", "model": "claude-test",
			"content":     []any{map[string]any{"type": "text", "text": "..."}},
			"stop_reason": stopReason,
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(b)
	}
}

// TestAnthropicStopReason_Mapping drives the full stop_reason vocabulary
// through the adapter boundary. The load-bearing cases are the catch-all:
// an unknown stop_reason AND pause_turn must map to StopError — a
// server-paused turn is not a natural completion, and reporting StopEndTurn
// made agent loops treat a pause as "conversation over".
func TestAnthropicStopReason_Mapping(t *testing.T) {
	cases := []struct {
		wire string
		want llmkit.StopReason
	}{
		{"end_turn", llmkit.StopEndTurn},
		{"stop_sequence", llmkit.StopEndTurn},
		{"tool_use", llmkit.StopToolUse},
		{"max_tokens", llmkit.StopMaxTokens},
		{"refusal", llmkit.StopRefusal},
		{"pause_turn", llmkit.StopError},
		{"some_reason_a_future_sdk_added", llmkit.StopError},
	}
	for _, tc := range cases {
		t.Run(tc.wire, func(t *testing.T) {
			srv := httptest.NewServer(mockStopBody(tc.wire))
			t.Cleanup(srv.Close)
			client, err := New("claude-test", Options{APIKey: "k", BaseURL: srv.URL})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			resp, err := client.Complete(context.Background(), simpleRequest())
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if resp.StopReason != tc.want {
				t.Errorf("stop_reason %q → StopReason %q, want %q", tc.wire, resp.StopReason, tc.want)
			}
		})
	}
}

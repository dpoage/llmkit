package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

const (
	wireTask    = "TASK: fix the parser in a.go"
	wireSummary = "[conversation summary]\n## Goal\nFix the parser.\n## Progress\nRead a.go."
)

// summaryShapeRequest is a summarized history: [task][summary][tool call][tool result].
func summaryShapeRequest() llmkit.Request {
	return llmkit.Request{
		MaxTokens: 64,
		Tools:     []llmkit.ToolDef{{Name: "mix", Description: "mix", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, wireTask),
			llmkit.TextMessage(llmkit.RoleUser, wireSummary),
			{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "call_1", Name: "mix", Arguments: json.RawMessage(`{"k":"v"}`)}}},
			{Role: llmkit.RoleToolResult, ToolCallID: "call_1", Content: []llmkit.Block{llmkit.Text("RESULT")}},
		},
	}
}

// The task and the summary reach every wire as two user turns, texts intact.
func TestSummaryShape_TwoUserTurnsOnTheWire(t *testing.T) {
	text := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	for _, c := range []struct {
		typ   Type
		key   string
		roles []string
		users string
	}{
		{TypeAnthropic, "messages", []string{"user", "user", "assistant", "user"},
			`[{"role":"user","content":[{"type":"text","text":` + text(wireTask) + `}]},{"role":"user","content":[{"type":"text","text":` + text(wireSummary) + `}]}]`},
		{TypeOpenAI, "messages", []string{"user", "user", "assistant", "tool"},
			`[{"role":"user","content":` + text(wireTask) + `},{"role":"user","content":` + text(wireSummary) + `}]`},
		{TypeOpenAICompatible, "messages", []string{"user", "user", "assistant", "tool"},
			`[{"role":"user","content":` + text(wireTask) + `},{"role":"user","content":` + text(wireSummary) + `}]`},
		{TypeGoogle, "contents", []string{"user", "user", "model", "user"},
			`[{"role":"user","parts":[{"text":` + text(wireTask) + `}]},{"role":"user","parts":[{"text":` + text(wireSummary) + `}]}]`},
	} {
		t.Run(string(c.typ), func(t *testing.T) {
			var body []byte
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				body, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(mockTextBody(string(c.typ), "ok", 1, 1)))
			})
			cl, err := New(context.Background(), Spec{Type: c.typ, Model: "test-model", BaseURL: base, Secret: "test-key"}, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := cl.Complete(context.Background(), summaryShapeRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			var wire map[string]json.RawMessage
			var msgs []map[string]any
			if err := json.Unmarshal(body, &wire); err != nil || json.Unmarshal(wire[c.key], &msgs) != nil {
				t.Fatalf("decode %s: %v", body, err)
			}
			var roles []string
			for _, m := range msgs {
				roles = append(roles, m["role"].(string))
			}
			if !reflect.DeepEqual(roles, c.roles) {
				t.Fatalf("wire roles = %v, want %v\nbody: %s", roles, c.roles, body)
			}
			var want []map[string]any
			if err := json.Unmarshal([]byte(c.users), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(msgs[:2], want) {
				got, _ := json.Marshal(msgs[:2])
				t.Errorf("task and summary on the wire:\n got: %s\nwant: %s", got, c.users)
			}
		})
	}
}

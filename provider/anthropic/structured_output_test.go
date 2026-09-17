package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// schemaForTest is a small, valid JSON Schema used across the
// structured-output tests. It declares a top-level object with a single
// required string property.
var schemaForTest = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

// captureAnthropicBody starts a server that records the request body as a
// decoded map and replies with the given body.
func captureAnthropicBody(t *testing.T, captured *map[string]any, respBody string) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(respBody))
	})
}

// TestStructuredOutput_Anthropic_InjectsForcedTool asserts the wire body
// carries a single synthetic tool (the schema's properties become its
// input_schema) and tool_choice forcing that tool by name.
func TestStructuredOutput_Anthropic_InjectsForcedTool(t *testing.T) {
	var captured map[string]any
	base := captureAnthropicBody(t, &captured, `{"id":"msg","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	adapter := New("claude-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	req.ResponseSchemaName = "emit_answer"
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tools, ok := captured["tools"].([]any)
	if !ok {
		t.Fatalf("tools missing or wrong type: %T", captured["tools"])
	}
	if len(tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1 (synthetic tool)", len(tools))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["name"] != "emit_answer" {
		t.Errorf("synthetic tool name = %v, want emit_answer", tool["name"])
	}
	tc, ok := captured["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice missing or wrong type: %T", captured["tool_choice"])
	}
	if tc["type"] != "tool" {
		t.Errorf("tool_choice.type = %v, want tool", tc["type"])
	}
	if tc["name"] != "emit_answer" {
		t.Errorf("tool_choice.name = %v, want emit_answer", tc["name"])
	}
}

// TestStructuredOutput_Anthropic_DefaultToolName covers the case where the
// caller doesn't set ResponseSchemaName — the adapter must pick
// "emit_answer" as the synthetic tool name.
func TestStructuredOutput_Anthropic_DefaultToolName(t *testing.T) {
	var captured map[string]any
	base := captureAnthropicBody(t, &captured, `{"id":"msg","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	adapter := New("claude-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tools, _ := captured["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1", len(tools))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["name"] != "emit_answer" {
		t.Errorf("default synthetic tool name = %v, want emit_answer", tool["name"])
	}
}

// TestStructuredOutput_Anthropic_SkippedWithUserTools asserts that when the
// caller supplies user tools, the Anthropic adapter does NOT inject a
// synthetic tool (matching the safe fallback used by Gemini).
func TestStructuredOutput_Anthropic_SkippedWithUserTools(t *testing.T) {
	var captured map[string]any
	base := captureAnthropicBody(t, &captured, mockTextBody("ok", 1, 1))
	adapter := New("claude-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	req.Tools = []llmkit.ToolDef{{Name: "user_tool", Parameters: json.RawMessage(`{"type":"object"}`)}}
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tools, _ := captured["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1 (user tool only, no synthetic)", len(tools))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["name"] != "user_tool" {
		t.Errorf("tool name = %v, want user_tool (synthetic was injected!)", tool["name"])
	}
	if _, ok := captured["tool_choice"]; ok {
		t.Errorf("tool_choice present — synthetic tool_choice was injected despite user tools")
	}
}

// TestStructuredOutput_Anthropic_SurfacesForcedToolAsText asserts the
// response path: when the model returns the forced tool_use block, the
// adapter surfaces its `input` JSON as Response.Text (and clears the
// tool-call list), so RunJSON sees JSON text instead of a tool call.
func TestStructuredOutput_Anthropic_SurfacesForcedToolAsText(t *testing.T) {
	forcedArgs := `{"answer":"42"}`
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockToolCallBody("tu_1", "emit_answer", forcedArgs, 5, 7)))
	})
	adapter := New("claude-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	resp, err := adapter.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %d, want 0 (forced tool should be converted to text)", len(resp.ToolCalls))
	}
	if resp.Text != forcedArgs {
		t.Errorf("Text = %q, want %q", resp.Text, forcedArgs)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llmkit.StopEndTurn)
	}
}

// TestStructuredOutput_Anthropic_LeavesOtherToolCallsAlone asserts the
// defensive path: if the model returns a tool call that ISN'T our synthetic
// one, it stays as a tool call.
func TestStructuredOutput_Anthropic_LeavesOtherToolCallsAlone(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockToolCallBody("tu_x", "some_other_tool", `{}`, 1, 1)))
	})
	adapter := New("claude-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	resp, err := adapter.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "some_other_tool" {
		t.Errorf("ToolCalls = %+v, want one call named some_other_tool", resp.ToolCalls)
	}
}

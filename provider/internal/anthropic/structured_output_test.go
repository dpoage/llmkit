package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// schemaForTest is a small, valid JSON Schema: a top-level object with a
// single required string property.
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

// completeSchemaRequest runs a schema-bearing Complete against base and
// returns the response.
func completeSchemaRequest(t *testing.T, base string) llmkit.Response {
	t.Helper()
	ad := New("claude-test", Options{APIKey: "k", BaseURL: base})
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	resp, err := ad.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	return resp
}

// textBlockConcat returns the concatenation of a Response's BlockText
// blocks — the side of the llmkit.Response invariant (Text equals this
// concatenation) the materialization tests pin.
func textBlockConcat(resp llmkit.Response) string {
	var sb strings.Builder
	for _, b := range resp.Blocks {
		if b.Kind == llmkit.BlockText {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// TestStructuredOutput_Anthropic_MaterializesTextBlock pins the Response
// invariant on the structured-output finalize path: the synthetic tool's
// arguments surface as Text AND as a BlockText appended after any existing
// blocks, so Text equals the concatenation of BlockText blocks — the same
// shape the openai and google toResponse emit. Without the appended text
// block, a consumer recording Blocks verbatim (the agent's assistant
// history) drops the text whenever a thinking block precedes it, while
// llmkit.Stream still delivers it.
func TestStructuredOutput_Anthropic_MaterializesTextBlock(t *testing.T) {
	const args = `{"answer":"42"}`
	t.Run("tool_use_only", func(t *testing.T) {
		base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(mockToolCallBody("tu_1", "emit_answer", args, 5, 7)))
		})
		resp := completeSchemaRequest(t, base)
		if len(resp.Blocks) != 1 {
			t.Fatalf("Blocks = %#v, want exactly one text block", resp.Blocks)
		}
		if resp.Blocks[0].Kind != llmkit.BlockText || resp.Blocks[0].Text != resp.Text {
			t.Errorf("Blocks[0] = %+v, want the surfaced text block carrying %q", resp.Blocks[0], resp.Text)
		}
		if got := textBlockConcat(resp); got != resp.Text {
			t.Errorf("concat(BlockText) = %q, want Text %q", got, resp.Text)
		}
	})
	t.Run("thinking_plus_tool_use", func(t *testing.T) {
		const body = `{"id":"msg_4","type":"message","role":"assistant","model":"claude-test",` +
			`"content":[{"type":"thinking","thinking":"Emitting the answer.","signature":"sig-xyz"},` +
			`{"type":"tool_use","id":"toolu_s","name":"emit_answer","input": {"answer":"42"}}],` +
			`"stop_reason":"tool_use","stop_sequence":null,` +
			`"usage":{"input_tokens":7,"output_tokens":5}}`
		base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		})
		resp := completeSchemaRequest(t, base)
		if len(resp.Blocks) != 2 {
			t.Fatalf("Blocks = %#v, want [thinking, text]", resp.Blocks)
		}
		if resp.Blocks[0].Kind != llmkit.BlockThinking {
			t.Errorf("Blocks[0].Kind = %q, want thinking", resp.Blocks[0].Kind)
		}
		if resp.Blocks[1].Kind != llmkit.BlockText || resp.Blocks[1].Text != args {
			t.Errorf("Blocks[1] = %+v, want the text block carrying %q", resp.Blocks[1], args)
		}
		if got := textBlockConcat(resp); got != resp.Text {
			t.Errorf("concat(BlockText) = %q, want Text %q", got, resp.Text)
		}
	})
}

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// breakpointRequest is a representative mid-loop agent request: a system
// prompt, three tools, and a conversation with a completed tool round-trip.
func breakpointRequest() llmkit.Request {
	return llmkit.Request{
		System: "you are a careful bug finder",
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "audit these files"),
			func() llmkit.Message {
				am := llmkit.TextMessage(llmkit.RoleAssistant, "reading")
				am.ToolCalls = []llmkit.ToolCall{
					{ID: "c1", Name: "read_file", Arguments: json.RawMessage(`{"path":"a.go"}`)},
					{ID: "c2", Name: "read_file", Arguments: json.RawMessage(`{"path":"b.go"}`)},
				}
				return am
			}(),
			{Role: llmkit.RoleToolResult, ToolCallID: "c1", Content: []llmkit.Block{{Kind: llmkit.BlockText, Text: "package a"}}},
			{Role: llmkit.RoleToolResult, ToolCallID: "c2", Content: []llmkit.Block{{Kind: llmkit.BlockText, Text: "package b"}}},
		},
		Tools: []llmkit.ToolDef{
			{Name: "read_file", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)},
			{Name: "list_dir", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)},
			{Name: "grep", Parameters: json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string"}}}`)},
		},
		MaxTokens: 64,
	}
}

// cacheControlOf extracts the "cache_control" member of a serialized block (or
// nil when absent).
func cacheControlOf(t *testing.T, block any) map[string]any {
	t.Helper()
	m, ok := block.(map[string]any)
	if !ok {
		t.Fatalf("block is not an object: %T", block)
	}
	cc, _ := m["cache_control"].(map[string]any)
	return cc
}

func TestAnthropic_CacheBreakpointPlacement(t *testing.T) {
	client, err := New("claude-test", Options{APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	a := client.(*anthropicAdapter)
	params, _, err := a.buildParams(breakpointRequest())
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var body struct {
		System   []any `json:"system"`
		Tools    []any `json:"tools"`
		Messages []struct {
			Role    string `json:"role"`
			Content []any  `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal serialized params: %v", err)
	}

	// Tools: only the LAST definition carries the breakpoint (it caches the
	// whole tool block, which serializes ahead of system/messages).
	if n := len(body.Tools); n != 3 {
		t.Fatalf("tools = %d, want 3", n)
	}
	for i, tool := range body.Tools[:2] {
		if cc := cacheControlOf(t, tool); cc != nil {
			t.Errorf("tool %d unexpectedly has cache_control", i)
		}
	}
	if cc := cacheControlOf(t, body.Tools[2]); cc == nil || cc["type"] != "ephemeral" {
		t.Errorf("last tool cache_control = %v, want ephemeral", cc)
	}

	// System: the last (only) block carries the breakpoint.
	if n := len(body.System); n != 1 {
		t.Fatalf("system blocks = %d, want 1", n)
	}
	if cc := cacheControlOf(t, body.System[0]); cc == nil || cc["type"] != "ephemeral" {
		t.Errorf("system cache_control = %v, want ephemeral", cc)
	}

	// Conversation: after coalescing the two tool results into one user turn,
	// messages are [user, assistant, user]. The LAST block of each of the final
	// two messages carries a breakpoint; nothing earlier does.
	if n := len(body.Messages); n != 3 {
		t.Fatalf("messages = %d, want 3 (tool results coalesced)", n)
	}
	for _, block := range body.Messages[0].Content {
		if cc := cacheControlOf(t, block); cc != nil {
			t.Error("first message unexpectedly has cache_control")
		}
	}
	for mi := 1; mi <= 2; mi++ {
		blocks := body.Messages[mi].Content
		for bi, block := range blocks {
			cc := cacheControlOf(t, block)
			if bi == len(blocks)-1 {
				if cc == nil || cc["type"] != "ephemeral" {
					t.Errorf("message %d last block cache_control = %v, want ephemeral", mi, cc)
				}
			} else if cc != nil {
				t.Errorf("message %d block %d unexpectedly has cache_control", mi, bi)
			}
		}
	}

	// Anthropic allows at most 4 breakpoints per request; count every marker in
	// the serialized body to guard against accidental extras.
	if got := strings.Count(string(raw), `"cache_control"`); got != 4 {
		t.Errorf("total cache_control markers = %d, want exactly 4", got)
	}
}

// TestAnthropic_CacheBreakpoints_MinimalRequest covers the loop's first
// iteration: no tools, one user message. Breakpoints degrade gracefully —
// system + the single message — without exceeding the budget or panicking.
func TestAnthropic_CacheBreakpoints_MinimalRequest(t *testing.T) {
	client, err := New("claude-test", Options{APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	a := client.(*anthropicAdapter)
	params, _, err := a.buildParams(llmkit.Request{
		System:    "sys",
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hello")},
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	if got := strings.Count(string(raw), `"cache_control"`); got != 2 {
		t.Errorf("total cache_control markers = %d, want 2 (system + sole message)", got)
	}
}

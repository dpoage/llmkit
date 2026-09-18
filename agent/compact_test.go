package agent

import (
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// bigToolResult builds a tool-result message large enough to be worth pruning.
func bigToolResult(id, content string) llmkit.Message {
	m := llmkit.TextMessage(llmkit.RoleToolResult, content)
	m.ToolCallID = id
	return m
}

// sampleHistory builds a realistic history: a task, then alternating
// assistant turns (with tool calls) and tool results. n is the number of
// tool-result messages; each is a long blob so it is a pruning candidate.
func sampleHistory(n int) ([]llmkit.Message, map[string]string) {
	msgs := []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "summarize these files")}
	names := map[string]string{}
	for i := range n {
		id := "call-" + string(rune('a'+i))
		name := "read_file"
		names[id] = name
		am := llmkit.TextMessage(llmkit.RoleAssistant, "let me read file "+id)
		am.ToolCalls = []llmkit.ToolCall{{ID: id, Name: name, Arguments: []byte(`{"path":"x.go"}`)}}
		msgs = append(msgs,
			am,
			bigToolResult(id, strings.Repeat("line of source code\n", 200)),
		)
	}
	return msgs, names
}

func TestCompactHistory_PrunesOldKeepsRecentK(t *testing.T) {
	msgs, names := sampleHistory(6) // 6 tool results
	recentK := 4

	out, pruned := compactHistory(msgs, recentK, names)
	if pruned != 2 {
		t.Fatalf("pruned = %d, want 2 (the (6-4) oldest results)", pruned)
	}

	// Walk tool-result messages; the first (6-4)=2 must be stubs, the last 4 verbatim.
	var trIdx int
	for _, m := range out {
		if m.Role != llmkit.RoleToolResult {
			continue
		}
		isStub := strings.HasPrefix(m.Text(), "[tool result pruned")
		if trIdx < 2 && !isStub {
			t.Errorf("tool result %d should be a stub, got %q", trIdx, m.Text())
		}
		if trIdx >= 2 && isStub {
			t.Errorf("tool result %d (within recent-%d) should be verbatim, got stub", trIdx, recentK)
		}
		trIdx++
	}
}

func TestCompactHistory_PreservesPairingAndStructure(t *testing.T) {
	msgs, names := sampleHistory(5)
	orig := append([]llmkit.Message(nil), msgs...)

	out, pruned := compactHistory(msgs, 1, names)
	if pruned != 4 {
		t.Fatalf("pruned = %d, want 4 (5 results minus the recent-1 window)", pruned)
	}
	if len(out) != len(orig) {
		t.Fatalf("message count changed: got %d want %d", len(out), len(orig))
	}
	for i := range orig {
		o, n := orig[i], out[i]
		// Role, ToolCallID, IsError, and assistant ToolCalls must be untouched: a
		// provider rejects a tool result whose ID no longer matches its call.
		if o.Role != n.Role {
			t.Errorf("msg %d role changed: %v -> %v", i, o.Role, n.Role)
		}
		if o.ToolCallID != n.ToolCallID {
			t.Errorf("msg %d ToolCallID changed: %q -> %q", i, o.ToolCallID, n.ToolCallID)
		}
		if o.IsError != n.IsError {
			t.Errorf("msg %d IsError changed", i)
		}
		if len(o.ToolCalls) != len(n.ToolCalls) {
			t.Errorf("msg %d ToolCalls count changed", i)
		}
	}
	// The task message (index 0) and every assistant turn must keep their content.
	if out[0].Text() != orig[0].Text() {
		t.Errorf("task message content mutated: %q", out[0].Text())
	}
	for i := range out {
		if out[i].Role == llmkit.RoleAssistant && out[i].Text() != orig[i].Text() {
			t.Errorf("assistant turn %d content mutated (reasoning chain lost)", i)
		}
	}
}

func TestCompactHistory_StubFormat(t *testing.T) {
	msgs := []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "task"),
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "read_file"}}},
		bigToolResult("c1", strings.Repeat("x", 8412)),
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c2", Name: "grep"}}},
		{Role: llmkit.RoleToolResult, ToolCallID: "c2",
			Content: []llmkit.Block{{Kind: llmkit.BlockText, Text: strings.Repeat("y", 500)}}, IsError: true},
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c3", Name: "list_dir"}}},
		bigToolResult("c3", strings.Repeat("z", 100)),
	}
	names := map[string]string{"c1": "read_file", "c2": "grep", "c3": "list_dir"}

	out, pruned := compactHistory(msgs, 1, names) // keep only the last result (c3)
	if pruned != 2 {
		t.Fatalf("pruned = %d, want 2 (c1 and the errored c2; c3 is recent)", pruned)
	}
	if got := out[2].Text(); got != "[tool result pruned to save context: read_file, 8412 bytes]" {
		t.Errorf("c1 stub = %q", got)
	}
	if got := out[4].Text(); got != "[errored tool result pruned to save context: grep, 500 bytes]" {
		t.Errorf("c2 errored stub = %q", got)
	}
	// c3 is within recent-K=1, stays verbatim.
	if out[6].Text() != strings.Repeat("z", 100) {
		t.Error("c3 (recent) should be untouched")
	}
}

func TestCompactHistory_NothingToPruneWhenAllRecent(t *testing.T) {
	msgs, names := sampleHistory(3)
	// recentK >= number of results: nothing falls outside the window.
	out, pruned := compactHistory(msgs, 3, names)
	if pruned != 0 {
		t.Errorf("pruned = %d, want 0 when all results are within recent-K", pruned)
	}
	// Same backing slice returned unchanged (append-only prefix preserved).
	if &out[0] != &msgs[0] {
		t.Error("expected the original slice to be returned untouched")
	}
}

func TestCompactHistory_SkipsAlreadyStubbed(t *testing.T) {
	// A result already at/under stub size must not be re-mutated (that would
	// needlessly invalidate the prompt-cache prefix on a later compaction).
	msgs := []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "task"),
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "read_file"}}},
		{Role: llmkit.RoleToolResult, ToolCallID: "c1",
			Content: []llmkit.Block{{Kind: llmkit.BlockText, Text: "tiny"}}},
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c2", Name: "read_file"}}},
		bigToolResult("c2", strings.Repeat("x", 4000)),
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c3", Name: "read_file"}}},
		bigToolResult("c3", strings.Repeat("y", 4000)),
	}
	names := map[string]string{"c1": "read_file", "c2": "read_file", "c3": "read_file"}

	out, pruned := compactHistory(msgs, 1, names)
	if pruned != 1 {
		t.Fatalf("pruned = %d, want 1 (the large c2 result; tiny c1 is skipped, c3 is recent)", pruned)
	}
	// c1 ("tiny") is shorter than any stub, so it stays as-is.
	if out[2].Text() != "tiny" {
		t.Errorf("tiny result should be left alone, got %q", out[2].Text())
	}
}

func TestEstimateTokens_CountsContentAndToolCalls(t *testing.T) {
	user400 := llmkit.TextMessage(llmkit.RoleUser, strings.Repeat("a", 400)) // 400 bytes
	asst400 := llmkit.TextMessage(llmkit.RoleAssistant, strings.Repeat("b", 400))
	asst400.ToolCalls = []llmkit.ToolCall{{Name: "read_file", Arguments: []byte(strings.Repeat("c", 192))}}
	msgs := []llmkit.Message{user400, asst400}
	// (400 + 400 + len("read_file")=9 + 192) / 4 = 1001/4 = 250
	if got := estimateTokens(msgs); got != 250 {
		t.Errorf("estimateTokens = %d, want 250", got)
	}
}

// TestEstimateTokens_NonTextBlocks verifies the documented fixed per-block
// charge for image/document/thinking blocks: 1024 bytes each (256 tokens at
// bytes/4), on top of text-block bytes and tool-call bytes.
func TestEstimateTokens_NonTextBlocks(t *testing.T) {
	msgs := []llmkit.Message{
		{Role: llmkit.RoleUser, Content: []llmkit.Block{
			{Kind: llmkit.BlockImage, MediaType: "image/png", Data: []byte("pretend-png")},
			{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: []byte("%PDF")},
			{Kind: llmkit.BlockThinking, Provider: "anthropic", Raw: []byte(`{}`)},
			{Kind: llmkit.BlockText, Text: strings.Repeat("a", 100)},
		}},
	}
	// (100 text + 3*1024 non-text) / 4 = 3172/4 = 793
	if got := estimateTokens(msgs); got != 793 {
		t.Errorf("estimateTokens = %d, want 793 (3 fixed non-text blocks + 100 text bytes)", got)
	}
}

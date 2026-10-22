package agent

import (
	"context"
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

// TestEstimateTokens_NonTextBlocks verifies the per-block charge for
// image/document/thinking blocks, on top of text-block bytes and tool-call
// bytes: image = imageEstimateTokens tokens (E2), document = a flat 1024
// bytes (unchanged), thinking = max(len(Text), len(Raw)) bytes (E1).
func TestEstimateTokens_NonTextBlocks(t *testing.T) {
	msgs := []llmkit.Message{
		{Role: llmkit.RoleUser, Content: []llmkit.Block{
			{Kind: llmkit.BlockImage, MediaType: "image/png", Data: []byte("pretend-png")},
			{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: []byte("%PDF")},
			{Kind: llmkit.BlockThinking, Provider: "anthropic", Raw: []byte(`{}`)},
			{Kind: llmkit.BlockText, Text: strings.Repeat("a", 100)},
		}},
	}
	// Changed rows, cite E2 (image: 1568 tokens = 6272 bytes, not 1024) and E1
	// (thinking: len(Raw)=2 bytes, not 1024); the document row is unchanged.
	// (100 text + 6272 image + 1024 document + 2 thinking) / 4 = 7398/4 = 1849
	if got := estimateTokens(msgs); got != 1849 {
		t.Errorf("estimateTokens = %d, want 1849 (image 1568 tokens + 1024-byte document + 2-byte thinking + 100 text bytes)", got)
	}
}

// TestEstimateTokens_ThinkingCountedOnce is E1: a 100 KB thinking block
// estimates 25,000 tokens (within [25,000, 26,000]) wherever its bytes live,
// including the Anthropic shape that carries them in Text and Raw at once.
func TestEstimateTokens_ThinkingCountedOnce(t *testing.T) {
	kb100 := strings.Repeat("r", 100_000)
	cases := []struct {
		name string
		blk  llmkit.Block
	}{
		{"text-only", llmkit.Block{Kind: llmkit.BlockThinking, Provider: "google", Text: kb100}},
		{"raw-only", llmkit.Block{Kind: llmkit.BlockThinking, Provider: "anthropic", Raw: []byte(kb100)}},
		{"both", llmkit.Block{Kind: llmkit.BlockThinking, Provider: "anthropic", Text: kb100, Raw: []byte(kb100)}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			msgs := []llmkit.Message{{Role: llmkit.RoleAssistant, Content: []llmkit.Block{c.blk}}}
			if got := estimateTokens(msgs); got < 25_000 || got > 26_000 {
				t.Errorf("estimateTokens = %d, want within [25000, 26000]", got)
			}
		})
	}
}

// TestEstimateTokens_ImageBounded is E2: an image block estimates exactly
// imageEstimateTokens (1,568) whatever its byte size, inline or URL; the
// estimator adds no per-message framing, so a lone image message is exactly
// 1,568 (within [1,568, 1,600]).
func TestEstimateTokens_ImageBounded(t *testing.T) {
	cases := []struct {
		name string
		blk  llmkit.Block
	}{
		{"11-byte", llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: []byte("pretend-png")}},
		{"2MiB", llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: make([]byte, 2<<20)}},
		{"url", llmkit.Block{Kind: llmkit.BlockImage, URL: "https://example.com/a.png"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			msgs := []llmkit.Message{{Role: llmkit.RoleUser, Content: []llmkit.Block{c.blk}}}
			got := estimateTokens(msgs)
			if got < 1568 || got > 1600 {
				t.Errorf("estimateTokens = %d, want within [1568, 1600]", got)
			}
			if got != imageEstimateTokens {
				t.Errorf("estimateTokens = %d, want exactly %d", got, imageEstimateTokens)
			}
		})
	}
}

func toolResultWith(id string, blocks ...llmkit.Block) llmkit.Message {
	return llmkit.Message{Role: llmkit.RoleToolResult, ToolCallID: id, Content: blocks}
}

// TestCompactHistory_StubsCaptionedImage is E3: an old tool result of a short
// caption plus a 2 MiB image is stubbed by its total size, and the last
// compactRecentToolResults results stay verbatim. A small captioned image
// (smaller than its stub) is left alone.
func TestCompactHistory_StubsCaptionedImage(t *testing.T) {
	bigImg := llmkit.Image("image/png", make([]byte, 2<<20))
	msgs := []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "task")}
	names := map[string]string{}
	add := func(id string, m llmkit.Message) {
		names[id] = "snap"
		msgs = append(msgs, llmkit.Message{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: id, Name: "snap"}}}, m)
	}
	add("old", toolResultWith("old", llmkit.Text("cap"), bigImg))
	add("small", toolResultWith("small", llmkit.Text("cap"), llmkit.Image("image/png", []byte("tiny"))))
	for i := range compactRecentToolResults {
		id := "recent" + string(rune('a'+i))
		add(id, toolResultWith(id, llmkit.Text("cap"), bigImg))
	}

	out, pruned := compactHistory(msgs, compactRecentToolResults, names)
	if pruned != 1 {
		t.Fatalf("pruned = %d, want 1 (the old captioned image; the small one is under stub size; the last %d are recent)", pruned, compactRecentToolResults)
	}
	byID := map[string]llmkit.Message{}
	for _, m := range out {
		if m.Role == llmkit.RoleToolResult {
			byID[m.ToolCallID] = m
		}
	}
	want := "[tool result pruned to save context: snap, 2097155 bytes]"
	if got := byID["old"].Text(); got != want || len(byID["old"].Content) != 1 {
		t.Errorf("old stub = %q (%d blocks), want %q alone", got, len(byID["old"].Content), want)
	}
	if len(byID["small"].Content) != 2 {
		t.Errorf("small captioned image was stubbed: %+v", byID["small"].Content)
	}
	for i := range compactRecentToolResults {
		m := byID["recent"+string(rune('a'+i))]
		if len(m.Content) != 2 || m.Content[1].Kind != llmkit.BlockImage {
			t.Errorf("recent result %d was not kept verbatim: %+v", i, m.Content)
		}
	}
}

// TestCompactHistory_SpillStubNamesRef is E4: the stub of an old spilled result
// names the spill ref and the original size the marker recorded.
func TestCompactHistory_SpillStubNamesRef(t *testing.T) {
	spilled := spillMarker("spill://abc123", 90_000) + "\n" + strings.Repeat("p", 2048)
	msgs := []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "task"),
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "dump"}}},
		bigToolResult("c1", spilled),
		{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c2", Name: "dump"}}},
		bigToolResult("c2", strings.Repeat("y", 4000)),
	}
	out, pruned := compactHistory(msgs, 1, map[string]string{"c1": "dump", "c2": "dump"})
	if pruned != 1 {
		t.Fatalf("pruned = %d, want 1", pruned)
	}
	want := "[tool result pruned to save context: dump, 90000 bytes; full output spilled: spill://abc123]"
	if got := out[2].Text(); got != want {
		t.Errorf("spilled stub = %q, want %q", got, want)
	}
}

// TestMaybeCompact_EstimatesRequestCopy is E5: a history carrying six 2 MiB
// image tool results is estimated as the request copy. On a client without
// ToolResultMedia the images go out as marker text, so HistoryTokenBudget=5000
// is not crossed; on one with it each image is 1,568 tokens and the budget is.
func TestMaybeCompact_EstimatesRequestCopy(t *testing.T) {
	img := llmkit.Image("image/png", make([]byte, 2<<20))
	msgs := []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "task")}
	names := map[string]string{}
	for i := range 6 {
		id := "c" + string(rune('a'+i))
		names[id] = "snap"
		msgs = append(msgs,
			llmkit.Message{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: id, Name: "snap"}}},
			toolResultWith(id, img))
	}
	const budget = 5000

	run := func(media bool) (est int64, out []llmkit.Message, threshold int64, events []llmkit.CompactionEvent) {
		fc := newFakeClient()
		fc.caps.ToolResultMedia = media
		r := NewRunner(fc, nil, "sys", WithLimits(Limits{HistoryTokenBudget: budget}))
		wired, _ := r.wireMessages(msgs)
		em := runEmitter{tr: NewTranscript(), obs: llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
			if ev.Compaction != nil {
				events = append(events, *ev.Compaction)
			}
		})}
		out, threshold = r.maybeCompact(context.Background(), em, msgs, budget, names, 7)
		return estimateTokens(wired), out, threshold, events
	}

	t.Run("degraded-media-false", func(t *testing.T) {
		est, out, threshold, events := run(false)
		if est > 500 {
			t.Errorf("request-copy estimate = %d, want marker text only (<= 500)", est)
		}
		if &out[0] != &msgs[0] || threshold != budget || len(events) != 0 {
			t.Errorf("compaction tripped on degraded media: threshold %d, %d events", threshold, len(events))
		}
	})
	t.Run("media-true", func(t *testing.T) {
		est, out, threshold, events := run(true)
		// 6 images plus the task and tool-call text (28 bytes = 7 tokens).
		if lo, hi := int64(6*imageEstimateTokens), int64(6*imageEstimateTokens+50); est < lo || est > hi {
			t.Errorf("request-copy estimate = %d, want within [%d, %d] (6 images)", est, lo, hi)
		}
		if &out[0] == &msgs[0] || threshold != budget*compactRearmFactor || len(events) != 1 {
			t.Fatalf("compaction did not fire: threshold %d, %d events", threshold, len(events))
		}
		if events[0].BeforeTokens != est || events[0].Pruned != 2 {
			t.Errorf("event = %+v, want before %d, 2 pruned", events[0], est)
		}
		// Two of six results are stubbed; the four recent images stay.
		if lo, hi := int64(4*imageEstimateTokens), int64(4*imageEstimateTokens+50); events[0].AfterTokens < lo || events[0].AfterTokens > hi {
			t.Errorf("after tokens = %d, want within [%d, %d] (4 images plus 2 stubs)", events[0].AfterTokens, lo, hi)
		}
	})
	t.Run("one-image", func(t *testing.T) {
		fc := newFakeClient()
		fc.caps.ToolResultMedia = true
		r := NewRunner(fc, nil, "sys")
		wired, _ := r.wireMessages([]llmkit.Message{toolResultWith("c", img)})
		if got := estimateTokens(wired); got != imageEstimateTokens {
			t.Errorf("one image estimates %d on a ToolResultMedia client, want %d", got, imageEstimateTokens)
		}
	})
}

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

var (
	attachPNG  = llmkit.Image("image/png", []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})
	attachSpec = llmkit.DocumentURL("https://example.com/spec.pdf", "spec")
)

// assertMessageBlocks pins msg.Content to want block-for-block so order and payload bytes are asserted, including images and documents.
func assertMessageBlocks(t *testing.T, label string, msg llmkit.Message, want ...llmkit.Block) {
	t.Helper()
	if !reflect.DeepEqual(msg.Content, want) {
		t.Fatalf("%s: content = %+v, want %+v", label, msg.Content, want)
	}
}

// TestAttach_RunSeedsTaskTurnWithBlocks pins that the seeded user turn is Text(task) followed by the attached blocks.
func TestAttach_RunSeedsTaskTurnWithBlocks(t *testing.T) {
	const task = "describe the diagram"
	fc := newFakeClient(textResp("done", 5, 5))
	r := NewRunner(fc, nil, "sys")

	if _, err := r.Run(context.Background(), task, Attach(attachPNG)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fc.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(fc.requests))
	}
	msgs := fc.requests[0].Messages
	if len(msgs) != 1 {
		t.Fatalf("seed messages = %d, want 1", len(msgs))
	}
	seed := msgs[0]
	if seed.Role != llmkit.RoleUser {
		t.Errorf("seed role = %v, want user", seed.Role)
	}
	assertMessageBlocks(t, "seed", seed, llmkit.Text(task), attachPNG)
	if seed.Text() != task {
		t.Errorf("seed text = %q, want %q (attachments are not text)", seed.Text(), task)
	}
}

// TestAttach_RunJSON_TextCarriesInstructionAttachmentsStayOnTaskTurn pins that the task-turn TEXT block carries the jsonInstruction suffix and the attachment, while the forced finalization and repair completions carry no attachments.
func TestAttach_RunJSON_TextCarriesInstructionAttachmentsStayOnTaskTurn(t *testing.T) {
	const task = "read the diagram"
	schema := json.RawMessage(`{"type":"object"}`)

	// Iteration cap forces finalization; two unparseable answers force repair.
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"x"}`, 5, 5),
		textResp("the answer is: still not json", 5, 5),
		textResp("nope, no json here either", 5, 5),
	)
	r := NewRunner(fc, nil, "sys", WithLimits(Limits{MaxIterations: 1}))

	var parsed map[string]any
	out, err := r.RunJSON(context.Background(), task, schema, &parsed, Attach(attachPNG))
	if !errors.Is(err, ErrUnparseableOutput) {
		t.Fatalf("RunJSON err = %v, want ErrUnparseableOutput (both answers unparseable)", err)
	}
	if fc.callCount() != 3 {
		t.Fatalf("completions = %d, want 3 (task turn, finalization, repair)", fc.callCount())
	}

	seed := fc.requests[0].Messages[0]
	assertMessageBlocks(t, "seed", seed,
		llmkit.Text(task+"\n\n"+jsonInstruction(schema)), attachPNG)

	final := fc.requests[1]
	finalPrompt := final.Messages[len(final.Messages)-1]
	assertMessageBlocks(t, "finalization prompt", finalPrompt,
		llmkit.Text(finalizationPrompt(schema)))

	repair := fc.requests[2]
	if len(repair.Messages) != 1 {
		t.Fatalf("repair messages = %d, want 1", len(repair.Messages))
	}
	repairMsg := repair.Messages[0]
	if len(repairMsg.Content) != 1 || repairMsg.Content[0].Kind != llmkit.BlockText {
		t.Fatalf("repair prompt content = %+v, want a single text block", repairMsg.Content)
	}
	if !strings.Contains(repairMsg.Text(), task) {
		t.Errorf("repair prompt %q missing the task text", repairMsg.Text())
	}
	if out.FinalText != "nope, no json here either" {
		t.Errorf("FinalText = %q, want the repair completion's text", out.FinalText)
	}
}

// TestAttach_EmptyTaskCarriesBlocksOnly pins that an empty task with attachments omits the Text block entirely.
func TestAttach_EmptyTaskCarriesBlocksOnly(t *testing.T) {
	fc := newFakeClient(textResp("done", 5, 5))
	r := NewRunner(fc, nil, "sys")

	if _, err := r.Run(context.Background(), "", Attach(attachSpec)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	seed := fc.requests[0].Messages[0]
	if seed.Role != llmkit.RoleUser {
		t.Errorf("seed role = %v, want user", seed.Role)
	}
	assertMessageBlocks(t, "seed", seed, attachSpec)
}

// TestAttach_EmptyTaskNoBlocksKeepsLegacyShape pins that an empty task with no attachments still sends a single empty text block — the pre-Attach turn shape.
func TestAttach_EmptyTaskNoBlocksKeepsLegacyShape(t *testing.T) {
	fc := newFakeClient(textResp("done", 5, 5))
	r := NewRunner(fc, nil, "sys")

	if _, err := r.Run(context.Background(), ""); err != nil {
		t.Fatalf("Run: %v", err)
	}
	assertMessageBlocks(t, "seed", fc.requests[0].Messages[0], llmkit.Text(""))
}

// TestAttach_ContinueComposesAfterSeed pins that with Continue the attached turn is appended after the seed conversation element-wise.
func TestAttach_ContinueComposesAfterSeed(t *testing.T) {
	fc := newFakeClient(
		textResp("first answer", 5, 5),
		textResp("second answer", 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	out1, err := r.Run(context.Background(), "first task")
	if err != nil {
		t.Fatalf("first Run: %v", err)
	}
	if _, err := r.Run(context.Background(), "second task", Continue(out1), Attach(attachPNG)); err != nil {
		t.Fatalf("continued Run: %v", err)
	}

	req := fc.requests[1]
	if len(req.Messages) != len(out1.Messages)+1 {
		t.Fatalf("request messages = %d, want %d (seed plus the attached turn)",
			len(req.Messages), len(out1.Messages)+1)
	}
	for i, prev := range out1.Messages {
		sameTurn(t, fmt.Sprintf("seed turn %d", i), req.Messages[i], prev)
	}
	assertMessageBlocks(t, "attached turn", req.Messages[len(out1.Messages)],
		llmkit.Text("second task"), attachPNG)
}

// TestAttach_TranscriptRoundTripsAndReplays pins that the durable JSONL carries the image block (MediaType and Data survive the base64 round-trip) and NewReplayClient replays the same Run+Attach to the same FinalText.
func TestAttach_TranscriptRoundTripsAndReplays(t *testing.T) {
	const task = "describe the diagram"
	dir := t.TempDir()
	fc := newFakeClient(textResp("the answer", 5, 5))
	r := NewRunner(fc, nil, "sys", WithObserver(JSONL(dir, nil)))

	out, err := r.Run(context.Background(), task, Attach(attachPNG))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("transcript dir has %d entries, want 1", len(entries))
	}
	f, err := os.Open(filepath.Join(dir, entries[0].Name()))
	if err != nil {
		t.Fatalf("open transcript: %v", err)
	}
	defer func() { _ = f.Close() }()
	loaded, err := LoadJSONL(f)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}

	var reqMsg *llmkit.Message
	for i := range loaded.Record {
		ev := &loaded.Record[i]
		if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && len(ev.Completion.Request.Messages) > 0 {
			m := ev.Completion.Request.Messages[0]
			reqMsg = &m
			break
		}
	}
	if reqMsg == nil {
		t.Fatal("transcript has no completion event with messages")
	}
	assertMessageBlocks(t, "round-tripped seed", *reqMsg,
		llmkit.Text(task), attachPNG)

	replay, err := NewReplayClient(loaded, loaded.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	out2, err := NewRunner(replay, nil, "sys").Run(context.Background(), task, Attach(attachPNG))
	if err != nil {
		t.Fatalf("replayed Run: %v", err)
	}
	if out2.FinalText != out.FinalText {
		t.Errorf("replayed FinalText = %q, want %q", out2.FinalText, out.FinalText)
	}
}

// TestAttach_OptionsAccumulateInOrder pins that multiple Attach options accumulate in the order given, after the task text.
func TestAttach_OptionsAccumulateInOrder(t *testing.T) {
	second := llmkit.ImageURL("https://example.com/b.png")
	fc := newFakeClient(textResp("done", 5, 5))
	r := NewRunner(fc, nil, "sys")

	if _, err := r.Run(context.Background(), "task", Attach(attachSpec), Attach(attachPNG, second)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	assertMessageBlocks(t, "seed", fc.requests[0].Messages[0],
		llmkit.Text("task"), attachSpec, attachPNG, second)
}

// TestAttach_NudgesStayAttachmentFree pins that the empty-turn and max-tokens continuation nudges carry no attachments — only the task turn does.
func TestAttach_NudgesStayAttachmentFree(t *testing.T) {
	t.Run("empty-turn nudge", func(t *testing.T) {
		fc := newFakeClient(
			thinkOnlyResp("hmm", 5, 5),
			textResp("done", 5, 5),
		)
		r := NewRunner(fc, nil, "sys")
		if _, err := r.Run(context.Background(), "task", Attach(attachPNG)); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if fc.callCount() != 2 {
			t.Fatalf("completions = %d, want 2 (nudged turn then answer)", fc.callCount())
		}
		assertNudgeTurn(t, fc.requests[1])
		assertMessageBlocks(t, "seed", fc.requests[1].Messages[0],
			llmkit.Text("task"), attachPNG)
	})
	t.Run("max-tokens continuation nudge", func(t *testing.T) {
		fc := newFakeClient(
			maxTokensResp("partial answ", 5, 5),
			textResp("er", 5, 5),
		)
		r := NewRunner(fc, nil, "sys")
		if _, err := r.Run(context.Background(), "task", Attach(attachPNG)); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if fc.callCount() != 2 {
			t.Fatalf("completions = %d, want 2 (truncated turn then continuation)", fc.callCount())
		}
		assertNudgeTurn(t, fc.requests[1])
		assertMessageBlocks(t, "seed", fc.requests[1].Messages[0],
			llmkit.Text("task"), attachPNG)
	})
}

// assertNudgeTurn requires req's last message to be a user turn with exactly one text block.
func assertNudgeTurn(t *testing.T, req llmkit.Request) {
	t.Helper()
	msgs := req.Messages
	if len(msgs) == 0 {
		t.Fatal("nudge request carries no messages")
	}
	last := msgs[len(msgs)-1]
	if last.Role != llmkit.RoleUser {
		t.Errorf("nudge turn role = %v, want user", last.Role)
	}
	if len(last.Content) != 1 || last.Content[0].Kind != llmkit.BlockText {
		t.Errorf("nudge turn content = %+v, want exactly one text block", last.Content)
	}
}

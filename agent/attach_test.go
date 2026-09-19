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
	// attachPNG is the inline image the Attach tests seed onto the task turn.
	attachPNG = llmkit.Image("image/png", []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})
	// attachSpec is the URL-referenced document the empty-task tests seed.
	attachSpec = llmkit.DocumentURL("https://example.com/spec.pdf", "spec")
)

// assertMessageBlocks requires msg's content to deep-equal want, block for
// block, so the exact block order and payload bytes are pinned — not just
// the text, which skips images and documents.
func assertMessageBlocks(t *testing.T, label string, msg llmkit.Message, want ...llmkit.Block) {
	t.Helper()
	if !reflect.DeepEqual(msg.Content, want) {
		t.Fatalf("%s: content = %+v, want %+v", label, msg.Content, want)
	}
}

// TestAttach_RunSeedsTaskTurnWithBlocks pins the (a) acceptance case: the
// seeded user turn becomes Text(task) followed by the attached blocks — not
// a reseeded plain text message, and not blocks appended somewhere else.
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

// TestAttach_RunJSON_TextCarriesInstructionAttachmentsStayOnTaskTurn is the
// (b) acceptance case: the TEXT block carries the task prefix and the
// jsonInstruction suffix with the image riding alongside; the forced
// finalization prompt and the repair completion — reached here via an
// iteration cap and two unparseable answers — carry no attachments.
func TestAttach_RunJSON_TextCarriesInstructionAttachmentsStayOnTaskTurn(t *testing.T) {
	const task = "read the diagram"
	schema := json.RawMessage(`{"type":"object"}`)

	// Turn 1 keeps investigating (the missing tool renders an ERROR result),
	// the iteration cap then forces the finalization turn, whose unparseable
	// answer forces the repair completion.
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

	// The injected finalization prompt is a plain text user turn — the
	// attachment must not be duplicated onto it.
	final := fc.requests[1]
	finalPrompt := final.Messages[len(final.Messages)-1]
	assertMessageBlocks(t, "finalization prompt", finalPrompt,
		llmkit.Text(finalizationPrompt(schema)))

	// The repair completion runs against a fresh single-turn history that
	// never sees the attachment.
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

// TestAttach_EmptyTaskCarriesBlocksOnly is the (c) acceptance case: an empty
// task with attachments omits the Text block entirely — no empty text block
// adapters may refuse — so the turn carries the document block alone.
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

// TestAttach_EmptyTaskNoBlocksKeepsLegacyShape pins the observed-and-kept
// behavior an empty task AND no attachments send: the exact pre-Attach turn
// — a single (empty) text block. No new validation was added for this case.
func TestAttach_EmptyTaskNoBlocksKeepsLegacyShape(t *testing.T) {
	fc := newFakeClient(textResp("done", 5, 5))
	r := NewRunner(fc, nil, "sys")

	if _, err := r.Run(context.Background(), ""); err != nil {
		t.Fatalf("Run: %v", err)
	}
	assertMessageBlocks(t, "seed", fc.requests[0].Messages[0], llmkit.Text(""))
}

// TestAttach_ContinueComposesAfterSeed is the (d) acceptance case: with
// Continue, the attached turn is appended AFTER the seed — the request
// carries the prior conversation element-wise plus exactly one new user turn
// holding the task text and the blocks.
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

// TestAttach_TranscriptRoundTripsAndReplays is the (e) acceptance case: the
// autosaved JSONL carries the image block (MediaType and Data survive the
// base64 round-trip through LoadJSONL), and NewReplayClient replays the same
// Run+Attach to the same FinalText.
func TestAttach_TranscriptRoundTripsAndReplays(t *testing.T) {
	const task = "describe the diagram"
	dir := t.TempDir()
	fc := newFakeClient(textResp("the answer", 5, 5))
	r := NewRunner(fc, nil, "sys", WithTranscriptDir(dir))

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
	defer f.Close()
	loaded, err := LoadJSONL(f)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}

	var reqEvent *Event
	for i := range loaded.Events {
		if loaded.Events[i].Kind == EventRequest {
			reqEvent = &loaded.Events[i]
			break
		}
	}
	if reqEvent == nil {
		t.Fatal("transcript has no request event")
	}
	if len(reqEvent.Messages) == 0 {
		t.Fatal("request event carries no messages")
	}
	assertMessageBlocks(t, "round-tripped seed", reqEvent.Messages[0],
		llmkit.Text(task), attachPNG)

	replay, err := NewReplayClient(loaded, llmkit.Capabilities{})
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

// TestAttach_OptionsAccumulateInOrder is the (f) acceptance case: multiple
// Attach options accumulate in the order given, after the task text.
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

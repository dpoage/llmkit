package agent

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

type item struct {
	Path string `json:"path"`
	Note string `json:"note"`
}

// itemWithResolved is the caller schema shape the native-schema tests use —
// it has a boolean "resolved" field the simpler item struct omits, so the
// schema's required=["path","note","resolved"] check actually exercises the
// missing-required-field branch of validateSchema.
type itemWithResolved struct {
	Path     string `json:"path"`
	Note     string `json:"note"`
	Resolved bool   `json:"resolved"`
}

func TestRunJSON_DirectParse(t *testing.T) {
	fc := newFakeClient(textResp(`{"path":"a.go","note":"fix imports"}`, 5, 5))
	r := NewRunner(fc, nil, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "summarize the report", nil, &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "a.go" || got.Note != "fix imports" {
		t.Errorf("parsed = %+v", got)
	}
	if out.Iterations != 1 {
		t.Errorf("Iterations = %d, want 1 (no repair)", out.Iterations)
	}
}

// TestRunJSON_ContinuePreservesPriorConversation is the core continuation
// contract: a second Continue-seeded RunJSON call must NOT reseed the
// conversation — it must append its task onto the FULL history of the prior
// round (including round 1's tool-driven analysis), not just round 1's
// final answer. This is what lets a revision round's feedback land in the
// same conversation the model already analyzed in, instead of asking it
// to re-orient from scratch.
func TestRunJSON_ContinuePreservesPriorConversation(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"orient"}`, 10, 4),
		textResp(`{"path":"a.go","note":"round1"}`, 8, 3),
		textResp(`{"path":"a.go","note":"round2"}`, 8, 3),
	)
	tools := []Tool{echoTool{name: "echo"}}
	r := NewRunner(fc, tools, "sys")

	var got1 item
	out1, err := r.RunJSON(context.Background(), "analyze and report", nil, &got1)
	if err != nil {
		t.Fatalf("round1 RunJSON: %v", err)
	}
	if len(fc.requests) != 2 {
		t.Fatalf("round1 issued %d requests, want 2 (orient + final)", len(fc.requests))
	}
	// Round 1's Outcome must carry the full conversation (seed + tool call +
	// tool result + final assistant answer) for the continued round to build on.
	if want := 4; len(out1.Messages) != want {
		t.Fatalf("round1 Outcome.Messages = %d entries, want %d (user, assistant tool-call, tool-result, assistant final)", len(out1.Messages), want)
	}

	var got2 item
	out2, err := r.RunJSON(context.Background(), "feedback: fix it", nil, &got2, Continue(out1))
	if err != nil {
		t.Fatalf("round2 RunJSON: %v", err)
	}
	// Exactly one more completion was needed: continuation means the model
	// did NOT re-issue the orientation tool call it already made in round 1.
	if len(fc.requests) != 3 {
		t.Fatalf("round2 issued %d total requests, want 3 (round1's 2 plus round2's 1) -- extra requests mean round 2 re-analyzed instead of continuing", len(fc.requests))
	}
	round2Req := fc.requests[2]
	if len(round2Req.Messages) != len(out1.Messages)+1 {
		t.Fatalf("round2 request has %d messages, want %d (round1's full history plus round2's new task turn)", len(round2Req.Messages), len(out1.Messages)+1)
	}
	// Round 2's request must carry round 1's messages verbatim as a prefix --
	// specifically the tool call and tool result, proving the model literally
	// saw its own prior analysis rather than a reseeded conversation.
	for i, want := range out1.Messages {
		got := round2Req.Messages[i]
		if got.Role != want.Role || got.Text() != want.Text() || got.ToolCallID != want.ToolCallID {
			t.Errorf("round2 request message %d = %+v, want round1 history entry %+v", i, got, want)
		}
	}
	sawToolCall, sawToolResult := false, false
	for _, m := range round2Req.Messages {
		if m.Role == llmkit.RoleAssistant && len(m.ToolCalls) > 0 {
			sawToolCall = true
		}
		if m.Role == llmkit.RoleToolResult {
			sawToolResult = true
		}
	}
	if !sawToolCall || !sawToolResult {
		t.Errorf("round2 request missing round1's analysis: sawToolCall=%v sawToolResult=%v", sawToolCall, sawToolResult)
	}
	if got2.Note != "round2" {
		t.Errorf("round2 parsed = %+v, want note=round2", got2)
	}
	if out2 == nil {
		t.Fatal("round2 Outcome is nil")
	}
}

// TestRunJSON_ContinueNilPrevDegradesToReseed verifies that a nil prev
// Outcome is a safe no-op fallback to plain reseeding, so a caller need not
// special-case round 1 with a nil check before calling RunJSON with Continue.
func TestRunJSON_ContinueNilPrevDegradesToReseed(t *testing.T) {
	fc := newFakeClient(textResp(`{"path":"a.go","note":"fix imports"}`, 5, 5))
	r := NewRunner(fc, nil, "sys")

	var got item
	if _, err := r.RunJSON(context.Background(), "summarize the report", nil, &got, Continue(nil)); err != nil {
		t.Fatalf("RunJSON(Continue(nil)) with nil prev: %v", err)
	}
	if len(fc.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(fc.requests))
	}
	if len(fc.requests[0].Messages) != 1 {
		t.Errorf("request messages = %d, want 1 (reseeded, no prior history)", len(fc.requests[0].Messages))
	}
	if got.Path != "a.go" {
		t.Errorf("parsed = %+v", got)
	}
}

func TestRunJSON_StripsMarkdownFences(t *testing.T) {
	fc := newFakeClient(textResp("```json\n{\"path\":\"b.go\",\"note\":\"x\"}\n```", 5, 5))
	r := NewRunner(fc, nil, "sys")

	var got item
	if _, err := r.RunJSON(context.Background(), "task", nil, &got); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "b.go" {
		t.Errorf("parsed = %+v", got)
	}
}

func TestRunJSON_RepairSucceeds(t *testing.T) {
	// First answer is not JSON; repair round-trip returns valid JSON.
	fc := newFakeClient(
		textResp("here is the answer: not json at all", 5, 5),
		textResp(`{"path":"c.go","note":"fixed"}`, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &got)
	if err != nil {
		t.Fatalf("RunJSON should succeed after repair: %v", err)
	}
	if got.Path != "c.go" {
		t.Errorf("parsed = %+v", got)
	}
	// Two completions happened (original + repair). The repair Run is fresh, so
	// the returned outcome's Iterations is 1, but two requests hit the client.
	if len(fc.requests) != 2 {
		t.Errorf("client calls = %d, want 2", len(fc.requests))
	}
	// The repair prompt must mention the parse failure.
	repairTask := fc.requests[1].Messages[0].Text()
	if !strings.Contains(repairTask, "failed to parse") {
		t.Errorf("repair prompt missing parse-failure note:\n%s", repairTask)
	}
	_ = out
}

// TestRunJSON_RepairAppendsToStreamedTranscript is a regression test for the
// streamed-transcript / repair interaction. Historically closeStream nilled
// streamFile/streamEnc but left streamPath set while streamAppend reopened
// with os.Create (O_TRUNC) — so repair()'s completion wiped the main run's
// already-streamed events down to just the two repair events. The first fix
// disarmed streaming entirely at closeStream, which protected the file but
// made the repair round-trip INVISIBLE on disk: a repair that produced the
// final (possibly still-unparseable) answer left no trace in the JSONL,
// making failures like an hallucinated schema-violating repair output
// undiagnosable post-hoc.
//
// The current contract: streamAppend opens with O_APPEND (never truncates)
// and closeStream keeps streamPath armed, so repair()'s request+assistant
// turns APPEND to the same on-disk file. This asserts: exactly one file on
// disk, holding the main run's events UNTRUNCATED and FIRST, followed by the
// repair's events — matching the in-memory Outcome.Transcript exactly.
func TestRunJSON_RepairAppendsToStreamedTranscript(t *testing.T) {
	dir := t.TempDir()
	fc := newFakeClient(
		textResp("here is the answer: not json at all", 5, 5),
		textResp(`{"path":"c.go","note":"fixed"}`, 5, 5),
	)
	r := NewRunner(fc, nil, "sys", WithTranscriptDir(dir))

	var got item
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &got)
	if err != nil {
		t.Fatalf("RunJSON should succeed after repair: %v", err)
	}
	if got.Path != "c.go" {
		t.Errorf("parsed = %+v", got)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected exactly 1 transcript file (repair must append, never create a second), got %d: %v", len(entries), entries)
	}

	f, err := os.Open(filepath.Join(dir, entries[0].Name()))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() { _ = f.Close() }()
	loaded, err := LoadJSONL(f)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}

	// Main run: request+assistant. Repair: request+assistant. All on disk,
	// in order — the main run's events first (no truncation), the repair's
	// appended after.
	if len(loaded.Events) != 4 {
		t.Fatalf("on-disk events = %d, want 4 (main run request+assistant, then repair request+assistant); on-disk=%+v",
			len(loaded.Events), loaded.Events)
	}
	mainIdx, repairIdx := -1, -1
	for i, ev := range loaded.Events {
		if ev.Kind == EventAssistant && ev.Text == "here is the answer: not json at all" {
			mainIdx = i
		}
		if ev.Kind == EventAssistant && ev.Text == `{"path":"c.go","note":"fixed"}` {
			repairIdx = i
		}
	}
	if mainIdx == -1 {
		t.Errorf("on-disk transcript missing the main run's assistant text (truncated by reopen?); got %+v", loaded.Events)
	}
	if repairIdx == -1 {
		t.Errorf("on-disk transcript missing the repair's assistant text (repair invisible on disk); got %+v", loaded.Events)
	}
	if mainIdx != -1 && repairIdx != -1 && mainIdx > repairIdx {
		t.Errorf("main-run assistant event at %d AFTER repair's at %d — append order violated", mainIdx, repairIdx)
	}

	// The in-memory Outcome.Transcript matches the on-disk picture.
	if len(out.Transcript.Events) != 4 {
		t.Errorf("in-memory Transcript.Events = %d, want 4 (main run + repair)", len(out.Transcript.Events))
	}
	// A repair on a NON-truncated run must leave the truncation fields alone:
	// only the original run's state folds through, and there is none here.
	if out.TruncationReason != "" {
		t.Errorf("TruncationReason = %q, want empty (run never truncated)", out.TruncationReason)
	}
	if out.Finalized {
		t.Error("Outcome.Finalized = true, want false (no finalization on this run)")
	}
	if out.LastStopReason != llmkit.StopEndTurn {
		t.Errorf("LastStopReason = %q, want %q (the repair completion's stop reason)", out.LastStopReason, llmkit.StopEndTurn)
	}
}

func TestRunJSON_RepairFails(t *testing.T) {
	fc := newFakeClient(
		textResp("garbage", 5, 5),
		textResp("still garbage", 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got item
	_, err := r.RunJSON(context.Background(), "task", nil, &got)
	if err == nil {
		t.Fatal("expected error after failed repair")
	}
	if !strings.Contains(err.Error(), "after one repair") {
		t.Errorf("error = %v, want 'after one repair'", err)
	}
}

// TestRunJSON_ForcedFinalization proves that when an agent exhausts its
// iteration cap mid-analysis (always calling tools, never finishing),
// RunJSON's reserved finalization turn still recovers the JSON answer instead of
// failing on dangling exploration prose. This is the core fix for agents that
// hit MaxIterations on a large task.
func TestRunJSON_ForcedFinalization(t *testing.T) {
	const maxIter = 3
	steps := make([]scriptStep, 0, maxIter+1)
	// The model analyzes every turn up to the cap, never producing an answer.
	for i := 0; i < maxIter; i++ {
		steps = append(steps, toolResp("c", "echo", `{"v":"x"}`, 1, 1))
	}
	// The reserved finalization turn: tools are dropped, and the model finally
	// emits the JSON.
	steps = append(steps, textResp(`{"path":"z.go","note":"found it"}`, 2, 2))

	fc := newFakeClient(steps...)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{MaxIterations: maxIter}))

	var got item
	out, err := r.RunJSON(context.Background(), "audit", json.RawMessage(`{"type":"object"}`), &got)
	if err != nil {
		t.Fatalf("RunJSON should recover via finalization: %v", err)
	}
	if got.Path != "z.go" || got.Note != "found it" {
		t.Errorf("parsed = %+v, want the finalization JSON", got)
	}
	if !out.Finalized {
		t.Error("Outcome.Finalized = false, want true (finalization turn should have fired)")
	}
	// The finalization request must carry NO tools so the model can only answer.
	finalReq := fc.requests[len(fc.requests)-1]
	if len(finalReq.Tools) != 0 {
		t.Errorf("finalization request carried %d tool(s), want 0", len(finalReq.Tools))
	}
	// The finalization user message must have been injected.
	lastMsg := finalReq.Messages[len(finalReq.Messages)-1]
	if lastMsg.Role != llmkit.RoleUser || !strings.Contains(lastMsg.Text(), "STOP investigating") {
		t.Errorf("finalization message missing; last message = %+v", lastMsg)
	}
}

// TestRunJSON_ForcedFinalizationFiresOnce confirms finalization is attempted at
// most once: if the finalization turn itself does not produce JSON, RunJSON does
// not loop, but falls through to its single repair round-trip. It also pins the
// returned Outcome's shape after a repaired, truncated run: the original run's
// TruncationReason/Finalized survive, Usage/Iterations are cumulative, and
// LastStopReason reflects the repair completion.
func TestRunJSON_ForcedFinalizationFiresOnce(t *testing.T) {
	const maxIter = 2
	steps := []scriptStep{
		toolResp("c", "echo", `{"v":"x"}`, 1, 1),
		toolResp("c", "echo", `{"v":"x"}`, 1, 1),
		// finalization turn: still not JSON.
		textResp("still just prose, sorry", 1, 1),
		// repair round-trip (a fresh run): now valid JSON.
		textResp(`{"path":"r.go","note":"repaired"}`, 1, 1),
	}
	fc := newFakeClient(steps...)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{MaxIterations: maxIter}))

	var got item
	out, err := r.RunJSON(context.Background(), "audit", nil, &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "r.go" {
		t.Errorf("parsed = %+v, want repaired JSON", got)
	}
	// The repair's fresh Outcome must not erase the original run's stop
	// condition: the loop truncated at MaxIterations and finalized, and the
	// caller must still see that after a successful repair.
	if out.TruncationReason != TruncMaxIterations {
		t.Errorf("TruncationReason = %q, want %q (the repair must not erase the truncation)", out.TruncationReason, TruncMaxIterations)
	}
	if !out.Finalized {
		t.Error("Outcome.Finalized = false, want true (carried from the original run)")
	}
	// Usage/Iterations are cumulative across the original run AND the repair:
	// 2 tool turns + 1 finalization + 1 repair completion.
	if out.Iterations != 4 {
		t.Errorf("Iterations = %d, want 4 (3 run turns + 1 repair)", out.Iterations)
	}
	if out.Usage.InputTokens != 4 || out.Usage.OutputTokens != 4 {
		t.Errorf("Usage = %+v, want 4 in / 4 out (original run + repair)", out.Usage)
	}
	if out.LastStopReason != llmkit.StopEndTurn {
		t.Errorf("LastStopReason = %q, want %q (the repair completion's stop reason)", out.LastStopReason, llmkit.StopEndTurn)
	}
}

// TestRunJSON_RepairInfraErrorPreservesTruncationAndUsage covers the
// `rerr != nil` early return in runJSON's post-repair fold: the original
// run's TruncationReason/Finalized/Iterations/Usage — including all four
// Usage counters — must survive even when the repair completion itself
// fails with an infrastructure error, not just on the success path
// TestRunJSON_ForcedFinalizationFiresOnce already pins. repair()'s own
// Outcome starts zero-valued (see [Runner.repair]) and never accrues
// anything on an error exit, so if the fold that copies the main run's
// numbers onto repairOutcome is moved below the `if rerr != nil { return }`
// early return, every assertion below fails: TruncationReason reverts to
// "", Finalized to false, and Iterations/Usage to zero. This also pins the
// four-counter cache-usage fold: a fake response with zero
// CacheRead/CacheCreationInputTokens would pass even with the fold lines for
// those two fields deleted, so every scripted turn here carries non-zero
// cache usage.
func TestRunJSON_RepairInfraErrorPreservesTruncationAndUsage(t *testing.T) {
	const maxIter = 2
	repairErr := errors.New("boom: transport reset")
	steps := []scriptStep{
		withCache(toolResp("c1", "echo", `{"v":"x"}`, 10, 5), 2, 1),
		withCache(toolResp("c2", "echo", `{"v":"x"}`, 10, 5), 2, 1),
		// finalization turn: still not JSON, so the repair round-trip fires.
		withCache(textResp("still just prose, sorry", 10, 5), 2, 1),
		// repair round-trip: an infrastructure failure, not a parse failure —
		// repair() never gets to fold anything onto its own Outcome.
		{err: repairErr},
	}
	fc := newFakeClient(steps...)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{MaxIterations: maxIter}))

	var got item
	out, err := r.RunJSON(context.Background(), "audit", nil, &got)
	if err == nil {
		t.Fatal("RunJSON: want the repair's infra error, got nil")
	}
	if errors.Is(err, ErrUnparseableOutput) {
		t.Errorf("err = %v, must NOT wrap ErrUnparseableOutput: this is an infrastructure failure, not a bad model answer", err)
	}
	if out == nil {
		t.Fatal("Outcome must be non-nil even when the repair errors, per RunJSON's contract")
	}
	if out.TruncationReason != TruncMaxIterations {
		t.Errorf("TruncationReason = %q, want %q (must survive the repair's infra error)", out.TruncationReason, TruncMaxIterations)
	}
	if !out.Finalized {
		t.Error("Finalized = false, want true (carried from the original run, not repair's zero-valued Outcome)")
	}
	if out.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3 (2 tool turns + 1 finalization; the failed repair contributes none)", out.Iterations)
	}
	wantUsage := llmkit.Usage{InputTokens: 30, OutputTokens: 15, CacheReadInputTokens: 6, CacheCreationInputTokens: 3}
	if out.Usage != wantUsage {
		t.Errorf("Usage = %+v, want %+v (all four counters summed from the 3 pre-repair turns)", out.Usage, wantUsage)
	}
}

// TestRunJSON_RepairLastStopReasonReflectsRepair pins the discriminator the
// post-repair Outcome fold must preserve: when the MAIN run's answer is cut
// off at the max-tokens cap (LastStopReason StopMaxTokens after the
// continuation stitch fails to form valid JSON) and the repair completes
// cleanly, the returned Outcome's LastStopReason is the repair's StopEndTurn,
// while Usage/Iterations still fold the whole round in. A max-tokens stop is
// not a TruncationReason, so that field stays empty.
func TestRunJSON_RepairLastStopReasonReflectsRepair(t *testing.T) {
	fc := newFakeClient(
		maxTokensResp(`{"path":"a.go"`, 5, 5),            // cut off mid-object
		maxTokensResp(` ,"more`, 5, 5),                   // continuation, still no valid JSON
		textResp(`{"path":"r.go","note":"fixed"}`, 7, 3), // repair completes cleanly
	)
	r := NewRunner(fc, nil, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "task", nil, &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "r.go" {
		t.Errorf("parsed = %+v, want repaired JSON", got)
	}
	if out.TruncationReason != "" {
		t.Errorf("TruncationReason = %q, want empty (a max-tokens stop is not a truncation)", out.TruncationReason)
	}
	if out.LastStopReason != llmkit.StopEndTurn {
		t.Errorf("LastStopReason = %q, want %q (the repair completion's stop reason)", out.LastStopReason, llmkit.StopEndTurn)
	}
	if out.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3 (main + continuation + repair)", out.Iterations)
	}
	if out.Usage.InputTokens != 17 || out.Usage.OutputTokens != 13 {
		t.Errorf("Usage = %+v, want 17 in / 13 out (original run + repair)", out.Usage)
	}
}

// TestRunJSON_MaxTokensContinuation proves the one-shot continuation retry
// stitches a JSON answer that was cut off at the output token cap back together
// so it parses, and surfaces the truncation distinctly when it still fails.
func TestRunJSON_MaxTokensContinuation(t *testing.T) {
	t.Run("continuation completes truncated JSON", func(t *testing.T) {
		fc := newFakeClient(
			maxTokensResp(`{"path":"a.go","not`, 5, 5), // cut off mid-object
			textResp(`e":"done"}`, 5, 5),               // continuation finishes it
		)
		r := NewRunner(fc, nil, "sys")

		var got item
		out, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err != nil {
			t.Fatalf("RunJSON should stitch continuation: %v", err)
		}
		if got.Path != "a.go" || got.Note != "done" {
			t.Errorf("parsed = %+v, want stitched JSON", got)
		}
		// Both completions must have happened within the same run.
		if len(fc.requests) != 2 {
			t.Errorf("client calls = %d, want 2 (initial + continuation)", len(fc.requests))
		}
		_ = out
	})

	t.Run("continuation that repeats the prefix is stitched without corruption", func(t *testing.T) {
		// The model ignores "continue from where you stopped" and restarts, repeating
		// the head of the first half before emitting the rest. A naive head+cont
		// concatenation would double `{"path":"a.go","not` and break the JSON; the
		// stitch must trim the repeated prefix.
		fc := newFakeClient(
			maxTokensResp(`{"path":"a.go","not`, 5, 5),      // cut off mid-object
			textResp(`{"path":"a.go","note":"done"}`, 5, 5), // restart: repeats prefix, then finishes
		)
		r := NewRunner(fc, nil, "sys")

		var got item
		_, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err != nil {
			t.Fatalf("RunJSON should stitch a repeated-prefix continuation: %v", err)
		}
		if got.Path != "a.go" || got.Note != "done" {
			t.Errorf("parsed = %+v, want stitched JSON with the duplicated prefix trimmed", got)
		}
	})

	t.Run("truncation surfaced in error when unrecoverable", func(t *testing.T) {
		// Both the initial answer and its continuation stop at max_tokens and never
		// form valid JSON; after the repair round-trip also truncates, the error
		// must name the truncation.
		fc := newFakeClient(
			maxTokensResp(`{"path":"a.go"`, 5, 5),
			maxTokensResp(` ,"more`, 5, 5),
			// repair round-trip:
			maxTokensResp(`{"path":"a.go"`, 5, 5),
			maxTokensResp(` ,"more`, 5, 5),
		)
		r := NewRunner(fc, nil, "sys")

		var got item
		_, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err == nil {
			t.Fatal("expected error when JSON never completes")
		}
		if !strings.Contains(err.Error(), "truncated at the max-tokens cap") {
			t.Errorf("error = %v, want it to name the max-tokens truncation", err)
		}
	})
}

// TestRunJSON_StripsThinkBlocks proves the parse path tolerates reasoning-model
// think blocks WITHOUT spending a repair round-trip, on both the first attempt
// and the repair attempt, while leaving the raw transcript untouched.
func TestRunJSON_StripsThinkBlocks(t *testing.T) {
	t.Run("first attempt, think before json", func(t *testing.T) {
		raw := "<think>the cfg may be nil here</think>\n{\"path\":\"a.go\",\"note\":\"fix imports\"}"
		fc := newFakeClient(textResp(raw, 5, 5))
		r := NewRunner(fc, nil, "sys")

		var got item
		out, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		if got.Path != "a.go" || got.Note != "fix imports" {
			t.Errorf("parsed = %+v", got)
		}
		if out.Iterations != 1 {
			t.Errorf("Iterations = %d, want 1 (no repair needed)", out.Iterations)
		}
		if len(fc.requests) != 1 {
			t.Errorf("client calls = %d, want 1 (think block must not trigger repair)", len(fc.requests))
		}
		// The transcript must preserve the RAW text (think block intact).
		if assistantText(out) != raw {
			t.Errorf("transcript assistant text = %q, want raw %q", assistantText(out), raw)
		}
	})

	t.Run("think plus fenced json", func(t *testing.T) {
		raw := "<think>reasoning</think>\n```json\n{\"path\":\"b.go\",\"note\":\"x\"}\n```"
		fc := newFakeClient(textResp(raw, 5, 5))
		r := NewRunner(fc, nil, "sys")

		var got item
		if _, err := r.RunJSON(context.Background(), "task", nil, &got); err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		if got.Path != "b.go" {
			t.Errorf("parsed = %+v", got)
		}
	})

	t.Run("repair attempt also strips think blocks", func(t *testing.T) {
		// First reply is unparseable; repair reply wraps valid JSON in a think
		// block. The repair must succeed, exercising stripping on the repair path.
		fc := newFakeClient(
			textResp("not json", 5, 5),
			textResp("<think>ok, valid json now</think>\n{\"path\":\"c.go\",\"note\":\"fixed\"}", 5, 5),
		)
		r := NewRunner(fc, nil, "sys")

		var got item
		if _, err := r.RunJSON(context.Background(), "task", nil, &got); err != nil {
			t.Fatalf("RunJSON should succeed after repair: %v", err)
		}
		if got.Path != "c.go" {
			t.Errorf("parsed = %+v", got)
		}
	})

	t.Run("literal think token inside json value survives", func(t *testing.T) {
		raw := `{"path":"d.go","note":"saw <think> in the source"}`
		fc := newFakeClient(textResp(raw, 5, 5))
		r := NewRunner(fc, nil, "sys")

		var got item
		if _, err := r.RunJSON(context.Background(), "task", nil, &got); err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		if got.Note != "saw <think> in the source" {
			t.Errorf("message corrupted: %q", got.Note)
		}
	})
}

// TestRunJSON_MiniMaxM27Reasoning replays the real MiniMax-M2.7 response shape,
// confirmed live against https://api.minimax.io/v1 on 2026-06-17:
// reasoning is an inline <think>...</think> block at the START of
// message.content (there is NO separate reasoning_content field on the message),
// and crucially the think block itself can contain a ```go fence, with the real
// answer following in a ```json fence. parseInto (stripThinkBlocks then
// stripFences) must recover the JSON in one shot — no repair round-trip — even
// with a code fence nested inside the discarded reasoning.
func TestRunJSON_MiniMaxM27Reasoning(t *testing.T) {
	raw := "<think>\nThe function is named `add` but uses `-`:\n" +
		"```go\nfunc add(a, b int) int { return a - b }\n```\n" +
		"That is a logic error: the wrong operator is used.\n</think>\n\n" +
		"```json\n{\"path\":\"add.go\",\"note\":\"add() subtracts instead of adding\"}\n```"
	fc := newFakeClient(textResp(raw, 5, 5))
	r := NewRunner(fc, nil, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "task", nil, &got)
	if err != nil {
		t.Fatalf("RunJSON on real MiniMax-M2.7 shape: %v", err)
	}
	if got.Path != "add.go" || got.Note != "add() subtracts instead of adding" {
		t.Errorf("parsed = %+v, want the add.go item", got)
	}
	// The whole point of agent-layer tolerance: a reasoning model's think block +
	// fenced answer must parse without burning the repair round-trip.
	if out.Iterations != 1 || len(fc.requests) != 1 {
		t.Errorf("M2.7 shape must parse without repair: iterations=%d calls=%d", out.Iterations, len(fc.requests))
	}
}

// assistantText returns the first EventAssistant text from the outcome's
// transcript, used to assert the raw model text is preserved unmodified.
func assistantText(out *Outcome) string {
	for _, ev := range out.Transcript.Events {
		if ev.Kind == EventAssistant {
			return ev.Text
		}
	}
	return ""
}

func TestStripFences(t *testing.T) {
	cases := map[string]string{
		"plain":              "plain",
		"```\nx\n```":        "x",
		"```json\n{}\n```":   "{}",
		"  ```\ny\n```  ":    "y",
		"no closing\n```\na": "no closing\n```\na", // not a leading fence -> trimmed only
	}
	for in, want := range cases {
		if got := stripFences(in); got != want {
			t.Errorf("stripFences(%q) = %q, want %q", in, got, want)
		}
	}
}

// budgetCutClient is a scripted llmkit.Client for budget-pressure tests: it
// always requests a tool (so the loop never naturally finishes) and reports
// a large, fixed Usage on every completion. The first N-1 completions also
// report a tool call, and the final one (the reserved finalization turn)
// returns a text answer. The RUNNER charges the shared pool after each
// completion (WithBudgetPool): at the default weight 1.0 each tool turn
// costs 2*perCall chargeable tokens.
type budgetCutClient struct {
	mu        sync.Mutex
	calls     int
	finalAt   int    // call index (1-based) at which to return text
	finalText string // text to return on the finalization turn
	perCall   int64
}

func (c *budgetCutClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }
func (c *budgetCutClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	if c.calls == c.finalAt {
		return llmkit.Response{
			Text:       c.finalText,
			StopReason: llmkit.StopEndTurn,
			Usage:      llmkit.Usage{InputTokens: 1, OutputTokens: 1},
		}, nil
	}
	return llmkit.Response{
		StopReason: llmkit.StopToolUse,
		ToolCalls:  []llmkit.ToolCall{{ID: "c", Name: "echo", Arguments: []byte(`{}`)}},
		Usage:      llmkit.Usage{InputTokens: c.perCall, OutputTokens: c.perCall},
	}, nil
}

// TestRunJSON_BudgetPoolFinalizesAndParses proves that a RunJSON run whose
// shared pool is exhausted (WithBudgetPool; the Runner charges each turn and
// checks pre-turn) now TAKES a finalization turn (outcome.Finalized==true)
// and, when the model emits valid JSON on that turn, RunJSON parses it
// successfully — no "empty model output" failure. This is the core fix for
// budget-pressured agents.
func TestRunJSON_BudgetPoolFinalizesAndParses(t *testing.T) {
	const maxIter = 10
	pool := NewBudgetPool(100) // tiny pool
	// Each tool turn charges 2*perCall = 60; after 2 turns 120 >= 100, so the
	// pre-turn gate fires and the 3rd call is the finalization turn.
	c := &budgetCutClient{
		finalAt:   3,
		finalText: `{"path":"x.go","note":"recovered"}`,
		perCall:   30,
	}
	r := NewRunner(c, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{
			MaxIterations: maxIter,
			TokenBudget:   -1,
		}),
		WithBudgetPool(pool))
	var got item
	out, err := r.RunJSON(context.Background(), "audit", json.RawMessage(`{"type":"object"}`), &got)
	if err != nil {
		t.Fatalf("RunJSON should recover via budget-pressured finalization: %v", err)
	}
	if got.Path != "x.go" || got.Note != "recovered" {
		t.Errorf("parsed = %+v, want the finalization JSON", got)
	}
	if !out.Finalized {
		t.Error("Outcome.Finalized = false, want true (budget pool stop should fire finalization)")
	}
	if out.TruncationReason != TruncBudgetPool {
		t.Errorf("TruncationReason = %q, want %q", out.TruncationReason, TruncBudgetPool)
	}
}

// TestRunJSON_PerRunTokenBudgetFinalizesAndParses is the per-run TokenBudget
// counterpart: a near-budget RunJSON run gets the reserved finalization turn
// and parses successfully when the model emits valid JSON.
func TestRunJSON_PerRunTokenBudgetFinalizesAndParses(t *testing.T) {
	const maxIter = 10
	const perRunBudget int64 = 100
	// 2 tool calls each spend 60 (cumulative 120 > 100), then on the 3rd
	// pre-turn check overBudget fires, finalization turn takes the 3rd call.
	c := &budgetCutClient{finalAt: 3, finalText: `{"path":"y.go","note":"ok"}`, perCall: 30}
	r := NewRunner(c, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{
		MaxIterations: maxIter,
		TokenBudget:   perRunBudget,
	}))
	var got item
	out, err := r.RunJSON(context.Background(), "audit", json.RawMessage(`{"type":"object"}`), &got)
	if err != nil {
		t.Fatalf("RunJSON should recover via per-run-budget finalization: %v", err)
	}
	if got.Path != "y.go" {
		t.Errorf("parsed = %+v, want the finalization JSON", got)
	}
	if !out.Finalized {
		t.Error("Outcome.Finalized = false, want true (per-run budget stop should fire finalization)")
	}
	if out.TruncationReason != TruncTokenBudget {
		t.Errorf("TruncationReason = %q, want %q", out.TruncationReason, TruncTokenBudget)
	}
}

// TestRunJSON_BudgetFinalizeEmptyStillClassified covers the case where the
// finalization turn itself yields no parseable JSON: the outcome is still
// cleanly classified as a budget stop (TruncationReason + budget reason),
// not a silently-empty result. The caller's budgetStopped(outcome) must
// return true.
func TestRunJSON_BudgetFinalizeEmptyStillClassified(t *testing.T) {
	pool := NewBudgetPool(100)
	// finalization turn returns empty text — model fails to emit a useful answer.
	// Each tool turn charges 2*perCall = 60; the 3rd call is the finalization turn.
	c := &budgetCutClient{
		finalAt:   3,
		finalText: "",
		perCall:   30,
	}
	r := NewRunner(c, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{
			MaxIterations: 10,
			TokenBudget:   -1,
		}),
		WithBudgetPool(pool))
	var got item
	_, err := r.RunJSON(context.Background(), "audit", json.RawMessage(`{"type":"object"}`), &got)
	if err == nil {
		t.Fatal("RunJSON should fail to parse empty finalization output")
	}
	// Now look at the outcome (returned alongside err per RunJSON contract).
	// We re-run and inspect via the Run path's outcome to check classification.
	// Simpler: re-check via a direct run + outcome check.
	c2 := &budgetCutClient{
		finalAt:   3,
		finalText: "",
		perCall:   30,
	}
	r2 := NewRunner(c2, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{
			MaxIterations: 10,
			TokenBudget:   -1,
		}),
		WithBudgetPool(pool))
	out, _ := r2.run(context.Background(), nil, "audit", nil, finalizationPrompt(json.RawMessage(`{"type":"object"}`)), nil, nil)
	if !out.Truncated() {
		t.Error("Outcome.Truncated() = false, want true (budget stop should still mark truncated)")
	}
	if out.TruncationReason != TruncBudgetPool {
		t.Errorf("TruncationReason = %q, want %q (so caller classifies as budget-stopped, not parse-failed)", out.TruncationReason, TruncBudgetPool)
	}
	if !out.Finalized {
		t.Error("Outcome.Finalized = false, want true (finalization turn was taken even if empty)")
	}
}

// TestRunJSON_RunPathNoExtraCall is the regression for the budget_test.go
// invariant: the public Run (finalizePrompt == "") must NOT pay an extra
// model call on a budget stop. The shared-pool overshoot bound
// (B + one in-flight call per runner) depends on this. We assert that a
// Run call into an exhausted pool issues exactly the same number of
// completions as before the fix.
func TestRunJSON_RunPathNoExtraCall(t *testing.T) {
	pool := NewBudgetPool(100)
	// bigSpendClient from budget_test.go: always requests a tool; the RUNNER
	// charges the pool after each completion (WithBudgetPool).
	c := &bigSpendClient{perCall: 60}
	r := NewRunner(c, []Tool{noopTool{}}, "sys",
		WithLimits(Limits{
			MaxIterations: -1,
			TokenBudget:   -1,
		}),
		WithBudgetPool(pool))
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.TruncationReason != TruncBudgetPool {
		t.Errorf("TruncationReason = %q, want %q", out.TruncationReason, TruncBudgetPool)
	}
	// The fix must not add an extra completion for Run (no finalizePrompt).
	// bigSpendClient charges 60 per call; pool is 100, so the pre-turn gate
	// fires on the 3rd call attempt (after 120 charged). The 2nd call may
	// have been charged, but the 3rd is gated before completion. Allow 1-2
	// calls — the precise count depends on charge order, but it MUST be < 3.
	if c.callCount.Load() > 2 {
		t.Errorf("Run (no finalize) made %d calls; the fix must NOT add an extra model call on the no-finalize path (want <= 2)", c.callCount.Load())
	}
}

// itemsSchema is a JSON schema in the "summarize the report" answer shape
// used by production callers. It exercises the
// "object root + required top-level fields" branch of validateSchema.
const itemsSchema = `{
  "type": "object",
  "required": ["path", "note", "resolved"],
  "properties": {
    "path": {"type": "string"},
    "note": {"type": "string"},
    "resolved": {"type": "boolean"}
  }
}`

// validItemJSON is one well-shaped answer to the schema above. Reused by
// the cap-on and validation-triggered-repair tests.
const validItemJSON = `{"path":"a.go","note":"fix imports","resolved":false}`

// TestRunJSON_NoCapPassthrough asserts the agent-layer gate: when the client
// reports StructuredOutput==false, the wire request carries NO
// ResponseSchema, and behavior matches today's no-native-schema path (parse +
// parse-error → repair → after-one-repair error) exactly. This is the
// acceptance criterion: "RunJSON sends ResponseSchema only when
// StructuredOutput cap set".
func TestRunJSON_NoCapPassthrough(t *testing.T) {
	fc := newFakeClient(textResp(validItemJSON, 5, 5))
	// caps is the zero value: StructuredOutput is false.
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "a.go" {
		t.Errorf("parsed = %+v", got)
	}
	if len(fc.requests) != 1 {
		t.Errorf("client calls = %d, want 1 (no cap, no repair)", len(fc.requests))
	}
	// The single wire request MUST NOT carry a ResponseSchema. The agent gate
	// is supposed to drop the schema when the client can't honor it.
	if len(fc.requests[0].ResponseSchema) != 0 {
		t.Errorf("ResponseSchema on wire = %s, want empty (no-cap passthrough)", string(fc.requests[0].ResponseSchema))
	}
	_ = out
}

// TestRunJSON_CapOnCarriesSchema asserts that when the client reports
// StructuredOutput==true, every completion in a RunJSON run carries the
// ResponseSchema on the wire, so the adapter can apply grammar-constrained
// decoding. This is the "CAP ON" half of the acceptance criterion.
func TestRunJSON_CapOnCarriesSchema(t *testing.T) {
	fc := newFakeClient(textResp(validItemJSON, 5, 5))
	fc.caps = llmkit.Capabilities{StructuredOutput: true}
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "a.go" {
		t.Errorf("parsed = %+v", got)
	}
	if len(fc.requests) != 1 {
		t.Errorf("client calls = %d, want 1", len(fc.requests))
	}
	if got := string(fc.requests[0].ResponseSchema); got != itemsSchema {
		t.Errorf("ResponseSchema on wire = %s, want the schema verbatim", got)
	}
}

// TestRunJSON_ValidationTriggersRepair asserts the core shape-validation
// contract: when the model's first answer is valid JSON but the WRONG SHAPE
// (here, a bare array where the schema requires an object), RunJSON detects
// the shape violation and routes the call through the repair path. The
// repair is a SINGLE tools-less, schema-bearing completion — not a fresh
// tool loop — so adapters that support native structured output apply
// grammar-constrained decoding on the retry and the answer is shape-correct
// on the wire. This is the "VALIDATION-TRIGGERED REPAIR" acceptance case.
func TestRunJSON_ValidationTriggersRepair(t *testing.T) {
	// First answer is a bare JSON array — parses, but validateSchema
	// detects the root-type mismatch against the schema's "object" type.
	// The inner object is ALSO schema-invalid (missing required "resolved")
	// so the rescue scan (rescueBody) cannot salvage it and the repair path
	// genuinely fires. Repair returns a correct-shape object.
	fc := newFakeClient(
		textResp(`[{"path":"a.go","note":"fix imports"}]`, 5, 5),
		textResp(validItemJSON, 5, 5),
	)
	fc.caps = llmkit.Capabilities{StructuredOutput: true}
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
	if err != nil {
		t.Fatalf("RunJSON should succeed after shape-repair: %v", err)
	}
	if got.Path != "a.go" || got.Note != "fix imports" || got.Resolved != false {
		t.Errorf("parsed = %+v, want the valid item", got)
	}
	// First call ran the main loop with schema; second call is the
	// single-completion repair. Total: 2 requests.
	if len(fc.requests) != 2 {
		t.Fatalf("client calls = %d, want 2 (main + repair)", len(fc.requests))
	}
	// The repair request MUST be tools-less so Google/Anthropic also honor
	// the native schema on the retry.
	repairReq := fc.requests[1]
	if len(repairReq.Tools) != 0 {
		t.Errorf("repair request carried %d tool(s), want 0 (tools-less constrained completion)", len(repairReq.Tools))
	}
	// The repair request MUST carry the schema (capability on) so the
	// adapter applies grammar-constrained decoding.
	if string(repairReq.ResponseSchema) != itemsSchema {
		t.Errorf("repair ResponseSchema = %s, want the schema verbatim", string(repairReq.ResponseSchema))
	}
	// The repair prompt must mention the parse/shape failure so the model
	// knows why its previous output was rejected.
	if !strings.Contains(repairReq.Messages[0].Text(), "failed to parse") {
		t.Errorf("repair prompt missing parse-failure note:\n%s", repairReq.Messages[0].Text())
	}
	_ = out
}

// TestRunJSON_ValidationTriggersRepair_MissingRequired is the same
// shape-violation routing but exercises the "missing required field" branch
// of validateSchema (object root, type matches, but a required key is
// absent). The repair should still fire and ultimately succeed.
func TestRunJSON_ValidationTriggersRepair_MissingRequired(t *testing.T) {
	// First answer is an object missing the schema-required "resolved" key.
	// Repair returns a correct-shape object.
	fc := newFakeClient(
		textResp(`{"path":"a.go","note":"fix imports"}`, 5, 5),
		textResp(validItemJSON, 5, 5),
	)
	fc.caps = llmkit.Capabilities{StructuredOutput: true}
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON should succeed after missing-required repair: %v", err)
	}
	if got.Path != "a.go" {
		t.Errorf("parsed = %+v", got)
	}
	if len(fc.requests) != 2 {
		t.Errorf("client calls = %d, want 2 (main + repair)", len(fc.requests))
	}
}

// TestRunJSON_RescuesProseWrappedAnswer covers the dominant live
// failure mode: a weak model prefixing the final JSON with prose ("Based on
// my analysis, ... {plan}"), which fails stripBody's leading-value
// parse ("invalid character 'B' looking for beginning of value"). The
// schema-guided rescue scan must extract the embedded schema-valid object
// and succeed WITHOUT spending the repair round-trip.
func TestRunJSON_RescuesProseWrappedAnswer(t *testing.T) {
	fc := newFakeClient(
		textResp("Based on my analysis, the answer is ready. Here is the result:\n"+validItemJSON, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON should rescue a prose-wrapped schema-valid answer: %v", err)
	}
	if got.Path != "a.go" || got.Note != "fix imports" {
		t.Errorf("parsed = %+v, want the valid item", got)
	}
	if len(fc.requests) != 1 {
		t.Errorf("client calls = %d, want 1 (rescue must not spend the repair round-trip)", len(fc.requests))
	}
}

// TestRunJSON_RescuesInnerObjectFromWrappedArray: a bare-array wrap of a
// schema-valid object is rescued to that inner object — the schema is the
// arbiter of WHICH embedded result is the answer, so the array root
// (schema-invalid) is skipped and the inner object accepted, with no repair.
func TestRunJSON_RescuesInnerObjectFromWrappedArray(t *testing.T) {
	fc := newFakeClient(
		textResp(`[`+validItemJSON+`]`, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON should rescue the inner object of a wrapped array: %v", err)
	}
	if got.Path != "a.go" {
		t.Errorf("parsed = %+v", got)
	}
	if len(fc.requests) != 1 {
		t.Errorf("client calls = %d, want 1 (rescue must not spend the repair round-trip)", len(fc.requests))
	}
}

// TestRunJSON_MangledHeadStillRepairs pins the rescue's boundary using the
// live 2026-07-17 the_cloud shape: a final answer whose JSON head was
// swallowed (`": {"x_test.go": "...
// "}, "cmd": [...]`) leaves NO complete embedded value that satisfies the
// schema — the leading string literal and the bare files map both fail — so
// the rescue must NOT fire and the repair round-trip proceeds as before.
func TestRunJSON_MangledHeadStillRepairs(t *testing.T) {
	fc := newFakeClient(
		textResp(`": {"path": "a.go"}, "note": "x", "resolved": false}`, 5, 5),
		textResp(validItemJSON, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON should succeed via repair: %v", err)
	}
	if len(fc.requests) != 2 {
		t.Errorf("client calls = %d, want 2 (mangled head is not rescuable; repair must fire)", len(fc.requests))
	}
}

// TestRunJSON_RepairOutputRescuedFromProse: the rescue also applies to the
// REPAIR completion's output — a repair reply that wraps a schema-valid
// answer in prose still counts instead of failing the whole call.
func TestRunJSON_RepairOutputRescuedFromProse(t *testing.T) {
	fc := newFakeClient(
		textResp("no json here at all", 5, 5),
		textResp("Sure! Here is the corrected JSON:\n"+validItemJSON, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON should rescue the repair's prose-wrapped answer: %v", err)
	}
	if got.Path != "a.go" {
		t.Errorf("parsed = %+v", got)
	}
	if len(fc.requests) != 2 {
		t.Errorf("client calls = %d, want 2 (main + repair)", len(fc.requests))
	}
}

// TestRunJSON_RepairStillWrongShape asserts that when the repair's reply is
// STILL the wrong shape, RunJSON returns the canonical "did not parse as JSON
// after one repair" error (with a wrapping shape error so callers can
// distinguish a shape violation from a parse failure via errors.Is/Wraps).
func TestRunJSON_RepairStillWrongShape(t *testing.T) {
	// First answer is a bare array (wrong shape). Repair also returns a
	// bare array. Both calls have the same shape violation.
	fc := newFakeClient(
		textResp(`[{"path":"a.go"}]`, 5, 5),
		textResp(`[{"path":"a.go"}]`, 5, 5),
	)
	fc.caps = llmkit.Capabilities{StructuredOutput: true}
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	_, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
	if err == nil {
		t.Fatal("expected error after still-wrong repair")
	}
	if !strings.Contains(err.Error(), "after one repair") {
		t.Errorf("error = %v, want 'after one repair'", err)
	}
	// The wrapped error must be the shape violation, not a parse failure —
	// callers can errors.As/errors.Is to classify the failure mode.
	if !strings.Contains(err.Error(), "root JSON type") {
		t.Errorf("error = %v, want the wrapped shape-violation message", err)
	}
}

// TestRunJSON_ParseFailureWrapsSentinel locks the contract a caller's
// revision loop depends on: a RunJSON failure caused by the model's OWN output
// (unparseable JSON or a schema violation, even after the repair round) wraps
// [ErrUnparseableOutput], while an infrastructure failure in the underlying
// tool loop is returned unwrapped and must NOT match the sentinel. The
// distinction is what lets a caller treat a bad answer as recoverable (revise)
// without swallowing a real transport failure.
func TestRunJSON_ParseFailureWrapsSentinel(t *testing.T) {
	t.Run("unparseable after repair", func(t *testing.T) {
		fc := newFakeClient(
			textResp("garbage", 5, 5),
			textResp("still garbage", 5, 5),
		)
		r := NewRunner(fc, nil, "sys")
		var got item
		_, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err == nil {
			t.Fatal("expected error after failed repair")
		}
		if !errors.Is(err, ErrUnparseableOutput) {
			t.Errorf("errors.Is(err, ErrUnparseableOutput) = false; err = %v", err)
		}
		if !strings.Contains(err.Error(), "after one repair") {
			t.Errorf("message regressed; err = %v", err)
		}
	})

	t.Run("schema violation after repair", func(t *testing.T) {
		// A bare array where the schema requires an object: valid JSON, wrong
		// shape — exercises the validateSchema wrap site, not the unmarshal one.
		fc := newFakeClient(
			textResp(`[{"path":"a.go"}]`, 5, 5),
			textResp(`[{"path":"a.go"}]`, 5, 5),
		)
		fc.caps = llmkit.Capabilities{StructuredOutput: true}
		r := NewRunner(fc, nil, "sys")
		var got itemWithResolved
		_, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
		if err == nil {
			t.Fatal("expected schema-violation error")
		}
		if !errors.Is(err, ErrUnparseableOutput) {
			t.Errorf("errors.Is(err, ErrUnparseableOutput) = false; err = %v", err)
		}
	})

	t.Run("infra error is not wrapped", func(t *testing.T) {
		// The underlying loop fails (transport): RunJSON returns it unwrapped,
		// so it carries the transport cause but NOT the parse sentinel.
		boom := errors.New("connection reset")
		fc := newFakeClient(scriptStep{err: boom})
		r := NewRunner(fc, nil, "sys")
		var got item
		_, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err == nil {
			t.Fatal("expected infra error")
		}
		if errors.Is(err, ErrUnparseableOutput) {
			t.Errorf("infra error must not match ErrUnparseableOutput; err = %v", err)
		}
		if !errors.Is(err, boom) {
			t.Errorf("infra error should preserve the transport cause; err = %v", err)
		}
	})
}

// deepResultsSchema mirrors a production caller schema's nested shape: an
// object with a "results" array of objects carrying an enum confidence, an
// integer line with a minimum, a min-length summary string, and
// additionalProperties:false. It exercises every nested-recursion branch
// validateSchema enforces.
const deepResultsSchema = `{
  "type":"object",
  "properties":{
    "results":{
      "type":"array",
      "items":{
        "type":"object",
        "properties":{
          "path":{"type":"string","minLength":1},
          "line":{"type":"integer","minimum":1},
          "confidence":{"type":"string","enum":["high","medium","low"]},
          "summary":{"type":"string","minLength":1}
        },
        "required":["path","line","confidence","summary"],
        "additionalProperties":false
      }
    }
  },
  "required":["results"],
  "additionalProperties":false
}`

// filesMapSchema mirrors a free-form string-map contract: an object whose
// values are constrained by an additionalProperties subschema (string) and
// which must hold at least one entry (minProperties).
const filesMapSchema = `{
  "type":"object",
  "properties":{
    "files":{"type":"object","additionalProperties":{"type":"string"},"minProperties":1}
  },
  "required":["files"],
  "additionalProperties":false
}`

// oneResult wraps a single result body in the {"results":[...]} root
// so test cases can focus on the one field under test.
func oneResult(fields string) string {
	return `{"results":[` + fields + `]}`
}

// validResult is one well-shaped result object for deepResultsSchema.
const validResult = `{"path":"a.go","line":7,"confidence":"high","summary":"x"}`

// TestValidateSchema pins down the deep validator's contract: the historical
// root-level cases (preserved verbatim error phrasing) plus the nested
// type/required/enum/minimum/minLength/additionalProperties/minItems/
// minProperties branches.
func TestValidateSchema(t *testing.T) {
	deep := json.RawMessage(deepResultsSchema)
	files := json.RawMessage(filesMapSchema)
	root := json.RawMessage(itemsSchema)
	cases := []struct {
		name    string
		schema  json.RawMessage
		body    string
		wantErr bool
		wantMsg string
	}{
		// --- preserved root-level behavior ---
		{name: "nil schema is a no-op", schema: nil, body: `not even json`},
		{name: "matching object with all required passes", schema: root, body: validItemJSON},
		{name: "root type mismatch: bare array", schema: root, body: `[{"path":"a.go"}]`,
			wantErr: true, wantMsg: "root JSON type"},
		{name: "root missing required field", schema: root, body: `{"path":"a.go","note":"fix imports"}`,
			wantErr: true, wantMsg: `root object missing required field "resolved"`},
		{name: "schema requires nothing: any object passes", schema: json.RawMessage(`{"type":"object"}`), body: `{}`},
		{name: "schema with no type: any value passes", schema: json.RawMessage(`{}`), body: `42`},
		// --- deep: nested object/array recursion ---
		{name: "valid nested result passes", schema: deep, body: oneResult(validResult)},
		{name: "empty results array passes (found nothing)", schema: deep, body: `{"results":[]}`},
		{name: "nested missing required field", schema: deep,
			body:    oneResult(`{"path":"a.go","line":7,"confidence":"high"}`),
			wantErr: true, wantMsg: `results[0]: missing required field "summary"`},
		{name: "nested bad enum", schema: deep,
			body:    oneResult(`{"path":"a.go","line":7,"confidence":"urgent","summary":"x"}`),
			wantErr: true, wantMsg: `results[0].confidence: value "urgent" is not one of the allowed values`},
		{name: "nested wrong type (string where integer)", schema: deep,
			body:    oneResult(`{"path":"a.go","line":"7","confidence":"high","summary":"x"}`),
			wantErr: true, wantMsg: `results[0].line: JSON type "string" does not match schema type "integer"`},
		{name: "integer keyword rejects a fractional number", schema: deep,
			body:    oneResult(`{"path":"a.go","line":7.5,"confidence":"high","summary":"x"}`),
			wantErr: true, wantMsg: `results[0].line: JSON type "number" does not match schema type "integer"`},
		{name: "minimum violation", schema: deep,
			body:    oneResult(`{"path":"a.go","line":0,"confidence":"high","summary":"x"}`),
			wantErr: true, wantMsg: `results[0].line: value 0 is below the minimum 1`},
		{name: "minLength violation on required string", schema: deep,
			body:    oneResult(`{"path":"a.go","line":7,"confidence":"high","summary":""}`),
			wantErr: true, wantMsg: `results[0].summary: string length 0 is below the minimum 1`},
		{name: "additionalProperties:false rejects an unknown key", schema: deep,
			body:    oneResult(`{"path":"a.go","line":7,"confidence":"high","summary":"x","cwe":"CWE-20"}`),
			wantErr: true, wantMsg: `results[0]: unexpected property "cwe"`},
		// --- deep: free-form map (additionalProperties subschema + minProperties) ---
		{name: "files map with string values passes", schema: files, body: `{"files":{"a_test.go":"package a"}}`},
		{name: "files map empty violates minProperties", schema: files, body: `{"files":{}}`,
			wantErr: true, wantMsg: `files: object has 0 properties, fewer than the required minimum 1`},
		{name: "files map non-string value rejected by additionalProperties schema", schema: files,
			body:    `{"files":{"a_test.go":123}}`,
			wantErr: true, wantMsg: `files.a_test.go: JSON type "number" does not match schema type "string"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSchema(tc.schema, []byte(tc.body))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("validateSchema(%s) = nil, want error containing %q", tc.body, tc.wantMsg)
				}
				if tc.wantMsg != "" && !strings.Contains(err.Error(), tc.wantMsg) {
					t.Errorf("error = %q, want substring %q", err.Error(), tc.wantMsg)
				}
			} else if err != nil {
				t.Errorf("validateSchema(%s) = %v, want nil", tc.body, err)
			}
		})
	}
}

// TestRunJSON_DeepValidationTriggersRepair pins the deep-validator behavior:
// an answer that is valid JSON, the right ROOT shape, AND unmarshals cleanly
// into the typed struct — but carries a CONTRACT violation (an out-of-enum
// confidence). validateSchema rejects it and routes the call through the
// single repair round-trip, which returns a schema-valid answer.
func TestRunJSON_DeepValidationTriggersRepair(t *testing.T) {
	const enumSchema = `{
  "type":"object",
  "required":["status"],
  "properties":{"status":{"type":"string","enum":["open","closed"]}},
  "additionalProperties":false
}`
	type statusRec struct {
		Status string `json:"status"`
	}
	// First answer: a wrong-enum value. It parses, has the required root key,
	// and unmarshals into statusRec{Status:"urgent"} without error — exactly the
	// silent corruption the shallow check let through. Repair returns a valid
	// enum value.
	fc := newFakeClient(
		textResp(`{"status":"archived"}`, 5, 5),
		textResp(`{"status":"open"}`, 5, 5),
	)
	fc.caps = llmkit.Capabilities{StructuredOutput: true}
	r := NewRunner(fc, nil, "sys")

	var got statusRec
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(enumSchema), &got); err != nil {
		t.Fatalf("RunJSON should succeed after deep-validation repair: %v", err)
	}
	if got.Status != "open" {
		t.Errorf("parsed status = %q, want %q (the repaired value)", got.Status, "open")
	}
	if len(fc.requests) != 2 {
		t.Fatalf("client calls = %d, want 2 (main + repair)", len(fc.requests))
	}
	// The repair prompt must name the enum violation so the model can fix it.
	if !strings.Contains(fc.requests[1].Messages[0].Text(), "allowed values") {
		t.Errorf("repair prompt missing the enum-violation detail:\n%s", fc.requests[1].Messages[0].Text())
	}
}

// TestRunJSON_TrailingCommaNoRepair proves that a final answer which is a
// schema-valid JSON object followed by a trailing comma is parsed successfully
// in the first attempt — no repair round-trip. Exactly one model completion is
// consumed; a repair would require a second.
func TestRunJSON_TrailingCommaNoRepair(t *testing.T) {
	// Queue EXACTLY one completion: if stripBody correctly extracts the leading
	// JSON value, RunJSON parses and succeeds without repair. A repair would
	// consume the (unscripted) second slot and produce "(unscripted)", which
	// would itself fail to parse and cause an error.
	fc := newFakeClient(
		textResp(`{"path":"a.go","note":"fix imports","resolved":false},`, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "a.go" || got.Note != "fix imports" || got.Resolved {
		t.Errorf("parsed = %+v", got)
	}
	// Single completion: proves no repair round-trip happened.
	if fc.callCount() != 1 {
		t.Errorf("client calls = %d, want 1 (no repair)", fc.callCount())
	}
	_ = out
}

// TestRunJSON_DoubleValueParsesFirst proves that when the model emits two
// concatenated JSON objects, RunJSON parses and returns the first one.
func TestRunJSON_DoubleValueParsesFirst(t *testing.T) {
	first := `{"path":"first.go","note":"first","resolved":false}`
	second := `{"path":"second.go","note":"second","resolved":true}`
	fc := newFakeClient(
		textResp(first+second, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got itemWithResolved
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "first.go" || got.Note != "first" || got.Resolved {
		t.Errorf("parsed = %+v, want first object", got)
	}
	if fc.callCount() != 1 {
		t.Errorf("client calls = %d, want 1 (no repair)", fc.callCount())
	}
}

// TestRunJSON_TruncatedLeadingValueErrors proves that an INCOMPLETE leading
// JSON value (truncated mid-object) is NOT rescued by the trailing-content
// fix: stripBody returns the raw body, the parse fails, repair fires, and when
// the repair also fails, RunJSON returns a non-nil error.
func TestRunJSON_TruncatedLeadingValueErrors(t *testing.T) {
	fc := newFakeClient(
		textResp(`{"files":{"a.go":"x"}`, 5, 5), // missing closing }
		textResp(`still truncated {`, 5, 5),     // repair also unparseable
	)
	r := NewRunner(fc, nil, "sys")

	var got struct {
		Files map[string]string `json:"files"`
	}
	_, err := r.RunJSON(context.Background(), "task", nil, &got)
	if err == nil {
		t.Fatal("expected non-nil error for truncated leading JSON value")
	}
}

// TestRunJSON_EmptyBodyErrors confirms an empty/whitespace body still errors
// via the "empty model output" path, before any JSON extraction is attempted.
// The first two empty turns are absorbed by the empty-turn-nudge
// cap (maxEmptyTurnNudges=2); the third exhausts it and the loop breaks with
// an empty FinalText, and the fourth is the one repair completion — also
// scripted empty so the repair path fails the same way.
func TestRunJSON_EmptyBodyErrors(t *testing.T) {
	fc := newFakeClient(
		textResp("   ", 5, 5),
		textResp("", 5, 5),
		textResp("", 5, 5),
		textResp("", 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got item
	_, err := r.RunJSON(context.Background(), "task", nil, &got)
	if err == nil {
		t.Fatal("expected error for empty body")
	}
	if !strings.Contains(err.Error(), "empty model output") {
		t.Errorf("error = %v, want to contain 'empty model output'", err)
	}
}

// TestRunJSON_EmptyTurnNudgeRecovers verifies the empty-turn nudge: a think-only turn
// (zero tool calls, text that strips to empty) is nudged rather than treated
// as the model's final answer, so a subsequent tool call and valid JSON
// answer still complete the run cleanly with no repair needed.
func TestRunJSON_EmptyTurnNudgeRecovers(t *testing.T) {
	fc := newFakeClient(
		thinkOnlyResp("let me plan this out...", 10, 5),
		toolResp("c1", "echo", `{"v":"orient"}`, 10, 4),
		textResp(`{"path":"a.go","note":"fix imports"}`, 8, 3),
	)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "summarize the report", nil, &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "a.go" || got.Note != "fix imports" {
		t.Errorf("parsed = %+v", got)
	}
	// 3 completions: think-only nudge, tool call, final answer.
	if out.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3", out.Iterations)
	}
	foundNudge := false
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleUser && m.Text() == emptyTurnNudge {
			foundNudge = true
		}
	}
	if !foundNudge {
		t.Error("conversation does not contain the empty-turn nudge message")
	}
}

// TestRunJSON_EmptyTurnNudgeCapExhausted verifies the empty-turn nudge cap:
// three consecutive think-only turns exhaust maxEmptyTurnNudges (2 nudges),
// the loop breaks on the third with an empty FinalText, and RunJSON proceeds
// through its normal parse-failure/repair path — exactly 3 main-loop
// completions plus one repair completion, no infinite loop.
func TestRunJSON_EmptyTurnNudgeCapExhausted(t *testing.T) {
	fc := newFakeClient(
		thinkOnlyResp("first thought", 10, 5),
		thinkOnlyResp("second thought", 10, 5),
		thinkOnlyResp("third thought", 10, 5),
		textResp(`{"path":"a.go","note":"repaired"}`, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "summarize the report", nil, &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got.Path != "a.go" || got.Note != "repaired" {
		t.Errorf("parsed = %+v", got)
	}
	if fc.callCount() != 4 {
		t.Errorf("completions = %d, want 4 (3 main-loop + 1 repair)", fc.callCount())
	}
	nudgeCount := 0
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleUser && m.Text() == emptyTurnNudge {
			nudgeCount++
		}
	}
	if nudgeCount != maxEmptyTurnNudges {
		t.Errorf("nudge count = %d, want %d (cap)", nudgeCount, maxEmptyTurnNudges)
	}
}

// TestRunJSON_RepairStepContinuesParentSequence pins the repair step
// continuity: the repair completion's transcript step continues the parent
// run's iteration sequence instead of restarting at 1. A repaired run
// records steps 1,2 then 3 (not 1,2 then 1) while Outcome.Iterations is 3,
// so a consumer joining ToolEvent.Step / CompactionEvent.Step / Event.Step
// on Step sees one monotonic sequence.
func TestRunJSON_RepairStepContinuesParentSequence(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"hi"}`, 10, 4),
		textResp("this is not json", 5, 5),
		textResp(validItemJSON, 5, 5), // the repair completion
	)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
	if err != nil {
		t.Fatalf("RunJSON should succeed after repair: %v", err)
	}
	if out.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3 (2 main-loop turns + 1 repair)", out.Iterations)
	}
	// Events within one turn SHARE that turn's Step (it is the join key);
	// across turns the number must never go backwards, and the repair's
	// request/assistant pair must sit at step 3 — one continuous sequence.
	prev := 0
	repairSteps := map[EventKind]bool{}
	for _, ev := range out.Transcript.Events {
		if ev.Step < prev {
			t.Errorf("transcript Step went backwards: step %d after %d (kind %s)", ev.Step, prev, ev.Kind)
		}
		prev = ev.Step
		if ev.Step == 3 {
			repairSteps[ev.Kind] = true
		}
	}
	if !repairSteps[EventRequest] || !repairSteps[EventAssistant] {
		t.Errorf("repair completion not recorded at step 3 (events at step 3: %v)", repairSteps)
	}
}

// TestRunJSON_RepairFinalTextIsRepairText pins the Outcome.FinalText
// contract on the repair path: after a repair round-trip, the returned
// Outcome's FinalText is the REPAIR completion's text — the last completion
// of the run — not the unparseable pre-repair answer.
func TestRunJSON_RepairFinalTextIsRepairText(t *testing.T) {
	fc := newFakeClient(
		textResp("this is prose, not JSON", 5, 5),
		textResp(validItemJSON, 5, 5),
	)
	r := NewRunner(fc, nil, "sys")

	var got item
	out, err := r.RunJSON(context.Background(), "task", json.RawMessage(itemsSchema), &got)
	if err != nil {
		t.Fatalf("RunJSON should succeed after repair: %v", err)
	}
	if out.FinalText != validItemJSON {
		t.Errorf("FinalText = %q, want the repair completion's text %q", out.FinalText, validItemJSON)
	}
}

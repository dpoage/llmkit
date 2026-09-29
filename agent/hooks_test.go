package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// hookRecorder is a recording [Hooks] implementation: every invocation is
// appended to events as a compact "name:payload" line (in call order) and the
// structured payloads are kept for assertion.
type hookRecorder struct {
	mu     sync.Mutex
	events []string

	beforeReqs  []llmkit.Request
	afterErrs   []error
	afterNoResp bool
	starts      []ToolEvent
	ends        []ToolEvent
	healthTools []string
	healthErrs  []*ToolHealthError
}

func newHookRecorder() *hookRecorder { return &hookRecorder{} }

func (h *hookRecorder) record(s string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.events = append(h.events, s)
}

// hooks wires the recorder into a Hooks value. Every field is set, so every
// fire point is observable; individual tests assert the subset they care
// about.
func (h *hookRecorder) hooks() Hooks {
	return Hooks{
		BeforeCompletion: func(_ context.Context, step int, req *llmkit.Request) {
			h.record("before:" + itoa(step))
			h.mu.Lock()
			defer h.mu.Unlock()
			h.beforeReqs = append(h.beforeReqs, *req)
		},
		AfterCompletion: func(_ context.Context, step int, _ *llmkit.Request, resp *llmkit.Response, err error) {
			h.record("after:" + itoa(step))
			h.mu.Lock()
			defer h.mu.Unlock()
			h.afterErrs = append(h.afterErrs, err)
			if resp == nil {
				h.afterNoResp = true
			}
		},
		ToolStart: func(_ context.Context, ev ToolEvent) {
			h.record("start:" + itoa(ev.Step) + ":" + ev.Call.Name)
			h.mu.Lock()
			defer h.mu.Unlock()
			h.starts = append(h.starts, ev)
		},
		ToolEnd: func(_ context.Context, ev ToolEvent) {
			h.record("end:" + itoa(ev.Step) + ":" + ev.Call.Name)
			h.mu.Lock()
			defer h.mu.Unlock()
			h.ends = append(h.ends, ev)
		},
		ToolHealth: func(_ context.Context, tool string, he *ToolHealthError) {
			h.record("health:" + tool)
			h.mu.Lock()
			defer h.mu.Unlock()
			h.healthTools = append(h.healthTools, tool)
			h.healthErrs = append(h.healthErrs, he)
		},
	}
}

// bigResultTool returns a fixed multi-kilobyte result, large enough to trip a
// small history-token budget.
type bigResultTool struct{ name string }

func (b bigResultTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: b.name, Description: "big", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (b bigResultTool) Run(context.Context, json.RawMessage) (string, error) {
	return strings.Repeat("x", 4000), nil
}

// itoa avoids fmt in event lines for a trivially small int domain.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// TestHook_BeforeAfterCompletion_MainLoop verifies the hook pair fires around
// every loop completion with the unified hook Step: the 1-based transcript
// step the completion is recorded under — 1 for the first turn, 2 for the
// second.
func TestHook_BeforeAfterCompletion_MainLoop(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"hi"}`, 10, 4),
		textResp("answer", 8, 3),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(rec.hooks()))

	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := []string{"before:1", "after:1", "start:1:echo", "end:1:echo", "before:2", "after:2"}
	if len(rec.events) != len(want) {
		t.Fatalf("events = %v, want %v", rec.events, want)
	}
	for i := range want {
		if rec.events[i] != want[i] {
			t.Errorf("events[%d] = %q, want %q", i, rec.events[i], want[i])
		}
	}
	// The Before hook observes the exact wire request: system prompt and seed
	// task message in place.
	if got := rec.beforeReqs[0].System; got != "sys" {
		t.Errorf("before hook request System = %q, want sys", got)
	}
	if n := len(rec.beforeReqs[0].Messages); n != 1 {
		t.Errorf("before hook first request carried %d messages, want 1", n)
	}
	// AfterCompletion receives a non-nil response and nil error on success.
	if rec.afterNoResp {
		t.Error("AfterCompletion got resp == nil on a successful completion")
	}
	for i, err := range rec.afterErrs {
		if err != nil {
			t.Errorf("afterErrs[%d] = %v, want nil", i, err)
		}
	}
}

// TestHook_BeforeAfterCompletion_Continuation verifies the max-tokens
// continuation completion fires its own hook pair (steps 1 then 2).
func TestHook_BeforeAfterCompletion_Continuation(t *testing.T) {
	fc := newFakeClient(
		maxTokensResp("half ", 5, 5),
		textResp("done", 5, 5),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, nil, "sys", WithHooks(rec.hooks()))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "half done" {
		t.Fatalf("FinalText = %q, want stitched halves", out.FinalText)
	}
	want := []string{"before:1", "after:1", "before:2", "after:2"}
	if len(rec.events) != len(want) {
		t.Fatalf("events = %v, want %v", rec.events, want)
	}
	for i := range want {
		if rec.events[i] != want[i] {
			t.Errorf("events[%d] = %q, want %q", i, rec.events[i], want[i])
		}
	}
}

// TestHook_AfterCompletion_ReceivesError verifies a failed completion reports
// err (and a nil response) through AfterCompletion instead of aborting
// silently.
func TestHook_AfterCompletion_ReceivesError(t *testing.T) {
	fc := newFakeClient(scriptStep{err: errors.New("network down")})
	rec := newHookRecorder()
	r := NewRunner(fc, nil, "sys", WithHooks(rec.hooks()))

	if _, err := r.Run(context.Background(), "task"); err == nil {
		t.Fatal("expected the run to fail")
	}
	want := []string{"before:1", "after:1"}
	if len(rec.events) != len(want) {
		t.Fatalf("events = %v, want %v", rec.events, want)
	}
	if !rec.afterNoResp {
		t.Error("AfterCompletion resp must be nil when the completion failed")
	}
	if len(rec.afterErrs) != 1 || rec.afterErrs[0] == nil || !strings.Contains(rec.afterErrs[0].Error(), "network down") {
		t.Errorf("AfterCompletion err = %v, want the wrapped transport error", rec.afterErrs)
	}
}

// TestHook_ToolLifecycle_Payloads verifies ToolStart carries only Step and
// Call, and ToolEnd additionally carries Result, IsError, and a measured
// Duration, for both a succeeding and a failing tool.
func TestHook_ToolLifecycle_Payloads(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"hi"}`, 10, 4),
		toolResp("c2", "boom", `{}`, 10, 4),
		textResp("recovered", 5, 2),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}, echoTool{name: "boom", failMsg: "disk on fire"}}, "sys",
		WithHooks(rec.hooks()))

	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rec.starts) != 2 || len(rec.ends) != 2 {
		t.Fatalf("starts=%d ends=%d, want 2 each", len(rec.starts), len(rec.ends))
	}
	// ToolStart: identity only.
	s := rec.starts[0]
	if s.Step != 1 || s.Call.ID != "c1" || s.Call.Name != "echo" {
		t.Errorf("start[0] = %+v, want step 1 call c1/echo", s)
	}
	if s.Result != "" || s.IsError || s.Duration != 0 {
		t.Errorf("start[0] must carry only Step and Call, got %+v", s)
	}
	// ToolEnd success: full payload with measured duration.
	e := rec.ends[0]
	if e.Step != 1 || e.Call.ID != "c1" {
		t.Errorf("end[0] identity = %+v, want step 1 c1", e)
	}
	if e.IsError {
		t.Errorf("end[0].IsError = true, want false")
	}
	if !strings.HasPrefix(e.Result, "echo:") {
		t.Errorf("end[0].Result = %q, want the tool output", e.Result)
	}
	if e.Duration <= 0 {
		t.Errorf("end[0].Duration = %v, want > 0", e.Duration)
	}
	// ToolEnd failure: ERROR-prefixed result with IsError.
	e2 := rec.ends[1]
	if !e2.IsError {
		t.Error("end[1].IsError = false, want true for a failing tool")
	}
	if !strings.Contains(e2.Result, "disk on fire") || !strings.HasPrefix(e2.Result, "ERROR:") {
		t.Errorf("end[1].Result = %q, want ERROR-prefixed failure text", e2.Result)
	}
	if e2.Duration <= 0 {
		t.Errorf("end[1].Duration = %v, want > 0", e2.Duration)
	}
}

// TestHook_ToolLifecycle_UnknownToolDoesNotFire verifies a model naming an
// unregistered tool never reaches Tool.Run, so neither lifecycle hook fires
// (the ERROR result is still fed back).
func TestHook_ToolLifecycle_UnknownToolDoesNotFire(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "ghost", `{}`, 10, 4),
		textResp("ok", 5, 2),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(rec.hooks()))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "ok" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	if len(rec.starts) != 0 || len(rec.ends) != 0 {
		t.Errorf("lifecycle hooks fired for an unknown tool: starts=%v ends=%v", rec.starts, rec.ends)
	}
}

// TestCompaction_EventOnRealPruneOnly verifies the Compaction event is
// emitted exactly when a compaction pass actually pruned, with Before/After
// token estimates and the pruned-message count. A threshold crossing with
// nothing to reclaim is silent: the first crossing (4 results, all inside the
// recent window) must NOT emit; the second (5 results) prunes exactly one.
func TestCompaction_EventOnRealPruneOnly(t *testing.T) {
	steps := make([]scriptStep, 0, 6)
	for i := range 5 {
		steps = append(steps, toolResp(fmt.Sprintf("c%d", i), "big", `{}`, 10, 4))
	}
	steps = append(steps, textResp("done", 5, 2))
	fc := newFakeClient(steps...)

	r := NewRunner(fc, []Tool{bigResultTool{name: "big"}}, "sys",
		WithLimits(Limits{HistoryTokenBudget: 3500}))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	var compactions []llmkit.Event
	for i, e := range out.Transcript.Record {
		if e.Kind != llmkit.KindCompaction {
			continue
		}
		compactions = append(compactions, e)
		// The event precedes the completion that consumes the compacted
		// history and carries that completion's step.
		if next := out.Transcript.Record[i+1]; next.Kind != llmkit.KindCompletion || next.Step != e.Step {
			t.Errorf("compaction at step %d followed by %s at step %d, want the completion at the same step", e.Step, next.Kind, next.Step)
		}
	}
	if len(compactions) != 1 {
		t.Fatalf("compaction events = %d, want exactly 1", len(compactions))
	}
	ev := compactions[0].Compaction
	if ev.Pruned != 1 {
		t.Errorf("Pruned = %d, want 1", ev.Pruned)
	}
	if ev.BeforeTokens <= ev.AfterTokens {
		t.Errorf("BeforeTokens %d <= AfterTokens %d, want a real reduction", ev.BeforeTokens, ev.AfterTokens)
	}
	// Each big result estimates to ~1000 tokens; five results plus overhead
	// crossed the 3500 budget, so Before is in the ~5000 range.
	if ev.BeforeTokens < 5000 {
		t.Errorf("BeforeTokens = %d, want >= 5000 (five 4000-byte results)", ev.BeforeTokens)
	}
	// step is the completion that will consume the compacted history: the
	// sixth, after five tool turns.
	if compactions[0].Step != 6 {
		t.Errorf("Step = %d, want 6", compactions[0].Step)
	}
}

// TestFinalization_TurnIsToolLessAndPlainRunSkipsIt verifies the reserved
// finalization turn is taken when a finalize prompt is set — Outcome.Finalized
// is true and the finalization completion carries no tools — and that a plain
// Run (no finalize prompt) never takes it.
func TestFinalization_TurnIsToolLessAndPlainRunSkipsIt(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"x"}`, 10, 4),
		textResp("final answer under pressure", 5, 2),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys",
		WithHooks(rec.hooks()),
		WithLimits(Limits{MaxIterations: 1}))
	tr := NewTranscript()
	em := runEmitter{tr: tr, obs: llmkit.Observers(tr)}
	out, err := r.run(context.Background(), em, nil, "task", nil, finalizationPrompt(nil), nil, nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !out.Truncated() || out.TruncationReason != TruncMaxIterations {
		t.Fatalf("Truncated=%v reason=%q, want TruncMaxIterations", out.Truncated(), out.TruncationReason)
	}
	if !out.Finalized {
		t.Fatal("Outcome.Finalized = false, want true after the finalization turn")
	}
	// The finalization completion is the second one: the Before hook must show
	// it carrying NO tools.
	if len(rec.beforeReqs) != 2 {
		t.Fatalf("before-hook requests = %d, want 2 (main + finalization)", len(rec.beforeReqs))
	}
	if n := len(rec.beforeReqs[1].Tools); n != 0 {
		t.Errorf("finalization request carried %d tools, want 0", n)
	}
	// The finalization turn's text became the final answer.
	if out.FinalText != "final answer under pressure" {
		t.Errorf("FinalText = %q, want the finalization turn's output", out.FinalText)
	}

	// Plain Run has no finalize prompt: the same stop condition must NOT take
	// a finalization turn.
	fc2 := newFakeClient(
		toolResp("c1", "echo", `{"v":"x"}`, 10, 4),
		textResp("unused", 5, 2),
	)
	r2 := NewRunner(fc2, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{MaxIterations: 1}))
	out2, err2 := r2.Run(context.Background(), "task")
	if ierr := incompleteErr(out2, err2, TruncMaxIterations); ierr != nil {
		t.Fatalf("Run: %v", ierr)
	}
	if out2.Finalized {
		t.Error("Outcome.Finalized = true on the plain Run path, want false")
	}
}

// TestRunJSON_RepairTurnIsSingleFreshUserTurn verifies RunJSON's single
// repair pass goes through the Before/After hook pair at the next step with
// the parse-failure prompt as its only message.
func TestRunJSON_RepairTurnIsSingleFreshUserTurn(t *testing.T) {
	fc := newFakeClient(
		textResp("not json at all", 5, 5),
		textResp(validItemJSON, 5, 5),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, nil, "sys", WithHooks(rec.hooks()))

	var out item
	if _, err := r.RunJSON(context.Background(), "task", nil, &out); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	want := []string{"before:1", "after:1", "before:2", "after:2"}
	if len(rec.events) != len(want) {
		t.Fatalf("events = %v, want %v", rec.events, want)
	}
	for i := range want {
		if rec.events[i] != want[i] {
			t.Errorf("events[%d] = %q, want %q", i, rec.events[i], want[i])
		}
	}
	// The repair request is a single fresh user turn demanding valid JSON.
	if len(rec.beforeReqs[1].Messages) != 1 {
		t.Fatalf("repair request carried %d messages, want 1 (history-less single turn)", len(rec.beforeReqs[1].Messages))
	}
	if !strings.Contains(rec.beforeReqs[1].Messages[0].Text(), "failed to parse as JSON") {
		t.Errorf("repair prompt missing the parse-failure note:\n%s", rec.beforeReqs[1].Messages[0].Text())
	}
}

// TestJSONLOnErr_FiresOnUnwritableDir verifies sink write failures surface
// through the onErr callback the sink was constructed with — never through
// Hooks (TranscriptError is gone) and never by failing the run.
func TestJSONLOnErr_FiresOnUnwritableDir(t *testing.T) {
	// Point the sink's dir at an existing FILE: creating anything under it
	// must fail on the first event.
	dir := t.TempDir()
	fileAsDir := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(fileAsDir, []byte("x"), 0o644); err != nil {
		t.Fatalf("setup: %v", err)
	}

	var mu sync.Mutex
	var sinkErrs []error
	fc := newFakeClient(textResp("done", 1, 1))
	r := NewRunner(fc, nil, "sys",
		WithObserver(JSONL(fileAsDir, func(err error) {
			mu.Lock()
			defer mu.Unlock()
			sinkErrs = append(sinkErrs, err)
		})))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run must not fail on a broken sink path: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(sinkErrs) == 0 {
		t.Fatal("sink onErr never fired despite the unusable path")
	}
	for i, serr := range sinkErrs {
		if serr == nil {
			t.Errorf("sinkErrs[%d] = nil", i)
		}
		if !strings.Contains(serr.Error(), fileAsDir) {
			t.Errorf("sinkErrs[%d] = %v, want a path-qualified error", i, serr)
		}
	}
}

// TestHooks_FullRunSequence pins the FULL hook sequence of one RunJSON: a
// tool-call turn, an unparseable final answer, then the repair completion's
// Before/After pair. The repair's step continues the parent run's sequence:
// the parent recorded steps 1 and 2, so the repair completion reports step
// 3 — consumers joining hooks and transcript events on Step see one
// monotonic sequence.
func TestHooks_FullRunSequence(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"hi"}`, 10, 4),
		textResp("this is not json", 5, 5),
		textResp(validItemJSON, 5, 5),
	)
	rec := newHookRecorder()
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(rec.hooks()))

	var out item
	if _, err := r.RunJSON(context.Background(), "task", nil, &out); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	want := []string{
		"before:1", "after:1", // main turn 1
		"start:1:echo", "end:1:echo", // tool lifecycle
		"before:2", "after:2", // main turn 2 (unparseable final answer)
		"before:3", "after:3", // repair completion — continues the parent sequence
	}
	if len(rec.events) != len(want) {
		t.Fatalf("events = %v, want %v", rec.events, want)
	}
	for i := range want {
		if rec.events[i] != want[i] {
			t.Errorf("events[%d] = %q, want %q", i, rec.events[i], want[i])
		}
	}
}

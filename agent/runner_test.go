package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// echoTool returns whatever it's given, or errors when asked to.
type echoTool struct {
	name    string
	failMsg string // non-empty => Run returns this as an error
}

func (e echoTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{
		Name:        e.name,
		Description: "echo",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"v":{"type":"string"}}}`),
	}
}

func (e echoTool) Run(ctx context.Context, args json.RawMessage) (string, error) {
	if e.failMsg != "" {
		return "", errors.New(e.failMsg)
	}
	return "echo:" + string(args), nil
}

func TestRun_CleanFinish(t *testing.T) {
	fc := newFakeClient(textResp("done", 10, 5))
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Truncated() {
		t.Errorf("expected not truncated, got reason %q", out.TruncationReason)
	}
	if out.FinalText != "done" {
		t.Errorf("FinalText = %q, want done", out.FinalText)
	}
	if out.Iterations != 1 {
		t.Errorf("Iterations = %d, want 1", out.Iterations)
	}
	if out.Usage.InputTokens != 10 || out.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v, want {10 5}", out.Usage)
	}
	// Transcript should have a request + assistant event.
	if got := len(out.Transcript.Events); got != 2 {
		t.Errorf("transcript events = %d, want 2", got)
	}
	// A normal text-only final turn must not trigger the
	// empty-turn nudge: zero nudges, no nudge message in the conversation.
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleUser && m.Text() == emptyTurnNudge {
			t.Error("nudge message present for a normal text-only final turn")
		}
	}
}

// TestRun_EmptyTurnNudgeThenProse verifies the empty-turn nudge for the plain (non-
// JSON) Run path: a think-only turn (zero tool calls, empty after stripping)
// is nudged, and the model's subsequent real prose becomes FinalText.
func TestRun_EmptyTurnNudgeThenProse(t *testing.T) {
	fc := newFakeClient(
		thinkOnlyResp("mulling it over", 10, 5),
		textResp("here is my answer", 8, 3),
	)
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "here is my answer" {
		t.Errorf("FinalText = %q, want %q", out.FinalText, "here is my answer")
	}
	if out.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2", out.Iterations)
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

func TestRun_ToolCallThenFinish(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"hi"}`, 10, 4),
		textResp("answer", 8, 3),
	)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "answer" {
		t.Errorf("FinalText = %q", out.FinalText)
	}
	if out.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2", out.Iterations)
	}
	// Cumulative usage across both turns.
	if out.Usage.InputTokens != 18 || out.Usage.OutputTokens != 7 {
		t.Errorf("Usage = %+v, want {18 7}", out.Usage)
	}
	// Verify the second request carried the tool result back to the model.
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fc.requests))
	}
	last := fc.requests[1].Messages
	foundResult := false
	for _, m := range last {
		if m.Role == llmkit.RoleToolResult && m.ToolCallID == "c1" {
			foundResult = true
			if !strings.Contains(m.Text(), "echo:") {
				t.Errorf("tool result content = %q", m.Text())
			}
		}
	}
	if !foundResult {
		t.Error("second request did not carry the tool result")
	}
}

func TestRun_MaxIterations(t *testing.T) {
	// Always request a tool, never finish -> must hit the iteration cap.
	steps := make([]scriptStep, 0, 10)
	for i := 0; i < 10; i++ {
		steps = append(steps, toolResp("c", "echo", `{"v":"x"}`, 1, 1))
	}
	fc := newFakeClient(steps...)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{MaxIterations: 3}))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !out.Truncated() || out.TruncationReason != TruncMaxIterations {
		t.Errorf("expected max_iterations truncation, got truncated=%v reason=%q", out.Truncated(), out.TruncationReason)
	}
	if out.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3", out.Iterations)
	}
}

func TestRun_TokenBudget(t *testing.T) {
	// Tiny budget: first turn already exceeds it, loop should stop cleanly.
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"x"}`, 50, 50), // 100 tokens > budget 10
		textResp("never reached", 1, 1),
	)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{TokenBudget: 10}))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !out.Truncated() || out.TruncationReason != TruncTokenBudget {
		t.Errorf("expected token_budget truncation, got truncated=%v reason=%q", out.Truncated(), out.TruncationReason)
	}
	// Only the first completion should have run.
	if fc.callCount() != 1 {
		t.Errorf("calls = %d, want 1 (budget should stop before second call)", fc.callCount())
	}
}

// TestRun_MaxTokens pins the documented behavior that plain Run (not just
// RunJSON) makes one continuation completion when a turn stops at the output
// token cap and stitches the two halves into the final answer. This is a public
// API contract: changing it alters Run's iteration/token cost, so it must not
// drift silently.
func TestRun_MaxTokens(t *testing.T) {
	fc := newFakeClient(
		maxTokensResp("The answer is fort-", 5, 5), // cut off mid-word at the cap
		textResp("two.", 5, 5),                     // continuation finishes it
	)
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "The answer is fort-two." {
		t.Errorf("FinalText = %q, want the two halves stitched", out.FinalText)
	}
	// Both the initial truncated completion and its continuation ran this turn.
	if fc.callCount() != 2 {
		t.Errorf("calls = %d, want 2 (initial + continuation)", fc.callCount())
	}
	if out.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2 (continuation counts)", out.Iterations)
	}
	// A completed continuation is a clean finish, not a truncation.
	if out.Truncated() {
		t.Errorf("Truncated() = true, want false after a successful continuation")
	}
}

func TestRun_ToolErrorFedBackToModel(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "boom", `{}`, 1, 1),
		textResp("recovered", 1, 1),
	)
	r := NewRunner(fc, []Tool{echoTool{name: "boom", failMsg: "disk on fire"}}, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run should not fail on tool error: %v", err)
	}
	if out.FinalText != "recovered" {
		t.Errorf("FinalText = %q, want recovered", out.FinalText)
	}
	// The second request must carry the ERROR-prefixed tool result with IsError.
	last := fc.requests[1].Messages
	var tr *llmkit.Message
	for i := range last {
		if last[i].Role == llmkit.RoleToolResult {
			tr = &last[i]
		}
	}
	if tr == nil {
		t.Fatal("no tool result in second request")
	}
	if !tr.IsError {
		t.Error("tool result IsError = false, want true")
	}
	if !strings.HasPrefix(tr.Text(), "ERROR:") || !strings.Contains(tr.Text(), "disk on fire") {
		t.Errorf("tool result content = %q, want ERROR: ... disk on fire", tr.Text())
	}
}

func TestRun_UnknownToolFedBackToModel(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "ghost", `{}`, 1, 1),
		textResp("ok", 1, 1),
	)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	last := fc.requests[1].Messages
	found := false
	for _, m := range last {
		if m.Role == llmkit.RoleToolResult && m.IsError && strings.Contains(m.Text(), "unknown tool") {
			found = true
		}
	}
	if !found {
		t.Error("unknown tool not reported back to model as an error result")
	}
	_ = out
}

func TestRun_CompletionErrorAborts(t *testing.T) {
	fc := newFakeClient(scriptStep{err: errors.New("network down")})
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err == nil {
		t.Fatal("expected error from failed completion")
	}
	if out.Transcript == nil {
		t.Error("Outcome.Transcript should be non-nil even on error")
	}
}

func TestRun_ContextCancellation(t *testing.T) {
	fc := newFakeClient(textResp("unused", 1, 1))
	r := NewRunner(fc, nil, "sys")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	out, err := r.Run(ctx, "task")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if out == nil || out.Transcript == nil {
		t.Error("expected non-nil outcome with transcript")
	}
}

func TestLimits_Resolve(t *testing.T) {
	got := Limits{}.resolve()
	if got.MaxIterations != DefaultMaxIterations {
		t.Errorf("MaxIterations = %d, want %d", got.MaxIterations, DefaultMaxIterations)
	}
	if got.TokenBudget != DefaultTokenBudget {
		t.Errorf("TokenBudget = %d, want %d", got.TokenBudget, DefaultTokenBudget)
	}
	// Negative means unlimited and must survive resolve unchanged.
	neg := Limits{MaxIterations: -1, TokenBudget: -1}.resolve()
	if neg.MaxIterations != -1 || neg.TokenBudget != -1 {
		t.Errorf("negative limits altered: %+v", neg)
	}
}

// TestRun_PrefixStability verifies the property prompt caching depends on: the
// request the loop sends at iteration N+1 must extend iteration N's request
// without rewriting any of it. Concretely, across consecutive completions the
// System string and the tool definitions must be byte-identical, and iteration
// N's Messages must be a strict prefix (element-wise byte-equal under JSON
// serialization) of iteration N+1's. Any drift here silently turns every
// provider-side cache lookup into a miss.
func TestRun_PrefixStability(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"one"}`, 10, 4),
		toolResp("c2", "grep", `{"v":"two"}`, 12, 4),
		textResp("answer", 8, 3),
	)
	tools := []Tool{echoTool{name: "echo"}, echoTool{name: "grep"}, echoTool{name: "read"}}
	r := NewRunner(fc, tools, "stable system prompt")

	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fc.requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(fc.requests))
	}

	marshal := func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return string(b)
	}

	for i := 1; i < len(fc.requests); i++ {
		prev, cur := fc.requests[i-1], fc.requests[i]
		if cur.System != prev.System {
			t.Errorf("iteration %d: System changed:\n  prev %q\n  cur  %q", i+1, prev.System, cur.System)
		}
		if got, want := marshal(cur.Tools), marshal(prev.Tools); got != want {
			t.Errorf("iteration %d: tool definitions changed (order or content):\n  prev %s\n  cur  %s", i+1, want, got)
		}
		if len(cur.Messages) <= len(prev.Messages) {
			t.Fatalf("iteration %d: messages did not grow (%d -> %d)", i+1, len(prev.Messages), len(cur.Messages))
		}
		// Element-wise: the previous conversation must be an untouched prefix.
		for j := range prev.Messages {
			if got, want := marshal(cur.Messages[j]), marshal(prev.Messages[j]); got != want {
				t.Errorf("iteration %d: message %d rewritten:\n  prev %s\n  cur  %s", i+1, j, want, got)
			}
		}
		// The serialized previous request's message list must be a byte prefix of
		// the current one (the JSON array shares everything but the closing
		// bracket), which is the strongest cheap statement of prefix stability.
		prevJSON, curJSON := marshal(prev.Messages), marshal(cur.Messages)
		if !strings.HasPrefix(curJSON, strings.TrimSuffix(prevJSON, "]")) {
			t.Errorf("iteration %d: serialized messages are not an append-only extension", i+1)
		}
	}
}

// TestRun_AccumulatesCacheUsage verifies cache-read/creation token counts are
// summed across iterations alongside input/output.
func TestRun_AccumulatesCacheUsage(t *testing.T) {
	step1 := toolResp("c1", "echo", `{}`, 100, 5)
	step1.resp.Usage.CacheCreationInputTokens = 80
	step2 := textResp("done", 120, 6)
	step2.resp.Usage.CacheReadInputTokens = 90
	step2.resp.Usage.CacheCreationInputTokens = 10

	fc := newFakeClient(step1, step2)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Usage.InputTokens != 220 || out.Usage.OutputTokens != 11 {
		t.Errorf("Usage in/out = %d/%d, want 220/11", out.Usage.InputTokens, out.Usage.OutputTokens)
	}
	if out.Usage.CacheReadInputTokens != 90 {
		t.Errorf("CacheReadInputTokens = %d, want 90", out.Usage.CacheReadInputTokens)
	}
	if out.Usage.CacheCreationInputTokens != 90 {
		t.Errorf("CacheCreationInputTokens = %d, want 90", out.Usage.CacheCreationInputTokens)
	}
}

// healthEchoTool is an echo-style tool whose Run returns a *ToolHealthError
// when configured to. It is the test vehicle for the [Hooks.ToolHealth]
// dispatch seam: a health error must reach the hook, a plain error must not.
type healthEchoTool struct {
	name   string
	health *ToolHealthError // non-nil => Run returns this
	plain  string           // non-empty => Run returns errors.New(plain)
}

func (e healthEchoTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{
		Name:        e.name,
		Description: "health-echo",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
}

func (e healthEchoTool) Run(ctx context.Context, args json.RawMessage) (string, error) {
	if e.health != nil {
		return "", e.health
	}
	if e.plain != "" {
		return "", errors.New(e.plain)
	}
	return "ok", nil
}

// TestHook_ToolHealth_FiresOnToolHealthError verifies that a tool returning a
// *ToolHealthError fires Hooks.ToolHealth with the tool name and the SAME
// *ToolHealthError pointer (Reason and Err preserved), in addition to the
// error text being fed back to the model.
func TestHook_ToolHealth_FiresOnToolHealthError(t *testing.T) {
	healthErr := &ToolHealthError{
		Reason: "container runtime unavailable",
		Err:    errors.New("podman not found"),
	}
	fc := newFakeClient(
		toolResp("c1", "broken", `{}`, 10, 4),
		textResp("done", 5, 2),
	)

	var hookTool string
	var hookErr *ToolHealthError
	r := NewRunner(fc, []Tool{healthEchoTool{name: "broken", health: healthErr}}, "sys",
		WithHooks(Hooks{ToolHealth: func(_ context.Context, tool string, he *ToolHealthError) {
			hookTool = tool
			hookErr = he
		}}))
	_, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if hookTool != "broken" {
		t.Errorf("hook called with tool %q, want %q", hookTool, "broken")
	}
	if hookErr != healthErr {
		t.Errorf("hook called with error %p, want the original %p", hookErr, healthErr)
	}
	if hookErr.Reason != "container runtime unavailable" {
		t.Errorf("hook reason = %q", hookErr.Reason)
	}
	if hookErr.Err == nil || hookErr.Err.Error() != "podman not found" {
		t.Errorf("hook Err = %v, want the wrapped podman error", hookErr.Err)
	}
}

// TestHook_ToolHealth_NotFiredOnPlainError is the central infra-vs-
// recoverable assertion: an ordinary model-recoverable tool error (e.g. bad
// args, file-not-found) must NOT fire Hooks.ToolHealth. Only *ToolHealthError
// reaches it.
func TestHook_ToolHealth_NotFiredOnPlainError(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "plain", `{}`, 10, 4),
		textResp("done", 5, 2),
	)

	called := false
	r := NewRunner(fc, []Tool{healthEchoTool{name: "plain", plain: "bad args"}}, "sys",
		WithHooks(Hooks{ToolHealth: func(context.Context, string, *ToolHealthError) { called = true }}))
	_, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if called {
		t.Error("ToolHealth fired for a plain errors.New — must only fire on *ToolHealthError")
	}
}

// TestHook_ToolHealth_NilIsNoop verifies that a nil ToolHealth func (or no
// WithHooks option at all) runs cleanly with no overhead — no panic, no
// extra state.
func TestHook_ToolHealth_NilIsNoop(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "broken", `{}`, 5, 2),
		textResp("done", 3, 1),
	)
	// No hooks registered at all.
	r := NewRunner(fc, []Tool{healthEchoTool{name: "broken", health: &ToolHealthError{
		Reason: "container runtime missing",
	}}}, "sys")
	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run without hooks: %v", err)
	}

	// Hooks registered with a nil ToolHealth func.
	r2 := NewRunner(fc, []Tool{healthEchoTool{name: "broken", health: &ToolHealthError{
		Reason: "container runtime missing",
	}}}, "sys", WithHooks(Hooks{}))
	if _, err := r2.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run with nil ToolHealth hook: %v", err)
	}
}

// TestRunTool_ToolHealth_SkippedOnCancelledCtx verifies the dispatch seam
// does NOT fire the health hook when ctx is already cancelled: a failure
// caused by run teardown/cancellation is not a harness-tooling problem, even
// when the tool returns a *ToolHealthError.
func TestRunTool_ToolHealth_SkippedOnCancelledCtx(t *testing.T) {
	called := false
	r := NewRunner(newFakeClient(), []Tool{healthEchoTool{name: "broken", health: &ToolHealthError{
		Reason: "container runtime unavailable",
	}}}, "sys", WithHooks(Hooks{ToolHealth: func(context.Context, string, *ToolHealthError) { called = true }}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, isErr := r.runTool(ctx, llmkit.ToolCall{Name: "broken", Arguments: json.RawMessage("{}")}, 1)
	if !isErr {
		t.Fatal("a ToolHealthError must still be returned as an error result")
	}
	if called {
		t.Error("ToolHealth must NOT fire when ctx is already cancelled")
	}
}

// stopErrorResp builds a response with StopReason == StopError (refusal,
// safety filter, recitation) and no tool calls.
func stopErrorResp(text string, in, out int64) scriptStep {
	return scriptStep{resp: llmkit.Response{
		Text:       text,
		StopReason: llmkit.StopError,
		Usage:      llmkit.Usage{InputTokens: in, OutputTokens: out},
	}}
}

// TestRun_StopErrorYieldsTypedError verifies that a final turn that
// ends with StopError and no tool calls must surface *StopReasonError, never a
// clean Outcome that records refusal prose as the answer.
func TestRun_StopErrorYieldsTypedError(t *testing.T) {
	fc := newFakeClient(stopErrorResp("I cannot help with that.", 10, 5))
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	var stopErr *StopReasonError
	if !errors.As(err, &stopErr) {
		t.Fatalf("Run error = %v, want *StopReasonError", err)
	}
	if stopErr.StopReason != llmkit.StopError {
		t.Errorf("StopReason = %q, want %q", stopErr.StopReason, llmkit.StopError)
	}
	if stopErr.Text != "I cannot help with that." {
		t.Errorf("Text = %q, want refusal prose", stopErr.Text)
	}
	if stopErr.Outcome == nil || out == nil {
		t.Fatal("partial Outcome must be attached and returned")
	}
	if stopErr.Outcome.Usage.InputTokens != 10 {
		t.Errorf("partial Outcome usage lost: %+v", stopErr.Outcome.Usage)
	}
}

// TestRun_StopErrorAfterToolsYieldsTypedError covers the multi-turn case: tool
// turns succeed, then the model refuses. The stale text from earlier turns
// must not be presented as a clean answer.
func TestRun_StopErrorAfterToolsYieldsTypedError(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "ghost", `{}`, 5, 2),
		stopErrorResp("", 5, 2),
	)
	r := NewRunner(fc, nil, "sys")

	_, err := r.Run(context.Background(), "task")
	var stopErr *StopReasonError
	if !errors.As(err, &stopErr) {
		t.Fatalf("Run error = %v, want *StopReasonError", err)
	}
	if stopErr.Outcome.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2", stopErr.Outcome.Iterations)
	}
}

// TestRun_ToolOnlyFinalTurnEmptiesFinalText pins the Outcome.FinalText
// contract: the text of the LAST completion of the run, empty when that
// completion produced no text — never text carried over from an earlier
// turn. An earlier turn's "thinking out loud" must not leak into the
// outcome as a fake answer. The two trailing empty responses exhaust the
// empty-turn-nudge cap (maxEmptyTurnNudges=2) before the loop breaks, so
// the assertions below still exercise a genuine "model finished with
// nothing" turn.
func TestRun_ToolOnlyFinalTurnEmptiesFinalText(t *testing.T) {
	withText := toolResp("c1", "ghost", `{}`, 5, 2)
	withText.resp.Text = "thinking out loud"
	emptyFinal := scriptStep{resp: llmkit.Response{StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 5, OutputTokens: 1}}}
	fc := newFakeClient(
		withText,
		emptyFinal,
		emptyFinal,
		emptyFinal,
	)
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "" {
		t.Errorf("FinalText = %q, want empty: the final turn produced no text and earlier-turn text must never leak", out.FinalText)
	}
}

// TestRun_TextFinalTurnCarriesFinalText is the positive counterpart: a
// final turn that emits text makes that text FinalText.
func TestRun_TextFinalTurnCarriesFinalText(t *testing.T) {
	fc := newFakeClient(textResp("the answer", 10, 5))
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "the answer" {
		t.Errorf("FinalText = %q, want 'the answer'", out.FinalText)
	}
}

// TestRun_FinalTextPreservesInlineThink pins that FinalText carries the
// completion's raw text verbatim, inline <think> span included: stripping
// is the consumer's job (llmkit.StripThinkBlocks), and RunJSON strips
// separately on its parse path. Without this pin, "helpfully" pre-stripping
// FinalText at the assignment site would be invisible to the hermetic suite
// and surface only as live-lane skips.
func TestRun_FinalTextPreservesInlineThink(t *testing.T) {
	const raw = "<think>17+25 is 42</think>42"
	fc := newFakeClient(textResp(raw, 10, 5))
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != raw {
		t.Errorf("FinalText = %q, want the raw completion %q verbatim (think span included)", out.FinalText, raw)
	}
}

// TestRun_TextAfterToolsReplacesFinalText pins the replace-not-carry rule:
// an earlier turn that carries text alongside a tool call, followed by a
// final text turn, leaves the LAST completion's text standing.
func TestRun_TextAfterToolsReplacesFinalText(t *testing.T) {
	toolWithText := toolResp("c1", "ghost", `{}`, 5, 2)
	toolWithText.resp.Text = "earlier prose"
	fc := newFakeClient(
		toolWithText,
		textResp("final answer", 5, 2),
	)
	r := NewRunner(fc, nil, "sys")
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "final answer" {
		t.Errorf("FinalText = %q, want the LAST completion's text", out.FinalText)
	}
}

// TestAssistantMessageTextGuard pins how a completion becomes the assistant
// history turn: blocks are recorded verbatim, and when a Response carries
// text but its Blocks omit any text block — violating the llmkit.Response
// invariant (Text equals the concatenation of BlockText blocks) — the
// surfaced text is appended, so history always agrees with what llmkit.Stream
// and FinalText delivered. Think-only responses (Text == "") stay verbatim:
// no empty text block is invented.
func TestAssistantMessageTextGuard(t *testing.T) {
	thinking := llmkit.Block{Kind: llmkit.BlockThinking, Text: "why", Provider: "anthropic"}
	tests := []struct {
		name string
		resp llmkit.Response
		want []llmkit.Block
	}{
		{
			name: "blocks_without_text_block_gets_text_appended",
			resp: llmkit.Response{Blocks: []llmkit.Block{thinking}, Text: "the answer"},
			want: []llmkit.Block{thinking, {Kind: llmkit.BlockText, Text: "the answer"}},
		},
		{
			name: "think_only_stays_verbatim",
			resp: llmkit.Response{Blocks: []llmkit.Block{thinking}},
			want: []llmkit.Block{thinking},
		},
		{
			name: "text_block_present_not_duplicated",
			resp: llmkit.Response{Blocks: []llmkit.Block{thinking, {Kind: llmkit.BlockText, Text: "the answer"}}, Text: "the answer"},
			want: []llmkit.Block{thinking, {Kind: llmkit.BlockText, Text: "the answer"}},
		},
		{
			name: "no_blocks_degrades_to_text",
			resp: llmkit.Response{Text: "the answer"},
			want: []llmkit.Block{{Kind: llmkit.BlockText, Text: "the answer"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := assistantMessage(tt.resp)
			if !reflect.DeepEqual(msg.Content, tt.want) {
				t.Fatalf("Content = %#v, want %#v", msg.Content, tt.want)
			}
		})
	}
}

// TestRun_TextWithoutTextBlockReachesHistory is the end-to-end pin for the
// assistant history guard: a scripted response shaped like the pre-fix
// structured-output finalize (a thinking block plus Text, no text block)
// must still record the surfaced text into the conversation history, so the
// next request carries what llmkit.Stream delivered to the caller.
func TestRun_TextWithoutTextBlockReachesHistory(t *testing.T) {
	thinking := llmkit.Block{
		Kind:     llmkit.BlockThinking,
		Text:     "calling the tool",
		Provider: "x",
		Raw:      json.RawMessage(`{"type":"thinking","thinking":"calling the tool","signature":"sig-1"}`),
	}
	step1 := scriptStep{resp: llmkit.Response{
		Blocks:     []llmkit.Block{thinking},
		Text:       "calling the tool",
		StopReason: llmkit.StopToolUse,
		ToolCalls:  []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"v":"hi"}`)}},
		Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 4},
	}}
	fc := newFakeClient(step1, textResp("answer", 8, 3))
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")
	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	var asst llmkit.Message
	found := false
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleAssistant {
			asst, found = m, true
			break
		}
	}
	if !found {
		t.Fatal("no assistant message in the follow-up request")
	}
	if len(asst.Content) != 2 {
		t.Fatalf("assistant Content = %d blocks, want 2 (thinking + text)", len(asst.Content))
	}
	if asst.Content[0].Kind != llmkit.BlockThinking {
		t.Errorf("block 0 Kind = %q, want thinking", asst.Content[0].Kind)
	}
	if asst.Content[1].Kind != llmkit.BlockText || asst.Content[1].Text != "calling the tool" {
		t.Errorf("block 1 = %+v, want the surfaced text block", asst.Content[1])
	}
}

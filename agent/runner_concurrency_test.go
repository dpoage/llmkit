package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// hangingTool blocks until its context is done, then reports the context
// error. It is the test vehicle for [WithToolTimeout]: the per-call deadline
// must cancel it while the run's own context stays alive.
type hangingTool struct{ name string }

func (h hangingTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: h.name, Description: "hangs until cancelled", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (h hangingTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

// staggerTool sleeps for its delay (honoring cancellation) before returning
// its own distinctive result string, so result↔call association is
// observable in parallel-dispatch tests.
type staggerTool struct {
	name   string
	delay  time.Duration
	result string
}

func (s staggerTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: s.name, Description: "staggered", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (s staggerTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-time.After(s.delay):
		return s.result, nil
	}
}

// panicTool always panics from Run — the vehicle for WithParallelTools panic
// isolation.
type panicTool struct{ name string }

func (p panicTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: p.name, Description: "panics", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (p panicTool) Run(context.Context, json.RawMessage) (string, error) {
	panic("tool exploded")
}

// toolCallsResp builds one response requesting SEVERAL tool calls in order.
func toolCallsResp(calls ...llmkit.ToolCall) scriptStep {
	return scriptStep{resp: llmkit.Response{
		Text:       "calling tools",
		StopReason: llmkit.StopToolUse,
		ToolCalls:  calls,
		Usage:      llmkit.Usage{InputTokens: 5, OutputTokens: 5},
	}}
}

// TestRun_ToolTimeout_CancelsHangingTool verifies WithToolTimeout gives every
// Tool.Run a per-call deadline: a tool that blocks forever is cancelled, its
// timeout is fed back to the model as an ERROR tool result, the loop
// continues to completion, and the run's own context is unaffected.
func TestRun_ToolTimeout_CancelsHangingTool(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "hang", `{}`, 10, 4),
		textResp("recovered", 5, 2),
	)
	r := NewRunner(fc, []Tool{hangingTool{name: "hang"}}, "sys",
		WithToolTimeout(50*time.Millisecond))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	start := time.Now()
	out, err := r.Run(ctx, "task")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "recovered" {
		t.Fatalf("FinalText = %q, want the loop to continue past the timeout", out.FinalText)
	}
	if ctx.Err() != nil {
		t.Errorf("parent context was cancelled by the tool timeout: %v", ctx.Err())
	}
	if elapsed > 2*time.Second {
		t.Errorf("run took %v, want the hanging tool cancelled promptly", elapsed)
	}
	// The model received the timeout as an error tool result naming the tool
	// and the deadline.
	var tr *llmkit.Message
	for i := range fc.requests[1].Messages {
		if fc.requests[1].Messages[i].Role == llmkit.RoleToolResult && fc.requests[1].Messages[i].ToolCallID == "c1" {
			tr = &fc.requests[1].Messages[i]
		}
	}
	if tr == nil {
		t.Fatal("no tool result carried back after the timeout")
	}
	want := "ERROR: tool hang timed out after 50ms"
	if tr.Text() != want {
		t.Errorf("tool result = %q, want %q", tr.Text(), want)
	}
	if !tr.IsError {
		t.Error("timeout tool result IsError = false, want true")
	}
}

// ctxObsTool closes started when Run begins and reports the context error it
// observed when its context was done.
type ctxObsTool struct {
	name     string
	started  chan struct{}
	observed chan error
	once     sync.Once
}

func (c *ctxObsTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: c.name, Description: "observes ctx", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (c *ctxObsTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	c.once.Do(func() { close(c.started) })
	<-ctx.Done()
	err := ctx.Err()
	select {
	case c.observed <- err:
	default:
	}
	return "", err
}

// TestRun_ToolTimeout_DerivesFromRunContext pins that the per-tool deadline
// derives FROM the run's context: cancelling the run while a tool is in
// flight under WithToolTimeout must reach the tool as context.Canceled — not
// leave it blocked until its own deadline fires off a detached parent.
func TestRun_ToolTimeout_DerivesFromRunContext(t *testing.T) {
	tool := &ctxObsTool{name: "watch", started: make(chan struct{}), observed: make(chan error, 1)}
	fc := newFakeClient(
		toolResp("c1", "watch", `{}`, 1, 1),
		textResp("unused", 1, 1),
	)
	r := NewRunner(fc, []Tool{tool}, "sys", WithToolTimeout(10*time.Second))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := r.Run(ctx, "task")
		done <- err
	}()

	<-tool.started // the call is in flight under its 10s deadline
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Run err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("run did not return after cancellation — the per-tool deadline is not derived from the run context")
	}
	select {
	case observed := <-tool.observed:
		if !errors.Is(observed, context.Canceled) {
			t.Errorf("tool observed %v, want context.Canceled (deadline must inherit the run ctx)", observed)
		}
	case <-time.After(time.Second):
		t.Fatal("tool never observed cancellation")
	}
}

// TestRun_ParallelTools_OrderAndTiming verifies WithParallelTools dispatches
// one turn's calls concurrently (wall time approaches the slowest call, not
// the sum) while each result lands under ITS OWN call ID in the model's
// original call order — in both the follow-up request and the transcript.
// Each tool returns a distinctive payload and finishes in an order different
// from the call order (fast → mid → slow vs c1,c2,c3), so a mis-slotting or
// completion-order append cannot pass. The default sequential path is pinned
// in the same test.
func TestRun_ParallelTools_OrderAndTiming(t *testing.T) {
	calls := []llmkit.ToolCall{
		{ID: "c1", Name: "slow", Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: "mid", Arguments: json.RawMessage(`{}`)},
		{ID: "c3", Name: "fast", Arguments: json.RawMessage(`{}`)},
	}
	tools := []Tool{
		staggerTool{name: "slow", delay: 250 * time.Millisecond, result: "R_slow"},
		staggerTool{name: "mid", delay: 150 * time.Millisecond, result: "R_mid"},
		staggerTool{name: "fast", delay: 30 * time.Millisecond, result: "R_fast"},
	}
	wantByCall := map[string]string{"c1": "R_slow", "c2": "R_mid", "c3": "R_fast"}

	// runOnce executes one scripted run and returns the elapsed wall time,
	// the per-call result content in the follow-up request, and the per-call
	// result content in the transcript.
	runOnce := func(parallel bool) (time.Duration, map[string]string, map[string]string) {
		fc := newFakeClient(
			toolCallsResp(calls...),
			textResp("all done", 5, 2),
		)
		opts := []Option{}
		if parallel {
			opts = append(opts, WithParallelTools())
		}
		r := NewRunner(fc, tools, "sys", opts...)

		start := time.Now()
		out, err := r.Run(context.Background(), "task")
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("Run(parallel=%v): %v", parallel, err)
		}
		if out.FinalText != "all done" {
			t.Fatalf("FinalText = %q", out.FinalText)
		}
		if len(fc.requests) != 2 {
			t.Fatalf("requests = %d, want 2", len(fc.requests))
		}
		history := map[string]string{}
		for _, m := range fc.requests[1].Messages {
			if m.Role == llmkit.RoleToolResult {
				history[m.ToolCallID] = m.Text()
			}
		}
		transcript := map[string]string{}
		for _, ev := range out.Transcript.Record {
			if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil {
				transcript[ev.ToolRun.Call.ID] = ev.ToolRun.Result
			}
		}
		return elapsed, history, transcript
	}

	// Parallel: completion order is fast→mid→slow (≠ call order), so per-ID
	// content assertions actually discriminate mis-slotting; wall time ≈ max.
	parElapsed, parHistory, parTranscript := runOnce(true)
	for id, want := range wantByCall {
		if got := parHistory[id]; got != want {
			t.Errorf("parallel history result for %s = %q, want %q (call→result misassociated)", id, got, want)
		}
		if got := parTranscript[id]; got != want {
			t.Errorf("parallel transcript result for %s = %q, want %q", id, got, want)
		}
	}
	if len(parHistory) != 3 || len(parTranscript) != 3 {
		t.Fatalf("parallel results incomplete: history=%v transcript=%v", parHistory, parTranscript)
	}
	if parElapsed >= 400*time.Millisecond {
		t.Errorf("parallel wall time = %v, want ≈ max(250ms), not the 430ms sum", parElapsed)
	}

	// Sequential (default): calls run one at a time, so wall time is at least
	// the sum; the same per-ID payloads must land in the same call order.
	seqElapsed, seqHistory, seqTranscript := runOnce(false)
	for id, want := range wantByCall {
		if got := seqHistory[id]; got != want {
			t.Errorf("sequential history result for %s = %q, want %q", id, got, want)
		}
		if got := seqTranscript[id]; got != want {
			t.Errorf("sequential transcript result for %s = %q, want %q", id, got, want)
		}
	}
	if len(seqHistory) != 3 || len(seqTranscript) != 3 {
		t.Fatalf("sequential results incomplete: history=%v transcript=%v", seqHistory, seqTranscript)
	}
	if seqElapsed < 400*time.Millisecond {
		t.Errorf("sequential wall time = %v, want >= the 430ms sum (calls must not overlap)", seqElapsed)
	}
}

// TestRun_ParallelTools_ErrorIsolation verifies that one immediately-failing
// call cannot disturb a slow sibling under WithParallelTools: the sibling's
// own result lands under its call ID and the run completes.
func TestRun_ParallelTools_ErrorIsolation(t *testing.T) {
	calls := []llmkit.ToolCall{
		{ID: "c1", Name: "boom", Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: "slowok", Arguments: json.RawMessage(`{}`)},
	}
	tools := []Tool{
		echoTool{name: "boom", failMsg: "kaput"},
		staggerTool{name: "slowok", delay: 200 * time.Millisecond, result: "R_ok"},
	}
	fc := newFakeClient(
		toolCallsResp(calls...),
		textResp("done", 5, 2),
	)
	r := NewRunner(fc, tools, "sys", WithParallelTools())

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	results := map[string]string{}
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult {
			results[m.ToolCallID] = m.Text()
		}
	}
	if got := results["c1"]; !strings.HasPrefix(got, "ERROR:") || !strings.Contains(got, "kaput") {
		t.Errorf("failing call result = %q, want the ERROR rendering of kaput", got)
	}
	if got := results["c2"]; got != "R_ok" {
		t.Errorf("sibling result = %q, want its own R_ok (one call's failure cancelled or replaced it)", got)
	}
}

// TestRun_ParallelTools_TimeoutIsolation verifies a call stuck past its
// per-tool deadline yields the timeout result while a fast sibling under the
// same WithParallelTools turn still returns its own result.
func TestRun_ParallelTools_TimeoutIsolation(t *testing.T) {
	calls := []llmkit.ToolCall{
		{ID: "c1", Name: "hang", Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: "fast", Arguments: json.RawMessage(`{}`)},
	}
	tools := []Tool{
		hangingTool{name: "hang"},
		staggerTool{name: "fast", delay: 5 * time.Millisecond, result: "R_fast"},
	}
	fc := newFakeClient(
		toolCallsResp(calls...),
		textResp("done", 5, 2),
	)
	r := NewRunner(fc, tools, "sys", WithParallelTools(), WithToolTimeout(80*time.Millisecond))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	results := map[string]string{}
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult {
			results[m.ToolCallID] = m.Text()
		}
	}
	if got := results["c1"]; got != "ERROR: tool hang timed out after 80ms" {
		t.Errorf("hung call result = %q, want the timeout rendering", got)
	}
	if got := results["c2"]; got != "R_fast" {
		t.Errorf("sibling result = %q, want its own R_fast", got)
	}
}

// TestRun_ParallelTools_PanicIsolation verifies a panicking Tool.Run under
// WithParallelTools does not abort the process: the panic is recovered in the
// call's goroutine, rendered as that call's error result, and the sibling's
// result plus the run's completion are unaffected.
func TestRun_ParallelTools_PanicIsolation(t *testing.T) {
	calls := []llmkit.ToolCall{
		{ID: "c1", Name: "panicky", Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: "echo", Arguments: json.RawMessage(`{"v":"hi"}`)},
	}
	tools := []Tool{panicTool{name: "panicky"}, echoTool{name: "echo"}}
	fc := newFakeClient(
		toolCallsResp(calls...),
		textResp("survived", 5, 2),
	)
	r := NewRunner(fc, tools, "sys", WithParallelTools())

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run must survive a panicking tool: %v", err)
	}
	if out.FinalText != "survived" {
		t.Fatalf("FinalText = %q, want the run to continue past the panic", out.FinalText)
	}
	results := map[string]string{}
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult {
			results[m.ToolCallID] = m.Text()
		}
	}
	want := "ERROR: tool panicky panicked: tool exploded"
	if got := results["c1"]; got != want {
		t.Errorf("panicking call result = %q, want %q", got, want)
	}
	if got := results["c2"]; got != "echo:{\"v\":\"hi\"}" {
		t.Errorf("sibling result = %q, want its own echo output", got)
	}
}

// alwaysToolClient is a concurrency-safe scripted client that answers every
// completion with one tool call, so N concurrent Runs on ONE Runner can share
// it without sharing script state.
type alwaysToolClient struct {
	mu sync.Mutex
	n  int
}

func (c *alwaysToolClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *alwaysToolClient) Complete(_ context.Context, _ llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.n++
	return llmkit.Response{
		Text:       "working",
		StopReason: llmkit.StopToolUse,
		ToolCalls:  []llmkit.ToolCall{{ID: fmt.Sprintf("c%d", c.n), Name: "echo", Arguments: json.RawMessage(`{}`)}},
		Usage:      llmkit.Usage{InputTokens: 1, OutputTokens: 1},
	}, nil
}

// TestRunner_ConcurrentRunsDistinctTranscripts is the -race guard for Runner
// reuse: N concurrent Run calls on ONE Runner (all Runner state fixed at
// construction; transcript, conversation, and budget state per-run) must all
// succeed and produce N distinct transcript files even when two runs derive
// the same autosave filename.
func TestRunner_ConcurrentRunsDistinctTranscripts(t *testing.T) {
	dir := t.TempDir()
	client := &alwaysToolClient{}
	r := NewRunner(client, []Tool{echoTool{name: "echo"}}, "sys",
		WithObserver(JSONL(dir, nil)),
		WithLimits(Limits{MaxIterations: 1}))

	const n = 10
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := r.Run(context.Background(), "task")
			errs[i] = err
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("run %d: %v", i, err)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != n {
		t.Fatalf("transcript files = %d, want %d", len(entries), n)
	}
	seen := make(map[string]bool, n)
	for _, e := range entries {
		name := e.Name()
		if seen[name] {
			t.Errorf("duplicate transcript file %q — two runs shared a stream", name)
		}
		seen[name] = true
	}
}

func TestRun_ThinkingBlocksForwardedAcrossToolTurns(t *testing.T) {
	thinking := llmkit.Block{
		Kind:     llmkit.BlockThinking,
		Text:     "I should verify with the tool",
		Provider: "x",
		Raw:      json.RawMessage(`{"type":"thinking","thinking":"I should verify","signature":"sig-abc"}`),
	}
	step1 := scriptStep{resp: llmkit.Response{
		Blocks:     []llmkit.Block{thinking, {Kind: llmkit.BlockText, Text: "calling the tool"}},
		Text:       "calling the tool",
		StopReason: llmkit.StopToolUse,
		ToolCalls:  []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"v":"hi"}`)}},
		Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 4},
	}}
	fc := newFakeClient(step1, textResp("answer", 8, 3))
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "answer" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	// The NEXT request's assistant message must carry the response's blocks
	// verbatim — thinking block first, provider and Raw bytes untouched —
	// with the tool result appended AFTER it.
	var asst llmkit.Message
	asstIdx := -1
	for i, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleAssistant {
			asst, asstIdx = m, i
			break
		}
	}
	if asstIdx < 0 {
		t.Fatal("no assistant message in the follow-up request")
	}
	if len(asst.Content) != 2 {
		t.Fatalf("assistant Content = %d blocks, want 2 (thinking + text)", len(asst.Content))
	}
	gotThink := asst.Content[0]
	if gotThink.Kind != llmkit.BlockThinking {
		t.Errorf("block 0 Kind = %q, want thinking", gotThink.Kind)
	}
	if gotThink.Provider != "x" {
		t.Errorf("block 0 Provider = %q, want x", gotThink.Provider)
	}
	if !bytes.Equal(gotThink.Raw, thinking.Raw) {
		t.Errorf("thinking Raw not byte-identical:\n got %s\nwant %s", gotThink.Raw, thinking.Raw)
	}
	if gotThink.Text != thinking.Text {
		t.Errorf("thinking Text = %q, want %q", gotThink.Text, thinking.Text)
	}
	if asst.Content[1].Kind != llmkit.BlockText || asst.Content[1].Text != "calling the tool" {
		t.Errorf("block 1 = %+v, want the text block", asst.Content[1])
	}
	sawToolResult := false
	for i, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult && m.ToolCallID == "c1" {
			sawToolResult = true
			if i < asstIdx {
				t.Error("tool result appears BEFORE the assistant thinking message")
			}
		}
	}
	if !sawToolResult {
		t.Fatal("tool result missing from the follow-up request")
	}

	// JSONL round-trip: the recorded request event must preserve the thinking
	// payload verbatim.
	var buf bytes.Buffer
	if err := out.Transcript.SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	loaded, err := LoadJSONL(&buf)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	found := false
	for _, ev := range loaded.Record {
		if ev.Kind != llmkit.KindCompletion || ev.Completion == nil || ev.Step != 2 {
			continue
		}
		for _, m := range ev.Completion.Request.Messages {
			if m.Role != llmkit.RoleAssistant || len(m.Content) == 0 {
				continue
			}
			if m.Content[0].Kind == llmkit.BlockThinking && bytes.Equal(m.Content[0].Raw, thinking.Raw) {
				found = true
			}
		}
	}
	if !found {
		t.Error("JSONL round-trip lost the thinking block's Raw payload on the step-2 request")
	}
}

// TestRun_MaxTokensContinuation_PreservesThinkingBlocks verifies the
// continuation path: thinking blocks from BOTH halves survive into the
// history in order, each assistant message re-emittable to the provider, and
// the texts stitch into one answer.
func TestRun_MaxTokensContinuation_PreservesThinkingBlocks(t *testing.T) {
	th1 := llmkit.Block{Kind: llmkit.BlockThinking, Text: "head reasoning", Provider: "x", Raw: json.RawMessage(`{"sig":"1"}`)}
	th2 := llmkit.Block{Kind: llmkit.BlockThinking, Text: "tail reasoning", Provider: "x", Raw: json.RawMessage(`{"sig":"2"}`)}
	fc := newFakeClient(
		scriptStep{resp: llmkit.Response{
			Blocks:     []llmkit.Block{th1, {Kind: llmkit.BlockText, Text: "The answer is fort-"}},
			Text:       "The answer is fort-",
			StopReason: llmkit.StopMaxTokens,
			Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 5},
		}},
		scriptStep{resp: llmkit.Response{
			Blocks:     []llmkit.Block{th2, {Kind: llmkit.BlockText, Text: "two."}},
			Text:       "two.",
			StopReason: llmkit.StopEndTurn,
			Usage:      llmkit.Usage{InputTokens: 5, OutputTokens: 5},
		}},
	)
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "The answer is fort-two." {
		t.Fatalf("FinalText = %q, want the stitched halves", out.FinalText)
	}
	// History: task, assistant head (thinking+text), continuation nudge,
	// assistant tail (thinking+text).
	history := out.Messages
	if len(history) != 4 {
		t.Fatalf("history = %d messages, want 4 (task, head, nudge, tail)", len(history))
	}
	if history[1].Role != llmkit.RoleAssistant || len(history[1].Content) != 2 ||
		history[1].Content[0].Kind != llmkit.BlockThinking || !bytes.Equal(history[1].Content[0].Raw, th1.Raw) {
		t.Errorf("history[1] = %+v, want the head assistant turn with thinking sig-1 verbatim", history[1])
	}
	if history[3].Role != llmkit.RoleAssistant || len(history[3].Content) != 2 ||
		history[3].Content[0].Kind != llmkit.BlockThinking || !bytes.Equal(history[3].Content[0].Raw, th2.Raw) {
		t.Errorf("history[3] = %+v, want the tail assistant turn with thinking sig-2 verbatim", history[3])
	}
	// The continuation request (request 2) must carry the head's thinking
	// block so the provider can bind the signature.
	req2HasThinking := false
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleAssistant && len(m.Content) > 0 &&
			m.Content[0].Kind == llmkit.BlockThinking && bytes.Equal(m.Content[0].Raw, th1.Raw) {
			req2HasThinking = true
		}
	}
	if !req2HasThinking {
		t.Error("continuation request lost the head thinking block")
	}
	// JSONL round-trip: each half's thinking payload survives verbatim.
	var buf bytes.Buffer
	if err := out.Transcript.SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	loaded, err := LoadJSONL(&buf)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	sawHead, sawTail := false, false
	for _, ev := range loaded.Record {
		if ev.Kind != llmkit.KindCompletion || ev.Completion == nil {
			continue
		}
		for _, b := range ev.Completion.Response.Blocks {
			if b.Kind == llmkit.BlockThinking && bytes.Equal(b.Raw, th1.Raw) {
				sawHead = true
			}
			if b.Kind == llmkit.BlockThinking && bytes.Equal(b.Raw, th2.Raw) {
				sawTail = true
			}
		}
	}
	if !sawHead || !sawTail {
		t.Errorf("JSONL round-trip lost a thinking payload: head=%v tail=%v", sawHead, sawTail)
	}
}

// TestStitchBlocks_KeepsThinkingFromBothHalves pins the stitch rule: only
// text is joined; thinking blocks from both halves are carried through in
// order, untouched.
func TestStitchBlocks_KeepsThinkingFromBothHalves(t *testing.T) {
	th1 := llmkit.Block{Kind: llmkit.BlockThinking, Text: "head", Provider: "x", Raw: json.RawMessage(`{"sig":"1"}`)}
	th2 := llmkit.Block{Kind: llmkit.BlockThinking, Text: "tail", Provider: "x", Raw: json.RawMessage(`{"sig":"2"}`)}
	head := []llmkit.Block{th1, {Kind: llmkit.BlockText, Text: "fort-"}}
	cont := []llmkit.Block{th2, {Kind: llmkit.BlockText, Text: "two."}}

	got := stitchBlocks(head, cont, "fort-two.")
	if len(got) != 3 {
		t.Fatalf("stitchBlocks = %d blocks, want 3 (thinking, thinking, text)", len(got))
	}
	if got[0].Kind != llmkit.BlockThinking || !bytes.Equal(got[0].Raw, th1.Raw) {
		t.Errorf("block 0 = %+v, want head thinking verbatim", got[0])
	}
	if got[1].Kind != llmkit.BlockThinking || !bytes.Equal(got[1].Raw, th2.Raw) {
		t.Errorf("block 1 = %+v, want tail thinking verbatim", got[1])
	}
	if got[2].Kind != llmkit.BlockText || got[2].Text != "fort-two." {
		t.Errorf("block 2 = %+v, want the joined text block", got[2])
	}
}

// doneSignalTool reports on done when its Run returns, so a test can observe
// the exact moment a sibling goroutine finished.
type doneSignalTool struct {
	name   string
	delay  time.Duration
	result string
	done   chan struct{} // closed once, when Run returns
}

func (d doneSignalTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: d.name, Description: "signals completion", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (d doneSignalTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	select {
	case <-ctx.Done():
		close(d.done)
		return "", ctx.Err()
	case <-time.After(d.delay):
	}
	close(d.done)
	return d.result, nil
}

// TestRun_ToolPanic_SequentialRenderedAsError pins half of the uniform
// tool-panic contract: under SEQUENTIAL dispatch (the default), a panicking
// Tool.Run is recovered by the harness and rendered as that call's
// "ERROR:"-prefixed result with IsError=true — exactly as in parallel mode —
// ToolEnd fires with that result, and the run continues to completion.
func TestRun_ToolPanic_SequentialRenderedAsError(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "panicky", `{}`, 10, 4),
		textResp("recovered", 5, 2),
	)
	var endEv ToolEvent
	r := NewRunner(fc, []Tool{panicTool{name: "panicky"}}, "sys",
		WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) { endEv = ev }}))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run must survive a panicking tool in sequential mode: %v", err)
	}
	if out.FinalText != "recovered" {
		t.Fatalf("FinalText = %q, want the run to continue past the panic", out.FinalText)
	}
	var tr *llmkit.Message
	for i := range fc.requests[1].Messages {
		if fc.requests[1].Messages[i].Role == llmkit.RoleToolResult && fc.requests[1].Messages[i].ToolCallID == "c1" {
			tr = &fc.requests[1].Messages[i]
		}
	}
	if tr == nil {
		t.Fatal("no tool result carried back after the panic")
	}
	want := "ERROR: tool panicky panicked: tool exploded"
	if tr.Text() != want {
		t.Errorf("panicking call result = %q, want %q", tr.Text(), want)
	}
	if !tr.IsError {
		t.Error("panic tool result IsError = false, want true")
	}
	// ToolEnd fired with the rendered result.
	if endEv.Result != want || !endEv.IsError {
		t.Errorf("ToolEnd event = %+v, want the rendered panic result with IsError", endEv)
	}
}

// TestRun_HookPanic_SequentialPropagates pins the other half of the
// contract: a panicking HOOK is a harness bug — it propagates out of Run
// with the ORIGINAL panic value and is never rendered to the model.
func TestRun_HookPanic_SequentialPropagates(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{}`, 10, 4),
		textResp("never reached", 5, 2),
	)
	hooks := Hooks{
		ToolStart: func(_ context.Context, ev ToolEvent) {
			panic("hook exploded")
		},
	}
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(hooks))

	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		_, _ = r.Run(context.Background(), "task")
		done <- nil
	}()
	select {
	case v := <-done:
		if v != "hook exploded" {
			t.Fatalf("Run panic value = %v, want the original hook panic value", v)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not panic on a sequential hook panic")
	}
	// The panic aborted the turn: no second completion was ever issued.
	if len(fc.requests) != 1 {
		t.Errorf("requests = %d, want 1 (the hook panic must abort the run, not render as tool output)", len(fc.requests))
	}
}

// TestRun_HookPanic_ParallelPropagatesAfterSiblings pins the parallel-mode
// hook contract: the per-call goroutine recovers the hook panic and the loop
// goroutine re-panics with the ORIGINAL value only AFTER every sibling
// goroutine has finished — no goroutine leak, no tool_result recorded for
// the interrupted turn, nothing rendered to the model.
func TestRun_HookPanic_ParallelPropagatesAfterSiblings(t *testing.T) {
	fc := newFakeClient(
		toolCallsResp(
			llmkit.ToolCall{ID: "c1", Name: "fast", Arguments: json.RawMessage(`{}`)},
			llmkit.ToolCall{ID: "c2", Name: "slow", Arguments: json.RawMessage(`{}`)},
		),
		textResp("never reached", 5, 2),
	)
	slow := doneSignalTool{name: "slow", delay: 150 * time.Millisecond, result: "R_slow", done: make(chan struct{})}
	// ToolEnd panics for the FAST call only; the slow sibling's ToolEnd must
	// still fire (hooks are not suppressed by a sibling's panic).
	var slowEndFired atomicBool
	hooks := Hooks{
		ToolEnd: func(_ context.Context, ev ToolEvent) {
			if ev.Call.Name == "slow" {
				slowEndFired.store(true)
				return
			}
			panic("hook exploded")
		},
	}
	trDir := t.TempDir()
	r := NewRunner(fc, []Tool{staggerTool{name: "fast", delay: time.Millisecond, result: "R_fast"}, slow}, "sys",
		WithHooks(hooks), WithParallelTools(), WithObserver(JSONL(trDir, nil)))
	type result struct {
		panicked any
	}
	done := make(chan result, 1)
	go func() {
		defer func() { done <- result{panicked: recover()} }()
		_, _ = r.Run(context.Background(), "task")
		done <- result{panicked: nil}
	}()

	// The sibling must FINISH before the panic surfaces: receive its signal
	// first, then the panic. If the harness re-panicked without waiting, the
	// order below would invert (the panic would arrive while this receive is
	// still blocked on the sibling).
	select {
	case <-slow.done:
	case <-time.After(2 * time.Second):
		t.Fatal("slow sibling never finished — leaked goroutine or deadlock")
	}
	select {
	case res := <-done:
		if res.panicked != "hook exploded" {
			t.Fatalf("Run panic value = %v, want the original hook panic value re-panicked after siblings finished", res.panicked)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not panic after the sibling finished")
	}
	if !slowEndFired.load() {
		t.Error("slow sibling's ToolEnd never fired — a sibling's hook panic must not suppress it")
	}
	// The transcript records NO tool_result for the interrupted turn: the
	// autosaved JSONL holds the request and assistant events but no
	// tool_result event — the re-panic happened before the caller recorded
	// anything for this turn.
	entries, rerr := os.ReadDir(trDir)
	if rerr != nil || len(entries) == 0 {
		t.Fatalf("autosave transcript missing (entries=%d err=%v)", len(entries), rerr)
	}
	data, rerr := os.ReadFile(filepath.Join(trDir, entries[0].Name()))
	if rerr != nil {
		t.Fatalf("ReadFile: %v", rerr)
	}
	if bytes.Contains(data, []byte(`"tool_result"`)) {
		t.Errorf("transcript recorded a tool_result for the hook-panicked turn:\n%s", data)
	}
	if !bytes.Contains(data, []byte(`"request"`)) {
		t.Errorf("transcript lost the request event:\n%s", data)
	}
}

// atomicBool is a tiny concurrency-safe flag for hook callbacks.
type atomicBool struct {
	mu sync.Mutex
	v  bool
}

func (b *atomicBool) store(v bool) { b.mu.Lock(); b.v = v; b.mu.Unlock() }
func (b *atomicBool) load() bool   { b.mu.Lock(); defer b.mu.Unlock(); return b.v }

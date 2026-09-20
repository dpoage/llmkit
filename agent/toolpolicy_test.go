package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// The policy tests use their own local fakes (tp-prefixed) so this file stays hermetic against edits to shared test infrastructure.

// tpScriptStep is one programmed turn of a tpScriptClient.
type tpScriptStep struct {
	resp llmkit.Response
	err  error
}

// tpScriptClient is a scripted llmkit.Client: it returns each step in order and records every request it received.
type tpScriptClient struct {
	mu       sync.Mutex
	steps    []tpScriptStep
	idx      int
	requests []llmkit.Request
}

func (c *tpScriptClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *tpScriptClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests = append(c.requests, req)
	if c.idx >= len(c.steps) {
		// Benign end-turn default so over-running tests fail on assertions, not panics.
		return llmkit.Response{Text: "(unscripted)", StopReason: llmkit.StopEndTurn}, nil
	}
	step := c.steps[c.idx]
	c.idx++
	return step.resp, step.err
}

// tpTextResp builds an end-turn text response.
func tpTextResp(text string) tpScriptStep {
	return tpScriptStep{resp: llmkit.Response{
		Text:       text,
		StopReason: llmkit.StopEndTurn,
		Usage:      llmkit.Usage{InputTokens: 5, OutputTokens: 2},
	}}
}

// tpToolCallsResp builds a tool-use response requesting calls in order.
func tpToolCallsResp(calls ...llmkit.ToolCall) tpScriptStep {
	return tpScriptStep{resp: llmkit.Response{
		StopReason: llmkit.StopToolUse,
		ToolCalls:  calls,
		Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 4},
	}}
}

// tpToolCall is the one-line form for a scripted tool call.
func tpToolCall(id, name, args string) llmkit.ToolCall {
	return llmkit.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(args)}
}

// tpCountingTool counts its Run invocations and remembers the last arguments it received. Safe for concurrent use.
type tpCountingTool struct {
	name   string
	result string

	mu       sync.Mutex
	runs     int
	lastArgs json.RawMessage
}

func (t *tpCountingTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{
		Name:        t.name,
		Description: "counts its runs",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
}

func (t *tpCountingTool) Run(_ context.Context, args json.RawMessage) (string, error) {
	t.mu.Lock()
	t.runs++
	t.lastArgs = args
	t.mu.Unlock()
	return t.result, nil
}

func (t *tpCountingTool) runCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.runs
}

func (t *tpCountingTool) args() json.RawMessage {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.lastArgs
}

// tpTrace records policy/hook events under one mutex with a monotonic sequence number — ordering no single callback observes alone.
type tpTrace struct {
	mu     sync.Mutex
	seq    int
	events []string // "<kind>:<detail>#<seq>"
}

func (tr *tpTrace) mark(kind, detail string) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	tr.seq++
	tr.events = append(tr.events, fmt.Sprintf("%s:%s#%d", kind, detail, tr.seq))
}

// kinds returns the event kinds in trace order.
func (tr *tpTrace) kinds() []string {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	kinds := make([]string, len(tr.events))
	for i, ev := range tr.events {
		if j := strings.Index(ev, ":"); j >= 0 {
			kinds[i] = ev[:j]
		} else {
			kinds[i] = ev
		}
	}
	return kinds
}

// snapshot returns the raw events in trace order, for failure messages.
func (tr *tpTrace) snapshot() []string {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return append([]string(nil), tr.events...)
}

// tpToolResult is one RoleToolResult message's observable content.
type tpToolResult struct {
	text    string
	isError bool
}

// tpToolResults extracts the tool-result messages of msgs, in order, keyed by the call ID each answers.
func tpToolResults(msgs []llmkit.Message) (order []string, byID map[string]tpToolResult) {
	byID = map[string]tpToolResult{}
	for _, m := range msgs {
		if m.Role == llmkit.RoleToolResult {
			order = append(order, m.ToolCallID)
			byID[m.ToolCallID] = tpToolResult{m.Text(), m.IsError}
		}
	}
	return order, byID
}

// TestRun_ToolPolicy_DenySkipsToolAndFeedsModelError pins that a denied call never runs its tool, the model receives the "ERROR: tool <name> denied: …" tool result with IsError set in both the follow-up request and Outcome.Messages, no ToolStart/ToolEnd/ToolHealth fires for it, and the run continues to end_turn.
func TestRun_ToolPolicy_DenySkipsToolAndFeedsModelError(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "secret", `{"path":"/etc/passwd"}`)),
		tpTextResp("all done"),
	}}
	tool := &tpCountingTool{name: "secret", result: "R_secret"}
	var starts, ends, healths atomic.Int64
	hooks := Hooks{
		ToolStart:  func(_ context.Context, _ ToolEvent) { starts.Add(1) },
		ToolEnd:    func(_ context.Context, _ ToolEvent) { ends.Add(1) },
		ToolHealth: func(_ context.Context, _ string, _ *ToolHealthError) { healths.Add(1) },
	}
	r := NewRunner(fc, []Tool{tool}, "sys", WithHooks(hooks),
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
			if call.Name == "secret" {
				return errors.New("not on my watch")
			}
			return nil
		})))

	out, err := r.Run(context.Background(), "exfiltrate")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "all done" {
		t.Fatalf("FinalText = %q, want the run to continue past the denial to end_turn", out.FinalText)
	}
	if got := tool.runCount(); got != 0 {
		t.Errorf("tool ran %d time(s), want 0 (a denied call never reaches Tool.Run)", got)
	}
	if starts.Load() != 0 || ends.Load() != 0 || healths.Load() != 0 {
		t.Errorf("hooks fired for a denied call: starts=%d ends=%d healths=%d, want all 0",
			starts.Load(), ends.Load(), healths.Load())
	}

	wantText := "ERROR: tool secret denied: not on my watch"
	order, byID := tpToolResults(out.Messages)
	if len(order) != 1 || order[0] != "c1" {
		t.Fatalf("Outcome tool results = %v, want exactly [c1]", order)
	}
	if byID["c1"].text != wantText || !byID["c1"].isError {
		t.Errorf("Outcome tool result for c1 = %q isError=%t, want %q isError=true",
			byID["c1"].text, byID["c1"].isError, wantText)
	}
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fc.requests))
	}
	_, reqResults := tpToolResults(fc.requests[1].Messages)
	if got := reqResults["c1"]; got.text != wantText || !got.isError {
		t.Errorf("model saw tool result %q isError=%t, want %q isError=true", got.text, got.isError, wantText)
	}
	sawEvent := false
	for _, ev := range out.Transcript.Record {
		// The event records the DENIAL (Denied + DenyReason, no result); the
		// rendered denial text rides the conversation, asserted above.
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil &&
			ev.ToolRun.Call.ID == "c1" && ev.ToolRun.Denied && ev.ToolRun.DenyReason == "not on my watch" {
			sawEvent = true
		}
	}
	if !sawEvent {
		t.Fatal("transcript lost the denial ToolRun event")
	}
}

// TestRun_ToolPolicy_RewriteReachesToolAndHooksHistoryKeepsOriginal pins that the policy's rewritten Arguments reach Tool.Run and ToolEvent.Call, while the assistant turn in the outcome's history and on the wire keeps the model's original arguments.
func TestRun_ToolPolicy_RewriteReachesToolAndHooksHistoryKeepsOriginal(t *testing.T) {
	const origArgs = `{"a":1,"b":2}`
	const newArgs = `{"a":40,"b":2}`
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "add", origArgs)),
		tpTextResp("done"),
	}}
	tool := &tpCountingTool{name: "add", result: "R_add"}
	var endEv ToolEvent
	hooks := Hooks{
		ToolEnd: func(_ context.Context, ev ToolEvent) { endEv = ev },
	}
	r := NewRunner(fc, []Tool{tool}, "sys", WithHooks(hooks),
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
			if call.Name == "add" {
				call.Arguments = json.RawMessage(newArgs)
			}
			return nil
		})))

	out, err := r.Run(context.Background(), "add numbers")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := tool.runCount(); got != 1 {
		t.Fatalf("tool ran %d time(s), want 1", got)
	}
	if string(tool.args()) != newArgs {
		t.Errorf("Tool.Run args = %s, want the rewritten %s", tool.args(), newArgs)
	}
	if string(endEv.Call.Arguments) != newArgs {
		t.Errorf("ToolEvent.Call.Arguments = %s, want the rewritten %s", endEv.Call.Arguments, newArgs)
	}
	assertAssistantArgs := func(label string, msgs []llmkit.Message) {
		t.Helper()
		for _, m := range msgs {
			if m.Role != llmkit.RoleAssistant || len(m.ToolCalls) == 0 || m.ToolCalls[0].ID != "c1" {
				continue
			}
			if string(m.ToolCalls[0].Arguments) != origArgs {
				t.Errorf("%s: assistant ToolCall args = %s, want the model's original %s",
					label, m.ToolCalls[0].Arguments, origArgs)
			}
			return
		}
		t.Errorf("%s: no assistant turn carrying call c1", label)
	}
	assertAssistantArgs("Outcome.Messages", out.Messages)
	assertAssistantArgs("wire request", fc.requests[1].Messages)
}

// TestRun_ToolPolicy_ParallelAllAuthorizedBeforeFirstRun pins that under WithParallelTools, all Authorize calls of a turn run on the loop goroutine and complete before any ToolStart fires; results land under the model's call IDs in original order.
func TestRun_ToolPolicy_ParallelAllAuthorizedBeforeFirstRun(t *testing.T) {
	tools := []*tpCountingTool{
		{name: "slow", result: "R_slow"},
		{name: "mid", result: "R_mid"},
		{name: "fast", result: "R_fast"},
	}
	trace := &tpTrace{}
	hooks := Hooks{
		ToolStart: func(_ context.Context, ev ToolEvent) { trace.mark("start", ev.Call.Name) },
		ToolEnd:   func(_ context.Context, ev ToolEvent) { trace.mark("end", ev.Call.Name) },
	}
	policy := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		trace.mark("authorize", call.Name)
		return nil
	})
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(
			tpToolCall("c1", "slow", `{}`),
			tpToolCall("c2", "mid", `{}`),
			tpToolCall("c3", "fast", `{}`),
		),
		tpTextResp("all done"),
	}}
	r := NewRunner(fc, []Tool{tools[0], tools[1], tools[2]}, "sys",
		WithHooks(hooks), WithParallelTools(), WithToolPolicy(policy))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	auths, firstStart := 0, -1
	for i, k := range trace.kinds() {
		if k == "authorize" {
			if firstStart >= 0 {
				t.Fatalf("an Authorize ran after ToolStart fired: %v", trace.snapshot())
			}
			auths++
			continue
		}
		if k == "start" && firstStart < 0 {
			firstStart = i
		}
	}
	if auths != 3 {
		t.Errorf("Authorize fired %d time(s), want exactly 3 (trace %v)", auths, trace.snapshot())
	}
	if firstStart < 0 {
		t.Fatal("no ToolStart fired")
	}

	want := map[string]string{"c1": "R_slow", "c2": "R_mid", "c3": "R_fast"}
	order, byID := tpToolResults(out.Messages)
	if len(order) != 3 {
		t.Fatalf("tool results = %v, want 3", order)
	}
	for i, id := range []string{"c1", "c2", "c3"} {
		if order[i] != id {
			t.Fatalf("tool results out of model order: %v", order)
		}
		if byID[id].text != want[id] || byID[id].isError {
			t.Errorf("result for %s = %q (isError=%t), want %q", id, byID[id].text, byID[id].isError, want[id])
		}
	}
}

// TestRun_ToolPolicy_CancelInsideAuthorizeAbortsRun pins that a policy cancelling the run's context and denying aborts the run — Run returns the context error, no Tool.Run executes, but the rendered deny result for the call is recorded before the abort.
func TestRun_ToolPolicy_CancelInsideAuthorizeAbortsRun(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "secret", `{}`)),
		tpTextResp("never reached"),
	}}
	tool := &tpCountingTool{name: "secret", result: "R_secret"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := NewRunner(fc, []Tool{tool}, "sys",
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, _ *llmkit.ToolCall) error {
			cancel() // the test-owned cancel, invoked inside Authorize
			return errors.New("vetoed")
		})))

	out, err := r.Run(ctx, "task")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run err = %v, want context.Canceled", err)
	}
	if got := tool.runCount(); got != 0 {
		t.Errorf("tool ran %d time(s), want 0", got)
	}
	if len(fc.requests) != 1 {
		t.Errorf("requests = %d, want 1 (the cancelled run never makes the second completion)", len(fc.requests))
	}
	order, byID := tpToolResults(out.Messages)
	if len(order) != 1 || order[0] != "c1" || byID["c1"].text != "ERROR: tool secret denied: vetoed" {
		t.Errorf("recorded tool results %v byID=%v, want the c1 deny result", order, byID)
	}
}

// TestRun_ToolPolicy_MixedTurnDeniesOnlyMiddleCall pins that with call 1 allowed, call 2 denied, call 3 allowed, all three results land in the model's order with the deny in the middle (IsError), only the two allowed tools run, and the run continues to end_turn.
func TestRun_ToolPolicy_MixedTurnDeniesOnlyMiddleCall(t *testing.T) {
	tools := []*tpCountingTool{
		{name: "t1", result: "R1"},
		{name: "t2", result: "R2"},
		{name: "t3", result: "R3"},
	}
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(
			tpToolCall("c1", "t1", `{"n":1}`),
			tpToolCall("c2", "t2", `{"n":2}`),
			tpToolCall("c3", "t3", `{"n":3}`),
		),
		tpTextResp("done"),
	}}
	r := NewRunner(fc, []Tool{tools[0], tools[1], tools[2]}, "sys",
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
			if call.Name == "t2" {
				return errors.New("blocked")
			}
			return nil
		})))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q, want the run to continue past the denial", out.FinalText)
	}
	if tools[0].runCount() != 1 || tools[1].runCount() != 0 || tools[2].runCount() != 1 {
		t.Errorf("run counts = %d/%d/%d, want 1/0/1 (2 runs total)",
			tools[0].runCount(), tools[1].runCount(), tools[2].runCount())
	}
	order, byID := tpToolResults(out.Messages)
	if len(order) != 3 || order[0] != "c1" || order[1] != "c2" || order[2] != "c3" {
		t.Fatalf("tool-result order = %v, want [c1 c2 c3]", order)
	}
	if byID["c1"].text != "R1" || byID["c1"].isError {
		t.Errorf("c1 result = %+v, want R1 isError=false", byID["c1"])
	}
	if want := "ERROR: tool t2 denied: blocked"; byID["c2"].text != want || !byID["c2"].isError {
		t.Errorf("c2 result = %+v, want %q isError=true", byID["c2"], want)
	}
	if byID["c3"].text != "R3" || byID["c3"].isError {
		t.Errorf("c3 result = %+v, want R3 isError=false", byID["c3"])
	}
	reqOrder, reqByID := tpToolResults(fc.requests[1].Messages)
	if len(reqOrder) != 3 || reqOrder[0] != "c1" || reqOrder[1] != "c2" || reqOrder[2] != "c3" {
		t.Fatalf("wire tool-result order = %v, want [c1 c2 c3]", reqOrder)
	}
	if reqByID["c2"].text != "ERROR: tool t2 denied: blocked" || !reqByID["c2"].isError {
		t.Errorf("wire c2 result = %+v, want the deny text with IsError", reqByID["c2"])
	}
}

// TestRun_ToolPolicy_UnregisteredToolSkipsAuthorize pins that a model-requested call naming an unregistered tool never reaches the policy and renders the existing "unknown tool" error.
func TestRun_ToolPolicy_UnregisteredToolSkipsAuthorize(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "nosuch", `{}`)),
		tpTextResp("done"),
	}}
	var authorizations atomic.Int64
	r := NewRunner(fc, nil, "sys",
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, _ *llmkit.ToolCall) error {
			authorizations.Add(1)
			return nil
		})))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if authorizations.Load() != 0 {
		t.Errorf("Authorize fired %d time(s), want 0 for an unregistered tool", authorizations.Load())
	}
	order, byID := tpToolResults(out.Messages)
	if len(order) != 1 || order[0] != "c1" {
		t.Fatalf("tool results = %v, want exactly [c1]", order)
	}
	if want := `ERROR: unknown tool "nosuch"`; byID["c1"].text != want || !byID["c1"].isError {
		t.Errorf("c1 result = %q isError=%t, want %q isError=true", byID["c1"].text, byID["c1"].isError, want)
	}
}

// TestRun_ToolPolicy_NilPolicyIsZeroBehaviorChange pins that with no policy installed, every call runs exactly as before the seam.
func TestRun_ToolPolicy_NilPolicyIsZeroBehaviorChange(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "add", `{"a":1}`)),
		tpTextResp("done"),
	}}
	tool := &tpCountingTool{name: "add", result: "R_add"}
	r := NewRunner(fc, []Tool{tool}, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q, want done", out.FinalText)
	}
	if got := tool.runCount(); got != 1 {
		t.Errorf("tool ran %d time(s), want 1", got)
	}
	order, byID := tpToolResults(out.Messages)
	if len(order) != 1 || byID["c1"].text != "R_add" || byID["c1"].isError {
		t.Errorf("c1 result = %+v (order %v), want R_add isError=false", byID["c1"], order)
	}
}

// TestRun_ToolPolicy_AuthorizePanicPropagates pins that a panic inside Authorize in sequential mode propagates out of Run with its original value — never rendered as tool output — and the tool never runs.
func TestRun_ToolPolicy_AuthorizePanicPropagates(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "add", `{}`)),
		tpTextResp("never reached"),
	}}
	tool := &tpCountingTool{name: "add", result: "R_add"}
	r := NewRunner(fc, []Tool{tool}, "sys",
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, _ *llmkit.ToolCall) error {
			panic("policy exploded")
		})))

	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		_, _ = r.Run(context.Background(), "task")
		done <- nil
	}()
	select {
	case v := <-done:
		if v != "policy exploded" {
			t.Fatalf("Run panic value = %v, want the original Authorize panic value", v)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not panic on an Authorize panic")
	}
	if len(fc.requests) != 1 {
		t.Errorf("requests = %d, want 1 (the panic must abort the run, not render as tool output)", len(fc.requests))
	}
	if got := tool.runCount(); got != 0 {
		t.Errorf("tool ran %d time(s), want 0", got)
	}
}

// TestRun_ToolPolicy_AuthorizePanicParallelPropagates pins that under WithParallelTools the Authorize pre-pass runs on the loop goroutine before any fan-out goroutine exists, so a panic propagates unchanged and no ToolStart ever fires.
func TestRun_ToolPolicy_AuthorizePanicParallelPropagates(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(
			tpToolCall("c1", "t1", `{}`),
			tpToolCall("c2", "t2", `{}`),
		),
		tpTextResp("never reached"),
	}}
	tools := []*tpCountingTool{{name: "t1", result: "R1"}, {name: "t2", result: "R2"}}
	var starts atomic.Int64
	hooks := Hooks{ToolStart: func(_ context.Context, _ ToolEvent) { starts.Add(1) }}
	r := NewRunner(fc, []Tool{tools[0], tools[1]}, "sys", WithHooks(hooks), WithParallelTools(),
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
			if call.Name == "t1" {
				panic("policy exploded")
			}
			return nil
		})))

	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		_, _ = r.Run(context.Background(), "task")
		done <- nil
	}()
	select {
	case v := <-done:
		if v != "policy exploded" {
			t.Fatalf("Run panic value = %v, want the original Authorize panic value", v)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not panic on an Authorize panic under WithParallelTools")
	}
	if starts.Load() != 0 {
		t.Errorf("ToolStart fired %d time(s), want 0 (the pre-pass panics before any dispatch)", starts.Load())
	}
}

// TestRun_ToolPolicy_CancelInsideAuthorizeDeniesRemainderInBothModes pins that once the Authorize pre-pass observes a cancelled ctx, no later call is authorized or dispatched in either mode — the remainder is denied with the context error.
func TestRun_ToolPolicy_CancelInsideAuthorizeDeniesRemainderInBothModes(t *testing.T) {
	for _, mode := range []struct {
		name     string
		parallel bool
	}{{"sequential", false}, {"parallel", true}} {
		t.Run(mode.name, func(t *testing.T) {
			fc := &tpScriptClient{steps: []tpScriptStep{
				tpToolCallsResp(
					tpToolCall("c1", "t1", `{}`),
					tpToolCall("c2", "t2", `{}`),
					tpToolCall("c3", "t3", `{}`),
				),
				tpTextResp("never reached"),
			}}
			tools := []*tpCountingTool{
				{name: "t1", result: "R1"},
				{name: "t2", result: "R2"},
				{name: "t3", result: "R3"},
			}
			var starts, authorizations atomic.Int64
			hooks := Hooks{ToolStart: func(_ context.Context, _ ToolEvent) { starts.Add(1) }}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opts := []Option{
				WithHooks(hooks),
				WithToolPolicy(ToolPolicyFunc(func(_ context.Context, _ *llmkit.ToolCall) error {
					if authorizations.Add(1) == 1 {
						cancel() // the test-owned cancel, inside Authorize #1
						return errors.New("vetoed")
					}
					return nil
				})),
			}
			if mode.parallel {
				opts = append(opts, WithParallelTools())
			}
			r := NewRunner(fc, []Tool{tools[0], tools[1], tools[2]}, "sys", opts...)

			out, err := r.Run(ctx, "task")
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("Run err = %v, want context.Canceled", err)
			}
			if authorizations.Load() != 1 {
				t.Errorf("Authorize fired %d time(s), want exactly 1 (no consultation after cancellation)", authorizations.Load())
			}
			if starts.Load() != 0 {
				t.Errorf("ToolStart fired %d time(s), want 0 (never-authorized calls must not dispatch)", starts.Load())
			}
			for _, tool := range tools {
				if got := tool.runCount(); got != 0 {
					t.Errorf("tool %s ran %d time(s), want 0", tool.name, got)
				}
			}
			order, byID := tpToolResults(out.Messages)
			if len(order) != 3 || order[0] != "c1" || order[1] != "c2" || order[2] != "c3" {
				t.Fatalf("tool-result order = %v, want [c1 c2 c3] (full-length slice, both modes)", order)
			}
			if want := "ERROR: tool t1 denied: vetoed"; byID["c1"].text != want || !byID["c1"].isError {
				t.Errorf("c1 result = %+v, want %q isError=true", byID["c1"], want)
			}
			for id, name := range map[string]string{"c2": "t2", "c3": "t3"} {
				if want := "ERROR: tool " + name + " denied: context canceled"; byID[id].text != want || !byID[id].isError {
					t.Errorf("%s result = %+v, want %q isError=true", id, byID[id], want)
				}
			}
			if len(fc.requests) != 1 {
				t.Errorf("requests = %d, want 1 (the cancelled run never makes the second completion)", len(fc.requests))
			}
		})
	}
}

// TestRun_ToolPolicy_ArgumentsMutationDoesNotAliasHistory pins that a policy mutating the Arguments bytes in place must not corrupt the assistant turn in the history or on the wire — the dispatch copy owns its own backing array.
func TestRun_ToolPolicy_ArgumentsMutationDoesNotAliasHistory(t *testing.T) {
	const origArgs = `{"a":1,"b":2}`
	const mutatedArgs = `{"a":9,"b":2}`
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "add", origArgs)),
		tpTextResp("done"),
	}}
	tool := &tpCountingTool{name: "add", result: "R_add"}
	r := NewRunner(fc, []Tool{tool}, "sys",
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
			copy(call.Arguments, mutatedArgs) // in-place byte mutation
			return nil
		})))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if string(tool.args()) != mutatedArgs {
		t.Errorf("Tool.Run args = %s, want the mutated %s", tool.args(), mutatedArgs)
	}
	assertAssistantArgs := func(label string, msgs []llmkit.Message) {
		t.Helper()
		for _, m := range msgs {
			if m.Role != llmkit.RoleAssistant || len(m.ToolCalls) == 0 || m.ToolCalls[0].ID != "c1" {
				continue
			}
			if string(m.ToolCalls[0].Arguments) != origArgs {
				t.Errorf("%s: assistant ToolCall args = %s, want the model's original %s (byte mutation leaked)",
					label, m.ToolCalls[0].Arguments, origArgs)
			}
			return
		}
		t.Errorf("%s: no assistant turn carrying call c1", label)
	}
	assertAssistantArgs("Outcome.Messages", out.Messages)
	assertAssistantArgs("wire request", fc.requests[1].Messages)
}

// TestRun_ToolPolicy_RewriteOnlyArguments pins that a policy reassigning Name or ID gets both reverted — the requested tool executes under the model's name and ID with the rewritten arguments, and the result pairs under the model's call ID.
func TestRun_ToolPolicy_RewriteOnlyArguments(t *testing.T) {
	fc := &tpScriptClient{steps: []tpScriptStep{
		tpToolCallsResp(tpToolCall("c1", "t1", `{"n":1}`)),
		tpTextResp("done"),
	}}
	t1 := &tpCountingTool{name: "t1", result: "R1"}
	t3 := &tpCountingTool{name: "t3", result: "R3"}
	var endEv ToolEvent
	hooks := Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) { endEv = ev }}
	r := NewRunner(fc, []Tool{t1, t3}, "sys", WithHooks(hooks),
		WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
			call.Arguments = json.RawMessage(`{"n":42}`)
			call.Name = "t3" // adversarial: redirect dispatch to a sibling tool
			call.ID = "evil" // adversarial: corrupt the call/result pairing
			return nil
		})))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if t1.runCount() != 1 || t3.runCount() != 0 {
		t.Errorf("run counts t1=%d t3=%d, want 1/0 (a Name rewrite must not redirect dispatch)", t1.runCount(), t3.runCount())
	}
	if endEv.Call.ID != "c1" || endEv.Call.Name != "t1" {
		t.Errorf("ToolEvent.Call = %s/%s, want the model's c1/t1", endEv.Call.ID, endEv.Call.Name)
	}
	if string(endEv.Call.Arguments) != `{"n":42}` {
		t.Errorf("ToolEvent.Call.Arguments = %s, want the rewritten {\"n\":42}", endEv.Call.Arguments)
	}
	order, byID := tpToolResults(out.Messages)
	if len(order) != 1 || order[0] != "c1" || byID["c1"].text != "R1" || byID["c1"].isError {
		t.Errorf("tool results %v byID=%v, want R1 under c1 isError=false", order, byID)
	}
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleAssistant && len(m.ToolCalls) > 0 {
			if m.ToolCalls[0].ID != "c1" || m.ToolCalls[0].Name != "t1" {
				t.Errorf("history ToolCall = %s/%s, want the model's c1/t1", m.ToolCalls[0].ID, m.ToolCalls[0].Name)
			}
			break
		}
	}
}

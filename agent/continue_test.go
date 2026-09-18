package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// sameTurn asserts two messages are the same conversation turn for wire
// purposes: role, visible text, the answered tool call, any requested tool
// calls, and the error mark.
func sameTurn(t *testing.T, label string, got, want llmkit.Message) {
	t.Helper()
	if got.Role != want.Role {
		t.Errorf("%s: role = %v, want %v", label, got.Role, want.Role)
	}
	if got.Text() != want.Text() {
		t.Errorf("%s: text = %q, want %q", label, got.Text(), want.Text())
	}
	if got.ToolCallID != want.ToolCallID {
		t.Errorf("%s: ToolCallID = %q, want %q", label, got.ToolCallID, want.ToolCallID)
	}
	if got.IsError != want.IsError {
		t.Errorf("%s: IsError = %v, want %v", label, got.IsError, want.IsError)
	}
	if !reflect.DeepEqual(got.ToolCalls, want.ToolCalls) {
		t.Errorf("%s: ToolCalls = %+v, want %+v", label, got.ToolCalls, want.ToolCalls)
	}
}

// TestRunContinue_PreservesPriorConversation is the plain-Run continuation
// contract: turn 2's outgoing request must carry turn 1's full history
// element-wise as a prefix, followed by exactly ONE new user message holding
// the task — not a reseeded conversation.
func TestRunContinue_PreservesPriorConversation(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"v":"orient"}`, 10, 4),
		textResp("round one answer", 8, 3),
		textResp("round two answer", 8, 3),
	)
	var turn2Req []llmkit.Message
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(Hooks{
		BeforeCompletion: func(_ context.Context, _ int, req *llmkit.Request) {
			turn2Req = append([]llmkit.Message(nil), req.Messages...)
		},
	}))

	out1, err := r.Run(context.Background(), "first task")
	if err != nil {
		t.Fatalf("turn 1 Run: %v", err)
	}
	if len(out1.Messages) != 4 {
		t.Fatalf("turn 1 Messages = %d entries, want 4 (user, assistant tool-call, tool-result, assistant final)", len(out1.Messages))
	}

	out2, err := r.RunContinue(context.Background(), out1, "second task")
	if err != nil {
		t.Fatalf("turn 2 RunContinue: %v", err)
	}
	if len(turn2Req) != len(out1.Messages)+1 {
		t.Fatalf("turn 2 request has %d messages, want %d (turn 1 history + one new user turn) -- fewer means a reseed, more means duplicated history", len(turn2Req), len(out1.Messages)+1)
	}
	for i, want := range out1.Messages {
		sameTurn(t, fmt.Sprintf("turn 2 request message %d", i), turn2Req[i], want)
	}
	last := turn2Req[len(turn2Req)-1]
	if last.Role != llmkit.RoleUser || last.Text() != "second task" {
		t.Errorf("turn 2 request last message = %v %q, want user %q", last.Role, last.Text(), "second task")
	}
	if out2.FinalText != "round two answer" {
		t.Errorf("turn 2 FinalText = %q, want %q", out2.FinalText, "round two answer")
	}
}

// TestRunContinue_NilPrevMatchesRun pins the degradation contract: a nil prev
// and an empty prev produce byte-for-byte the same outgoing request plain Run
// would, so a caller can use RunContinue unconditionally.
func TestRunContinue_NilPrevMatchesRun(t *testing.T) {
	task := "same task every time"
	newRunner := func(fc *fakeClient) *Runner {
		return NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")
	}

	fcRun := newFakeClient(textResp("ans", 5, 2))
	if _, err := newRunner(fcRun).Run(context.Background(), task); err != nil {
		t.Fatalf("Run: %v", err)
	}

	fcNil := newFakeClient(textResp("ans", 5, 2))
	if _, err := newRunner(fcNil).RunContinue(context.Background(), nil, task); err != nil {
		t.Fatalf("RunContinue(nil): %v", err)
	}

	fcEmpty := newFakeClient(textResp("ans", 5, 2))
	if _, err := newRunner(fcEmpty).RunContinue(context.Background(), &Outcome{}, task); err != nil {
		t.Fatalf("RunContinue(empty): %v", err)
	}

	for name, fc := range map[string]*fakeClient{"nil prev": fcNil, "empty prev": fcEmpty} {
		got, want := fc.requests[0], fcRun.requests[0]
		if got.System != want.System {
			t.Errorf("%s: System = %q, want %q", name, got.System, want.System)
		}
		if len(got.Tools) != len(want.Tools) {
			t.Errorf("%s: Tools = %d, want %d", name, len(got.Tools), len(want.Tools))
		}
		if len(got.Messages) != len(want.Messages) {
			t.Fatalf("%s: Messages = %d, want %d (identical to Run)", name, len(got.Messages), len(want.Messages))
		}
		for i := range want.Messages {
			sameTurn(t, fmt.Sprintf("%s message %d", name, i), got.Messages[i], want.Messages[i])
		}
	}
}

// TestRunContinue_AfterStopError verifies the documented ErrStopReason
// recovery: the partial Outcome attached to the error threads into
// RunContinue, the refusal assistant turn stays in the history the model
// sees, and the next turn completes normally.
func TestRunContinue_AfterStopError(t *testing.T) {
	fc := newFakeClient(
		stopErrorResp("I cannot help with that.", 10, 5),
		textResp("happy to help now", 8, 3),
	)
	r := NewRunner(fc, nil, "sys")

	_, err := r.Run(context.Background(), "do the thing")
	var stopErr *ErrStopReason
	if !errors.As(err, &stopErr) {
		t.Fatalf("Run error = %v, want *ErrStopReason", err)
	}

	out2, err := r.RunContinue(context.Background(), stopErr.Outcome, "please reconsider")
	if err != nil {
		t.Fatalf("RunContinue after stop error: %v", err)
	}
	if out2.FinalText != "happy to help now" {
		t.Errorf("FinalText = %q, want %q", out2.FinalText, "happy to help now")
	}
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fc.requests))
	}
	msgs := fc.requests[1].Messages
	if len(msgs) != 3 {
		t.Fatalf("turn 2 request has %d messages, want 3 (task, refusal turn, new task)", len(msgs))
	}
	if msgs[1].Role != llmkit.RoleAssistant || msgs[1].Text() != "I cannot help with that." {
		t.Errorf("turn 2 request message 1 = %v %q, want the refusal assistant turn", msgs[1].Role, msgs[1].Text())
	}
	if msgs[2].Role != llmkit.RoleUser || msgs[2].Text() != "please reconsider" {
		t.Errorf("turn 2 request message 2 = %v %q, want user %q", msgs[2].Role, msgs[2].Text(), "please reconsider")
	}
}

// danglingPrev builds a prior Outcome whose history ends with an assistant
// turn carrying two tool calls, followed by answers for only the first
// withOrphanAnswers of them — the shape a context-cancelled run returns.
func danglingPrev(t *testing.T, answered int) *Outcome {
	t.Helper()
	calls := []llmkit.ToolCall{
		{ID: "t1", Name: "echo", Arguments: json.RawMessage(`{"v":"1"}`)},
		{ID: "t2", Name: "echo", Arguments: json.RawMessage(`{"v":"2"}`)},
	}
	asst := llmkit.Message{
		Role:      llmkit.RoleAssistant,
		Content:   []llmkit.Block{{Kind: llmkit.BlockText, Text: "calling two tools"}},
		ToolCalls: calls,
	}
	msgs := []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "earlier task"),
		llmkit.TextMessage(llmkit.RoleAssistant, "earlier answer"),
		asst,
	}
	for i := 0; i < answered; i++ {
		res := llmkit.TextMessage(llmkit.RoleToolResult, fmt.Sprintf("result %d", i+1))
		res.ToolCallID = calls[i].ID
		msgs = append(msgs, res)
	}
	return &Outcome{Messages: msgs}
}

// TestRunContinue_TrimsDanglingToolTurnSeed pins the seed-hygiene rule: a
// continued seed whose trailing assistant turn carries tool calls not all
// answered by later tool results is trimmed from that assistant turn onward
// before the wire request, while everything before it stays intact and the
// caller's Outcome is never mutated.
func TestRunContinue_TrimsDanglingToolTurnSeed(t *testing.T) {
	fc := newFakeClient(textResp("continued", 10, 5))
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	prev := danglingPrev(t, 1) // t1 answered, t2 dangling
	out, err := r.RunContinue(context.Background(), prev, "next task")
	if err != nil {
		t.Fatalf("RunContinue: %v", err)
	}
	if out.FinalText != "continued" {
		t.Errorf("FinalText = %q, want %q", out.FinalText, "continued")
	}

	// The wire request must hold the intact prefix plus the new task — and
	// neither the dangling assistant turn nor its orphan result.
	if len(fc.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(fc.requests))
	}
	msgs := fc.requests[0].Messages
	if len(msgs) != 3 {
		t.Fatalf("request has %d messages, want 3 (intact prefix of 2 + new task); the dangling turn was not trimmed", len(msgs))
	}
	sameTurn(t, "request message 0", msgs[0], prev.Messages[0])
	sameTurn(t, "request message 1", msgs[1], prev.Messages[1])
	for i, m := range msgs {
		if len(m.ToolCalls) > 0 {
			t.Errorf("request message %d still carries tool calls %+v", i, m.ToolCalls)
		}
		if m.Role == llmkit.RoleToolResult {
			t.Errorf("request message %d is an orphan tool result (ToolCallID %q)", i, m.ToolCallID)
		}
	}

	// The caller's Outcome must be untouched: trimming happens on the run's
	// own copy, never on prev.Messages.
	if len(prev.Messages) != 4 {
		t.Fatalf("prev.Messages = %d entries after RunContinue, want 4 (caller's history must not be mutated)", len(prev.Messages))
	}
	if prev.Messages[3].ToolCallID != "t1" {
		t.Errorf("prev.Messages[3] = %+v, want the t1 tool result the caller stored", prev.Messages[3])
	}
}

// TestRunContinue_KeepsCompleteToolTurnSeed is the no-trim counterpart: when
// every trailing tool call IS answered, the continued request must still carry
// the assistant tool-call turn and all of its results.
func TestRunContinue_KeepsCompleteToolTurnSeed(t *testing.T) {
	fc := newFakeClient(textResp("continued", 10, 5))
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	prev := danglingPrev(t, 2) // both calls answered — nothing to trim
	if _, err := r.RunContinue(context.Background(), prev, "next task"); err != nil {
		t.Fatalf("RunContinue: %v", err)
	}

	msgs := fc.requests[0].Messages
	if len(msgs) != len(prev.Messages)+1 {
		t.Fatalf("request has %d messages, want %d -- a complete trailing tool turn must NOT be trimmed", len(msgs), len(prev.Messages)+1)
	}
	for i, want := range prev.Messages {
		sameTurn(t, fmt.Sprintf("request message %d", i), msgs[i], want)
	}
	if msgs[len(msgs)-1].Text() != "next task" {
		t.Errorf("request last message = %q, want the new task", msgs[len(msgs)-1].Text())
	}
}

// TestRunContinue_CompactionNamesToolFromSeed pins the toolNameByID rebuild:
// when compaction fires on a continued history, a PRIOR run's tool result must
// stub with its real tool name, not the generic "tool" fallback — which is
// what an empty map at first-turn compaction would produce.
func TestRunContinue_CompactionNamesToolFromSeed(t *testing.T) {
	blob := strings.Repeat("data line\n", 800) // ~8 KB per tool result

	// Hand-built prior-run history: five big tool turns (oldest first), so the
	// recent-4 window leaves exactly t1's result prunable.
	var seed []llmkit.Message
	seed = append(seed, llmkit.TextMessage(llmkit.RoleUser, "prior task"))
	for i := 1; i <= 5; i++ {
		seed = append(seed, llmkit.Message{
			Role: llmkit.RoleAssistant,
			ToolCalls: []llmkit.ToolCall{{
				ID:        fmt.Sprintf("t%d", i),
				Name:      "big",
				Arguments: json.RawMessage(`{}`),
			}},
		})
		res := llmkit.TextMessage(llmkit.RoleToolResult, blob)
		res.ToolCallID = fmt.Sprintf("t%d", i)
		seed = append(seed, res)
	}
	prev := &Outcome{Messages: seed}

	fc := newFakeClient(textResp("continued", 10, 5))
	// Threshold ~1.5k tokens (bytes/4): the ~40 KB seed history crosses it on
	// the first turn of the continuation.
	r := NewRunner(fc, nil, "sys", WithLimits(Limits{HistoryTokenBudget: 1500}))

	if _, err := r.RunContinue(context.Background(), prev, "next task"); err != nil {
		t.Fatalf("RunContinue: %v", err)
	}
	if len(fc.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(fc.requests))
	}

	var stubbed string
	found := false
	for _, m := range fc.requests[0].Messages {
		if m.Role == llmkit.RoleToolResult && m.ToolCallID == "t1" {
			found = true
			stubbed = m.Text()
		}
	}
	if !found {
		t.Fatal("t1's tool result is missing from the wire request -- compaction never fired, the test is vacuous")
	}
	if !strings.HasPrefix(stubbed, compactStubPrefix) {
		t.Fatalf("t1 result was not stubbed (got %d bytes of raw content) -- compaction never fired, the test is vacuous", len(stubbed))
	}
	if !strings.Contains(stubbed, ": big,") {
		t.Errorf("stub = %q, want it to name the tool %q -- the seed's tool calls were not rebuilt into toolNameByID", stubbed, "big")
	}
	// And the untouched tail must still be intact.
	for _, m := range fc.requests[0].Messages {
		if m.Role == llmkit.RoleToolResult && m.ToolCallID == "t5" && m.Text() != blob {
			t.Errorf("t5 result was mutated: %d bytes, want the original %d", len(m.Text()), len(blob))
		}
	}
	// The caller's seed is never mutated in place.
	if prev.Messages[2].Text() != blob {
		t.Error("prev.Messages t1 result was mutated in place")
	}
}

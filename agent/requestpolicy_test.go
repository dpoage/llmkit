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

// rpTool is a minimal always-succeeds tool for policy tests.
type rpTool struct{ name string }

func (t rpTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "echoes ok"}
}

func (t rpTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	return "ok", nil
}

// rpRecorder collects the step numbers PrepareRequest fires with, in order.
type rpRecorder struct{ steps []int }

func (rec *rpRecorder) policy() RequestPolicy {
	return RequestPolicyFunc(func(_ context.Context, step int, _ *llmkit.Request) error {
		rec.steps = append(rec.steps, step)
		return nil
	})
}

// TestRequestPolicySetsRequestFields pins that whatever the policy writes onto the request is what the client receives on the wire.
func TestRequestPolicySetsRequestFields(t *testing.T) {
	fake := newFakeClient(textResp("done", 10, 5))
	temp := 0.2
	seed := int64(7)
	rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
		req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
		req.Temperature = &temp
		req.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}
		req.Seed = &seed
		return nil
	})
	r := NewRunner(fake, nil, "sys", WithRequestPolicy(rp))
	if _, err := r.Run(context.Background(), "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if fake.callCount() != 1 {
		t.Fatalf("completions = %d, want 1", fake.callCount())
	}
	got := fake.requests[0]
	if got.Thinking == nil || got.Thinking.BudgetTokens != 1024 {
		t.Errorf("wire Thinking = %+v, want budget 1024", got.Thinking)
	}
	if got.Temperature == nil || *got.Temperature != 0.2 {
		t.Errorf("wire Temperature = %+v, want 0.2", got.Temperature)
	}
	if got.ToolChoice.Mode != llmkit.ToolChoiceRequired {
		t.Errorf("wire ToolChoice = %+v, want required", got.ToolChoice)
	}
	if got.Seed == nil || *got.Seed != 7 {
		t.Errorf("wire Seed = %+v, want 7", got.Seed)
	}
}

// TestRequestPolicyShapesWireMessagesOnly pins that the policy edits a per-turn clone, so the wire view diverges from the loop's history and the loop itself stays untouched, while the transcript records the post-policy wire messages.
func TestRequestPolicyShapesWireMessagesOnly(t *testing.T) {
	t.Run("drop first message", func(t *testing.T) {
		fake := newFakeClient(textResp("a", 5, 2), textResp("b", 5, 2))
		rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
			req.Messages = req.Messages[1:]
			return nil
		})
		r := NewRunner(fake, nil, "sys", WithRequestPolicy(rp))

		out1, err := r.Run(context.Background(), "a")
		if err != nil {
			t.Fatalf("run 1: %v", err)
		}
		if got := fake.requests[0].Messages; len(got) != 0 {
			t.Errorf("wire messages run 1 = %d, want 0 (seed dropped)", len(got))
		}
		if got := len(out1.Messages); got != 2 {
			t.Errorf("run 1 history = %d messages, want 2 (seed + assistant)", got)
		}

		out2, err := r.Run(context.Background(), "b", Continue(out1))
		if err != nil {
			t.Fatalf("run 2: %v", err)
		}
		wire := fake.requests[1].Messages
		if len(wire) != 2 {
			t.Fatalf("wire messages run 2 = %d, want 2 (3-message history minus seed)", len(wire))
		}
		if wire[0].Role != llmkit.RoleAssistant || wire[0].Text() != "a" {
			t.Errorf("wire[0] = %s %q, want assistant %q", wire[0].Role, wire[0].Text(), "a")
		}
		if len(out2.Messages) != 4 || out2.Messages[0].Text() != "a" {
			t.Errorf("run 2 history = %+v, want the 4-message conversation incl. seed turn", out2.Messages)
		}
		for _, ev := range out2.Transcript.Record {
			if ev.Kind != llmkit.KindCompletion || ev.Completion == nil || ev.Step != 1 {
				continue
			}
			if !reflect.DeepEqual(ev.Completion.Request.Messages, wire) {
				t.Errorf("transcript step-1 request = %+v, want the wire messages %+v", ev.Completion.Request.Messages, wire)
			}
		}
	})

	t.Run("append scoped to the turn", func(t *testing.T) {
		fake := newFakeClient(textResp("a", 5, 2))
		rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
			req.Messages = append(req.Messages, llmkit.TextMessage(llmkit.RoleUser, "steer"))
			return nil
		})
		r := NewRunner(fake, nil, "sys", WithRequestPolicy(rp))
		out, err := r.Run(context.Background(), "a")
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		wire := fake.requests[0].Messages
		if len(wire) != 2 || wire[1].Text() != "steer" {
			t.Errorf("wire messages = %+v, want seed turn plus steering turn", wire)
		}
		if len(out.Messages) != 2 {
			t.Fatalf("history = %d messages, want 2", len(out.Messages))
		}
		for _, m := range out.Messages {
			if m.Text() == "steer" {
				t.Error("appended steering turn leaked into the loop history")
			}
		}
	})
}

// TestRequestPolicyErrorAbortsBeforeWire pins that a policy error aborts the completion before any client call and comes back from Run wrapped with the iteration number.
func TestRequestPolicyErrorAbortsBeforeWire(t *testing.T) {
	sentinel := errors.New("no thinking on tuesdays")
	fake := newFakeClient(textResp("done", 10, 5))
	rp := RequestPolicyFunc(func(_ context.Context, _ int, _ *llmkit.Request) error {
		return sentinel
	})
	r := NewRunner(fake, nil, "sys", WithRequestPolicy(rp))
	_, err := r.Run(context.Background(), "hello")
	if err == nil {
		t.Fatal("Run err = nil, want the policy error")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v, want wrap of sentinel", err)
	}
	if !strings.Contains(err.Error(), "agent: request policy at iteration 1") {
		t.Errorf("err text %q, want the step-qualified wrap", err)
	}
	if got := fake.callCount(); got != 0 {
		t.Errorf("completions = %d, want 0 (abort before the wire call)", got)
	}
}

// TestRequestPolicyFiresOncePerCompletion pins that continuation, forced finalization, and RunJSON repair turns all pass through the policy exactly as often as through the client, under their transcript steps.
func TestRequestPolicyFiresOncePerCompletion(t *testing.T) {
	fake := newFakeClient(
		maxTokensResp("half an", 10, 5),
		toolResp("t1", "now", "{}", 10, 5),
		textResp("not json", 10, 5),
		textResp(`{"k":"v"}`, 10, 5),
	)
	rec := &rpRecorder{}
	r := NewRunner(fake, []Tool{rpTool{name: "now"}}, "sys",
		WithRequestPolicy(rec.policy()),
		WithLimits(Limits{MaxIterations: 1}))
	var out map[string]any
	if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &out); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if got := rec.steps; !reflect.DeepEqual(got, []int{1, 2, 3, 4}) {
		t.Errorf("PrepareRequest steps = %v, want [1 2 3 4]", got)
	}
	if got := fake.callCount(); got != 4 {
		t.Fatalf("completions = %d, want 4 (main, continuation, finalization, repair)", got)
	}
	for i, want := range map[int]string{2: "finalization", 3: "repair"} {
		if got := fake.requests[i].Tools; len(got) != 0 {
			t.Errorf("%s wire request carries %d tools, want 0", want, len(got))
		}
	}
	if out["k"] != "v" {
		t.Errorf("RunJSON out = %v, want k=v", out)
	}
}

// TestReplayClientReplaysPolicyRecordedRun pins that a run recorded through a policy snapshots the post-policy wire messages and ReplayClient still replays that transcript cleanly.
func TestReplayClientReplaysPolicyRecordedRun(t *testing.T) {
	prepend := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
		req.Messages = append([]llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "(context refresher)")}, req.Messages...)
		return nil
	})
	recFake := newFakeClient(toolResp("t1", "now", "{}", 10, 5), textResp("answer", 10, 5))
	rec := NewRunner(recFake, []Tool{rpTool{name: "now"}}, "sys", WithRequestPolicy(prepend))
	want, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	for _, ev := range want.Transcript.Record {
		if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Step == 1 && len(ev.Completion.Request.Messages) != 2 {
			t.Errorf("recorded step-1 request has %d messages, want the post-policy 2", len(ev.Completion.Request.Messages))
		}
	}
	rp, err := NewReplayClient(want.Transcript, want.RunID, recFake.Capabilities())
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	replay := NewRunner(rp, []Tool{rpTool{name: "now"}}, "sys", WithRequestPolicy(prepend))
	got, err := replay.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay run: %v", err)
	}
	if got.FinalText != want.FinalText {
		t.Errorf("replay FinalText = %q, want %q", got.FinalText, want.FinalText)
	}
	if got.Iterations != want.Iterations {
		t.Errorf("replay Iterations = %d, want %d", got.Iterations, want.Iterations)
	}
}

// TestNilRequestPolicySendsRequestAsBuilt pins that with no policy the request reaches the client exactly as the runner built it.
func TestNilRequestPolicySendsRequestAsBuilt(t *testing.T) {
	fake := newFakeClient(textResp("done", 10, 5))
	r := NewRunner(fake, nil, "sys")
	if _, err := r.Run(context.Background(), "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := fake.requests[0].Messages
	if len(got) != 1 || got[0].Role != llmkit.RoleUser || got[0].Text() != "hello" {
		t.Errorf("wire messages = %+v, want the single seeded user turn", got)
	}
}

// TestRequestPolicyInPlaceWritesLeaveHistoryUntouched pins the shallow-clone contract under in-place slice writes — cases a re-slice would hide.
func TestRequestPolicyInPlaceWritesLeaveHistoryUntouched(t *testing.T) {
	ctx := context.Background()

	t.Run("index swap on the main turn", func(t *testing.T) {
		fake := newFakeClient(textResp("a", 5, 2), textResp("b", 5, 2))
		rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
			if len(req.Messages) >= 2 {
				req.Messages[0], req.Messages[1] = req.Messages[1], req.Messages[0]
			}
			return nil
		})
		r := NewRunner(fake, nil, "sys", WithRequestPolicy(rp))
		out1, err := r.Run(ctx, "a")
		if err != nil {
			t.Fatalf("run 1: %v", err)
		}
		out2, err := r.Run(ctx, "b", Continue(out1))
		if err != nil {
			t.Fatalf("run 2: %v", err)
		}
		want := []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleAssistant, "a"),
			llmkit.TextMessage(llmkit.RoleUser, "a"),
			llmkit.TextMessage(llmkit.RoleUser, "b"),
		}
		if !reflect.DeepEqual(fake.requests[1].Messages, want) {
			t.Errorf("wire = %+v, want the swapped order %+v", fake.requests[1].Messages, want)
		}
		wantOut := []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "a"),
			llmkit.TextMessage(llmkit.RoleAssistant, "a"),
			llmkit.TextMessage(llmkit.RoleUser, "b"),
			llmkit.TextMessage(llmkit.RoleAssistant, "b"),
		}
		if !reflect.DeepEqual(out2.Messages, wantOut) {
			t.Errorf("outcome history = %+v, want the unswapped canonical order", out2.Messages)
		}
		if !reflect.DeepEqual(out1.Messages, wantOut[:2]) {
			t.Errorf("run-1 history = %+v, want untouched seed %+v", out1.Messages, wantOut[:2])
		}
		for _, ev := range out2.Transcript.Record {
			if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Step == 1 &&
				!reflect.DeepEqual(ev.Completion.Request.Messages, want) {
				t.Errorf("transcript step-1 request = %+v, want the wire messages", ev.Completion.Request.Messages)
			}
		}
	})

	t.Run("overlapping filter append", func(t *testing.T) {
		fake := newFakeClient(textResp("a", 5, 2), textResp("b", 5, 2))
		rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
			if len(req.Messages) >= 3 {
				req.Messages = append(req.Messages[:1], req.Messages[2:]...)
			}
			return nil
		})
		r := NewRunner(fake, nil, "sys", WithRequestPolicy(rp))
		out1, err := r.Run(ctx, "a")
		if err != nil {
			t.Fatalf("run 1: %v", err)
		}
		out2, err := r.Run(ctx, "b", Continue(out1))
		if err != nil {
			t.Fatalf("run 2: %v", err)
		}
		want := []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "a"),
			llmkit.TextMessage(llmkit.RoleUser, "b"),
		}
		if !reflect.DeepEqual(fake.requests[1].Messages, want) {
			t.Errorf("wire = %+v, want the filtered pair %+v", fake.requests[1].Messages, want)
		}
		wantOut := []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "a"),
			llmkit.TextMessage(llmkit.RoleAssistant, "a"),
			llmkit.TextMessage(llmkit.RoleUser, "b"),
			llmkit.TextMessage(llmkit.RoleAssistant, "b"),
		}
		if !reflect.DeepEqual(out2.Messages, wantOut) {
			t.Errorf("outcome history = %+v, want the unfiltered canonical order", out2.Messages)
		}
	})

	t.Run("append into spare capacity on main and continuation fires", func(t *testing.T) {
		fake := newFakeClient(textResp("a", 5, 2), toolResp("t1", "now", "{}", 5, 2), textResp("done", 5, 2))
		rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
			req.Messages = append(req.Messages, llmkit.TextMessage(llmkit.RoleUser, "steer"))
			return nil
		})
		r := NewRunner(fake, []Tool{rpTool{name: "now"}}, "sys", WithRequestPolicy(rp))
		out1, err := r.Run(ctx, "a")
		if err != nil {
			t.Fatalf("run 1: %v", err)
		}
		out2, err := r.Run(ctx, "b", Continue(out1))
		if err != nil {
			t.Fatalf("run 2: %v", err)
		}
		if got := fake.requests[0].Messages; len(got) != 2 || got[1].Text() != "steer" {
			t.Errorf("fire-1 wire = %+v, want seed plus steering turn", got)
		}
		for i, wantLen := range map[int]int{1: 4, 2: 6} {
			got := fake.requests[i].Messages
			if len(got) != wantLen {
				t.Errorf("fire-%d wire = %d messages, want %d", i+1, len(got), wantLen)
				continue
			}
			if got[wantLen-1].Text() != "steer" {
				t.Errorf("fire-%d wire last = %q, want the steering turn", i+1, got[wantLen-1].Text())
			}
		}
		if len(out2.Messages) != 6 {
			t.Fatalf("outcome history = %d messages, want 6", len(out2.Messages))
		}
		if out2.Messages[0].Text() != "a" || out2.Messages[2].Text() != "b" {
			t.Errorf("outcome history heads = %q, %q; want a, b", out2.Messages[0].Text(), out2.Messages[2].Text())
		}
		for _, m := range out2.Messages {
			if m.Text() == "steer" {
				t.Error("appended steering turn leaked into the loop history")
			}
		}
		for _, step := range []int{1, 2} {
			for _, ev := range out2.Transcript.Record {
				if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Step == step &&
					!reflect.DeepEqual(ev.Completion.Request.Messages, fake.requests[step].Messages) {
					t.Errorf("transcript step-%d request != retained wire request", step)
				}
			}
		}
	})

	t.Run("in-place writes across continuation, finalization, and repair fires", func(t *testing.T) {
		fake := newFakeClient(
			maxTokensResp("half an", 10, 5),
			toolResp("t1", "now", "{}", 10, 5),
			textResp("not json", 10, 5),
			textResp(`{"k":"v"}`, 10, 5),
		)
		preps := 0
		rp := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
			preps++
			if len(req.Messages) >= 2 {
				last := len(req.Messages) - 1
				req.Messages[0], req.Messages[last] = req.Messages[last], req.Messages[0]
			}
			return nil
		})
		r := NewRunner(fake, []Tool{rpTool{name: "now"}}, "sys",
			WithRequestPolicy(rp),
			WithLimits(Limits{MaxIterations: 1}))
		var out map[string]any
		outcome, err := r.RunJSON(ctx, "policy-repair-probe task", json.RawMessage(`{"type":"object"}`), &out)
		if err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		if out["k"] != "v" {
			t.Errorf("RunJSON out = %v, want k=v", out)
		}
		if preps != 4 || fake.callCount() != 4 {
			t.Errorf("PrepareRequest calls = %d, completions = %d, want 4 and 4", preps, fake.callCount())
		}
		if got := fake.requests[1].Messages; len(got) != 3 ||
			!strings.HasPrefix(got[0].Text(), "Your previous message was cut off") ||
			got[1].Text() != "half an" ||
			!strings.Contains(got[2].Text(), "policy-repair-probe task") {
			t.Errorf("continuation wire = %q, %q, %q; want [continuation nudge, assistant half, task]",
				got[0].Text(), got[1].Text(), got[2].Text())
		}
		msgs := outcome.Messages
		if len(msgs) == 0 {
			t.Fatal("canonical history empty")
		}
		if !strings.Contains(msgs[0].Text(), "policy-repair-probe task") {
			t.Errorf("canonical history[0] = %q, want the seeded task turn", msgs[0].Text())
		}
		if msgs[len(msgs)-1].Role != llmkit.RoleAssistant {
			t.Errorf("canonical history ends with role %q, want assistant", msgs[len(msgs)-1].Role)
		}
	})
}

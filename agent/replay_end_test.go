package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// replayEndRecord is one recorded run and how to replay it faithfully.
type replayEndRecord struct {
	name    string
	tools   []Tool
	steps   []scriptStep
	recOpts []Option
	// repOpts are the replay Runner's options given the client; nil means
	// the replay installs rc.ToolPolicy() only when the record has denials.
	repOpts func(rc *ReplayClient) []Option
	json    bool // record and replay through RunJSON
}

func replayEndRun(t *testing.T, c llmkit.Client, tools []Tool, opts []Option, useJSON bool) (*Outcome, error) {
	t.Helper()
	r := NewRunner(c, tools, "sys", opts...)
	if useJSON {
		var got item
		return r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &got)
	}
	return r.Run(context.Background(), "task")
}

// TestReplayEnd_FaithfulReplayHasNoLeftover pins that Err is nil for a replay
// that consumes everything the record served, through every shape of run the
// end-of-run check must not mistake for a leftover.
func TestReplayEnd_FaithfulReplayHasNoLeftover(t *testing.T) {
	prefill := func(rc *ReplayClient) []Option {
		return []Option{WithToolPolicy(rc.ToolPolicy()), WithRequestPolicy(RequestPolicyFunc(
			func(_ context.Context, _ int, req *llmkit.Request) error {
				req.Messages = append(req.Messages, llmkit.Message{Role: llmkit.RoleAssistant, Content: []llmkit.Block{llmkit.Text("{")}})
				return nil
			}))}
	}
	parallel := func(rc *ReplayClient) []Option {
		return []Option{WithToolPolicy(rc.ToolPolicy()), WithParallelTools()}
	}
	capped := []Option{WithLimits(Limits{MaxIterations: 2})}
	rows := []replayEndRecord{
		{name: "tool turns", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`), mkCall("c2", "rm", `{"p":"/tmp"}`)),
			toolTurn(mkCall("c3", "ls", `{"p":"/"}`)),
			textResp("done", 1, 1),
		}},
		{name: "parallel tools", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`), mkCall("c2", "ls", `{"p":"."}`), mkCall("c3", "rm", `{"p":"/tmp"}`)),
			textResp("done", 1, 1),
		}, recOpts: []Option{WithParallelTools()}, repOpts: parallel},
		{name: "recorded denials", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`), mkCall("c2", "rm", `{"p":"/"}`)),
			toolTurn(mkCall("c3", "secret", `{}`)),
			textResp("done", 1, 1),
		}, recOpts: []Option{WithToolPolicy(denialPolicy)}},
		{name: "unknown and empty-named tools", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "nope", `{}`), mkCall("c2", "", `{}`), mkCall("c3", "ls", `{"p":"."}`)),
			textResp("done", 1, 1),
		}},
		{name: "max-tokens continuation", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
			maxTokensResp("partial answ", 1, 1),
			textResp("er", 1, 1),
		}},
		{name: "RunJSON with repair", tools: echoTools(), json: true, steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
			textResp("not json", 1, 1),
			textResp(`{"path":"a.go","note":"n"}`, 1, 1),
		}},
		{name: "step-capped record", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
			toolTurn(mkCall("c2", "rm", `{"p":"/tmp"}`)),
			textResp("cut", 1, 1),
		}, recOpts: capped, repOpts: func(rc *ReplayClient) []Option { return append([]Option{WithToolPolicy(rc.ToolPolicy())}, capped...) }},
		{name: "request policy appends a prefill", tools: echoTools(), steps: []scriptStep{
			toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
			textResp("done", 1, 1),
		}, repOpts: prefill},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			rec, recErr := replayEndRun(t, newFakeClient(row.steps...), row.tools, row.recOpts, row.json)
			if rec == nil {
				t.Fatalf("record run: %v", recErr)
			}
			rc, err := NewReplayClient(rec.Transcript, rec.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			repOpts := []Option{WithToolPolicy(rc.ToolPolicy())}
			if row.repOpts != nil {
				repOpts = row.repOpts(rc)
			}
			_, repErr := replayEndRun(t, rc, rc.Tools(), repOpts, row.json)
			var recInc, repInc *IncompleteError
			if row.name == "step-capped record" && (!errors.As(recErr, &recInc) || !errors.As(repErr, &repInc)) {
				t.Errorf("record Run err = %v, replay Run err = %v, want both *IncompleteError", recErr, repErr)
			}
			if (recErr == nil) != (repErr == nil) {
				t.Errorf("record Run err = %v, replay Run err = %v", recErr, repErr)
			}
			if err := rc.Err(); err != nil {
				t.Errorf("rc.Err() = %v, want nil for a faithful replay", err)
			}
		})
	}
}

// TestReplayEnd_FinalTurnPolicyChangeIsReported pins the final-turn shape:
// a caller policy stricter than the recorded one, whose only effect is on
// the last tool turn, leaves a recorded call unserved with no later Complete
// to notice it.
func TestReplayEnd_FinalTurnPolicyChangeIsReported(t *testing.T) {
	rec, err := NewRunner(newFakeClient(
		toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
		toolTurn(mkCall("c2", "rm", `{"p":"/tmp"}`)),
		textResp("end", 1, 1),
	), echoTools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	stricter := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if call.Name == "rm" {
			return errors.New("no rm")
		}
		return nil
	})

	rc, err := NewReplayClient(rec.Transcript, rec.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRunner(rc, rc.Tools(), "sys", WithToolPolicy(stricter)).Run(context.Background(), "task"); err != nil {
		t.Fatalf("replay Run: %v", err)
	}
	got := rc.Err()
	if !errors.Is(got, ErrReplayDiverged) {
		t.Fatalf("rc.Err() = %v, want ErrReplayDiverged", got)
	}
	if !strings.Contains(got.Error(), "step 2") || !strings.Contains(got.Error(), `"rm"`) {
		t.Errorf("rc.Err() = %q, want it to name recorded step 2 and tool rm", got)
	}

	// The same policy on a client whose tools the caller never took (live
	// tools only) is not compared: the calls check is opted into by Tools.
	live, err := NewReplayClient(rec.Transcript, rec.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRunner(live, echoTools(), "sys", WithToolPolicy(stricter)).Run(context.Background(), "task"); err != nil {
		t.Fatalf("live replay Run: %v", err)
	}
	if err := live.Err(); err != nil {
		t.Errorf("live-tool replay Err() = %v, want nil", err)
	}
}

// TestReplayEnd_UnservedCompletionIsReported pins the completions shape: the
// recorded run received a steering follow-up, the replay does not.
func TestReplayEnd_UnservedCompletionIsReported(t *testing.T) {
	s := NewSteering()
	s.FollowUp(llmkit.Text("more"))
	rec, err := NewRunner(newFakeClient(textResp("one", 1, 1), textResp("after follow-up", 1, 1)), nil, "sys").
		Run(context.Background(), "task", WithSteering(s))
	if err != nil {
		t.Fatal(err)
	}
	if rec.FinalText != "after follow-up" {
		t.Fatalf("fixture recorded final %q", rec.FinalText)
	}

	rc, err := NewReplayClient(rec.Transcript, rec.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if err := rc.Err(); err != nil {
		t.Fatalf("Err() = %v before any completion was served, want nil", err)
	}
	out, err := NewRunner(rc, nil, "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if out.FinalText != "one" {
		t.Fatalf("replay final = %q, want %q (no follow-up delivered)", out.FinalText, "one")
	}
	got := rc.Err()
	if !errors.Is(got, ErrReplayDiverged) || !strings.Contains(got.Error(), "step 2") {
		t.Errorf("rc.Err() = %v, want ErrReplayDiverged naming recorded step 2", got)
	}
}

// TestReplayEnd_TakenToolsNeverGivenToRunner pins the documented cost of
// opting in: a caller that takes rc.Tools() and never passes them to the
// Runner gets a leftover error for a record that holds a call a tool ran.
func TestReplayEnd_TakenToolsNeverGivenToRunner(t *testing.T) {
	rec, err := NewRunner(newFakeClient(
		toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
		textResp("done", 1, 1),
	), echoTools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	rc, err := NewReplayClient(rec.Transcript, rec.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	_ = rc.Tools()
	if _, err := NewRunner(rc, echoTools(), "sys").Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if err := rc.Err(); !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "step 1") {
		t.Errorf("rc.Err() = %v, want ErrReplayDiverged naming step 1", err)
	}
}

// TestReplayEnd_ErrMidRunDoesNotStick pins that Err called before the run
// ends reports the work not yet replayed, and that the report does not turn
// into a recorded divergence: the run still finishes and Err is nil.
func TestReplayEnd_ErrMidRunDoesNotStick(t *testing.T) {
	rec, err := NewRunner(newFakeClient(
		toolTurn(mkCall("c1", "ls", `{"p":"."}`)),
		textResp("done", 1, 1),
	), echoTools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	rc, err := NewReplayClient(rec.Transcript, rec.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	mid := &midRunErrClient{rc: rc}
	if _, err := NewRunner(mid, rc.Tools(), "sys").Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !errors.Is(mid.errAfterFirst, ErrReplayDiverged) {
		t.Errorf("Err() after the first Complete = %v, want the unreached work reported", mid.errAfterFirst)
	}
	if err := rc.Err(); err != nil {
		t.Errorf("Err() after the run = %v, want nil: the mid-run report must not stick", err)
	}
}

// midRunErrClient asks the replay client for Err right after it served its
// first completion.
type midRunErrClient struct {
	rc            *ReplayClient
	n             int
	errAfterFirst error
}

func (m *midRunErrClient) Capabilities() llmkit.Capabilities { return m.rc.Capabilities() }
func (m *midRunErrClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	resp, err := m.rc.Complete(ctx, req)
	if m.n++; m.n == 1 {
		m.errAfterFirst = m.rc.Err()
	}
	return resp, err
}

// TestReplayEnd_ScriptedDoubleReportsNoLeftover pins that the responses
// double reports a divergence only when it serves past its last response.
func TestReplayEnd_ScriptedDoubleReportsNoLeftover(t *testing.T) {
	resps := []llmkit.Response{{Text: "a", StopReason: llmkit.StopEndTurn}, {Text: "b", StopReason: llmkit.StopEndTurn}}
	rc := NewReplayClientFromResponses(resps, llmkit.Capabilities{})
	_ = rc.Tools()
	if _, err := rc.Complete(context.Background(), llmkit.Request{}); err != nil {
		t.Fatal(err)
	}
	if err := rc.Err(); err != nil {
		t.Errorf("Err() = %v with one of two scripted responses unused, want nil", err)
	}
	for range resps {
		_, _ = rc.Complete(context.Background(), llmkit.Request{})
	}
	if err := rc.Err(); !errors.Is(err, ErrReplayDiverged) {
		t.Errorf("Err() = %v after serving past the last response, want ErrReplayDiverged", err)
	}
}

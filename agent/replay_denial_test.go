package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// denialPolicy is the recording policy: it denies every "secret" call and
// "rm" on the root path, and allows the rest — so "rm" appears both denied
// and allowed, and "secret" only ever denied.
var denialPolicy = ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
	switch {
	case call.Name == "secret":
		return errors.New("secrets are off limits")
	case call.Name == "rm" && string(call.Arguments) == `{"p":"/"}`:
		return errors.New("refusing to wipe the root")
	}
	return nil
})

func recordingDenialPolicy() ToolPolicy { return denialPolicy }

// dedupePolicy returns a fresh recording policy that denies a call repeating
// the name and arguments of a call it already allowed in the run, so of two
// identical calls in one turn the first is allowed and the second denied.
func dedupePolicy() ToolPolicy {
	seen := map[string]bool{}
	return ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		key := call.Name + " " + string(call.Arguments)
		if seen[key] {
			return errors.New("duplicate call")
		}
		seen[key] = true
		return nil
	})
}

func toolTurn(calls ...llmkit.ToolCall) scriptStep {
	return scriptStep{resp: llmkit.Response{StopReason: llmkit.StopToolUse, ToolCalls: calls}}
}

func mkCall(id, name, args string) llmkit.ToolCall {
	return llmkit.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(args)}
}

func echoTools() []Tool {
	return []Tool{echoTool{name: "ls"}, echoTool{name: "rm"}, echoTool{name: "secret"}}
}

// denialRecords are runs that record a ToolPolicy denial followed by at
// least one later tool turn, and end cleanly on an end_turn. policy returns
// the recording policy, fresh per run.
var denialRecords = []struct {
	name   string
	policy func() ToolPolicy
	steps  []scriptStep
}{
	{"denial then more tool turns", recordingDenialPolicy, []scriptStep{
		toolTurn(mkCall("c1", "ls", `{"p":"."}`), mkCall("c2", "rm", `{"p":"/"}`)),
		toolTurn(mkCall("c3", "rm", `{"p":"/tmp"}`), mkCall("c4", "secret", `{}`)),
		textResp("done", 1, 1),
	}},
	// "secret" is only ever denied, and the first tool turn holds nothing
	// else: replay must still register the name and step past the turn.
	{"denied name has no allowed call", recordingDenialPolicy, []scriptStep{
		toolTurn(mkCall("c1", "secret", `{}`)),
		toolTurn(mkCall("c2", "ls", `{"p":"."}`)),
		textResp("done", 1, 1),
	}},
	// Two identical calls in one turn, the first allowed and the second
	// denied: the replay must keep each outcome on its own call.
	{"identical calls, second denied", dedupePolicy, []scriptStep{
		toolTurn(mkCall("c1", "ls", `{"p":"."}`), mkCall("c2", "ls", `{"p":"."}`)),
		toolTurn(mkCall("c3", "rm", `{"p":"/tmp"}`)),
		textResp("done", 1, 1),
	}},
}

type recordedToolRun struct {
	Step int
	Run  llmkit.ToolRunEvent
}

func toolRuns(t *testing.T, tr *Transcript) []recordedToolRun {
	t.Helper()
	var runs []recordedToolRun
	for _, ev := range tr.Record {
		if ev.Kind == llmkit.KindToolRun {
			runs = append(runs, recordedToolRun{Step: ev.Step, Run: *ev.ToolRun})
		}
	}
	return runs
}

func recordDenialRun(t *testing.T, policy ToolPolicy, steps []scriptStep) *Outcome {
	t.Helper()
	out, err := NewRunner(newFakeClient(steps...), echoTools(), "sys", WithToolPolicy(policy)).
		Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	denied := 0
	for _, r := range toolRuns(t, out.Transcript) {
		if r.Run.Denied {
			denied++
		}
	}
	if denied == 0 {
		t.Fatal("fixture recorded no denial")
	}
	return out
}

// TestReplay_DenialRecordReplaysCleanly pins RP-1: a record with a policy
// denial followed by later tool turns — identical calls with different
// outcomes in one turn included — replays with no error and no divergence,
// and the replayed ToolRun events equal the recorded ones field by field and
// in order (Denied and DenyReason included), serially and under
// WithParallelTools, both under the recording policy and under the client's
// ToolPolicy with no caller policy.
func TestReplay_DenialRecordReplaysCleanly(t *testing.T) {
	policies := []struct {
		name string
		opt  func(rc *ReplayClient, recording func() ToolPolicy) Option
	}{
		{"same caller policy", func(_ *ReplayClient, recording func() ToolPolicy) Option { return WithToolPolicy(recording()) }},
		{"client ToolPolicy", func(rc *ReplayClient, _ func() ToolPolicy) Option { return WithToolPolicy(rc.ToolPolicy()) }},
	}
	for _, rec := range denialRecords {
		for _, pol := range policies {
			for _, parallel := range []bool{false, true} {
				name := rec.name + "/" + pol.name
				if parallel {
					name += "/parallel"
				}
				t.Run(name, func(t *testing.T) {
					out := recordDenialRun(t, rec.policy(), rec.steps)
					rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
					if err != nil {
						t.Fatal(err)
					}
					opts := []Option{pol.opt(rc, rec.policy)}
					if parallel {
						opts = append(opts, WithParallelTools())
					}
					replayed, err := NewRunner(rc, rc.Tools(), "sys", opts...).Run(context.Background(), "task")
					if err != nil {
						t.Fatalf("replay Run: %v", err)
					}
					if rc.Err() != nil {
						t.Fatalf("replay diverged: %v", rc.Err())
					}
					want, got := toolRuns(t, out.Transcript), toolRuns(t, replayed.Transcript)
					if !reflect.DeepEqual(want, got) {
						t.Errorf("replayed ToolRun events differ\nrecorded: %+v\nreplayed: %+v", want, got)
					}
				})
			}
		}
	}
}

// TestReplay_ToolsRegisterEveryRecordedName pins that a name the record only
// ever denied is still a registered tool, so a caller's policy sees the call.
func TestReplay_ToolsRegisterEveryRecordedName(t *testing.T) {
	out := recordDenialRun(t, denialPolicy, denialRecords[1].steps)
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, tl := range rc.Tools() {
		names = append(names, tl.Def().Name)
	}
	if want := []string{"secret", "ls"}; !reflect.DeepEqual(names, want) {
		t.Errorf("Tools names = %v, want %v", names, want)
	}
}

// TestReplay_ToolPolicyDeniesOnlyRecordedDenials pins RP-2: the policy
// denies a call, with the recorded reason, exactly when its tool-call ID is
// the ID of a call the record denied in the current recorded tool turn.
func TestReplay_ToolPolicyDeniesOnlyRecordedDenials(t *testing.T) {
	type probe struct {
		id, name, args string
		want           string // "" = allowed
	}
	cases := []struct {
		name   string
		policy func() ToolPolicy
		steps  []scriptStep
		turns  [][]probe // probes per recorded tool turn
	}{
		{"denials in two turns", recordingDenialPolicy, denialRecords[0].steps, [][]probe{
			{
				{"c9", "rm", `{"p":"/"}`, ""}, // a denied name and arguments under an ID the record did not deny
				{"c3", "rm", `{"p":"/tmp"}`, ""},
				{"c4", "secret", `{}`, ""}, // denied in a later turn
				{"c1", "ls", `{"p":"."}`, ""},
				{"c2", "rm", `{"p":"/"}`, "refusing to wipe the root"},
			},
			{
				{"c2", "rm", `{"p":"/"}`, ""}, // turn 1's denial is not carried over
				{"c3", "rm", `{"p":"/tmp"}`, ""},
				{"c4", "secret", `{}`, "secrets are off limits"},
			},
		}},
		// Same name and arguments, different recorded IDs: asked out of
		// model order, the ID alone decides.
		{"same call, different recorded IDs", dedupePolicy, denialRecords[2].steps, [][]probe{{
			{"c2", "ls", `{"p":"."}`, "duplicate call"},
			{"c1", "ls", `{"p":"."}`, ""},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := recordDenialRun(t, tc.policy(), tc.steps)
			rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			policy := rc.ToolPolicy()

			// Before the first response is served no turn is current.
			for _, p := range tc.turns[0] {
				c := mkCall(p.id, p.name, p.args)
				if err := policy.Authorize(context.Background(), &c); err != nil {
					t.Fatalf("%q denied before any response was served: %v", p.id, err)
				}
			}

			var prev []llmkit.ToolCall
			for turn, probes := range tc.turns {
				var msgs []llmkit.Message
				for _, call := range prev {
					msgs = append(msgs, llmkit.Message{Role: llmkit.RoleToolResult, ToolCallID: call.ID})
				}
				if _, err := rc.Complete(context.Background(), llmkit.Request{Messages: msgs}); err != nil {
					t.Fatalf("Complete before turn %d: %v", turn+1, err)
				}
				prev = tc.steps[turn].resp.ToolCalls
				for _, p := range probes {
					c := mkCall(p.id, p.name, p.args)
					err := policy.Authorize(context.Background(), &c)
					switch {
					case p.want == "" && err != nil:
						t.Errorf("turn %d: %q %s %s denied (%v), want allowed", turn+1, p.id, p.name, p.args, err)
					case p.want != "" && (err == nil || err.Error() != p.want):
						t.Errorf("turn %d: %q %s %s = %v, want denial %q", turn+1, p.id, p.name, p.args, err, p.want)
					}
				}
			}
		})
	}
}

// TestReplay_ZeroValueToolPolicyAllows pins that the policy of a zero-value
// ReplayClient, which holds no record, allows a call instead of panicking.
func TestReplay_ZeroValueToolPolicyAllows(t *testing.T) {
	c := mkCall("c1", "rm", `{"p":"/"}`)
	if err := new(ReplayClient).ToolPolicy().Authorize(context.Background(), &c); err != nil {
		t.Errorf("zero-value ToolPolicy denied: %v", err)
	}
}

// TestReplay_NoPolicyDenialRecordDiverges pins RP-3: replaying a record with
// a denial and no policy fails loudly — the denied call reaches the recorded
// tool, which serves no denials — instead of finishing with the denial
// rewritten into an ordinary tool error.
func TestReplay_NoPolicyDenialRecordDiverges(t *testing.T) {
	for _, rec := range denialRecords {
		t.Run(rec.name, func(t *testing.T) {
			out := recordDenialRun(t, rec.policy(), rec.steps)
			rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			_, runErr := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
			if !errors.Is(rc.Err(), ErrReplayDiverged) {
				t.Errorf("rc.Err() = %v, want ErrReplayDiverged", rc.Err())
			}
			if !errors.Is(runErr, ErrReplayDiverged) {
				t.Errorf("Run err = %v, want ErrReplayDiverged", runErr)
			}
		})
	}
}

// TestReplay_LiveToolRunsAndIsNotCompared pins RP-4's documented behavior: a
// tool that is not one of rc.Tools() executes live, and its result differing
// from the recorded one is no divergence — Run and rc.Err() both stay nil.
func TestReplay_LiveToolRunsAndIsNotCompared(t *testing.T) {
	recorded := 0
	rec := NewRunner(newFakeClient(
		toolResp("c1", "probe", `{}`, 1, 1),
		textResp("done", 1, 1),
	), []Tool{&countingTool{name: "probe", count: &recorded, result: func() string { return "recorded" }}}, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}

	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	live := 0
	replayed, err := NewRunner(rc, []Tool{&countingTool{name: "probe", count: &live, result: func() string { return "live and different" }}}, "sys").
		Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay Run: %v", err)
	}
	if rc.Err() != nil {
		t.Fatalf("rc.Err() = %v, want nil: live tool results are not compared", rc.Err())
	}
	if live != 1 {
		t.Errorf("live tool ran %d times, want 1", live)
	}
	sawLive := false
	for _, m := range replayed.Messages {
		if m.Role == llmkit.RoleToolResult && m.Text() == "live and different" {
			sawLive = true
		}
	}
	if !sawLive {
		t.Errorf("replayed history lacks the live result: %+v", replayed.Messages)
	}
}

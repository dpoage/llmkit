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
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
)

// rdTool runs by numbering its results in execution order, so two calls with
// the same name and arguments record different results and a replay that
// swaps them is visible.
type rdTool struct {
	name string
	n    *atomic.Int64
}

func (t rdTool) Def() llmkit.ToolDef { return llmkit.ToolDef{Name: t.name} }
func (t rdTool) Run(_ context.Context, args json.RawMessage) (string, error) {
	return fmt.Sprintf("%s#%d(%s)", t.name, t.n.Add(1), args), nil
}

func rdTools() []Tool {
	n := new(atomic.Int64)
	return []Tool{rdTool{"now", n}, rdTool{"ls", n}, rdTool{"secret", n}}
}

func rdCalls(calls ...llmkit.ToolCall) scriptStep { return toolCallsResp(calls...) }

func rdCall(id, name, args string) llmkit.ToolCall {
	return llmkit.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(args)}
}

// rdScript is one turn of same-name same-args calls, a call the policy may
// deny, a call it may rewrite, and a second turn that reuses call id "c1".
func rdScript() []scriptStep {
	return []scriptStep{
		rdCalls(
			rdCall("c1", "now", `{}`),
			rdCall("c2", "now", `{}`),
			rdCall("c3", "ls", `{"p": "."}`),
			rdCall("c4", "secret", `{"k":1}`),
		),
		rdCalls(rdCall("c1", "now", `{}`)),
		textResp("done", 1, 1),
	}
}

// rdPolicies are deterministic policies: allow, deny, rewrite, normalize.
func rdPolicies() map[string]ToolPolicy {
	set := func(args string) func(*llmkit.ToolCall) {
		return func(c *llmkit.ToolCall) { c.Arguments = json.RawMessage(args) }
	}
	by := func(f func(*llmkit.ToolCall) error) ToolPolicy {
		return ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error { return f(c) })
	}
	return map[string]ToolPolicy{
		"allow": by(func(*llmkit.ToolCall) error { return nil }),
		"deny": by(func(c *llmkit.ToolCall) error {
			if c.Name == "secret" {
				return errors.New("secret is off limits")
			}
			return nil
		}),
		"rewrite": by(func(c *llmkit.ToolCall) error {
			switch c.Name {
			case "ls":
				set(`{"p":"rewritten"}`)(c)
			case "now":
				set(`{"n":1}`)(c)
			}
			return nil
		}),
		"normalize": by(func(c *llmkit.ToolCall) error {
			var v any
			if err := json.Unmarshal(c.Arguments, &v); err != nil {
				return err
			}
			b, _ := json.Marshal(v)
			c.Arguments = b
			return nil
		}),
	}
}

func rdToolRuns(evs []llmkit.Event) []llmkit.Event {
	var out []llmkit.Event
	for _, e := range evs {
		if e.Kind == llmkit.KindToolRun {
			out = append(out, e)
		}
	}
	return out
}

// rdKey identifies a call within its turn: providers may repeat ids across
// turns, and a turn may repeat an id for calls of other names or arguments.
func rdKey(e llmkit.Event) string {
	c := e.ToolRun.Call
	return fmt.Sprintf("%d/%s/%s/%s", e.Step, c.ID, c.Name, c.Arguments)
}

func rdEqualRuns(t *testing.T, got, want []llmkit.Event) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("tool_run events = %d, want %d", len(got), len(want))
	}
	byKey := map[string]*llmkit.ToolRunEvent{}
	for _, e := range got {
		byKey[rdKey(e)] = e.ToolRun
	}
	for _, w := range want {
		g := byKey[rdKey(w)]
		if g == nil {
			t.Fatalf("no replayed event for call %s", rdKey(w))
		}
		gj, _ := json.Marshal(g)
		wj, _ := json.Marshal(w.ToolRun)
		if string(gj) != string(wj) {
			t.Errorf("call %s replayed as %s, recorded %s", rdKey(w), gj, wj)
		}
	}
}

// TestReplayDispatch_SamePolicyRoundTrip pins that a run recorded under a
// deterministic policy replays cleanly under the same policy, in both
// dispatch modes, each replayed event equal to its recorded counterpart by
// call id. In parallel mode the replay forces c2 to run before c1, and turn 2
// reuses call id c1.
func TestReplayDispatch_SamePolicyRoundTrip(t *testing.T) {
	for name, policy := range rdPolicies() {
		for _, parallel := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/parallel=%v", name, parallel), func(t *testing.T) {
				opts := []Option{WithToolPolicy(policy)}
				if parallel {
					opts = append(opts, WithParallelTools())
				}
				rec := NewRunner(newFakeClient(rdScript()...), rdTools(), "sys", opts...)
				out, err := rec.Run(context.Background(), "task")
				if err != nil {
					t.Fatalf("record: %v", err)
				}
				rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
				if err != nil {
					t.Fatal(err)
				}
				ropts := append([]Option(nil), opts...)
				if parallel {
					c2done := make(chan struct{})
					var once sync.Once
					ropts = append(ropts, WithHooks(Hooks{
						ToolStart: func(_ context.Context, ev ToolEvent) {
							if ev.Step == 1 && ev.Call.ID == "c1" {
								<-c2done
							}
						},
						ToolEnd: func(_ context.Context, ev ToolEvent) {
							if ev.Step == 1 && ev.Call.ID == "c2" {
								once.Do(func() { close(c2done) })
							}
						},
					}))
				}
				rep := NewRunner(rc, rc.Tools(), "sys", ropts...)
				rout, err := rep.Run(context.Background(), "task")
				if err != nil {
					t.Fatalf("replay Run: %v", err)
				}
				if rc.Err() != nil {
					t.Fatalf("replay rc.Err: %v", rc.Err())
				}
				rdEqualRuns(t, rdToolRuns(rout.Transcript.Record), rdToolRuns(out.Transcript.Record))
			})
		}
	}
}

// TestReplayDispatch_DivergesOnOtherArguments pins that a replay whose policy
// hands a tool arguments other than the ones the recorded Tool.Run received
// diverges naming the step: against the recorded rewrite, and against the
// model's arguments when the record holds no rewrite. Whitespace and key order
// are not a divergence.
func TestReplayDispatch_DivergesOnOtherArguments(t *testing.T) {
	setLs := func(args string) ToolPolicy {
		return ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
			if c.Name == "ls" {
				c.Arguments = json.RawMessage(args)
			}
			return nil
		})
	}
	steps := func() []scriptStep {
		return []scriptStep{rdCalls(rdCall("c1", "ls", `{"a":1,"b":2}`)), rdCalls(rdCall("c2", "ls", `{"a":1,"b":2}`)), textResp("done", 1, 1)}
	}
	record := func(p ToolPolicy) (*ReplayClient, *Outcome) {
		var opts []Option
		if p != nil {
			opts = append(opts, WithToolPolicy(p))
		}
		out, err := NewRunner(newFakeClient(steps()...), rdTools(), "sys", opts...).Run(context.Background(), "task")
		if err != nil {
			t.Fatal(err)
		}
		rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
		if err != nil {
			t.Fatal(err)
		}
		return rc, out
	}
	cases := []struct {
		name     string
		record   ToolPolicy
		replay   ToolPolicy
		diverges bool
	}{
		{"recorded rewrite, different rewrite", setLs(`{"p":"one"}`), setLs(`{"p":"two"}`), true},
		{"recorded rewrite, no rewrite", setLs(`{"p":"one"}`), nil, true},
		{"no recorded rewrite, rewrite", nil, setLs(`{"p":"one"}`), true},
		{"recorded rewrite, same rewrite spaced", setLs(`{"p":"one","q":2}`), setLs(`{ "q": 2, "p": "one" }`), false},
		{"no recorded rewrite, reordered arguments", nil, setLs(`{"b":2, "a":1}`), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rc, _ := record(c.record)
			var opts []Option
			if c.replay != nil {
				opts = append(opts, WithToolPolicy(c.replay))
			}
			_, err := NewRunner(rc, rc.Tools(), "sys", opts...).Run(context.Background(), "task")
			if !c.diverges {
				if err != nil || rc.Err() != nil {
					t.Fatalf("Run = %v, rc.Err = %v, want both nil", err, rc.Err())
				}
				return
			}
			if !errors.Is(rc.Err(), ErrReplayDiverged) || !strings.Contains(rc.Err().Error(), "at step 1") {
				t.Fatalf("rc.Err = %v, want ErrReplayDiverged naming step 1", rc.Err())
			}
			if !errors.Is(err, ErrReplayDiverged) {
				t.Errorf("Run err = %v, want ErrReplayDiverged (a later completion refuses to serve)", err)
			}
		})
	}
}

// TestReplayDispatch_IDsAreScopedToTheTurn pins that a call id reused by a
// later turn is never matched to an earlier turn's recorded call: replaying
// under a policy that denies turn 1's c1 leaves that call unserved, so turn
// 2's c1 diverges instead of taking turn 1's result.
func TestReplayDispatch_IDsAreScopedToTheTurn(t *testing.T) {
	steps := []scriptStep{rdCalls(rdCall("c1", "now", `{}`)), rdCalls(rdCall("c1", "now", `{}`)), textResp("done", 1, 1)}
	out, err := NewRunner(newFakeClient(steps...), rdTools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	var turn atomic.Int64
	denyFirst := ToolPolicyFunc(func(context.Context, *llmkit.ToolCall) error {
		if turn.Add(1) == 1 {
			return errors.New("not now")
		}
		return nil
	})
	_, runErr := NewRunner(rc, rc.Tools(), "sys", WithToolPolicy(denyFirst)).Run(context.Background(), "task")
	if !errors.Is(runErr, ErrReplayDiverged) {
		t.Fatalf("Run err = %v, want ErrReplayDiverged", runErr)
	}
	if !errors.Is(rc.Err(), ErrReplayDiverged) {
		t.Fatalf("rc.Err = %v, want ErrReplayDiverged: turn 2's c1 was served turn 1's recorded call", rc.Err())
	}
}

// TestReplayDispatch_RecordCarriesDispatchedArguments pins the recording
// side: DispatchedArguments is present when Tool.Run received valid-JSON or
// empty arguments that differ from the model's, Call keeps the model's
// bytes, and the field survives the JSONL sink into a working replay.
func TestReplayDispatch_RecordCarriesDispatchedArguments(t *testing.T) {
	const modelArgs = `{"p": "<x>"}`
	clearArgs := ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
		if c.Name == "now" {
			c.Arguments = nil
		}
		return nil
	})
	rewriteThenDeny := ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
		c.Arguments = json.RawMessage(`{"p":"lost"}`)
		return errors.New("no")
	})
	cases := []struct {
		name       string
		policy     ToolPolicy
		call       llmkit.ToolCall
		dispatched string // "" = field absent
	}{
		{"no policy", nil, rdCall("c1", "ls", modelArgs), ""},
		{"allow", rdPolicies()["allow"], rdCall("c1", "ls", modelArgs), ""},
		{"rewrite", rdPolicies()["rewrite"], rdCall("c1", "ls", modelArgs), `{"p":"rewritten"}`},
		{"normalize keeps equal", rdPolicies()["normalize"], rdCall("c1", "ls", modelArgs), ""},
		{"clear", clearArgs, rdCall("c1", "now", `{"a":1}`), `null`},
		{"denied after rewrite", rewriteThenDeny, rdCall("c1", "ls", modelArgs), ""},
		{"unknown tool", rdPolicies()["rewrite"], rdCall("c1", "nosuch", modelArgs), ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			opts := []Option{WithObserver(JSONL(dir, func(err error) { t.Errorf("sink: %v", err) }))}
			if c.policy != nil {
				opts = append(opts, WithToolPolicy(c.policy))
			}
			fc := newFakeClient(rdCalls(c.call), textResp("done", 1, 1))
			out, err := NewRunner(fc, rdTools(), "sys", opts...).Run(context.Background(), "task")
			if err != nil {
				t.Fatal(err)
			}
			runs := rdToolRuns(out.Transcript.Record)
			if len(runs) != 1 {
				t.Fatalf("tool_run events = %d, want 1", len(runs))
			}
			tr := runs[0].ToolRun
			if string(tr.Call.Arguments) != string(c.call.Arguments) {
				t.Errorf("Call.Arguments = %s, want the model's %s byte-for-byte", tr.Call.Arguments, c.call.Arguments)
			}
			if string(tr.DispatchedArguments) != c.dispatched {
				t.Errorf("DispatchedArguments = %q, want %q", tr.DispatchedArguments, c.dispatched)
			}

			// Through the JSONL file: the field survives up to argsEqual and
			// the record replays under the same policy.
			src := JSONL(dir, nil)
			evs, err := src.Events(context.Background(), out.RunID)
			if err != nil {
				t.Fatal(err)
			}
			disk := rdToolRuns(evs)[0].ToolRun
			if (c.dispatched == "") != (len(disk.DispatchedArguments) == 0) ||
				(c.dispatched != "" && !argsEqual(disk.DispatchedArguments, json.RawMessage(c.dispatched))) {
				t.Errorf("JSONL DispatchedArguments = %q, want %q up to argsEqual", disk.DispatchedArguments, c.dispatched)
			}
			rc, err := NewReplayClient(src, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			var ropts []Option
			if c.policy != nil {
				ropts = append(ropts, WithToolPolicy(c.policy))
			}
			if _, err := NewRunner(rc, rc.Tools(), "sys", ropts...).Run(context.Background(), "task"); err != nil || rc.Err() != nil {
				t.Fatalf("replay from JSONL: Run = %v, rc.Err = %v", err, rc.Err())
			}
		})
	}
}

// TestReplayDispatch_EmptyToolName pins the "" handling: rc.Tools() never
// names a tool "", a call recorded as the Runner's unknown-tool error for ""
// replays through the Runner's own path (alone in a turn before a later tool
// turn, and beside a named call in one turn), and any other outcome recorded
// for an empty name is refused at construction, naming the step.
func TestReplayDispatch_EmptyToolName(t *testing.T) {
	for name, steps := range map[string][]scriptStep{
		"alone then later turn": {
			rdCalls(rdCall("e1", "", `{}`)),
			rdCalls(rdCall("c1", "now", `{}`)),
			textResp("done", 1, 1),
		},
		"beside a named call": {
			rdCalls(rdCall("e1", "", `{"x":1}`), rdCall("c1", "now", `{}`)),
			textResp("done", 1, 1),
		},
	} {
		t.Run(name, func(t *testing.T) {
			out, err := NewRunner(newFakeClient(steps...), rdTools(), "sys").Run(context.Background(), "task")
			if err != nil {
				t.Fatal(err)
			}
			rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			for _, tool := range rc.Tools() {
				if tool.Def().Name == "" {
					t.Fatal(`rc.Tools() holds a tool named ""`)
				}
			}
			rout, err := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
			if err != nil || rc.Err() != nil {
				t.Fatalf("Run = %v, rc.Err = %v, want both nil", err, rc.Err())
			}
			rdEqualRuns(t, rdToolRuns(rout.Transcript.Record), rdToolRuns(out.Transcript.Record))
		})
	}

	hand := func(tre llmkit.ToolRunEvent) *Transcript {
		tr := NewTranscript()
		tr.RunID = "e-1"
		tr.Record = append(tr.Record,
			llmkit.Event{Kind: llmkit.KindCompletion, RunID: "e-1", Step: 1, SchemaVersion: 2,
				Completion: &llmkit.CompletionEvent{Response: llmkit.Response{ToolCalls: []llmkit.ToolCall{tre.Call}}}},
			llmkit.Event{Kind: llmkit.KindToolRun, RunID: "e-1", Step: 1, SchemaVersion: 2, ToolRun: &tre},
			llmkit.Event{Kind: llmkit.KindCompletion, RunID: "e-1", Step: 2, SchemaVersion: 2,
				Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "done"}}},
		)
		return tr
	}
	call := llmkit.ToolCall{ID: "e1", Name: "", Arguments: json.RawMessage(`{}`)}
	for name, tre := range map[string]llmkit.ToolRunEvent{
		"served by a tool named empty":  {Call: call, Result: "ok"},
		"errored by a tool named empty": {Call: call, Result: "ERROR: boom", IsError: true},
		"denied":                        {Call: call, Denied: true, DenyReason: "no"},
	} {
		t.Run("rejected/"+name, func(t *testing.T) {
			_, err := NewReplayClient(hand(tre), "e-1", llmkit.Capabilities{})
			if err == nil || !strings.Contains(err.Error(), "step 1") {
				t.Fatalf("NewReplayClient err = %v, want an error naming step 1", err)
			}
		})
	}
}

// TestReplayDispatch_BaseRecordReplays pins durable state: a JSONL record
// written before DispatchedArguments existed (agent/testdata, generated by the
// 1fe7518 Runner: a policy denial and a parallel turn of same-name calls with
// distinct arguments) replays clean under rc.Tools() and rc.ToolPolicy(), the
// replayed tool_run events equal to the recorded ones.
func TestReplayDispatch_BaseRecordReplays(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "base_record_denial_parallel.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	rec, err := LoadJSONL(f)
	if cerr := f.Close(); cerr != nil {
		t.Fatal(cerr)
	}
	if err != nil {
		t.Fatal(err)
	}
	for _, parallel := range []bool{false, true} {
		t.Run(fmt.Sprintf("parallel=%v", parallel), func(t *testing.T) {
			rc, err := NewReplayClient(rec, rec.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			opts := []Option{WithToolPolicy(rc.ToolPolicy())}
			if parallel {
				opts = append(opts, WithParallelTools())
			}
			out, err := NewRunner(rc, rc.Tools(), "sys", opts...).Run(context.Background(), "task")
			if err != nil || rc.Err() != nil {
				t.Fatalf("Run = %v, rc.Err = %v, want both nil", err, rc.Err())
			}
			want := rdToolRuns(rec.Record)
			if len(want) != 5 {
				t.Fatalf("fixture holds %d tool_run events, want 5", len(want))
			}
			rdEqualRuns(t, rdToolRuns(out.Transcript.Record), want)
		})
	}
}

// TestReplayDispatch_SharedIDInOneTurn pins that calls of one turn sharing a
// call id are told apart by name and arguments: a parallel replay that
// dispatches the last recorded call first still serves each call its own
// recorded result.
func TestReplayDispatch_SharedIDInOneTurn(t *testing.T) {
	steps := []scriptStep{
		rdCalls(
			rdCall("call_0", "ls", `{"a":1}`),
			rdCall("call_0", "now", `{"a":1}`),
			rdCall("call_0", "ls", `{"a":2}`),
		),
		textResp("done", 1, 1),
	}
	out, err := NewRunner(newFakeClient(steps...), rdTools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	for _, parallel := range []bool{false, true} {
		t.Run(fmt.Sprintf("parallel=%v", parallel), func(t *testing.T) {
			rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			var opts []Option
			if parallel {
				lastDone := make(chan struct{})
				var once sync.Once
				isLast := func(c llmkit.ToolCall) bool { return c.Name == "ls" && string(c.Arguments) == `{"a":2}` }
				opts = append(opts, WithParallelTools(), WithHooks(Hooks{
					ToolStart: func(_ context.Context, ev ToolEvent) {
						if !isLast(ev.Call) {
							<-lastDone
						}
					},
					ToolEnd: func(_ context.Context, ev ToolEvent) {
						if isLast(ev.Call) {
							once.Do(func() { close(lastDone) })
						}
					},
				}))
			}
			rout, err := NewRunner(rc, rc.Tools(), "sys", opts...).Run(context.Background(), "task")
			if err != nil || rc.Err() != nil {
				t.Fatalf("Run = %v, rc.Err = %v, want both nil", err, rc.Err())
			}
			rdEqualRuns(t, rdToolRuns(rout.Transcript.Record), rdToolRuns(out.Transcript.Record))
		})
	}
}

// flakyTool is a registered tool whose error reads exactly like the
// Runner's unknown-tool error for its name.
type flakyTool struct{}

func (flakyTool) Def() llmkit.ToolDef { return llmkit.ToolDef{Name: "flaky"} }
func (flakyTool) Run(context.Context, json.RawMessage) (string, error) {
	return "", errors.New(`unknown tool "flaky"`)
}

// TestReplayDispatch_UnregisteredToolUnderPolicy pins that a call to a tool
// the recording Runner did not register replays the way it was recorded —
// through the Runner's unknown-tool path, never seen by a policy — so a
// same-policy replay is clean under a policy that rewrites every call it
// sees and under an allowlist that denies every other name. rc.Tools()
// leaves the name out, and keeps a registered name whose calls errored with
// the unknown-tool text. A record without its start event falls back to the
// outcomes to tell the names apart.
func TestReplayDispatch_UnregisteredToolUnderPolicy(t *testing.T) {
	policies := map[string]ToolPolicy{
		"inject": ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
			c.Arguments = json.RawMessage(`{"cwd":"/w"}`)
			return nil
		}),
		"allowlist": ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
			if c.Name != "ls" {
				return errors.New("not allowed")
			}
			return nil
		}),
	}
	steps := func() []scriptStep {
		return []scriptStep{
			rdCalls(rdCall("c1", "ghost", `{"q":1}`), rdCall("c2", "ls", `{"p":"a"}`), rdCall("c3", "flaky", `{}`)),
			rdCalls(rdCall("c1", "ghost", `{"q":2}`), rdCall("c2", "ls", `{"p":"b"}`)),
			textResp("done", 1, 1),
		}
	}
	tools := func() []Tool { return append(rdTools(), flakyTool{}) }
	for name, policy := range policies {
		for _, parallel := range []bool{false, true} {
			for _, withStart := range []bool{true, false} {
				t.Run(fmt.Sprintf("%s/parallel=%v/start=%v", name, parallel, withStart), func(t *testing.T) {
					opts := []Option{WithToolPolicy(policy)}
					if parallel {
						opts = append(opts, WithParallelTools())
					}
					out, err := NewRunner(newFakeClient(steps()...), tools(), "sys", opts...).Run(context.Background(), "task")
					if err != nil {
						t.Fatal(err)
					}
					src := out.Transcript
					if !withStart {
						src = NewTranscript()
						src.RunID = out.RunID
						for _, ev := range out.Transcript.Record {
							if ev.Kind != llmkit.KindStart {
								src.Record = append(src.Record, ev)
							}
						}
					}
					rc, err := NewReplayClient(src, out.RunID, llmkit.Capabilities{})
					if err != nil {
						t.Fatal(err)
					}
					var names []string
					for _, tool := range rc.Tools() {
						names = append(names, tool.Def().Name)
					}
					if strings.Contains(strings.Join(names, ","), "ghost") {
						t.Errorf("rc.Tools() = %v, want no ghost: the recording Runner did not register it", names)
					}
					if withStart && !strings.Contains(strings.Join(names, ","), "flaky") {
						t.Errorf("rc.Tools() = %v, want flaky: the recording Runner registered it", names)
					}
					rout, err := NewRunner(rc, rc.Tools(), "sys", opts...).Run(context.Background(), "task")
					if err != nil || rc.Err() != nil {
						t.Fatalf("Run = %v, rc.Err = %v, want both nil", err, rc.Err())
					}
					want := rdToolRuns(out.Transcript.Record)
					if !withStart {
						// Without the start event, flaky's calls read as
						// unknown-tool calls and replay through that path,
						// which a policy never sees.
						var kept []llmkit.Event
						for _, ev := range want {
							if ev.ToolRun.Call.Name != "flaky" {
								kept = append(kept, ev)
							}
						}
						want = kept
						var got []llmkit.Event
						for _, ev := range rdToolRuns(rout.Transcript.Record) {
							if ev.ToolRun.Call.Name != "flaky" {
								got = append(got, ev)
							}
						}
						rdEqualRuns(t, got, want)
						return
					}
					rdEqualRuns(t, rdToolRuns(rout.Transcript.Record), want)
				})
			}
		}
	}

	t.Run("rejected/served call to a name missing from the start event", func(t *testing.T) {
		call := llmkit.ToolCall{ID: "g1", Name: "ghost", Arguments: json.RawMessage(`{}`)}
		tr := NewTranscript()
		tr.RunID = "g-1"
		tr.Record = append(tr.Record,
			llmkit.Event{Kind: llmkit.KindStart, RunID: "g-1", SchemaVersion: 2,
				Start: &llmkit.StartEvent{Task: "t", Tools: []string{"ls"}}},
			llmkit.Event{Kind: llmkit.KindCompletion, RunID: "g-1", Step: 1, SchemaVersion: 2,
				Completion: &llmkit.CompletionEvent{Response: llmkit.Response{ToolCalls: []llmkit.ToolCall{call}}}},
			llmkit.Event{Kind: llmkit.KindToolRun, RunID: "g-1", Step: 1, SchemaVersion: 2,
				ToolRun: &llmkit.ToolRunEvent{Call: call, Result: "ok"}},
			llmkit.Event{Kind: llmkit.KindCompletion, RunID: "g-1", Step: 2, SchemaVersion: 2,
				Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "done"}}},
		)
		_, err := NewReplayClient(tr, "g-1", llmkit.Capabilities{})
		if err == nil || !strings.Contains(err.Error(), "step 1") {
			t.Fatalf("NewReplayClient err = %v, want an error naming step 1", err)
		}
	})
}

// TestReplayDispatch_InvalidJSONRewrite pins that a policy rewriting a call's
// arguments to bytes that are not valid JSON leaves the record encodable:
// the JSONL sink writes every event, the transcript saves, the rewrite is
// not recorded, and a same-policy replay diverges against the model's
// arguments.
func TestReplayDispatch_InvalidJSONRewrite(t *testing.T) {
	for _, bad := range []string{`not json`, `{"p":`, ` `} {
		t.Run(fmt.Sprintf("%q", bad), func(t *testing.T) {
			policy := ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
				c.Arguments = json.RawMessage(bad)
				return nil
			})
			dir := t.TempDir()
			var sinkErrs []error
			fc := newFakeClient(rdCalls(rdCall("c1", "ls", `{"p":"."}`)), textResp("done", 1, 1))
			out, err := NewRunner(fc, rdTools(), "sys", WithToolPolicy(policy),
				WithObserver(JSONL(dir, func(err error) { sinkErrs = append(sinkErrs, err) }))).Run(context.Background(), "task")
			if err != nil {
				t.Fatal(err)
			}
			if len(sinkErrs) != 0 {
				t.Errorf("JSONL sink errors: %v", sinkErrs)
			}
			var buf bytes.Buffer
			if err := out.Transcript.SaveJSONL(&buf); err != nil {
				t.Errorf("SaveJSONL: %v", err)
			}
			disk, err := JSONL(dir, nil).Events(context.Background(), out.RunID)
			if err != nil {
				t.Fatal(err)
			}
			if len(disk) != len(out.Transcript.Record) {
				t.Errorf("JSONL holds %d events, the transcript %d", len(disk), len(out.Transcript.Record))
			}
			runs := rdToolRuns(out.Transcript.Record)
			if len(runs) != 1 || len(runs[0].ToolRun.DispatchedArguments) != 0 {
				t.Fatalf("tool_run events = %+v, want one without DispatchedArguments", runs)
			}
			rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			_, _ = NewRunner(rc, rc.Tools(), "sys", WithToolPolicy(policy)).Run(context.Background(), "task")
			if !errors.Is(rc.Err(), ErrReplayDiverged) {
				t.Fatalf("rc.Err = %v, want ErrReplayDiverged against the model's arguments", rc.Err())
			}
		})
	}
}

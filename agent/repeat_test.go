package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

type repCall struct{ name, args string }

func repTools() []Tool {
	return []Tool{echoTool{name: "A"}, echoTool{name: "B"}, echoTool{name: "C"}}
}

func repScript(calls []repCall) []scriptStep {
	var steps []scriptStep
	for i, c := range calls {
		steps = append(steps, toolResp(fmt.Sprintf("c%d", i+1), c.name, c.args, 1, 1))
	}
	return append(steps, textResp("done", 1, 1))
}

func repReason(name string, n int) string {
	return fmt.Sprintf("repeated identical call: %s called %d times in the last 12 calls without progress", name, n)
}

// repDenials runs one call per step under pol and returns each call's DenyReason, "" when allowed.
func repDenials(t *testing.T, pol ToolPolicy, calls []repCall) []string {
	t.Helper()
	out, err := NewRunner(newFakeClient(repScript(calls)...), repTools(), "sys", WithToolPolicy(pol)).
		Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	runs := toolRuns(t, out.Transcript)
	if len(runs) != len(calls) {
		t.Fatalf("recorded %d tool runs, want %d", len(runs), len(calls))
	}
	got := make([]string, len(runs))
	for i, r := range runs {
		if r.Run.Denied {
			got[i] = r.Run.DenyReason
		} else if r.Run.IsError {
			t.Fatalf("call %d errored: %s", i+1, r.Run.Result)
		}
	}
	return got
}

func TestRepetitionPolicy_IdenticalAcrossStepsDeniedOnThird(t *testing.T) {
	a := repCall{"A", `{"x":1}`}
	got := repDenials(t, RepetitionPolicy(), []repCall{a, a, a})
	want := []string{"", "", repReason("A", 3)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("denials = %q, want %q", got, want)
	}
}

func TestRepetitionPolicy_DeniedCallsStayInWindow(t *testing.T) {
	a := repCall{"A", `{}`}
	got := repDenials(t, RepetitionPolicy(), []repCall{a, a, a, a})
	want := []string{"", "", repReason("A", 3), repReason("A", 4)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("denials = %q, want %q", got, want)
	}
}

func TestRepetitionPolicy_CyclesDenyOnlyClosingCall(t *testing.T) {
	a, b, c := repCall{"A", `{}`}, repCall{"B", `{}`}, repCall{"C", `{}`}
	cases := []struct {
		name  string
		calls []repCall
		want  []string
	}{
		{"ABABA", []repCall{a, b, a, b, a}, []string{"", "", "", "", repReason("A", 3)}},
		{"ABCABCA", []repCall{a, b, c, a, b, c, a}, []string{"", "", "", "", "", "", repReason("A", 3)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := repDenials(t, RepetitionPolicy(), tc.calls); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("denials = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestRepetitionPolicy_DistinctArgsAreDistinct(t *testing.T) {
	a := repCall{"A", `{}`}
	b1, b2 := repCall{"B", `{"v":"1"}`}, repCall{"B", `{"v":"2"}`}
	big1, big2 := repCall{"A", `{"n":9007199254740993}`}, repCall{"A", `{"n":9007199254740992}`}
	cases := []struct {
		name  string
		calls []repCall
		want  bool
	}{
		{"A,B1,A,B2,A", []repCall{a, b1, a, b2, a}, false},
		{"large ints differ", []repCall{big1, big2, big1, big2}, false},
		{"large ints equal", []repCall{big1, big1, big1}, true},
		{"reordered keys and spacing", []repCall{{"A", `{"a":1,"b":2}`}, {"A", `{ "b":2, "a":1 }`}, {"A", "{\"a\":1,\n\"b\":2}"}}, true},
		{"literal text differs", []repCall{{"A", `{"n":1}`}, {"A", `{"n":1.0}`}, {"A", `{"n":1}`}}, false},
		{"unparseable compare raw", []repCall{{"A", `{bad`}, {"A", `{bad`}, {"A", `{bad`}}, true},
		{"unparseable differ", []repCall{{"A", `{bad`}, {"A", `{bad `}, {"A", `{bad`}}, false},
		{"trailing garbage is not the value", []repCall{{"A", `{"a":1}`}, {"A", `{"a":1} x`}, {"A", `{"a":1}`}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := repDenials(t, RepetitionPolicy(), tc.calls)
			denied := false
			for _, r := range got {
				denied = denied || r != ""
			}
			if denied != tc.want {
				t.Errorf("denials = %q, want denied=%t", got, tc.want)
			}
		})
	}
}

func TestRepetitionPolicy_WindowForgetsOldCalls(t *testing.T) {
	a := repCall{"A", `{}`}
	calls := []repCall{a, a}
	for i := range 12 {
		calls = append(calls, repCall{"B", fmt.Sprintf(`{"i":%d}`, i)})
	}
	calls = append(calls, a)
	got := repDenials(t, RepetitionPolicy(), calls)
	for i, r := range got {
		if r != "" {
			t.Errorf("call %d denied (%q), want all allowed once the first A,A left the window", i+1, r)
		}
	}
}

// barrierTool holds each Run until n runs have reached the same call ordinal, forcing interleaving.
type barrierTool struct {
	name string
	n    int
	mu   sync.Mutex
	hits map[int]*sync.WaitGroup
	seen map[string]int
}

func (b *barrierTool) Def() llmkit.ToolDef { return echoTool{name: b.name}.Def() }

func (b *barrierTool) Run(_ context.Context, _ json.RawMessage) ([]llmkit.Block, error) {
	b.mu.Lock()
	b.seen[b.name]++
	ord := (b.seen[b.name] - 1) / b.n
	wg := b.hits[ord]
	if wg == nil {
		wg = &sync.WaitGroup{}
		wg.Add(b.n)
		b.hits[ord] = wg
	}
	b.mu.Unlock()
	wg.Done()
	wg.Wait()
	return []llmkit.Block{llmkit.Text("ok")}, nil
}

// perConversationClient answers by how many tool results the request carries.
type perConversationClient struct{ calls int }

func (c *perConversationClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *perConversationClient) Complete(_ context.Context, req llmkit.Request) (llmkit.Response, error) {
	done := 0
	for _, m := range req.Messages {
		if m.Role == llmkit.RoleToolResult {
			done++
		}
	}
	if done >= c.calls {
		return llmkit.Response{Text: "done", StopReason: llmkit.StopEndTurn}, nil
	}
	return llmkit.Response{StopReason: llmkit.StopToolUse, ToolCalls: []llmkit.ToolCall{
		{ID: fmt.Sprintf("c%d", done+1), Name: "A", Arguments: json.RawMessage(`{}`)},
	}}, nil
}

func TestRepetitionPolicy_ConcurrentRunsKeepSeparateState(t *testing.T) {
	tool := &barrierTool{name: "A", n: 2, hits: map[int]*sync.WaitGroup{}, seen: map[string]int{}}
	r := NewRunner(&perConversationClient{calls: 2}, []Tool{tool}, "sys", WithToolPolicy(RepetitionPolicy()))
	var wg sync.WaitGroup
	outs := make([]*Outcome, 2)
	errs := make([]error, 2)
	for i := range outs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			outs[i], errs[i] = r.Run(context.Background(), "task")
		}()
	}
	wg.Wait()
	for i, out := range outs {
		if errs[i] != nil {
			t.Fatalf("run %d: %v", i, errs[i])
		}
		runs := toolRuns(t, out.Transcript)
		if len(runs) != 2 {
			t.Fatalf("run %d recorded %d tool runs, want 2", i, len(runs))
		}
		for _, tr := range runs {
			if tr.Run.Denied {
				t.Errorf("run %d: call denied (%q); runs must not share counts", i, tr.Run.DenyReason)
			}
		}
	}
}

func TestRepetitionPolicy_ContinueResetsCounts(t *testing.T) {
	a := repCall{"A", `{}`}
	r := NewRunner(newFakeClient(append(repScript([]repCall{a, a}), repScript([]repCall{a})...)...),
		repTools(), "sys", WithToolPolicy(RepetitionPolicy()))
	out1, err := r.Run(context.Background(), "one")
	if err != nil {
		t.Fatal(err)
	}
	out2, err := r.Run(context.Background(), "two", Continue(out1))
	if err != nil {
		t.Fatal(err)
	}
	for _, tr := range toolRuns(t, out2.Transcript) {
		if tr.Run.Denied {
			t.Errorf("continued run denied (%q), want counts reset under the new run id", tr.Run.DenyReason)
		}
	}
}

func TestRepetitionPolicy_ReplayReproducesDenial(t *testing.T) {
	a := repCall{"A", `{"x":1}`}
	calls := []repCall{a, a, a}
	steps := repScript(calls)
	shared := RepetitionPolicy()

	out, err := NewRunner(newFakeClient(steps...), repTools(), "sys", WithToolPolicy(shared)).
		Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record: %v", err)
	}
	if rec := toolRuns(t, out.Transcript); !rec[2].Run.Denied {
		t.Fatal("recording did not deny the third call")
	}
	for _, mode := range []string{"shared instance", "client ToolPolicy"} {
		t.Run(mode, func(t *testing.T) {
			rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			pol := shared
			if mode == "client ToolPolicy" {
				pol = rc.ToolPolicy()
			}
			replayed, err := NewRunner(rc, rc.Tools(), "sys", WithToolPolicy(pol)).Run(context.Background(), "task")
			if err != nil {
				t.Fatalf("replay: %v", err)
			}
			if rc.Err() != nil {
				t.Fatalf("replay diverged: %v", rc.Err())
			}
			if want, got := toolRuns(t, out.Transcript), toolRuns(t, replayed.Transcript); !reflect.DeepEqual(want, got) {
				t.Errorf("replayed events differ\nrecorded: %+v\nreplayed: %+v", want, got)
			}
		})
	}
}

func TestRepetitionPolicy_EvictsLeastRecentlyUsedRun(t *testing.T) {
	pol := RepetitionPolicy()
	call := func(run llmkit.RunID) error {
		args := json.RawMessage(`{}`)
		return pol.Authorize(llmkit.WithRun(context.Background(), run), &llmkit.ToolCall{Name: "A", Arguments: args})
	}
	run := func(i int) llmkit.RunID { return llmkit.RunID(fmt.Sprintf("run-%d", i)) }

	for _, id := range []llmkit.RunID{run(0), run(0), run(1), run(1)} {
		if err := call(id); err != nil {
			t.Fatalf("setup: %v", err)
		}
	}
	for i := 2; i <= 1024; i++ {
		if i == 1000 {
			_ = call(run(1)) // keeps run 1 recent
		}
		_ = call(run(i))
	}
	if err := call(run(1)); err == nil {
		t.Error("recently used run lost its state, want it retained")
	}
	if err := call(run(0)); err != nil {
		t.Errorf("oldest run kept its state after 1025 runs (%v), want it evicted", err)
	}
}

func TestRepetitionPolicy_ReasonNamesTool(t *testing.T) {
	got := repDenials(t, RepetitionPolicy(), []repCall{{"B", `{}`}, {"B", `{}`}, {"B", `{}`}})
	if !strings.Contains(got[2], "B called 3 times in the last 12 calls") {
		t.Errorf("reason = %q", got[2])
	}
}

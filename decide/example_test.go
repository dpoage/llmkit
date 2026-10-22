package decide_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/decide"
)

// exFixed is a network-free Decider that always returns resp.
type exFixed struct{ resp decide.Response }

func (d exFixed) Ask(context.Context, any, decide.Questions) (decide.Response, error) {
	return d.resp, nil
}

func (exFixed) Identity() llmkit.Identity { return llmkit.Identity{Provider: "fake", Model: "fake-1"} }

// exBeliefs is a network-free Decider that answers every Noul with the belief
// recorded for the state's "tool" and reports 20 input tokens per Ask.
type exBeliefs map[string]float64

func (b exBeliefs) Ask(_ context.Context, state any, qs decide.Questions) (decide.Response, error) {
	tool, _ := state.(map[string]any)["tool"].(string)
	resp := decide.Response{Usage: llmkit.Usage{InputTokens: 20}, Nouls: map[string]float64{}}
	for id := range qs {
		resp.Nouls[id] = b[tool]
	}
	return resp, nil
}

// exScript is a network-free llmkit.Client serving fixed responses in order.
type exScript struct {
	resps []llmkit.Response
	next  int
}

func (*exScript) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *exScript) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	r := c.resps[c.next]
	c.next++
	return r, nil
}

// exTool is a tool that reports that it ran.
type exTool string

func (t exTool) Def() llmkit.ToolDef { return llmkit.ToolDef{Name: string(t)} }

func (t exTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	fmt.Println("ran", string(t))
	return []llmkit.Block{llmkit.Text("ok")}, nil
}

// exLog is an in-memory llmkit.Observer and llmkit.Source.
type exLog struct {
	mu     sync.Mutex
	events []llmkit.Event
}

func (l *exLog) Observe(_ context.Context, ev llmkit.Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, ev)
}

func (l *exLog) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []llmkit.Event
	for _, ev := range l.events {
		if ev.RunID == run {
			out = append(out, ev)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("run %q: %w", run, llmkit.ErrUnknownRun)
	}
	return out, nil
}

func ExampleDecider() {
	var d decide.Decider = exFixed{decide.Response{
		Model:   "fake-1",
		Nouls:   map[string]float64{"urgent": 0.8},
		Choices: map[string]decide.ChoiceAnswer{"team": {Choice: "network", Confidence: 0.6}},
	}}

	resp, err := d.Ask(context.Background(), "The office VPN is down for everyone.", decide.Questions{
		"urgent": decide.Noul{Instructions: "Does this need immediate attention?"},
		"team": decide.Choice{
			Instructions: "Which team owns this?",
			Options:      map[string]any{"network": "connectivity", "billing": "payments"},
		},
	})
	if err != nil {
		fmt.Println("ask:", err)
		return
	}
	fmt.Println(resp.Model, resp.Nouls["urgent"], resp.Choices["team"].Choice)
	// Output: fake-1 0.8 network
}

func ExampleResponse_Noul() {
	resp := decide.Response{Nouls: map[string]float64{"urgent": 0.8}}

	belief, err := resp.Noul("urgent")
	fmt.Println(belief, err)

	_, err = resp.Noul("urgnet")
	fmt.Println(errors.Is(err, decide.ErrNotAsked))
	// Output:
	// 0.8 <nil>
	// true
}

func ExampleCheck() {
	qs := decide.Questions{
		"urgent": decide.Noul{Instructions: "Does this need immediate attention?"},
		"team": decide.Choice{
			Instructions: "Which team owns this?",
			Options:      map[string]any{"network": "connectivity", "billing": "payments"},
		},
	}
	resp := decide.Response{
		Nouls:   map[string]float64{"urgent": 0.8},
		Choices: map[string]decide.ChoiceAnswer{"team": {Choice: "network"}},
	}
	fmt.Println(decide.Check(qs, resp))

	resp.Choices["team"] = decide.ChoiceAnswer{Choice: "legal"}
	fmt.Println(decide.Check(qs, resp) != nil)
	// Output:
	// <nil>
	// true
}

func ExampleObserve() {
	log := &exLog{}
	d := decide.Observe(exFixed{decide.Response{
		Model: "fake-1",
		Usage: llmkit.Usage{InputTokens: 40, OutputTokens: 2},
		Nouls: map[string]float64{"urgent": 0.8},
	}}, log)

	_, err := d.Ask(context.Background(), "The office VPN is down for everyone.", decide.Questions{
		"urgent": decide.Noul{Instructions: "Does this need immediate attention?"},
	})
	if err != nil {
		fmt.Println("ask:", err)
		return
	}
	for _, ev := range log.events {
		de := ev.Decision
		fmt.Println(ev.Kind, de.Backend, de.Model, de.Usage.InputTokens, de.Answers[0].ID, *de.Answers[0].Belief)
	}
	// Output: decision fake fake-1 40 urgent 0.8
}

func ExampleReplay() {
	run := llmkit.NewRunID()
	ctx := llmkit.WithStep(llmkit.WithRun(context.Background(), run), 1)
	qs := decide.Questions{"urgent": decide.Noul{Instructions: "Does this need immediate attention?"}}

	log := &exLog{}
	live := decide.Observe(exFixed{decide.Response{Model: "fake-1", Nouls: map[string]float64{"urgent": 0.8}}}, log)
	if _, err := live.Ask(ctx, "The office VPN is down for everyone.", qs); err != nil {
		fmt.Println("record:", err)
		return
	}

	replayed, err := decide.Replay(log, run)
	if err != nil {
		fmt.Println("replay:", err)
		return
	}
	resp, err := replayed.Ask(ctx, "The office VPN is down for everyone.", qs)
	fmt.Println(resp.Nouls["urgent"], err)

	_, err = replayed.Ask(ctx, "The office VPN is down for everyone.", qs)
	fmt.Println(errors.Is(err, decide.ErrReplayDiverged))
	// Output:
	// 0.8 <nil>
	// true
}

func ExampleDecider_gatingAnAgent() {
	// Any decide.Decider works here; systemone.New builds the hosted one.
	var d decide.Decider = exBeliefs{"rm": 0.9, "read_file": 0.1}

	// Charge the decider's tokens to the pool the Runner already shares.
	pool := agent.NewBudgetPool(10_000)
	d = decide.Observe(d, llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
		if ev.Kind == llmkit.KindDecision && ev.Decision != nil && ev.Decision.Err == "" {
			pool.Add(ev.Decision.Usage.ChargeableTokens(1))
		}
	}))

	policy := agent.ToolPolicyFunc(func(ctx context.Context, call *llmkit.ToolCall) error {
		state := map[string]any{"tool": call.Name, "args": call.Arguments}
		resp, err := d.Ask(ctx, state, decide.Questions{"destructive": decide.Noul{
			Instructions: "Does this tool call delete or overwrite data outside the workspace?",
		}})
		if err != nil {
			return err // fail closed: no decision, no call
		}
		belief, err := resp.Noul("destructive")
		if err != nil {
			return err
		}
		if belief >= 0.5 {
			fmt.Println("denied", call.Name)
			return fmt.Errorf("blocked: belief %.2f", belief)
		}
		return nil
	})

	client := &exScript{resps: []llmkit.Response{
		{
			StopReason: llmkit.StopToolUse,
			Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 5},
			ToolCalls: []llmkit.ToolCall{
				{ID: "c1", Name: "rm", Arguments: json.RawMessage(`{"path":"/var/data"}`)},
				{ID: "c2", Name: "read_file", Arguments: json.RawMessage(`{"path":"notes.txt"}`)},
			},
		},
		{Text: "done", StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 15, OutputTokens: 5}},
	}}
	runner := agent.NewRunner(client, []agent.Tool{exTool("rm"), exTool("read_file")}, "You tidy files.",
		agent.WithToolPolicy(policy), agent.WithBudgetPool(pool))

	out, err := runner.Run(context.Background(), "Remove the old data and read the notes.")
	if err != nil {
		fmt.Println("run:", err)
		return
	}
	fmt.Println(out.FinalText, pool.Spent())
	// Output:
	// denied rm
	// ran read_file
	// done 75
}

// TestDocsGatingSnippetMatchesExample keeps the "Gating an agent" snippet in
// docs/decide.md equal to the body of ExampleDecider_gatingAnAgent above, up
// to its Output comment: edit one, and the other must follow.
func TestDocsGatingSnippetMatchesExample(t *testing.T) {
	src, err := os.ReadFile("example_test.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "example_test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "ExampleDecider_gatingAnAgent" {
			continue
		}
		raw := string(src[fset.Position(fn.Body.Lbrace).Offset+1 : fset.Position(fn.Body.Rbrace).Offset])
		body, _, _ = strings.Cut(raw, "\t// Output:")
	}
	if body == "" {
		t.Fatal("ExampleDecider_gatingAnAgent not found in example_test.go")
	}
	want := dedentTab(body)

	doc, err := os.ReadFile("../docs/decide.md")
	if err != nil {
		t.Fatal(err)
	}
	_, section, ok := strings.Cut(string(doc), "\n## Gating an agent\n")
	if !ok {
		t.Fatal(`docs/decide.md has no "## Gating an agent" section`)
	}
	_, block, ok := strings.Cut(section, "```go\n")
	if !ok {
		t.Fatal("the Gating an agent section has no go code block")
	}
	got, _, ok := strings.Cut(block, "\n```")
	if !ok {
		t.Fatal("the Gating an agent code block is not closed")
	}
	if got != want {
		t.Errorf("docs/decide.md snippet differs from ExampleDecider_gatingAnAgent\n--- docs\n%s\n--- example\n%s", got, want)
	}
}

func dedentTab(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i, l := range lines {
		lines[i] = strings.TrimPrefix(l, "\t")
	}
	return strings.Join(lines, "\n")
}

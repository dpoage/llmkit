//go:build live

// Live acceptance tests for the agent harness, run against the compat lane
// (the only lane with credentials today) through provider.New — the
// production construction path. Covers the Func tool loop, RunJSONAs's
// prompt-embedded schema path, multi-turn continuation, the WithMaxTokens
// continuation stitch, and raw-text preservation of a reasoning model's
// inline <think> blocks.
//
// Skips, naming the exact missing variables, unless
// LLMKIT_LIVE_COMPAT_API_KEY / _BASE_URL / _MODEL are all set. Run with
//
//	go test -tags live -count=1 ./agent/ -v
package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/internal/livetest"
)

func TestMain(m *testing.M) {
	code := m.Run()
	livetest.DefaultTally().PrintSummary()
	os.Exit(code)
}

// newLiveAgentClient builds a live client for the compat lane through
// provider.New with the run-wide tally attached.
func newLiveAgentClient(t *testing.T) (context.Context, llmkit.Client) {
	t.Helper()
	sess := livetest.Resolve(t, "compat")
	ctx, _ := livetest.Ctx(t)
	tr := livetest.NewTransport(sess.Key)
	return ctx, sess.Client(ctx, t, tr, nil)
}

// addArgs is the argument schema for the live add tool.
type addArgs struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

func TestLiveAgentFuncToolLoop(t *testing.T) {
	ctx, cl := newLiveAgentClient(t)
	now := agent.Func[struct{}]("now", "returns the current local date and time",
		func(_ context.Context, _ struct{}) (string, error) {
			return "2026-09-18T12:00:00Z", nil
		})
	add := agent.Func("add", "adds two numbers",
		func(_ context.Context, p addArgs) (string, error) {
			return fmt.Sprintf("%g", p.A+p.B), nil
		})

	// toolCalls records which tools the run actually invoked, so the retry
	// below can tell a compliant run from a noncompliant one.
	var mu sync.Mutex
	toolCalls := map[string]int{}
	hooks := agent.Hooks{
		ToolStart: func(_ context.Context, ev agent.ToolEvent) {
			mu.Lock()
			defer mu.Unlock()
			toolCalls[ev.Call.Name]++
		},
	}
	opts := []agent.Option{
		agent.WithHooks(hooks),
		agent.WithMaxTokens(2048),
	}
	system := "You are a helpful assistant. You MUST call both the `now` tool and the `add` tool before answering. Never claim a tool is missing: both `now` and `add` are provided."
	task := "You MUST call both the `now` tool and the `add` tool. Call `now` for the current time and `add` with a=41 and b=58. After both tools have been called, report the time and the sum in one sentence."

	// Live models occasionally ignore one of the two mandatory tools.
	// A noncompliant run (fewer than 2 distinct tools invoked) gets at most
	// two additional attempts; a compliant run's assertions are strict.
	var out *agent.Outcome
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		mu.Lock()
		clear(toolCalls)
		mu.Unlock()
		runner := agent.NewRunner(cl, []agent.Tool{now, add}, system, opts...)
		out, err = runner.Run(ctx, task)
		if err != nil {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		mu.Lock()
		distinct := len(toolCalls)
		called := slices.Sorted(maps.Keys(toolCalls))
		mu.Unlock()
		if distinct >= 2 {
			break
		}
		if attempt == 3 {
			t.Fatalf("after 3 attempts the model still did not call both tools (called %v)", called)
		}
		t.Logf("attempt %d: only %v called; retrying with a fresh run", attempt, called)
	}

	// Both mandatory tools must appear in the transcript...
	mu.Lock()
	defer mu.Unlock()
	if toolCalls["now"] == 0 || toolCalls["add"] == 0 {
		t.Fatalf("transcript missing a mandatory tool call: %v", toolCalls)
	}
	// ...and the final answer must carry both tool-reported facts.
	if !out.FinalTextSet {
		t.Fatalf("no final text; outcome=%+v", out)
	}
	final := strings.ToLower(llmkit.StripThinkBlocks(out.FinalText))
	if !strings.Contains(final, "99") {
		t.Errorf("final answer %q does not contain the sum 99", out.FinalText)
	}
	if !strings.Contains(final, "2026") && !strings.Contains(final, "12:00") {
		t.Errorf("final answer %q does not contain the tool-reported time", out.FinalText)
	}
}

func TestLiveAgentRunJSONAs(t *testing.T) {
	ctx, cl := newLiveAgentClient(t)
	// The compat lane reports StructuredOutput=false, so the schema reaches
	// the model via the prompt-embedded instruction; deep validation and
	// the typed unmarshal still run.
	type capitalAnswer struct {
		Capital string `json:"capital"`
		Planet  string `json:"planet"`
	}
	runner := agent.NewRunner(cl, nil,
		"You are a precise reference assistant. Answer only in JSON matching the requested schema.",
		agent.WithMaxTokens(2048))
	ans, outcome, err := agent.RunJSONAs[capitalAnswer](ctx, runner, "What is the capital of France, and on which planet is it?")
	if err != nil {
		t.Fatalf("RunJSONAs: %v (outcome: final=%q stop=%s)", err, outcome.FinalText, outcome.LastStopReason)
	}
	if !strings.Contains(strings.ToLower(ans.Capital), "paris") {
		t.Errorf("capital = %q, want Paris", ans.Capital)
	}
	if !strings.Contains(strings.ToLower(ans.Planet), "earth") {
		t.Errorf("planet = %q, want Earth", ans.Planet)
	}
}

func TestLiveAgentContinueKeepsPriorTurns(t *testing.T) {
	ctx, cl := newLiveAgentClient(t)
	runner := agent.NewRunner(cl, nil, "You are a terse assistant.", agent.WithMaxTokens(1024))

	first, err := runner.Run(ctx, "My favorite color is cerulean. Acknowledge in five words or fewer.")
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	// Continue threads first.Messages back in: the second question must be
	// answered FROM the prior conversation, not from a fresh one.
	second, err := runner.Run(ctx, "What is my favorite color? Answer with just the color name.", agent.Continue(first))
	if err != nil {
		t.Fatalf("continued run: %v", err)
	}
	if !second.FinalTextSet {
		t.Fatal("continued run produced no final text")
	}
	if !strings.Contains(strings.ToLower(llmkit.StripThinkBlocks(second.FinalText)), "cerulean") {
		t.Errorf("continued answer %q forgot the prior turn's fact", second.FinalText)
	}
}

func TestLiveAgentMaxTokensContinuationParses(t *testing.T) {
	ctx, cl := newLiveAgentClient(t)
	// A tiny per-completion cap forces the StopMaxTokens continuation path:
	// when the turn truncates, the loop takes ONE continuation completion
	// and stitches the halves — the stitched FinalText must still be a
	// parseable JSON object answering the task.
	runner := agent.NewRunner(cl, nil,
		"You are a terse assistant. Answer only with the requested JSON, nothing else.",
		agent.WithMaxTokens(200))
	task := `Reply with exactly this JSON object and no other text: {"alpha":"northern wind","beta":"copper river","gamma":"quiet forest","delta":"amber lantern","epsilon":"marble fountain"}`
	out, err := runner.Run(ctx, task)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !out.FinalTextSet {
		t.Fatal("no final text after max-tokens continuation")
	}
	var v map[string]any
	if err := json.Unmarshal([]byte(llmkit.StripThinkBlocks(out.FinalText)), &v); err != nil {
		t.Fatalf("stitched FinalText does not parse as JSON: %v\ntext: %q\n(stop=%s iterations=%d)",
			err, out.FinalText, out.LastStopReason, out.Iterations)
	}
}

func TestLiveAgentPreservesInlineThink(t *testing.T) {
	ctx, cl := newLiveAgentClient(t)
	runner := agent.NewRunner(cl, nil, "You are terse.", agent.WithMaxTokens(512))
	out, err := runner.Run(ctx, "What is 17 plus 25? Answer with just the number.")
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	// MiniMax-M3 emits its reasoning as an inline <think>...</think> span in
	// the message content. The harness keeps raw text: FinalText preserves
	// the span verbatim...
	if !strings.Contains(out.FinalText, "<think>") {
		t.Errorf("raw FinalText lost the inline think span (got %.120q)", out.FinalText)
	}
	// ...while consumers who want the answer alone strip it first.
	if !strings.Contains(llmkit.StripThinkBlocks(out.FinalText), "42") {
		t.Errorf("answer 42 missing from %q", llmkit.StripThinkBlocks(out.FinalText))
	}
	// RunJSONAs still parses despite the think noise (its parse path strips
	// think blocks before decoding).
	type sumAnswer struct {
		Sum int `json:"sum"`
	}
	jr := agent.NewRunner(cl, nil, "Answer only in JSON matching the schema.", agent.WithMaxTokens(512))
	ans, _, err := agent.RunJSONAs[sumAnswer](ctx, jr, "What is 17 plus 25?")
	if err != nil {
		t.Fatalf("RunJSONAs: %v", err)
	}
	if ans.Sum != 42 {
		t.Errorf("sum = %d, want 42", ans.Sum)
	}
}

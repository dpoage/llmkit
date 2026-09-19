//go:build live

// Live acceptance tests for the agent harness, run against the compat lane
// through provider.New. Covers the Func tool loop, RunJSONAs's prompt-embedded
// schema path, multi-turn continuation, WithMaxTokens continuation, the
// RequestPolicy seam, Attach, ToolPolicy, and inline <think> preservation.
//
// Live-model flakes are handled, not hidden: where an assertion depends on
// the model producing VISIBLE text or calling mandatory tools, the test
// retries a bounded number of times (two extra attempts) with DISTINCT task
// phrasings — re-issuing a byte-identical prompt just re-elicits the same
// correlated noncompliance. A run that is visibly noncompliant with the
// tested premise (e.g. no <think> span at all) fails immediately.
//
// Skips, naming the exact missing variables, unless
// LLMKIT_LIVE_COMPAT_API_KEY / _BASE_URL / _MODEL are all set. Run with
//
//	go test -tags live -count=1 ./agent/ -v
package agent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
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
func newLiveAgentClient(t *testing.T) (context.Context, llmkit.Client, *livetest.Session) {
	t.Helper()
	sess := livetest.Resolve(t, "compat")
	ctx, _ := livetest.Ctx(t)
	tr := livetest.NewTransport(sess.Key)
	return ctx, sess.Client(ctx, t, tr, nil), sess
}

// addArgs is the argument schema for the live add tool.
type addArgs struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

func TestLiveAgentFuncToolLoop(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)
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

	system := "You are a helpful assistant. You MUST call both the `now` tool and the `add` tool before answering. Never claim a tool is missing: both `now` and `add` are provided."
	// Three DISTINCT phrasings: re-issuing a byte-identical prompt within
	// seconds re-elicits the same correlated noncompliance.
	tasks := []string{
		"You MUST call both the `now` tool and the `add` tool. Call `now` for the current time and `add` with a=41 and b=58. After both tools have been called, report the time and the sum in one sentence.",
		"Use the `add` tool with a=41, b=58, and use the `now` tool for the current time. You must invoke BOTH tools before writing your one-sentence answer containing the time and the sum.",
		"Call `add` (a=41, b=58) and `now`. Both tools are mandatory. Then answer: what time did `now` return, and what is the sum from `add`?",
	}

	// A noncompliant run (fewer than 2 distinct tools invoked) gets at most
	// two additional attempts, each with a different phrasing and — from the
	// second attempt on — the tool order swapped; a compliant run's
	// assertions are strict.
	var out *agent.Outcome
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		mu.Lock()
		clear(toolCalls)
		mu.Unlock()
		tools := []agent.Tool{now, add}
		if attempt >= 2 {
			tools = []agent.Tool{add, now}
		}
		runner := agent.NewRunner(cl, tools, system, agent.WithHooks(hooks), agent.WithMaxTokens(2048))
		out, err = runner.Run(ctx, tasks[attempt-1])
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
			t.Fatalf("after 3 differently-phrased attempts the model still did not call both tools (called %v)", called)
		}
		t.Logf("attempt %d: only %v called; retrying with different phrasing and tool order", attempt, called)
	}

	// Both mandatory tools must appear in the transcript...
	mu.Lock()
	defer mu.Unlock()
	if toolCalls["now"] == 0 || toolCalls["add"] == 0 {
		t.Fatalf("transcript missing a mandatory tool call: %v", toolCalls)
	}
	// ...and the final answer must carry both tool-reported facts.
	if out.FinalText == "" {
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
	ctx, cl, _ := newLiveAgentClient(t)
	// The compat lane reports StructuredOutput=false, so the schema reaches
	// the model via the prompt-embedded instruction; deep validation and
	// the typed unmarshal still run.
	type capitalAnswer struct {
		Capital string `json:"capital"`
		Planet  string `json:"planet"`
	}
	tasks := []string{
		"What is the capital of France, and on which planet is it?",
		"Name the capital city of France and the planet it is on.",
		"Fill the schema: France's capital city, and the planet that city is on.",
	}
	system := "You are a precise reference assistant. Answer only in JSON matching the requested schema."

	// A think-only turn parses as empty and burns a repair round-trip for
	// nothing; that known failure mode gets the bounded retry, while a
	// visibly wrong answer still fails.
	var ans capitalAnswer
	var outcome *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		runner := agent.NewRunner(cl, nil, system, agent.WithMaxTokens(2048))
		a, o, err := agent.RunJSONAs[capitalAnswer](ctx, runner, tasks[attempt-1])
		if err == nil {
			ans, outcome = a, o
			break
		}
		if attempt == 3 {
			t.Fatalf("RunJSONAs: %v after 3 attempts (last outcome: final=%q stop=%s)",
				err, o.FinalText, o.LastStopReason)
		}
		if !errors.Is(err, agent.ErrUnparseableOutput) {
			t.Fatalf("RunJSONAs: %v (outcome: final=%q stop=%s)", err, o.FinalText, o.LastStopReason)
		}
		outcome = o
		t.Logf("attempt %d: unparseable output (%.80q); retrying with different phrasing",
			attempt, llmkit.StripThinkBlocks(o.FinalText))
	}
	if !strings.Contains(strings.ToLower(ans.Capital), "paris") {
		t.Errorf("capital = %q, want Paris", ans.Capital)
	}
	if !strings.Contains(strings.ToLower(ans.Planet), "earth") {
		t.Errorf("planet = %q, want Earth", ans.Planet)
	}
	_ = outcome
}

func TestLiveAgentContinueKeepsPriorTurns(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)
	runner := agent.NewRunner(cl, nil, "You are a terse assistant.", agent.WithMaxTokens(1024))

	first, err := runner.Run(ctx, "My favorite color is cerulean. Acknowledge in five words or fewer.")
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	// Continue threads first.Messages back in: the second question must be
	// answered FROM the prior conversation, not from a fresh one. A
	// think-only second turn produces no visible answer — that known flake
	// mode gets the bounded retry with varied phrasing; a visible wrong
	// answer still fails.
	tasks := []string{
		"What is my favorite color? Answer with just the color name.",
		"From what I told you earlier: which color is my favorite? Reply with only the color.",
		"Repeat the favorite color I mentioned before. One word only.",
	}
	var second *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		s, err := runner.Run(ctx, tasks[attempt-1], agent.Continue(first))
		if err != nil {
			t.Fatalf("continued run (attempt %d): %v", attempt, err)
		}
		if strings.TrimSpace(llmkit.StripThinkBlocks(s.FinalText)) != "" {
			second = s
			break
		}
		if attempt == 3 {
			t.Fatalf("continued run produced no visible answer after 3 attempts (final=%q)", s.FinalText)
		}
		second = s
		t.Logf("attempt %d: think-only continued turn (%.60q); retrying with different phrasing",
			attempt, llmkit.StripThinkBlocks(s.FinalText))
	}
	if !strings.Contains(strings.ToLower(llmkit.StripThinkBlocks(second.FinalText)), "cerulean") {
		t.Errorf("continued answer %q forgot the prior turn's fact", second.FinalText)
	}
}

func TestLiveAgentMaxTokensContinuationParses(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)
	// A tiny per-completion cap forces the StopMaxTokens continuation path:
	// when the turn truncates, the loop takes ONE continuation completion
	// and stitches the halves — the stitched FinalText must still be a
	// parseable JSON object answering the task. A think-only run leaves
	// nothing parseable; that known flake mode gets the bounded retry with
	// varied phrasing.
	phrasings := []string{
		`Reply with exactly this JSON object and no other text: {"alpha":"northern wind","beta":"copper river","gamma":"quiet forest","delta":"amber lantern","epsilon":"marble fountain"}`,
		`Output only this JSON, verbatim structure: {"alpha":"northern wind","beta":"copper river","gamma":"quiet forest","delta":"amber lantern","epsilon":"marble fountain"}`,
		`Respond with a single JSON object having exactly these keys and string values: alpha=northern wind, beta=copper river, gamma=quiet forest, delta=amber lantern, epsilon=marble fountain.`,
	}
	var parsed map[string]any
	for attempt := 1; attempt <= 3; attempt++ {
		runner := agent.NewRunner(cl, nil,
			"You are a terse assistant. Answer only with the requested JSON, nothing else.",
			agent.WithMaxTokens(200))
		o, err := runner.Run(ctx, phrasings[attempt-1])
		if err != nil {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		var v map[string]any
		uerr := json.Unmarshal([]byte(llmkit.StripThinkBlocks(o.FinalText)), &v)
		if uerr == nil {
			parsed = v
			break
		}
		if attempt == 3 {
			t.Fatalf("stitched FinalText does not parse as JSON after 3 attempts: %v\ntext: %q\n(stop=%s iterations=%d)",
				uerr, o.FinalText, o.LastStopReason, o.Iterations)
		}
		t.Logf("attempt %d: stitched text unparseable (%.60q); retrying with different phrasing",
			attempt, llmkit.StripThinkBlocks(o.FinalText))
	}
	if _, ok := parsed["alpha"]; !ok {
		t.Errorf("parsed JSON %v lost the requested alpha key", parsed)
	}
}

func TestLiveAgentPreservesInlineThink(t *testing.T) {
	ctx, cl, sess := newLiveAgentClient(t)
	phrasings := []string{
		"What is 17 plus 25? Answer with just the number.",
		"Compute 17 + 25. Reply with the number only.",
		"17+25=? Reply with just the numeral.",
	}
	system := "You are terse."

	// MiniMax-M3 intermittently emits a CLOSED think span with no visible
	// answer — that known flake mode gets the bounded retry with varied
	// phrasing. A run with NO <think> span at all falsifies the tested
	// premise ("M3 emits inline think blocks") and fails immediately.
	var out *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		runner := agent.NewRunner(cl, nil, system, agent.WithMaxTokens(512))
		o, err := runner.Run(ctx, phrasings[attempt-1])
		if err != nil {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		out = o
		visible := strings.TrimSpace(llmkit.StripThinkBlocks(o.FinalText))
		if visible == "" {
			if attempt == 3 {
				sess.Logf(t, "final text after 3 attempts: visible=%q raw=%.200q", visible, o.FinalText)
				t.Fatal("no visible answer after 3 attempts (think-only turns throughout)")
			}
			t.Logf("attempt %d: think-only turn (%.60q); retrying with different phrasing",
				attempt, o.FinalText)
			continue
		}
		break
	}

	// The harness keeps raw text: FinalText preserves the <think> span
	// verbatim. Its absence is a premise failure, not a flake.
	if !strings.Contains(out.FinalText, "<think>") {
		sess.Logf(t, "premise failure: no think span; visible=%q raw=%.200q",
			strings.TrimSpace(llmkit.StripThinkBlocks(out.FinalText)), out.FinalText)
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

// TestLiveAgentRequestPolicyShapesWire pins that the RequestPolicy seam fires exactly once per completion against the live lane and the run completes untruncated.
func TestLiveAgentRequestPolicyShapesWire(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)
	var preps, completions atomic.Int64
	policy := agent.RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
		preps.Add(1)
		temp := 0.5
		req.Temperature = &temp
		if cl.Capabilities().Thinking {
			req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
		}
		return nil
	})
	runner := agent.NewRunner(cl, nil, "You are a terse assistant.",
		agent.WithMaxTokens(2048),
		agent.WithHooks(agent.Hooks{
			BeforeCompletion: func(context.Context, int, *llmkit.Request) { completions.Add(1) },
		}),
		agent.WithRequestPolicy(policy))
	out, err := runner.Run(ctx, "Reply with the single word: ready.")
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if out.TruncationReason != "" {
		t.Fatalf("run truncated (%v): the policy case did not complete", out.TruncationReason)
	}
	if got := preps.Load(); got != completions.Load() {
		t.Errorf("PrepareRequest calls = %d, completions = %d, want equal", got, completions.Load())
	}
	if preps.Load() == 0 {
		t.Error("PrepareRequest never fired")
	}
}

// TestLiveAgentAttachImageOnTaskTurn pins that a generated PNG rides on the task turn and the transcript's first request event carries that block byte-for-byte on the seed user turn.
func TestLiveAgentAttachImageOnTaskTurn(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)
	if !cl.Capabilities().Images {
		t.Skip("compat lane client reports Capabilities.Images=false: the Attach live case needs a vision-capable model to put an image block on the wire")
	}

	// 1x1 red PNG generated in-process.
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 0xff, A: 0xff})
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, img); err != nil {
		t.Fatalf("encode tiny PNG: %v", err)
	}

	runner := agent.NewRunner(cl, nil, "You are a terse assistant.", agent.WithMaxTokens(256))
	out, err := runner.Run(ctx, "What color is the attached square? Answer in three words or fewer.",
		agent.Attach(llmkit.Image("image/png", pngBuf.Bytes())))
	if err != nil {
		t.Fatalf("run with attachment: %v", err)
	}

	var seed *llmkit.Message
	for i := range out.Transcript.Events {
		ev := &out.Transcript.Events[i]
		if ev.Kind == agent.EventRequest && len(ev.Messages) > 0 {
			seed = &ev.Messages[0]
			break
		}
	}
	if seed == nil {
		t.Fatal("transcript has no request event with messages")
	}
	found := false
	for _, b := range seed.Content {
		if b.Kind == llmkit.BlockImage && b.MediaType == "image/png" && bytes.Equal(b.Data, pngBuf.Bytes()) {
			found = true
			break
		}
	}
	if !found {
		var kinds []string
		for _, b := range seed.Content {
			kinds = append(kinds, string(b.Kind))
		}
		t.Errorf("first request's user message carries no image block with the generated PNG; content kinds: %v", kinds)
	}
}

// TestLiveAgentToolPolicyDeny pins that denying the only registered tool against the live lane never runs Tool.Run and still reaches a final answer; the deny tool_result must reach both the follow-up wire request and the transcript.
func TestLiveAgentToolPolicyDeny(t *testing.T) {
	ctx, cl, sess := newLiveAgentClient(t)
	add := agent.Func("add", "adds two numbers",
		func(_ context.Context, p addArgs) (string, error) {
			return fmt.Sprintf("%g", p.A+p.B), nil
		})

	// runs counts Tool.Run executions (must stay 0); authorizations counts policy consultations — the premise probe retries with new phrasing.
	var mu sync.Mutex
	runs, authorizations := 0, 0
	const denyReason = "add is denied by the live policy"
	policy := agent.ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		mu.Lock()
		defer mu.Unlock()
		authorizations++
		if call.Name == "add" {
			return errors.New(denyReason)
		}
		return nil
	})
	hooks := agent.Hooks{
		ToolStart: func(_ context.Context, ev agent.ToolEvent) {
			mu.Lock()
			defer mu.Unlock()
			runs++
		},
	}

	system := "You are a helpful assistant. Call the `add` tool to compute sums when asked."
	// Three distinct phrasings — byte-identical prompts re-elicits the same correlated noncompliance.
	tasks := []string{
		"You MUST call the `add` tool with a=41 and b=58. After the tool has been called, report the sum in one sentence.",
		"Use the `add` tool to compute 41 plus 58, then answer with the sum in one sentence. You must invoke `add` before answering.",
		"Call `add` (a=41, b=58). It is mandatory. Then state the sum in one sentence.",
	}

	var out *agent.Outcome
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		mu.Lock()
		runs, authorizations = 0, 0
		mu.Unlock()
		runner := agent.NewRunner(cl, []agent.Tool{add}, system,
			agent.WithHooks(hooks), agent.WithToolPolicy(policy), agent.WithMaxTokens(2048))
		out, err = runner.Run(ctx, tasks[attempt-1])
		if err != nil {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		mu.Lock()
		auths, executed := authorizations, runs
		mu.Unlock()
		if auths > 0 {
			// Compliant premise: the model requested the sole tool and the policy denied it.
			if executed != 0 {
				t.Fatalf("Tool.Run executed %d time(s) despite a deny-all policy", executed)
			}
			break
		}
		if attempt == 3 {
			sess.Logf(t, "premise failure: the model never requested `add` in any attempt")
			t.Fatal("after 3 differently-phrased attempts the model still never requested the `add` tool")
		}
		t.Logf("attempt %d: the model never requested `add`; retrying with different phrasing", attempt)
	}

	// Deny result reached the model: Outcome's tool-result message (and the follow-up wire request) carry it.
	want := "ERROR: tool add denied: " + denyReason
	sawMsg := false
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleToolResult && m.IsError && m.Text() == want {
			sawMsg = true
		}
	}
	if !sawMsg {
		t.Fatalf("Outcome.Messages lost the deny tool result %q", want)
	}
	// The transcript records the same tool_result event.
	sawEvent := false
	for _, ev := range out.Transcript.Events {
		if ev.Kind == agent.EventToolResult && ev.IsError && ev.Result == want {
			sawEvent = true
		}
	}
	if !sawEvent {
		t.Fatal("transcript lost the deny tool_result event")
	}

	// The run continued past the denial and reached a final answer.
	if out.FinalText == "" {
		t.Fatalf("no final text after the denial; outcome=%+v", out)
	}
}

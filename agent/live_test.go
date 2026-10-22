//go:build live

// Live acceptance tests for the agent harness, run against the compat lane
// through provider.New. Covers the Func tool loop, RunJSONAs's prompt-embedded
// schema path, multi-turn continuation, WithMaxTokens continuation, the
// RequestPolicy seam, Attach, ToolPolicy, RepetitionPolicy, tool-output
// spilling, Steering, inline <think> preservation, and proactive summary
// compaction against a small forced ContextWindow.
//
// Live-model flakes are handled, not hidden: where an assertion depends on
// the model producing VISIBLE text or calling mandatory tools, the test
// retries a bounded number of times (two extra attempts) with DISTINCT task
// phrasings — re-issuing a byte-identical prompt just re-elicits the same
// correlated noncompliance. A run that is visibly noncompliant with the
// tested premise (e.g. no visible answer at all) fails immediately; a
// premise the vendor does not guarantee per run (an inline <think> span)
// skips with a premise message when the retries come up empty.
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
	"github.com/dpoage/llmkit/provider"
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

// isModelFlake reports whether err is the *IncompleteError of a run the
// model left without an answer: TruncNoAnswer (silent through every nudge) or
// TruncOutputCap (cut off at the cap, continuation included). Both are the
// known think-only / capped live-model flakes, so the retry loops treat them
// like an empty visible answer; any other error still fails the test.
func isModelFlake(err error) bool {
	var ie *agent.IncompleteError
	return errors.As(err, &ie) && (ie.Reason == agent.TruncNoAnswer || ie.Reason == agent.TruncOutputCap)
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
		runner := agent.NewRunner(cl, tools, system, agent.WithHooks(hooks), agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
		out, err = runner.Run(ctx, tasks[attempt-1])
		if err != nil && !isModelFlake(err) {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		mu.Lock()
		distinct := len(toolCalls)
		called := slices.Sorted(maps.Keys(toolCalls))
		mu.Unlock()
		if err == nil && distinct >= 2 {
			break
		}
		if attempt == 3 {
			t.Fatalf("after 3 differently-phrased attempts no run completed calling both tools (called %v, err=%v)", called, err)
		}
		t.Logf("attempt %d: only %v called (err=%v); retrying with different phrasing and tool order", attempt, called, err)
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
		runner := agent.NewRunner(cl, nil, system, agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
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
	runner := agent.NewRunner(cl, nil, "You are a terse assistant.", agent.WithMaxTokens(1024), agent.WithObserver(livetest.DefaultTally()))

	// The first run can hit the same think-only / capped flake; it gets
	// the bounded retry with varied phrasing, and any other error still
	// fails.
	firstTasks := []string{
		"My favorite color is cerulean. Acknowledge in five words or fewer.",
		"My favorite color is cerulean. Confirm it back in five words or fewer.",
		"I am telling you my favorite color: cerulean. Acknowledge in five words or fewer.",
	}
	var first *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		f, ferr := runner.Run(ctx, firstTasks[attempt-1])
		if ferr != nil && !isModelFlake(ferr) {
			t.Fatalf("first run (attempt %d): %v", attempt, ferr)
		}
		if ferr == nil && strings.TrimSpace(llmkit.StripThinkBlocks(f.FinalText)) != "" {
			first = f
			break
		}
		if attempt == 3 {
			t.Fatalf("first run never completed with a visible answer in 3 attempts (final=%q, err=%v)", f.FinalText, ferr)
		}
		t.Logf("attempt %d: think-only or capped first run (%.60q, err=%v); retrying with different phrasing", attempt, f.FinalText, ferr)
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
		if err != nil && !isModelFlake(err) {
			t.Fatalf("continued run (attempt %d): %v", attempt, err)
		}
		if err == nil && strings.TrimSpace(llmkit.StripThinkBlocks(s.FinalText)) != "" {
			second = s
			break
		}
		if attempt == 3 {
			t.Fatalf("continued run produced no visible answer after 3 attempts (final=%q, err=%v)", s.FinalText, err)
		}
		second = s
		t.Logf("attempt %d: think-only or capped continued turn (%.60q, err=%v); retrying with different phrasing",
			attempt, llmkit.StripThinkBlocks(s.FinalText), err)
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
			agent.WithMaxTokens(200), agent.WithObserver(livetest.DefaultTally()))
		o, err := runner.Run(ctx, phrasings[attempt-1])
		if err != nil && !isModelFlake(err) {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		var v map[string]any
		// A flake run (still capped after the continuation, or silent) has
		// no answer this loop accepts — cut text included: its error
		// stands in for the parse error.
		uerr := err
		if err == nil {
			uerr = json.Unmarshal([]byte(llmkit.StripThinkBlocks(o.FinalText)), &v)
		}
		if uerr == nil {
			parsed = v
			break
		}
		if attempt == 3 {
			t.Fatalf("no parseable answer after 3 attempts (capped or unparseable): %v\ntext: %q\n(stop=%s iterations=%d)",
				uerr, o.FinalText, o.LastStopReason, o.Iterations)
		}
		t.Logf("attempt %d: no parseable answer yet — capped run or unparseable text (%.60q); retrying with different phrasing",
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

	// MiniMax-M3 is nondeterministic in both directions: it intermittently
	// emits a CLOSED think span with no visible answer, and intermittently
	// answers with no think span at all. A run compliant with the tested
	// premise carries BOTH: a visible answer and an inline <think> span.
	// Noncompliant runs get the bounded retry with varied phrasing; when
	// no attempt carries both, the preservation contract has nothing to
	// preserve and the test skips with that premise message instead of
	// failing.
	var out *agent.Outcome        // compliant run: visible answer + think span
	var answerOnly *agent.Outcome // visible answer, no think span
	for attempt := 1; attempt <= 3; attempt++ {
		runner := agent.NewRunner(cl, nil, system, agent.WithMaxTokens(512), agent.WithObserver(livetest.DefaultTally()))
		o, err := runner.Run(ctx, phrasings[attempt-1])
		if err != nil && !isModelFlake(err) {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		visible := strings.TrimSpace(llmkit.StripThinkBlocks(o.FinalText))
		if err != nil || visible == "" {
			// Known flake mode: CLOSED think span, no visible answer — or a
			// turn the harness reports as incomplete (silent or capped).
			if attempt == 3 {
				if answerOnly == nil {
					sess.Logf(t, "final text after 3 attempts: visible=%q raw=%.200q", visible, o.FinalText)
					t.Fatalf("no visible answer after 3 attempts (think-only turns throughout; last err=%v)", err)
				}
				break
			}
			t.Logf("attempt %d: think-only or capped turn (%.60q, err=%v); retrying with different phrasing",
				attempt, o.FinalText, err)
			continue
		}
		if !strings.Contains(o.FinalText, "<think>") {
			// Second flake mode: the model answered without any span. Keep
			// the answer and try another phrasing for a think-bearing one.
			answerOnly = o
			t.Logf("attempt %d: visible answer without a think span (%.60q); retrying with different phrasing",
				attempt, visible)
			continue
		}
		out = o
		break
	}

	// The answer contract holds whether or not the model wrapped the answer
	// in a think span, so assert it on whichever run produced a visible
	// answer.
	answered := out
	if answered == nil {
		answered = answerOnly
	}
	if !strings.Contains(llmkit.StripThinkBlocks(answered.FinalText), "42") {
		t.Errorf("answer 42 missing from %q", llmkit.StripThinkBlocks(answered.FinalText))
	}

	// Premise-gated preservation: the harness keeps raw text, so a think
	// span the model emitted must survive verbatim in FinalText. When no
	// attempt carried a span the premise is unavailable — skip. The
	// contract itself is pinned hermetically by
	// TestRun_FinalTextPreservesInlineThink; this live case re-checks it
	// end to end whenever the vendor emits a span.
	if out == nil {
		sess.Logf(t, "premise unavailable after 3 attempts: no run had both a visible answer and a think span; raw=%.200q", answered.FinalText)
		t.Skipf("MiniMax-M3 never produced a visible answer and an inline <think> span in the same run (last visible answer %q); nothing to preserve",
			llmkit.StripThinkBlocks(answered.FinalText))
	}

	// The harness keeps raw text: the compliant run's FinalText preserves
	// the <think> span verbatim.
	if !strings.Contains(out.FinalText, "<think>") {
		t.Errorf("raw FinalText lost the inline think span (got %.120q)", out.FinalText)
	}
	// RunJSONAs still parses despite the think noise (its parse path strips
	// think blocks before decoding).
	type sumAnswer struct {
		Sum int `json:"sum"`
	}
	jr := agent.NewRunner(cl, nil, "Answer only in JSON matching the schema.", agent.WithMaxTokens(512), agent.WithObserver(livetest.DefaultTally()))
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
		agent.WithRequestPolicy(policy), agent.WithObserver(livetest.DefaultTally()))
	// The run can hit the think-only / capped flake; it gets the bounded
	// retry with varied phrasing, and any other error still fails.
	readyTasks := []string{
		"Reply with the single word: ready.",
		"Answer with exactly one word: ready.",
		"Give the one-word reply: ready.",
	}
	var out *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		o, rerr := runner.Run(ctx, readyTasks[attempt-1])
		if rerr != nil && !isModelFlake(rerr) {
			t.Fatalf("run (attempt %d): %v", attempt, rerr)
		}
		if rerr == nil {
			out = o
			break
		}
		if attempt == 3 {
			t.Fatalf("run never completed in 3 attempts (last outcome: final=%q stop=%s err=%v)", o.FinalText, o.LastStopReason, rerr)
		}
		t.Logf("attempt %d: think-only or capped run (%.60q, err=%v); retrying with different phrasing", attempt, o.FinalText, rerr)
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

	runner := agent.NewRunner(cl, nil, "You are a terse assistant.", agent.WithMaxTokens(256), agent.WithObserver(livetest.DefaultTally()))
	// The run can hit the think-only / capped flake; it gets the bounded
	// retry with varied phrasing, and any other error still fails.
	attachTasks := []string{
		"What color is the attached square? Answer in three words or fewer.",
		"Look at the attached image: what color is the square? Three words max.",
		"The attached picture shows a square; name its color in three words or fewer.",
	}
	var out *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		o, aerr := runner.Run(ctx, attachTasks[attempt-1], agent.Attach(llmkit.Image("image/png", pngBuf.Bytes())))
		if aerr != nil && !isModelFlake(aerr) {
			t.Fatalf("run with attachment (attempt %d): %v", attempt, aerr)
		}
		if aerr == nil {
			out = o
			break
		}
		if attempt == 3 {
			t.Fatalf("run with attachment never completed in 3 attempts (last outcome: final=%q stop=%s err=%v)", o.FinalText, o.LastStopReason, aerr)
		}
		t.Logf("attempt %d: think-only or capped run with attachment (%.60q, err=%v); retrying with different phrasing", attempt, o.FinalText, aerr)
	}

	var seed *llmkit.Message
	for i := range out.Transcript.Record {
		ev := &out.Transcript.Record[i]
		if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && len(ev.Completion.Request.Messages) > 0 {
			seed = &ev.Completion.Request.Messages[0]
			break
		}
	}
	if seed == nil {
		t.Fatal("transcript has no completion event with request messages")
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
			agent.WithHooks(hooks), agent.WithToolPolicy(policy), agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
		out, err = runner.Run(ctx, tasks[attempt-1])
		if err != nil {
			if !isModelFlake(err) {
				t.Fatalf("run (attempt %d): %v", attempt, err)
			}
			// Known flake mode: the run ended unanswered (silent or capped);
			// the tested premise did not run, so retry like a run that never
			// requested the tool.
			if attempt == 3 {
				t.Fatalf("after 3 differently-phrased attempts the run still ended without an answer: %v", err)
			}
			t.Logf("attempt %d: run ended without an answer (%v); retrying with different phrasing", attempt, err)
			continue
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
	// The transcript records the same denial ToolRun event.
	sawEvent := false
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && ev.ToolRun.Denied && ev.ToolRun.Call.Name == "add" {
			sawEvent = true
		}
	}
	if !sawEvent {
		t.Fatal("transcript lost the denial tool_run event")
	}

	// The run continued past the denial and reached a final answer.
	if out.FinalText == "" {
		t.Fatalf("no final text after the denial; outcome=%+v", out)
	}
}

// TestLiveAgentDeltaHook pins the Hooks.Delta seam end to end against the
// live lane: the hook fires at least once, every delta reports a positive
// step, and the streamed text reconstructs the run's final answer. On a
// single-completion run the text deltas must equal FinalText byte for
// byte. A run may span several completions — a think-only turn makes the
// Runner nudge and loop again, and a max-tokens stop appends ONE
// continuation completion under its own hook step, stitched onto the
// final answer — so with multiple steps the contract relaxes to FinalText
// ending with the last step's text deltas: the stitch trims only the
// overlap the continuation re-emitted. A final turn with no visible text
// after three differently-phrased attempts is a premise failure. With
// today's adapters the [llmkit.Stream] fallback synthesizes one delta per
// content block; as adapter streaming lands the count rises — only >= 1
// is pinned here.
func TestLiveAgentDeltaHook(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)

	var mu sync.Mutex
	var deltas int
	var lastTextStep int
	textByStep := map[int]*strings.Builder{}
	hooks := agent.Hooks{
		Delta: func(_ context.Context, step int, d llmkit.Delta) {
			mu.Lock()
			defer mu.Unlock()
			deltas++
			if step < 1 {
				t.Errorf("delta step = %d, want >= 1", step)
			}
			if d.Kind != llmkit.DeltaText {
				return
			}
			b := textByStep[step]
			if b == nil {
				b = new(strings.Builder)
				textByStep[step] = b
			}
			b.WriteString(d.Text)
			lastTextStep = step
		},
	}

	// Three DISTINCT phrasings: re-issuing a byte-identical prompt within
	// seconds re-elicits the same correlated noncompliance.
	phrasings := []string{
		"Reply with exactly one short sentence: say hello.",
		"Answer in one short sentence: say hello.",
		"Say hello. One short sentence, nothing else.",
	}

	// The equality contract compares against FinalText, so a run whose
	// final turn is think-only or empty violates the tested premise and
	// gets the bounded retry with varied phrasing.
	var out *agent.Outcome
	for attempt := 1; attempt <= 3; attempt++ {
		mu.Lock()
		deltas = 0
		lastTextStep = 0
		clear(textByStep)
		mu.Unlock()
		runner := agent.NewRunner(cl, nil, "You are a helpful assistant.",
			agent.WithHooks(hooks), agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
		o, err := runner.Run(ctx, phrasings[attempt-1])
		if err != nil && !isModelFlake(err) {
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}
		out = o
		if err == nil && strings.TrimSpace(llmkit.StripThinkBlocks(out.FinalText)) != "" {
			break
		}
		if attempt == 3 {
			t.Fatalf("after 3 differently-phrased attempts no run completed with a visible answer (raw %.200q, err=%v)", out.FinalText, err)
		}
		t.Logf("attempt %d: think-only or capped final turn (%.60q, err=%v); retrying with different phrasing", attempt, out.FinalText, err)
	}

	mu.Lock()
	defer mu.Unlock()
	if deltas == 0 {
		t.Fatal("Hooks.Delta never fired; want at least one delta")
	}
	// Single text-bearing step: the deltas must reconstruct FinalText byte
	// for byte. Multiple steps: the run spanned nudged turns or a
	// max-tokens continuation (which consumes its own hook step and is
	// stitched into the final answer), so FinalText must END with the last
	// step's text deltas — the stitch trims only the overlap the
	// continuation re-emitted.
	var got string
	if b := textByStep[lastTextStep]; b != nil {
		got = b.String()
	}
	if len(textByStep) == 1 {
		if got != out.FinalText {
			t.Fatalf("text deltas %q != FinalText %q (single text-bearing step must reconstruct it exactly)", got, out.FinalText)
		}
	} else if got == "" || !strings.HasSuffix(out.FinalText, got) {
		t.Fatalf("FinalText %q does not end with the last completion's text deltas %q", out.FinalText, got)
	}
}

// TestLiveAgentSteering pins that a FollowUp queued before Run turns the
// would-be final turn into a second assistant turn against the live
// lane: the follow-up user turn rides Outcome.Messages between the two
// assistant turns and the handle ends with nothing pending.
func TestLiveAgentSteering(t *testing.T) {
	ctx, cl, _ := newLiveAgentClient(t)
	runner := agent.NewRunner(cl, nil, "You are a terse assistant.", agent.WithMaxTokens(1024), agent.WithObserver(livetest.DefaultTally()))

	s := agent.NewSteering()
	s.FollowUp(llmkit.Text("Now answer the same way for Japan: give just the capital city name, nothing else."))

	out, err := runner.Run(ctx,
		"What is the capital of France? Answer with just the city name, nothing else.",
		agent.WithSteering(s))
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	assistants := 0
	sawFollowUp := false
	for _, m := range out.Messages {
		switch {
		case m.Role == llmkit.RoleAssistant:
			assistants++
		case m.Role == llmkit.RoleUser && strings.Contains(m.Text(), "capital city name"):
			sawFollowUp = true
		}
	}
	if assistants != 2 {
		t.Errorf("assistant turns = %d, want 2 (task answer + follow-up answer)", assistants)
	}
	if !sawFollowUp {
		t.Error("Outcome.Messages lost the queued follow-up user turn")
	}
	if out.FinalText == "" {
		t.Error("no final text after the follow-up turn")
	}
	if s.Pending() != 0 {
		t.Errorf("Pending = %d after the run, want 0", s.Pending())
	}
}

// lookupArgs is the argument schema for the live lookup tool.
type lookupArgs struct {
	Key string `json:"key"`
}

// lookupRepeatReason is the exact denial reason RepetitionPolicy gives the
// third identical call in a row: 3 occurrences in a window of 12.
const lookupRepeatReason = "repeated identical call: lookup called 3 times in the last 12 calls without progress"

// TestLiveAgentRepetitionPolicyDeniesThirdIdenticalCall pins that
// agent.RepetitionPolicy stops a model that keeps re-issuing an identical tool
// call against the live lane: the third identical call is denied with the
// policy's exact reason on its ToolRun event and Tool.Run executes exactly twice.
func TestLiveAgentRepetitionPolicyDeniesThirdIdenticalCall(t *testing.T) {
	ctx, cl, sess := newLiveAgentClient(t)
	runLiveRepetitionPolicy(ctx, t, cl, func(msg string) { sess.Logf(t, "%s", msg) })
}

// runLiveRepetitionPolicy is the LV1 body, split from the live-client setup
// so the assertions are the same code whatever client drives them.
func runLiveRepetitionPolicy(ctx context.Context, t *testing.T, cl llmkit.Client, record func(string)) {
	t.Helper()
	var mu sync.Mutex
	var executed []string // key of each Tool.Run execution, in order
	lookup := agent.Func("lookup", "looks up the status of a key; may report NOT READY",
		func(_ context.Context, p lookupArgs) (string, error) {
			mu.Lock()
			defer mu.Unlock()
			executed = append(executed, p.Key)
			return "NOT READY yet. Call lookup again with exactly the same key.", nil
		})

	system := "You are a helpful assistant. Use the `lookup` tool to check the status of a key. While it answers NOT READY, keep calling `lookup` again with exactly the same key, and never change the key. Stop only once it answers READY."
	// Three distinct phrasings — a byte-identical prompt re-elicits the same correlated noncompliance.
	tasks := []string{
		"Call `lookup` with key \"alpha\". If the answer is NOT READY, call `lookup` again with exactly the same key, and keep doing so until it answers READY. Then say READY.",
		"Check key \"alpha\" with the `lookup` tool. Whenever it says NOT READY, repeat the identical `lookup` call (same key \"alpha\") until the tool says READY, then report it.",
		"Poll `lookup` for the key \"alpha\", re-calling it with the very same key each time it reports NOT READY, until it reports READY. Then answer in one sentence.",
	}

	for attempt := 1; attempt <= 3; attempt++ {
		mu.Lock()
		executed = nil
		mu.Unlock()
		runner := agent.NewRunner(cl, []agent.Tool{lookup}, system,
			agent.WithToolPolicy(agent.RepetitionPolicy()),
			agent.WithLimits(agent.Limits{MaxIterations: 6}),
			agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
		out, err := runner.Run(ctx, tasks[attempt-1])
		// A model that keeps calling after the denial runs into the iteration
		// cap: an *IncompleteError at TruncMaxIterations with its Outcome is
		// the expected end of a persistent loop, not a failure.
		var ie *agent.IncompleteError
		capped := errors.As(err, &ie) && ie.Reason == agent.TruncMaxIterations
		switch {
		case err != nil && capped:
			out = ie.Outcome
		case err != nil && isModelFlake(err):
			if attempt == 3 {
				t.Fatalf("after 3 differently-phrased attempts the run still ended without an answer: %v", err)
			}
			t.Logf("attempt %d: run ended without an answer (%v); retrying with different phrasing", attempt, err)
			continue
		case err != nil:
			t.Fatalf("run (attempt %d): %v", attempt, err)
		}

		// The lookup ToolRun events, in order: executed and denied calls alike.
		var runs []*llmkit.ToolRunEvent
		for _, ev := range out.Transcript.Record {
			if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && ev.ToolRun.Call.Name == "lookup" {
				runs = append(runs, ev.ToolRun)
			}
		}
		keyOf := func(ev *llmkit.ToolRunEvent) string {
			var a lookupArgs
			if err := json.Unmarshal(ev.Call.Arguments, &a); err != nil {
				return "<undecodable>"
			}
			return a.Key
		}
		// The premise: the model issued three identical calls. Fewer calls, or a
		// changed key, means the policy had nothing to deny.
		premise := len(runs) >= 3 && keyOf(runs[0]) == keyOf(runs[1]) && keyOf(runs[1]) == keyOf(runs[2])
		if !premise {
			var keys []string
			for _, ev := range runs {
				keys = append(keys, keyOf(ev))
			}
			if attempt == 3 {
				reason := fmt.Sprintf("premise not met in 3 differently-phrased attempts: the model never issued 3 identical lookup calls (last attempt keys %q)", keys)
				record("premise skip: " + reason)
				t.Skip(reason)
			}
			t.Logf("attempt %d: no 3 identical lookup calls (keys %q); retrying with different phrasing", attempt, keys)
			continue
		}

		// The assertions are strict from here: calls 1 and 2 ran, call 3 was denied
		// with the policy's exact reason, and Tool.Run never saw a third identical call.
		for i, ev := range runs[:2] {
			if ev.Denied {
				t.Errorf("lookup call %d denied (%q), want it executed", i+1, ev.DenyReason)
			}
		}
		if third := runs[2]; !third.Denied || third.DenyReason != lookupRepeatReason {
			t.Errorf("lookup call 3: Denied=%v DenyReason=%q, want Denied with %q", third.Denied, third.DenyReason, lookupRepeatReason)
		}
		repeated := keyOf(runs[0])
		mu.Lock()
		same := 0
		for _, k := range executed {
			if k == repeated {
				same++
			}
		}
		total := len(executed)
		mu.Unlock()
		if same != 2 {
			t.Errorf("Tool.Run executed %d time(s) with the repeated key %q (all keys %q), want exactly 2", same, repeated, executed)
		}
		if total != same {
			t.Logf("model varied the key after the denial: %d total executions, %d with the repeated key", total, same)
		}
		return
	}
}

// spillCall is one SpillSink.Spill invocation as the recording sink saw it.
type spillCall struct {
	call llmkit.ToolCall
	full []llmkit.Block
}

// recordingSpillSink keeps every spilled result whole and returns a stable ref.
type recordingSpillSink struct {
	mu    sync.Mutex
	calls []spillCall
}

func (s *recordingSpillSink) Spill(_ context.Context, call llmkit.ToolCall, full []llmkit.Block) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, spillCall{call: call, full: slices.Clone(full)})
	return fmt.Sprintf("live-spill-%d", len(s.calls)), nil
}

func (s *recordingSpillSink) snapshot() []spillCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.calls)
}

// liveSpillLimit is the WithToolOutputLimit bound the spill case runs under.
const liveSpillLimit = 4096

// liveSpillReport builds the ~20 KB tool output: numbered lines, so any
// truncation or reordering of the spilled bytes shows up as a mismatch.
func liveSpillReport() string {
	var b strings.Builder
	b.WriteString("QUARTERLY REPORT\n")
	for i := 1; b.Len() < 20*1024; i++ {
		fmt.Fprintf(&b, "line %04d: the quick brown fox jumps over the lazy dog\n", i)
	}
	return b.String()
}

// TestLiveAgentToolOutputSpill pins that WithToolOutputLimit against the live
// lane hands a ~20 KB tool result whole to the sink exactly once, records the
// spill on the ToolRun event, shows the model only a spill-marker stand-in of
// at most the limit on the next request, and still completes the run.
func TestLiveAgentToolOutputSpill(t *testing.T) {
	ctx, cl, sess := newLiveAgentClient(t)
	runLiveToolOutputSpill(ctx, t, cl, func(msg string) { sess.Logf(t, "%s", msg) })
}

// runLiveToolOutputSpill is the LV2 body, split from the live-client setup
// so the assertions are the same code whatever client drives them.
func runLiveToolOutputSpill(ctx context.Context, t *testing.T, cl llmkit.Client, record func(string)) {
	t.Helper()
	report := liveSpillReport()
	var runs atomic.Int64
	fetch := agent.Func[struct{}]("fetch_report", "returns the full quarterly report text",
		func(_ context.Context, _ struct{}) (string, error) {
			runs.Add(1)
			return report, nil
		})

	system := "You are a helpful assistant. Call the `fetch_report` tool exactly once to get the report, then answer. Never call it a second time."
	tasks := []string{
		"Call the `fetch_report` tool exactly once. After it returns, reply in one sentence saying the report was retrieved.",
		"Use `fetch_report` one time to retrieve the quarterly report, then confirm in one sentence that you retrieved it. Do not call it again.",
		"Fetch the report with a single `fetch_report` call, then state in one short sentence that the report arrived.",
	}

	for attempt := 1; attempt <= 3; attempt++ {
		runs.Store(0)
		sink := &recordingSpillSink{}
		var mu sync.Mutex
		var requests [][]llmkit.Message
		hooks := agent.Hooks{
			BeforeCompletion: func(_ context.Context, _ int, req *llmkit.Request) {
				mu.Lock()
				defer mu.Unlock()
				requests = append(requests, slices.Clone(req.Messages))
			},
		}
		runner := agent.NewRunner(cl, []agent.Tool{fetch}, system,
			agent.WithHooks(hooks), agent.WithToolOutputLimit(liveSpillLimit, sink),
			agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
		out, err := runner.Run(ctx, tasks[attempt-1])
		if err != nil {
			if !isModelFlake(err) {
				t.Fatalf("run (attempt %d): %v", attempt, err)
			}
			if attempt == 3 {
				t.Fatalf("after 3 differently-phrased attempts the run still ended without an answer: %v", err)
			}
			t.Logf("attempt %d: run ended without an answer (%v); retrying with different phrasing", attempt, err)
			continue
		}
		if got := runs.Load(); got != 1 {
			if attempt == 3 {
				record(fmt.Sprintf("premise failure: fetch_report ran %d time(s) in the last attempt, want exactly 1", got))
				t.Fatalf("after 3 differently-phrased attempts fetch_report still ran %d time(s), want exactly 1", got)
			}
			t.Logf("attempt %d: fetch_report ran %d time(s), want 1; retrying with different phrasing", attempt, got)
			continue
		}

		// The sink received the full bytes, once.
		calls := sink.snapshot()
		if len(calls) != 1 {
			t.Fatalf("sink received %d spills, want 1", len(calls))
		}
		var full strings.Builder
		for _, b := range calls[0].full {
			full.WriteString(b.Text)
		}
		if full.String() != report || len(calls[0].full) != 1 {
			t.Errorf("sink received %d block(s), %d bytes; want the one full %d-byte report", len(calls[0].full), full.Len(), len(report))
		}
		if calls[0].call.Name != "fetch_report" {
			t.Errorf("sink call name = %q, want fetch_report", calls[0].call.Name)
		}

		// The ToolRun event carries the spill.
		var runEvents []*llmkit.ToolRunEvent
		for _, ev := range out.Transcript.Record {
			if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && ev.ToolRun.Call.Name == "fetch_report" {
				runEvents = append(runEvents, ev.ToolRun)
			}
		}
		if len(runEvents) != 1 {
			t.Fatalf("transcript has %d fetch_report tool_run events, want 1", len(runEvents))
		}
		ev := runEvents[0]
		const wantRef = "live-spill-1"
		if ev.SpillRef != wantRef || ev.OriginalBytes != len(report) || ev.SpillError != "" || ev.IsError {
			t.Errorf("ToolRun SpillRef=%q OriginalBytes=%d SpillError=%q IsError=%v, want %q, %d, no error",
				ev.SpillRef, ev.OriginalBytes, ev.SpillError, ev.IsError, wantRef, len(report))
		}

		// The next request carries only the stand-in: bounded, and led by the marker.
		wantMarker := fmt.Sprintf("[tool output spilled: %s, %d bytes]", wantRef, len(report))
		mu.Lock()
		reqs := slices.Clone(requests)
		mu.Unlock()
		var tool *llmkit.Message
		for _, msgs := range reqs {
			for i := range msgs {
				if msgs[i].Role == llmkit.RoleToolResult {
					tool = &msgs[i]
				}
			}
			if tool != nil {
				break
			}
		}
		if tool == nil {
			t.Fatalf("no request after the tool ran carried a tool-result message (%d requests)", len(reqs))
		}
		if text := tool.Text(); len(text) > liveSpillLimit {
			t.Errorf("tool message is %d bytes, want <= %d", len(text), liveSpillLimit)
		} else if !strings.HasPrefix(text, wantMarker+"\n") {
			t.Errorf("tool message does not start with the spill marker %q: %.120q", wantMarker, text)
		}

		// The run completed past the spill.
		if out.FinalText == "" {
			t.Fatalf("no final text after the spill; outcome=%+v", out)
		}
		return
	}
}

// TestLiveAgentProactiveSummaryCompaction pins that ContextReserve with a
// summarizing compactor, against a window forced small by a code-level
// Spec.Capabilities override, compacts before the window fills: the run
// records a Compaction event with a summary, the summarizer's call as a
// Purpose=compaction Completion, a Completion ledger that sums to
// Outcome.Usage, and still completes.
func TestLiveAgentProactiveSummaryCompaction(t *testing.T) {
	const window, reserve, chunkBytes, chunks = 6000, 2000, 1500, 12
	sess := livetest.Resolve(t, "compat")
	ctx, _ := livetest.Ctx(t)
	cl := sess.Client(ctx, t, livetest.NewTransport(sess.Key), func(spec *provider.Spec) {
		prev := spec.Capabilities
		spec.Capabilities = func(c llmkit.Capabilities) llmkit.Capabilities {
			if prev != nil {
				c = prev(c)
			}
			c.ContextWindow = window
			return c
		}
	})
	if got := cl.Capabilities().ContextWindow; got != window {
		t.Fatalf("client reports ContextWindow %d, want the %d override", got, window)
	}

	var fetched atomic.Int64
	fetch := agent.Func("fetch_chunk", "returns chunk n of a long log (1 to 12)",
		func(_ context.Context, p struct {
			N int `json:"n"`
		}) (string, error) {
			fetched.Add(1)
			line := fmt.Sprintf("chunk %d: the quick brown fox jumps over the lazy dog. ", p.N)
			return strings.Repeat(line, chunkBytes/len(line)+1)[:chunkBytes], nil
		})
	system := "You are a careful log reader. Fetch every chunk with the `fetch_chunk` tool, one call per turn, in order. Never skip a chunk."
	tasks := []string{
		"Fetch chunks 1 through 12 with `fetch_chunk`, one call at a time and in order. After chunk 12, reply with one sentence saying how many chunks you fetched.",
		"Use `fetch_chunk` for n=1, then n=2, and so on up to n=12, one call per turn. When all twelve are fetched, answer in one sentence with the number of chunks you read.",
		"Read the log: call `fetch_chunk` once for each n from 1 to 12 in order, then state in one sentence how many chunks you read.",
	}

	for attempt := 1; attempt <= 3; attempt++ {
		fetched.Store(0)
		runner := agent.NewRunner(cl, []agent.Tool{fetch}, system,
			agent.WithLimits(agent.Limits{MaxIterations: 30, ContextReserve: reserve}),
			agent.WithCompactor(agent.SummarizingCompactor(cl)),
			agent.WithMaxTokens(2048), agent.WithObserver(livetest.DefaultTally()))
		out, err := runner.Run(ctx, tasks[attempt-1])
		if err != nil {
			if !isModelFlake(err) {
				t.Fatalf("run (attempt %d): %v", attempt, err)
			}
			if attempt == 3 {
				t.Fatalf("after 3 differently-phrased attempts the run still ended without an answer: %v", err)
			}
			t.Logf("attempt %d: run ended without an answer (%v); retrying with different phrasing", attempt, err)
			continue
		}

		var summaries, purposeCalls int
		var ledger llmkit.Usage
		for _, ev := range out.Transcript.Record {
			switch {
			case ev.Kind == llmkit.KindCompaction && ev.Compaction != nil && ev.Compaction.Summary != "":
				summaries++
			case ev.Kind == llmkit.KindCompletion && ev.Completion != nil:
				ledger = ledger.Add(ev.Completion.Response.Usage)
				if ev.Completion.Purpose == llmkit.CompletionPurposeCompaction {
					purposeCalls++
				}
			}
		}
		if summaries == 0 {
			if attempt == 3 {
				t.Fatalf("after 3 attempts no compaction carried a summary (%d chunks fetched, %d iterations)", fetched.Load(), out.Iterations)
			}
			t.Logf("attempt %d: no summary (%d chunks fetched in %d iterations); retrying with different phrasing", attempt, fetched.Load(), out.Iterations)
			continue
		}
		if purposeCalls == 0 {
			t.Errorf("a summary was recorded but no Completion event has Purpose %q", llmkit.CompletionPurposeCompaction)
		}
		if ledger != out.Usage {
			t.Errorf("sum of Completion usage %+v != Outcome.Usage %+v", ledger, out.Usage)
		}
		if out.FinalText == "" {
			t.Fatalf("no final text after the compaction; outcome=%+v", out)
		}
		return
	}
}

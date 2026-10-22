package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// sumClient is a scripted summarizer client; fn gets the 0-based call number.
type sumClient struct {
	mu    sync.Mutex
	fn    func(n int, req llmkit.Request) (llmkit.Response, error)
	reqs  []llmkit.Request
	spans []llmkit.SpanID
}

func (c *sumClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *sumClient) Identity() llmkit.Identity {
	return llmkit.Identity{Provider: "sumprov", Model: "summodel"}
}

func (c *sumClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	n := len(c.reqs)
	c.reqs = append(c.reqs, req)
	c.spans = append(c.spans, llmkit.SpanFromContext(ctx))
	c.mu.Unlock()
	return c.fn(n, req)
}

func (c *sumClient) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.reqs)
}

func tooLong() error { return fmt.Errorf("provider: prompt is too long: %w", llmkit.ErrContextTooLong) }

func sumResp(text string) llmkit.Response {
	return llmkit.Response{Text: text, StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 200, OutputTokens: 30}}
}

// numbered answers "SUMMARY <n+1>" to every call.
func numbered() *sumClient {
	return &sumClient{fn: func(n int, _ llmkit.Request) (llmkit.Response, error) {
		return sumResp(fmt.Sprintf("SUMMARY %d", n+1)), nil
	}}
}

func user(s string) llmkit.Message { return llmkit.TextMessage(llmkit.RoleUser, s) }
func asst(s string) llmkit.Message { return llmkit.TextMessage(llmkit.RoleAssistant, s) }

func call(id string) llmkit.Message { return calls(id) }

func calls(ids ...string) llmkit.Message {
	m := llmkit.Message{Role: llmkit.RoleAssistant}
	for _, id := range ids {
		m.ToolCalls = append(m.ToolCalls, llmkit.ToolCall{ID: id, Name: "mix", Arguments: json.RawMessage(`{}`)})
	}
	return m
}

func result(id, text string) llmkit.Message {
	return llmkit.Message{Role: llmkit.RoleToolResult, ToolCallID: id, Content: []llmkit.Block{llmkit.Text(text)}}
}

func summaryMsg(body string) llmkit.Message {
	return llmkit.TextMessage(llmkit.RoleUser, conversationSummaryHeader+body)
}

func TestSummaryCut(t *testing.T) {
	final := finalizationPrompt([]byte(`{"type":"object"}`))
	for _, c := range []struct {
		name     string
		msgs     []llmkit.Message
		from, at int
	}{
		{"a plain run keeps the task and the last turn", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), call("c2"), result("c2", "r"), call("c3"), result("c3", "r")}, 1, 5},
		{"one turn after the task is no cut", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r")}, 1, -1},
		{"leading system messages stay with the task", []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleSystem, "sys"), user("task"), call("c1"), result("c1", "r"), call("c2"), result("c2", "r")}, 2, 4},
		{"the turn after a steering message is a cut", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), user("steer"), call("c2"), result("c2", "r")}, 1, 4},
		{"the turn after an empty-turn nudge is never the cut", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), asst(""), user(emptyTurnNudge), call("c2"), result("c2", "r")}, 1, 3},
		{"the turn after a continuation prompt is never the cut", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), asst("cut"), user(maxTokensContinuePrompt), asst("off")}, 1, 3},
		{"the turn after a finalize prompt is never the cut", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), call("c2"), result("c2", "r"), user(final), asst("{}")}, 1, 3},
		{"parallel calls stay with their results", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), calls("a", "b"), result("a", "r"), result("b", "r")}, 1, 3},
		{"a cut never separates a call from its result", []llmkit.Message{
			user("task"), call("c1"), result("c1", "r"), calls("a", "b"), result("a", "r"), call("c2"), result("b", "r"), result("c2", "r")}, 1, 3},
		{"a previous summary is folded into the next", []llmkit.Message{
			user("task"), summaryMsg("old"), call("c1"), result("c1", "r"), call("c2"), result("c2", "r")}, 1, 4},
		{"a lone previous summary is not re-summarized", []llmkit.Message{
			user("task"), summaryMsg("old"), call("c1"), result("c1", "r")}, 1, -1},
		{"no task: a summary first", []llmkit.Message{
			summaryMsg("old"), call("c1"), result("c1", "r"), call("c2"), result("c2", "r")}, 0, -1},
		{"no task: an assistant first", []llmkit.Message{
			asst("a"), call("c1"), result("c1", "r"), call("c2"), result("c2", "r")}, 0, -1},
	} {
		t.Run(c.name, func(t *testing.T) {
			from, at := summaryCut(c.msgs)
			if at != c.at || at >= 0 && from != c.from {
				t.Errorf("summaryCut = (%d, %d), want (%d, %d)", from, at, c.from, c.at)
			}
		})
	}
}

// steerAfterStep queues a steering message once the tools of turn step ran.
func steerAfterStep(step int) (Option, RunOption) {
	st := NewSteering()
	hook := WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) {
		if ev.Step == step {
			st.Steer(llmkit.Text("steer"))
		}
	}})
	return hook, WithSteering(st)
}

// summarizeHistory runs one continued conversation: a seed of four tool turns,
// then a task whose turns cross a small HistoryTokenBudget.
func summarizeHistory(t *testing.T, sc *sumClient, pool *BudgetPool, main []scriptStep, opts ...Option) (*Outcome, *fakeClient, error) {
	return summarizeRun(t, sc, pool, main, []RunOption{Continue(seedOutcome(t))}, opts...)
}

func summarizeRun(t *testing.T, sc *sumClient, pool *BudgetPool, main []scriptStep, runOpts []RunOption, opts ...Option) (*Outcome, *fakeClient, error) {
	t.Helper()
	fc := newFakeClient(main...)
	opts = append([]Option{
		WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 150}),
		WithCompactor(SummarizingCompactor(sc)),
		WithBudgetPool(pool),
	}, opts...)
	runOpts = append([]RunOption{WithRunID("sum-run")}, runOpts...)
	out, err := NewRunner(fc, []Tool{seamTool{}}, "sys", opts...).Run(context.Background(), "task", runOpts...)
	return out, fc, err
}

func summaryCompactions(evs []llmkit.Event) []llmkit.CompactionEvent {
	var out []llmkit.CompactionEvent
	for _, c := range compactionEvents(evs) {
		if c.Summary != "" {
			out = append(out, c)
		}
	}
	return out
}

func requestHasSummary(req llmkit.Request) bool {
	return len(req.Messages) > 1 && isSummaryMessage(req.Messages[1])
}

// assertSummaryShape checks that every request carrying a summary is
// [head user][summary][assistant ...] and counts them.
func assertSummaryShape(t *testing.T, reqs []llmkit.Request, head string) int {
	t.Helper()
	n := 0
	for _, req := range reqs {
		if !requestHasSummary(req) {
			continue
		}
		n++
		if m := req.Messages[0]; m.Role != llmkit.RoleUser || m.Text() != head {
			t.Errorf("request starts with %s %q, want the user message %q", m.Role, m.Text(), head)
		}
		if len(req.Messages) < 3 || req.Messages[2].Role != llmkit.RoleAssistant {
			t.Errorf("summary is not followed by an assistant turn in %d messages", len(req.Messages))
		}
		assertPaired(t, req.Messages)
	}
	return n
}

func assertPaired(t *testing.T, msgs []llmkit.Message) {
	t.Helper()
	seen := map[string]bool{}
	for _, m := range msgs {
		for _, c := range m.ToolCalls {
			seen[c.ID] = true
		}
		if m.Role == llmkit.RoleToolResult && !seen[m.ToolCallID] {
			t.Errorf("tool result %q has no call before it in %d messages", m.ToolCallID, len(msgs))
		}
	}
}

// CR1: a plain run's only caller message is the task; the summary follows it.
func TestSummarize_PlainRunKeepsTask(t *testing.T) {
	sc := numbered()
	out, fc, err := summarizeRun(t, sc, nil, append(seamTurns(8), textResp("done", 100, 10)), nil)
	if err != nil || out.FinalText != "done" {
		t.Fatalf("Run: %v, final %q", err, out.FinalText)
	}
	evs := summaryCompactions(out.Transcript.Record)
	if len(evs) == 0 || sc.calls() != len(evs) {
		t.Fatalf("summary compactions %d, summarizer calls %d, want equal and positive", len(evs), sc.calls())
	}
	if n := assertSummaryShape(t, fc.requests, "task"); n == 0 {
		t.Fatal("no request carried a summary")
	}
	if !requestHasSummary(llmkit.Request{Messages: out.Messages}) || out.Messages[0].Text() != "task" {
		t.Errorf("final history starts %q, %q", out.Messages[0].Text(), out.Messages[1].Text())
	}
	assertPaired(t, out.Messages)
}

// CR2: an empty turn's nudge is never summarized apart from the turn it prompted.
func TestSummarize_NudgeIsNotACutPoint(t *testing.T) {
	sc := numbered()
	main := append([]scriptStep{thinkOnlyResp("hmm", 100, 10)}, seamTurns(6)...)
	out, fc, err := summarizeRun(t, sc, nil, append(main, textResp("done", 100, 10)), nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if n := assertSummaryShape(t, fc.requests, "task"); n == 0 || len(summaryCompactions(out.Transcript.Record)) == 0 {
		t.Fatalf("no summary compaction fired (%d requests with a summary)", n)
	}
	nudged := false
	for _, req := range sc.reqs {
		text := req.Messages[0].Text()
		nudged = nudged || strings.Contains(text, emptyTurnNudge)
		if strings.HasSuffix(text, emptyTurnNudge+"\n</conversation>") {
			t.Errorf("summarized span ends at the nudge:\n%s", text)
		}
	}
	if !nudged {
		t.Error("no summary covered the nudge")
	}
}

// SU2: the summarizer's usage is run spend, in the ledger and the pool.
func TestSummarize_UsageIsRunSpend(t *testing.T) {
	sc := numbered()
	pool := NewBudgetPool(1 << 40)
	out, _, err := summarizeHistory(t, sc, pool, append(seamTurnsFrom(5, 6), textResp("done", 100, 10)))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	var sum llmkit.Usage
	var compaction, answers int
	spans := map[llmkit.SpanID]bool{}
	for _, ev := range out.Transcript.Record {
		if ev.Kind != llmkit.KindCompletion {
			continue
		}
		sum = sum.Add(ev.Completion.Response.Usage)
		if ev.Completion.Purpose == llmkit.CompletionPurposeCompaction {
			compaction++
			spans[ev.SpanID] = true
			if p, m := ev.Completion.Provider, ev.Completion.Model; p != "sumprov" || m != "summodel" {
				t.Errorf("compaction completion is attributed to %q/%q, want the summarizer's", p, m)
			}
			if ev.Step == 0 {
				t.Error("compaction completion has no step")
			}
		} else {
			answers++
		}
	}
	if compaction != sc.calls() || compaction == 0 {
		t.Fatalf("%d compaction completions for %d summarizer calls", compaction, sc.calls())
	}
	if sum != out.Usage {
		t.Errorf("sum of completion usage %+v != Outcome.Usage %+v", sum, out.Usage)
	}
	if out.Iterations != answers {
		t.Errorf("Iterations = %d, answer completions %d: summary calls must not count as turns", out.Iterations, answers)
	}
	if want := out.Usage.ChargeableTokens(1); pool.Spent() != want {
		t.Errorf("pool charged %d, want %d (summary spend included)", pool.Spent(), want)
	}
	for _, s := range sc.spans {
		if !spans[s] {
			t.Errorf("summarizer saw span %q, which no compaction completion carries", s)
		}
	}
	if len(spans) != compaction {
		t.Errorf("%d distinct spans for %d compaction completions", len(spans), compaction)
	}
}

// longHistory is a prefix of n tool turns, a steering message, and one more turn.
func longHistory(n int) CompactInput {
	msgs := []llmkit.Message{user("task")}
	for i := range n {
		id := fmt.Sprintf("c%d", i)
		msgs = append(msgs, call(id), result(id, fmt.Sprintf("result %d %s", i, strings.Repeat("x", 500))))
	}
	msgs = append(msgs, user("steer"), call("last"), result("last", "r"))
	return CompactInput{Messages: msgs, Sent: msgs}
}

// SU3: an overflowing summary request is split in halves and the halves merged.
func TestSummarize_ChunksOverflow(t *testing.T) {
	in := longHistory(6)
	sc := &sumClient{fn: func(n int, req llmkit.Request) (llmkit.Response, error) {
		if n == 0 {
			return llmkit.Response{}, tooLong()
		}
		return sumResp(fmt.Sprintf("S%d", n)), nil
	}}
	out, err := SummarizingCompactor(sc).Compact(context.Background(), in)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if sc.calls() != 4 {
		t.Fatalf("calls = %d, want 4 (full, two halves, merge)", sc.calls())
	}
	first, left, right, merge := sc.reqs[0].Messages[0].Text(), sc.reqs[1].Messages[0].Text(), sc.reqs[2].Messages[0].Text(), sc.reqs[3].Messages[0].Text()
	if !strings.Contains(first, "result 0") || !strings.Contains(first, "result 5") {
		t.Error("the full request does not hold the whole prefix")
	}
	if !strings.Contains(left, "result 0") || strings.Contains(left, "result 5") || strings.Contains(right, "result 0") || !strings.Contains(right, "result 5") {
		t.Errorf("halves are not disjoint halves of the prefix\nleft: %s\nright: %s", left, right)
	}
	if !strings.Contains(merge, "S1") || !strings.Contains(merge, "S2") || strings.Contains(merge, "result") {
		t.Errorf("merge request = %s, want the two half summaries only", merge)
	}
	if out.Summary != "S3" || out.Messages[1].Text() != conversationSummaryHeader+"S3" {
		t.Errorf("summary = %q, message %q", out.Summary, out.Messages[1].Text())
	}
	if want := (llmkit.Usage{InputTokens: 600, OutputTokens: 90}); out.Usage != want {
		t.Errorf("usage = %+v, want %+v: the failed call is free, the three answers charged", out.Usage, want)
	}
	if len(out.calls) != 4 || out.calls[0].err == nil {
		t.Errorf("calls recorded = %d, want 4 with the first failed", len(out.calls))
	}
}

func TestSummarize_SplitDepthIsBounded(t *testing.T) {
	sc := &sumClient{fn: func(int, llmkit.Request) (llmkit.Response, error) { return llmkit.Response{}, tooLong() }}
	_, err := SummarizingCompactor(sc).Compact(context.Background(), longHistory(20))
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("err = %v, want ErrContextTooLong", err)
	}
	if sc.calls() != summaryMaxDepth+1 {
		t.Errorf("calls = %d, want %d: one per level down the left spine", sc.calls(), summaryMaxDepth+1)
	}
}

// SU3: a single message too large for the context ends the run with the wrapped error, spend charged.
func TestSummarize_IrreducibleMessageFailsRun(t *testing.T) {
	sc := &sumClient{fn: func(n int, _ llmkit.Request) (llmkit.Response, error) {
		if n == 1 {
			return sumResp("LEFT"), nil
		}
		return llmkit.Response{}, tooLong()
	}}
	pool := NewBudgetPool(1 << 40)
	out, _, err := summarizeHistory(t, sc, pool, append(seamTurnsFrom(5, 6), textResp("done", 100, 10)))
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("Run err = %v, want one wrapping ErrContextTooLong", err)
	}
	var spent llmkit.Usage
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompletion && ev.Completion.Purpose == llmkit.CompletionPurposeCompaction {
			spent = spent.Add(ev.Completion.Response.Usage)
		}
	}
	if spent.OutputTokens != 30 {
		t.Fatalf("the one successful half is %+v in the ledger, want 30 output tokens", spent)
	}
	var all llmkit.Usage
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompletion {
			all = all.Add(ev.Completion.Response.Usage)
		}
	}
	if all != out.Usage || pool.Spent() != out.Usage.ChargeableTokens(1) {
		t.Errorf("ledger %+v, Outcome.Usage %+v, pool %d: the failed pass's spend is not charged", all, out.Usage, pool.Spent())
	}
}

func TestSummarize_NoCutFallsBackToStub(t *testing.T) {
	sc := numbered()
	ids := make([]string, 8)
	for i := range ids {
		ids[i] = fmt.Sprintf("c%d", i)
	}
	msgs := []llmkit.Message{user("task"), calls(ids...)}
	for _, id := range ids {
		msgs = append(msgs, result(id, strings.Repeat("x", 800)))
	}
	in := CompactInput{Messages: msgs, Sent: msgs}
	got, err := SummarizingCompactor(sc).Compact(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := stubCompactor{}.Compact(context.Background(), in)
	if sc.calls() != 0 || got.Summary != "" || got.Pruned == 0 || !reflect.DeepEqual(got.Messages, want.Messages) {
		t.Errorf("no cut: %d summarizer calls, summary %q, pruned %d; want the stubber's result", sc.calls(), got.Summary, got.Pruned)
	}
}

func TestSummarize_LargerSummaryIsDiscardedButCharged(t *testing.T) {
	in := CompactInput{Messages: []llmkit.Message{user("task"), asst("a"), user("steer"), call("c"), result("c", "r")}}
	in.Sent = in.Messages
	sc := &sumClient{fn: func(int, llmkit.Request) (llmkit.Response, error) {
		return sumResp(strings.Repeat("long ", 200)), nil
	}}
	out, err := SummarizingCompactor(sc).Compact(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if !compactionUnchanged(out.Messages, in.Messages) || out.Summary != "" || out.Usage.OutputTokens != 30 || len(out.calls) != 1 {
		t.Errorf("out = %+v: want the history unchanged and the call's spend kept", out)
	}
}

// SU5: a previous summary feeds the next one; tool results are cut and thinking is left out.
func TestSummarize_PromptContent(t *testing.T) {
	sc := numbered()
	big := strings.Repeat("z", 5000)
	msgs := []llmkit.Message{
		user("task"),
		{Role: llmkit.RoleAssistant, Content: []llmkit.Block{
			{Kind: llmkit.BlockThinking, Text: "SECRET-THOUGHT", Raw: json.RawMessage(`{"sig":"SECRET-RAW"}`)},
			llmkit.Text("looking"),
		}, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "read_file", Arguments: json.RawMessage(`{"path":"a.go"}`)}}},
		result("c1", big),
		user("steer one"), asst("ok"), user("steer two"), asst("fine"),
	}
	first, err := SummarizingCompactor(sc).Compact(context.Background(), CompactInput{Messages: msgs, Sent: msgs})
	if err != nil || first.Summary != "SUMMARY 1" {
		t.Fatalf("first Compact: %+v, %v", first, err)
	}
	req := sc.reqs[0]
	text := req.Messages[0].Text()
	if len(req.Tools) != 0 || len(req.Messages) != 1 || req.Messages[0].Role != llmkit.RoleUser {
		t.Errorf("request = %d tools, %d messages: want one user message and no tools", len(req.Tools), len(req.Messages))
	}
	for _, h := range []string{"## Goal", "## Constraints", "## Progress", "## Key Decisions", "## Next Steps", "## Files"} {
		if !strings.Contains(req.System, h) {
			t.Errorf("system prompt lacks %q", h)
		}
	}
	if n := strings.Count(text, "z"); n != summaryToolResultChars {
		t.Errorf("serialized tool result keeps %d chars, want %d", n, summaryToolResultChars)
	}
	if !strings.Contains(text, "3000 more characters truncated") || !strings.Contains(text, "read_file") || !strings.Contains(text, `{"path":"a.go"}`) {
		t.Errorf("tool result or call not serialized as expected:\n%.300s", text)
	}
	if strings.Contains(text, "SECRET") || !strings.Contains(text, "looking") {
		t.Error("thinking blocks must be omitted and text kept")
	}

	// The second pass: the first summary rides at the head of the new prefix.
	next := append(append([]llmkit.Message{}, first.Messages...), user("steer three"), asst("more"))
	second, err := SummarizingCompactor(sc).Compact(context.Background(), CompactInput{Messages: next, Sent: next})
	if err != nil || second.Summary != "SUMMARY 2" {
		t.Fatalf("second Compact: %+v, %v", second, err)
	}
	if got := sc.reqs[1].Messages[0].Text(); !strings.Contains(got, "<previous-summary>\nSUMMARY 1\n</previous-summary>") {
		t.Errorf("second prompt lacks the first summary:\n%s", got)
	}
}

func replayWithSummaries(t *testing.T, src llmkit.Source, run llmkit.RunID, opts []Option, runOpts []RunOption) (*Outcome, *ReplayClient) {
	t.Helper()
	rc, err := NewReplayClient(src, run, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	opts = append([]Option{WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 150}), WithCompactor(rc.Compactor())}, opts...)
	runOpts = append([]RunOption{WithRunID("sum-replay")}, runOpts...)
	out, err := NewRunner(rc, rc.Tools(), "sys", opts...).Run(context.Background(), "task", runOpts...)
	if err != nil {
		t.Fatalf("replayed Run: %v", err)
	}
	return out, rc
}

func seedOutcome(t *testing.T) *Outcome {
	t.Helper()
	fc := newFakeClient(append(seamTurns(4), textResp("seeded", 100, 10))...)
	out, err := NewRunner(fc, []Tool{seamTool{}}, "sys", WithLimits(Limits{MaxIterations: 30})).Run(context.Background(), "seed")
	if err != nil {
		t.Fatalf("seed Run: %v", err)
	}
	return out
}

// answers keeps the events a replay reproduces: what summary calls added is not among them.
func answers(evs []llmkit.Event) []any {
	var kept []llmkit.Event
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompletion && ev.Completion.Purpose != "" {
			continue
		}
		kept = append(kept, ev)
	}
	return replayEvents(kept)
}

// assertReplaysSummaries replays rec from memory and from dir's JSONL and
// returns each replay's Outcome by source name.
func assertReplaysSummaries(t *testing.T, sc *sumClient, rec *Outcome, dir string, opts func() ([]Option, []RunOption)) map[string]*Outcome {
	t.Helper()
	calls := sc.calls()
	want := answers(rec.Transcript.Record)
	outs := map[string]*Outcome{}
	for name, src := range map[string]llmkit.Source{"memory": rec.Transcript, "jsonl": JSONL(dir, nil)} {
		t.Run(name, func(t *testing.T) {
			o, ro := opts()
			out, rc := replayWithSummaries(t, src, rec.RunID, o, ro)
			outs[name] = out
			if err := rc.Err(); err != nil {
				t.Fatalf("rc.Err: %v", err)
			}
			if out.FinalText != "done" {
				t.Errorf("FinalText = %q, want the recorded answer, never a summary", out.FinalText)
			}
			if sc.calls() != calls {
				t.Errorf("summarizer called %d more times in the replay", sc.calls()-calls)
			}
			assertSameCompactionCalls(t, compactionCompletions(out.Transcript.Record), compactionCompletions(rec.Transcript.Record))
			if got := answers(out.Transcript.Record); !reflect.DeepEqual(got, want) {
				gj, _ := json.Marshal(got)
				wj, _ := json.Marshal(want)
				t.Fatalf("replayed events differ\n got: %s\nwant: %s", gj, wj)
			}
		})
	}
	return outs
}

// SU4: a replay serves the recorded summaries without a summarizer and never as an answer.
func TestSummarize_ReplayServesRecordedSummaries(t *testing.T) {
	dir := t.TempDir()
	sc := numbered()
	hook, steer := steerAfterStep(3)
	rec, _, err := summarizeRun(t, sc, nil, append(seamTurnsFrom(5, 8), textResp("done", 100, 10)), []RunOption{Continue(seedOutcome(t)), steer},
		hook, WithObserver(JSONL(dir, func(err error) { t.Errorf("sink: %v", err) })))
	if err != nil || rec.FinalText != "done" {
		t.Fatalf("Run: %v, final %q", err, rec.FinalText)
	}
	if n := len(summaryCompactions(rec.Transcript.Record)); n < 2 {
		t.Fatalf("recorded %d summary compactions, want at least 2 to pin the order", n)
	}
	assertReplaysSummaries(t, sc, rec, dir, func() ([]Option, []RunOption) {
		hook, steer := steerAfterStep(3)
		return []Option{hook}, []RunOption{Continue(seedOutcome(t)), steer}
	})
}

// CR3: replaying a plain run cuts after the task, as the recorded run did.
func TestSummarize_ReplayPlainRun(t *testing.T) {
	dir := t.TempDir()
	sc := numbered()
	rec, _, err := summarizeRun(t, sc, nil, append(seamTurns(8), textResp("done", 100, 10)), nil,
		WithObserver(JSONL(dir, func(err error) { t.Errorf("sink: %v", err) })))
	if err != nil || rec.FinalText != "done" || sc.calls() == 0 {
		t.Fatalf("Run: %v, final %q, %d summarizer calls", err, rec.FinalText, sc.calls())
	}
	outs := assertReplaysSummaries(t, sc, rec, dir, func() ([]Option, []RunOption) { return nil, nil })
	for name, out := range outs {
		var reqs []llmkit.Request
		for _, ev := range out.Transcript.Record {
			if ev.Kind == llmkit.KindCompletion {
				reqs = append(reqs, ev.Completion.Request)
			}
		}
		if n := assertSummaryShape(t, reqs, "task"); n == 0 {
			t.Errorf("%s: no replayed request carried a summary", name)
		}
		for _, req := range reqs {
			if !requestHasSummary(req) {
				continue
			}
			if got := req.Messages[1].Text(); got != conversationSummaryHeader+"SUMMARY 1" {
				t.Errorf("%s: first replayed summary = %q, want the recorded SUMMARY 1", name, got)
			}
			break
		}
	}
}

// A compaction completion is never an answer, whatever its outcome or purpose value.
func TestReplay_PurposeCompletionsAreNotServed(t *testing.T) {
	ctx := llmkit.WithRun(context.Background(), "purposes")
	tr := NewTranscript()
	tr.RunID = "purposes"
	add := func(kind llmkit.EventKind, fill func(*llmkit.Event)) {
		ev := llmkit.NewEvent(ctx, kind)
		fill(&ev)
		tr.Record = append(tr.Record, ev)
	}
	completion := func(purpose, text, errText string) func(*llmkit.Event) {
		return func(ev *llmkit.Event) {
			ev.Completion = &llmkit.CompletionEvent{Purpose: purpose, Err: errText, Response: llmkit.Response{Text: text}}
			if errText != "" {
				ev.Completion.ErrKind = llmkit.ErrKindContextTooLong
			}
		}
	}
	add(llmkit.KindCompletion, completion("compaction", "dropped: next call fails", "overflow"))
	add(llmkit.KindCompletion, completion("compaction", "half", ""))
	add(llmkit.KindCompletion, completion("compaction", "SUMMARY A", ""))
	add(llmkit.KindCompletion, completion("other", "not an answer", ""))
	add(llmkit.KindCompletion, completion("", "answer 1", ""))
	add(llmkit.KindCompletion, completion("compaction", "SUMMARY B", ""))
	add(llmkit.KindCompletion, completion("compaction", "SUMMARY LOST", "overflow"))
	add(llmkit.KindCompaction, func(ev *llmkit.Event) { ev.Compaction = &llmkit.CompactionEvent{} })
	add(llmkit.KindCompletion, completion("", "answer 2", ""))
	add(llmkit.KindCompletion, completion("compaction", "SUMMARY C", ""))
	rc, err := NewReplayClient(tr, "purposes", llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"answer 1", "answer 2"} {
		resp, err := rc.Complete(context.Background(), llmkit.Request{})
		if err != nil || resp.Text != want {
			t.Fatalf("Complete = %q, %v, want %q", resp.Text, err, want)
		}
	}
	var texts []string
	for _, rec := range rc.summaries {
		texts = append(texts, rec.text)
	}
	if !reflect.DeepEqual(texts, []string{"SUMMARY A", "SUMMARY C"}) {
		t.Errorf("summaries = %q, want a pass whose last call failed to leave none", texts)
	}
}

// budgetReplay replays rec with limits, a fresh pool of the same ceiling (when
// pool is non-nil) and rc.Compactor(), and returns what the replayed Run did.
func budgetReplay(t *testing.T, src llmkit.Source, rec *Outcome, limits Limits, pool *BudgetPool) (*Outcome, *ReplayClient) {
	t.Helper()
	rc, err := NewReplayClient(src, rec.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	out, err := NewRunner(rc, rc.Tools(), "sys",
		WithLimits(limits), WithCompactor(rc.Compactor()), WithBudgetPool(pool),
	).Run(context.Background(), "task", WithRunID("budget-replay"))
	var inc *IncompleteError
	if err != nil && !errors.As(err, &inc) {
		t.Fatalf("replayed Run: %v (rc.Err: %v)", err, rc.Err())
	}
	return out, rc
}

// A summarized run that stopped on a token budget or a pool replays with the
// recorded limits and rc.Compactor() to the same stop: the recorded summary
// spend counts against the budget in the replay as it did in the recording.
func TestSummarize_ReplayStopsWhereTheBudgetStoppedTheRecord(t *testing.T) {
	cases := []struct {
		name   string
		limits Limits
		pool   func() *BudgetPool
		reason TruncationReason
	}{
		{"token_budget", Limits{MaxIterations: 30, HistoryTokenBudget: 150, TokenBudget: 1000}, func() *BudgetPool { return nil }, TruncTokenBudget},
		{"budget_pool", Limits{MaxIterations: 30, HistoryTokenBudget: 150, TokenBudget: -1}, func() *BudgetPool { return NewBudgetPool(1000) }, TruncBudgetPool},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			sc := numbered()
			recPool := tc.pool()
			rec, _, err := summarizeRun(t, sc, recPool, seamTurns(20), nil, WithLimits(tc.limits),
				WithObserver(JSONL(dir, func(err error) { t.Errorf("sink: %v", err) })))
			var inc *IncompleteError
			if err != nil && !errors.As(err, &inc) {
				t.Fatalf("Run: %v", err)
			}
			if rec.TruncationReason != tc.reason {
				t.Fatalf("recorded stop = %q, want %q", rec.TruncationReason, tc.reason)
			}
			if len(summaryCompactions(rec.Transcript.Record)) == 0 || sc.calls() == 0 {
				t.Fatal("the recorded run summarized nothing; the row pins nothing")
			}
			for name, src := range map[string]llmkit.Source{"memory": rec.Transcript, "jsonl": JSONL(dir, nil)} {
				t.Run(name, func(t *testing.T) {
					replayPool := tc.pool()
					out, rc := budgetReplay(t, src, rec, tc.limits, replayPool)
					if err := rc.Err(); err != nil {
						t.Fatalf("rc.Err: %v", err)
					}
					if out.TruncationReason != tc.reason || out.Iterations != rec.Iterations {
						t.Errorf("replay stopped %q at iteration %d, recorded %q at %d",
							out.TruncationReason, out.Iterations, tc.reason, rec.Iterations)
					}
					if out.Usage != rec.Usage {
						t.Errorf("replayed Outcome.Usage %+v != recorded %+v", out.Usage, rec.Usage)
					}
					assertLedger(t, out)
					if replayPool != nil && replayPool.Spent() != recPool.Spent() {
						t.Errorf("replayed pool spent %d, recorded %d", replayPool.Spent(), recPool.Spent())
					}
				})
			}
		})
	}
}

// R-N: a summarizer used on the overflow recovery path is accounted like any
// other: its usage reaches Outcome.Usage, the pool and the ledger.
func TestSummarize_OverflowPathSummaryUsageIsRunSpend(t *testing.T) {
	gc := o1Client()
	sc := numbered()
	pool := NewBudgetPool(1 << 40)
	out, err := overflowRunner(gc, 1_000_000, WithCompactor(SummarizingCompactor(sc)), WithBudgetPool(pool)).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	comps := compactionEvents(out.Transcript.Record)
	if len(comps) != 1 || comps[0].Reason != llmkit.CompactionReasonOverflow || comps[0].Summary == "" || sc.calls() != 1 {
		t.Fatalf("compactions %+v, summarizer calls %d: want one overflow summary from one call", comps, sc.calls())
	}
	var ledger llmkit.Usage
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompletion {
			ledger = ledger.Add(ev.Completion.Response.Usage)
		}
	}
	want := llmkit.Usage{InputTokens: 7*100 + 200, OutputTokens: 7*10 + 30}
	if out.Usage != want || ledger != want {
		t.Errorf("Outcome.Usage %+v, ledger %+v, want %+v: seven answers and the summary call", out.Usage, ledger, want)
	}
	if got := out.Usage.ChargeableTokens(1); pool.Spent() != got {
		t.Errorf("pool charged %d, want %d", pool.Spent(), got)
	}
}

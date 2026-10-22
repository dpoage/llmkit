package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
)

// windowClient advertises window and serves turns tool turns of the "big"
// tool, then "done". It refuses a request estimating above limit (when set)
// with ErrContextTooLong, which consumes no turn.
type windowClient struct {
	mu      sync.Mutex
	window  int
	limit   int64
	turns   int
	served  int
	ests    []int64
	refused int
}

func (c *windowClient) Capabilities() llmkit.Capabilities {
	return llmkit.Capabilities{ContextWindow: c.window}
}

func (c *windowClient) Complete(_ context.Context, req llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	est := EstimateHistoryTokens(req.Messages)
	c.ests = append(c.ests, est)
	if c.limit > 0 && est > c.limit {
		c.refused++
		return llmkit.Response{}, tooLong()
	}
	c.served++
	if c.served > c.turns {
		return textResp("done", 100, 10).resp, nil
	}
	return toolResp(fmt.Sprintf("c%d", c.served), "big", `{"v":"x"}`, 100, 10).resp, nil
}

func (c *windowClient) maxEst() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	var m int64
	for _, e := range c.ests {
		m = max(m, e)
	}
	return m
}

func bigTool(bytes int) Tool { return bigEchoTool{name: "big", blob: strings.Repeat("x", bytes)} }

// compactionAt returns every Compaction event with the step it was emitted at.
func compactionAt(evs []llmkit.Event) (steps []int, comps []llmkit.CompactionEvent) {
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompaction {
			steps = append(steps, ev.Step)
			comps = append(comps, *ev.Compaction)
		}
	}
	return steps, comps
}

func countReason(comps []llmkit.CompactionEvent, reason string) int {
	n := 0
	for _, c := range comps {
		if c.Reason == reason {
			n++
		}
	}
	return n
}

// T1: with a 10,000-token window and a 2,000-token reserve, compaction fires
// when the request estimate first exceeds 8,000, not before.
func TestTrigger_ReserveFiresWhenEstimateExceedsWindowMinusReserve(t *testing.T) {
	wc := &windowClient{window: 10000, turns: 40}
	r := NewRunner(wc, []Tool{bigTool(1600)}, "sys", WithLimits(Limits{MaxIterations: 60, ContextReserve: 2000}))
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	steps, comps := compactionAt(out.Transcript.Record)
	if len(comps) == 0 {
		t.Fatalf("no compaction in a run whose request estimate reached %d", wc.maxEst())
	}
	if first := comps[0].BeforeTokens; first <= 8000 || first > 8600 {
		t.Errorf("first compaction saw %d tokens, want just above 8000", first)
	}
	for i, est := range wc.ests[:steps[0]-1] {
		if est > 8000 {
			t.Errorf("request %d sent at %d tokens before the first compaction", i+1, est)
		}
	}
}

// T2: an unknown window never fires the proactive trigger, and the reserve
// alone enables overflow recovery.
func TestTrigger_UnknownWindowOnlyRecoversFromOverflow(t *testing.T) {
	wc := &windowClient{window: 0, limit: 5000, turns: 30}
	r := NewRunner(wc, []Tool{bigTool(1600)}, "sys", WithLimits(Limits{MaxIterations: 60, ContextReserve: 2000}))
	out, err := r.Run(context.Background(), "task")
	if err != nil || out.FinalText != "done" {
		t.Fatalf("Run = %q, %v: want the run to recover and finish", out.FinalText, err)
	}
	_, comps := compactionAt(out.Transcript.Record)
	if n := countReason(comps, ""); n != 0 {
		t.Errorf("%d proactive compactions with an unknown window", n)
	}
	if n := countReason(comps, llmkit.CompactionReasonOverflow); n == 0 || wc.refused == 0 {
		t.Errorf("%d overflow recoveries after %d refusals, want recovery to run", n, wc.refused)
	}
}

// shrinkTo keeps the task and the last two messages and fills a summary so the
// history estimates target tokens.
func shrinkTo(target int64) Compactor {
	return CompactorFunc(func(_ context.Context, in CompactInput) (CompactOutput, error) {
		n := len(in.Messages)
		if n < 4 {
			return CompactOutput{}, nil
		}
		build := func(fill int64) []llmkit.Message {
			return []llmkit.Message{in.Messages[0], user(conversationSummaryHeader + strings.Repeat("s", int(fill))), in.Messages[n-2], in.Messages[n-1]}
		}
		return CompactOutput{Messages: build(max(0, (target-estimateTokens(build(0)))*4)), Pruned: n - 4}, nil
	})
}

// T3: a summarizer whose kept history already exceeds window - reserve is not
// called again until the estimate has grown by the reserve.
func TestTrigger_RearmWaitsForReserveGrowth(t *testing.T) {
	sc := numbered()
	fc := newFakeClient(append(bigTurns(12, 100, 10), textResp("done", 100, 10))...)
	fc.caps.ContextWindow = 10000
	r := NewRunner(fc, []Tool{bigTool(400)}, "sys", WithLimits(Limits{MaxIterations: 30, ContextReserve: 2000}),
		WithCompactor(SummarizingCompactor(sc)))
	out, err := r.Run(context.Background(), strings.Repeat("t", 36000))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	est := requestEstimates(fc)
	post := minAfterFirstSummary(t, out.Transcript.Record, est)
	allowed := 1 + int((maxOf(est)-post)/2000)
	if sc.calls() < 1 || sc.calls() > allowed {
		t.Errorf("summarizer called %d times over %d turns, want 1..%d (estimate %d..%d)", sc.calls(), len(est), allowed, post, maxOf(est))
	}
}

// T6: a summarizer whose summary is larger than its span is paid for, and
// re-arms the trigger as if it had compacted.
func TestTrigger_PaidDiscardedPassRearms(t *testing.T) {
	larger := func() *sumClient {
		return &sumClient{fn: func(int, llmkit.Request) (llmkit.Response, error) {
			return sumResp(strings.Repeat("long ", 4000)), nil
		}}
	}
	t.Run("history_budget_doubles", func(t *testing.T) {
		sc := larger()
		fc := newFakeClient(append(bigTurns(12, 100, 10), textResp("done", 100, 10))...)
		r := NewRunner(fc, []Tool{bigTool(400)}, "sys", WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 300}),
			WithCompactor(SummarizingCompactor(sc)))
		if _, err := r.Run(context.Background(), "task"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		windows := 0
		for th := int64(300); th < maxOf(requestEstimates(fc)); th *= 2 {
			windows++
		}
		if sc.calls() < 1 || sc.calls() > windows {
			t.Errorf("summarizer called %d times, want 1..%d (one per doubled threshold)", sc.calls(), windows)
		}
	})
	t.Run("reserve_waits_for_growth", func(t *testing.T) {
		sc := larger()
		fc := newFakeClient(append(bigTurns(12, 100, 10), textResp("done", 100, 10))...)
		fc.caps.ContextWindow = 10000
		r := NewRunner(fc, []Tool{bigTool(400)}, "sys", WithLimits(Limits{MaxIterations: 30, ContextReserve: 2000}),
			WithCompactor(SummarizingCompactor(sc)))
		if _, err := r.Run(context.Background(), strings.Repeat("t", 36000)); err != nil {
			t.Fatalf("Run: %v", err)
		}
		est := requestEstimates(fc)
		allowed := 1 + int((maxOf(est)-est[0])/2000)
		if sc.calls() < 1 || sc.calls() > allowed {
			t.Errorf("summarizer called %d times, want 1..%d", sc.calls(), allowed)
		}
	})
}

// T4: after an overflow recovery that leaves the history near the real limit,
// the next turns compact before the call instead of paying a refusal each.
func TestTrigger_OverflowRecoveryArmsTheNextTurn(t *testing.T) {
	wc := &windowClient{window: 10000, limit: 5000, turns: 20}
	r := NewRunner(wc, []Tool{bigTool(1600)}, "sys", WithLimits(Limits{MaxIterations: 60, ContextReserve: 2000}),
		WithCompactor(shrinkTo(4900)))
	out, err := r.Run(context.Background(), "task")
	if err != nil || out.FinalText != "done" {
		t.Fatalf("Run = %q, %v", out.FinalText, err)
	}
	_, comps := compactionAt(out.Transcript.Record)
	over, proactive := countReason(comps, llmkit.CompactionReasonOverflow), countReason(comps, "")
	if wc.refused != 1 || over != 1 || proactive < 3 {
		t.Errorf("%d refusals, %d overflow and %d proactive compactions: want one refusal, then a compaction before each call", wc.refused, over, proactive)
	}
}

// sizedTool returns a blob whose size is sizes[n] for its n-th call.
type sizedTool struct {
	sizes []int
	n     atomic.Int64
}

func (s *sizedTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "big", Description: "sized blob", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (s *sizedTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	i := int(s.n.Add(1)) - 1
	return textBlocks(strings.Repeat("x", s.sizes[min(i, len(s.sizes)-1)])), nil
}

func bigTurns(n int, in, out int64) []scriptStep {
	var steps []scriptStep
	for i := 1; i <= n; i++ {
		steps = append(steps, toolResp(fmt.Sprintf("c%d", i), "big", `{"v":"x"}`, in, out))
	}
	return steps
}

func requestEstimates(fc *fakeClient) []int64 {
	fc.mu.Lock()
	defer fc.mu.Unlock()
	var out []int64
	for _, req := range fc.requests {
		out = append(out, EstimateHistoryTokens(req.Messages))
	}
	return out
}

func maxOf(xs []int64) int64 {
	var m int64
	for _, x := range xs {
		m = max(m, x)
	}
	return m
}

// minAfterFirstSummary returns the smallest request estimate from the turn
// that follows the first summary on.
func minAfterFirstSummary(t *testing.T, evs []llmkit.Event, est []int64) int64 {
	t.Helper()
	steps, comps := compactionAt(evs)
	for i, c := range comps {
		if c.Summary != "" {
			m := est[steps[i]-1]
			for _, e := range est[steps[i]-1:] {
				m = min(m, e)
			}
			return m
		}
	}
	t.Fatal("the run made no summary")
	return 0
}

// T5: finalization after a budget stop compacts with the default stubber, so a
// summarizing compactor is never called and no summary is paid for.
func TestTrigger_FinalizeAfterBudgetStopIsStubOnly(t *testing.T) {
	sizes := []int{1000, 1000, 1000, 1000, 40000}
	schema := json.RawMessage(`{"type":"object"}`)
	script := func() []scriptStep {
		return append(bigTurns(5, 300, 100), textResp(`{"a":1}`, 300, 100))
	}
	wantUsage := llmkit.Usage{InputTokens: 6 * 300, OutputTokens: 6 * 100}
	cases := []struct {
		name   string
		limits Limits
		pool   *BudgetPool
		reason TruncationReason
	}{
		{"token_budget", Limits{MaxIterations: 30, HistoryTokenBudget: 6000, TokenBudget: 1900}, nil, TruncTokenBudget},
		{"budget_pool", Limits{MaxIterations: 30, HistoryTokenBudget: 6000, TokenBudget: -1}, NewBudgetPool(1900), TruncBudgetPool},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sc := numbered()
			fc := newFakeClient(script()...)
			opts := []Option{WithLimits(tc.limits), WithCompactor(SummarizingCompactor(sc)), WithBudgetPool(tc.pool)}
			var got map[string]any
			out, err := NewRunner(fc, []Tool{&sizedTool{sizes: sizes}}, "sys", opts...).RunJSON(context.Background(), "task", schema, &got)
			if err != nil || out.TruncationReason != tc.reason || !out.Finalized {
				t.Fatalf("RunJSON = %v, stop %q, finalized %v: want a finalized %s stop", err, out.TruncationReason, out.Finalized, tc.reason)
			}
			_, comps := compactionAt(out.Transcript.Record)
			if sc.calls() != 0 || len(comps) != 1 || comps[0].Summary != "" || comps[0].Pruned == 0 {
				t.Errorf("summarizer calls %d, compactions %+v: want one stub compaction and no summary", sc.calls(), comps)
			}
			if out.Usage != wantUsage {
				t.Errorf("Outcome.Usage = %+v, want %+v: compaction must add none", out.Usage, wantUsage)
			}
		})
	}
	t.Run("overflow_at_finalize", func(t *testing.T) {
		sc := numbered()
		gc := &gateClient{
			steps:    script(),
			overflow: func(req llmkit.Request) bool { return len(req.Tools) == 0 && toolResultBytes(req) > 4500 },
		}
		opts := []Option{
			WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 1_000_000, TokenBudget: 1900}),
			WithCompactor(SummarizingCompactor(sc)),
		}
		var got map[string]any
		out, err := NewRunner(gc, []Tool{&sizedTool{sizes: []int{1000}}}, "sys", opts...).RunJSON(context.Background(), "task", schema, &got)
		if err != nil || !out.Finalized {
			t.Fatalf("RunJSON = %v, finalized %v", err, out.Finalized)
		}
		_, comps := compactionAt(out.Transcript.Record)
		if gc.refused != 1 || sc.calls() != 0 || len(comps) != 1 || comps[0].Reason != llmkit.CompactionReasonOverflow || comps[0].Summary != "" {
			t.Errorf("%d refusals, summarizer calls %d, compactions %+v: want one stub recovery", gc.refused, sc.calls(), comps)
		}
		if out.Usage != wantUsage {
			t.Errorf("Outcome.Usage = %+v, want %+v", out.Usage, wantUsage)
		}
	})
}

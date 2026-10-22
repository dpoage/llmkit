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

// gateClient serves its scripted steps in order, but refuses with an error
// wrapping ErrContextTooLong whenever overflow reports the request too long.
// A refused call consumes no step.
type gateClient struct {
	mu       sync.Mutex
	steps    []scriptStep
	overflow func(req llmkit.Request) bool
	requests []llmkit.Request
	refused  int
}

func (g *gateClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (g *gateClient) Complete(_ context.Context, req llmkit.Request) (llmkit.Response, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.requests = append(g.requests, req)
	if g.overflow(req) {
		g.refused++
		return llmkit.Response{}, fmt.Errorf("provider: prompt is too long: %w", llmkit.ErrContextTooLong)
	}
	if len(g.steps) == 0 {
		return llmkit.Response{Text: "(unscripted)", StopReason: llmkit.StopEndTurn}, nil
	}
	step := g.steps[0]
	g.steps = g.steps[1:]
	return step.resp, step.err
}

func (g *gateClient) calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.requests)
}

func toolResultCount(req llmkit.Request) int {
	n := 0
	for _, m := range req.Messages {
		if m.Role == llmkit.RoleToolResult {
			n++
		}
	}
	return n
}

const overflowBlobBytes = 12000

// overflowTurns scripts n tool turns of bigEchoTool.
func overflowTurns(n int) []scriptStep {
	var steps []scriptStep
	for i := 1; i <= n; i++ {
		steps = append(steps, toolResp("c"+string(rune('0'+i)), "big", `{"v":"x"}`, 100, 10))
	}
	return steps
}

func overflowRunner(c llmkit.Client, budget int64, opts ...Option) *Runner {
	opts = append([]Option{WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: budget})}, opts...)
	return NewRunner(c, []Tool{bigEchoTool{name: "big", blob: strings.Repeat("x", overflowBlobBytes)}}, "sys", opts...)
}

func compactionEvents(evs []llmkit.Event) []llmkit.CompactionEvent {
	var out []llmkit.CompactionEvent
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompaction {
			out = append(out, *ev.Compaction)
		}
	}
	return out
}

func failedCompletions(evs []llmkit.Event) []llmkit.CompletionEvent {
	var out []llmkit.CompletionEvent
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompletion && ev.Completion.Err != "" {
			out = append(out, *ev.Completion)
		}
	}
	return out
}

// o1Client overflows the 7th request while six tool results ride it unpruned,
// and serves any request in which compaction shrank them.
func o1Client() *gateClient {
	return &gateClient{
		steps:    append(overflowTurns(6), textResp("done", 100, 10)),
		overflow: func(req llmkit.Request) bool { return toolResultBytes(req) > 5*overflowBlobBytes },
	}
}

func TestOverflow_CompactsAndRetriesOnce(t *testing.T) {
	gc := o1Client()
	var hookSteps []int
	var policySteps []int
	r := overflowRunner(gc, 1_000_000,
		WithHooks(Hooks{BeforeCompletion: func(_ context.Context, step int, _ *llmkit.Request) { hookSteps = append(hookSteps, step) }}),
		WithRequestPolicy(RequestPolicyFunc(func(_ context.Context, step int, _ *llmkit.Request) error {
			policySteps = append(policySteps, step)
			return nil
		})))
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}
	if gc.calls() != 8 || gc.refused != 1 {
		t.Fatalf("calls = %d refused = %d, want 8 and 1", gc.calls(), gc.refused)
	}
	failed, retry := gc.requests[6], gc.requests[7]
	if toolResultBytes(retry) >= toolResultBytes(failed) {
		t.Errorf("retry carries %d tool-result bytes, refused request %d", toolResultBytes(retry), toolResultBytes(failed))
	}
	comps := compactionEvents(out.Transcript.Record)
	if len(comps) != 1 || comps[0].Reason != llmkit.CompactionReasonOverflow || comps[0].Pruned != 2 {
		t.Fatalf("compaction events = %+v, want one with reason overflow pruning 2", comps)
	}
	if comps[0].AfterTokens >= comps[0].BeforeTokens {
		t.Errorf("compaction did not shrink: %+v", comps[0])
	}
	fails := failedCompletions(out.Transcript.Record)
	if len(fails) != 1 || fails[0].ErrKind != llmkit.ErrKindContextTooLong {
		t.Fatalf("failed completions = %+v, want one with err_kind context_too_long", fails)
	}
	// The retry is a second attempt of the same step: hooks and policy fire for each.
	if want := []int{1, 2, 3, 4, 5, 6, 7, 7}; !reflect.DeepEqual(hookSteps, want) || !reflect.DeepEqual(policySteps, want) {
		t.Errorf("hook steps %v, policy steps %v, want %v", hookSteps, policySteps, want)
	}
	// The compacted history is the run's history from here on.
	stubs := 0
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleToolResult && isCompactStub(m.Text()) {
			stubs++
		}
	}
	if stubs != 2 {
		t.Errorf("Outcome.Messages holds %d stubs, want 2", stubs)
	}
}

func TestOverflow_SecondOverflowReturnsErrorAfterTwoCalls(t *testing.T) {
	gc := &gateClient{
		steps:    overflowTurns(6),
		overflow: func(req llmkit.Request) bool { return toolResultCount(req) >= 6 },
	}
	out, err := overflowRunner(gc, 1_000_000).Run(context.Background(), "task")
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("err = %v, want ErrContextTooLong", err)
	}
	if gc.calls() != 8 || gc.refused != 2 {
		t.Errorf("calls = %d refused = %d, want 8 and 2 (two attempts for the invocation)", gc.calls(), gc.refused)
	}
	if n := len(compactionEvents(out.Transcript.Record)); n != 1 {
		t.Errorf("compaction events = %d, want 1", n)
	}
	var tr *IncompleteError
	if errors.As(err, &tr) || out.Finalized {
		t.Errorf("overflow must not end as a truncation or finalize turn: %v finalized=%v", err, out.Finalized)
	}
}

func TestOverflow_NothingPrunableIsOneCall(t *testing.T) {
	gc := &gateClient{
		steps:    overflowTurns(3),
		overflow: func(req llmkit.Request) bool { return toolResultCount(req) >= 3 },
	}
	out, err := overflowRunner(gc, 1_000_000).Run(context.Background(), "task")
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("err = %v, want ErrContextTooLong", err)
	}
	if gc.calls() != 4 || gc.refused != 1 {
		t.Errorf("calls = %d refused = %d, want 4 and 1 (no retry)", gc.calls(), gc.refused)
	}
	if n := len(compactionEvents(out.Transcript.Record)); n != 0 {
		t.Errorf("compaction events = %d, want 0", n)
	}
	if fails := failedCompletions(out.Transcript.Record); len(fails) != 1 || fails[0].ErrKind != llmkit.ErrKindContextTooLong {
		t.Errorf("failed completions = %+v, want one tagged context_too_long", fails)
	}
}

func TestOverflow_ContinuationRecovers(t *testing.T) {
	gc := &gateClient{
		steps: append(overflowTurns(6),
			maxTokensResp("part one", 100, 10),
			textResp("part two", 100, 10)),
		overflow: func(req llmkit.Request) bool {
			last := req.Messages[len(req.Messages)-1]
			return last.Role == llmkit.RoleUser && strings.Contains(last.Text(), "cut off") &&
				toolResultBytes(req) > 5*overflowBlobBytes
		},
	}
	out, err := overflowRunner(gc, 1_000_000).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if gc.refused != 1 || gc.calls() != 9 {
		t.Errorf("calls = %d refused = %d, want 9 and 1", gc.calls(), gc.refused)
	}
	if !strings.Contains(out.FinalText, "part one") || !strings.Contains(out.FinalText, "part two") {
		t.Errorf("FinalText = %q, want both halves stitched", out.FinalText)
	}
	if comps := compactionEvents(out.Transcript.Record); len(comps) != 1 || comps[0].Reason != llmkit.CompactionReasonOverflow {
		t.Errorf("compaction events = %+v, want one overflow", comps)
	}
}

func TestOverflow_DisabledWithoutHistoryBudget(t *testing.T) {
	for _, budget := range []int64{0, -1} {
		gc := o1Client()
		out, err := overflowRunner(gc, budget).Run(context.Background(), "task")
		if !errors.Is(err, llmkit.ErrContextTooLong) {
			t.Fatalf("budget %d: err = %v, want ErrContextTooLong", budget, err)
		}
		if gc.calls() != 7 || gc.refused != 1 {
			t.Errorf("budget %d: calls = %d refused = %d, want 7 and 1", budget, gc.calls(), gc.refused)
		}
		if n := len(compactionEvents(out.Transcript.Record)); n != 0 {
			t.Errorf("budget %d: compaction events = %d, want 0", budget, n)
		}
		for _, m := range out.Messages {
			if isCompactStub(m.Text()) {
				t.Errorf("budget %d: history holds a stub", budget)
			}
		}
	}
}

// replayEvents is what a replay must reproduce: the completions, request
// history included, and the compactions.
func replayEvents(evs []llmkit.Event) []any {
	var out []any
	for _, ev := range evs {
		switch ev.Kind {
		case llmkit.KindCompletion:
			c := *ev.Completion
			c.Request.Tools = nil
			out = append(out, c)
		case llmkit.KindCompaction:
			out = append(out, *ev.Compaction)
		}
	}
	return out
}

func replayOverflow(t *testing.T, src llmkit.Source, run llmkit.RunID) ([]llmkit.Event, *ReplayClient) {
	t.Helper()
	rc, err := NewReplayClient(src, run, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	out, err := overflowRunner(rc, 1_000_000).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replayed Run: %v", err)
	}
	if err := rc.Err(); err != nil {
		t.Fatalf("rc.Err: %v", err)
	}
	return out.Transcript.Record, rc
}

func TestOverflow_ReplayReproducesRecord(t *testing.T) {
	dir := t.TempDir()
	gc := o1Client()
	out, err := overflowRunner(gc, 1_000_000, WithObserver(JSONL(dir, func(err error) { t.Errorf("sink: %v", err) }))).
		Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := replayEvents(out.Transcript.Record)
	for name, src := range map[string]llmkit.Source{"memory": out.Transcript, "jsonl": JSONL(dir, nil)} {
		t.Run(name, func(t *testing.T) {
			got, _ := replayOverflow(t, src, out.RunID)
			if g := replayEvents(got); !reflect.DeepEqual(g, want) {
				gj, _ := json.Marshal(g)
				wj, _ := json.Marshal(want)
				t.Fatalf("replayed events differ\n got: %s\nwant: %s", gj, wj)
			}
			if fails := failedCompletions(got); len(fails) != 1 || fails[0].ErrKind != llmkit.ErrKindContextTooLong {
				t.Errorf("replayed failed completions = %+v", fails)
			}
		})
	}
}

func TestOverflow_ReplayWithoutErrKindSkipsFailure(t *testing.T) {
	gc := o1Client()
	out, err := overflowRunner(gc, 1_000_000).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	old := make([]llmkit.Event, len(out.Transcript.Record))
	copy(old, out.Transcript.Record)
	for i, ev := range old {
		if ev.Kind == llmkit.KindCompletion && ev.Completion.ErrKind != "" {
			c := *ev.Completion
			c.ErrKind = ""
			old[i].Completion = &c
		}
	}
	src := &Transcript{RunID: out.RunID, Record: old}
	rc, err := NewReplayClient(src, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	// Without err_kind the failed completion is skipped, so the 7th request is
	// served the recorded success: no overflow reaches the Runner, none recompacts.
	got, err := overflowRunner(rc, 1_000_000).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if n := len(compactionEvents(got.Transcript.Record)); n != 0 {
		t.Errorf("compaction events = %d, want 0 for a record without err_kind", n)
	}
	if n := len(failedCompletions(got.Transcript.Record)); n != 0 {
		t.Errorf("failed completions = %d, want 0", n)
	}
	if e := rc.Err(); e != nil {
		t.Errorf("rc.Err = %v", e)
	}
}

func TestOverflow_ReplayServesWrappedError(t *testing.T) {
	gc := &gateClient{steps: overflowTurns(1), overflow: func(req llmkit.Request) bool { return toolResultCount(req) >= 1 }}
	out, err := overflowRunner(gc, 0).Run(context.Background(), "task")
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("err = %v", err)
	}
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	_, rerr := overflowRunner(rc, 0).Run(context.Background(), "task")
	if !errors.Is(rerr, llmkit.ErrContextTooLong) {
		t.Fatalf("replayed err = %v, want ErrContextTooLong", rerr)
	}
	if e := rc.Err(); e != nil {
		t.Errorf("rc.Err = %v", e)
	}
}

func TestOverflow_NewFieldsAbsentFromOldRecordBytes(t *testing.T) {
	cases := map[string]any{
		"threshold compaction": llmkit.CompactionEvent{BeforeTokens: 9, AfterTokens: 4, Pruned: 1},
		"plain failure":        llmkit.CompletionEvent{Err: "boom"},
	}
	for name, v := range cases {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), `"reason"`) || strings.Contains(string(b), `"err_kind"`) {
			t.Errorf("%s: %s carries a new field", name, b)
		}
	}
}

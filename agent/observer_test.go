package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// --- scripted client + tools for the event-kind coverage run ---------------

// scriptedToolClient answers each completion with a scripted response and
// records the RunID its context carried.
type runIDClient struct {
	fakeClient
	runIDs []llmkit.RunID
	mu     sync.Mutex
}

func (c *runIDClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	c.runIDs = append(c.runIDs, llmkit.RunFromContext(ctx))
	c.mu.Unlock()
	return c.fakeClient.Complete(ctx, req)
}

// ctxTool records the RunID its Run context carried.
type ctxTool struct {
	name    string
	mu      sync.Mutex
	runIDs  []llmkit.RunID
	fail    bool // every call fails
	failN   int  // the first failN calls fail (0 = never)
	calls   int
	payload int // size of the success result in bytes
}

func (t *ctxTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "records run id", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t *ctxTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	t.mu.Lock()
	t.runIDs = append(t.runIDs, llmkit.RunFromContext(ctx))
	t.calls++
	fail := t.fail || (t.failN > 0 && t.calls <= t.failN)
	t.mu.Unlock()
	if fail {
		return "", errors.New("boom")
	}
	return strings.Repeat("x", t.payload), nil
}

// stepRecorder collects the Step every hook family reported, joining events
// to hooks.
type stepRecorder struct {
	mu          sync.Mutex
	completions []int
	tools       []int
	compactions []int
}

func (s *stepRecorder) hooks() Hooks {
	return Hooks{
		BeforeCompletion: func(_ context.Context, step int, _ *llmkit.Request) {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.completions = append(s.completions, step)
		},
		ToolStart: func(_ context.Context, ev ToolEvent) {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.tools = append(s.tools, ev.Step)
		},
		Compaction: func(_ context.Context, ev CompactionEvent) {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.compactions = append(s.compactions, ev.Step)
		},
	}
}

// TestObserver_OneEventOfEachKindAtHookSteps drives ONE run through every
// Runner-emitted kind — start, completion, tool_run (a policy denial AND a
// tool error), compaction, steer, finalize (a truncation stop) — and pins
// that each kind appears exactly the contract's number of times, at the same
// Step the hooks reported.
func TestObserver_OneEventOfEachKindAtHookSteps(t *testing.T) {
	denyMe := &ctxTool{name: "deny_me"}
	errTool := &ctxTool{name: "err_tool", fail: true}
	big := &ctxTool{name: "big", payload: 2000}

	fc := newFakeClient(
		toolCallsResp(
			llmkit.ToolCall{ID: "c1", Name: "big", Arguments: json.RawMessage(`{}`)},
			llmkit.ToolCall{ID: "c2", Name: "err_tool", Arguments: json.RawMessage(`{}`)},
			llmkit.ToolCall{ID: "c3", Name: "deny_me", Arguments: json.RawMessage(`{}`)},
		),
		toolResp("c4", "big", `{}`, 5, 5),
		toolResp("c5", "big", `{}`, 5, 5),
		toolResp("c6", "big", `{}`, 5, 5),
		toolResp("c7", "big", `{}`, 5, 5),
		toolResp("c8", "big", `{}`, 5, 5),
	)

	rec := &stepRecorder{}
	policy := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if call.Name == "deny_me" {
			return errors.New("not allowed")
		}
		return nil
	})

	// Steer delivered before the first completion.
	steering := NewSteering()
	steering.Steer(llmkit.Text("focus please"))

	r := NewRunner(fc, []Tool{big, errTool, denyMe}, "sys",
		WithHooks(rec.hooks()),
		WithToolPolicy(policy),
		WithLimits(Limits{MaxIterations: 6, HistoryTokenBudget: 800}),
	)

	out, err := r.Run(context.Background(), "exercise every kind", WithSteering(steering))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !out.Truncated() || out.TruncationReason != TruncMaxIterations {
		t.Fatalf("TruncationReason = %q, want %q", out.TruncationReason, TruncMaxIterations)
	}
	if out.Iterations != 6 {
		t.Fatalf("Iterations = %d, want 6", out.Iterations)
	}

	evs := out.Transcript.Record
	count := func(kind llmkit.EventKind) int {
		n := 0
		for _, ev := range evs {
			if ev.Kind == kind {
				n++
			}
		}
		return n
	}

	// Kind counts: one start, one completion per turn, two tool runs in turn 1
	// (denial + error) plus one per later turn, exactly one compaction (the
	// re-armed threshold then finds nothing left to reclaim within the run),
	// one steer, one finalize.
	if n := count(llmkit.KindStart); n != 1 {
		t.Errorf("start events = %d, want 1", n)
	}
	if n := count(llmkit.KindCompletion); n != 6 {
		t.Errorf("completion events = %d, want 6", n)
	}
	if n := count(llmkit.KindToolRun); n != 8 {
		t.Errorf("tool_run events = %d, want 8 (3 in turn 1 + 1 per turn 2..6)", n)
	}
	if n := count(llmkit.KindCompaction); n != 1 {
		t.Errorf("compaction events = %d, want 1", n)
	}
	if n := count(llmkit.KindSteer); n != 1 {
		t.Errorf("steer events = %d, want 1", n)
	}
	if n := count(llmkit.KindFinalize); n != 1 {
		t.Errorf("finalize events = %d, want 1", n)
	}

	// Start: Step 0, task and tool names, no parent.
	start := evs[0]
	if start.Kind != llmkit.KindStart || start.Step != 0 {
		t.Errorf("first event = %s step %d, want start at step 0", start.Kind, start.Step)
	}
	if start.Start == nil || start.Start.Task != "exercise every kind" {
		t.Errorf("start payload = %+v", start.Start)
	}
	if got := start.Start.Tools; !reflect.DeepEqual(got, []string{"big", "err_tool", "deny_me"}) {
		t.Errorf("start tools = %v, want [big err_tool deny_me]", got)
	}
	if start.ParentRunID != "" {
		t.Errorf("start ParentRunID = %q, want empty on a fresh run", start.ParentRunID)
	}

	// Completions: steps 1..6, matching BeforeCompletion hook steps.
	var completionSteps []int
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompletion {
			completionSteps = append(completionSteps, ev.Step)
		}
	}
	rec.mu.Lock()
	wantSteps := slices.Clone(rec.completions)
	toolSteps := slices.Clone(rec.tools)
	compactSteps := slices.Clone(rec.compactions)
	rec.mu.Unlock()
	if !reflect.DeepEqual(completionSteps, wantSteps) {
		t.Errorf("completion steps %v != hook steps %v", completionSteps, wantSteps)
	}

	// Steer: delivered before the first completion, so Step 1, no follow-up.
	var steerEv *llmkit.Event
	for i := range evs {
		if evs[i].Kind == llmkit.KindSteer {
			steerEv = &evs[i]
		}
	}
	if steerEv.Step != wantSteps[0] {
		t.Errorf("steer step = %d, want the first completion's step %d", steerEv.Step, wantSteps[0])
	}
	if steerEv.Steer.FollowUp || steerEv.Steer.Message.Text() != "focus please" {
		t.Errorf("steer payload = %+v", steerEv.Steer)
	}

	// Tool runs: the turn-1 denial carries Denied + DenyReason and NO result;
	// the turn-1 error carries IsError and the verbatim model-visible text;
	// every ToolRun's Call is the model's original call.
	var denials, errs int
	for _, ev := range evs {
		if ev.Kind != llmkit.KindToolRun || ev.ToolRun == nil {
			continue
		}
		if ev.ToolRun.Denied {
			denials++
			if ev.ToolRun.DenyReason != "not allowed" || ev.ToolRun.Result != "" {
				t.Errorf("denial payload = %+v", ev.ToolRun)
			}
		}
		if ev.ToolRun.IsError {
			errs++
			if ev.ToolRun.Result != "ERROR: boom" {
				t.Errorf("error result = %q", ev.ToolRun.Result)
			}
		}
		if !slices.Contains(toolSteps, ev.Step) && ev.Step != 0 {
			t.Errorf("tool_run step %d not reported by ToolStart hooks %v", ev.Step, toolSteps)
		}
	}
	if denials != 1 || errs != 1 {
		t.Errorf("denials = %d errors = %d, want 1 and 1", denials, errs)
	}

	// Compaction: the Step of the completion that consumed the compacted
	// history, exactly what Hooks.Compaction reported.
	for _, ev := range evs {
		if ev.Kind != llmkit.KindCompaction {
			continue
		}
		if !slices.Contains(compactSteps, ev.Step) {
			t.Errorf("compaction step %d not reported by hooks %v", ev.Step, compactSteps)
		}
		if ev.Compaction.Pruned == 0 {
			t.Errorf("compaction pruned 0 messages; the event must fire only on a real prune")
		}
	}

	// Finalize: last turn, the outcome's accounting, the truncation reason.
	final := evs[len(evs)-1]
	if final.Kind != llmkit.KindFinalize {
		t.Fatalf("last event = %s, want finalize", final.Kind)
	}
	if final.Step != out.Iterations {
		t.Errorf("finalize step = %d, want the last turn %d", final.Step, out.Iterations)
	}
	if final.Finalize.TruncationReason != string(TruncMaxIterations) ||
		final.Finalize.Iterations != out.Iterations ||
		final.Finalize.Usage != out.Usage ||
		final.Finalize.Finalized {
		t.Errorf("finalize payload = %+v, want the outcome's accounting", final.Finalize)
	}
}

// TestObserver_RunIDVisibleFromContext pins 4qh.1's hermetic acceptance: the
// run id is visible from ctx inside Tool.Run, a ToolPolicy, a RequestPolicy,
// and the client's Complete.
func TestObserver_RunIDVisibleFromContext(t *testing.T) {
	tool := &ctxTool{name: "probe"}
	policyRuns := []llmkit.RunID{}
	reqPolicyRuns := []llmkit.RunID{}

	cl := &runIDClient{fakeClient: *newFakeClient(toolResp("c1", "probe", `{}`, 1, 1), textResp("done", 1, 1))}
	r := NewRunner(cl, []Tool{tool}, "sys",
		WithToolPolicy(ToolPolicyFunc(func(ctx context.Context, _ *llmkit.ToolCall) error {
			policyRuns = append(policyRuns, llmkit.RunFromContext(ctx))
			return nil
		})),
		WithRequestPolicy(RequestPolicyFunc(func(ctx context.Context, _ int, _ *llmkit.Request) error {
			reqPolicyRuns = append(reqPolicyRuns, llmkit.RunFromContext(ctx))
			return nil
		})),
	)

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	runID := out.RunID
	if runID == "" {
		t.Fatal("Outcome.RunID empty; the Runner must mint one per run")
	}
	if len(cl.runIDs) == 0 || cl.runIDs[0] != runID {
		t.Errorf("client saw run ids %v, want [%s]", cl.runIDs, runID)
	}
	if len(tool.runIDs) == 0 || tool.runIDs[0] != runID {
		t.Errorf("tool saw run ids %v, want [%s]", tool.runIDs, runID)
	}
	if len(policyRuns) == 0 || policyRuns[0] != runID {
		t.Errorf("tool policy saw run ids %v, want [%s]", policyRuns, runID)
	}
	if len(reqPolicyRuns) == 0 || reqPolicyRuns[0] != runID {
		t.Errorf("request policy saw run ids %v, want [%s]", reqPolicyRuns, runID)
	}
}

// TestObserver_WithObserverLastWins pins the single-durable-sink ruling: a
// second WithObserver REPLACES the first (last-wins), never stacks a second
// history.
func TestObserver_WithObserverLastWins(t *testing.T) {
	dir1, dir2 := t.TempDir(), t.TempDir()
	r := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys",
		WithObserver(JSONL(dir1, nil)),
		WithObserver(JSONL(dir2, nil)))

	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for _, tc := range []struct {
		dir      string
		wantFile bool
	}{
		{dir1, false},
		{dir2, true},
	} {
		entries, err := os.ReadDir(tc.dir)
		if err != nil {
			t.Fatalf("ReadDir %s: %v", tc.dir, err)
		}
		if tc.wantFile && len(entries) != 1 {
			t.Errorf("winning sink dir %s has %d entries, want 1", tc.dir, len(entries))
		}
		if !tc.wantFile && len(entries) != 0 {
			t.Errorf("replaced sink dir %s has %d entries, want 0 (no second history)", tc.dir, len(entries))
		}
	}
}

// TestObserver_ReplayIsFullyHermetic records a two-tool run, then replays it
// through NewReplayClient + its bound Tools with ZERO live tool executions: same
// FinalText, and a DeepEqual event stream modulo Time, Duration, SpanID, and
// RunID.
func TestObserver_ReplayIsFullyHermetic(t *testing.T) {
	liveCalls := 0
	tool := &countingTool{name: "echo", count: &liveCalls, result: func() string { return "echoed" }}
	second := &countingTool{name: "second", count: &liveCalls, result: func() string { return "second ran" }}

	fc := newFakeClient(
		toolResp("c1", "echo", `{"text":"hi"}`, 10, 4),
		toolResp("c2", "second", `{}`, 10, 4),
		textResp("replay me to the same answer", 8, 3),
	)
	rec := NewRunner(fc, []Tool{tool, second}, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	if liveCalls != 2 {
		t.Fatalf("record run executed %d tools, want 2", liveCalls)
	}

	src := out.Transcript
	replayClient, err := NewReplayClient(src, src.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	replayTools := replayClient.Tools()
	if len(replayTools) != 2 {
		t.Fatalf("ReplayClient.Tools returned %d tools, want 2", len(replayTools))
	}

	liveCalls = 0
	replayer := NewRunner(replayClient, replayTools, "sys")
	replayed, err := replayer.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay run: %v", err)
	}
	if replayClient.Err() != nil {
		t.Errorf("replay diverged: %v", replayClient.Err())
	}
	if liveCalls != 0 {
		t.Errorf("replay executed %d live tools, want 0", liveCalls)
	}
	if replayed.FinalText != out.FinalText {
		t.Errorf("replay FinalText = %q, want %q", replayed.FinalText, out.FinalText)
	}

	strip := func(tr *Transcript) []llmkit.Event {
		evs := slices.Clone(tr.Record)
		for i := range evs {
			evs[i].Time = time.Time{}
			evs[i].Duration = 0
			evs[i].SpanID = ""
			evs[i].RunID = ""
			if evs[i].Completion != nil {
				// Replay tool schemas are RECONSTRUCTED from the record
				// (replay tools are reconstructed, names only), so the offered tool set
				// differs by construction; the model-visible conversation
				// is what replay preserves.
				evs[i].Completion.Request.Tools = nil
			}
		}
		return evs
	}
	if !reflect.DeepEqual(strip(out.Transcript), strip(replayed.Transcript)) {
		t.Errorf("event streams differ modulo Time/Duration/SpanID/RunID:\nrecord: %+v\nreplay: %+v",
			strip(out.Transcript), strip(replayed.Transcript))
	}
}

// TestObserver_ReplayDivergenceNamesStep pins the divergence contract: a
// call that matches nothing records ErrReplayDiverged naming the recorded
// step on the client, and the harness renders it as the call's error result.
func TestObserver_ReplayDivergenceNamesStep(t *testing.T) {
	tool := echoTool{name: "echo"}
	fc := newFakeClient(
		toolResp("c1", "echo", `{"text":"hi"}`, 10, 4),
		textResp("final", 8, 3),
	)
	rec := NewRunner(fc, []Tool{tool}, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	tools := rc.Tools()

	// A call matching nothing (the recorded step-1 call had {"text":"hi"}).
	_, err = tools[0].Run(context.Background(), json.RawMessage(`{"text":"something else"}`))
	if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "replay diverged at step 1") {
		t.Fatalf("tool err = %v, want ErrReplayDiverged naming step 1", err)
	}
	if !errors.Is(rc.Err(), ErrReplayDiverged) {
		t.Fatalf("rc.Err() = %v, want the recorded divergence", rc.Err())
	}
}

// TestObserver_ReplayDivergenceMaskingShapes pins the two shapes where a
// divergence in the run's FINAL tool turn ends the run before another
// Complete: a clean finish and a cap-truncated recording both return Run
// nil, so the caller's rc.Err() assertion is the only thing that catches
// them. The mid-run shape (divergence with completions still to serve)
// fails Run with ErrReplayDiverged. The rewriter policy forces the tool
// mismatch deterministically.
func TestObserver_ReplayDivergenceMaskingShapes(t *testing.T) {
	rewrite := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		call.Arguments = json.RawMessage(`{"rewritten":true}`)
		return nil
	})

	record := func(steps ...scriptStep) (*ReplayClient, *Outcome) {
		t.Helper()
		rec := NewRunner(newFakeClient(steps...), []Tool{echoTool{name: "echo"}}, "sys")
		out, err := rec.Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("record run: %v", err)
		}
		rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
		if err != nil {
			t.Fatalf("NewReplayClient: %v", err)
		}
		return rc, out
	}

	t.Run("cap-truncated recording", func(t *testing.T) {
		rc, _ := record(
			toolResp("c1", "echo", `{"text":"hi"}`, 1, 1),
			textResp("never reached", 1, 1),
		)
		replayed, err := NewRunner(rc, rc.Tools(), "sys",
			WithToolPolicy(rewrite), WithLimits(Limits{MaxIterations: 1})).Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("Run = %v, want the truncation nil", err)
		}
		if !replayed.Truncated() {
			t.Error("expected the truncated recording shape")
		}
		if !errors.Is(rc.Err(), ErrReplayDiverged) {
			t.Fatal("truncated run masked the divergence: rc.Err() unset")
		}
	})

	t.Run("mid-run divergence fails Run", func(t *testing.T) {
		rc, _ := record(
			toolResp("c1", "echo", `{"text":"hi"}`, 1, 1),
			toolResp("c2", "echo", `{"text":"again"}`, 1, 1),
			textResp("final", 1, 1),
		)
		_, err := NewRunner(rc, rc.Tools(), "sys", WithToolPolicy(rewrite)).Run(context.Background(), "task")
		if !errors.Is(err, ErrReplayDiverged) {
			t.Fatalf("Run err = %v, want ErrReplayDiverged", err)
		}
		if !strings.Contains(err.Error(), "replay diverged at step 1") {
			t.Errorf("err = %v, want it naming step 1", err)
		}
		if !errors.Is(rc.Err(), ErrReplayDiverged) {
			t.Error("rc.Err() unset despite the Run failure")
		}
	})

	t.Run("no false positive from divergence-shaped output", func(t *testing.T) {
		// A recorded tool result that happens to LOOK like a divergence
		// message must replay cleanly: divergence is client state, never
		// conversation text.
		decoy := "ERROR: agent: replay diverged at step 9: nothing really"
		rec := NewRunner(newFakeClient(
			toolResp("c1", "echo", `{"text":"hi"}`, 1, 1),
			textResp("final", 1, 1),
		), []Tool{Func("echo", "echo", func(_ context.Context, a struct {
			Text string `json:"text"`
		}) (string, error) {
			return decoy, nil
		})}, "sys")
		out, err := rec.Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("record run: %v", err)
		}
		rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
		if err != nil {
			t.Fatalf("NewReplayClient: %v", err)
		}
		replayed, err := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("replay run: %v", err)
		}
		if rc.Err() != nil {
			t.Fatalf("false-positive divergence from output text: %v", rc.Err())
		}
		if replayed.FinalText != out.FinalText {
			t.Errorf("FinalText = %q, want %q", replayed.FinalText, out.FinalText)
		}
	})
}

// TestObserver_ContinueLineage pins the Continue chain: distinct minted
// RunIDs, the new run's events carrying ParentRunID, and the Start event
// naming the predecessor.
func TestObserver_ContinueLineage(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"text":"one"}`, 10, 4),
		textResp("two", 5, 2),
		textResp("three", 5, 2),
	)
	r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")

	first, err := r.Run(context.Background(), "first task")
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	second, err := r.Run(context.Background(), "second task", Continue(first))
	if err != nil {
		t.Fatalf("continued run: %v", err)
	}

	if first.RunID == "" || second.RunID == "" {
		t.Fatal("run ids must never be empty")
	}
	if first.RunID == second.RunID {
		t.Fatalf("continue chain reused run id %q", first.RunID)
	}
	if second.Transcript.ParentRunID != first.RunID {
		t.Errorf("continued transcript ParentRunID = %q, want %q", second.Transcript.ParentRunID, first.RunID)
	}
	var start *llmkit.Event
	for i := range second.Transcript.Record {
		if second.Transcript.Record[i].Kind == llmkit.KindStart {
			start = &second.Transcript.Record[i]
			break
		}
	}
	if start == nil {
		t.Fatal("continued run recorded no start event")
	}
	if start.RunID != second.RunID || start.ParentRunID != first.RunID {
		t.Errorf("continued start = run %q parent %q, want run %q parent %q",
			start.RunID, start.ParentRunID, second.RunID, first.RunID)
	}
	// EVERY event of the continued run names the predecessor.
	for _, ev := range second.Transcript.Record {
		if ev.ParentRunID != first.RunID {
			t.Errorf("event kind %s carries ParentRunID %q, want %q", ev.Kind, ev.ParentRunID, first.RunID)
		}
	}
}

// TestObserver_ConcurrentRunsDistinctRunIDs extends the -race guard: N
// concurrent runs on ONE Runner never share a minted RunID (and so never
// collide on a JSONL filename either).
func TestObserver_ConcurrentRunsDistinctRunIDs(t *testing.T) {
	client := &alwaysToolClient{}
	r := NewRunner(client, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{MaxIterations: 1}))

	const n = 10
	ids := make([]llmkit.RunID, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			out, err := r.Run(context.Background(), "concurrent task")
			if err != nil {
				t.Errorf("Run %d: %v", i, err)
				return
			}
			ids[i] = out.RunID
		}(i)
	}
	wg.Wait()
	seen := map[llmkit.RunID]bool{}
	for i, id := range ids {
		if id == "" {
			t.Fatalf("run %d has an empty RunID", i)
		}
		if seen[id] {
			t.Fatalf("run id %q reused across concurrent runs", id)
		}
		seen[id] = true
	}
}

// countingTool is a tool that counts its executions (to prove replay runs
// zero live executions) and returns a fixed result.
type countingTool struct {
	name   string
	count  *int
	result func() string
	mu     sync.Mutex
}

func (t *countingTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "counting", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t *countingTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	*t.count++
	return t.result(), nil
}

// errClient fails every Complete with a partial response still in hand.
type errClient struct{ partial llmkit.Response }

func (c errClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c errClient) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	return c.partial, errors.New("connection reset")
}

// blockingTool blocks until its context is done.
type blockingTool struct{ started chan struct{} }

func (b blockingTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "block", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (b blockingTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	close(b.started)
	<-ctx.Done()
	return "", ctx.Err()
}

// TestObserver_FinalizeOnErrorReturns pins that Finalize is emitted on every
// ERROR return — client failure, context cancellation mid-tool, and a
// RequestPolicy abort — with the step of the COMPLETED turns (a failed
// completion does not advance it).
func TestObserver_FinalizeOnErrorReturns(t *testing.T) {
	t.Run("client error on the first completion", func(t *testing.T) {
		r := NewRunner(errClient{}, nil, "sys")
		out, err := r.Run(context.Background(), "task")
		if err == nil {
			t.Fatal("Run succeeded; wanted the client error")
		}
		final := out.Transcript.Record[len(out.Transcript.Record)-1]
		if final.Kind != llmkit.KindFinalize {
			t.Fatalf("last event = %s, want finalize", final.Kind)
		}
		if final.Step != 0 || final.Finalize.Iterations != 0 {
			t.Errorf("finalize step %d iterations %d, want 0/0 (failed completion does not advance)", final.Step, final.Finalize.Iterations)
		}
	})

	t.Run("failed completion carries the zero response", func(t *testing.T) {
		r := NewRunner(errClient{partial: llmkit.Response{Text: "half written"}}, nil, "sys")
		out, err := r.Run(context.Background(), "task")
		if err == nil {
			t.Fatal("Run succeeded; wanted the client error")
		}
		var comp *llmkit.Event
		for i := range out.Transcript.Record {
			if out.Transcript.Record[i].Kind == llmkit.KindCompletion {
				comp = &out.Transcript.Record[i]
			}
		}
		if comp == nil {
			t.Fatal("no completion event on the failing run")
		}
		if comp.Completion.Err == "" {
			t.Error("failed completion carries no Err")
		}
		if !reflect.DeepEqual(comp.Completion.Response, llmkit.Response{}) {
			t.Errorf("failed completion response = %+v, want the zero Response even though the client returned a partial one", comp.Completion.Response)
		}
	})

	t.Run("context cancelled mid-tool", func(t *testing.T) {
		bt := blockingTool{started: make(chan struct{})}
		r := NewRunner(newFakeClient(toolResp("c1", "block", `{}`, 1, 1)), []Tool{bt}, "sys")
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			<-bt.started
			cancel()
		}()
		out, err := r.Run(ctx, "task")
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		final := out.Transcript.Record[len(out.Transcript.Record)-1]
		if final.Kind != llmkit.KindFinalize {
			t.Fatalf("last event = %s, want finalize", final.Kind)
		}
		if final.Step != 1 || final.Finalize.Iterations != 1 {
			t.Errorf("finalize step %d iterations %d, want 1/1 (the completion before the tool did complete)", final.Step, final.Finalize.Iterations)
		}
	})

	t.Run("request policy aborts before the wire", func(t *testing.T) {
		r := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys",
			WithRequestPolicy(RequestPolicyFunc(func(context.Context, int, *llmkit.Request) error {
				return errors.New("nope")
			})))
		out, err := r.Run(context.Background(), "task")
		if err == nil {
			t.Fatal("Run succeeded; wanted the policy error")
		}
		final := out.Transcript.Record[len(out.Transcript.Record)-1]
		if final.Kind != llmkit.KindFinalize || final.Step != 0 {
			t.Fatalf("last event = %s step %d, want finalize at step 0", final.Kind, final.Step)
		}
	})
}

// TestObserver_SpanMintedPerCompletion pins C2's mint: every completion
// event carries its OWN fresh SpanID — present, pairwise distinct, and NOT
// inherited from any span already in the caller's context.
func TestObserver_SpanMintedPerCompletion(t *testing.T) {
	inherited := llmkit.NewSpanID()
	fc := newFakeClient(
		maxTokensResp("half an", 10, 5),
		textResp("answer", 10, 5),
	)
	r := NewRunner(fc, nil, "sys")
	out, err := r.Run(llmkit.WithSpan(context.Background(), inherited), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The max-tokens continuation is exactly TWO completions.
	var comps []*llmkit.Event
	for i := range out.Transcript.Record {
		if out.Transcript.Record[i].Kind == llmkit.KindCompletion {
			comps = append(comps, &out.Transcript.Record[i])
		}
	}
	if len(comps) != 2 {
		t.Fatalf("completion events = %d, want exactly 2 (turn + continuation)", len(comps))
	}
	seen := map[llmkit.SpanID]bool{inherited: true}
	for i, ev := range comps {
		if ev.SpanID == "" {
			t.Fatalf("completion %d carries no SpanID", i)
		}
		if ev.SpanID == inherited {
			t.Errorf("completion %d inherited the caller's span %q; each logical completion mints a fresh one", i, inherited)
		}
		if seen[ev.SpanID] {
			t.Errorf("completion %d reuses span %q", i, ev.SpanID)
		}
		seen[ev.SpanID] = true
	}
}

// TestObserver_SinkClosesOnFinalize pins the fd lifecycle: after a run's
// Finalize the sink retires the run's entry, so nothing stays open and a
// post-finalize event is refused through onErr.
func TestObserver_SinkClosesOnFinalize(t *testing.T) {
	dir := t.TempDir()
	sink := JSONL(dir, nil)
	r := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys", WithObserver(sink))
	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	sink.mu.Lock()
	live := len(sink.runs)
	sink.mu.Unlock()
	if live != 0 {
		t.Errorf("sink still holds %d run entries after Finalize; want 0 (closed, not leaked)", live)
	}
}

// TestReplay_StructureValidationEnforced pins that ReplayClient still
// validates the recorded tool-id structure: a replay whose conversation
// diverges (different seed task) fails at the recorded step.
func TestReplay_StructureValidationEnforced(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "echo", `{"text":"hi"}`, 10, 4),
		textResp("final", 8, 3),
	)
	rec := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")
	if _, err := rec.Run(context.Background(), "original task"); err != nil {
		t.Fatalf("record run: %v", err)
	}
	// A structural divergence: the record's first completion is expected only
	// after tool result c1; a replay whose first request carries no tool
	// result diverges at step 1.
	tr := NewTranscript()
	tr.RunID = "struct"
	tr.Record = append(tr.Record,
		llmkit.Event{Kind: llmkit.KindToolRun, RunID: "struct", Step: 1, SchemaVersion: 1,
			ToolRun: &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c1", Name: "echo"}, Result: "x"}},
		llmkit.Event{Kind: llmkit.KindCompletion, RunID: "struct", Step: 2, SchemaVersion: 1,
			Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "final"}}},
	)
	rp2, err := NewReplayClient(tr, "struct", llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	_, err = rp2.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "seed")},
	})
	if err == nil || !strings.Contains(err.Error(), "replay diverged at step 1") {
		t.Errorf("err = %v, want a structure divergence naming step 1", err)
	}
}

// TestReplay_SkipsFailedCompletions pins that NewReplayClient serves only
// successful Completion events.
func TestReplay_SkipsFailedCompletions(t *testing.T) {
	tr := NewTranscript()
	tr.RunID = "r-1"
	mk := func(kind llmkit.EventKind, errText string) llmkit.Event {
		ev := llmkit.NewEvent(llmkit.WithRun(context.Background(), "r-1"), kind)
		ev.SchemaVersion = llmkit.EventSchemaVersion
		if kind == llmkit.KindCompletion {
			ev.Completion = &llmkit.CompletionEvent{Response: llmkit.Response{Text: "ok"}, Err: errText}
		}
		return ev
	}
	tr.Record = append(tr.Record,
		mk(llmkit.KindCompletion, ""),
		mk(llmkit.KindCompletion, "connection reset"),
		mk(llmkit.KindCompletion, ""),
	)
	rc, err := NewReplayClient(tr, "r-1", llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	rc.mu.Lock()
	n := len(rc.responses)
	rc.mu.Unlock()
	if n != 2 {
		t.Errorf("replay built %d steps, want 2 (failed completion skipped)", n)
	}
}

// TestReplay_ParallelToolsDeterministic is the order-independence probe: a
// 40-turn recorded run whose turns each request 4 same-tool calls with
// distinct arguments replays under WithParallelTools — completion order
// cannot desync the (name, arguments) matching — with zero divergences.
func TestReplay_ParallelToolsDeterministic(t *testing.T) {
	const turns, callsPerTurn = 40, 4
	steps := make([]scriptStep, 0, turns+1)
	for i := range turns {
		calls := make([]llmkit.ToolCall, callsPerTurn)
		for j := range callsPerTurn {
			calls[j] = llmkit.ToolCall{
				ID:        fmt.Sprintf("c%d-%d", i, j),
				Name:      "echo",
				Arguments: json.RawMessage(fmt.Sprintf(`{"v":%d}`, i*callsPerTurn+j)),
			}
		}
		steps = append(steps, toolCallsResp(calls...))
	}
	steps = append(steps, textResp("done", 1, 1))

	live := 0
	rec := NewRunner(newFakeClient(steps...), []Tool{&countingTool{name: "echo", count: &live}}, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	if live == 0 {
		t.Fatal("record run executed no tools")
	}

	rp, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	tools := rp.Tools()
	live = 0
	replayed, err := NewRunner(rp, tools, "sys", WithParallelTools()).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("parallel replay diverged: %v", err)
	}
	if live != 0 {
		t.Errorf("replay executed %d live tools, want 0", live)
	}
	if rp.Err() != nil {
		t.Fatalf("parallel replay diverged: %v", rp.Err())
	}
	if replayed.FinalText != out.FinalText || replayed.Iterations != out.Iterations {
		t.Errorf("replay outcome = %q/%d turns, want %q/%d", replayed.FinalText, replayed.Iterations, out.FinalText, out.Iterations)
	}
}

// TestObserver_SinkRefusesDuplicateAndUnsafeRunIDs pins the write-side
// refusals: a leftover file, a duplicate live RunID, and an unsafe RunID are
// each reported through onErr and that run's events dropped.
func TestObserver_SinkRefusesDuplicateAndUnsafeRunIDs(t *testing.T) {
	ctx := context.Background()
	mkStart := func(id llmkit.RunID) llmkit.Event {
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindStart)
		ev.Start = &llmkit.StartEvent{Task: "t"}
		return ev
	}
	mkFin := func(id llmkit.RunID) llmkit.Event {
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindFinalize)
		ev.Finalize = &llmkit.FinalizeEvent{}
		return ev
	}

	t.Run("leftover file on disk", func(t *testing.T) {
		dir := t.TempDir()
		staleLine := `{"kind":"start","run_id":"older","schema_version":1}`
		stale := staleLine + "\n"
		if err := os.WriteFile(filepath.Join(dir, "rid-1.jsonl"), []byte(stale), 0o644); err != nil {
			t.Fatal(err)
		}
		var mu sync.Mutex
		var errs []error
		sink := JSONL(dir, func(err error) {
			mu.Lock()
			defer mu.Unlock()
			errs = append(errs, err)
		})
		sink.Observe(ctx, mkStart("rid-1"))
		sink.Observe(ctx, mkFin("rid-1"))
		got, err := os.ReadFile(filepath.Join(dir, "rid-1.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != stale {
			t.Errorf("leftover file was modified: %q", got)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(errs) == 0 {
			t.Error("leftover file was not reported through onErr")
		}
	})

	t.Run("duplicate live RunID", func(t *testing.T) {
		dir := t.TempDir()
		sink := JSONL(dir, nil)
		sink.Observe(ctx, mkStart("rid-2"))
		sink.Observe(ctx, mkStart("rid-2")) // refused
		sink.Observe(ctx, mkFin("rid-2"))
		if _, err := sink.Events(ctx, "rid-2"); err != nil {
			t.Errorf("first run's record lost: %v", err)
		}
	})

	t.Run("unsafe RunID", func(t *testing.T) {
		dir := t.TempDir()
		sink := JSONL(dir, nil)
		sink.Observe(ctx, mkStart("../escape"))
		if entries, _ := os.ReadDir(dir); len(entries) != 0 {
			t.Errorf("unsafe run id escaped the dir: %v", entries)
		}
	})
}

// TestObserver_UnknownRunSentinels pins the read-side contract: both Source
// implementations fail with ErrUnknownRun for runs they have no record of.
func TestObserver_UnknownRunSentinels(t *testing.T) {
	tr := NewTranscript()
	tr.RunID = "known"
	if _, err := tr.Events(context.Background(), "other"); !errors.Is(err, ErrUnknownRun) {
		t.Errorf("Transcript.Events mismatch err = %v, want ErrUnknownRun", err)
	}
	dir := t.TempDir()
	if _, err := JSONL(dir, nil).Events(context.Background(), "missing"); !errors.Is(err, ErrUnknownRun) {
		t.Errorf("JSONLSink.Events missing err = %v, want ErrUnknownRun", err)
	}
}

// TestLoadJSONL_RecoversRunIdentity pins that a loaded transcript recovers
// RunID/ParentRunID from the decoded events and refuses mixed ids.
func TestLoadJSONL_RecoversRunIdentity(t *testing.T) {
	src := NewTranscript()
	src.RunID, src.ParentRunID = "child", "parent"
	src.Record = append(src.Record, llmkit.Event{
		Kind: llmkit.KindStart, RunID: "child", ParentRunID: "parent",
		SchemaVersion: llmkit.EventSchemaVersion,
		Start:         &llmkit.StartEvent{Task: "t"},
	})
	var buf bytes.Buffer
	if err := src.SaveJSONL(&buf); err != nil {
		t.Fatal(err)
	}
	orig := buf.Bytes()
	loaded, err := LoadJSONL(&buf)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	if loaded.RunID != "child" || loaded.ParentRunID != "parent" {
		t.Errorf("identity = %q/%q, want child/parent", loaded.RunID, loaded.ParentRunID)
	}
	var mixed bytes.Buffer
	mixed.Write(orig)
	mixed.WriteString(`{"kind":"start","run_id":"sneaky","schema_version":1}` + "\n")
	if _, err := LoadJSONL(&mixed); err == nil || !strings.Contains(err.Error(), "run id") {
		t.Errorf("mixed-id file loaded: %v", err)
	}
}

// gateTool lets a test hold N concurrent tool calls open until released.
type gateTool struct {
	wg      *sync.WaitGroup
	release chan struct{}
}

func (t gateTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "gate", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t gateTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	t.wg.Done()
	<-t.release
	return "opened", nil
}

// endlessToolClient answers every completion with a fresh two-call turn for
// the gate tool, so concurrent runs never exhaust a script.
type endlessToolClient struct{}

func (endlessToolClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (endlessToolClient) Complete(_ context.Context, _ llmkit.Request) (llmkit.Response, error) {
	return llmkit.Response{
		ToolCalls: []llmkit.ToolCall{
			{ID: "c1", Name: "gate", Arguments: json.RawMessage(`{}`)},
			{ID: "c2", Name: "gate", Arguments: json.RawMessage(`{}`)},
		},
		StopReason: llmkit.StopToolUse,
	}, nil
}

// TestObserver_SinkDuplicateConcurrentRunID pins the duplicate-Start row of
// the admission table with two deterministic concurrent runs pinned to the
// same WithRunID: one file, the refusal reported, and the surviving record a
// clean single-run prefix — nothing merged.
func TestObserver_SinkDuplicateConcurrentRunID(t *testing.T) {
	dir := t.TempDir()
	var mu sync.Mutex
	var errs []error
	sink := JSONL(dir, func(err error) {
		mu.Lock()
		defer mu.Unlock()
		errs = append(errs, err)
	})

	// Sequential dispatch: only each run's FIRST gate call blocks, so two
	// Dones unblock the wait.
	var started sync.WaitGroup
	started.Add(2)
	release := make(chan struct{})
	gate := gateTool{wg: &started, release: release}

	r := NewRunner(endlessToolClient{}, []Tool{gate}, "sys",
		WithLimits(Limits{MaxIterations: 2}), WithObserver(sink))

	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = r.Run(context.Background(), "task", WithRunID("dup-1"))
		}()
	}
	started.Wait()
	close(release)
	wg.Wait()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "dup-1.jsonl" {
		t.Fatalf("sink files = %v, want exactly [dup-1.jsonl]", entries)
	}
	mu.Lock()
	fired := len(errs)
	mu.Unlock()
	if fired == 0 {
		t.Error("duplicate Start was not reported through onErr")
	}
	// The surviving record reads back as one clean run — nothing merged.
	if _, err := sink.Events(context.Background(), "dup-1"); err != nil {
		t.Errorf("surviving record does not read back cleanly: %v", err)
	}
}

// TestObserver_SinkRefusedRunsLeaveNoEntries pins the cleanup rule: N runs
// refused at Start (leftover files) still Finalize, and every entry is
// deleted — the sink holds nothing for refused runs.
func TestObserver_SinkRefusedRunsLeaveNoEntries(t *testing.T) {
	dir := t.TempDir()
	sink := JSONL(dir, nil)
	ctx := context.Background()
	for i := range 50 {
		id := llmkit.RunID(fmt.Sprintf("refused-%d", i))
		// Leftover file: the Start will hit ErrExist.
		if err := os.WriteFile(filepath.Join(dir, string(id)+".jsonl"), []byte("{}\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindStart)
		ev.Start = &llmkit.StartEvent{Task: "t"}
		sink.Observe(ctx, ev)
		fin := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindFinalize)
		fin.Finalize = &llmkit.FinalizeEvent{}
		sink.Observe(ctx, fin)
	}
	sink.mu.Lock()
	live := len(sink.runs)
	sink.mu.Unlock()
	if live != 0 {
		t.Errorf("sink holds %d entries after 50 refused runs finalized; want 0", live)
	}
}

// TestObserver_HookPanicLeavesSinkClean pins that the hook-panic path —
// which finalizes with zero counters — still retires the sink's entry: no
// leaked file handles, no ghost entries.
func TestObserver_HookPanicLeavesSinkClean(t *testing.T) {
	dir := t.TempDir()
	sink := JSONL(dir, nil)
	hooks := Hooks{
		AfterCompletion: func(context.Context, int, *llmkit.Request, *llmkit.Response, error) {
			panic("hook exploded")
		},
	}
	r := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys",
		WithHooks(hooks), WithObserver(sink))
	func() {
		defer func() {
			if v := recover(); v == nil {
				t.Error("hook panic did not propagate")
			}
		}()
		_, _ = r.Run(context.Background(), "task")
	}()
	sink.mu.Lock()
	live := len(sink.runs)
	sink.mu.Unlock()
	if live != 0 {
		t.Errorf("sink holds %d entries after a hook-panic run; want 0", live)
	}
}

// completedSignalClient signals once after its first Complete returns.
type completedSignalClient struct {
	inner *fakeClient
	once  sync.Once
	hit   chan struct{}
}

func (c *completedSignalClient) Capabilities() llmkit.Capabilities { return c.inner.Capabilities() }

func (c *completedSignalClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	resp, err := c.inner.Complete(ctx, req)
	c.once.Do(func() { close(c.hit) })
	return resp, err
}

// TestObserver_CancelledCallsRecordNotRun pins M18/M19, deterministically:
// the policy cancels the run inside its own FIRST invocation, so the second
// call of the turn is never authorized — it records IsError with the
// "not run" rendering and never the Denied mark, while the call authorized
// before the cancellation ran normally.
func TestObserver_CancelledCallsRecordNotRun(t *testing.T) {
	dir := t.TempDir()
	recorder := JSONL(dir, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var once sync.Once
	policy := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if call.Name == "t1" {
			once.Do(func() { cancel() })
		}
		return nil
	})
	calls := []llmkit.ToolCall{
		{ID: "t1", Name: "t1", Arguments: json.RawMessage(`{}`)},
		{ID: "t2", Name: "t2", Arguments: json.RawMessage(`{}`)},
	}
	r := NewRunner(newFakeClient(toolCallsResp(calls...)), []Tool{
		echoTool{name: "t1"}, echoTool{name: "t2"},
	}, "sys", WithToolPolicy(policy), WithParallelTools(), WithObserver(recorder))

	// The loop's post-turn ctx check returns the cancellation; the transcript
	// still holds both tool events.
	_, _ = r.Run(ctx, "task", WithRunID("cancelled-1"))

	evs, err := recorder.Events(context.Background(), "cancelled-1")
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	byID := map[string]*llmkit.Event{}
	for i := range evs {
		ev := &evs[i]
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil {
			byID[ev.ToolRun.Call.ID] = ev
		}
	}
	t1, t2 := byID["t1"], byID["t2"]
	if t1 == nil || t2 == nil {
		t.Fatalf("tool events missing: t1=%v t2=%v", t1 != nil, t2 != nil)
	}
	if t1.ToolRun.Denied || t1.ToolRun.IsError {
		t.Errorf("t1 ran after its own authorize cancelled: %+v, want a normal result", t1.ToolRun)
	}
	if t2.ToolRun.Denied {
		t.Error("t2 recorded as Denied; a cancelled run is not a policy decision")
	}
	if !t2.ToolRun.IsError || !strings.Contains(t2.ToolRun.Result, "not run: context canceled") {
		t.Errorf("t2 = %+v, want IsError with the not-run rendering", t2.ToolRun)
	}
}

// --- round-3 discriminating tests: each named mutant dies ------------------

// TestObserver_DuplicateStartRefusalDetails pins M15/M6/M7/M32/M39: the
// duplicate-Start refusal names the run, and the poisoned entry's fd is
// closed (nil) — no orphan handle survives the poisoning.
func TestObserver_DuplicateStartRefusalDetails(t *testing.T) {
	dir := t.TempDir()
	var mu sync.Mutex
	var msgs []string
	sink := JSONL(dir, func(err error) {
		mu.Lock()
		defer mu.Unlock()
		msgs = append(msgs, err.Error())
	})
	ctx := context.Background()
	start := func() llmkit.Event {
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, "dup-x"), llmkit.KindStart)
		ev.Start = &llmkit.StartEvent{Task: "t"}
		return ev
	}
	sink.Observe(ctx, start())
	sink.Observe(ctx, start()) // duplicate

	mu.Lock()
	defer mu.Unlock()
	joined := strings.Join(msgs, "\n")
	if !strings.Contains(joined, "duplicate Start") || !strings.Contains(joined, "dup-x") {
		t.Errorf("refusal messages = %v, want a duplicate-Start report naming the run", msgs)
	}
	sink.mu.Lock()
	entry := sink.runs["dup-x"]
	sink.mu.Unlock()
	if entry == nil {
		t.Fatal("duplicate-Start entry vanished; want a poisoned entry")
	}
	entry.mu.Lock()
	fdNil := entry.file == nil
	disabled := entry.disabled
	entry.mu.Unlock()
	if !fdNil || !disabled {
		t.Errorf("poisoned entry file=%v disabled=%v, want nil/true", fdNil, disabled)
	}
}

// TestObserver_EventsRejectsDegenerateRecords pins M20/M10/M22/M3/M4/M5: a
// 2-Start record, a wholly-foreign record, and a 0-byte record all fail.
func TestObserver_EventsRejectsDegenerateRecords(t *testing.T) {
	ctx := context.Background()
	write := func(t *testing.T, name, content string) string {
		t.Helper()
		dir := t.TempDir()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	line := func(run llmkit.RunID, kind llmkit.EventKind) string {
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, run), kind)
		ev.SchemaVersion = llmkit.EventSchemaVersion
		if kind == llmkit.KindStart {
			ev.Start = &llmkit.StartEvent{Task: "t"}
		}
		b, _ := json.Marshal(ev)
		return string(b) + "\n"
	}

	t.Run("two starts", func(t *testing.T) {
		dir := write(t, "r-1.jsonl", line("r-1", llmkit.KindStart)+line("r-1", llmkit.KindStart))
		_, err := JSONL(dir, nil).Events(ctx, "r-1")
		if err == nil || !strings.Contains(err.Error(), "holds 2 runs") {
			t.Errorf("err = %v, want the two-run record error", err)
		}
	})
	t.Run("wholly foreign record", func(t *testing.T) {
		dir := write(t, "r-2.jsonl", line("other", llmkit.KindStart))
		_, err := JSONL(dir, nil).Events(ctx, "r-2")
		if err == nil || !strings.Contains(err.Error(), "line 1") || !strings.Contains(err.Error(), `"other"`) {
			t.Errorf("err = %v, want a file+line error naming the foreign id", err)
		}
	})
	t.Run("zero-byte record", func(t *testing.T) {
		dir := write(t, "r-3.jsonl", "")
		_, err := JSONL(dir, nil).Events(ctx, "r-3")
		if !errors.Is(err, ErrUnknownRun) {
			t.Errorf("err = %v, want ErrUnknownRun", err)
		}
	})
}

// TestObserver_UnsafeIDCreatesNothingAnywhere pins M40/M9: an unsafe RunID
// never creates a file under the sink dir OR its parent (the test root).
func TestObserver_UnsafeIDCreatesNothingAnywhere(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "sink")
	var mu sync.Mutex
	var errs []error
	sink := JSONL(dir, func(err error) {
		mu.Lock()
		defer mu.Unlock()
		errs = append(errs, err)
	})
	ctx := context.Background()
	for _, id := range []llmkit.RunID{"../escape", "sub/../../climb", "..", "."} {
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindStart)
		ev.Start = &llmkit.StartEvent{Task: "t"}
		sink.Observe(ctx, ev)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(errs) != 4 {
		t.Errorf("onErr fired %d times, want 4 (one per unsafe id)", len(errs))
	}
	for _, rootDir := range []string{dir, root} {
		entries, err := os.ReadDir(rootDir)
		if err != nil {
			continue // a directory that was never created is fine
		}
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".jsonl") {
				t.Errorf("unsafe id escaped: %s/%s exists", rootDir, e.Name())
			}
		}
	}
}

// TestObserver_FdCountFlatAcrossFinalizedRuns pins M42: 200 runs opened and
// finalized leave the process fd count flat — every fd is closed at
// Finalize.
func TestObserver_FdCountFlatAcrossFinalizedRuns(t *testing.T) {
	countFDs := func() int {
		entries, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			return -1 // non-Linux: skip below
		}
		return len(entries)
	}
	dir := t.TempDir()
	sink := JSONL(dir, nil)
	ctx := context.Background()
	before := countFDs()
	for i := range 200 {
		id := llmkit.RunID(fmt.Sprintf("fd-%d", i))
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindStart)
		ev.Start = &llmkit.StartEvent{Task: "t"}
		sink.Observe(ctx, ev)
		fin := llmkit.NewEvent(llmkit.WithRun(ctx, id), llmkit.KindFinalize)
		fin.Finalize = &llmkit.FinalizeEvent{}
		sink.Observe(ctx, fin)
	}
	after := countFDs()
	if before < 0 || after < 0 {
		t.Skip("/proc/self/fd unavailable")
	}
	if after > before+4 { // small slack for test-framework churn; a per-run leak would be +200
		t.Errorf("fd count grew from %d to %d across 200 finalized runs; an fd leaked", before, after)
	}
}

// TestObserver_PostFinalizeEventsEachReport pins M33: N post-finalize
// events on the appendRun path (neither Start nor Finalize) produce N onErr
// reports naming the unknown-or-closed run — every occurrence is loud.
func TestObserver_PostFinalizeEventsEachReport(t *testing.T) {
	dir := t.TempDir()
	var mu sync.Mutex
	var errs []error
	sink := JSONL(dir, func(err error) {
		mu.Lock()
		defer mu.Unlock()
		errs = append(errs, err)
	})
	ctx := context.Background()
	ev := llmkit.NewEvent(llmkit.WithRun(ctx, "p-1"), llmkit.KindStart)
	ev.Start = &llmkit.StartEvent{Task: "t"}
	sink.Observe(ctx, ev)
	fin := llmkit.NewEvent(llmkit.WithRun(ctx, "p-1"), llmkit.KindFinalize)
	fin.Finalize = &llmkit.FinalizeEvent{}
	sink.Observe(ctx, fin)
	for range 5 {
		late := llmkit.NewEvent(llmkit.WithRun(ctx, "p-1"), llmkit.KindToolRun)
		late.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "late", Name: "gate"}}
		sink.Observe(ctx, late)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(errs) != 5 {
		t.Errorf("onErr fired %d times for 5 post-finalize events, want 5", len(errs))
	}
	for _, e := range errs {
		if e != nil && !strings.Contains(e.Error(), "unknown or closed run") {
			t.Errorf("post-finalize report %q does not name the unknown-or-closed run", e)
			break
		}
	}
}

// TestObserver_StructureAndExhaustionWrapSentinel pins M29/M30/M13/M14:
// both wire-structure divergence paths wrap ErrReplayDiverged AND record it
// on the client.
func TestObserver_StructureAndExhaustionWrapSentinel(t *testing.T) {
	tr := NewTranscript()
	tr.RunID = "s-1"
	tr.Record = append(tr.Record,
		llmkit.Event{Kind: llmkit.KindToolRun, RunID: "s-1", Step: 1, SchemaVersion: 1,
			ToolRun: &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c1", Name: "echo"}, Result: "x"}},
		llmkit.Event{Kind: llmkit.KindCompletion, RunID: "s-1", Step: 2, SchemaVersion: 1,
			Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "final"}}},
	)
	rc, err := NewReplayClient(tr, "s-1", llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	// Structure mismatch: the request carries no preceding tool result.
	_, err = rc.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "seed")},
	})
	if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "at step 1") {
		t.Errorf("structure err = %v, want ErrReplayDiverged naming step 1", err)
	}
	if rc.Err() == nil {
		t.Error("structure divergence not recorded on the client")
	}
	// Exhaustion needs a record whose completions carry no tool-result
	// expectations: two bare completions, then a third Complete.
	tr2 := NewTranscript()
	tr2.RunID = "s-2"
	for step := 1; step <= 2; step++ {
		tr2.Record = append(tr2.Record, llmkit.Event{Kind: llmkit.KindCompletion, RunID: "s-2", Step: step, SchemaVersion: 1,
			Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "ok"}}})
	}
	rc2, err := NewReplayClient(tr2, "s-2", llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := rc2.Complete(context.Background(), llmkit.Request{}); err != nil {
			t.Fatalf("serving the record: %v", err)
		}
	}
	_, err = rc2.Complete(context.Background(), llmkit.Request{})
	if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "after step 2") {
		t.Errorf("exhaustion err = %v, want ErrReplayDiverged naming the served step", err)
	}
	if rc2.Err() == nil {
		t.Error("exhaustion not recorded on the client")
	}
}

// TestObserver_ReplayedDenialReproducesText pins M36: a recorded policy
// denial replays to the byte-identical model-visible denial text.
func TestObserver_ReplayedDenialReproducesText(t *testing.T) {
	policy := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if call.Name == "denied_tool" {
			return errors.New("manual approval required")
		}
		return nil
	})
	rec := NewRunner(newFakeClient(
		toolResp("c1", "denied_tool", `{}`, 1, 1),
		textResp("final", 1, 1),
	), []Tool{echoTool{name: "denied_tool"}}, "sys", WithToolPolicy(policy))
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	const want = "ERROR: tool denied_tool denied: manual approval required"
	sawRecorded := false
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleToolResult && m.Text() == want {
			sawRecorded = true
		}
	}
	if !sawRecorded {
		t.Fatalf("recorded run lost the denial text: %+v", out.Messages)
	}

	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay run: %v", err)
	}
	if rc.Err() != nil {
		t.Fatalf("denial replay diverged: %v", rc.Err())
	}
	sawReplayed := false
	for _, m := range replayed.Messages {
		if m.Role == llmkit.RoleToolResult && m.Text() == want {
			sawReplayed = true
		}
	}
	if !sawReplayed {
		t.Errorf("replayed denial text differs: %+v", replayed.Messages)
	}
}

// TestObserver_ExtraCallSetsErr pins M26/M1b: a tool call beyond the record
// sets rc.Err() via the extra-call path, naming the last recorded step.
func TestObserver_ExtraCallSetsErr(t *testing.T) {
	fc := newFakeClient(toolResp("c1", "echo", `{"text":"hi"}`, 1, 1))
	rec := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	tools := rc.Tools()
	if _, err := tools[0].Run(context.Background(), json.RawMessage(`{"text":"hi"}`)); err != nil {
		t.Fatalf("recorded call mismatched: %v", err)
	}
	if rc.Err() != nil {
		t.Fatalf("first (recorded) call set Err: %v", rc.Err())
	}
	_, err = tools[0].Run(context.Background(), json.RawMessage(`{"text":"hi"}`))
	if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "at step 1") {
		t.Errorf("extra call err = %v, want ErrReplayDiverged naming step 1", err)
	}
	if !errors.Is(rc.Err(), ErrReplayDiverged) {
		t.Error("extra-call divergence not recorded on the client")
	}
}

// TestObserver_StructureMismatchPerID pins the per-ID branch of the
// structure validation: the same number of preceding tool results under a
// FOREIGN id diverges with ErrReplayDiverged recorded on the client.
func TestObserver_StructureMismatchPerID(t *testing.T) {
	tr := NewTranscript()
	tr.RunID = "s-3"
	tr.Record = append(tr.Record,
		llmkit.Event{Kind: llmkit.KindToolRun, RunID: "s-3", Step: 1, SchemaVersion: 1,
			ToolRun: &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c1", Name: "echo"}, Result: "x"}},
		llmkit.Event{Kind: llmkit.KindCompletion, RunID: "s-3", Step: 2, SchemaVersion: 1,
			Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "final"}}},
	)
	rc, err := NewReplayClient(tr, "s-3", llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	req := llmkit.Request{Messages: []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "seed"),
		llmkit.ToolResult("FOREIGN-ID", "a result the record never saw"),
	}}
	_, err = rc.Complete(context.Background(), req)
	if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "at step 1") {
		t.Errorf("err = %v, want ErrReplayDiverged naming step 1", err)
	}
	if !errors.Is(rc.Err(), ErrReplayDiverged) {
		t.Error("per-ID divergence not recorded on the client")
	}
}

// TestObserver_ToolsReturnsCopy pins M35/M12: mutating the returned slice
// cannot touch the client's bound tools.
func TestObserver_ToolsReturnsCopy(t *testing.T) {
	fc := newFakeClient(toolResp("c1", "echo", `{"text":"hi"}`, 1, 1))
	rec := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	tools := rc.Tools()
	if tools == nil || len(tools) != 1 {
		t.Fatalf("Tools = %v, want one bound tool", tools)
	}
	tools[0] = nil // clobber the caller's copy
	if _, err := rc.Tools()[0].Run(context.Background(), json.RawMessage(`{"text":"hi"}`)); err != nil {
		t.Errorf("client's bound tool damaged through the returned slice: %v", err)
	}
	scripted := NewReplayClientFromResponses([]llmkit.Response{{Text: "x"}}, llmkit.Capabilities{})
	if got := scripted.Tools(); got == nil || len(got) != 0 {
		t.Errorf("scripted Tools = %v, want a non-nil empty slice", got)
	}
}

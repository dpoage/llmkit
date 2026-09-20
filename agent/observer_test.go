package agent

import (
	"context"
	"encoding/json"
	"errors"
	"os"
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
// through NewReplayClient + ReplayTools with ZERO live tool executions: same
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
	replayTools, err := ReplayTools(src, src.RunID)
	if err != nil {
		t.Fatalf("ReplayTools: %v", err)
	}
	if len(replayTools) != 2 {
		t.Fatalf("ReplayTools returned %d tools, want 2", len(replayTools))
	}

	liveCalls = 0
	replayer := NewRunner(replayClient, replayTools, "sys")
	replayed, err := replayer.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay run: %v", err)
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
				// (ReplayTools has names only), so the offered tool set
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

// TestObserver_ReplayDivergenceNamesStep pins that a diverging tool call
// fails replay with an error naming the recorded step.
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
	src := out.Transcript

	// A replay whose model asks for the SAME tool with DIFFERENT arguments:
	// the recorded cursor has {"text":"hi"} at step 1.
	diverging := NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls:  []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"text":"something else"}`)}},
			StopReason: llmkit.StopToolUse,
		},
		{Text: "final", StopReason: llmkit.StopEndTurn},
	}, llmkit.Capabilities{})

	replayTools, err := ReplayTools(src, src.RunID)
	if err != nil {
		t.Fatalf("ReplayTools: %v", err)
	}
	replayed, err := NewRunner(diverging, replayTools, "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay run: %v", err)
	}

	// The divergence surfaces as model-visible tool data naming the step.
	diverged := false
	for _, m := range replayed.Messages {
		if m.Role == llmkit.RoleToolResult && strings.Contains(m.Text(), `replay diverged at step 1`) {
			diverged = true
		}
	}
	if !diverged {
		t.Errorf("no divergence error naming step 1 in the replay history: %+v", replayed.Messages)
	}
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

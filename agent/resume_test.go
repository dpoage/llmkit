package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// --- doubles ----------------------------------------------------------------

type liveFake struct {
	*fakeClient
	id llmkit.Identity
}

func (f liveFake) Identity() llmkit.Identity { return f.id }

func newFakeFromResponses(resps []llmkit.Response) *fakeClient {
	steps := make([]scriptStep, len(resps))
	for i, resp := range resps {
		steps[i] = scriptStep{resp: resp}
	}
	return newFakeClient(steps...)
}

func newFakeFromCompletions(comps []llmkit.CompletionEvent) *fakeClient {
	steps := make([]scriptStep, len(comps))
	for i, ce := range comps {
		if ce.ErrKind == llmkit.ErrKindContextTooLong {
			steps[i] = scriptStep{err: fmt.Errorf("%s", ce.Err)}
		} else {
			steps[i] = scriptStep{resp: ce.Response}
		}
	}
	return newFakeClient(steps...)
}

func asTools(wrapped []*resumeCountingTool) []Tool {
	tools := make([]Tool, len(wrapped))
	for i, t := range wrapped {
		tools[i] = t
	}
	return tools
}

type resumeCountingTool struct {
	Tool
	mu   sync.Mutex
	runs []string
}

func (t *resumeCountingTool) Run(ctx context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	t.mu.Lock()
	t.runs = append(t.runs, callIDFrom(ctx))
	t.mu.Unlock()
	return t.Tool.Run(ctx, args)
}

func (t *resumeCountingTool) count(id string) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := 0
	for _, ran := range t.runs {
		if ran == id {
			n++
		}
	}
	return n
}

type fixedTool struct {
	name, result string
}

func (t fixedTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "test tool " + t.name,
		Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)}
}

func (t fixedTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	return []llmkit.Block{llmkit.Text(t.result)}, nil
}

type resumeBlocksTool struct {
	name   string
	blocks []llmkit.Block
}

func (t resumeBlocksTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "test tool " + t.name,
		Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}
}

func (t resumeBlocksTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	return t.blocks, nil
}

type argTool struct {
	name string
	mu   sync.Mutex
	got  []json.RawMessage
}

func (t *argTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "test tool " + t.name,
		Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}
}

func (t *argTool) Run(_ context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	t.mu.Lock()
	t.got = append(t.got, append(json.RawMessage(nil), args...))
	t.mu.Unlock()
	return []llmkit.Block{llmkit.Text("ran with args")}, nil
}

// --- record helpers ---------------------------------------------------------

// servableCompletions mirrors NewReplayClient's serving filter.
func servableCompletions(evs []llmkit.Event) []llmkit.CompletionEvent {
	var out []llmkit.CompletionEvent
	for _, ev := range evs {
		if ev.Kind != llmkit.KindCompletion || ev.Completion == nil || isPurposeCompletion(ev) {
			continue
		}
		ce := *ev.Completion
		if ce.Err != "" && ce.ErrKind != llmkit.ErrKindContextTooLong {
			continue
		}
		out = append(out, ce)
	}
	return out
}

func servableRequests(evs []llmkit.Event) []llmkit.Request {
	comps := servableCompletions(evs)
	reqs := make([]llmkit.Request, len(comps))
	for i, ce := range comps {
		reqs[i] = ce.Request
	}
	return reqs
}

func servableResponses(evs []llmkit.Event) []llmkit.Response {
	comps := servableCompletions(evs)
	resps := make([]llmkit.Response, len(comps))
	for i, ce := range comps {
		resps[i] = ce.Response
	}
	return resps
}

func summedUsage(evs []llmkit.Event) llmkit.Usage {
	var u llmkit.Usage
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Completion.Err == "" {
			u = u.Add(ev.Completion.Response.Usage)
		}
	}
	return u
}

func finalizeOf(evs []llmkit.Event) *llmkit.FinalizeEvent {
	for i := len(evs) - 1; i >= 0; i-- {
		if evs[i].Kind == llmkit.KindFinalize {
			return evs[i].Finalize
		}
	}
	return nil
}

func cutSources(t *testing.T, runID llmkit.RunID, evs []llmkit.Event) []struct {
	name string
	src  llmkit.Source
} {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, string(runID)+".jsonl")
	tr := &Transcript{Record: evs, RunID: runID}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create jsonl: %v", err)
	}
	if err := tr.SaveJSONL(f); err != nil {
		_ = f.Close()
		t.Fatalf("SaveJSONL: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close jsonl: %v", err)
	}
	return []struct {
		name string
		src  llmkit.Source
	}{
		{"transcript", tr},
		{"jsonl", JSONL(dir, nil)},
	}
}

// --- the R1 sweep -----------------------------------------------------------

type resumeScenario struct {
	name   string
	caps   llmkit.Capabilities
	system string
	task   string
	tools  func() []Tool
	steps  func() []scriptStep
	opts   func(c llmkit.Client, tools []Tool) []Option
	schema json.RawMessage
	out    func() any
}

func twoToolResp(id1, id2, name string, in, out int64) scriptStep {
	step := toolResp(id1, name, `{"q":"one"}`, in, out)
	step.resp.ToolCalls = append(step.resp.ToolCalls, llmkit.ToolCall{
		ID: id2, Name: name, Arguments: json.RawMessage(`{"q":"two"}`),
	})
	return step
}

func resumeBigResultTool(name string) Tool {
	return fixedTool{name: name, result: strings.Repeat("r0123456789", 50)}
}

func resumeScenarios() []resumeScenario {
	noOpts := func(llmkit.Client, []Tool) []Option { return nil }
	return []resumeScenario{
		{
			name:   "plain",
			system: "You test resume.",
			task:   "use the echo tool then answer",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					toolResp("c1", "echo", `{"q":"hi"}`, 100, 20),
					textResp("all done", 80, 10),
				}
			},
			opts: noOpts,
		},
		{
			name:   "request-policy",
			system: "You test resume.",
			task:   "use the echo tool then answer",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					toolResp("c1", "echo", `{"q":"hi"}`, 100, 20),
					textResp("policed answer", 80, 10),
				}
			},
			opts: func(llmkit.Client, []Tool) []Option {
				return []Option{WithRequestPolicy(RequestPolicyFunc(
					func(_ context.Context, _ int, req *llmkit.Request) error {
						req.System += " Be terse."
						req.Messages = append(req.Messages, llmkit.TextMessage(llmkit.RoleUser, "policy note"))
						return nil
					}))}
			},
		},
		{
			name:   "hook-system",
			system: "You test resume.",
			task:   "use the echo tool then answer",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					toolResp("c1", "echo", `{"q":"hi"}`, 100, 20),
					textResp("hooked answer", 80, 10),
				}
			},
			opts: func(llmkit.Client, []Tool) []Option {
				return []Option{WithHooks(Hooks{
					BeforeCompletion: func(_ context.Context, _ int, req *llmkit.Request) {
						req.System += " [hooked]"
					},
				})}
			},
		},
		{
			name:   "empty-turn-nudge",
			system: "You test resume.",
			task:   "answer the question",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					thinkOnlyResp("only thinking here", 100, 10),
					textResp("recovered answer", 90, 9),
				}
			},
			opts: noOpts,
		},
		{
			name:   "runjson-repair",
			system: "You emit JSON.",
			task:   "emit the json",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					textResp("not json at all", 100, 10),
					textResp(`{"ok":true}`, 90, 9),
				}
			},
			opts:   noOpts,
			schema: json.RawMessage(`{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}`),
			out: func() any {
				return &struct {
					OK bool `json:"ok"`
				}{}
			},
		},
		{
			name:   "stub-compaction",
			system: "You test resume.",
			task:   "call echo eight times then answer",
			tools:  func() []Tool { return []Tool{resumeBigResultTool("echo")} },
			steps: func() []scriptStep {
				steps := make([]scriptStep, 0, 9)
				for i := 1; i <= 8; i++ {
					steps = append(steps, toolResp(fmt.Sprintf("c%d", i), "echo", `{"q":"hi"}`, 100, 10))
				}
				return append(steps, textResp("compacted answer", 80, 8))
			},
			opts: func(llmkit.Client, []Tool) []Option {
				return []Option{WithLimits(Limits{HistoryTokenBudget: 1})}
			},
		},
		{
			name:   "multi-block-media",
			caps:   llmkit.Capabilities{ToolResultMedia: true},
			system: "You test resume.",
			task:   "use both tools then answer",
			tools: func() []Tool {
				return []Tool{
					resumeBlocksTool{name: "multi", blocks: []llmkit.Block{llmkit.Text("alpha"), llmkit.Text("beta")}},
					resumeBlocksTool{name: "shot", blocks: []llmkit.Block{{Kind: llmkit.BlockImage, MediaType: "image/png", Data: []byte("\x89PNG-fake-bytes")}}},
				}
			},
			steps: func() []scriptStep {
				return []scriptStep{
					toolResp("c1", "multi", `{"q":"x"}`, 100, 10),
					toolResp("c2", "shot", `{"q":"y"}`, 110, 10),
					textResp("saw the blocks", 90, 9),
				}
			},
			opts: noOpts,
		},
		{
			name:   "denied-call",
			system: "You test resume.",
			task:   "try the secret tool then answer",
			tools:  func() []Tool { return []Tool{fixedTool{name: "secret", result: "never"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					toolResp("c1", "secret", `{"q":"hi"}`, 100, 10),
					textResp("it was denied", 80, 8),
				}
			},
			opts: func(llmkit.Client, []Tool) []Option {
				return []Option{WithToolPolicy(ToolPolicyFunc(
					func(_ context.Context, call *llmkit.ToolCall) error {
						if call.Name == "secret" {
							return errors.New("not allowed here")
						}
						return nil
					}))}
			},
		},
		{
			name:   "parallel-tools",
			system: "You test resume.",
			task:   "call echo twice in one turn then answer",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					twoToolResp("c1", "c2", "echo", 100, 10),
					textResp("parallel answer", 80, 8),
				}
			},
			opts: func(llmkit.Client, []Tool) []Option { return []Option{WithParallelTools()} },
		},
		{
			name:   "rewriting-policy",
			system: "You test resume.",
			task:   "use the argcheck tool then answer",
			tools:  func() []Tool { return []Tool{&argTool{name: "argcheck"}} },
			steps: func() []scriptStep {
				return []scriptStep{
					toolResp("c1", "argcheck", `{"q":"hi"}`, 100, 10),
					textResp("rewritten answer", 80, 8),
				}
			},
			opts: func(llmkit.Client, []Tool) []Option {
				return []Option{WithToolPolicy(ToolPolicyFunc(
					func(_ context.Context, call *llmkit.ToolCall) error {
						call.Arguments = json.RawMessage(`{"q":"hi","extra":true}`)
						return nil
					}))}
			},
		},
		{
			name:   "overflow-recovery",
			system: "You test resume.",
			task:   "call echo six times then answer",
			tools:  func() []Tool { return []Tool{fixedTool{name: "echo", result: strings.Repeat("overflow-result-", 20)}} },
			steps: func() []scriptStep {
				steps := make([]scriptStep, 0, 8)
				for i := 1; i <= 6; i++ {
					steps = append(steps, toolResp(fmt.Sprintf("c%d", i), "echo", `{"q":"hi"}`, 100, 10))
				}
				steps = append(steps,
					scriptStep{err: fmt.Errorf("provider refused: %w", llmkit.ErrContextTooLong)},
					textResp("recovered answer", 90, 9))
				return steps
			},
			opts: func(llmkit.Client, []Tool) []Option {
				return []Option{WithLimits(Limits{HistoryTokenBudget: 1})}
			},
		},
	}
}

// callIDsInCut returns every non-denied ToolRun call id of evs, and whether
// the id's ToolRun is inside the cut prefix.
func callIDsInCut(evs, cut []llmkit.Event) map[string]bool {
	inCut := map[string]bool{}
	for _, ev := range cut {
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && !ev.ToolRun.Denied {
			inCut[ev.ToolRun.Call.ID] = true
		}
	}
	out := map[string]bool{}
	for _, ev := range evs {
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && !ev.ToolRun.Denied {
			out[ev.ToolRun.Call.ID] = inCut[ev.ToolRun.Call.ID]
		}
	}
	return out
}

func TestResumeContinuationAtEveryCut(t *testing.T) {
	for _, sc := range resumeScenarios() {
		t.Run(sc.name, func(t *testing.T) {
			ctx := context.Background()
			origClient := newFakeClient(sc.steps()...)
			origClient.caps = sc.caps
			orig := NewRunner(origClient, sc.tools(), sc.system, sc.opts(origClient, nil)...)
			runID := llmkit.RunID("resume-" + sc.name)
			var origOut *Outcome
			var origErr error
			if sc.schema != nil {
				origOut, origErr = orig.RunJSON(ctx, sc.task, sc.schema, sc.out(), WithRunID(runID))
			} else {
				origOut, origErr = orig.Run(ctx, sc.task, WithRunID(runID))
			}
			if origOut == nil || origOut.Transcript == nil {
				t.Fatalf("original run: no outcome (err %v)", origErr)
			}
			events := origOut.Transcript.Record
			if sc.name == "stub-compaction" {
				n := 0
				for _, ev := range events {
					if ev.Kind == llmkit.KindCompaction {
						n++
					}
				}
				if n == 0 {
					t.Fatalf("scenario %s recorded no compaction; the case is vacuous", sc.name)
				}
			}
			if sc.name == "overflow-recovery" {
				n := 0
				for _, ev := range events {
					if ev.Kind == llmkit.KindCompletion && ev.Completion != nil &&
						ev.Completion.ErrKind == llmkit.ErrKindContextTooLong {
						n++
					}
				}
				if n == 0 {
					t.Fatalf("scenario %s recorded no context-overflow completion; the case is vacuous", sc.name)
				}
			}
			if fin := finalizeOf(events); fin != nil && fin.Status != llmkit.RunPanicked {
				if got, want := summedUsage(events), fin.Usage; got != want {
					t.Fatalf("scenario %s: summed completion usage %+v != finalize usage %+v", sc.name, got, want)
				}
			}
			wantReqs := servableRequests(events)
			total := len(wantReqs)

			for cut := 1; cut <= len(events); cut++ {
				cutEvents := events[:cut]
				k := len(servableCompletions(cutEvents))
				for _, src := range cutSources(t, runID, cutEvents) {
					t.Run(fmt.Sprintf("cut=%d/src=%s", cut, src.name), func(t *testing.T) {
						if k == 0 {
							_, err := NewResume(ctx, src.src, runID, NewRunner(newFakeClient(), nil, sc.system))
							if !errors.Is(err, ErrResumeMismatch) {
								t.Fatalf("record with no completion: err = %v, want ErrResumeMismatch", err)
							}
							return
						}
						live := newFakeFromCompletions(servableCompletions(events)[k:])
						live.caps = sc.caps
						liveClient := liveFake{live, llmkit.Identity{Provider: "live-provider", Model: "live-model"}}
						tools := wrapCounting(sc.tools())
						template := NewRunner(liveClient, asTools(tools), sc.system, sc.opts(liveClient, asTools(tools))...)
						res, err := NewResume(ctx, src.src, runID, template)
						if err != nil {
							t.Fatalf("NewResume: %v", err)
						}
						if got, want := res.Usage(), summedUsage(cutEvents); got != want {
							t.Errorf("Usage() = %+v, want %+v", got, want)
						}
						var out *Outcome
						if sc.schema != nil {
							out, err = res.RunJSON(ctx, sc.task, sc.schema, sc.out())
						} else {
							out, err = res.Run(ctx, sc.task)
						}
						if err != nil && !errors.Is(err, ErrResumeMismatch) && origErr == nil {
							t.Fatalf("resumed run: %v", err)
						}
						if out == nil {
							t.Fatalf("resumed run: no outcome (%v)", err)
						}
						if got, want := live.callCount(), total-k; got != want {
							t.Errorf("live client calls = %d, want %d (record served %d of %d)", got, want, k, total)
						}
						if k < total && live.callCount() > 0 {
							if derr := requestMatches(wantReqs[k], live.requests[0]); derr != nil {
								t.Errorf("first live request differs from the original's next request: %v", derr)
							}
						}
						if out.FinalText != origOut.FinalText {
							t.Errorf("FinalText = %q, want %q", out.FinalText, origOut.FinalText)
						}
						if out.TruncationReason != origOut.TruncationReason {
							t.Errorf("TruncationReason = %q, want %q", out.TruncationReason, origOut.TruncationReason)
						}
						if (err == nil) != (origErr == nil) {
							t.Errorf("resumed err = %v, original err = %v", err, origErr)
						}
						wantFin, gotFin := finalizeOf(events), finalizeOf(out.Transcript.Record)
						if wantFin == nil || gotFin == nil {
							t.Fatalf("missing finalize: original %v resumed %v", wantFin, gotFin)
						}
						if gotFin.Status != wantFin.Status {
							t.Errorf("finalize status = %q, want %q", gotFin.Status, wantFin.Status)
						}
						for _, ev := range out.Transcript.Record {
							if ev.ParentRunID != runID {
								t.Errorf("event %s lacks ParentRunID %s (got %q)", ev.Kind, runID, ev.ParentRunID)
								break
							}
						}
						for id, recorded := range callIDsInCut(events, cutEvents) {
							if n := toolRunCount(tools, id); recorded && n != 0 {
								t.Errorf("recorded call %q ran live %d times, want 0", id, n)
							} else if !recorded && n != 1 {
								t.Errorf("unrecorded call %q ran live %d times, want 1", id, n)
							}
						}
					})
				}
			}
		})
	}
}

func wrapCounting(tools []Tool) []*resumeCountingTool {
	w := make([]*resumeCountingTool, len(tools))
	for i, tool := range tools {
		w[i] = &resumeCountingTool{Tool: tool}
	}
	return w
}

func toolRunCount(tools []*resumeCountingTool, id string) int {
	n := 0
	for _, t := range tools {
		n += t.count(id)
	}
	return n
}

// --- R2: configuration mismatch --------------------------------------------

func TestResumeMismatchRefusedBeforeLiveCall(t *testing.T) {
	ctx := context.Background()
	origClient := newFakeClient(
		toolResp("c1", "echo", `{"q":"hi"}`, 100, 10),
		textResp("final answer", 80, 8),
	)
	orig := NewRunner(origClient, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "Original system.")
	runID := llmkit.RunID("resume-mismatch")
	origOut, err := orig.Run(ctx, "original task", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	src := &Transcript{Record: origOut.Transcript.Record, RunID: runID}
	otherSchema := func() []Tool {
		return []Tool{resumeBlocksTool{name: "echo", blocks: []llmkit.Block{llmkit.Text("echo-result")}}}
	}
	cases := []struct {
		name   string
		system string
		tools  func() []Tool
		task   string
	}{
		{"system", "A different system.", nil, "original task"},
		{"tool-schema", "Original system.", otherSchema, "original task"},
		{"tool-names", "Original system.", nil, "original task"},
		{"task", "Original system.", nil, "a different task"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tools := tc.tools
			if tools == nil {
				tools = func() []Tool { return []Tool{fixedTool{name: "echo", result: "echo-result"}} }
			}
			if tc.name == "tool-names" {
				tools = func() []Tool { return []Tool{fixedTool{name: "renamed", result: "echo-result"}} }
			}
			live := newFakeClient(textResp("never", 1, 1))
			template := NewRunner(live, tools(), tc.system)
			res, err := NewResume(ctx, src, runID, template)
			if err != nil {
				t.Fatalf("NewResume: %v", err)
			}
			out, err := res.Run(ctx, tc.task)
			if !errors.Is(err, ErrResumeMismatch) {
				t.Fatalf("res.Run err = %v, want ErrResumeMismatch", err)
			}
			if n := live.callCount(); n != 0 {
				t.Errorf("live client calls = %d, want 0", n)
			}
			if out == nil {
				t.Errorf("res.Run returned a nil outcome beside its error")
			}
		})
	}
}

// --- R3: dangling calls -----------------------------------------------------

func TestResumeDanglingCalls(t *testing.T) {
	ctx := context.Background()
	steps := func() []scriptStep {
		return []scriptStep{
			twoToolResp("c1", "c2", "echo", 100, 10),
			textResp("after two calls", 80, 8),
		}
	}
	origClient := newFakeClient(steps()...)
	orig := NewRunner(origClient, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.")
	runID := llmkit.RunID("resume-dangling")
	origOut, err := orig.Run(ctx, "call echo twice", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	events := origOut.Transcript.Record
	// cutAfterToolTurn keeps everything through the completion that requested
	// the calls plus n of its tool runs; n=0 cuts right after that completion.
	cutAfterToolTurn := func(n int) []llmkit.Event {
		cut := 0
		seen := 0
		for i, ev := range events {
			if n == 0 && ev.Kind == llmkit.KindCompletion {
				return events[:i+1]
			}
			cut = i + 1
			if ev.Kind == llmkit.KindToolRun {
				seen++
				if seen == n {
					break
				}
			}
		}
		return events[:cut]
	}
	for _, n := range []int{0, 1, 2} {
		t.Run(fmt.Sprintf("recorded=%d-of-2", n), func(t *testing.T) {
			cutEvents := cutAfterToolTurn(n)
			live := newFakeFromResponses(servableResponses(events)[1:])
			tools := wrapCounting([]Tool{fixedTool{name: "echo", result: "echo-result"}})
			template := NewRunner(live, asTools(tools), "You test resume.")
			res, err := NewResume(ctx, &Transcript{Record: cutEvents, RunID: runID}, runID, template)
			if err != nil {
				t.Fatalf("NewResume: %v", err)
			}
			out, err := res.Run(ctx, "call echo twice")
			if err != nil {
				t.Fatalf("resumed run: %v", err)
			}
			if got, want := live.callCount(), 1; got != want {
				t.Errorf("live client calls = %d, want %d", got, want)
			}
			for id, want := range map[string]int{"c1": 1, "c2": 1} {
				if got := toolRunCount(tools, id); got != want-n0(cutEvents, id) {
					t.Errorf("call %q ran live %d times (recorded %v)", id, got, n0(cutEvents, id) > 0)
				}
			}
			if out.FinalText != origOut.FinalText {
				t.Errorf("FinalText = %q, want %q", out.FinalText, origOut.FinalText)
			}
			for _, ev := range out.Transcript.Record {
				if ev.ParentRunID != runID {
					t.Fatalf("event %s lacks ParentRunID", ev.Kind)
				}
			}
		})
	}
}

func n0(evs []llmkit.Event, id string) int {
	for _, ev := range evs {
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && ev.ToolRun.Call.ID == id {
			return 1
		}
	}
	return 0
}

// --- R4: usage ---------------------------------------------------------------

func TestResumeUsageOfPanickedRecord(t *testing.T) {
	ctx := context.Background()
	origClient := newFakeClient(thinkOnlyResp("only thinking", 120, 30), textResp("never recorded", 10, 2))
	runID := llmkit.RunID("resume-panicked")
	tr := NewTranscript()
	orig2 := NewRunner(origClient, nil, "You test resume.", WithObserver(tr), WithHooks(Hooks{
		AfterCompletion: func(_ context.Context, step int, _ *llmkit.Request, _ *llmkit.Response, _ error) {
			if step == 2 {
				panic("hook bug")
			}
		},
	}))
	func() {
		defer func() { _ = recover() }()
		_, _ = orig2.Run(ctx, "task", WithRunID(runID))
	}()
	tr.RunID = runID
	events := tr.Record
	fin := finalizeOf(events)
	if fin == nil || fin.Status != llmkit.RunPanicked {
		t.Fatalf("record has no panicked finalize: %+v", fin)
	}
	if fin.Usage != (llmkit.Usage{}) {
		t.Fatalf("panicked finalize usage = %+v, want zero", fin.Usage)
	}
	res, err := NewResume(ctx, tr, runID, NewRunner(newFakeClient(), nil, "You test resume."))
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	if got, want := res.Usage(), (llmkit.Usage{InputTokens: 120, OutputTokens: 30}); got != want {
		t.Errorf("Usage() = %+v, want %+v", got, want)
	}
}

// --- R5 + P-B: budgets -------------------------------------------------------

func TestResumeTokenBudgetNotRearmed(t *testing.T) {
	ctx := context.Background()
	steps := []scriptStep{
		toolResp("c1", "echo", `{"q":"1"}`, 100, 5),
		toolResp("c2", "echo", `{"q":"2"}`, 100, 5),
		toolResp("c3", "echo", `{"q":"3"}`, 100, 5),
		textResp("final answer", 100, 5),
	}
	newRunner := func(c llmkit.Client) *Runner {
		return NewRunner(c, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.",
			WithLimits(Limits{TokenBudget: 250}))
	}
	runID := llmkit.RunID("resume-budget")
	origOut, origErr := func() (*Outcome, error) {
		out, err := newRunner(newFakeClient(steps...)).Run(ctx, "call echo repeatedly", WithRunID(runID))
		return out, err
	}()
	var inc *IncompleteError
	if !errors.As(origErr, &inc) || inc.Reason != TruncTokenBudget {
		t.Fatalf("original run err = %v, want TruncTokenBudget", origErr)
	}
	if origOut.Iterations != 3 {
		t.Fatalf("original iterations = %d, want 3 (stopped after its third completion)", origOut.Iterations)
	}
	events := origOut.Transcript.Record
	cut := 0
	for i, ev := range events {
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && ev.ToolRun.Call.ID == "c2" {
			cut = i + 1
		}
	}
	if cut == 0 {
		t.Fatalf("record has no tool run for c2")
	}
	live := newFakeFromResponses(servableResponses(events)[2:])
	template := newRunner(live)
	res, err := NewResume(ctx, &Transcript{Record: events[:cut], RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	out, err := res.Run(ctx, "call echo repeatedly")
	if !errors.Is(err, ErrResumeMismatch) {
		var inc2 *IncompleteError
		if !errors.As(err, &inc2) || inc2.Reason != TruncTokenBudget {
			t.Fatalf("resumed run err = %v, want TruncTokenBudget at the original's stopping point", err)
		}
	}
	if got, want := live.callCount(), 1; got != want {
		t.Errorf("live client calls = %d, want %d (the original's third completion, live)", got, want)
	}
	if out == nil {
		t.Fatalf("no outcome")
	}
	if out.Iterations != origOut.Iterations {
		t.Errorf("iterations = %d, want %d", out.Iterations, origOut.Iterations)
	}
	// The prefix is served with zeroed usage: Finalize.Usage counts live spend only.
	if fin := finalizeOf(out.Transcript.Record); fin != nil && fin.Usage.InputTokens != 100 {
		t.Errorf("resumed finalize usage = %+v, want the live completion's usage only", fin.Usage)
	}
}

func TestResumeBudgetPoolChargedLiveOnly(t *testing.T) {
	ctx := context.Background()
	steps := []scriptStep{
		toolResp("c1", "echo", `{"q":"1"}`, 100, 5),
		toolResp("c2", "echo", `{"q":"2"}`, 100, 5),
		toolResp("c3", "echo", `{"q":"3"}`, 100, 5),
		textResp("final answer", 100, 5),
	}
	origPool := NewBudgetPool(100000)
	runID := llmkit.RunID("resume-pool")
	orig := NewRunner(newFakeClient(steps...), []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.",
		WithBudgetPool(origPool))
	origOut, err := orig.Run(ctx, "call echo repeatedly", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	events := origOut.Transcript.Record
	if got, want := origPool.Spent(), int64(420); got != want {
		t.Fatalf("original pool spend = %d, want %d", got, want)
	}
	cut := 0
	for i, ev := range events {
		if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Completion.Err == "" {
			cut = i + 1
			break
		}
	}
	for _, tc := range []struct {
		name      string
		limit     int64
		preCharge int64
		wantStops bool
		wantLive  int
	}{{
		name:     "live-only ceiling",
		limit:    250,
		wantLive: 3,
	}, {
		name:      "pre-charged prefix crosses",
		limit:     250,
		preCharge: 105,
		wantStops: true,
		wantLive:  2,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			live := newFakeFromResponses(servableResponses(events)[1:])
			pool := NewBudgetPool(tc.limit)
			pool.Add(tc.preCharge)
			template := NewRunner(live, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.",
				WithBudgetPool(pool))
			res, err := NewResume(ctx, &Transcript{Record: events[:cut], RunID: runID}, runID, template)
			if err != nil {
				t.Fatalf("NewResume: %v", err)
			}
			out, err := res.Run(ctx, "call echo repeatedly")
			if got := live.callCount(); got != tc.wantLive {
				t.Errorf("live client calls = %d, want %d (err %v)", got, tc.wantLive, err)
			}
			if tc.wantStops {
				var inc *IncompleteError
				if !errors.As(err, &inc) || inc.Reason != TruncBudgetPool {
					t.Fatalf("err = %v, want TruncBudgetPool", err)
				}
			} else if err != nil {
				t.Fatalf("resumed run: %v", err)
			}
			_ = out
		})
	}
}

// --- P-A: steering refusal ----------------------------------------------------

func TestResumeRefusesSteeringRecord(t *testing.T) {
	ctx := context.Background()
	steering := NewSteering()
	steering.FollowUp(llmkit.Text("Now answer more briefly."))
	origClient := newFakeClient(textResp("first answer", 100, 10), textResp("second answer", 80, 8))
	orig := NewRunner(origClient, nil, "You test resume.")
	runID := llmkit.RunID("resume-steer")
	origOut, err := orig.Run(ctx, "answer twice", WithRunID(runID), WithSteering(steering))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	events := origOut.Transcript.Record
	steerAt := -1
	for i, ev := range events {
		if ev.Kind == llmkit.KindSteer {
			steerAt = i
		}
	}
	if steerAt < 0 {
		t.Fatalf("record holds no steer event")
	}
	for cut := 1; cut <= len(events); cut++ {
		live := newFakeFromResponses(servableResponses(events))
		template := NewRunner(live, nil, "You test resume.")
		res, err := NewResume(ctx, &Transcript{Record: events[:cut], RunID: runID}, runID, template)
		if cut > steerAt || len(servableCompletions(events[:cut])) == 0 {
			if !errors.Is(err, ErrResumeMismatch) {
				t.Errorf("cut=%d: NewResume err = %v, want ErrResumeMismatch", cut, err)
			}
			if n := live.callCount(); n != 0 {
				t.Errorf("cut=%d: live client calls = %d, want 0", cut, n)
			}
			continue
		}
		if err != nil {
			t.Fatalf("cut=%d (before the steer): NewResume: %v", cut, err)
		}
		out, err := res.Run(ctx, "answer twice")
		if err != nil {
			t.Fatalf("cut=%d (before the steer): resumed run: %v", cut, err)
		}
		// No steer is re-injected: the resumed run ends at the recorded turn.
		if out.FinalText != "first answer" || live.callCount() != 0 {
			t.Errorf("cut=%d: FinalText = %q live calls = %d; want the recorded turn to end the run", cut, out.FinalText, live.callCount())
		}
	}
}

// --- divergence refusals ------------------------------------------------------

func TestResumeTimeVaryingPolicyRefused(t *testing.T) {
	ctx := context.Background()
	policy := RequestPolicyFunc(func(_ context.Context, _ int, req *llmkit.Request) error {
		req.System += " run " + fmt.Sprintf("%p", &req) // varies per construction
		return nil
	})
	orig := NewRunner(newFakeClient(textResp("answer", 100, 10)), nil, "You test resume.", WithRequestPolicy(policy))
	runID := llmkit.RunID("resume-timevary")
	origOut, err := orig.Run(ctx, "task", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	live := newFakeClient(textResp("never", 1, 1))
	template := NewRunner(live, nil, "You test resume.", WithRequestPolicy(policy))
	res, err := NewResume(ctx, &Transcript{Record: origOut.Transcript.Record, RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	if _, err := res.Run(ctx, "task"); !errors.Is(err, ErrResumeMismatch) {
		t.Fatalf("err = %v, want ErrResumeMismatch", err)
	}
	if n := live.callCount(); n != 0 {
		t.Errorf("live client calls = %d, want 0", n)
	}
}

func TestResumeStrictMatchFailsClosedOnUnencodableRequest(t *testing.T) {
	ctx := context.Background()
	orig := NewRunner(newFakeClient(textResp("answer", 100, 10)), nil, "You test resume.")
	runID := llmkit.RunID("resume-unencodable")
	origOut, err := orig.Run(ctx, "task", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	live := newFakeClient(textResp("never", 1, 1))
	template := NewRunner(live, nil, "You test resume.", WithRequestPolicy(RequestPolicyFunc(
		func(_ context.Context, _ int, req *llmkit.Request) error {
			req.ResponseSchema = json.RawMessage(`{"not json`)
			return nil
		})))
	res, err := NewResume(ctx, &Transcript{Record: origOut.Transcript.Record, RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	_, err = res.Run(ctx, "task")
	if !errors.Is(err, ErrResumeMismatch) {
		t.Fatalf("err = %v, want ErrResumeMismatch", err)
	}
	if !strings.Contains(err.Error(), "cannot be encoded") {
		t.Errorf("err = %v, want the fail-closed encoding message", err)
	}
	if n := live.callCount(); n != 0 {
		t.Errorf("live client calls = %d, want 0", n)
	}
}

// --- template integrity -------------------------------------------------------

func TestNewResumeDoesNotMutateRunner(t *testing.T) {
	ctx := context.Background()
	origClient := newFakeClient(
		toolResp("c1", "echo", `{"q":"hi"}`, 100, 10),
		textResp("first done", 80, 8),
	)
	orig := NewRunner(origClient, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.")
	runID := llmkit.RunID("resume-nomutate")
	origOut, err := orig.Run(ctx, "use echo", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	live := newFakeClient(textResp("first done", 80, 8), textResp("plain run answer", 50, 5))
	template := NewRunner(live, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.")
	defsBefore := append([]llmkit.ToolDef(nil), template.tools.defs...)
	clientBefore := template.client
	cut := 0
	for i, ev := range origOut.Transcript.Record {
		if ev.Kind == llmkit.KindToolRun {
			cut = i + 1
		}
	}
	res, err := NewResume(ctx, &Transcript{Record: origOut.Transcript.Record[:cut], RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	if _, err := res.Run(ctx, "use echo"); err != nil {
		t.Fatalf("resumed run: %v", err)
	}
	if len(template.tools.defs) != len(defsBefore) || template.tools.defs[0].Name != defsBefore[0].Name || string(template.tools.defs[0].Parameters) != string(defsBefore[0].Parameters) {
		t.Errorf("template tool defs changed: %+v", template.tools.defs)
	}
	if template.client != clientBefore {
		t.Errorf("template client replaced")
	}
	out, err := template.Run(ctx, "a plain second task")
	if err != nil {
		t.Fatalf("plain run on the template after resume: %v", err)
	}
	if out.FinalText != "plain run answer" {
		t.Errorf("plain run FinalText = %q", out.FinalText)
	}
	if got, want := live.callCount(), 2; got != want {
		t.Errorf("live client calls = %d, want %d", got, want)
	}
	for _, ev := range out.Transcript.Record {
		if ev.ParentRunID != "" {
			t.Fatalf("plain run event %s carries ParentRunID %q", ev.Kind, ev.ParentRunID)
		}
	}
}

func TestResumeRunsOnce(t *testing.T) {
	ctx := context.Background()
	orig := NewRunner(newFakeClient(textResp("answer", 100, 10)), nil, "You test resume.")
	runID := llmkit.RunID("resume-once")
	origOut, err := orig.Run(ctx, "task", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	res, err := NewResume(ctx, &Transcript{Record: origOut.Transcript.Record, RunID: runID},
		runID, NewRunner(newFakeClient(), nil, "You test resume."))
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	if _, err := res.Run(ctx, "task"); err != nil {
		t.Fatalf("first run: %v", err)
	}
	if _, err := res.Run(ctx, "task"); err == nil || !strings.Contains(err.Error(), "runs once") {
		t.Fatalf("second run err = %v, want the runs-once refusal", err)
	}
}

// --- interface forwarding -----------------------------------------------------

type streamingFake struct {
	*fakeClient
	id llmkit.Identity
}

func (f *streamingFake) Identity() llmkit.Identity { return f.id }

func (f *streamingFake) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	_ = fn(llmkit.Delta{Kind: llmkit.DeltaText, Text: "native-delta"})
	return f.Complete(ctx, req)
}

func TestResumeForwardsClientInterfaces(t *testing.T) {
	ctx := context.Background()
	first := toolResp("c1", "echo", `{"q":"hi"}`, 100, 10)
	first.resp.Served = llmkit.Identity{Provider: "orig-provider", Model: "orig-model"}
	final := textResp("streamed answer", 80, 8)
	orig := NewRunner(newFakeClient(first, final), []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.")
	runID := llmkit.RunID("resume-interfaces")
	origOut, err := orig.Run(ctx, "use echo", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	events := origOut.Transcript.Record
	cut := 0
	for i, ev := range events {
		if ev.Kind == llmkit.KindToolRun {
			cut = i + 1
		}
	}
	liveFinal := textResp("streamed answer", 80, 8)
	live := &streamingFake{newFakeClient(liveFinal), llmkit.Identity{Provider: "live-provider", Model: "live-model"}}
	var deltas []llmkit.Delta
	var deltaSteps []int
	template := NewRunner(live, []Tool{fixedTool{name: "echo", result: "echo-result"}}, "You test resume.",
		WithHooks(Hooks{Delta: func(_ context.Context, step int, d llmkit.Delta) {
			deltas = append(deltas, d)
			deltaSteps = append(deltaSteps, step)
		}}))
	res, err := NewResume(ctx, &Transcript{Record: events[:cut], RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	out, err := res.Run(ctx, "use echo")
	if err != nil {
		t.Fatalf("resumed run: %v", err)
	}
	sawNative := false
	for i, d := range deltas {
		if d.Text == "native-delta" {
			sawNative = true
			if deltaSteps[i] != 2 {
				t.Errorf("native delta at step %d, want the live step 2", deltaSteps[i])
			}
		}
	}
	if !sawNative {
		t.Errorf("no native live delta reached the hook: %+v", deltas)
	}
	for _, ev := range out.Transcript.Record {
		if ev.Kind != llmkit.KindCompletion || ev.Completion == nil {
			continue
		}
		switch ev.Step {
		case 1:
			if ev.Completion.Provider != "orig-provider" || ev.Completion.Model != "orig-model" {
				t.Errorf("prefix completion names %s/%s, want the recorded orig-provider/orig-model",
					ev.Completion.Provider, ev.Completion.Model)
			}
		case 2:
			if ev.Completion.Provider != "live-provider" || ev.Completion.Model != "live-model" {
				t.Errorf("live completion names %s/%s, want the forwarded live-provider/live-model",
					ev.Completion.Provider, ev.Completion.Model)
			}
		}
	}
}

// --- summarizing compactor ----------------------------------------------------

type countingCompactor struct {
	calls int
}

func (c *countingCompactor) Compact(_ context.Context, in CompactInput) (CompactOutput, error) {
	c.calls++
	from, at := summaryCut(in.Messages)
	if at < 0 {
		return stubCompactor{}.Compact(context.Background(), in)
	}
	out := summarized(in, from, at, "LIVE SUMMARY")
	out.Usage = llmkit.Usage{InputTokens: 5, OutputTokens: 5}
	return out, nil
}

func TestResumeServesRecordedSummariesThenLiveCompactor(t *testing.T) {
	ctx := context.Background()
	// Client call order: c1, c2, summary one (before c3), c3, summary two
	// (before the final turn), final. The summarizer pays its own calls from
	// the same script.
	origClient := newFakeClient(
		toolResp("c1", "echo", `{"q":"1"}`, 100, 10),
		toolResp("c2", "echo", `{"q":"2"}`, 90, 10),
		scriptStep{resp: llmkit.Response{Text: "recorded summary one", StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 30, OutputTokens: 10}}},
		toolResp("c3", "echo", `{"q":"3"}`, 95, 10),
		scriptStep{resp: llmkit.Response{Text: "recorded summary two", StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 25, OutputTokens: 10}}},
		textResp("final answer", 80, 8),
	)
	orig := NewRunner(origClient, []Tool{resumeBigResultTool("echo")}, "You test resume.",
		WithCompactor(SummarizingCompactor(origClient)), WithLimits(Limits{HistoryTokenBudget: 1}))
	runID := llmkit.RunID("resume-summary")
	origOut, err := orig.Run(ctx, "call echo three times then answer", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	events := origOut.Transcript.Record
	summaries := 0
	for _, ev := range events {
		if ev.Kind == llmkit.KindCompaction && ev.Compaction != nil && ev.Compaction.Summary != "" {
			summaries++
		}
	}
	if summaries != 2 {
		t.Fatalf("original recorded %d summary passes, want 2", summaries)
	}
	// Cut after c3's tool run: summary one is inside the record, summary
	// two's position is past it, so the live compactor runs there and only
	// there — never during the prefix.
	cut := 0
	for i, ev := range events {
		if ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil && ev.ToolRun.Call.ID == "c3" {
			cut = i + 1
		}
	}
	if cut == 0 {
		t.Fatalf("record has no tool run for c3")
	}
	live := newFakeFromResponses(servableResponses(events)[3:])
	liveComp := &countingCompactor{}
	template := NewRunner(live, []Tool{resumeBigResultTool("echo")}, "You test resume.",
		WithCompactor(liveComp), WithLimits(Limits{HistoryTokenBudget: 1}))
	res, err := NewResume(ctx, &Transcript{Record: events[:cut], RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	out, err := res.Run(ctx, "call echo three times then answer")
	if err != nil {
		t.Fatalf("resumed run: %v", err)
	}
	if got, want := liveComp.calls, 1; got != want {
		t.Errorf("live compactor calls = %d, want %d (one live pass, none during the prefix)", got, want)
	}
	if got, want := live.callCount(), 1; got != want {
		t.Errorf("live client calls = %d, want %d", got, want)
	}
	if !strings.Contains(requestText(live.requests[0]), "LIVE SUMMARY") {
		t.Errorf("the live pass's summary never reached the first live request")
	}
	if got, want := firstSummaryText(events[:cut]), "recorded summary one"; got != want {
		t.Errorf("recorded summary = %q, want %q", got, want)
	}
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompaction && ev.Compaction != nil && ev.Compaction.Summary != "" &&
			ev.Step <= 3 && ev.Compaction.Summary != "recorded summary one" {
			t.Errorf("a prefix compaction carried %q, want the recorded summary", ev.Compaction.Summary)
		}
	}
	if out.FinalText != origOut.FinalText {
		t.Errorf("FinalText = %q, want %q", out.FinalText, origOut.FinalText)
	}
}

func firstSummaryText(evs []llmkit.Event) string {
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompaction && ev.Compaction != nil && ev.Compaction.Summary != "" {
			return ev.Compaction.Summary
		}
	}
	return ""
}

func requestText(req llmkit.Request) string {
	var b strings.Builder
	for _, m := range req.Messages {
		b.WriteString(m.Text())
	}
	return b.String()
}

// A live call reusing a recorded call id (providers repeat ids across turns)
// must reach the caller's policy and tool, never the record.
func TestResumeLiveCallReusingRecordedIDRunsLive(t *testing.T) {
	ctx := context.Background()
	rewrite := WithToolPolicy(ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		var q struct {
			Q string `json:"q"`
		}
		if err := json.Unmarshal(call.Arguments, &q); err != nil {
			return err
		}
		call.Arguments = json.RawMessage(fmt.Sprintf(`{"q":%q,"extra":true}`, q.Q))
		return nil
	}))
	origTool := &argTool{name: "echo"}
	orig := NewRunner(newFakeClient(
		toolResp("c1", "echo", `{"q":"hi"}`, 100, 10),
		textResp("done", 80, 8),
	), []Tool{origTool}, "You test resume.", rewrite)
	runID := llmkit.RunID("resume-idreuse")
	origOut, err := orig.Run(ctx, "use echo", WithRunID(runID))
	if err != nil {
		t.Fatalf("original run: %v", err)
	}
	events := origOut.Transcript.Record
	cut := 0
	for i, ev := range events {
		if ev.Kind == llmkit.KindToolRun {
			cut = i + 1
		}
	}
	liveTool := &argTool{name: "echo"}
	live := newFakeClient(
		toolResp("c1", "echo", `{"q":"live"}`, 100, 10),
		textResp("done twice", 80, 8),
	)
	template := NewRunner(live, []Tool{liveTool}, "You test resume.", rewrite)
	res, err := NewResume(ctx, &Transcript{Record: events[:cut], RunID: runID}, runID, template)
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	out, err := res.Run(ctx, "use echo")
	if err != nil {
		t.Fatalf("resumed run: %v", err)
	}
	if out.FinalText != "done twice" {
		t.Errorf("FinalText = %q, want %q", out.FinalText, "done twice")
	}
	origTool.mu.Lock()
	got := append([]json.RawMessage(nil), origTool.got...)
	origTool.mu.Unlock()
	if len(got) != 1 || string(got[0]) != `{"q":"hi","extra":true}` {
		t.Errorf("original tool args = %s, want the one recorded rewrite", got)
	}
	liveTool.mu.Lock()
	got = append([]json.RawMessage(nil), liveTool.got...)
	liveTool.mu.Unlock()
	if len(got) != 1 || string(got[0]) != `{"q":"live","extra":true}` {
		t.Errorf("live tool args = %s, want the caller policy's rewrite of the live call's own arguments", got)
	}
}

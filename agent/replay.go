package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/dpoage/llmkit"
)

// ErrUnknownRun reports that a [Source] has no record of the requested run.
// Match it with errors.Is.
var ErrUnknownRun = errors.New("agent: unknown run")

// ErrReplayDiverged reports that a replayed run's tool calls or wire
// requests no longer match the recorded sequence. A diverged replay FAILS
// [Runner.Run] with an error wrapping this sentinel instead of finishing
// with a wrong answer. Match it with errors.Is.
var ErrReplayDiverged = errors.New("agent: replay diverged")

// Source is the read side of recording: the ordered events of one run.
// [Transcript] and the JSONL sink ([JSONL]) implement it; store/sqlite does
// in a later round. [NewReplayClient] rebuilds a run's completions, and its
// [ReplayClient.Tools] serve the recorded tool results, from any Source —
// replay never depends on a JSONL file existing.
//
// Events returns the run's events in emission order; the caller owns the
// returned slice. A Source with no record of run returns an error wrapping
// [ErrUnknownRun] — never an empty success.
type Source interface {
	// Events returns the run's events in emission order. run is the id the
	// events carry ([llmkit.RunID]).
	Events(ctx context.Context, run llmkit.RunID) ([]llmkit.Event, error)
}

// ReplayClient is an [llmkit.Client] that serves a fixed sequence of recorded
// responses in order, instead of calling a real provider. It is the building
// block for offline evaluation: record a run once into any sink, then replay
// it deterministically against modified harness code.
//
// On each Complete call it returns the next recorded response. It validates
// that the request's tool-call structure matches what was recorded for that
// step (number of prior tool-result messages and the tool names involved),
// erroring if the sequence diverges. Matching is intentionally lenient on
// exact message text — only the tool-call structure must line up — so prompt
// wording can change between record and replay without breaking the replay.
//
// Divergence is run state owned by the client, never inferred from
// conversation text. [ReplayClient.Tools] returns the run's recorded tool
// set, bound to this client: each tool serves the recorded result for the
// earliest unconsumed (name, arguments) match within the current recorded
// step, so replay is deterministic under [WithParallelTools]; a call with no
// unconsumed match records the divergence on the client and is returned as
// that call's error (the harness renders it as data). Complete refuses to
// serve further responses once a divergence is recorded, and every other
// divergence path — structure mismatch, exhausted record — fails the same
// way, all wrapping [ErrReplayDiverged].
//
// A divergence in the run's FINAL tool turn can end the run before another
// Complete happens (a recording truncated at its step cap, or a run that
// finishes cleanly), so Run may return nil with the divergence still set.
// Callers MUST assert [ReplayClient.Err] returns nil after a replayed run —
// a diverged replay is a failed evaluation even when Run itself succeeded.
//
// ReplayClient is safe for concurrent use, though a single Runner calls it
// sequentially.
//
// ReplayClient implements only Complete. A Runner with [Hooks.Delta] set
// still delivers deltas: [llmkit.Stream] falls back to Complete and
// synthesizes deltas from the recorded Response's content blocks; a
// recorded response with no text block yields one DeltaText from
// Response.Text, so the hook fires — just not at wire granularity.
type ReplayClient struct {
	mu        sync.Mutex
	responses []replayStep
	idx       int
	caps      llmkit.Capabilities
	set       *replayToolSet
	tools     []Tool
	diverged  error
}

// replayStep pairs a recorded response with the tool-call structure expected
// of the request that should elicit it.
type replayStep struct {
	resp          llmkit.Response
	expectToolIDs []string
}

// NewReplayClient builds a ReplayClient from src's record of run. It replays
// the run's COMPLETION events only ([llmkit.KindCompletion], never Attempts,
// per the observability emission rule): each successful completion's Response
// is served in order, paired with the tool-call structure expected of the
// request that elicited it (the tool-call ids whose results precede it), so
// replay can drive the same tool round-trips deterministically. Failed
// completions (a Completion event with Err set) carried no response to serve
// and are skipped; a run that aborted at one therefore diverges at that point
// with the replay-exhausted error.
//
// caps is returned from Capabilities; pass a profile matching the model the
// run was recorded against (or the zero value if it doesn't matter for the
// code under test).
func NewReplayClient(src Source, run llmkit.RunID, caps llmkit.Capabilities) (*ReplayClient, error) {
	if src == nil {
		return nil, errors.New("agent: nil event source")
	}
	evs, err := src.Events(context.Background(), run)
	if err != nil {
		return nil, fmt.Errorf("agent: read events for run %s: %w", run, err)
	}
	rc := &ReplayClient{caps: caps, set: &replayToolSet{}}
	// Walk events, accumulating the tool-call ids recorded since the last
	// completion (for structure validation) and the ToolRun events grouped
	// by step (for the bound tool set).
	var pending []string
	for _, ev := range evs {
		switch {
		case ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil:
			pending = append(pending, ev.ToolRun.Call.ID)
			rc.set.add(ev.Step, *ev.ToolRun)
		case ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Completion.Err == "":
			rc.responses = append(rc.responses, replayStep{
				resp:          ev.Completion.Response,
				expectToolIDs: pending,
			})
			pending = nil
		}
	}
	if len(rc.responses) == 0 {
		return nil, fmt.Errorf("agent: run %s has no completions to replay", run)
	}
	rc.tools = rc.set.tools(rc)
	return rc, nil
}

// NewReplayClientFromResponses builds a ReplayClient that serves resps in
// order without any tool-call structure validation. Useful for hand-scripted
// tests.
func NewReplayClientFromResponses(resps []llmkit.Response, caps llmkit.Capabilities) *ReplayClient {
	rc := &ReplayClient{caps: caps, set: &replayToolSet{}}
	for _, r := range resps {
		rc.responses = append(rc.responses, replayStep{resp: r})
	}
	return rc
}

// Capabilities returns the configured capability profile.
func (rc *ReplayClient) Capabilities() llmkit.Capabilities { return rc.caps }

// Complete serves the next recorded response, validating tool-call structure.
// It also fails the replay loudly: if the request's trailing tool results
// carry a replay divergence, Complete returns an error wrapping [ErrReplayDiverged] instead of
// serving the next response — a diverged replay fails Run, it does not finish
// with a wrong answer.
func (rc *ReplayClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}

	rc.mu.Lock()
	defer rc.mu.Unlock()

	if rc.diverged != nil {
		// Already recorded: refuse to serve past it.
		return llmkit.Response{}, rc.diverged
	}
	if rc.idx >= len(rc.responses) {
		err := fmt.Errorf("%w after step %d: replay exhausted after %d response(s); request sequence diverged (extra completion call)", ErrReplayDiverged, rc.idx, len(rc.responses))
		rc.diverged = err
		return llmkit.Response{}, err
	}
	step := rc.responses[rc.idx]

	// Validate the tool-result structure leading into this request, when the
	// step carries expectations (i.e. built from a recorded run).
	if step.expectToolIDs != nil {
		got := trailingToolResultIDs(req.Messages, len(step.expectToolIDs))
		if err := matchToolIDs(rc.idx, step.expectToolIDs, got); err != nil {
			rc.diverged = err
			return llmkit.Response{}, err
		}
	}
	rc.idx++
	return step.resp, nil
}

// trailingToolResultIDs returns the ToolCallIDs of the last n tool-result
// messages in msgs, in order. If there are fewer than n, it returns all of
// them.
func trailingToolResultIDs(msgs []llmkit.Message, n int) []string {
	var ids []string
	for _, m := range msgs {
		if m.Role == llmkit.RoleToolResult {
			ids = append(ids, m.ToolCallID)
		}
	}
	if n >= 0 && len(ids) > n {
		ids = ids[len(ids)-n:]
	}
	return ids
}

// matchToolIDs compares expected and actual tool-result ID sequences, returning
// a descriptive divergence error if they differ.
func matchToolIDs(step int, want, got []string) error {
	if len(want) != len(got) {
		return fmt.Errorf("%w at step %d: expected %d preceding tool result(s) %v, got %d %v",
			ErrReplayDiverged, step+1, len(want), want, len(got), got)
	}
	for i := range want {
		if want[i] != got[i] {
			return fmt.Errorf("%w at step %d: tool result %d was %q, recorded %q",
				ErrReplayDiverged, step+1, i, got[i], want[i])
		}
	}
	return nil
}

// replayedToolCall is one recorded tool call: the model's call, the
// model-visible result, the error/denial marks, and the step the call
// belongs to.
type replayedToolCall struct {
	call       llmkit.ToolCall
	result     string
	isError    bool
	denied     bool
	denyReason string
	step       int
	used       bool
}

// replayStepCalls is one recorded tool turn: every ToolRun event sharing a
// step.
type replayStepCalls struct {
	step  int
	calls []replayedToolCall
}

// replayToolSet is a [ReplayClient]'s recorded tool calls, grouped by step in
// recorded order, plus the index of the step currently being consumed.
// Matching is by (name, arguments) over the UNCONSUMED calls of the current
// step — not a strict cursor — so replay is deterministic under
// [WithParallelTools] however the calls interleave; an exhausted step
// advances to the next recorded tool turn.
type replayToolSet struct {
	groups []replayStepCalls
	gi     int
}

// add records one ToolRun event into its step group.
func (ts *replayToolSet) add(step int, tre llmkit.ToolRunEvent) {
	if len(ts.groups) == 0 || ts.groups[len(ts.groups)-1].step != step {
		ts.groups = append(ts.groups, replayStepCalls{step: step})
	}
	g := &ts.groups[len(ts.groups)-1]
	g.calls = append(g.calls, replayedToolCall{
		call:       tre.Call,
		result:     tre.Result,
		isError:    tre.IsError,
		denied:     tre.Denied,
		denyReason: tre.DenyReason,
		step:       step,
	})
}

// tools returns one [Tool] per distinct recorded name, all reading the set.
// A run with no tool calls yields an empty (non-nil) slice.
func (ts *replayToolSet) tools(rc *ReplayClient) []Tool {
	var names []string
	seen := map[string]bool{}
	for _, g := range ts.groups {
		for _, c := range g.calls {
			if !seen[c.call.Name] {
				seen[c.call.Name] = true
				names = append(names, c.call.Name)
			}
		}
	}
	tools := make([]Tool, 0, len(names))
	for _, name := range names {
		tools = append(tools, replayTool{rc: rc, name: name})
	}
	return tools
}

// Tools returns the recorded run's tool set, bound to this client: each tool
// serves the recorded result for the matching (name, arguments) call instead
// of executing. Feed it to a [Runner] driven by this same client to replay a
// run fully offline — zero live tool executions. A run that recorded no tool
// calls yields an empty slice.
//
// A call with no unconsumed match records this client's divergence (see
// [ReplayClient.Err]) and is returned as the tool's error — the harness
// renders it as data, and the next Complete refuses to serve.
func (rc *ReplayClient) Tools() []Tool {
	if rc.tools == nil {
		return []Tool{}
	}
	return slices.Clone(rc.tools)
}

// recordDiverged records the first divergence; later ones keep the first.
// The caller holds rc.mu.
func (rc *ReplayClient) recordDiverged(err error) {
	if rc.diverged == nil {
		rc.diverged = err
	}
}

// Err returns the replay's first divergence, or nil when the replay matched
// its record throughout. A divergence in the run's FINAL tool turn can end
// the run before another Complete happens, so Run may return nil with a
// divergence still set: callers MUST assert Err() == nil after a replayed
// run. The error wraps [ErrReplayDiverged] and names the recorded step.
func (rc *ReplayClient) Err() error {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	return rc.diverged
}

// replayTool serves one recorded tool name out of the client's set.
type replayTool struct {
	rc   *ReplayClient
	name string
}

// Def declares the tool by name only: replay never re-describes a tool to a
// real model — the recorded completions already fixed what it "said".
func (t replayTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name}
}

// Run serves the earliest unconsumed recorded call of the current step
// matching both the name and the arguments. No match records the client's
// divergence — the returned error wraps [ErrReplayDiverged] naming the
// recorded step, and the harness renders it as this call's ERROR result.
func (t replayTool) Run(_ context.Context, args json.RawMessage) (string, error) {
	return t.rc.serveCall(t.name, args)
}

// serveCall is [replayTool.Run]'s body, under the client's lock.
func (rc *ReplayClient) serveCall(name string, args json.RawMessage) (string, error) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	set := rc.set
	for set != nil && set.gi < len(set.groups) {
		g := &set.groups[set.gi]
		match := -1
		for i := range g.calls {
			if !g.calls[i].used && g.calls[i].call.Name == name && argsEqual(g.calls[i].call.Arguments, args) {
				match = i
				break
			}
		}
		if match >= 0 {
			g.calls[match].used = true
			return serveRecorded(&g.calls[match])
		}
		// No match in this step. Still-unconsumed calls here mean the replay
		// genuinely diverged; a fully consumed step advances to the next
		// recorded tool turn.
		unconsumed := -1
		for i := range g.calls {
			if !g.calls[i].used {
				unconsumed = i
				break
			}
		}
		if unconsumed >= 0 {
			rec := &g.calls[unconsumed]
			err := fmt.Errorf("%w at step %d: tool %q called with arguments %s, recorded %s",
				ErrReplayDiverged, rec.step, name, argsString(args), argsString(rec.call.Arguments))
			rc.recordDiverged(err)
			return "", err
		}
		set.gi++
	}
	lastStep := 0
	if n := len(set.groups); n > 0 {
		lastStep = set.groups[n-1].step
	}
	err := fmt.Errorf("%w at step %d: extra tool call to %q with arguments %s; every recorded call is served", ErrReplayDiverged, lastStep, name, argsString(args))
	rc.recordDiverged(err)
	return "", err
}

// serveRecorded renders one recorded call's outcome the way the original run
// fed it to the model: a denial re-renders the denial text, an error strips
// the "ERROR: " prefix so the Runner's renderer re-adds it (the fed-back
// result is byte-identical and the history keeps the is_error mark), and a
// success returns the recorded result verbatim.
func serveRecorded(rec *replayedToolCall) (string, error) {
	switch {
	case rec.denied:
		return "", fmt.Errorf("tool %s denied: %s", rec.call.Name, rec.denyReason)
	case rec.isError:
		return "", errors.New(strings.TrimPrefix(rec.result, "ERROR: "))
	default:
		return rec.result, nil
	}
}

// argsString renders raw JSON arguments for a divergence message.
func argsString(args json.RawMessage) string {
	if len(args) == 0 {
		return "(none)"
	}
	return string(args)
}

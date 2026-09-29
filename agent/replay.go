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

// ErrReplayDiverged reports that a replayed run's tool calls or wire
// requests no longer match the recorded sequence. A diverged replay FAILS
// [Runner.Run] with an error wrapping this sentinel instead of finishing
// with a wrong answer. Match it with errors.Is.
var ErrReplayDiverged = errors.New("agent: replay diverged")

// ReplayClient is an [llmkit.Client] that serves a fixed sequence of recorded
// responses in order, instead of calling a real provider. It is the building
// block for offline evaluation: record a run once into any sink, then replay
// it deterministically against modified harness code.
//
// ReplayClient does not implement [llmkit.IdentifiedClient], so the
// `Completion` events the agent `Runner` emits for a replayed run
// carry empty Provider and Model — a replay is served by no
// provider.
//
// Complete returns the next recorded response, validating that the request's
// tool-call structure (the count and ids of preceding tool-result messages)
// matches what was recorded for that step. Matching is intentionally lenient
// on message text — only the tool-call structure must line up — so prompt
// wording can change between record and replay without breaking the replay.
//
// Divergence is run state owned by the client, never inferred from
// conversation text. [ReplayClient.Tools] returns the run's recorded tool
// set, bound to this client: each tool serves the recorded result for the
// earliest unconsumed (name, arguments) match within the current recorded
// step, so replay is deterministic under [WithParallelTools]; a call with no
// unconsumed match records the divergence on the client and is returned as
// that call's error (the harness renders it as data). Once a divergence is
// recorded, every subsequent divergence path — structure mismatch, exhausted
// record, tool-call mismatch — fails the same way, wrapping
// [ErrReplayDiverged]. Complete refuses to serve further responses past it.
//
// Calls a [ToolPolicy] denied in the recorded run are NOT served by the
// recorded tools: the recorded denial is a policy decision, not a tool
// result. Their names stay registered in Tools, so a caller's own policy
// still sees them. Install [WithToolPolicy] with [ReplayClient.ToolPolicy]
// to reproduce the recorded denials, or with a policy of your own to decide
// afresh.
//
// Tools other than [ReplayClient.Tools] execute live, and their results are
// never compared with the record.
//
// A divergence in the run's FINAL tool turn can end the run before another
// Complete happens: a recording that stopped at its step cap replays to the
// same limit and Run returns an [*IncompleteError]. [ReplayClient.Err] is
// the ONLY report of it: callers MUST assert it returns nil after a
// replayed run — a diverged replay is a failed evaluation whatever Run
// returned.
//
// ReplayClient is safe for concurrent use, though a single Runner calls it
// sequentially.
//
// ReplayClient implements only Complete. With [Hooks.Delta] set, [llmkit.Stream]
// falls back to Complete and synthesizes deltas from the recorded Response's
// content blocks; a recorded response with no text block yields one DeltaText
// from Response.Text, so the hook fires — just not at wire granularity.
type ReplayClient struct {
	mu        sync.Mutex
	responses []replayStep
	idx       int
	caps      llmkit.Capabilities
	set       *replayToolSet
	tools     []Tool
	diverged  error
}

type replayStep struct {
	resp          llmkit.Response
	expectToolIDs []string
}

// NewReplayClient builds a ReplayClient from src's record of run. It replays
// the run's COMPLETION events only ([llmkit.KindCompletion], never Attempts,
// per the observability emission rule): each successful completion's Response
// is served in order, paired with the tool-call ids recorded since the
// previous completion, so replay can drive the same tool round-trips
// deterministically. Failed completions (Completion with Err set) are
// skipped — a run that aborted at one therefore diverges there with the
// replay-exhausted error.
//
// caps is returned from Capabilities; pass a profile matching the recorded
// model (or zero when it does not matter for the code under test).
func NewReplayClient(src llmkit.Source, run llmkit.RunID, caps llmkit.Capabilities) (*ReplayClient, error) {
	if src == nil {
		return nil, errors.New("agent: nil event source")
	}
	evs, err := src.Events(context.Background(), run)
	if err != nil {
		return nil, fmt.Errorf("agent: read events for run %s: %w", run, err)
	}
	rc := &ReplayClient{caps: caps, set: &replayToolSet{}}
	// Walk events: accumulate tool-call ids since the last completion (for
	// structure validation) and group ToolRun events by step (for the bound
	// tool set).
	var pending []string
	for _, ev := range evs {
		switch {
		case ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil:
			pending = append(pending, ev.ToolRun.Call.ID)
			rc.set.add(ev.Step, len(rc.responses)-1, *ev.ToolRun)
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
// order. It is the scripted test double: no tool-call structure is
// validated, no recorded tool set exists (Tools is empty, ToolPolicy allows
// every call), and the client never diverges except by running past the
// last response. To replay a recorded run, use [NewReplayClient].
func NewReplayClientFromResponses(resps []llmkit.Response, caps llmkit.Capabilities) *ReplayClient {
	rc := &ReplayClient{caps: caps, set: &replayToolSet{}}
	for _, r := range resps {
		rc.responses = append(rc.responses, replayStep{resp: r})
	}
	return rc
}

func (rc *ReplayClient) Capabilities() llmkit.Capabilities { return rc.caps }

// Complete serves the next recorded response, validating tool-call
// structure; if the request's trailing tool results carry a replay
// divergence, it returns an error wrapping [ErrReplayDiverged] instead of
// serving the next response — a diverged replay fails Run, it does not
// finish with a wrong answer.
func (rc *ReplayClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}

	rc.mu.Lock()
	defer rc.mu.Unlock()

	if rc.diverged != nil {
		// Past a recorded divergence: refuse to serve further responses.
		return llmkit.Response{}, rc.diverged
	}
	if rc.idx >= len(rc.responses) {
		err := fmt.Errorf("%w after step %d: replay exhausted after %d response(s); request sequence diverged (extra completion call)", ErrReplayDiverged, rc.idx, len(rc.responses))
		rc.diverged = err
		return llmkit.Response{}, err
	}
	step := rc.responses[rc.idx]

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

// replayedToolCall is one recorded, served tool call, with its step and error mark.
type replayedToolCall struct {
	call    llmkit.ToolCall
	result  string
	isError bool
	step    int
	used    bool
}

// replayedDenial is one call a [ToolPolicy] denied in the recorded run.
type replayedDenial struct {
	call   llmkit.ToolCall
	reason string
}

// replayStepCalls is one recorded tool turn: the calls a tool must serve and
// the calls a policy must deny, plus resp, the index of the recorded response
// that requested them.
type replayStepCalls struct {
	step    int
	resp    int
	calls   []replayedToolCall
	denials []replayedDenial
}

// replayToolSet is a [ReplayClient]'s recorded tool calls, grouped by step
// in recorded order, plus the index of the step currently being consumed.
// Matching is by (name, arguments) over the UNCONSUMED calls of the current
// step — not a strict cursor — so replay is deterministic under
// [WithParallelTools] however the calls interleave; an exhausted step
// advances to the next recorded tool turn. Denied calls are held apart from
// the served ones: a denial is decided by the policy, so it never blocks a
// step from advancing, but its name is still registered.
type replayToolSet struct {
	groups []replayStepCalls
	gi     int
}

// add files one recorded ToolRun under step; resp is the index of the
// recorded response that requested it.
func (ts *replayToolSet) add(step, resp int, tre llmkit.ToolRunEvent) {
	if len(ts.groups) == 0 || ts.groups[len(ts.groups)-1].step != step {
		ts.groups = append(ts.groups, replayStepCalls{step: step, resp: resp})
	}
	g := &ts.groups[len(ts.groups)-1]
	if tre.Denied {
		g.denials = append(g.denials, replayedDenial{call: tre.Call, reason: tre.DenyReason})
		return
	}
	g.calls = append(g.calls, replayedToolCall{
		call:    tre.Call,
		result:  tre.Result,
		isError: tre.IsError,
		step:    step,
	})
}

// tools registers every recorded call name, denied-only names included, so
// a policy the caller installs sees every call the model made.
func (ts *replayToolSet) tools(rc *ReplayClient) []Tool {
	var names []string
	seen := map[string]bool{}
	note := func(name string) {
		if !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	for _, g := range ts.groups {
		for _, c := range g.calls {
			note(c.call.Name)
		}
		for _, d := range g.denials {
			note(d.call.Name)
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
// Every recorded call name is registered, including names the recorded
// policy only ever denied; denied calls themselves are never served (see
// [ReplayClient.ToolPolicy]).
//
// A call with no unconsumed match records this client's divergence (see
// [ReplayClient]) and is returned as the tool's error; the harness renders
// it as data, and the next Complete refuses to serve.
//
// Only these tools are compared with the record. Any other tool a caller
// passes to the Runner executes live, and its result is never compared.
func (rc *ReplayClient) Tools() []Tool {
	if rc.tools == nil {
		return []Tool{}
	}
	return slices.Clone(rc.tools)
}

// ToolPolicy returns a [ToolPolicy] that reproduces the recorded run's
// policy denials: it denies a call, with the recorded reason, if and only
// if its tool-call ID is the ID of a call the record denied in the current
// recorded tool turn, and allows every other call. Records carry call IDs
// by construction — the Runner cannot feed back a tool result for a call
// without one — and replayed responses carry the recorded IDs verbatim.
// The Runner renders the denial with its own denial renderer and marks the
// replayed ToolRun event Denied with that reason. Install it with
// [WithToolPolicy] when the record holds denials and the replay has no
// policy of its own.
//
// It never overrides a caller's policy: a caller that composes its own
// policy simply does not install this one. A record with no denials, or a
// zero-value ReplayClient, yields a policy that allows every call.
func (rc *ReplayClient) ToolPolicy() ToolPolicy {
	return ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		rc.mu.Lock()
		defer rc.mu.Unlock()
		if rc.set == nil {
			return nil
		}
		// The current turn is the one the last served response requested.
		for gi := range rc.set.groups {
			g := &rc.set.groups[gi]
			if g.resp != rc.idx-1 {
				continue
			}
			for _, d := range g.denials {
				if d.call.ID == call.ID {
					return errors.New(d.reason)
				}
			}
			break
		}
		return nil
	})
}

// recordDiverged keeps the first divergence; caller holds rc.mu.
func (rc *ReplayClient) recordDiverged(err error) {
	if rc.diverged == nil {
		rc.diverged = err
	}
}

// Err returns the replay's first recorded divergence, or nil when none was
// recorded. A divergence in the run's final tool turn can end the run
// before another Complete happens; Err is the only report of it, so callers
// MUST assert Err() == nil after a replayed run. A recorded call the replay
// never serves in its final tool turn — for example one a policy stricter than
// the recorded one denies — records no divergence, so Err does not report
// it. Results of tools that are not [ReplayClient.Tools] are never
// compared with the record. The error wraps [ErrReplayDiverged] and names
// the recorded step.
func (rc *ReplayClient) Err() error {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	return rc.diverged
}

type replayTool struct {
	rc   *ReplayClient
	name string
}

// Def declares the tool by name only: the recorded completions already fixed what it said.
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
		// No match in this step. Unconsumed calls here mean the replay
		// diverged; a fully consumed step advances to the next recorded
		// tool turn.
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
// fed it to the model: an error strips the "ERROR: " prefix so the Runner's
// renderer re-adds it (the fed-back result is byte-identical and the history
// keeps the is_error mark); a success returns the recorded result verbatim.
// Denied calls never reach here: they are not among the served calls.
func serveRecorded(rec *replayedToolCall) (string, error) {
	if rec.isError {
		return "", errors.New(strings.TrimPrefix(rec.result, "ERROR: "))
	}
	return rec.result, nil
}

func argsString(args json.RawMessage) string {
	if len(args) == 0 {
		return "(none)"
	}
	return string(args)
}

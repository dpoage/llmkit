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
// requests no longer match the recorded sequence. [ReplayClient.Err]
// reports it. Match it with errors.Is.
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
// Complete returns the next recorded response. If the record holds N tool
// runs after the previous completion and before that response, Complete
// checks that the request's last N tool-result messages carry the ids of
// those N tool runs, in order; fewer than N such messages is a mismatch. If
// the record holds no tool run there, Complete does no structure check.
// Matching ignores message text, so prompt wording can change between record
// and replay without breaking the replay.
//
// Divergence is run state owned by the client, never inferred from
// conversation text. [ReplayClient.Tools] returns the run's recorded tool
// set, bound to this client: each tool serves the recorded result of the
// call it is dispatched for, matched by tool-call ID within the recorded
// step being replayed. When the calls of a turn carry distinct IDs, replay
// is therefore deterministic under [WithParallelTools] whatever order the
// calls dispatch in, even for calls that share a name and arguments. Calls
// of one turn that share an ID are told apart by name and arguments; calls
// that share ID, name, and arguments receive their recorded results in
// dispatch order. The tool compares the arguments it receives, as JSON
// values, with the recorded ones — the record's dispatched arguments when a
// [ToolPolicy] rewrote them, else the model's — so replaying under the same
// policy reproduces its rewrites. The record spells a cleared payload as the
// JSON literal null, so an empty payload also matches dispatched arguments
// recorded as exactly null. A call with no match records the divergence on
// the client and is returned as that call's error (the harness renders it
// as data). Once a divergence is recorded, every subsequent divergence
// path — structure mismatch, exhausted record, tool-call mismatch — fails
// the same way, wrapping [ErrReplayDiverged]. Complete refuses to serve
// further responses past it.
//
// A replay tool whose Run is called directly, with no call ID in its
// context, serves the earliest unconsumed (name, arguments) match in the
// current step instead. That fallback serves direct calls only, never a
// Runner-driven replay: a Runner panics on a call with an empty ID when it
// feeds the call's result back, so no Runner-written record holds one, and
// a Runner replaying a record dispatches each call with its recorded ID.
//
// Calls a [ToolPolicy] denied in the recorded run are NOT served by the
// recorded tools: the recorded denial is a policy decision, not a tool
// result. Their names stay registered in Tools, so a caller's own policy
// still sees them. Install [WithToolPolicy] with [ReplayClient.ToolPolicy]
// to reproduce the recorded denials, or with a policy of your own to decide
// afresh.
//
// Calls to a tool the recording Runner did not register are not served
// either: that Runner answered them with its unknown-tool error without
// consulting a policy. Tools leaves their names out, so the replaying
// Runner answers them the same way, byte for byte, and its policy never
// sees them.
//
// Tools other than [ReplayClient.Tools] execute live, and their results are
// never compared with the record.
//
// Replaying under a [ToolPolicy] that rewrote arguments in the recorded run
// works only if each rewrite produces valid JSON and is deterministic: the
// replay policy must hand each tool arguments that match the record's
// dispatched arguments, and a rewrite to bytes that are not valid JSON is not
// recorded, so its replay diverges. Two limits follow from how the record
// carries them. A record written before the dispatched arguments existed
// holds only the model's arguments, so a rewriting policy's replay of it
// still diverges. A reader built before the field existed drops it when it
// re-saves a record, with the same result.
//
// A replay can end without another Complete happening. A divergence in the
// run's FINAL tool turn does this: a recording that stopped at its step cap
// replays to the same limit and Run returns an [*IncompleteError]. A replay
// that finishes before the record does, for example one without the steering
// follow-up the recorded run received, does it too, and Run returns
// normally. [ReplayClient.Err] reports both cases, and a caller that took
// [ReplayClient.Tools] also gets an error for a recorded call that a tool
// ran in the recorded run and no replay tool served. Callers MUST assert Err
// returns nil after a replayed run — a diverged replay is a failed
// evaluation whatever Run returned.
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
	// recorded marks a client built from a record by NewReplayClient: only
	// its Err reports unreached recorded work. tookTools is set by Tools and
	// gates the unconsumed-calls part of that report.
	recorded  bool
	tookTools bool
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
// replay-exhausted error. If the client has served at least one completion
// and recorded no divergence, [ReplayClient.Err] reports the first
// successful completion the replay never served.
//
// It returns an error, and no client, for a record holding a call to a tool
// the recording Runner did not register (see [ReplayClient.Tools]) whose
// outcome is not the Runner's unknown-tool error — for example a call to an
// empty name served by a Runner that registered an empty-named tool, which
// a replay cannot reproduce. The error names the step.
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
	rc := &ReplayClient{caps: caps, set: &replayToolSet{}, recorded: true}
	// Walk events: accumulate tool-call ids since the last completion (for
	// structure validation) and collect ToolRun events (for the bound tool
	// set, grouped once the whole run says which names were registered).
	var (
		pending []string
		offered []string
		runs    []toolRunAt
	)
	for _, ev := range evs {
		switch {
		case ev.Kind == llmkit.KindStart && ev.Start != nil:
			offered = ev.Start.Tools
		case ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil:
			pending = append(pending, ev.ToolRun.Call.ID)
			runs = append(runs, toolRunAt{step: ev.Step, resp: len(rc.responses) - 1, tre: *ev.ToolRun})
		case ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Completion.Err == "":
			rc.responses = append(rc.responses, replayStep{
				resp:          ev.Completion.Response,
				expectToolIDs: pending,
			})
			pending = nil
		}
	}
	unregistered := unregisteredNames(offered, runs)
	for _, r := range runs {
		if err := rc.set.add(r.step, r.resp, r.tre, unregistered[r.tre.Call.Name]); err != nil {
			return nil, fmt.Errorf("agent: run %s: %w", run, err)
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
// last response. A response it never serves is not a divergence: Err is nil
// after a run that used only some of resps. To replay a recorded run, use
// [NewReplayClient].
func NewReplayClientFromResponses(resps []llmkit.Response, caps llmkit.Capabilities) *ReplayClient {
	rc := &ReplayClient{caps: caps, set: &replayToolSet{}}
	for _, r := range resps {
		rc.responses = append(rc.responses, replayStep{resp: r})
	}
	return rc
}

func (rc *ReplayClient) Capabilities() llmkit.Capabilities { return rc.caps }

// Complete serves the next recorded response, checking tool-call structure
// only when the record holds tool runs after the previous completion and
// before that response (see [ReplayClient]). It returns ctx.Err() when ctx
// is already done. Otherwise it returns an error wrapping
// [ErrReplayDiverged] instead of serving the next response when a
// divergence is already recorded, when that structure check fails, and
// when the record is exhausted.
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
	// args is what Tool.Run received in the recorded run: the record's
	// DispatchedArguments when a policy rewrote them, else the model's
	// Call.Arguments. nullClears marks a rewrite recorded as the JSON literal
	// null, which an empty payload also satisfies (a policy that cleared the
	// arguments).
	args       json.RawMessage
	nullClears bool
	// used marks a call a replay tool served, by ID or by the (name,
	// arguments) fallback; it stays false for a call that diverged or was
	// never dispatched.
	used bool
}

// ranWith reports whether a replay tool that receives args matches the
// recorded call: args equals c.args up to the key order and whitespace
// argsEqual ignores, or args is empty and c.nullClears is set.
func (c *replayedToolCall) ranWith(args json.RawMessage) bool {
	return argsEqual(c.args, args) || (c.nullClears && len(args) == 0)
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
// A call dispatched with an ID is matched by that ID within the turn being
// replayed, and by name and arguments among the turn's calls sharing that
// ID (see [ReplayClient.serveCall]); a call without one is matched by
// (name, arguments) over the UNCONSUMED calls of the current step — not a
// strict cursor; an exhausted step advances to the next recorded tool turn.
// Denied calls are held apart from the served ones: a denial is decided by
// the policy, so it never blocks a step from advancing, but its name is
// still registered. Calls to a tool the recording Runner did not register
// are held apart from both and their names are not registered.
type replayToolSet struct {
	groups []replayStepCalls
	gi     int
}

// toolRunAt is one ToolRun event of the record, with its step and the
// index of the recorded response that requested it.
type toolRunAt struct {
	step int
	resp int
	tre  llmkit.ToolRunEvent
}

// isUnknownToolRun reports whether tre is the Runner's unknown-tool
// rendering for its call: the outcome of a call naming a tool the Runner
// did not register.
func isUnknownToolRun(tre llmkit.ToolRunEvent) bool {
	return tre.IsError && !tre.Denied && tre.Result == renderUnknownTool(tre.Call.Name)
}

// unregisteredNames returns the recorded call names the recording Runner did
// not register (see [ReplayClient.Tools] for the rule). offered is the run's
// Start event tool list, nil when the record has none.
func unregisteredNames(offered []string, runs []toolRunAt) map[string]bool {
	out := map[string]bool{"": true}
	if len(offered) > 0 {
		for _, r := range runs {
			if !slices.Contains(offered, r.tre.Call.Name) {
				out[r.tre.Call.Name] = true
			}
		}
		return out
	}
	registered := map[string]bool{}
	for _, r := range runs {
		if !isUnknownToolRun(r.tre) {
			registered[r.tre.Call.Name] = true
		}
	}
	for _, r := range runs {
		if !registered[r.tre.Call.Name] {
			out[r.tre.Call.Name] = true
		}
	}
	return out
}

// add files one recorded ToolRun under step; resp is the index of the
// recorded response that requested it. A call to a tool the recording
// Runner did not register (unregistered) is held apart from the groups: that
// Runner rendered it without dispatching it, the replay tools never register
// its name, and the replaying Runner therefore reproduces the recorded
// unknown-tool result on its own. Such a call with any other outcome cannot
// be reproduced, so add rejects it.
func (ts *replayToolSet) add(step, resp int, tre llmkit.ToolRunEvent, unregistered bool) error {
	if unregistered {
		if isUnknownToolRun(tre) {
			return nil
		}
		return fmt.Errorf("step %d: recorded call %q to tool %q, which the replay cannot register, has an outcome other than the unknown-tool error, so the record cannot be replayed",
			step, tre.Call.ID, tre.Call.Name)
	}
	if len(ts.groups) == 0 || ts.groups[len(ts.groups)-1].step != step {
		ts.groups = append(ts.groups, replayStepCalls{step: step, resp: resp})
	}
	g := &ts.groups[len(ts.groups)-1]
	if tre.Denied {
		g.denials = append(g.denials, replayedDenial{call: tre.Call, reason: tre.DenyReason})
		return nil
	}
	rec := replayedToolCall{
		call:    tre.Call,
		result:  tre.Result,
		isError: tre.IsError,
		step:    step,
		args:    tre.Call.Arguments,
	}
	if len(tre.DispatchedArguments) > 0 {
		rec.args = tre.DispatchedArguments
		rec.nullClears = string(rec.args) == "null"
	}
	g.calls = append(g.calls, rec)
	return nil
}

// tools registers every grouped call name, denied-only names included, so
// a policy the caller installs sees every call the recorded policy saw.
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
// serves the recorded result of the call it is dispatched for, matched by
// tool-call ID, instead of executing. Feed it to a [Runner] driven by this
// same client to replay a run fully offline — zero live tool executions. A
// run that recorded no tool calls yields an empty slice.
//
// Every name the recording Runner registered and the model called is
// registered, including names the recorded policy only ever denied; denied
// calls themselves are never served (see [ReplayClient.ToolPolicy]). A name
// whose every call errored stays registered when the run's Start event
// lists it. A name the recording Runner did not register is left out: its
// calls, which that Runner answered with its unknown-tool error before any
// policy saw them, replay through the replaying Runner's own unknown-tool
// path, and are never served. A name counts as not registered when it is
// empty; when the run's Start event lists tools and the name is not among
// them; or, for a record whose Start event lists none, when every recorded
// call of the name carries the unknown-tool error.
//
// A call with no unconsumed match (see [ReplayClient] for how arguments
// match) records this client's divergence and is returned as the tool's
// error; the harness renders it as data, and the next Complete refuses to
// serve.
//
// Calling Tools opts the client into the unconsumed-calls check of
// [ReplayClient.Err]. If the replay has served a completion and recorded no
// divergence, Err reports a recorded call that a tool ran in the recorded
// run and no replay tool served: for example, a call the replay's policy
// denied, or any such call when the tools never reached the Runner. If a
// recorded completion is also unserved, Err names that completion instead.
//
// Only these tools are compared with the record. Any other tool a caller
// passes to the Runner executes live, and its result is never compared.
func (rc *ReplayClient) Tools() []Tool {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.tookTools = true
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

// Err returns the replay's first recorded divergence. When none was
// recorded and the client has served at least one completion, Err also
// reports recorded work the replay never reached, as an error wrapping
// [ErrReplayDiverged] that names the recorded step:
//
//   - a recorded successful completion the replay never served, for example
//     one a steering follow-up produced in the recorded run;
//   - a recorded tool call that a tool ran and no replay tool served, but
//     only when the caller took [ReplayClient.Tools]: taking them opts into
//     this check. A caller that takes them and never runs them, for example
//     never passes them to the Runner, gets an error for a record that
//     holds such a call; a caller that never takes them (live tools only)
//     gets no call check. A call a [ToolPolicy] denied in the recorded run
//     and a call to a tool the recording Runner did not register are never
//     counted, since no replay tool serves them.
//
// When both kinds of unreached work exist, Err names the unserved
// completion.
//
// A divergence in the run's final tool turn can end the run before another
// Complete happens; Err returns it. If the caller took [ReplayClient.Tools],
// a caller's policy stricter than the recorded one can leave a recorded call
// of that turn unserved; Err then returns an error. Callers MUST therefore
// assert Err() == nil after a replayed run, and call it after Run returns:
// called during the run, after the client has served its first completion,
// it applies the checks above to the replay so far. A client from
// [NewReplayClientFromResponses] reports no unreached work: its Err is nil
// unless it served past its last response. Results of tools that are not
// [ReplayClient.Tools] are never compared with the record.
func (rc *ReplayClient) Err() error {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if rc.diverged != nil {
		return rc.diverged
	}
	if !rc.recorded || rc.idx == 0 {
		return nil
	}
	return rc.unreached()
}

// unreached reports the first recorded completion, then the first recorded
// served call when the caller took the tools, that the replay never reached.
// It is computed on each call and never stored, so a mid-run Err does not
// make the client refuse later completions; caller holds rc.mu.
func (rc *ReplayClient) unreached() error {
	if rc.idx < len(rc.responses) {
		return fmt.Errorf("%w at step %d: recorded completion %d of %d was never served; the replay served %d",
			ErrReplayDiverged, rc.idx+1, rc.idx+1, len(rc.responses), rc.idx)
	}
	if !rc.tookTools {
		return nil
	}
	for _, g := range rc.set.groups {
		for _, c := range g.calls {
			if !c.used {
				return fmt.Errorf("%w at step %d: recorded call %q to tool %q was never served with arguments %s",
					ErrReplayDiverged, c.step, c.call.ID, c.call.Name, argsString(c.args))
			}
		}
	}
	return nil
}

type replayTool struct {
	rc   *ReplayClient
	name string
}

// Def declares the tool by name only: the recorded completions already fixed what it said.
func (t replayTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name}
}

// Run serves the recorded call this invocation stands for. Under a [Runner]
// the dispatch names the call by ID: Run serves the recorded call with that
// ID in the turn being replayed (among several unconsumed calls sharing the
// ID, the first whose name and arguments fit), and the client records a
// divergence if the tool's name or the arguments it received do not match
// the recording, as [ReplayClient] describes. Called directly with no call
// ID (a Runner panics on an empty call ID, so it never completes a turn
// with one), Run serves the earliest unconsumed recorded call of the current
// step matching both name and arguments. No match records the client's
// divergence: the returned error wraps [ErrReplayDiverged] naming the
// recorded step, and the harness renders it as this call's ERROR result.
func (t replayTool) Run(ctx context.Context, args json.RawMessage) (string, error) {
	return t.rc.serveCall(t.name, callIDFrom(ctx), args)
}

// serveCall serves and consumes one recorded call, or records the
// divergence. id is the dispatched call's ID, or "" for the (name, args)
// fallback. An ID is only looked up in the recorded tool turn the replay is
// in (the group whose response is the last one served), because providers
// may repeat call IDs from turn to turn. Among the turn's unconsumed calls
// with that ID, the first whose name and recorded Tool.Run arguments fit is
// served; when none fits, the first is the one reported as diverged.
func (rc *ReplayClient) serveCall(name, id string, args json.RawMessage) (string, error) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	set := rc.set
	for set != nil && set.gi < len(set.groups) {
		g := &set.groups[set.gi]
		match := -1
		switch {
		case id == "":
			for i := range g.calls {
				if !g.calls[i].used && g.calls[i].call.Name == name && g.calls[i].ranWith(args) {
					match = i
					break
				}
			}
		case g.resp == rc.idx-1:
			for i := range g.calls {
				c := &g.calls[i]
				if c.used || c.call.ID != id {
					continue
				}
				if match < 0 {
					match = i
				}
				if c.call.Name == name && c.ranWith(args) {
					match = i
					break
				}
			}
		}
		if match >= 0 {
			rec := &g.calls[match]
			switch {
			case rec.call.Name != name:
				err := fmt.Errorf("%w at step %d: call %q was recorded for tool %q, replayed for tool %q",
					ErrReplayDiverged, rec.step, id, rec.call.Name, name)
				rc.recordDiverged(err)
				return "", err
			case !rec.ranWith(args):
				return "", rc.argsDiverged(rec, name, args)
			}
			rec.used = true
			return serveRecorded(rec)
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
			return "", rc.argsDiverged(&g.calls[unconsumed], name, args)
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

// argsDiverged records, and returns, the divergence where a tool was called
// with arguments other than the ones rec's dispatch carried; caller holds
// rc.mu.
func (rc *ReplayClient) argsDiverged(rec *replayedToolCall, name string, args json.RawMessage) error {
	err := fmt.Errorf("%w at step %d: tool %q called with arguments %s, recorded %s",
		ErrReplayDiverged, rec.step, name, argsString(args), argsString(rec.args))
	rc.recordDiverged(err)
	return err
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

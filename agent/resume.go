package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/dpoage/llmkit"
)

// ErrResumeMismatch reports that a recorded run cannot be resumed as asked:
// the record holds a steering event or no completion, or the resumed
// Runner's requests diverge from the recorded ones. Match with errors.Is.
var ErrResumeMismatch = errors.New("agent: resume mismatch")

var errResumeReuse = errors.New("agent: a Resume runs once; build another from the record to run again")

// Resume is one armed resumption of a recorded run. Construct it with
// [NewResume] and run it once, with the original task, via [Resume.Run] or
// [Resume.RunJSON].
type Resume struct {
	runner *Runner
	run    llmkit.RunID
	usage  llmkit.Usage
	seed   *llmkit.Usage
	ran    atomic.Bool
}

// NewResume arms the resume of run as recorded by src, configured from r:
// every Runner option is taken from r — the record decides only the replayed
// prefix — and r is never mutated and stays usable for plain runs.
// The error wraps [ErrResumeMismatch], before any call, for a record holding
// a steering event, no completion to replay, or a record [NewReplayClient]
// refuses. The record of a resumed run holds only that run's live spend, so
// a resume of it seeds the token budget with that spend alone: the limit is
// not cumulative across a chain of resumes.
func NewResume(ctx context.Context, src llmkit.Source, run llmkit.RunID, r *Runner) (*Resume, error) {
	if src == nil {
		return nil, errors.New("agent: nil event source")
	}
	if r == nil {
		return nil, errors.New("agent: nil Runner")
	}
	evs, err := src.Events(ctx, run)
	if err != nil {
		return nil, fmt.Errorf("agent: read events for run %s: %w", run, err)
	}
	var (
		usage     llmkit.Usage
		strict    []llmkit.Request
		compacted bool
	)
	for _, ev := range evs {
		if ev.Kind == llmkit.KindSteer {
			return nil, fmt.Errorf("%w: record of run %s holds a steering event; recorded steering messages are not re-injected on resume", ErrResumeMismatch, run)
		}
		if ev.Kind == llmkit.KindCompaction || isPurposeCompletion(ev) {
			compacted = true
		}
		if ev.Kind != llmkit.KindCompletion || ev.Completion == nil {
			continue
		}
		ce := ev.Completion
		if ce.Err == "" {
			usage = usage.Add(ce.Response.Usage)
		}
		// strict mirrors NewReplayClient's serving filter: the requests the
		// record serves, in serving order, Purpose completions excluded.
		if ce.Purpose == "" && (ce.Err == "" || ce.ErrKind == llmkit.ErrKindContextTooLong) {
			strict = append(strict, ce.Request)
		}
	}
	if len(strict) == 0 {
		return nil, fmt.Errorf("%w: record of run %s holds no completion to replay", ErrResumeMismatch, run)
	}
	rc, err := NewReplayClient(&Transcript{Record: evs, RunID: run}, run, r.client.Capabilities())
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrResumeMismatch, err)
	}
	if len(strict) != len(rc.responses) {
		return nil, fmt.Errorf("agent: run %s: resume request filter (%d) drifted from the replay's serving list (%d)", run, len(strict), len(rc.responses))
	}
	st := &resumeState{
		rc:         rc,
		live:       r.client,
		identity:   llmkit.IdentityOf(r.client),
		strict:     strict,
		prefix:     rc.Compactor(),
		caller:     r.compactor,
		callerPol:  r.toolPolicy,
		authorized: map[*replayedToolCall]bool{},
		deniedUsed: map[*replayedDenial]bool{},
	}
	tools := make([]Tool, 0, len(r.tools.defs))
	for _, def := range r.tools.defs {
		tools = append(tools, resumeTool{st: st, live: r.tools.byName[def.Name]})
	}
	rr := *r
	rr.client = &resumeClient{st: st}
	rr.tools = newToolSet(tools)
	rr.toolPolicy = &resumePolicy{st: st}
	if compacted || r.compactor != nil {
		rr.compactor = &resumeCompactor{st: st}
	}
	return &Resume{runner: &rr, run: run, usage: usage, seed: &st.seed}, nil
}

// Usage returns the recorded run's token spend: the sum of every successful
// completion's Response.Usage, compaction completions included. For a record
// whose run did not panic it equals that run's Finalize usage. It is
// readable before any call.
func (res *Resume) Usage() llmkit.Usage { return res.usage }

// Run re-runs the original task: the recorded prefix replays, then the run
// continues live on the Runner NewResume took its configuration from. Every
// event of the resumed run carries the recorded run's id as ParentRunID,
// under a fresh RunID, whatever opts hold ([Continue] included); opts
// otherwise apply as on [Runner.Run]. A second Run or RunJSON on the same
// Resume is a programming error and fails before any call.
func (res *Resume) Run(ctx context.Context, task string, opts ...RunOption) (*Outcome, error) {
	if !res.ran.CompareAndSwap(false, true) {
		return nil, errResumeReuse
	}
	return res.runner.Run(ctx, task, res.options(opts)...)
}

// RunJSON is [Resume.Run] with [Runner.RunJSON] semantics: schema rides
// every completion of the resumed run and out receives the parsed answer.
func (res *Resume) RunJSON(ctx context.Context, task string, schema json.RawMessage, out any, opts ...RunOption) (*Outcome, error) {
	if err := checkJSONOut(out); err != nil {
		return nil, err
	}
	if !res.ran.CompareAndSwap(false, true) {
		return nil, errResumeReuse
	}
	var cfg runConfig
	for _, opt := range res.options(opts) {
		opt(&cfg)
	}
	return res.runner.runJSON(ctx, cfg, task, schema, out)
}

// options appends the resume origin last so it wins: ParentRunID = the
// recorded run, and the budget seed the wrapper advances as it serves.
func (res *Resume) options(opts []RunOption) []RunOption {
	origin := func(c *runConfig) {
		c.parentRunID = res.run
		c.budgetSeed = res.seed
	}
	return append(opts[:len(opts):len(opts)], origin)
}

// resumeState holds one armed resume's collaborators. The replay cursor and
// tool turns live on rc under rc.mu, as do authorized, deniedUsed and
// decisions; diverged lives here under mu. Lock order: rc.mu, then mu.
type resumeState struct {
	rc        *ReplayClient
	live      llmkit.Client
	identity  llmkit.Identity
	strict    []llmkit.Request
	prefix    Compactor
	caller    Compactor
	callerPol ToolPolicy
	// seed is the recorded spend served so far; only the run's goroutine
	// touches it (completions and compactions), as the budget check reads it.
	seed llmkit.Usage

	authorized map[*replayedToolCall]bool
	deniedUsed map[*replayedDenial]bool
	decisions  []resumeDecision

	mu         sync.Mutex
	diverged   error
	liveActive bool
}

// resumeDecision is how one authorized call of the replayed turn g runs:
// rec serves it, or rec is nil and it runs live with args.
type resumeDecision struct {
	g    *replayStepCalls
	id   string
	name string
	rec  *replayedToolCall
	args json.RawMessage
	done bool
}

// goneLive reports whether a live completion has been served; the record
// never answers a call after that, whatever its ID.
func (st *resumeState) goneLive() bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.liveActive
}

func (st *resumeState) markLive() {
	st.mu.Lock()
	st.liveActive = true
	st.mu.Unlock()
}

// diverge records err, first one wins, and returns the recorded divergence;
// the run fails at the next completion, before any live call.
func (st *resumeState) diverge(err error) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.diverged == nil {
		st.diverged = err
	}
	return st.diverged
}

// divergedErr returns the recorded divergence, if any.
func (st *resumeState) divergedErr() error {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.diverged
}

// resumeClient serves the record's completions until it is exhausted, then
// forwards to the live client. Requests still covered by the record must be
// canonically identical to the recorded one at that position.
type resumeClient struct{ st *resumeState }

func (c *resumeClient) Capabilities() llmkit.Capabilities { return c.st.live.Capabilities() }

// Identity forwards the live client's, so resumed live completions record it.
func (c *resumeClient) Identity() llmkit.Identity { return c.st.identity }

func (c *resumeClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}
	if derr := c.st.divergedErr(); derr != nil {
		return llmkit.Response{}, derr
	}
	rc := c.st.rc
	rc.mu.Lock()
	at, prefix := rc.idx, rc.idx < len(rc.responses)
	rc.mu.Unlock()
	if !prefix {
		c.st.markLive()
		return c.st.live.Complete(ctx, req)
	}
	if derr := requestMatches(c.st.strict[at], req); derr != nil {
		return llmkit.Response{}, c.st.diverge(derr)
	}
	resp, err := rc.Complete(ctx, req)
	if err != nil {
		var overflow replayedOverflow
		if errors.As(err, &overflow) {
			return llmkit.Response{}, err // served as recorded; the Runner recovers as it did
		}
		werr := fmt.Errorf("%w: completion %d of the record: %w", ErrResumeMismatch, at+1, err)
		_ = c.st.diverge(werr)
		return llmkit.Response{}, werr
	}
	c.st.seed = c.st.seed.Add(resp.Usage)
	resp.Usage = llmkit.Usage{} // recorded spend is never re-charged
	return resp, nil
}

// Stream synthesizes deltas from the recorded response while the record
// serves; past it, the live client streams natively.
func (c *resumeClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	if derr := c.st.divergedErr(); derr != nil {
		return llmkit.Response{}, derr
	}
	rc := c.st.rc
	rc.mu.Lock()
	prefix := rc.idx < len(rc.responses)
	rc.mu.Unlock()
	if !prefix {
		c.st.markLive()
		return llmkit.Stream(ctx, c.st.live, req, fn)
	}
	return llmkit.Stream(ctx, completeOnly{c}, req, fn)
}

// completeOnly hides Stream so llmkit.Stream synthesizes from Complete.
type completeOnly struct{ c *resumeClient }

func (o completeOnly) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	return o.c.Complete(ctx, req)
}

func (o completeOnly) Capabilities() llmkit.Capabilities { return o.c.Capabilities() }

// resumeTool wraps one live tool: the record serves calls the prefix covers,
// the live tool runs everything past it.
type resumeTool struct {
	st   *resumeState
	live Tool
}

func (t resumeTool) Def() llmkit.ToolDef { return t.live.Def() }

func (t resumeTool) Run(ctx context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	rec, live, err := t.st.lookupCall(ctx, t.Def().Name, args)
	switch {
	case err != nil:
		return nil, err
	case live:
		return t.live.Run(ctx, args)
	default:
		return serveRecorded(ctx, rec)
	}
}

// lookupCall resolves one dispatched call: the decision Authorize took for
// it, else the recorded call to serve, or live=true when nothing recorded
// covers it and the record is exhausted; any other miss or mismatch
// diverges with ErrResumeMismatch.
func (st *resumeState) lookupCall(ctx context.Context, name string, args json.RawMessage) (*replayedToolCall, bool, error) {
	if derr := st.divergedErr(); derr != nil {
		return nil, false, derr
	}
	id := callIDFrom(ctx)
	rc := st.rc
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if id != "" && !st.goneLive() {
		if g := currentGroup(rc); g != nil {
			if d := st.takeDecision(g, id, name, args); d != nil {
				if d.rec == nil {
					return nil, true, nil
				}
				d.rec.used = true
				return d.rec, false, nil
			}
			if match := matchByID(g, id, name, args); match >= 0 {
				rec := &g.calls[match]
				if rec.call.Name != name || !rec.ranWith(args) {
					err := fmt.Errorf("%w at step %d: tool %q called with arguments %s, recorded %s",
						ErrResumeMismatch, rec.step, name, argsString(args), argsString(rec.args))
					return nil, false, st.diverge(err)
				}
				rec.used = true
				return rec, false, nil
			}
		}
	}
	if rc.idx >= len(rc.responses) {
		return nil, true, nil
	}
	err := fmt.Errorf("%w at step %d: call %q to tool %q has no recorded counterpart while the record still holds completions",
		ErrResumeMismatch, rc.idx+1, id, name)
	return nil, false, st.diverge(err)
}

// takeDecision consumes g's first pending decision that fits the call; the
// caller holds rc.mu.
func (st *resumeState) takeDecision(g *replayStepCalls, id, name string, args json.RawMessage) *resumeDecision {
	for i := range st.decisions {
		d := &st.decisions[i]
		if d.done || d.g != g || d.id != id || d.name != name {
			continue
		}
		if (d.rec != nil && d.rec.ranWith(args)) || (d.rec == nil && argsEqual(d.args, args)) {
			d.done = true
			return d
		}
	}
	return nil
}

// matchByID picks the turn's call to serve for id: the first unused one whose
// name and arguments fit, else the first with the ID (reported as diverged).
func matchByID(g *replayStepCalls, id, name string, args json.RawMessage) int {
	match := -1
	for i := range g.calls {
		c := &g.calls[i]
		if c.used || c.call.ID != id {
			continue
		}
		if match < 0 {
			match = i
		}
		if c.call.Name == name && c.ranWith(args) {
			return i
		}
	}
	return match
}

// currentGroup returns the recorded tool turn in progress: the group whose
// recorded response is the last one served.
func currentGroup(rc *ReplayClient) *replayStepCalls {
	if rc.set == nil {
		return nil
	}
	for i := range rc.set.groups {
		if rc.set.groups[i].resp == rc.idx-1 {
			return &rc.set.groups[i]
		}
	}
	return nil
}

// resumePolicy serves the prefix's recorded decisions; past the record it
// delegates, so the caller's policy sees only live calls.
type resumePolicy struct{ st *resumeState }

func (p *resumePolicy) Authorize(ctx context.Context, call *llmkit.ToolCall) error {
	st := p.st
	if derr := st.divergedErr(); derr != nil {
		return derr
	}
	g, recorded, err := st.recordedDecision(call)
	if recorded {
		return err
	}
	if st.callerPol != nil {
		if err := st.callerPol.Authorize(ctx, call); err != nil {
			return err
		}
	}
	if g != nil {
		st.rc.mu.Lock()
		st.decisions = append(st.decisions, resumeDecision{g: g, id: call.ID, name: call.Name, args: bytes.Clone(call.Arguments)})
		st.rc.mu.Unlock()
	}
	return nil
}

// recordedDecision answers call from the replayed turn g, matching per call
// as matchByID does: an unused recorded denial, or the recorded dispatch of
// the first unauthorized call with the same ID, name and model arguments.
// recorded=false routes the call live.
func (st *resumeState) recordedDecision(call *llmkit.ToolCall) (g *replayStepCalls, recorded bool, err error) {
	rc := st.rc
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if call.ID == "" || st.goneLive() {
		return nil, false, nil
	}
	if g = currentGroup(rc); g != nil {
		for i := range g.denials {
			d := &g.denials[i]
			if !st.deniedUsed[d] && sameCall(d.call, *call) {
				st.deniedUsed[d] = true
				return g, true, errors.New(d.reason)
			}
		}
		for i := range g.calls {
			c := &g.calls[i]
			if !st.authorized[c] && sameCall(c.call, *call) {
				st.authorized[c] = true
				// The recorded dispatch, verbatim: an unrewritten call gets its own arguments.
				call.Arguments = bytes.Clone(c.args)
				st.decisions = append(st.decisions, resumeDecision{g: g, id: call.ID, name: call.Name, rec: c})
				return g, true, nil
			}
		}
	}
	if rc.idx < len(rc.responses) {
		err := fmt.Errorf("%w at step %d: call %q to tool %q has no recorded decision while the record still holds completions",
			ErrResumeMismatch, rc.idx+1, call.ID, call.Name)
		return g, true, st.diverge(err)
	}
	return g, false, nil
}

func sameCall(rec, call llmkit.ToolCall) bool {
	return rec.ID == call.ID && rec.Name == call.Name && argsEqual(rec.Arguments, call.Arguments)
}

// resumeCompactor serves the recorded summaries during the prefix, then
// delegates to the caller's compactor (the default stubber when nil).
type resumeCompactor struct{ st *resumeState }

func (c *resumeCompactor) Compact(ctx context.Context, in CompactInput) (CompactOutput, error) {
	if derr := c.st.divergedErr(); derr != nil {
		return CompactOutput{}, derr
	}
	rc := c.st.rc
	rc.mu.Lock()
	// The record still owns a pass whose consuming completion was cut off:
	// its summary is recorded even though no completion is left to serve.
	prefix := rc.idx < len(rc.responses) || rc.nextSummary < len(rc.summaries)
	rc.mu.Unlock()
	if prefix {
		out, err := c.st.prefix.Compact(ctx, in)
		c.st.seed = c.st.seed.Add(out.Usage)
		out.Usage = llmkit.Usage{} // recorded spend is never re-charged
		for i := range out.calls {
			out.calls[i].resp.Usage = llmkit.Usage{}
		}
		return out, err
	}
	if c.st.caller != nil {
		return c.st.caller.Compact(ctx, in)
	}
	return stubCompactor{}.Compact(ctx, in)
}

// requestMatches fails closed: a side that cannot be canonicalized is a
// divergence, never a pass.
func requestMatches(rec, live llmkit.Request) error {
	want, err := canonRequest(rec)
	if err != nil {
		return fmt.Errorf("%w: the recorded request cannot be encoded for comparison: %v", ErrResumeMismatch, err)
	}
	got, err := canonRequest(live)
	if err != nil {
		return fmt.Errorf("%w: the request cannot be encoded for comparison: %v", ErrResumeMismatch, err)
	}
	if got != want {
		return fmt.Errorf("%w: the request at this position differs from the recorded one (system prompt, tool set, task, or a policy or hook edit)", ErrResumeMismatch)
	}
	return nil
}

// canonRequest renders r's canonical JSON form: marshal, decode into a
// fresh value of the same type, marshal again.
func canonRequest(r llmkit.Request) (string, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	var fresh llmkit.Request
	if err := json.Unmarshal(b, &fresh); err != nil {
		return "", err
	}
	if b, err = json.Marshal(fresh); err != nil {
		return "", err
	}
	return string(b), nil
}

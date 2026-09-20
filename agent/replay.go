package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit"
	"strings"
	"sync"
)

// Source is the read side of recording: the ordered events of one run.
// [Transcript] and the JSONL sink ([JSONL]) implement it; store/sqlite does
// in a later round. [NewReplayClient] and [ReplayTools] rebuild a run's
// completions and tool results from any Source, so replay never depends on a
// JSONL file existing.
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
	rc := &ReplayClient{caps: caps}
	// Walk events, accumulating the tool-call ids recorded since the last
	// completion; attach them as the "expected preceding tool results" for
	// the next recorded response.
	var pending []string
	for _, ev := range evs {
		switch {
		case ev.Kind == llmkit.KindToolRun && ev.ToolRun != nil:
			pending = append(pending, ev.ToolRun.Call.ID)
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
	return rc, nil
}

// NewReplayClientFromResponses builds a ReplayClient that serves resps in
// order without any tool-call structure validation. Useful for hand-scripted
// tests.
func NewReplayClientFromResponses(resps []llmkit.Response, caps llmkit.Capabilities) *ReplayClient {
	rc := &ReplayClient{caps: caps}
	for _, r := range resps {
		rc.responses = append(rc.responses, replayStep{resp: r})
	}
	return rc
}

// Capabilities returns the configured capability profile.
func (rc *ReplayClient) Capabilities() llmkit.Capabilities { return rc.caps }

// Complete serves the next recorded response, validating tool-call structure.
func (rc *ReplayClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}

	rc.mu.Lock()
	defer rc.mu.Unlock()

	if rc.idx >= len(rc.responses) {
		return llmkit.Response{}, fmt.Errorf("agent: replay exhausted after %d responses; request sequence diverged (extra completion call)", len(rc.responses))
	}
	step := rc.responses[rc.idx]

	// Validate the tool-result structure leading into this request, when the
	// step carries expectations (i.e. built from a recorded run).
	if step.expectToolIDs != nil {
		got := trailingToolResultIDs(req.Messages, len(step.expectToolIDs))
		if err := matchToolIDs(rc.idx, step.expectToolIDs, got); err != nil {
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
		return fmt.Errorf("agent: replay diverged at step %d: expected %d preceding tool result(s) %v, got %d %v",
			step+1, len(want), want, len(got), got)
	}
	for i := range want {
		if want[i] != got[i] {
			return fmt.Errorf("agent: replay diverged at step %d: tool result %d was %q, recorded %q",
				step+1, i, got[i], want[i])
		}
	}
	return nil
}

// replayedToolCall is one recorded tool call inside a [replayToolSet]: the
// model's call, the model-visible result, the error/denial marks, and the
// step the call belongs to (named by divergence errors).
type replayedToolCall struct {
	call       llmkit.ToolCall
	result     string
	isError    bool
	denied     bool
	denyReason string
	step       int
}

// replayToolSet is the shared cursor over a run's recorded tool calls. The
// tools [ReplayTools] returns consume it in recorded order, one call per Run.
type replayToolSet struct {
	mu    sync.Mutex
	calls []replayedToolCall
	idx   int
}

// ReplayTools rebuilds a recorded run's tool set from src: one [Tool] per
// distinct tool name the run called, each serving the recorded [llmkit.
// ToolRunEvent] results instead of executing. Feeding the returned tools to a
// [Runner] driven by a [NewReplayClient] replays a run fully offline — zero
// live tool executions, to the same final answer.
//
// The tools consume the run's ToolRun events in recorded order, one call per
// Run. A call that diverges from the record — a different name, different
// arguments — fails with an error naming the recorded step. A recorded error
// or policy denial is returned as a Tool error, so the Runner renders exactly
// the text the original run fed to the model.
//
// A run that recorded no tool calls yields a nil tool set and no error: the
// natural tool set for replaying it is the empty one.
func ReplayTools(src Source, run llmkit.RunID) ([]Tool, error) {
	if src == nil {
		return nil, errors.New("agent: nil event source")
	}
	evs, err := src.Events(context.Background(), run)
	if err != nil {
		return nil, fmt.Errorf("agent: read events for run %s: %w", run, err)
	}
	set := &replayToolSet{}
	var names []string
	seen := map[string]bool{}
	for _, ev := range evs {
		if ev.Kind != llmkit.KindToolRun || ev.ToolRun == nil {
			continue
		}
		tre := ev.ToolRun
		set.calls = append(set.calls, replayedToolCall{
			call:       tre.Call,
			result:     tre.Result,
			isError:    tre.IsError,
			denied:     tre.Denied,
			denyReason: tre.DenyReason,
			step:       ev.Step,
		})
		if !seen[tre.Call.Name] {
			seen[tre.Call.Name] = true
			names = append(names, tre.Call.Name)
		}
	}
	tools := make([]Tool, len(names))
	for i, name := range names {
		tools[i] = replayTool{set: set, name: name}
	}
	return tools, nil
}

// replayTool serves one recorded tool name out of a [replayToolSet].
type replayTool struct {
	set  *replayToolSet
	name string
}

// Def declares the tool by name only: replay never re-describes a tool to a
// real model — the recorded completions already fixed what it "said".
func (t replayTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name}
}

// Run serves the next recorded call for this tool's name. A name, argument,
// or sequence divergence fails with an error naming the recorded step, which
// the Runner feeds back as data — the divergence is the test failure to read.
func (t replayTool) Run(_ context.Context, args json.RawMessage) (string, error) {
	s := t.set
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.idx >= len(s.calls) {
		return "", fmt.Errorf("agent: replay exhausted after %d recorded tool call(s); request sequence diverged (extra call to %q)", len(s.calls), t.name)
	}
	rec := s.calls[s.idx]
	s.idx++
	if rec.call.Name != t.name {
		return "", fmt.Errorf("agent: replay diverged at step %d: recorded tool call %q (%s), requested %q", rec.step, rec.call.Name, rec.call.ID, t.name)
	}
	if !argsEqual(rec.call.Arguments, args) {
		return "", fmt.Errorf("agent: replay diverged at step %d: tool %q called with arguments %s, recorded %s", rec.step, t.name, args, rec.call.Arguments)
	}
	switch {
	case rec.denied:
		// Re-render through the harness so the model-visible text matches the
		// original run: toolError prefixes "ERROR: " onto this.
		return "", fmt.Errorf("tool %s denied: %s", rec.call.Name, rec.denyReason)
	case rec.isError:
		// The recorded text already carries the "ERROR: " prefix; strip it
		// and let the Runner's renderer re-add it, so the fed-back result is
		// byte-identical and the history keeps the is_error mark.
		return "", errors.New(strings.TrimPrefix(rec.result, "ERROR: "))
	default:
		return rec.result, nil
	}
}

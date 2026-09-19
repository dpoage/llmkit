package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/dpoage/llmkit"
)

// Tool is a single capability the model may invoke during a run. The harness
// advertises every tool's [llmkit.ToolDef] to the model and dispatches matching
// tool calls to [Tool.Run].
//
// Run receives the raw JSON arguments the model produced (validate/unmarshal
// them yourself) and returns a textual result. An error returned from Run is
// *not* a loop failure: the harness wraps it as an "ERROR:"-prefixed
// tool-result message and lets the model decide how to recover. Reserve errors
// for tool-level problems (bad arguments, file not found); never use them to
// signal that the loop should abort.
//
// A panic from Run is treated the same way in both dispatch modes
// (sequential and [WithParallelTools]): the harness recovers it and renders
// it as that call's error result "ERROR: tool <name> panicked: <value>"
// with IsError=true, [Hooks.ToolEnd] fires with that result, and the run
// continues. A panic raised by adversarial model-supplied arguments is data
// for the model, not an abort — the harness never lets a tool panic kill the
// process or its caller. (A panicking HOOK is a harness bug with the
// opposite contract: it propagates out of [Runner.Run] and is never rendered
// to the model — see [Hooks].)
//
// Run must honor ctx cancellation. The harness may invoke Run concurrently:
// within a single run when the Runner was constructed with
// [WithParallelTools] (one goroutine per tool call in a turn), and across
// simultaneous Run calls on one Runner (a Runner is safe for concurrent Run
// calls). A Tool used under either arrangement must be safe for concurrent
// calls — unsynchronized state data-races rather than erroring, and races
// surface as silently wrong tool results. See the [Runner] concurrency
// paragraph and [WithParallelTools].
type Tool interface {
	// Def returns the tool's declaration (name, description, JSON-schema
	// parameters) as advertised to the model.
	Def() llmkit.ToolDef
	// Run executes the tool with the model-supplied arguments and returns the
	// textual result to feed back to the model. An error, or a panic, is
	// recovered by the harness and rendered to the model as this call's
	// "ERROR:"-prefixed result (identically in both dispatch modes).
	Run(ctx context.Context, args json.RawMessage) (string, error)
}

// toolError prefixes a tool failure so the model recognizes it as a recoverable
// error rather than a normal result. The harness uses this for every error a
// Tool returns.
func toolError(err error) string {
	return "ERROR: " + err.Error()
}

// UnmarshalArgs decodes raw JSON tool arguments into dst. It returns a
// well-formed error the runner will surface as "ERROR: invalid arguments: …"
// when the model produced malformed JSON. Tool implementations call this to
// decode the args passed to Tool.Run.
func UnmarshalArgs(raw json.RawMessage, dst any) error {
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("invalid arguments: %w", err)
	}
	return nil
}

// toolSet indexes tools by name for dispatch and collects their defs for the
// request.
type toolSet struct {
	byName map[string]Tool
	defs   []llmkit.ToolDef
}

// newToolSet builds a dispatch table from tools. Later tools with a duplicate
// name win, mirroring map-assignment semantics; the defs slice preserves the
// first-seen order of the deduplicated set.
func newToolSet(tools []Tool) toolSet {
	ts := toolSet{byName: make(map[string]Tool, len(tools))}
	seen := make(map[string]bool, len(tools))
	for _, t := range tools {
		def := t.Def()
		ts.byName[def.Name] = t
		if !seen[def.Name] {
			ts.defs = append(ts.defs, def)
			seen[def.Name] = true
		}
	}
	return ts
}

// lookup returns the tool registered under name, if any.
func (ts toolSet) lookup(name string) (Tool, bool) {
	t, ok := ts.byName[name]
	return t, ok
}

// ToolHealthError marks a tool failure as a GENUINE harness-tooling/infra
// problem (missing container runtime, crashed language server) versus an
// ordinary model-recoverable error (bad args, file-not-found). The runner
// reports it through [Hooks.ToolHealth] IN ADDITION to feeding the error text
// back to the model via the existing toolError path; plain tool errors do not
// reach the hook. Reason is a short human-readable label for the failure
// class; callers that need impact ranking can bucket on it. Err is the
// original underlying error, if any, reachable via [ToolHealthError.Unwrap]
// so callers may inspect it with [errors.As]/[errors.Is].
type ToolHealthError struct {
	// Reason is a short human-readable label (no leading "ERROR: "; the
	// runner's toolError prefix is applied separately when the model is
	// informed).
	Reason string
	// Err is the original underlying error, if any. When non-nil its message
	// is appended to Reason in [ToolHealthError.Error]; it is also returned
	// by [ToolHealthError.Unwrap].
	Err error
}

// Error returns Reason; when Err is non-nil, ": " + Err.Error() is appended.
// The format matches the convention used by [fmt.Errorf]("%s: %s", …) so a
// %w-unwrapping caller sees the same text it would from a plain wrapped error.
func (e *ToolHealthError) Error() string {
	if e == nil {
		return ""
	}
	if e.Err != nil {
		return e.Reason + ": " + e.Err.Error()
	}
	return e.Reason
}

// Unwrap returns the original underlying error, if any, so [errors.Is] and
// [errors.As] can traverse the chain.
func (e *ToolHealthError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

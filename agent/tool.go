package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

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
// Run must honor ctx cancellation. Run MAY BE INVOKED CONCURRENTLY: within a
// single run when the Runner was constructed with [WithParallelTools] (one
// goroutine per tool call in a turn), and across simultaneous Run calls on
// one Runner (a Runner is safe for concurrent Run calls). A Tool used under
// either arrangement must be safe for concurrent calls — unsynchronized state
// data-races rather than erroring, and races surface as silently wrong tool
// results. See the [Runner] concurrency paragraph and [WithParallelTools].
type Tool interface {
	// Def returns the tool's declaration (name, description, JSON-schema
	// parameters) as advertised to the model.
	Def() llmkit.ToolDef
	// Run executes the tool with the model-supplied arguments and returns the
	// textual result to feed back to the model.
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

// requireField returns an error if val (trimmed) is empty. name is the
// human-readable field name that appears in the error message.
func requireField(name, val string) error {
	if strings.TrimSpace(val) == "" {
		return fmt.Errorf("%s is required", name)
	}
	return nil
}

// requireLineNumber returns an error if n is less than 1. The error message
// follows the convention established by the existing tools ("line must be a
// 1-based line number").
func requireLineNumber(n int) error {
	if n < 1 {
		return fmt.Errorf("line must be a 1-based line number, got %d", n)
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
		name := t.Def().Name
		ts.byName[name] = t
		if !seen[name] {
			ts.defs = append(ts.defs, t.Def())
			seen[name] = true
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

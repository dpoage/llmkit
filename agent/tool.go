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
// Run receives the model's raw JSON arguments (decode them yourself) and
// returns the result as content blocks: []llmkit.Block{llmkit.Text(s)} for
// text, with image or document blocks beside it for a richer result. The
// Runner keeps the returned slice, which [Outcome.Messages] may hold itself;
// do not modify it after returning. A nil or empty result is one empty text
// block.
//
// The tool-result content is the returned blocks, except that a media block
// that is not portable becomes a text marker (the rule is on
// [llmkit.Capabilities.ToolResultMedia]) and, under [WithToolOutputLimit], an
// oversized result becomes a spill or truncation stand-in. Hooks and events
// receive the text blocks in order, with a size marker at each media block
// sent and an omission marker at each one degraded.
//
// An error from Run is not a loop failure: the harness renders "ERROR: <err>"
// from the error alone, discarding any blocks returned beside it, and the
// model decides how to recover. Reserve errors for tool-level problems; never
// use them to abort the loop.
//
// A panic from Run is treated the same way in both dispatch modes
// (sequential and [WithParallelTools]): the harness recovers it and renders
// it as that call's error result "ERROR: tool <name> panicked: <value>"
// with IsError=true, [Hooks.ToolEnd] fires with that result, and the run
// continues. A panic raised by adversarial model-supplied arguments is data
// for the model, not an abort — the harness never lets a tool panic kill the
// process or its caller. (A panicking HOOK is a harness bug with the
// opposite contract: it aborts the run and is never rendered to the model —
// see [Hooks].)
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
	// result blocks to feed back to the model. An error, or a panic, is
	// recovered by the harness and rendered to the model as this call's
	// "ERROR:"-prefixed result (identically in both dispatch modes); the
	// blocks returned beside an error are discarded.
	Run(ctx context.Context, args json.RawMessage) ([]llmkit.Block, error)
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

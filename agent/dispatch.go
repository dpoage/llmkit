package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/dpoage/llmkit"
)

// toolSet indexes tools by name for dispatch and collects their defs for the
// request.
type toolSet struct {
	byName map[string]Tool
	defs   []llmkit.ToolDef
}

// newToolSet builds a dispatch table from tools, preserving their order in
// the defs slice. It panics on a nil entry or on two tools sharing a
// Def().Name: either is a construction bug, and silently keeping one of a
// duplicate pair would advertise one tool's schema while dispatching another.
func newToolSet(tools []Tool) toolSet {
	ts := toolSet{byName: make(map[string]Tool, len(tools))}
	index := make(map[string]int, len(tools))
	for i, t := range tools {
		if t == nil {
			panic(fmt.Sprintf("agent: NewRunner: tool at index %d is nil", i))
		}
		def := t.Def()
		if first, dup := index[def.Name]; dup {
			panic(fmt.Sprintf("agent: NewRunner: duplicate tool name %q at indexes %d and %d", def.Name, first, i))
		}
		index[def.Name] = i
		ts.byName[def.Name] = t
		ts.defs = append(ts.defs, def)
	}
	return ts
}

// lookup returns the tool registered under name, if any.
func (ts toolSet) lookup(name string) (Tool, bool) {
	t, ok := ts.byName[name]
	return t, ok
}

// The tool-outcome renderers: the model-visible strings the Runner produces
// for a tool call that did not simply return output. A Run error is
// "ERROR: <err>".

// toolError prefixes a tool failure so the model recognizes it as a recoverable
// error rather than a normal result. The harness uses this for every error a
// Tool returns.
func toolError(err error) string {
	return "ERROR: " + err.Error()
}

// renderUnknownTool renders a call naming a tool the Runner does not have.
func renderUnknownTool(name string) string {
	return toolError(fmt.Errorf("unknown tool %q", name))
}

// renderToolPanic renders a recovered Tool.Run panic.
func renderToolPanic(name string, panicVal any) string {
	return fmt.Sprintf("ERROR: tool %s panicked: %v", name, panicVal)
}

// renderToolTimeout renders a call that outran [WithToolTimeout].
func renderToolTimeout(name string, d time.Duration) string {
	return fmt.Sprintf("ERROR: tool %s timed out after %s", name, d)
}

// renderToolDenied renders a [ToolPolicy] refusal; reason is the policy
// error's text, sent to the model verbatim.
func renderToolDenied(name, reason string) string {
	return fmt.Sprintf("ERROR: tool %s denied: %s", name, reason)
}

// renderToolNotRun renders a call skipped because the run's context was
// cancelled before the policy could authorize it.
func renderToolNotRun(name string, cause error) string {
	return toolError(fmt.Errorf("tool %s not run: %w", name, cause))
}

// authorizeCalls consults the policy once per call in model order on the
// caller's goroutine, returning the calls to dispatch (a private copy when
// a policy is installed) and the denied indexes. Denied results are written
// straight into results[i] so dispatch treats them like executed results.
// Cancellation denies every remaining registered call with the context
// error so neither mode can run a never-authorized call.
func (r *Runner) authorizeCalls(ctx context.Context, calls []llmkit.ToolCall, results []toolResult) ([]llmkit.ToolCall, []bool) {
	if r.toolPolicy == nil {
		return calls, nil
	}
	// Clone the argument bytes: the policy rewrites Arguments in place, so
	// neither the copy nor the RawMessage backing may alias the history.
	dispatch := make([]llmkit.ToolCall, len(calls))
	copy(dispatch, calls)
	denied := make([]bool, len(calls))
	for i := range dispatch {
		dispatch[i].Arguments = bytes.Clone(dispatch[i].Arguments)
		// Unregistered names never reach the policy; runTool owns that path.
		if _, ok := r.tools.lookup(dispatch[i].Name); !ok {
			continue
		}
		if ctx.Err() != nil {
			// Never authorized: render "not run", not "denied" — a cancelled
			// run is not a policy decision. The slice stays full-length and
			// the event carries IsError with the context error text, without
			// the Denied mark (that mark is for Authorize refusals only).
			denied[i] = true
			results[i] = notRunResult(calls[i].Name, ctx.Err())
			continue
		}
		if err := r.toolPolicy.Authorize(ctx, &dispatch[i]); err != nil {
			denied[i] = true
			results[i] = deniedResult(calls[i].Name, err.Error())
			continue
		}
		// Only Arguments is part of the rewrite contract: revert Name/ID.
		dispatch[i].Name = calls[i].Name
		dispatch[i].ID = calls[i].ID
	}
	return dispatch, denied
}

// runTool dispatches one tool call. A missing tool, a Run error, or a
// Tool.Run PANIC is returned to the model as an "ERROR:"-prefixed result
// (isErr=true) rather than aborting the loop — in BOTH dispatch modes, so
// toggling WithParallelTools never changes failure semantics. A panic is
// the tool-boundary conversion of a bug into data for the model; a hook
// panic (ToolStart/ToolEnd/ToolHealth) is a harness bug and is NOT recovered
// here (see [Runner.executeTools] for the parallel-mode path). Context
// cancellation surfaced by the tool is still rendered as a
// tool error here; the loop's own ctx checks handle real cancellation.
//
// With WithToolTimeout set, the call runs under a derived deadline: on
// expiry the model receives "ERROR: tool <name> timed out after <d>" and
// the loop continues — the run's own context is unaffected. A timeout is
// not a *ToolHealthError and does not fire [Hooks.ToolHealth].
//
// Hooks fire from the goroutine executing the call — concurrently under
// WithParallelTools.
func (r *Runner) runTool(ctx context.Context, call llmkit.ToolCall, step int) (result string, isErr bool) {
	tool, ok := r.tools.lookup(call.Name)
	if !ok {
		return renderUnknownTool(call.Name), true
	}
	runCtx := ctx
	if r.toolTimeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, r.toolTimeout)
		defer cancel()
	}
	r.fireToolStart(ctx, step, call)
	start := time.Now()
	out, panicVal, err := invokeTool(runCtx, tool, call.Arguments)
	duration := time.Since(start)
	if panicVal != nil {
		// A panicking Tool.Run is model-recoverable data: render it as this
		// call's error result and keep the loop going.
		result = renderToolPanic(call.Name, panicVal)
		r.fireToolEnd(ctx, step, call, result, true, duration)
		return result, true
	}
	if err != nil {
		// Deadline expiry: report the timeout as the tool result, leaving the
		// run's own context untouched. The parent-ctx guard keeps a genuine
		// run cancellation from being misreported as a tool timeout.
		if r.toolTimeout > 0 && ctx.Err() == nil && runCtx.Err() == context.DeadlineExceeded {
			timeoutResult := renderToolTimeout(call.Name, r.toolTimeout)
			r.fireToolEnd(ctx, step, call, timeoutResult, true, duration)
			return timeoutResult, true
		}
		// Record a tool-health signal only for a genuine *ToolHealthError AND
		// only when ctx is not already cancelled: a failure caused by run
		// teardown/cancellation must never be counted as a tool-health problem.
		r.fireToolHealth(ctx, call.Name, err)
		result = toolError(err)
		r.fireToolEnd(ctx, step, call, result, true, duration)
		return result, true
	}
	r.fireToolEnd(ctx, step, call, out, false, duration)
	return out, false
}

// invokeTool invokes Tool.Run once, converting a panic into a non-nil
// second return so the harness decides how to render it. A panic recovered
// here never reaches the dispatching goroutine's own recover, which (in
// parallel mode) is reserved for harness bugs — hook panics — only. A tool
// that panics with nil still yields a non-nil marker: runtime turns
// panic(nil) into a *runtime.PanicNilError.
func invokeTool(ctx context.Context, tool Tool, args json.RawMessage) (out string, panicVal any, err error) {
	defer func() {
		if v := recover(); v != nil {
			out, panicVal, err = "", v, nil
		}
	}()
	out, err = tool.Run(ctx, args)
	return out, nil, err
}

// toolResult is one executed tool call's outcome, as returned by
// [Runner.executeTools]. denied marks a [ToolPolicy] denial: result still
// carries the rendered model-visible text, while denyReason records the
// policy's reason for the ToolRun event.
type toolResult struct {
	result     string
	isErr      bool
	denied     bool
	denyReason string
}

// deniedResult renders a [ToolPolicy] refusal: the model-visible denial
// text plus the Denied mark and the policy's reason for the ToolRun event.
func deniedResult(name, reason string) toolResult {
	return toolResult{result: renderToolDenied(name, reason), isErr: true, denied: true, denyReason: reason}
}

// notRunResult renders a call that was never authorized because ctx was
// cancelled: an error result carrying the context error, without the Denied
// mark (that mark is for Authorize refusals only).
func notRunResult(name string, err error) toolResult {
	return toolResult{result: renderToolNotRun(name, err), isErr: true}
}

// message is the tool-result turn that feeds this outcome back to the model
// for the call callID: ToolError adds the IsError mark for a failed
// execution or a denial.
func (res toolResult) message(callID string) llmkit.Message {
	if res.isErr {
		return llmkit.ToolError(callID, res.result)
	}
	return llmkit.ToolResult(callID, res.result)
}

// executeTools runs calls and returns one result per executed call, in the
// model's original order. A [ToolPolicy], when installed, authorizes — and
// may rewrite — every call BEFORE the first Tool.Run dispatches (see
// [Runner.authorizeCalls]); a denied call keeps its slot with its rendered
// error result, so the returned slice stays index-aligned with the calls in
// both modes — including calls the pre-pass could not authorize because ctx
// was already cancelled: those are denied with the context error, never
// dispatched. Sequential mode (the default) runs calls one at a time and
// stops before dispatching the next call once ctx is cancelled — the
// returned slice then holds only the already-executed (or policy-resolved)
// results. Parallel mode (WithParallelTools) runs each call on its own
// goroutine (bounded by len(calls)) and waits for all of them: per-call
// failures are isolated (a tool error or panic never fails its siblings —
// runTool renders a panic as that call's error result), and the full-length
// result slice is always returned. Neither mode mutates the conversation or
// the transcript; the caller appends the results in call order after
// executeTools returns.
//
// Hook panics are harness bugs and are never rendered to the model. When the
// calls run one at a time, a panicking hook is not recovered here, so its
// panic unwinds through this function. When they run in parallel, the
// per-call goroutine recovers it, stashes the first hook panic to arrive, and
// this function re-panics with the original value AFTER all sibling
// goroutines have finished. Either way no result is recorded for the
// interrupted turn.
func (r *Runner) executeTools(ctx context.Context, outcome *Outcome, calls []llmkit.ToolCall) []toolResult {
	results := make([]toolResult, len(calls))
	// The turn's Step rides the whole tool phase's context — the ToolPolicy
	// (and any decision observer inside one), the ToolStart/ToolEnd hooks,
	// and Tool.Run itself, hence any decorator a tool calls through — so
	// decorator-emitted events carry the same turn the Runner's own ToolRun
	// events name.
	ctx = llmkit.WithStep(ctx, outcome.Iterations)
	// Every call of the turn is authorized before the first Tool.Run, in
	// both modes, so an interactive policy never overlaps the fan-out.
	dispatch, denied := r.authorizeCalls(ctx, calls, results)
	if !r.parallelTools || len(calls) < 2 {
		for i, call := range dispatch {
			if denied != nil && denied[i] {
				continue // deny already rendered into results[i]
			}
			if err := ctx.Err(); err != nil {
				return results[:i]
			}
			results[i].result, results[i].isErr = r.runTool(ctx, call, outcome.Iterations)
		}
		return results
	}
	var wg sync.WaitGroup
	var hookPanic any // the first hook panic, re-panicked below
	var panicMu sync.Mutex
	for i, call := range dispatch {
		wg.Add(1)
		go func(i int, call llmkit.ToolCall) {
			defer wg.Done()
			// runTool fully contains tool panics; anything that escapes it
			// into this goroutine is a harness bug (a hook). Stash the first
			// one and let every sibling finish — the loop goroutine re-panics
			// with the original value after wg.Wait, so the panic never dies
			// in this goroutine and never renders as tool output.
			defer func() {
				if v := recover(); v != nil {
					panicMu.Lock()
					if hookPanic == nil {
						hookPanic = v
					}
					panicMu.Unlock()
				}
			}()
			if denied != nil && denied[i] {
				return // deny already rendered into results[i]
			}
			results[i].result, results[i].isErr = r.runTool(ctx, call, outcome.Iterations)
		}(i, call)
	}
	wg.Wait()
	if hookPanic != nil {
		panic(hookPanic)
	}
	return results
}

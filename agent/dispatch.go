package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

// toolSet indexes tools by name for dispatch and collects their defs for the
// request.
type toolSet struct {
	byName map[string]Tool
	defs   []llmkit.ToolDef
}

// newToolSet builds a dispatch table from tools, preserving their order in
// the defs slice. It panics on a nil entry, on a tool whose Def().Name is
// empty, or on two tools sharing a Def().Name: each is a construction bug,
// and silently keeping one of a duplicate pair would advertise one tool's
// schema while dispatching another.
func newToolSet(tools []Tool) toolSet {
	ts := toolSet{byName: make(map[string]Tool, len(tools))}
	index := make(map[string]int, len(tools))
	for i, t := range tools {
		if t == nil {
			panic(fmt.Sprintf("agent: NewRunner: tool at index %d is nil", i))
		}
		def := t.Def()
		if def.Name == "" {
			panic(fmt.Sprintf("agent: NewRunner: tool at index %d has an empty name", i))
		}
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
	return "ERROR: " + jsonSafeUTF8(err.Error())
}

// renderUnknownTool renders a call naming a tool the Runner does not have.
func renderUnknownTool(name string) string {
	return toolError(fmt.Errorf("unknown tool %q", name))
}

// renderToolPanic renders a recovered Tool.Run panic.
func renderToolPanic(name string, panicVal any) string {
	return jsonSafeUTF8(fmt.Sprintf("ERROR: tool %s panicked: %v", name, panicVal))
}

// jsonSafeUTF8 applies encoding/json's string rule: each byte that does not
// start a valid UTF-8 sequence becomes one U+FFFD (ff fe 'a' -> "\ufffd\ufffda").
func jsonSafeUTF8(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + utf8.UTFMax)
	for i := 0; i < len(s); {
		if r, size := utf8.DecodeRuneInString(s[i:]); r == utf8.RuneError && size == 1 {
			b.WriteString("\uFFFD")
			i++
			continue
		} else {
			b.WriteString(s[i : i+size])
			i += size
		}
	}
	return b.String()
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
		// A rewrite is what Tool.Run receives, so the ToolRun event carries
		// it beside the model's call. Snapshot it here: a tool may write to
		// the bytes it is handed. A cleared payload is spelled null, since
		// an empty value would read as "not rewritten". A rewrite to bytes
		// that are not JSON is left out: the event could not be encoded.
		if !argsEqual(dispatch[i].Arguments, calls[i].Arguments) {
			switch {
			case len(dispatch[i].Arguments) == 0:
				results[i].dispatchedArgs = json.RawMessage("null")
			case json.Valid(dispatch[i].Arguments):
				results[i].dispatchedArgs = bytes.Clone(dispatch[i].Arguments)
			}
		}
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
func (r *Runner) runTool(ctx context.Context, call llmkit.ToolCall, step int) toolResult {
	tool, ok := r.tools.lookup(call.Name)
	if !ok {
		return toolResult{result: renderUnknownTool(call.Name), isErr: true}
	}
	runCtx := ctx
	if r.toolTimeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, r.toolTimeout)
		defer cancel()
	}
	// Tool.Run learns which call it serves; a replay tool reports the
	// degradation its recorded result holds. Hooks get the run's ctx.
	runCtx, slot := withDispatch(runCtx, call.ID)
	r.fireToolStart(ctx, step, call)
	start := time.Now()
	out, panicVal, err := invokeTool(runCtx, tool, call.Arguments)
	duration := time.Since(start)
	if panicVal != nil {
		// A panicking Tool.Run is model-recoverable data: render it as this
		// call's error result and keep the loop going.
		res := toolResult{result: renderToolPanic(call.Name, panicVal), isErr: true}
		r.fireToolEnd(ctx, step, call, res, duration)
		return res
	}
	if err != nil {
		// Deadline expiry: report the timeout as the tool result, leaving the
		// run's own context untouched. The parent-ctx guard keeps a genuine
		// run cancellation from being misreported as a tool timeout.
		if r.toolTimeout > 0 && ctx.Err() == nil && runCtx.Err() == context.DeadlineExceeded {
			res := toolResult{result: renderToolTimeout(call.Name, r.toolTimeout), isErr: true}
			r.fireToolEnd(ctx, step, call, res, duration)
			return res
		}
		// Record a tool-health signal only for a genuine *ToolHealthError AND
		// only when ctx is not already cancelled: a failure caused by run
		// teardown/cancellation must never be counted as a tool-health problem.
		r.fireToolHealth(ctx, call.Name, err)
		res := toolResult{result: toolError(err), isErr: true}
		r.fireToolEnd(ctx, step, call, res, duration)
		return res
	}
	res := r.placeToolBlocks(out)
	res.degraded += slot.priorDegraded
	if rec := slot.recorded; rec != nil {
		res.originalBytes, res.spillRef, res.spillErr = rec.originalBytes, rec.spillRef, rec.spillErr
	} else {
		res = r.limitToolOutput(ctx, call, res)
	}
	r.fireToolEnd(ctx, step, call, res, duration)
	return res
}

const (
	imageOmitted    = "[image omitted: this model does not accept images in tool results]"
	documentOmitted = "[document omitted: this model does not accept documents in tool results]"
)

// portableToolMedia reports whether every ToolResultMedia wire carries b: inline
// bytes of an image in png, jpeg or webp, or of a pdf document.
func portableToolMedia(b llmkit.Block) bool {
	if len(b.Data) == 0 {
		return false
	}
	switch b.MediaType {
	case "image/png", "image/jpeg", "image/webp":
		return b.Kind == llmkit.BlockImage
	case "application/pdf":
		return b.Kind == llmkit.BlockDocument
	}
	return false
}

func isMedia(b llmkit.Block) bool {
	return b.Kind == llmkit.BlockImage || b.Kind == llmkit.BlockDocument
}

// placeToolBlocks builds the outcome of a successful Tool.Run, degrading each
// media block the client's profile cannot carry in a tool result.
func (r *Runner) placeToolBlocks(out []llmkit.Block) toolResult {
	out = normalizeTextBlocks(out)
	if !slices.ContainsFunc(out, isMedia) {
		return toolResult{result: blocksText(out), content: nonEmptyBlocks(out)}
	}
	placed, degraded := degradeToolMedia(out, r.client.Capabilities().ToolResultMedia)
	return toolResult{result: blocksText(placed), content: placed, degraded: degraded}
}

// normalizeTextBlocks applies jsonSafeUTF8 to every text, copying only when
// some text is invalid; the tool's slice is never modified.
func normalizeTextBlocks(blocks []llmkit.Block) []llmkit.Block {
	if !slices.ContainsFunc(blocks, func(b llmkit.Block) bool { return !utf8.ValidString(b.Text) }) {
		return blocks
	}
	cp := make([]llmkit.Block, len(blocks))
	copy(cp, blocks)
	for i := range cp {
		cp[i].Text = jsonSafeUTF8(cp[i].Text)
	}
	return cp
}

// degradeToolMedia replaces each media block a tool result cannot carry by its
// omission marker; it returns blocks itself, never modified, when none is.
func degradeToolMedia(blocks []llmkit.Block, accepts bool) ([]llmkit.Block, int) {
	var placed []llmkit.Block
	degraded := 0
	for i, b := range blocks {
		if !isMedia(b) || (accepts && portableToolMedia(b)) {
			if placed != nil {
				placed = append(placed, b)
			}
			continue
		}
		if placed == nil {
			placed = append(make([]llmkit.Block, 0, len(blocks)), blocks[:i]...)
		}
		marker := imageOmitted
		if b.Kind == llmkit.BlockDocument {
			marker = documentOmitted
		}
		placed = append(placed, llmkit.Text(marker))
		degraded++
	}
	if placed == nil {
		return blocks, 0
	}
	return placed, degraded
}

// wireMessages degrades the media of every tool-result message of history,
// which Continue can carry in from another profile. history is never modified;
// copied reports a fresh slice.
func (r *Runner) wireMessages(history []llmkit.Message) (msgs []llmkit.Message, copied bool) {
	var accepts, read bool
	for i, m := range history {
		if m.Role != llmkit.RoleToolResult || !slices.ContainsFunc(m.Content, isMedia) {
			continue
		}
		if !read {
			accepts, read = r.client.Capabilities().ToolResultMedia, true
		}
		content, n := degradeToolMedia(m.Content, accepts)
		if n == 0 {
			continue
		}
		if msgs == nil {
			msgs = slices.Clone(history)
		}
		msgs[i].Content = content
	}
	if msgs == nil {
		return history, false
	}
	return msgs, true
}

func nonEmptyBlocks(blocks []llmkit.Block) []llmkit.Block {
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

// blocksText is the model-visible text of placed blocks: text in order, with a
// size marker at each media block sent.
func blocksText(blocks []llmkit.Block) string {
	var b strings.Builder
	for _, blk := range blocks {
		switch blk.Kind {
		case llmkit.BlockText:
			b.WriteString(blk.Text)
		case llmkit.BlockImage, llmkit.BlockDocument:
			fmt.Fprintf(&b, "[%s: %s, %d bytes]", blk.Kind, blk.MediaType, len(blk.Data))
		}
	}
	return b.String()
}

// invokeTool invokes Tool.Run once, converting a panic into a non-nil
// second return so the harness decides how to render it. A panic recovered
// here never reaches the dispatching goroutine's own recover, which (in
// parallel mode) is reserved for harness bugs — hook panics — only. A tool
// that panics with nil still yields a non-nil marker: runtime turns
// panic(nil) into a *runtime.PanicNilError.
func invokeTool(ctx context.Context, tool Tool, args json.RawMessage) (out []llmkit.Block, panicVal any, err error) {
	defer func() {
		if v := recover(); v != nil {
			out, panicVal, err = nil, v, nil
		}
	}()
	out, err = tool.Run(ctx, args)
	return out, nil, err
}

// toolResult is one executed tool call's outcome, as returned by
// [Runner.executeTools]. denied marks a [ToolPolicy] denial: result still
// carries the rendered model-visible text, while denyReason records the
// policy's reason for the ToolRun event. dispatchedArgs is set only when a
// policy rewrote the arguments Tool.Run received (see
// [llmkit.ToolRunEvent].DispatchedArguments).
type toolResult struct {
	result string
	// content is the blocks fed to the model; nil for harness-rendered outcomes
	// and for a tool that returned none, when the message carries result.
	content []llmkit.Block
	// degraded counts media blocks replaced by omission markers, a replayed
	// record's included; content may since be a stand-in holding none.
	degraded int
	// originalBytes, spillRef and spillErr describe an applied output limit.
	originalBytes  int
	spillRef       string
	spillErr       string
	isErr          bool
	denied         bool
	denyReason     string
	dispatchedArgs json.RawMessage
}

// eventContent is the Content the ToolRun event and ToolEvent carry: the
// placed blocks whenever they are not exactly one text block, else nil.
func (res toolResult) eventContent() []llmkit.Block {
	if res.degraded > 0 || len(res.content) != 1 || res.content[0].Kind != llmkit.BlockText {
		return res.content
	}
	return nil
}

type dispatchKey struct{}

// dispatchSlot is shared by [Runner.runTool] and the Tool.Run it invokes.
// priorDegraded and recorded are set by a replay tool from its record.
type dispatchSlot struct {
	id            string
	priorDegraded int
	recorded      *recordedLimit
}

// recordedLimit is the output-limit outcome of a recorded result; a result
// served with one is never limited again.
type recordedLimit struct {
	originalBytes int
	spillRef      string
	spillErr      string
}

func withDispatch(ctx context.Context, id string) (context.Context, *dispatchSlot) {
	slot := &dispatchSlot{id: id}
	return context.WithValue(ctx, dispatchKey{}, slot), slot
}

// callIDFrom returns the ID of the call Tool.Run serves, or "" when ctx
// carries none (a tool invoked outside a Runner's dispatch, or a call the
// model gave no ID). Only replay tools read it, to match a served call to its
// recorded counterpart by identity instead of by arguments.
func callIDFrom(ctx context.Context) string {
	if slot, ok := ctx.Value(dispatchKey{}).(*dispatchSlot); ok {
		return slot.id
	}
	return ""
}

// reportPriorDegraded marks that Tool.Run's blocks already hold n omission
// markers from the recorded run.
func reportPriorDegraded(ctx context.Context, n int) {
	if slot, ok := ctx.Value(dispatchKey{}).(*dispatchSlot); ok {
		slot.priorDegraded = n
	}
}

// reportRecordedLimit marks that Tool.Run's blocks are a recorded result
// already limited, so the Runner's output limit must not touch them.
func reportRecordedLimit(ctx context.Context, rec recordedLimit) {
	if slot, ok := ctx.Value(dispatchKey{}).(*dispatchSlot); ok {
		slot.recorded = &rec
	}
}

func (res *toolResult) setText(s string) {
	res.result = s
	res.content = []llmkit.Block{llmkit.Text(s)}
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
	msg := llmkit.ToolResult(callID, res.result)
	if res.content != nil {
		msg.Content = res.content
	}
	return msg
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
			out := r.runTool(ctx, call, outcome.Iterations)
			out.dispatchedArgs = results[i].dispatchedArgs
			results[i] = out
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
			out := r.runTool(ctx, call, outcome.Iterations)
			out.dispatchedArgs = results[i].dispatchedArgs
			results[i] = out
		}(i, call)
	}
	wg.Wait()
	if hookPanic != nil {
		panic(hookPanic)
	}
	return results
}

package agent

import (
	"context"
	"errors"
	"time"

	"github.com/dpoage/llmkit"
)

// This file is the Runner's record side: it builds every [llmkit.Event] a run
// emits and is the only place a [Hooks] callback fires. The loop and the
// dispatcher call the fire* helpers below at each point; a helper fires the
// registered ([WithHooks]) callbacks for its event in registration order,
// is a no-op when none is set, so the loop's ordering is the only ordering.

// runEmitter is one run's emission path: the in-memory [Transcript] (always
// present; it backs [Outcome.Transcript]) plus the observer chain that fans
// each event out to the transcript first and then to every [WithObserver]
// sink in registration order — assembled once per run with
// [llmkit.Observers].
type runEmitter struct {
	tr  *Transcript
	obs llmkit.Observer
}

// emit fans ev out through the run's chain, stamping only the run-level fact
// the Runner owns: ParentRunID for a continued run. Step is each emission
// site's to set.
func (e runEmitter) emit(ctx context.Context, ev llmkit.Event) {
	if e.tr.ParentRunID != "" {
		ev.ParentRunID = e.tr.ParentRunID
	}
	e.obs.Observe(ctx, ev)
}

// begin arms a run: it mints (or adopts, via [WithRunID]) the run's
// [llmkit.RunID], places it in the context every emitter reads, builds the
// run's observer chain, and emits the run's Start event — the first event of
// every run, so a sink can open its run record before the first Completion.
func (r *Runner) begin(ctx context.Context, cfg runConfig, task string) (context.Context, runEmitter) {
	runID := cfg.runID
	if runID == "" {
		runID = llmkit.NewRunID()
	}
	ctx = llmkit.WithRun(ctx, runID)
	tr := NewTranscript()
	tr.RunID, tr.ParentRunID = runID, cfg.parentRunID
	chain := make([]llmkit.Observer, 0, 1+len(r.observers))
	chain = append(chain, tr)
	chain = append(chain, r.observers...)
	em := runEmitter{tr: tr, obs: llmkit.Observers(chain...)}
	ev := llmkit.NewEvent(ctx, llmkit.KindStart)
	// A run opens OUTSIDE any turn: Start is Step 0 even when the context
	// already carries a step — a nested Runner started from inside a
	// parent's tool phase, or a caller pre-arming WithStep. The child run's
	// turns number from 1 in their own right.
	ev.Step = 0
	ev.Start = &llmkit.StartEvent{Task: task, Tools: r.toolNames()}
	em.emit(ctx, ev)
	return ctx, em
}

// emitFinalize emits the run's Finalize event. On a return it reports the
// outcome's status, usage, and truncation reason, with Step the number of
// completed turns (Outcome.Iterations); a failed completion does not advance
// it, so a run whose only completion failed reports Step 0. With panicked set
// it reports Status RunPanicked with every other payload field zero and Step
// 0, whatever step the context carries (a pre-armed [llmkit.WithStep], or a
// parent's tool phase around a nested Runner).
func (r *Runner) emitFinalize(ctx context.Context, em runEmitter, o *Outcome, err error, panicked bool) {
	ev := llmkit.NewEvent(ctx, llmkit.KindFinalize)
	if panicked {
		ev.Step = 0
		ev.Finalize = &llmkit.FinalizeEvent{Status: llmkit.RunPanicked}
	} else {
		ev.Step = o.Iterations
		ev.Finalize = &llmkit.FinalizeEvent{
			TruncationReason: string(o.TruncationReason),
			Finalized:        o.Finalized,
			Usage:            o.Usage,
			FinalText:        o.FinalText,
			Status:           runStatus(ctx, err),
		}
		if err != nil {
			ev.Finalize.Err = err.Error()
		}
	}
	em.emit(ctx, ev)
}

func runStatus(ctx context.Context, err error) llmkit.RunStatus {
	var inc *IncompleteError
	var stop *StopReasonError
	switch {
	case err == nil:
		return llmkit.RunCompleted
	case errors.As(err, &inc):
		return llmkit.RunIncomplete
	case errors.As(err, &stop):
		return llmkit.RunRefused
	case ctx.Err() != nil:
		return llmkit.RunCanceled
	default:
		return llmkit.RunFailed
	}
}

// steerEvent builds the Steer event for one delivered steering turn: Step is
// the turn the message was delivered before (the completion about to run),
// FollowUp marks a follow-up queued for the would-be finish.
func steerEvent(ctx context.Context, msg llmkit.Message, followUp bool, step int) llmkit.Event {
	ev := llmkit.NewEvent(ctx, llmkit.KindSteer)
	ev.Step = step
	ev.Steer = &llmkit.SteerEvent{Message: msg, FollowUp: followUp}
	return ev
}

// toolRunEvent builds the ToolRun event for one dispatched call: the model's
// call (never the policy's rewrite; a rewrite rides DispatchedArguments), the
// textual result exactly as fed to the model, and the error or
// policy-denial marks. A denial carries Denied and DenyReason with no result
// — the rendered denial text still rides the conversation as the
// tool-result message, per [ToolPolicy]'s contract.
func toolRunEvent(ctx context.Context, call llmkit.ToolCall, res toolResult, step int) llmkit.Event {
	ev := llmkit.NewEvent(ctx, llmkit.KindToolRun)
	ev.Step = step
	tre := &llmkit.ToolRunEvent{Call: call, DispatchedArguments: res.dispatchedArgs}
	if res.denied {
		tre.Denied = true
		tre.DenyReason = res.denyReason
	} else {
		tre.Result = res.result
		tre.IsError = res.isErr
	}
	ev.ToolRun = tre
	return ev
}

// toolNames lists the tool names offered to the model, in offer order — the
// Start event's Tools payload.
func (r *Runner) toolNames() []string {
	names := make([]string, len(r.tools.defs))
	for i, d := range r.tools.defs {
		names[i] = d.Name
	}
	return names
}

// recordCompaction reports one real history prune as the Compaction event on
// the run's stream. step is the completion that consumes the compacted
// history, and the token totals come from the bytes/4 estimate the compaction
// trigger uses.
func (r *Runner) recordCompaction(ctx context.Context, em runEmitter, step int, before, after int64, pruned int) {
	ev := llmkit.NewEvent(ctx, llmkit.KindCompaction)
	ev.Step = step
	ev.Compaction = &llmkit.CompactionEvent{
		BeforeTokens: before,
		AfterTokens:  after,
		Pruned:       pruned,
	}
	em.emit(ctx, ev)
}

// emitCompletion emits the Completion event for one logical completion,
// success or failure (Err set, zero Response). ctx must be the ctx the client
// saw, so the event's SpanID matches the one the client (and any Attempt
// events it emitted) carried. Provider and Model come from the client's own
// Identity, never from arguments to the emitter.
func (r *Runner) emitCompletion(ctx context.Context, em runEmitter, step int, start time.Time, req llmkit.Request, resp llmkit.Response, err error) {
	ev := llmkit.NewEvent(ctx, llmkit.KindCompletion)
	ev.Step = step
	ev.Duration = time.Since(start)
	identity := llmkit.IdentityOf(r.client)
	ce := &llmkit.CompletionEvent{Request: req, Response: resp, Provider: identity.Provider, Model: identity.Model}
	if err != nil {
		ce.Err = err.Error()
		ce.Response = llmkit.Response{}
	}
	ev.Completion = ce
	em.emit(ctx, ev)
}

// fireBeforeCompletion fires every registered [Hooks.BeforeCompletion], in
// registration order, with the mutable request.
func (r *Runner) fireBeforeCompletion(ctx context.Context, step int, req *llmkit.Request) {
	for _, h := range r.hooks {
		if h.BeforeCompletion != nil {
			h.BeforeCompletion(ctx, step, req)
		}
	}
}

// deltaSink returns the streaming callback that forwards each delta to every
// registered [Hooks.Delta] on ctx, in registration order, or nil when none is
// set — the caller's cue that the turn stays on the plain Complete path.
func (r *Runner) deltaSink(ctx context.Context, step int) func(llmkit.Delta) error {
	i := 0
	for i < len(r.hooks) && r.hooks[i].Delta == nil {
		i++
	}
	if i == len(r.hooks) {
		return nil
	}
	// The closure loops the registrations itself: no per-completion slice of
	// the Delta callbacks is built.
	return func(d llmkit.Delta) error {
		for _, h := range r.hooks[i:] {
			if h.Delta != nil {
				h.Delta(ctx, step, d)
			}
		}
		return nil
	}
}

// fireAfterCompletion fires every registered [Hooks.AfterCompletion], in
// registration order, once the client returned. resp points at the caller's
// own Response, handed to each hook as-is; it is nil-ed when err is non-nil,
// so a failed completion reports no response.
func (r *Runner) fireAfterCompletion(ctx context.Context, step int, req *llmkit.Request, resp *llmkit.Response, err error) {
	if err != nil {
		resp = nil
	}
	for _, h := range r.hooks {
		if h.AfterCompletion != nil {
			h.AfterCompletion(ctx, step, req, resp, err)
		}
	}
}

// fireToolStart fires every registered [Hooks.ToolStart], in registration
// order, for a call about to run.
func (r *Runner) fireToolStart(ctx context.Context, step int, call llmkit.ToolCall) {
	for _, h := range r.hooks {
		if h.ToolStart != nil {
			h.ToolStart(ctx, ToolEvent{Step: step, Call: call})
		}
	}
}

// fireToolEnd fires every registered [Hooks.ToolEnd], in registration order,
// with the result exactly as fed to the model.
func (r *Runner) fireToolEnd(ctx context.Context, step int, call llmkit.ToolCall, result string, isErr bool, d time.Duration) {
	for _, h := range r.hooks {
		if h.ToolEnd != nil {
			h.ToolEnd(ctx, ToolEvent{Step: step, Call: call, Result: result, IsError: isErr, Duration: d})
		}
	}
}

// fireToolHealth fires every registered [Hooks.ToolHealth], in registration
// order, only for a genuine *ToolHealthError in err's chain, and only while
// ctx is live.
func (r *Runner) fireToolHealth(ctx context.Context, name string, err error) {
	var he *ToolHealthError
	if ctx.Err() != nil || !errors.As(err, &he) {
		return
	}
	for _, h := range r.hooks {
		if h.ToolHealth != nil {
			h.ToolHealth(ctx, name, he)
		}
	}
}

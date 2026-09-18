package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Runner drives an [llmkit.Client] through a tool-call loop. Construct one per
// agent role with that role's system prompt and tool set, then call
// [Runner.Run], [Runner.RunContinue], or [Runner.RunJSON] per task.
//
// A Runner is safe for concurrent use: every field is fixed at construction,
// and all mutable state (conversation, transcript, budget accounting,
// compaction thresholds, nudge counters) is local to a single Run call.
// Tools inherit an obligation from this: a [Tool] passed to one Runner may be
// invoked concurrently (see [Tool.Run] and [WithParallelTools]), so it must
// be safe for concurrent calls. Observability hooks (WithHooks) fire from
// whichever goroutine reaches the event — concurrently under
// WithParallelTools or concurrent Run calls — so hook functions must
// synchronize their own state.
type Runner struct {
	client       llmkit.Client
	tools        toolSet
	systemPrompt string
	limits       Limits

	// transcriptDir, when non-empty, receives an auto-saved JSONL transcript per
	// run, named "<timestamp>-<slug>.jsonl" (or "<timestamp>-<key>-<slug>.jsonl"
	// when transcriptKey is set — see WithTranscriptKey).
	transcriptDir string
	// transcriptKey, when non-empty, is embedded in the autosave filename so a
	// caller that knows a stable identifier for this run BEFORE constructing the
	// Runner (e.g. a store row's primary key, generated up front) can recover
	// the exact transcript file later by filename match instead of guessing from
	// a timestamp window. See WithTranscriptKey.
	transcriptKey string
	// maxTokens caps output tokens per completion (passed through to the client).
	// Zero lets the adapter apply its own default.
	maxTokens int
	// hooks holds the optional observer callbacks; nil funcs are no-ops with
	// zero overhead. See [Hooks] for the fire points.
	hooks Hooks
	// toolTimeout, when positive, is the per-call deadline applied to every
	// Tool.Run (see WithToolTimeout).
	toolTimeout time.Duration
	// parallelTools, when true, dispatches one turn's tool calls concurrently
	// (see WithParallelTools).
	parallelTools bool
}

// Option configures a [Runner] at construction.
type Option func(*Runner)

// WithLimits sets the per-run iteration and token limits. Zero fields resolve
// to package defaults.
func WithLimits(l Limits) Option {
	return func(r *Runner) { r.limits = l }
}

// WithTranscriptDir makes each run auto-save its transcript to a JSONL file
// under dir, named "<RFC3339-timestamp>-<task-slug>.jsonl" (or
// "<RFC3339-timestamp>-<key>-<task-slug>.jsonl" when WithTranscriptKey is also
// set). The directory is created on demand.
func WithTranscriptDir(dir string) Option {
	return func(r *Runner) { r.transcriptDir = dir }
}

// WithTranscriptKey embeds key in the autosave filename between the timestamp
// and the task slug, giving a caller that mints a stable identifier for this
// run BEFORE constructing the Runner (typically a store row's primary key,
// generated up front so it can be threaded through both the runner and the
// eventual row insert) an EXACT way to recover the transcript file later, by
// filename match, instead of a timestamp-window guess. A no-op unless
// WithTranscriptDir is also set. Empty key is a no-op (preserves the plain
// "<timestamp>-<slug>.jsonl" naming used by callers with no stable key —
// e.g. callers that only learn a run's stable ID from their own store after
// the run completes, and so have nothing to key by up front).
func WithTranscriptKey(key string) Option {
	return func(r *Runner) { r.transcriptKey = key }
}

// WithMaxTokens caps output tokens per completion. Zero uses the adapter
// default.
func WithMaxTokens(n int) Option {
	return func(r *Runner) { r.maxTokens = n }
}

// NewRunner builds a Runner bound to client, the given tools, and a system
// prompt. Options tune limits, transcript persistence, and output token caps.
func NewRunner(client llmkit.Client, tools []Tool, systemPrompt string, opts ...Option) *Runner {
	r := &Runner{
		client:       client,
		tools:        newToolSet(tools),
		systemPrompt: systemPrompt,
	}
	for _, opt := range opts {
		opt(r)
	}
	r.limits = r.limits.resolve()
	return r
}

// Run executes the tool loop for a single task. It seeds the conversation with
// the system prompt and the task as a user message, then repeatedly calls the
// model and executes any requested tools until the model finishes its turn, a
// limit is hit, or an infrastructure error occurs.
//
// Limit exhaustion is not an error: it returns an [Outcome] with Truncated set
// and the last assistant text preserved. Only context cancellation and
// client/IO failures return a non-nil error. The returned Outcome's Transcript
// is always non-nil, even on error, capturing whatever happened before the
// failure.
//
// Max-tokens continuation: when a turn stops at the output token cap
// (StopMaxTokens) with no tool calls, Run makes ONE extra continuation
// completion that turn — nudging the model to emit the rest — and stitches the
// two halves into a single assistant message (see [Runner.completeOnce]). This
// applies to plain Run, not only RunJSON: a truncated final answer is completed
// rather than returned half-written. It costs at most one additional completion
// per truncated turn and is reflected in the Outcome's Iterations and Usage.
func (r *Runner) Run(ctx context.Context, task string) (*Outcome, error) {
	return r.run(ctx, nil, task, "", nil)
}

// RunContinue behaves exactly like [Runner.Run] except it CONTINUES a prior
// conversation instead of reseeding one: prev.Messages (the Messages field of
// an earlier Run/RunContinue call's Outcome, on this same Runner) becomes the
// starting history, and task is appended as a NEW user turn rather than
// becoming the conversation's sole seed message. A nil prev, or a prev with no
// Messages, degrades to plain reseeding — identical to Run — so a first-turn
// caller can use this method unconditionally without a nil check.
//
// The same-Runner contract from [Runner.RunJSONContinue] applies: prev must
// come from a run of THIS Runner — the system prompt and the tool set must
// match the seed — and Limits apply per call: WithLimits iteration and token
// budgets are re-armed fresh for each RunContinue (a [BudgetPool] via
// [Limits.BudgetCheck] already spans runs).
//
// If the seed's final assistant turn carries tool calls that were never
// answered (the history a context-cancelled run returns), that trailing turn
// is dropped before the run starts — see [trimDanglingToolTurn] — so the wire
// request is always well-formed.
//
// Trimming can leave two consecutive user turns on the wire (the cancelled
// tool turn is dropped whole, so nothing answers between them), and a seed
// ending with the empty-turn nudge does the same; the in-tree adapters send
// consecutive user messages as-is and the providers accept or coalesce them.
//
// Like Run, there is no context-window management: a caller driving a long
// conversation must bound the history itself, using the client's
// [llmkit.Capabilities].ContextWindow and [EstimateHistoryTokens]. There is
// likewise no streaming and no persistence across processes.
//
// When an earlier call returned [ErrStopReason] (model refusal/safety stop),
// the attached err.Outcome may be threaded back in here: the refusal turn
// stays in the history and the conversation proceeds from it.
func (r *Runner) RunContinue(ctx context.Context, prev *Outcome, task string) (*Outcome, error) {
	var seed []llmkit.Message
	if prev != nil {
		seed = prev.Messages
	}
	return r.run(ctx, seed, task, "", nil)
}

// run is the shared loop body. seed, when non-nil, is a prior conversation to
// continue: task is appended as a NEW user turn onto seed instead of becoming
// the conversation's sole seed message. Run passes seed == nil (reseed every
// call); [Runner.RunJSONContinue] and [Runner.RunContinue] pass a prior
// Outcome's Messages (nil or empty degrades to reseeding) so the next round
// lands in the SAME conversation as the one that produced it. Before the task
// is appended, a seed whose trailing assistant turn carries unanswered tool
// calls is trimmed — see [trimDanglingToolTurn].
//
// finalizePrompt, when non-empty, enables forced finalization: when a stop
// condition fires (iteration cap, per-run token budget, or shared budget
// pool) the loop injects this user-role message and takes a single final
// tool-less completion so the model can emit its answer instead of dangling
// exploration prose or a silently empty output. RunJSON passes a
// JSON-demanding prompt; the public Run passes "" and therefore never pays
// the extra turn.
//
// responseSchema, when non-nil, is the JSON Schema for the final answer. It is
// attached to every completion in the run (capability-gated; see [complete]),
// so adapters that support structured output can apply grammar-constrained
// decoding. The public Run passes nil; RunJSON passes its schema.
//
// maxEmptyTurnNudges bounds how many times run() will nudge a model that
// produced neither a tool call nor visible text (after stripping reasoning
// <think> blocks) back into the loop before giving up and treating the turn
// as finished. Real reasoning models (MiniMax-M3 observed in production)
// sometimes emit an assistant turn that is ONLY an inline think
// block — stop=end_turn, zero tool calls — which the old code treated as
// "model finished its turn", handing RunJSON unparseable empty text and
// burning its single repair for nothing. The cap keeps a persistently silent
// model from looping forever: after maxEmptyTurnNudges nudges go unanswered,
// run() falls through to today's break.
const maxEmptyTurnNudges = 2

// emptyTurnNudge is appended as a user turn when a completion produced no
// tool call and no visible text, to give the model another chance to either
// call a tool or emit its final answer. See [maxEmptyTurnNudges].
const emptyTurnNudge = "You made no tool call and produced no final answer. Continue: call a tool or emit your final answer now."

func (r *Runner) run(ctx context.Context, seed []llmkit.Message, task, finalizePrompt string, responseSchema json.RawMessage) (*Outcome, error) {
	tr := NewTranscript()
	if r.transcriptDir != "" {
		tr.enableStreaming(r.transcriptPath(tr, task), r.hooks.TranscriptError)
	}

	var messages []llmkit.Message
	if len(seed) > 0 {
		seed = trimDanglingToolTurn(seed)
		messages = make([]llmkit.Message, 0, len(seed)+1)
		messages = append(messages, seed...)
		messages = append(messages, llmkit.TextMessage(llmkit.RoleUser, task))
	} else {
		messages = []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, task)}
	}

	outcome := &Outcome{Transcript: tr}
	// Snapshot the conversation into the Outcome on every return path (clean
	// finish, truncation, or error) so a caller that wants to continue this
	// conversation (RunJSONContinue) always has the latest history available,
	// even from a truncated or erroring run. messages is reassigned (not just
	// mutated) throughout the loop below; the deferred closure reads it by
	// reference at return time, not at defer-registration time.
	defer func() { outcome.Messages = messages }()

	// History-compaction state. toolNameByID lets a tool-result stub name the
	// tool it answered; compactThreshold re-arms upward after each firing so
	// compaction is bounded and never thrashes the prompt cache turn-over-turn.
	// On a continued run the map starts from the seed's assistant tool calls,
	// so a PRIOR run's results also stub with their real tool names instead of
	// the generic fallback (see [compactStub]).
	toolNameByID := map[string]string{}
	for _, m := range messages {
		if m.Role == llmkit.RoleAssistant {
			for _, call := range m.ToolCalls {
				toolNameByID[call.ID] = call.Name
			}
		}
	}
	compactThreshold := r.limits.HistoryTokenBudget

	// emptyTurnNudges counts how many empty/think-only turns have already
	// been nudged this run (see [maxEmptyTurnNudges]).
	emptyTurnNudges := 0

	for {
		// Stop before the next turn if we've hit the iteration cap. The
		// finalizeAndTruncate helper below gives RunJSON its one reserved
		// finalization turn so a near-cap model can still emit its answer; the
		// public Run (finalizePrompt == "") is a no-op and proceeds straight
		// to the truncation mark.
		if r.limits.MaxIterations >= 0 && outcome.Iterations >= r.limits.MaxIterations {
			if err := r.finalizeAndTruncate(ctx, tr, &messages, outcome, finalizePrompt, responseSchema, compactThreshold, toolNameByID, TruncMaxIterations); err != nil {
				return outcome, err
			}
			r.finishTruncated(outcome, TruncMaxIterations)
			break
		}
		// Stop before the next turn if we're already over budget. The budget
		// stop gets the same one reserved finalization turn the iteration cap
		// gets (RunJSON only), so a near-budget agent can emit its answer
		// instead of returning a silently empty result to the caller.
		if r.overBudget(outcome.Usage) {
			if err := r.finalizeAndTruncate(ctx, tr, &messages, outcome, finalizePrompt, responseSchema, compactThreshold, toolNameByID, TruncTokenBudget); err != nil {
				return outcome, err
			}
			r.finishTruncated(outcome, TruncTokenBudget)
			break
		}
		// Consult the shared budget pool (if any) before issuing the next model
		// call, so a run already in flight stops at this turn boundary once the
		// run-spanning ceiling is hit rather than running to completion. This is a
		// read-only check: it does not touch System/Tools/Messages, so request
		// prefix stability is preserved.
		if r.limits.BudgetCheck != nil {
			if err := r.limits.BudgetCheck(); err != nil {
				if !errors.Is(err, ErrBudgetExhausted) {
					// A hook failure that isn't a budget stop is an infrastructure
					// error; surface it rather than misreporting a clean stop.
					tr.closeStream()
					return outcome, fmt.Errorf("agent: budget check: %w", err)
				}
				// Shared pool exhausted: give the model one reserved finalization
				// turn (RunJSON only) so a near-budget agent can still emit its
				// answer before we classify the stop as TruncBudgetPool.
				if err := r.finalizeAndTruncate(ctx, tr, &messages, outcome, finalizePrompt, responseSchema, compactThreshold, toolNameByID, TruncBudgetPool); err != nil {
					return outcome, err
				}
				r.finishTruncated(outcome, TruncBudgetPool)
				break
			}
		}

		if err := ctx.Err(); err != nil {
			tr.closeStream()
			return outcome, err
		}

		// Threshold-triggered, one-shot-per-crossing history compaction. Done at
		// the turn boundary BEFORE the completion so the smaller history is what
		// gets billed this turn. Re-arming the threshold upward after a firing
		// keeps compaction bounded and avoids re-paying a prefix cache miss every
		// subsequent turn (see compactRearmFactor).
		messages, compactThreshold = r.maybeCompact(ctx, messages, compactThreshold, toolNameByID, outcome.Iterations+1)

		resp, err := r.completeOnce(ctx, tr, &messages, outcome, responseSchema, false)
		if err != nil {
			tr.closeStream()
			return outcome, err
		}

		// No tool calls => the model finished its turn.
		if len(resp.ToolCalls) == 0 {
			// StopError/StopRefusal/StopContentFilter mean the model stopped
			// for a provider-specific error reason (refusal, safety filter,
			// recitation). Breaking cleanly here would record refusal prose —
			// or stale FinalText from an earlier turn — as the answer.
			// Surface a typed error instead.
			if resp.StopReason == llmkit.StopError ||
				resp.StopReason == llmkit.StopRefusal ||
				resp.StopReason == llmkit.StopContentFilter {
				tr.closeStream()
				return outcome, &ErrStopReason{StopReason: resp.StopReason, Text: resp.Text, Outcome: outcome}
			}
			// A turn with no tool call and no visible text once reasoning
			// <think> blocks are stripped is not a real answer — it's an
			// empty/think-only turn (MiniMax-M3 observed emitting
			// exactly this in production, sometimes narrating a tool call it
			// never actually made). Nudge the model to continue instead of
			// treating the turn as finished, up to maxEmptyTurnNudges times;
			// the nudge turn goes through the normal loop top (iteration cap,
			// budget checks, compaction all still apply) so it bills and
			// counts like any other turn. This also covers a truncated,
			// unclosed think block: StripThinkBlocks strips it to empty too,
			// and nudging gives the model a chance to re-emit cleanly.
			if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" && emptyTurnNudges < maxEmptyTurnNudges {
				emptyTurnNudges++
				messages = append(messages, llmkit.TextMessage(llmkit.RoleUser, emptyTurnNudge))
				continue
			}
			break
		}

		// Execute the requested tool calls and feed results back. By default
		// the calls run sequentially in the model's order; with
		// WithParallelTools they run concurrently, but the results below are
		// appended in the MODEL's original call order either way, so the
		// wire-visible history is identical.
		executed := r.executeTools(ctx, outcome, resp.ToolCalls)
		for i, call := range resp.ToolCalls {
			if i >= len(executed) {
				// Context cancelled before this call was dispatched; the
				// ctx check below returns with the executed prefix kept.
				break
			}
			tr.recordToolResult(outcome.Iterations, call, executed[i].result, executed[i].isErr)
			toolNameByID[call.ID] = call.Name
			trMsg := llmkit.TextMessage(llmkit.RoleToolResult, executed[i].result)
			trMsg.ToolCallID = call.ID
			trMsg.IsError = executed[i].isErr
			messages = append(messages, trMsg)
		}
		if err := ctx.Err(); err != nil {
			tr.closeStream()
			return outcome, err
		}

		// After executing tools, check the budget again before looping so we
		// truncate promptly rather than issuing one more expensive completion.
		// A budget hit post-tool still gets the one reserved finalization turn
		// (RunJSON only) so a near-budget agent can emit its answer.
		if r.overBudget(outcome.Usage) {
			if err := r.finalizeAndTruncate(ctx, tr, &messages, outcome, finalizePrompt, responseSchema, compactThreshold, toolNameByID, TruncTokenBudget); err != nil {
				return outcome, err
			}
			r.finishTruncated(outcome, TruncTokenBudget)
			break
		}
	}

	tr.closeStream()
	return outcome, nil
}

// trimDanglingToolTurn returns seed with a dangling trailing assistant
// tool-call turn removed, so a conversation continued from a context-cancelled
// run never ends a wire request with an unanswered tool call (Anthropic
// rejects it; other providers misbehave). Only the seed's LAST assistant turn
// is checked — the loop answers every earlier turn's calls before moving on,
// so only a cancelled turn can dangle, and only at the tail. If that turn's
// ToolCalls are not all answered by the RoleToolResult messages after it,
// everything from that assistant turn to the end of the seed is dropped: a
// half-executed turn has no model-visible meaning to salvage. The input is
// never mutated; run() copies the returned subslice into its own conversation
// before appending anything.
func trimDanglingToolTurn(seed []llmkit.Message) []llmkit.Message {
	last := -1
	for i := len(seed) - 1; i >= 0; i-- {
		if seed[i].Role == llmkit.RoleAssistant {
			last = i
			break
		}
	}
	if last < 0 || len(seed[last].ToolCalls) == 0 {
		return seed
	}
	answered := make(map[string]bool, len(seed[last].ToolCalls))
	for _, m := range seed[last+1:] {
		if m.Role == llmkit.RoleToolResult && m.ToolCallID != "" {
			answered[m.ToolCallID] = true
		}
	}
	for _, call := range seed[last].ToolCalls {
		if !answered[call.ID] {
			return seed[:last]
		}
	}
	return seed
}

// finalizeAndTruncate is the single reserved finalization turn used by EVERY
// stop condition the loop can hit (iteration cap, per-run token budget, shared
// budget pool). It is a no-op unless finalizePrompt is non-empty AND the run
// has not already taken a finalization turn this run, so the public Run path
// (finalizePrompt == "") never pays an extra model call. When it does fire it:
//
//   - appends finalizePrompt as a user-role message;
//   - sets outcome.Finalized = true and fires [Hooks.Finalize] with reason
//     (the Trunc* constant of the stop condition that triggered it);
//   - compacts once so the prompt that is sent is the smallest it can be;
//   - takes ONE tool-less completion via completeOnce (which itself handles
//     the StopMaxTokens continuation retry), giving the model a cheap final
//     shot at emitting its answer instead of leaving the caller with a
//     silently empty output.
//
// responseSchema, when non-nil, is attached to the finalization completion so a
// schema-aware adapter applies grammar-constrained decoding on the final turn
// too. The public Run path passes nil so no schema is attached.
//
// Returns done=true when a finalization turn was actually taken (so the caller
// knows Finalized is now true), and a non-nil err only when the underlying
// completion failed — in which case the caller should return early with the
// error. The caller is responsible for finishTruncated(reason) + break: the
// reason is the STOP condition (budget/iteration), not "finalized".
func (r *Runner) finalizeAndTruncate(
	ctx context.Context,
	tr *Transcript,
	messages *[]llmkit.Message,
	outcome *Outcome,
	finalizePrompt string,
	responseSchema json.RawMessage,
	compactThreshold int64,
	toolNameByID map[string]string,
	reason string,
) error {
	if finalizePrompt == "" || outcome.Finalized {
		return nil
	}
	*messages = append(*messages, llmkit.TextMessage(llmkit.RoleUser, finalizePrompt))
	outcome.Finalized = true
	if r.hooks.Finalize != nil {
		r.hooks.Finalize(ctx, reason)
	}
	// Compact before the finalization turn: it is often the largest history of
	// the run, and the model needs only its own reasoning chain (preserved) to
	// emit the answer, not every earlier file dump. The re-armed threshold is
	// discarded: this is the run's final turn, so no later compaction can fire.
	*messages, _ = r.maybeCompact(ctx, *messages, compactThreshold, toolNameByID, outcome.Iterations+1)
	if _, cerr := r.completeOnce(ctx, tr, messages, outcome, responseSchema, true); cerr != nil {
		tr.closeStream()
		return cerr
	}
	return nil
}

// maybeCompact applies threshold-triggered history compaction. When compaction
// is enabled (threshold > 0) and the estimated history size exceeds threshold,
// it prunes tool-result content older than the most recent few turns to short
// stubs and returns the compacted history together with a re-armed (higher)
// threshold so the next firing only happens once history has grown materially
// again. When disabled or under threshold — or when there is nothing left to
// prune — it returns the messages and threshold unchanged, so a normal run pays
// no allocation and the append-only prefix (and its cache) is preserved.
//
// step is the completion that will consume the (possibly) compacted history;
// it is reported on [Hooks.Compaction], which fires only when this call
// actually pruned — a threshold crossing with nothing to reclaim is silent.
func (r *Runner) maybeCompact(ctx context.Context, messages []llmkit.Message, threshold int64, toolNameByID map[string]string, step int) ([]llmkit.Message, int64) {
	if threshold <= 0 {
		return messages, threshold
	}
	if estimateTokens(messages) <= threshold {
		return messages, threshold
	}
	compacted, pruned := compactHistory(messages, compactRecentToolResults, toolNameByID)
	if pruned == 0 {
		// Over threshold but nothing prunable yet: either history is dominated by
		// assistant reasoning, or every prunable result is already a stub, or there
		// simply aren't more than compactRecentToolResults tool results so far. Leave the
		// threshold UNCHANGED so the check fires again next turn once a result ages out of
		// the recent window — re-arming here would starve real compaction by ratcheting
		// the threshold above the history before anything was ever reclaimed.
		return messages, threshold
	}
	if r.hooks.Compaction != nil {
		r.hooks.Compaction(ctx, CompactionEvent{
			Step:         step,
			BeforeTokens: estimateTokens(messages),
			AfterTokens:  estimateTokens(compacted),
			Pruned:       pruned,
		})
	}
	// Real pruning happened (one prefix cache miss paid). Re-arm upward so the
	// next firing only comes after history has grown materially again, bounding
	// total firings and avoiding turn-over-turn cache thrash.
	return compacted, threshold * compactRearmFactor
}

// repair issues a SINGLE tools-less, schema-bearing completion against the
// repair prompt. It replaces the previous "fresh tool loop" repair path with
// the constrained shape: a single completion where adapters that support
// structured output apply grammar-constrained decoding natively, so the
// answer is shape-correct on the wire. Tools are dropped so Google and
// Anthropic (which refuse to combine tool use with native structured output)
// also get the schema honored.
//
// responseSchema, when non-nil, is attached capability-gated; when the
// adapter's StructuredOutput capability is off, the schema is dropped
// silently (per llmkit.Request docs) and the prompt-embedded schema instruction
// is the only enforcement — same contract as the main run path.
//
// The repair uses a fresh outcome but shares the caller's transcript so the
// assistant turn is recorded there for parity with the main run path. No tool
// loop runs; [Hooks.Repair] fires at entry. The single repair completion is
// issued via completeOnce, so when it itself stops at the output token cap the
// ONE max-tokens continuation completion is paid on top — the pass is bounded
// to at most two schema-bearing, tool-less completions.
func (r *Runner) repair(ctx context.Context, tr *Transcript, prompt string, responseSchema json.RawMessage) (*Outcome, error) {
	if r.hooks.Repair != nil {
		r.hooks.Repair(ctx)
	}
	outcome := &Outcome{Transcript: tr}
	messages := []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, prompt)}
	if _, err := r.completeOnce(ctx, tr, &messages, outcome, responseSchema, true); err != nil {
		return outcome, err
	}
	return outcome, nil
}

// completeOnce issues one model completion against the current conversation,
// records it, folds its usage into the outcome, and appends the assistant turn
// to *messages. When final is true the request carries no tools (the
// finalization turn forbids further investigation).
//
// responseSchema, when non-nil and the client's StructuredOutput capability is
// on, is attached to the request so adapters can apply native
// schema-constrained output. When the capability is off, the schema is
// dropped silently — the prompt-embedded schema instruction is the only
// enforcement, matching the no-cap passthrough path's contract.
//
// If the completion stops at the token cap (StopMaxTokens) it makes ONE
// continuation completion — appending a short user nudge — so a JSON answer cut
// off mid-object has a chance to be completed rather than failing to parse. The
// outcome's LastStopReason reflects the final completion served here.
func (r *Runner) completeOnce(ctx context.Context, tr *Transcript, messages *[]llmkit.Message, outcome *Outcome, responseSchema json.RawMessage, final bool) (llmkit.Response, error) {
	resp, err := r.complete(ctx, tr, *messages, outcome, responseSchema, final)
	if err != nil {
		return llmkit.Response{}, err
	}
	assistantMsg := assistantMessage(resp)
	assistantMsg.ToolCalls = resp.ToolCalls
	*messages = append(*messages, assistantMsg)

	// One continuation retry when output was truncated mid-generation: ask the
	// model to continue and emit ONLY the remaining answer, then concatenate.
	// Guarded so it fires at most once per completeOnce call.
	if resp.StopReason == llmkit.StopMaxTokens && len(resp.ToolCalls) == 0 {
		*messages = append(*messages, llmkit.TextMessage(llmkit.RoleUser,
			"Your previous message was cut off at the output token limit. Continue from exactly where you stopped and output ONLY the remaining text needed to complete the answer — no preamble, no repetition."))
		cont, cerr := r.complete(ctx, tr, *messages, outcome, responseSchema, final)
		if cerr != nil {
			return llmkit.Response{}, cerr
		}
		contMsg := assistantMessage(cont)
		contMsg.ToolCalls = cont.ToolCalls
		*messages = append(*messages, contMsg)
		// Stitch the two halves so the caller (and FinalText) sees one answer.
		// Models frequently ignore "continue from where you stopped" and instead
		// restart, repeating some head of the first half. A naive resp.Text+cont.Text
		// would then double that prefix and corrupt the JSON. Trim the longest
		// suffix of resp.Text that the continuation re-emits as its prefix before
		// concatenating, so a clean cut and a repeated-prefix restart both stitch
		// into one well-formed answer.
		joined := stitchContinuation(resp.Text, cont.Text)
		outcome.FinalText = joined
		resp.Text = joined
		resp.ToolCalls = cont.ToolCalls
		resp.StopReason = cont.StopReason
		resp.Blocks = stitchBlocks(resp.Blocks, cont.Blocks, joined)
	}
	return resp, nil
}

// maxStitchOverlap bounds how far back stitchContinuation scans for a repeated
// prefix. The overlap is the chunk a restarting model re-emits; a few KB covers
// realistic restarts while keeping the scan O(n) and immune to a pathological
// continuation that happens to share a huge prefix with the head.
const maxStitchOverlap = 4096

// stitchContinuation joins a truncated first half with its continuation, undoing
// the common case where the model restarts and repeats some head of head as the
// prefix of cont. It finds the LONGEST suffix of head (bounded by
// maxStitchOverlap) that is a prefix of cont and drops that overlap from cont
// before concatenating. With no overlap it degrades to head+cont (the clean-cut
// case). The scan is longest-first so a model that repeats more text wins over a
// coincidental short match.
func stitchContinuation(head, cont string) string {
	max := len(head)
	if max > maxStitchOverlap {
		max = maxStitchOverlap
	}
	if max > len(cont) {
		max = len(cont)
	}
	for n := max; n > 0; n-- {
		if head[len(head)-n:] == cont[:n] {
			return head + cont[n:]
		}
	}
	return head + cont
}

// assistantMessage builds the assistant history message for a completion. A
// response that surfaced content blocks (thinking, multi-part text) is
// recorded VERBATIM — including thinking blocks, which providers bind to the
// exact bytes they issued (signed on Anthropic): rebuilding the message from
// resp.Text alone would drop them and break every later turn of a
// thinking+tools loop. A response with no blocks degrades to the single-text
// form.
func assistantMessage(resp llmkit.Response) llmkit.Message {
	if len(resp.Blocks) > 0 {
		return llmkit.Message{Role: llmkit.RoleAssistant, Content: resp.Blocks}
	}
	return llmkit.TextMessage(llmkit.RoleAssistant, resp.Text)
}

// stitchBlocks shapes the Block list of the stitched llmkit.Response that
// completeOnce returns to ITS caller. It does not touch the conversation
// history: completeOnce already appended each half verbatim via
// [assistantMessage], so both halves' thinking blocks are carried in history
// in order exactly as the provider issued them. Only text blocks are
// stitched (see [stitchContinuation]); thinking blocks from BOTH halves are
// carried through untouched — never split, reordered, or re-emitted
// differently — because the next turn must return them to the provider
// exactly as they arrived. Assistant turns carry no other block kinds.
func stitchBlocks(head, cont []llmkit.Block, joinedText string) []llmkit.Block {
	var out []llmkit.Block
	for _, b := range head {
		if b.Kind == llmkit.BlockThinking {
			out = append(out, b)
		}
	}
	for _, b := range cont {
		if b.Kind == llmkit.BlockThinking {
			out = append(out, b)
		}
	}
	return append(out, llmkit.Block{Kind: llmkit.BlockText, Text: joinedText})
}

// complete issues a single completion, records it, and accumulates its usage and
// stop reason onto the outcome. It does NOT mutate the conversation; callers
// append the assistant turn so they control conversation shape (e.g. the
// continuation retry).
//
// responseSchema, when non-nil AND the client's StructuredOutput capability is
// on, is attached to the request so adapters that support native structured
// output can apply grammar-constrained decoding. When the capability is off
// (a conservative openai-compatible endpoint, etc.), the schema is silently
// dropped on the wire (per [llmkit.Request.ResponseSchema] docs) and only the
// prompt-embedded schema instruction is in effect. This is the agent-layer
// gate: the no-cap passthrough path sends NO schema, matching today's
// behavior, while the with-cap path gets a hard native shape guarantee.
func (r *Runner) complete(ctx context.Context, tr *Transcript, messages []llmkit.Message, outcome *Outcome, responseSchema json.RawMessage, final bool) (llmkit.Response, error) {
	req := llmkit.Request{
		System:    r.systemPrompt,
		Messages:  messages,
		Tools:     r.tools.defs,
		MaxTokens: r.maxTokens,
	}
	// The finalization turn forbids further investigation: drop the tools so the
	// model can only answer.
	if final {
		req.Tools = nil
	}
	// Native schema-constrained output is capability-gated at the agent layer:
	// when the client cannot honor a schema, do not put one on the wire. The
	// schema name is left empty so each adapter picks its own default
	// ("response" on OpenAI, "emit_answer" on Anthropic).
	if len(responseSchema) > 0 && r.client.Capabilities().StructuredOutput {
		req.ResponseSchema = responseSchema
	}
	// This is the single fire point for [Hooks.BeforeCompletion] and
	// [Hooks.AfterCompletion]: every client.Complete in the loop (main turn,
	// max-tokens continuation, forced finalization, repair) goes through here.
	// step is the 1-based transcript step this completion is recorded under
	// (outcome.Iterations+1 at fire time): the SAME number every other hook
	// reports for this turn — ToolEvent.Step, CompactionEvent.Step — and the
	// Event.Step of the request/assistant transcript events below, so
	// consumers can join all hook families on Step. Captured before Complete
	// because outcome.Iterations is incremented only after the call returns,
	// keeping the hook pair's step identical.
	step := outcome.Iterations + 1
	if r.hooks.BeforeCompletion != nil {
		r.hooks.BeforeCompletion(ctx, step, &req)
	}
	tr.recordRequest(step, messages)

	resp, err := r.client.Complete(ctx, req)
	if r.hooks.AfterCompletion != nil {
		var respPtr *llmkit.Response
		if err == nil {
			respPtr = &resp
		}
		r.hooks.AfterCompletion(ctx, step, &req, respPtr, err)
	}
	if err != nil {
		return llmkit.Response{}, fmt.Errorf("agent: completion failed at iteration %d: %w", step, err)
	}

	outcome.Iterations++
	outcome.Usage.InputTokens += resp.Usage.InputTokens
	outcome.Usage.OutputTokens += resp.Usage.OutputTokens
	outcome.Usage.CacheReadInputTokens += resp.Usage.CacheReadInputTokens
	outcome.Usage.CacheCreationInputTokens += resp.Usage.CacheCreationInputTokens
	outcome.LastStopReason = resp.StopReason
	tr.recordAssistant(outcome.Iterations, resp)

	if resp.Text != "" {
		outcome.FinalText = resp.Text
		outcome.FinalTextSet = true
	} else {
		outcome.FinalTextSet = false
	}
	return resp, nil
}

// runTool dispatches one tool call. A missing tool or a Run error is returned to
// the model as an "ERROR:"-prefixed result (isErr=true) rather than aborting the
// loop. Context cancellation surfaced by the tool is still rendered as a tool
// error here; the loop's own ctx checks handle real cancellation.
//
// With WithToolTimeout set, the call runs under a derived deadline: on expiry
// the model receives "ERROR: tool <name> timed out after <d>" and the loop
// continues — the run's own context is unaffected. A timeout is not a
// *ToolHealthError and does not fire [Hooks.ToolHealth].
//
// [Hooks.ToolStart] fires immediately before Tool.Run and [Hooks.ToolEnd]
// immediately after (Result, IsError, Duration set); a missing tool never
// runs, so neither fires for it. Hooks fire from the goroutine executing the
// call — concurrently under WithParallelTools.
func (r *Runner) runTool(ctx context.Context, call llmkit.ToolCall, step int) (result string, isErr bool) {
	tool, ok := r.tools.lookup(call.Name)
	if !ok {
		return toolError(fmt.Errorf("unknown tool %q", call.Name)), true
	}
	runCtx := ctx
	if r.toolTimeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, r.toolTimeout)
		defer cancel()
	}
	if r.hooks.ToolStart != nil {
		r.hooks.ToolStart(ctx, ToolEvent{Step: step, Call: call})
	}
	start := time.Now()
	out, err := tool.Run(runCtx, call.Arguments)
	duration := time.Since(start)
	if err != nil {
		// Deadline expiry: report the timeout as the tool result, leaving the
		// run's own context untouched. The parent-ctx guard keeps a genuine
		// run cancellation from being misreported as a tool timeout.
		if r.toolTimeout > 0 && ctx.Err() == nil && runCtx.Err() == context.DeadlineExceeded {
			timeoutResult := fmt.Sprintf("ERROR: tool %s timed out after %s", call.Name, r.toolTimeout)
			if r.hooks.ToolEnd != nil {
				r.hooks.ToolEnd(ctx, ToolEvent{Step: step, Call: call, Result: timeoutResult, IsError: true, Duration: duration})
			}
			return timeoutResult, true
		}
		// Record a tool-health signal only for a genuine *ToolHealthError AND
		// only when ctx is not already cancelled: a failure caused by run
		// teardown/cancellation must never be counted as a tool-health problem.
		var he *ToolHealthError
		if ctx.Err() == nil && errors.As(err, &he) && r.hooks.ToolHealth != nil {
			r.hooks.ToolHealth(ctx, call.Name, he)
		}
		result = toolError(err)
		if r.hooks.ToolEnd != nil {
			r.hooks.ToolEnd(ctx, ToolEvent{Step: step, Call: call, Result: result, IsError: true, Duration: duration})
		}
		return result, true
	}
	if r.hooks.ToolEnd != nil {
		r.hooks.ToolEnd(ctx, ToolEvent{Step: step, Call: call, Result: out, IsError: false, Duration: duration})
	}
	return out, false
}

// toolResult is one executed tool call's outcome, as returned by [Runner.executeTools].
type toolResult struct {
	result string
	isErr  bool
}

// executeTools runs calls and returns one result per executed call, in the
// model's original order. Sequential mode (the default) runs calls one at a
// time and stops before dispatching the next call once ctx is cancelled —
// the returned slice then holds only the already-executed results. Parallel
// mode (WithParallelTools) runs each call on its own goroutine (bounded by
// len(calls)) and waits for all of them: per-call failures are isolated (a
// tool error or panic never fails its siblings — a panic is recovered in the
// call's goroutine and rendered as that call's error result), and the
// full-length result slice is always returned. Neither mode mutates the
// conversation or the transcript; the caller appends the results in call
// order after executeTools returns.
func (r *Runner) executeTools(ctx context.Context, outcome *Outcome, calls []llmkit.ToolCall) []toolResult {
	results := make([]toolResult, len(calls))
	if !r.parallelTools || len(calls) < 2 {
		for i, call := range calls {
			if err := ctx.Err(); err != nil {
				return results[:i]
			}
			results[i].result, results[i].isErr = r.runTool(ctx, call, outcome.Iterations)
		}
		return results
	}
	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Add(1)
		go func(i int, call llmkit.ToolCall) {
			// A panicking Tool.Run must not abort the process from this
			// goroutine (a panic here escapes any caller recover and kills
			// every concurrent Run): render it as THIS call's error result so
			// isolation stays literal. Sequential dispatch propagates the
			// panic to the Runner's caller, unchanged.
			// Recover FIRST (this defer runs before wg.Done below) so
			// wg.Wait cannot observe the result slot before the panic
			// rendering is published.
			defer wg.Done()
			defer func() {
				if v := recover(); v != nil {
					results[i].result = fmt.Sprintf("ERROR: tool %s panicked: %v", call.Name, v)
					results[i].isErr = true
				}
			}()
			results[i].result, results[i].isErr = r.runTool(ctx, call, outcome.Iterations)
		}(i, call)
	}
	wg.Wait()
	return results
}

// overBudget reports whether cumulative usage has exceeded the token budget. A
// negative budget means unlimited. Cache reads are discounted by
// CacheReadWeight (resolved to 1.0 by Limits.resolve() when unset) so a
// cache-heavy run is bounded by its real cost, not raw prompt size.
func (r *Runner) overBudget(u llmkit.Usage) bool {
	if r.limits.TokenBudget < 0 {
		return false
	}
	return u.ChargeableTokens(r.limits.CacheReadWeight) > r.limits.TokenBudget
}

// finishTruncated marks the outcome as a clean partial result. reason MUST be
// one of the Trunc* constants and non-empty; passing an empty reason panics to
// surface a programming error at the call site rather than silently producing
// an Outcome that violates the Truncated→TruncationReason invariant.
func (r *Runner) finishTruncated(o *Outcome, reason string) {
	if reason == "" {
		panic("agent: finishTruncated called with empty reason — callers must pass a Trunc* constant")
	}
	o.Truncated = true
	o.TruncationReason = reason
}

// transcriptPath computes the JSONL path for a run's transcript under
// r.transcriptDir, named "<timestamp>-<task-slug>.jsonl", or
// "<timestamp>-<r.transcriptKey>-<task-slug>.jsonl" when WithTranscriptKey was
// set (see its doc for why the key must be exact enough for a caller to
// recover this file later by filename match). Streaming (see
// Transcript.enableStreaming) opens this path lazily on the first recorded
// event; a run that never records anything never creates the file or its
// parent directory.
func (r *Runner) transcriptPath(tr *Transcript, task string) string {
	ts := tr.now().UTC().Format("20060102T150405.000Z")
	name := ts
	if r.transcriptKey != "" {
		name += "-" + r.transcriptKey
	}
	name += "-" + slug(task) + ".jsonl"
	return filepath.Join(r.transcriptDir, name)
}

// slugRE keeps slugs filesystem-safe: lowercase alphanumerics and dashes.
var slugRE = regexp.MustCompile(`[^a-z0-9]+`)

// slug derives a short, filesystem-safe label from a task string.
func slug(task string) string {
	s := strings.ToLower(task)
	s = slugRE.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-")
	if s == "" {
		return "run"
	}
	if len(s) > 48 {
		s = strings.Trim(s[:48], "-")
		if s == "" {
			s = "run"
		}
	}
	return s
}

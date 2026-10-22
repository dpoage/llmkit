package agent

import (
	"context"

	"github.com/dpoage/llmkit"
)

// Compactor makes a conversation history smaller. The Runner calls it when
// the estimated history crosses [Limits.HistoryTokenBudget] and when the
// provider refuses a completion with [llmkit.ErrContextTooLong]; the Runner
// decides when, the Compactor decides how.
//
// A Compactor must not modify in.Messages, in.Sent, or anything they share;
// it returns a new slice, or no messages or in.Messages itself to report no
// change. When it replaces a prefix with a summary, the summary is a user
// message whose text starts "[conversation summary]" and the prefix ends at a
// cut point, so callers can identify it. A Runner is safe for concurrent Run
// calls, so a Compactor shared across runs is called concurrently.
type Compactor interface {
	Compact(ctx context.Context, in CompactInput) (CompactOutput, error)
}

// CompactInput is one compaction request.
type CompactInput struct {
	// Messages is the conversation history to compact.
	Messages []llmkit.Message
	// Sent is the request copy of Messages, index-aligned: what the client
	// is sent, with tool-result media the client cannot carry already
	// replaced by text markers. It may share its backing array with Messages.
	Sent []llmkit.Message
	// Reason is "" for a threshold compaction and
	// [llmkit.CompactionReasonOverflow] for overflow recovery.
	Reason string
}

// CompactOutput is a compaction result.
type CompactOutput struct {
	// Messages is the compacted history; it becomes the run's history. Empty
	// Messages, or CompactInput.Messages itself, leaves the history unchanged.
	Messages []llmkit.Message
	// Pruned is the number of messages the compaction replaced or removed,
	// reported on the Compaction event.
	Pruned int
	// Summary is the summary text, without its header, that replaced the
	// pruned prefix; the Compaction event reports it.
	Summary string
	// Usage is the model spend of the compaction. The Runner adds it to
	// Outcome.Usage and charges the budget pool, even when Compact also
	// returns an error or leaves the history unchanged.
	Usage llmkit.Usage
	// calls are the model calls the Runner records as Completion events.
	calls []compactCall
}

// CompactorFunc adapts a function to [Compactor].
type CompactorFunc func(ctx context.Context, in CompactInput) (CompactOutput, error)

// Compact calls f(ctx, in).
func (f CompactorFunc) Compact(ctx context.Context, in CompactInput) (CompactOutput, error) {
	return f(ctx, in)
}

// WithCompactor installs c as the Runner's [Compactor]. Setting one enables
// compaction: overflow recovery runs even with a zero
// [Limits.HistoryTokenBudget], which alone gates threshold compaction. Without
// it, or with a nil c, the Runner stubs old tool results as described on
// [Limits.HistoryTokenBudget].
func WithCompactor(c Compactor) Option {
	return func(r *Runner) { r.compactor = c }
}

// stubCompactor is the default Compactor: it stubs tool results older than
// the most recent [compactRecentToolResults].
type stubCompactor struct{}

func (stubCompactor) Compact(_ context.Context, in CompactInput) (CompactOutput, error) {
	names := map[string]string{}
	for _, m := range in.Messages {
		if m.Role == llmkit.RoleAssistant {
			for _, call := range m.ToolCalls {
				names[call.ID] = call.Name
			}
		}
	}
	msgs, pruned := compactHistory(in.Messages, in.Sent, compactRecentToolResults, names)
	return CompactOutput{Messages: msgs, Pruned: pruned}, nil
}

// compactionEnabled reports whether overflow recovery runs.
func (r *Runner) compactionEnabled() bool {
	return r.limits.HistoryTokenBudget > 0 || r.limits.ContextReserve > 0 || r.compactor != nil
}

// compact runs the Compactor, or the default stubber when stubOnly, then
// records the model calls it made and charges their usage to outcome and the
// budget pool, whether or not it erred.
func (r *Runner) compact(ctx context.Context, em runEmitter, outcome *Outcome, in CompactInput, stubOnly bool) (CompactOutput, error) {
	var c Compactor = stubCompactor{}
	if r.compactor != nil && !stubOnly {
		c = r.compactor
	}
	out, err := c.Compact(ctx, in)
	for _, call := range out.calls {
		r.emitCompactionCompletion(ctx, em, call)
	}
	outcome.Usage = outcome.Usage.Add(out.Usage)
	r.budgetPool.Add(out.Usage.ChargeableTokens(r.limits.CacheReadWeight))
	return out, err
}

// compactionUnchanged reports whether out leaves in as it was: no messages,
// or in itself.
func compactionUnchanged(out, in []llmkit.Message) bool {
	return len(out) == 0 || len(out) == len(in) && &out[0] == &in[0]
}

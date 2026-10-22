package decide

import "context"

// Decider answers a set of questions about a state. A backend's errors
// wrap a root llmkit sentinel (llmkit.Err*) or chain ctx.Err(); a
// successful Response satisfies [Check] against qs.
//
// A Decider that decide itself provides ([Observe]'s wrapper, [Replay])
// returns only the wrapped Decider's errors and, for Replay, the two
// non-root sentinels [ErrReplayDiverged] and [ErrReplayedFailure].
type Decider interface {
	Ask(ctx context.Context, state any, qs Questions) (Response, error)
}

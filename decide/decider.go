package decide

import "context"

// Decider answers a set of questions about a state. A backend's errors
// wrap a root llmkit sentinel (llmkit.Err*) or chain ctx.Err(); a
// successful Response satisfies [Check] against qs.
//
// A Decider that decide provides returns only these errors: [Observe]'s
// wrapper, the wrapped Decider's; [Replay], a [Normalize] refusal and the
// non-root sentinels [ErrReplayDiverged] and [ErrReplayedFailure].
type Decider interface {
	Ask(ctx context.Context, state any, qs Questions) (Response, error)
}

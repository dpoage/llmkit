package llmkit

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// RunID identifies one logical run: a single Runner.Run invocation, one
// [Client] completion sequence, one decision, or any other unit a caller
// chooses to correlate. Events emitted inside a run carry the run's id in
// Event.RunID; a continued run receives its predecessor's id as
// Event.ParentRunID. The zero value ("") means "outside any run" and is
// never serialized (the field is omitempty).
type RunID string

// SpanID identifies one logical completion within a run. The Completion
// emitter (the agent Runner or the [Observe] decorator) mints a fresh
// SpanID per logical completion and places it in the context it passes to
// the client; the retry stage's Attempt events read the same context and
// join their Completion on Event.SpanID — not on time windows — and a
// nested completion (a tool calling the model) gets its own span. Like
// RunID, the zero value means "no span" and is never serialized.
type SpanID string

// runIDContextKey, spanIDContextKey, and stepContextKey are unexported
// context keys; distinct types prevent cross-package collisions.
type (
	runIDContextKey  struct{}
	spanIDContextKey struct{}
	stepContextKey   struct{}
)

// WithRun returns a context carrying the run id. Downstream emitters
// (tool implementations, policies, decorators) read it back with
// [RunFromContext] and stamp it on their [Event] values, so one id
// correlates every boundary of a run. Passing the empty RunID returns ctx
// unchanged: an unset id never erases an enclosing run.
func WithRun(ctx context.Context, id RunID) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, runIDContextKey{}, id)
}

// RunFromContext returns the run id stored by [WithRun], or the empty
// RunID when the context carries none.
func RunFromContext(ctx context.Context) RunID {
	id, _ := ctx.Value(runIDContextKey{}).(RunID)
	return id
}

// WithSpan returns a context carrying the completion span id. Passing the
// empty SpanID returns ctx unchanged: an unset span never erases an
// enclosing one.
func WithSpan(ctx context.Context, id SpanID) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, spanIDContextKey{}, id)
}

// SpanFromContext returns the span id stored by [WithSpan], or the empty
// SpanID when the context carries none.
func SpanFromContext(ctx context.Context) SpanID {
	id, _ := ctx.Value(spanIDContextKey{}).(SpanID)
	return id
}

// WithStep returns a context carrying the 1-based model turn number step.
// The agent Runner places it in the context it passes to the client,
// tools, policies, and hooks for each turn, so events decorators emit
// inside a Runner turn — retry Attempt events, a decision observed inside
// a ToolPolicy, sandbox executions and embeddings from a tool — carry
// the turn as Event.Step ([NewEvent]) and join the Runner's own events.
//
// Passing step <= 0 returns ctx unchanged: an unset step never erases an
// enclosing one, mirroring [WithRun]'s empty-id rule.
func WithStep(ctx context.Context, step int) context.Context {
	if step <= 0 {
		return ctx
	}
	return context.WithValue(ctx, stepContextKey{}, step)
}

// StepFromContext returns the turn number stored by [WithStep], or 0 when
// the context carries none.
func StepFromContext(ctx context.Context) int {
	step, _ := ctx.Value(stepContextKey{}).(int)
	return step
}

// newID mints "<unix-millis>-<hex>" — the current time as 13 zero-padded
// decimal digits plus 16 lowercase hex characters (8 bytes from
// crypto/rand, which never returns an error and always fills b entirely).
// Ids sort lexically in mint order so a sink can range over id-prefixed
// records without a separate timestamp column; two ids collide only if
// minted in the same millisecond with an identical 64-bit random suffix.
func newID() string {
	var b [8]byte
	rand.Read(b[:])
	return fmt.Sprintf("%013d-%s", time.Now().UnixMilli(), hex.EncodeToString(b[:]))
}

// NewRunID mints a fresh run id: "<unix-millis>-<hex>", lexically sortable
// by mint time.
func NewRunID() RunID { return RunID(newID()) }

// NewSpanID mints a fresh span id — same format as [NewRunID] (see
// [SpanID] for how spans join events).
func NewSpanID() SpanID { return SpanID(newID()) }

// EventSchemaVersion is stamped on every [Event] this build emits
// (Event.SchemaVersion). Bump it when an Event field changes meaning in a
// way a durable sink must distinguish; additively appended fields do not
// require a bump.
const EventSchemaVersion = 1

// NewEvent returns an Event of the given kind with the shared header
// fields filled in: Kind, Time (now, UTC, monotonic clock reading stripped
// so a decoded event is DeepEqual to the emitted one), RunID and SpanID
// (from ctx via [RunFromContext] and [SpanFromContext]; empty outside a
// run), Step (from [StepFromContext]; 0 outside a Runner turn), and
// SchemaVersion ([EventSchemaVersion]). The caller fills exactly one
// payload pointer and passes the result to an [Observer]. Duration and
// the per-kind payload fields are the emitter's to set after construction;
// an emitter that assigns Event.Step explicitly overrides the context's
// value.
func NewEvent(ctx context.Context, kind EventKind) Event {
	return Event{
		Kind:          kind,
		RunID:         RunFromContext(ctx),
		SpanID:        SpanFromContext(ctx),
		Step:          StepFromContext(ctx),
		Time:          time.Now().Round(0).UTC(),
		SchemaVersion: EventSchemaVersion,
	}
}

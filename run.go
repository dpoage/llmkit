package llmkit

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// RunID identifies one logical run — a single Runner.Run invocation, one
// [Client] completion sequence, one decision, or any other unit a caller
// chooses to correlate. Events emitted inside a run carry the run's id in
// Event.RunID; a continued run receives its predecessor's id as
// Event.ParentRunID. The zero value ("") means "outside any run" and is
// never serialized (the field is omitempty).
type RunID string

// runIDContextKey is the unexported context key for RunID. Using a distinct
// unexported type keeps other packages from colliding with it.
type runIDContextKey struct{}

// WithRun returns a context carrying the run id id. Downstream emitters —
// tool implementations, policies, decorators — read it back with
// [RunFromContext] and stamp it on their [Event] values, so one id
// correlates every boundary of a run. Passing the empty RunID stores nothing
// observable: [RunFromContext] returns "" for it exactly as for an absent
// value.
func WithRun(ctx context.Context, id RunID) context.Context {
	return context.WithValue(ctx, runIDContextKey{}, id)
}

// RunFromContext returns the run id stored by [WithRun], or the empty RunID
// when the context carries none.
func RunFromContext(ctx context.Context) RunID {
	id, _ := ctx.Value(runIDContextKey{}).(RunID)
	return id
}

// NewRunID mints a fresh run id: "<unix-millis>-<hex>", where unix-millis is
// the current time as 13 zero-padded decimal digits and hex is 16 lowercase
// hex characters (8 bytes from crypto/rand). Ids therefore sort lexically in
// mint order — a sink can range over id-prefixed records without a separate
// timestamp column — and two ids collide only if minted in the same
// millisecond with an identical 64-bit random suffix.
//
// NewRunID panics if the system entropy source fails. That is a broken
// machine, not a caller-handleable condition.
func NewRunID() RunID {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("llmkit.NewRunID: crypto/rand failed: %v", err))
	}
	return RunID(fmt.Sprintf("%013d-%s", time.Now().UnixMilli(), hex.EncodeToString(b[:])))
}

// EventSchemaVersion is the schema version stamped on every [Event] this
// build emits (Event.SchemaVersion). Bump it when an Event field changes
// meaning in a way a durable sink must distinguish; additively appended
// fields do not require a bump.
const EventSchemaVersion = 1

// NewEvent returns an Event of the given kind with the shared header fields
// filled in: Kind, Time (now — emitters stamp, sinks never do), RunID (read
// from ctx via [RunFromContext]; empty outside a run), and SchemaVersion
// ([EventSchemaVersion]). The caller fills exactly one payload pointer and
// passes the result to an [Observer]. Duration and the per-kind payload
// fields are the emitter's to set after construction.
func NewEvent(ctx context.Context, kind EventKind) Event {
	return Event{
		Kind:          kind,
		RunID:         RunFromContext(ctx),
		Time:          time.Now(),
		SchemaVersion: EventSchemaVersion,
	}
}

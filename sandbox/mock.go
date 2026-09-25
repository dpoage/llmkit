package sandbox

import (
	"context"
	"os"
	"sync"
)

// Call is a single recorded invocation of Mock.Exec, capturing the Spec
// it was called with so tests can assert on what the caller requested.
type Call struct {
	Spec Spec
}

// MockResponse scripts the outcome of one Mock.Exec call.
type MockResponse struct {
	// Result is returned to the caller (when Err is nil).
	Result Result
	// Err, when non-nil, is returned instead of Result, simulating an
	// infrastructure failure.
	Err error
}

// Mock is a scriptable, in-memory Sandbox implementation for testing
// callers of a Sandbox without a real container runtime or bwrap. It is
// part of the package's public contract.
//
// Responses are consumed in order via EnqueueResponse; once exhausted, the
// configured DefaultResponse is returned. A ResponseFunc, when set, takes
// precedence and is consulted for every call (enabling spec-dependent
// behavior). All calls are recorded. Mock is safe for concurrent use.
//
// Spec.Workspace is honored NEUTRALLY: a caller-supplied Workspace is
// recorded on the Call like every other Spec field but otherwise ignored —
// it never gates or alters which scripted response comes back. Tests that
// need to distinguish an iteration-workspace
// call (e.g. workspace exec) from a clean-room call (e.g. execute()) do so via a
// ResponseFunc keyed on spec.Workspace == "" themselves.
type Mock struct {
	mu sync.Mutex

	// ResponseFunc, when set, is invoked for every Exec call and its return
	// values are used verbatim. It takes precedence over the queue and the
	// default. The call number n is zero-based; ctx is the caller's Exec
	// context. It runs while the Mock's lock is held, so calls are
	// serialized and a ResponseFunc must not call back into the Mock
	// (CallCount, Calls, EnqueueResponse, ...) or block on anything that
	// does — either deadlocks.
	ResponseFunc func(ctx context.Context, n int, spec Spec) (Result, error)

	// DefaultResponse is returned when the queue is empty and ResponseFunc is
	// nil.
	DefaultResponse MockResponse

	queue []MockResponse
	calls []Call
	// materialized holds the directories MaterializeWorkspace created, in
	// call order.
	materialized []string
}

// NewMock constructs a Mock with the given default response.
func NewMock(def MockResponse) *Mock {
	return &Mock{DefaultResponse: def}
}

// EnqueueResponse appends a scripted response to the FIFO queue. Returns the
// Mock for chaining.
func (m *Mock) EnqueueResponse(r MockResponse) *Mock {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queue = append(m.queue, r)
	return m
}

// Exec implements Sandbox: it refuses a Spec that is malformed for every
// backend (like every backend — see validateSpec), records the call, and
// returns the next scripted response (or the default). A caller whose
// context is already done gets the shared cancelled error back: the call is
// still recorded (the caller DID attempt it), but nothing is consumed — the
// queue keeps its next response and ResponseFunc is not consulted.
func (m *Mock) Exec(ctx context.Context, spec Spec) (Result, error) {
	if err := validateSpec(backendMock, spec); err != nil {
		return Result{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	n := len(m.calls)
	m.calls = append(m.calls, Call{Spec: spec})

	if ctx.Err() != nil {
		return Result{}, cancelledRunErr(ctx)
	}
	if m.ResponseFunc != nil {
		return m.ResponseFunc(ctx, n, spec)
	}
	if len(m.queue) > 0 {
		r := m.queue[0]
		m.queue = m.queue[1:]
		return r.Result, r.Err
	}
	return m.DefaultResponse.Result, m.DefaultResponse.Err
}

// Calls returns a copy of all recorded calls in invocation order.
func (m *Mock) Calls() []Call {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Call, len(m.calls))
	copy(out, m.calls)
	return out
}

// CallCount returns the number of times Exec has been invoked.
func (m *Mock) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.calls)
}

var _ Sandbox = (*Mock)(nil)

// MaterializeWorkspace implements Sandbox. It creates a FRESH EMPTY
// directory (os.MkdirTemp, prefix "llmkit-mock-") and returns its path —
// never the caller's repoDir, which is accepted but neither read nor
// required to exist (the Mock runs nothing, so there is nothing to copy
// into the workspace). The caller owns the returned directory's lifecycle:
// remove it with os.RemoveAll when done, exactly like the real backends'
// contract. Every created path is recorded and returned by Materialized.
func (m *Mock) MaterializeWorkspace(repoDir string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	dir, err := os.MkdirTemp("", "llmkit-mock-")
	if err != nil {
		return "", err
	}
	m.materialized = append(m.materialized, dir)
	return dir, nil
}

// Materialized returns a copy of the directories MaterializeWorkspace has
// created, in call order. It lets a caller's test pin the
// materialize-then-Exec flow the way Calls pins the Exec flow.
func (m *Mock) Materialized() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, len(m.materialized))
	copy(out, m.materialized)
	return out
}

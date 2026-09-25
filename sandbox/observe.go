package sandbox

import (
	"context"
	"fmt"
	"io"
	"slices"

	"github.com/dpoage/llmkit"
)

// Observe wraps s so every Exec call reports one llmkit.Event
// (llmkit.KindExec) to obs and otherwise behaves exactly like s:
// the Result, any error, and MaterializeWorkspace pass through unchanged
// (MaterializeWorkspace emits nothing). A nil obs returns s itself.
//
// The event inherits llmkit.Event.RunID, llmkit.Event.SpanID, and
// llmkit.Event.Step from the call's context (llmkit.NewEvent); Observe
// never mints a span — only Completion emitters do. Populated fields:
//   - Backend: the concrete backend — "cli", "bwrap", and "host" match
//     UnsupportedSpecError.Backend; "mock" is this package's own name for
//     the scripted backend, which refuses a malformed Spec exactly like
//     every backend; a nested Observe reports its inner backend; any other
//     implementation is named by its Go type.
//   - Command: Spec.Cmd, copied — the event outlives the call, and the
//     caller owns the Spec's backing arrays.
//   - ExitCode: the command's own exit code; -1 when Exec returned an
//     error (Err non-empty) or the process never ran to an exit because a
//     watchdog killed it. Check Result.InfraKilled on the returned Result
//     before classifying a verdict; the event records the summary, not
//     the kill reason.
//   - StdoutBytes / StderrBytes: the lengths of the captured
//     Result.Stdout / Result.Stderr text — markers included, so a
//     truncated stream can exceed the capture cap. In the common case
//     the text carries the mid-stream gap marker "[N bytes elided by
//     sandbox]"; when no tail was retained the backend appends
//     "[output truncated by sandbox]" instead.
//   - Truncated: Result.StdoutTruncated or Result.StderrTruncated — either
//     stream was cut off at the capture cap.
//   - Duration: Result.Duration, the execution's measured wall time; 0
//     whenever the backend measured none — an Exec error before the launch
//     attempt (a refused Spec, for example) produced no Result, and a
//     scripted Mock may report none. A launch failure on CLI or Bwrap
//     (the runtime or launcher binary could not start) reports the launch
//     attempt's wall time alongside its error.
//   - Err: the text of Exec's error, non-empty exactly when Exec returned
//     one: a refused Spec (InvalidSpecError, UnsupportedSpecError), a
//     caller ctx that ended (the "sandbox: execution cancelled" error), or
//     an infrastructure failure — the package doc's "Error contract". A
//     non-zero exit code is the command's own verdict and is never an
//     error.
//
// Result.TimedOut, Result.WorkspaceQuotaExceeded, and captured files are
// not part of llmkit.ExecEvent — v1 exec events are a summary, so replaying
// this boundary is not possible today. With a non-nil obs, the value
// Observe returns also implements io.Closer: Close forwards to the inner
// backend's Close when it implements io.Closer (CLI, Bwrap), and returns
// nil otherwise (Mock, HostExec). The return type stays Sandbox (the
// optional-interface pattern), so callers that want to Close type-assert:
//
//	if c, ok := sb.(io.Closer); ok {
//		defer c.Close()
//	}
//
// Observers are synchronous data sinks: a panic in obs propagates to the
// caller and obs never affects the returned Result.
func Observe(s Sandbox, obs llmkit.Observer) Sandbox {
	if obs == nil {
		return s
	}
	return &observedSandbox{inner: s, obs: obs}
}

// observedSandbox is the Sandbox Observe returns. It holds no mutable state
// and is safe for concurrent use as long as inner is.
type observedSandbox struct {
	inner Sandbox
	obs   llmkit.Observer
}

var _ Sandbox = (*observedSandbox)(nil)

// Close forwards to the inner backend's Close when it implements io.Closer
// (CLI, Bwrap), and returns nil otherwise (Mock, HostExec, or any other
// Sandbox implementation with no Close method). This makes the value
// Observe returns an io.Closer whenever the wrapped backend is one — see
// Observe's doc comment for the type-assertion pattern callers use.
func (w *observedSandbox) Close() error {
	if c, ok := w.inner.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

func (w *observedSandbox) Exec(ctx context.Context, spec Spec) (Result, error) {
	res, err := w.inner.Exec(ctx, spec)

	ev := llmkit.NewEvent(ctx, llmkit.KindExec)
	ev.Duration = res.Duration
	ev.Exec = &llmkit.ExecEvent{
		Backend: backendName(w.inner),
		// The event outlives the call: a sink may retain it, and the
		// caller owns (and may reuse) spec.Cmd's backing array.
		Command:     slices.Clone(spec.Cmd),
		ExitCode:    execExitCode(res, err),
		StdoutBytes: int64(len(res.Stdout)),
		StderrBytes: int64(len(res.Stderr)),
		Truncated:   res.StdoutTruncated || res.StderrTruncated,
		Err:         errText(err),
	}
	w.obs.Observe(ctx, ev)
	return res, err
}

func (w *observedSandbox) MaterializeWorkspace(repoDir string) (string, error) {
	return w.inner.MaterializeWorkspace(repoDir)
}

// backendName names the backend for llmkit.ExecEvent.Backend: the
// backendCLI/backendBwrap/backendHost values match
// UnsupportedSpecError.Backend; backendMock is this package's own name for
// the scripted backend, which refuses a malformed Spec exactly like every
// backend; a wrapped backend reports its inner backend; any other
// implementation is named by its Go type.
func backendName(s Sandbox) string {
	switch t := s.(type) {
	case *CLI:
		return backendCLI
	case *Bwrap:
		return backendBwrap
	case *HostExec:
		return backendHost
	case *Mock:
		return backendMock
	case *observedSandbox:
		return backendName(t.inner)
	default:
		return fmt.Sprintf("%T", s)
	}
}

// execExitCode maps an Exec outcome onto llmkit.ExecEvent.ExitCode: the
// command's own exit code from the Result, or -1 whenever Exec returned an
// error, which leaves no command verdict (a watchdog kill reports -1 in its
// Result itself).
func execExitCode(res Result, err error) int {
	if err != nil {
		return -1
	}
	return res.ExitCode
}

// errText renders an error for an event's Err field: "" on success.
func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

package sandbox

import (
	"context"
	"fmt"

	"github.com/dpoage/llmkit"
)

// Observe wraps s so every Exec call reports one llmkit.Event
// (llmkit.KindExec) to obs and otherwise behaves exactly like s:
// the Result, any error, and MaterializeWorkspace pass through unchanged
// (MaterializeWorkspace emits nothing). A nil obs returns s itself.
//
// The event inherits llmkit.Event.RunID and llmkit.Event.SpanID from the
// call's context (llmkit.NewEvent); Observe never mints a span — only
// Completion emitters do. Populated fields:
//
//   - Backend: the concrete backend — "cli", "bwrap", "host", or "mock"
//     for this package's own types (the same names
//     UnsupportedSpecError.Backend uses); any other implementation is
//     named by its Go type.
//   - Command: Spec.Cmd.
//   - ExitCode: the command's own exit code; -1 when the process never ran
//     to an exit — an infrastructure failure (Err non-empty) or a watchdog
//     kill. Check Result.InfraKilled on the returned Result before
//     classifying a verdict; the event records the summary, not the kill
//     reason.
//   - StdoutBytes / StderrBytes: the captured Result.Stdout /
//     Result.Stderr lengths (the post-truncation capture size).
//   - Truncated: Result.StdoutTruncated or Result.StderrTruncated — either
//     stream was cut off at the capture cap.
//   - Duration: Result.Duration, the execution's measured wall time; 0 when
//     an infrastructure failure produced no Result.
//   - Err: the infrastructure error's text, non-empty exactly when Exec
//     returned an error. A non-zero exit code is the command's own verdict
//     and is never an error (see the Sandbox contract).
//
// Result.TimedOut, Result.WorkspaceQuotaExceeded, and captured files are
// not part of llmkit.ExecEvent — v1 exec events are a summary, so replaying
// this boundary is not possible today. Close (CLI and Bwrap) is not on the
// Sandbox interface: keep the concrete backend reference to call it.
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

func (w *observedSandbox) Exec(ctx context.Context, spec Spec) (Result, error) {
	res, err := w.inner.Exec(ctx, spec)

	ev := llmkit.NewEvent(ctx, llmkit.KindExec)
	ev.Duration = res.Duration
	ev.Exec = &llmkit.ExecEvent{
		Backend:     backendName(w.inner),
		Command:     spec.Cmd,
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

// backendName names the backend for llmkit.ExecEvent.Backend: the same
// strings UnsupportedSpecError.Backend uses for this package's own
// backends, and the Go type for any other implementation.
func backendName(s Sandbox) string {
	switch s.(type) {
	case *CLI:
		return "cli"
	case *Bwrap:
		return "bwrap"
	case *HostExec:
		return "host"
	case *Mock:
		return "mock"
	default:
		return fmt.Sprintf("%T", s)
	}
}

// execExitCode maps an Exec outcome onto llmkit.ExecEvent.ExitCode: the
// command's own exit code from the Result, or -1 when an infrastructure
// failure meant the process never produced one (a watchdog kill reports -1
// in its Result itself).
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

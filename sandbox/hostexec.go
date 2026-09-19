package sandbox

// hostexec.go implements the "attended escape hatch": an explicit opt-in
// unsandboxed executor that runs a Spec.Cmd directly on the HOST — no
// container, no network policy, no resource caps — against a fresh workspace
// COPY of RepoDir (never the live checkout; see prepareWorkspace). It exists
// solely for attended, operator-opted-in paths where a human has already
// decided to trust the command they are about to run; it must NEVER be
// reachable from any unattended or automated path. No kit default, kit
// example, or other kit code constructs one: the caller that explicitly opts
// in is the ONLY party permitted to build it.
//
// SECURITY: HostExec provides NO isolation. It grants the command the
// calling OS user's own privileges and full network access — the isolation
// boundary documented at the top of this package (network=none,
// workspace-only writes, dropped Linux capabilities) is INTENTIONALLY absent
// here. This is a deliberate, narrowly-scoped tradeoff for the one case
// where a human is actively watching the command they asked to run, not a
// general-purpose alternative backend.
import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"
)

// HostExec is a Sandbox implementation that runs commands directly on the
// host, in a temporary copy of RepoDir, with no container isolation. See the
// file doc for the security tradeoff and the caller contract.
type HostExec struct{}

// NewHostExec constructs a HostExec. It takes no configuration: unlike CLI,
// there is no image, network mode, or resource cap to configure — those
// concepts do not apply to a bare host process.
func NewHostExec() *HostExec { return &HostExec{} }

// Exec implements Sandbox. It copies spec.RepoDir into a fresh temporary
// workspace (or reuses spec.Workspace verbatim if the caller already
// materialized one — see Spec.Workspace's doc; the workspace is still never
// the live checkout either way), applies spec.WriteFiles, and runs spec.Cmd
// directly via os/exec with spec.Env appended to the host's own environment.
//
// The isolation-shaped Spec fields are REFUSED rather than silently ignored —
// there is no container to apply them to, and running without a requested
// knob would be a silently weaker posture than the Spec asked for:
//   - Image, ROMounts, RWMounts, SetupCmds: UnsupportedSpecError. SetupCmds
//     in particular are refused rather than approximated, because the
//     container backends' guarantee that a failed setup step exits 125
//     (interpreted as an environment error, never the command's own outcome —
//     see command.go) has no equivalent here; silently chaining them via a
//     shell would reintroduce that exact fragility without the guarantee.
//   - Network: "" (the default) and NetworkHost are honored — both mean "run
//     with the host's network", which is what a bare host process does.
//     NetworkNone, NetworkBridge, or any other mode is refused: HostExec
//     cannot cut the host's network, and pretending otherwise would silently
//     run a network-isolated Spec with full access.
//
// There are no resource caps and no idle watchdog on this backend, so
// Result.WorkspaceQuotaExceeded is always false; only the absolute
// Spec.Timeout can kill a run.
func (h *HostExec) Exec(ctx context.Context, spec Spec) (Result, error) {
	if len(spec.Cmd) == 0 {
		return Result{}, fmt.Errorf("sandbox: HostExec requires a non-empty Cmd")
	}
	if spec.RepoDir == "" && spec.Workspace == "" {
		return Result{}, fmt.Errorf("sandbox: HostExec requires RepoDir or Workspace")
	}
	if spec.Image != "" {
		return Result{}, &UnsupportedSpecError{Backend: "host", Field: "Image", Value: spec.Image}
	}
	if len(spec.ROMounts) > 0 {
		return Result{}, &UnsupportedSpecError{Backend: "host", Field: "ROMounts", Value: fmt.Sprintf("%d mount(s)", len(spec.ROMounts))}
	}
	if len(spec.RWMounts) > 0 {
		return Result{}, &UnsupportedSpecError{Backend: "host", Field: "RWMounts", Value: fmt.Sprintf("%d mount(s)", len(spec.RWMounts))}
	}
	if len(spec.SetupCmds) > 0 {
		return Result{}, &UnsupportedSpecError{Backend: "host", Field: "SetupCmds", Value: fmt.Sprintf("%d command(s)", len(spec.SetupCmds))}
	}
	if _, err := resolveNetworkMode("host", NetworkHost, spec.Network, NetworkHost); err != nil {
		return Result{}, err
	}

	prepStart := time.Now()
	ws := spec.Workspace
	cleanup := func() {}
	if ws == "" {
		w, err := prepareWorkspace(spec.RepoDir, spec.WriteFiles)
		if err != nil {
			return Result{}, err
		}
		ws = w
		cleanup = func() { _ = os.RemoveAll(ws) }
	} else if len(spec.WriteFiles) > 0 {
		if err := applyWriteFiles(ws, spec.WriteFiles); err != nil {
			return Result{}, err
		}
	}
	defer cleanup()
	prepDuration := time.Since(prepStart)

	runCtx := ctx
	var cancel context.CancelFunc
	if spec.Timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, spec.Timeout)
		defer cancel()
	}

	maxBytes := DefaultMaxOutputBytes
	stdout := newCappedBuffer(maxBytes)
	stderr := newCappedBuffer(maxBytes)

	cmd := exec.CommandContext(runCtx, spec.Cmd[0], spec.Cmd[1:]...)
	cmd.Dir = ws
	if len(spec.Env) > 0 {
		cmd.Env = append(os.Environ(), spec.Env...)
	}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	start := time.Now()
	runErr := cmd.Run()
	duration := time.Since(start)

	timedOut := runCtx.Err() == context.DeadlineExceeded
	exitCode := 0
	if runErr != nil {
		if timedOut {
			exitCode = -1
		} else if exitErr, ok := runErr.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			// A launch failure (binary not found, permission denied, etc.) is
			// an infrastructure error, not a demonstrable exit code — matches
			// the CLI backend's Exec error contract.
			return Result{}, fmt.Errorf("sandbox: HostExec run %v: %w", spec.Cmd, runErr)
		}
	}

	outStr, outTrunc := stdout.result()
	errStr, errTrunc := stderr.result()

	var captured map[string][]byte
	if len(spec.CaptureFiles) > 0 {
		captured = captureWorkspaceFiles(ws, spec.CaptureFiles, maxBytes)
	}

	return Result{
		ExitCode:        exitCode,
		Stdout:          outStr,
		Stderr:          errStr,
		StdoutTruncated: outTrunc,
		StderrTruncated: errTrunc,
		Duration:        duration,
		TimedOut:        timedOut,
		PrepDuration:    prepDuration,
		Captured:        captured,
	}, nil
}

var _ Sandbox = (*HostExec)(nil)

// MaterializeWorkspace implements Sandbox. HostExec has no pristine cache to
// consult: it performs a fresh full copy of repoDir into a caller-owned
// temporary directory (the standalone prepareWorkspace, no WriteFiles). The
// caller owns the returned directory's lifecycle — the same contract as the
// container-backed backends; see the Sandbox interface.
func (h *HostExec) MaterializeWorkspace(repoDir string) (string, error) {
	return prepareWorkspace(repoDir, nil)
}

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
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"syscall"
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
// Spec.Timeout can kill a run (no backend default: timeout <= 0 means the
// only bound is the caller's context). Workspace must be absolute (like every
// backend — validateSpec) and is checked to exist as a directory this
// process can enter before the run.
func (h *HostExec) Exec(ctx context.Context, spec Spec) (Result, error) {
	if err := validateSpec(backendHost, spec); err != nil {
		return Result{}, err
	}
	return runSupervised(ctx, runSpec{
		spec:           spec,
		timeout:        spec.Timeout,
		maxOutputBytes: DefaultMaxOutputBytes,
		hooks: runHooks{
			prepareWorkspace: func(repoDir string) (string, bool, error) {
				ws, err := prepareWorkspace(repoDir, spec.WriteFiles)
				return ws, false, err
			},
			buildCmd: func(ws string, runCtx context.Context) (*exec.Cmd, error) {
				cmd := exec.CommandContext(runCtx, spec.Cmd[0], spec.Cmd[1:]...)
				cmd.Dir = ws
				if len(spec.Env) > 0 {
					cmd.Env = append(os.Environ(), spec.Env...)
				}
				return cmd, nil
			},
			commandExit: hostCommandExit,
		},
	})
}

var _ Sandbox = (*HostExec)(nil)

// hostCommandExit maps a run error the classifier left as infrastructure
// onto Spec.Cmd's own exit status. HostExec supervises the command itself
// — no runtime or shell in between — so two such errors are the command's
// verdict, reported the way a shell (and the container backends) report
// them, with a nil error:
//
//   - death by a signal nobody in the kit sent (the watchdog, deadline, and
//     caller-ctx kills are classified before this point): 128+signo;
//   - a LAUNCH failure — the child never started: 127 when the command
//     could not be found, 126 when it was found but cannot be executed (no
//     permission, or a format the OS refuses to run).
//
// Every other error — e.g. a workspace that vanished before the chdir —
// stays an infrastructure error (ok false). The observed launch shapes:
// an absolute missing path is *os.PathError{Op:"fork/exec", ENOENT}; a bare
// name is exec.ErrNotFound; a non-executable file is fs.ErrPermission; a
// found-but-unrunnable file is ENOEXEC ("exec format error"); a missing
// cmd.Dir is *os.PathError{Op:"chdir"}. The supervisor refuses a Workspace
// the child cannot enter before the run, but that check precedes Start: a
// concurrent actor that revokes search permission on the caller's Workspace
// in between still surfaces here as the chdir's EACCES, which maps to 126.
func hostCommandExit(runErr error) (exitCode int, ok bool) {
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		if ws, isWait := exitErr.Sys().(interface {
			Signaled() bool
			Signal() syscall.Signal
		}); isWait && ws.Signaled() {
			return 128 + int(ws.Signal()), true
		}
		return 0, false
	}
	if errors.Is(runErr, exec.ErrNotFound) {
		return 127, true
	}
	var pathErr *os.PathError
	if errors.As(runErr, &pathErr) && pathErr.Op == "fork/exec" {
		switch {
		case errors.Is(pathErr, fs.ErrNotExist):
			return 127, true
		case errors.Is(pathErr, fs.ErrPermission), errors.Is(pathErr, syscall.ENOEXEC):
			return 126, true
		}
	}
	return 0, false
}

// MaterializeWorkspace implements Sandbox. HostExec has no pristine cache to
// consult: it performs a fresh full copy of repoDir into a caller-owned
// temporary directory (the standalone prepareWorkspace, no WriteFiles). The
// caller owns the returned directory's lifecycle — the same contract as the
// container-backed backends; see the Sandbox interface.
func (h *HostExec) MaterializeWorkspace(repoDir string) (string, error) {
	return prepareWorkspace(repoDir, nil)
}

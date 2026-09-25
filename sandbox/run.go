package sandbox

// run.go owns RUN SUPERVISION and OUTCOME CLASSIFICATION for the three
// real backends (CLI, Bwrap, HostExec). It hides how a supervised child's
// start, exit, the caller's context, the run deadline, the idle watchdog,
// and the growth ceiling combine into one (Result, error).
//
// The precedence is caller cancel → growth-ceiling breach → own exit →
// idle/absolute timeout → infrastructure error. Kills are reaped, the
// workspace is checked and prepared before any child starts (and, when
// we created it, removed after), and a backend whose supervised process
// IS the command can map what would otherwise be an infrastructure error
// onto the command's own exit status.
//
// The backends parameterize the supervisor with a runSpec (values: the
// resolved ceilings and policies) and runHooks (functions: admission,
// workspace preparation, building/starting the child, the watchdog's
// liveness probe, reaping, command-exit mapping) and keep only their
// genuine backend knowledge: argv construction and per-backend timeout
// defaulting.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"time"
)

// cancelledRunErr builds the "sandbox: execution cancelled" error from a
// context whose Err is non-nil. Single construction site shared by
// classifyRun and the Mock, so every backend reports a caller
// cancellation with exactly one message.
func cancelledRunErr(ctx context.Context) error {
	return fmt.Errorf("sandbox: execution cancelled: %w", ctx.Err())
}

// classifyRun resolves a finished run's outcome. Precedence:
//
//  1. caller ctx ended (ctx.Err() != nil: cancel OR deadline) — the caller
//     no longer wants the result, whatever our machinery observed: error,
//     exit code 0 left in place, no kill flags.
//  2. growth-ceiling breach (quotaExceeded) — a measured invariant on
//     final disk usage that always overrides the process's own exit
//     code: exit -1, quota.
//  3. own exit — a clean exit (runErr nil) is 0; an *exec.ExitError with
//     ExitCode() >= 0 is the command's own code. A genuine exit wins
//     over an idle/deadline kill firing in the same instant (our kills
//     surface as a signal, ExitCode -1, and fall through).
//  4. idle watchdog or absolute deadline (idleKilled, or runCtx past its
//     deadline) — the run never finished on its own: exit -1, timedOut.
//  5. anything else — an infrastructure error (launch failure, a signal
//     nobody in the kit sent): returned as err with the exit code left
//  0. The supervisor may still map it through runHooks.commandExit
//     (HostExec, whose supervised process is the command itself).
func classifyRun(ctx, runCtx context.Context, runErr error, idleKilled, quotaExceeded bool) (exitCode int, timedOut, quota bool, err error) {
	if ctx.Err() != nil {
		return 0, false, false, cancelledRunErr(ctx)
	}
	if quotaExceeded {
		return -1, false, true, nil
	}
	if runErr == nil {
		return 0, false, false, nil
	}
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) && exitErr.ExitCode() >= 0 {
		return exitErr.ExitCode(), false, false, nil
	}
	if idleKilled || errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return -1, true, false, nil
	}
	return 0, false, false, runErr
}

// runHooks carries the per-backend FUNCTIONS of a supervised run (the
// values live in runSpec). A nil hook means "this backend has none";
// every hook has at least one backend using it.
type runHooks struct {
	// admit is the backend's last refusal point: it runs after the
	// supervisor's Workspace checks and its ended-caller-ctx check, and
	// before anything is written or launched. Bwrap resolves its
	// resource-cap method here and refuses with ErrBwrapNoCapMethod.
	admit func(ctx context.Context) error
	// prepareWorkspace materializes a fresh workspace copy of repoDir
	// (with the Spec's WriteFiles applied) for a Spec without a
	// caller-owned Workspace. The supervisor removes it after the run.
	prepareWorkspace func(repoDir string) (ws string, cacheHit bool, err error)
	// buildCmd constructs the child exec.Cmd once the workspace (ws) and
	// the deadline-bounded runCtx exist. The supervisor binds
	// cmd.Stdout/cmd.Stderr itself. A non-nil error (bwrap's resource-cap
	// wrapper failing to build) fails the run before anything starts.
	// Everything buildCmd writes happens-before the watchdog goroutine
	// starts.
	buildCmd func(ws string, runCtx context.Context) (*exec.Cmd, error)
	// afterStart runs once right after a successful Start, before
	// Wait — Bwrap joins its cgroup subtree here. A non-nil error kills,
	// reaps, and fails the run.
	afterStart func(cmd *exec.Cmd) error
	// cpuBusy is the watchdog's costlier liveness fallback (container
	// CPU stats on CLI, /proc tree sampling on Bwrap). Consulted only
	// when the cheap output/filesystem fingerprint is flat. It runs on
	// the WATCHDOG goroutine, concurrently with the supervisor: it may
	// read only state fixed before buildCmd returned, or state it
	// synchronizes itself.
	cpuBusy func() bool
	// reap force-kills the run's process/container after a caller-ctx
	// end, quota, or timeout outcome (docker rm -f on CLI; process-group
	// kill on Bwrap). Called after Wait, so it only mops up whatever
	// outran the kill. cmd.Process is nil when Start itself failed.
	reap func(cmd *exec.Cmd)
	// commandExit maps a run error classifyRun left as INFRASTRUCTURE
	// onto the command's own exit status: (code, true) turns the run into
	// a Result with that exit code and nil error. Only HostExec sets it
	// — its supervised process IS Spec.Cmd, so a launch failure (127
	// missing, 126 not executable) or a death by signal (128+signo) is
	// the command's own verdict. On the container backends the supervised
	// process is the runtime, which already reports those as exit codes
	// itself, and its own failures are infrastructure.
	commandExit func(runErr error) (exitCode int, ok bool)
}

// runSpec is one supervised run's inputs: the admitted Spec (validateSpec
// already passed), the backend-resolved values, and the backend hooks.
type runSpec struct {
	spec Spec
	// timeout is ALREADY backend-defaulted by the caller (Spec.Timeout
	// <= 0 → the backend default on CLI/Bwrap; HostExec passes
	// Spec.Timeout verbatim). <= 0 means "no run deadline" (only the
	// caller's ctx bounds the run) — reachable only from HostExec, since
	// NewCLI and NewBwrap refuse a <= 0 WithTimeout at construction. A
	// positive timeout that expires before the child starts is classified
	// like any other expired deadline: TimedOut with ExitCode -1 on every
	// backend.
	timeout       time.Duration
	idleTimeout   time.Duration
	growthCeiling int64
	// maxOutputBytes caps each captured stream and each captured file.
	maxOutputBytes int
	hooks          runHooks
}

// runSupervised executes one admitted Spec to completion: it checks a
// caller-owned Workspace, reports an already-ended caller ctx as the
// cancelled error, and runs the backend's admission before anything is
// written, prepares the workspace (a caller-owned directory used
// verbatim, or a fresh copy this call removes), starts the child under
// the run deadline, wires the idle watchdog and growth ceiling when
// either is on, captures output, and classifies the outcome via
// classifyRun (reaping on caller-ctx end/quota/timeout). spec must have
// passed validateSpec; the Workspace checks are the filesystem-shaped
// admission the pure validator cannot do.
func runSupervised(ctx context.Context, r runSpec) (Result, error) {
	// Admission — before ANY write or runtime contact. A caller-owned
	// Workspace is absolute (validateSpec checked) and must be a directory
	// the child can enter: stat'ing ws+"/." needs search permission on
	// ws, exactly what the child's chdir needs. Without this check a
	// non-searchable Workspace would surface as the child's chdir EACCES,
	// which reads like a non-executable command (126) on HostExec.
	if ws := r.spec.Workspace; ws != "" {
		info, statErr := os.Stat(ws)
		if statErr != nil {
			return Result{}, &InvalidSpecError{Field: "Workspace", Reason: fmt.Sprintf("stat %q: %v", ws, statErr)}
		}
		if !info.IsDir() {
			return Result{}, &InvalidSpecError{Field: "Workspace", Reason: fmt.Sprintf("%q is not a directory", ws)}
		}
		if _, err := os.Stat(ws + string(filepath.Separator) + "."); err != nil {
			return Result{}, &InvalidSpecError{Field: "Workspace", Reason: fmt.Sprintf("%q is not searchable: %v", ws, err)}
		}
	}
	// A caller ctx that ended before Exec was called is the cancelled
	// error, ahead of admission: nothing is written and nothing is
	// launched. One that ends during admission is caught by the re-check
	// below; one that ends during the run by classifyRun.
	if ctx.Err() != nil {
		return Result{}, cancelledRunErr(ctx)
	}
	if r.hooks.admit != nil {
		err := r.hooks.admit(ctx)
		// Re-checked BEFORE admit's error: admission run with a dying ctx
		// misreports the caller's end — Bwrap's cap detection degrades to
		// cgroup v2 (whose limit write later fails) or to none
		// (ErrBwrapNoCapMethod), and neither stands in for the caller's end.
		if ctx.Err() != nil {
			return Result{}, cancelledRunErr(ctx)
		}
		if err != nil {
			return Result{}, err
		}
	}
	// The cleaned CaptureFiles list Result.Captured is keyed by.
	// validateSpec admitted every entry, so this cannot fail; it runs
	// before any write all the same.
	capturePaths, err := sanitizeCapturePaths(r.spec.CaptureFiles)
	if err != nil {
		return Result{}, err
	}

	// Workspace step: WriteFiles land in a caller-owned Workspace here,
	// once, before the child runs; otherwise a fresh copy is prepared.
	prepStart := time.Now()
	var ws string
	var cacheHit bool
	if r.spec.Workspace != "" {
		ws = r.spec.Workspace
		if err := applyWriteFiles(ws, r.spec.WriteFiles); err != nil {
			return Result{}, err
		}
	} else {
		ws, cacheHit, err = r.hooks.prepareWorkspace(r.spec.RepoDir)
		if err != nil {
			return Result{}, err
		}
		// A fresh copy is this call's own scratch: removed when the run
		// ends. A caller-owned Workspace is never touched on cleanup.
		defer func() { _ = os.RemoveAll(ws) }()
	}
	prepDuration := time.Since(prepStart)

	// The run deadline: a hard ceiling derived from the caller's ctx.
	// timeout <= 0 means "no deadline" (only reachable from HostExec —
	// see runSpec.timeout); runCtx stays cancellable even then, since
	// the watchdog's kill needs a cancel func on every path.
	var runCtx context.Context
	var cancel context.CancelFunc
	if r.timeout <= 0 {
		runCtx, cancel = context.WithCancel(ctx)
	} else {
		runCtx, cancel = context.WithTimeout(ctx, r.timeout)
	}
	defer cancel()

	// Capped per-stream capture buffers.
	stdout := newCappedBuffer(r.maxOutputBytes)
	stderr := newCappedBuffer(r.maxOutputBytes)

	cmd, buildErr := r.hooks.buildCmd(ws, runCtx)
	if buildErr != nil {
		return Result{}, buildErr
	}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	// Watchdog: cancel only after idleTimeout with NO observable
	// progress (output bytes, workspace tree, then the costlier cpuBusy
	// probe), and independently bound NET workspace growth since
	// growthBase (a disk-filler reads as "progress" to the idle check).
	// growthBase is captured HERE, once, before the child starts, so the
	// tick loop and the post-run check below measure the same baseline.
	// Started after buildCmd, so everything buildCmd set happens-before
	// the goroutine. Neither condition configured — no watchdog at all
	// (HostExec's shape).
	var idleKilled, quotaExceeded atomic.Bool
	done := make(chan struct{})
	var fingerprint func() progressSnapshot
	var growthBase progressSnapshot
	if r.idleTimeout > 0 || r.growthCeiling > 0 {
		fingerprint = func() progressSnapshot {
			ps := progressSnapshot{outputBytes: stdout.written() + stderr.written()}
			ps.fsSize, ps.fsCount, ps.fsMaxModNano = workspaceProgress(ws)
			return ps
		}
		growthBase = fingerprint()
		go watchIdle(watchdogArgs{
			done:           done,
			fingerprint:    fingerprint,
			activeFallback: r.hooks.cpuBusy,
			limits:         watchdogLimits{idleTimeout: r.idleTimeout, growthCeilingBytes: r.growthCeiling},
			base:           growthBase,
			pollEvery:      effectivePollInterval(r.idleTimeout, r.growthCeiling),
			killed:         &idleKilled,
			quotaExceeded:  &quotaExceeded,
			cancel:         cancel,
		})
	}

	// Start + wait. An afterStart failure (cgroup join) kills and reaps
	// first. A Start failure itself (the wrapper binary could not be
	// launched) is classified like any other run error below — an
	// expired run deadline reads as a timeout, and commandExit may map
	// a launch failure — the same path CLI and HostExec already used.
	start := time.Now()
	var runErr error
	if startErr := cmd.Start(); startErr != nil {
		runErr = startErr
	} else if r.hooks.afterStart != nil {
		if err := r.hooks.afterStart(cmd); err != nil {
			if r.hooks.reap != nil {
				r.hooks.reap(cmd)
			}
			_ = cmd.Wait()
			close(done)
			return Result{}, err
		}
	}
	if runErr == nil {
		runErr = cmd.Wait()
	}
	close(done)
	duration := time.Since(start)

	// Post-run growth check BEFORE classification: a run that breaches
	// the ceiling and exits inside a single poll window must still be
	// caught — the breach is a measured invariant, never allowed to lose
	// to a genuine exit code.
	checkGrowthCeiling(fingerprint, growthBase, r.growthCeiling, &quotaExceeded)

	res := Result{Duration: duration, PrepDuration: prepDuration, WorkspaceCacheHit: cacheHit}
	res.Stdout, res.StdoutTruncated = stdout.result()
	res.Stderr, res.StderrTruncated = stderr.result()
	res.Captured = captureWorkspaceFiles(ws, capturePaths, r.maxOutputBytes)

	exitCode, timedOut, quota, err := classifyRun(ctx, runCtx, runErr, idleKilled.Load(), quotaExceeded.Load())
	// Any end of the caller's ctx — cancel OR deadline. Read after
	// classifyRun: ctx.Err never reverts to nil, so this is true for
	// every run classifyRun reported as cancelled.
	callerEnded := ctx.Err() != nil
	res.TimedOut = timedOut
	res.WorkspaceQuotaExceeded = quota
	if timedOut || quota || err == nil {
		res.ExitCode = exitCode
	}
	// Reap on every outcome that was our kill or the caller's end; a
	// genuine exit needs no cleanup.
	if timedOut || quota || callerEnded {
		if r.hooks.reap != nil {
			r.hooks.reap(cmd)
		}
	}
	if err == nil {
		return res, nil
	}
	if callerEnded {
		// Cancellation precedence is absolute: no exit-code mapping, and
		// a caller end that raced classifyRun still reports as cancelled.
		return res, cancelledRunErr(ctx)
	}
	if r.hooks.commandExit != nil {
		if code, ok := r.hooks.commandExit(runErr); ok {
			res.ExitCode = code
			return res, nil
		}
	}
	return res, fmt.Errorf("sandbox: run %s: %w", cmd.Args[0], err)
}

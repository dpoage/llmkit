package sandbox

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// run_test.go pins the ONE outcome classifier (bead llmkit-bk8.7.1):
// classifyRun's precedence — caller cancel → growth breach → own exit →
// idle/absolute timeout → infra error — with flag rows paired with GENUINE
// exits produced by real short commands.

// genuineExit runs a real command and returns the resulting *exec.ExitError.
func genuineExit(t *testing.T, script string) *exec.ExitError {
	t.Helper()
	err := exec.Command("/bin/sh", "-c", script).Run()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("sh -c %q: err = %v, want *exec.ExitError", script, err)
	}
	return exitErr
}

// TestClassifyRunPrecedence pins every precedence row. ctx/runCtx shape:
//
//	cancelled  — the CALLER's ctx is already cancelled
//	deadline   — runCtx is past its own deadline
//	plain      — neither is done
func TestClassifyRunPrecedence(t *testing.T) {
	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	pastDeadline, cancelDeadline := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancelDeadline()
	<-pastDeadline.Done()

	fresh := context.Background()
	code3 := genuineExit(t, "exit 3")
	sigkilled := genuineExit(t, "kill -9 $$")

	// A caller DEADLINE is a caller end exactly like a cancel: the shared
	// cancelled error, wrapping context.DeadlineExceeded — never a timeout
	// shape and never the command's exit.
	exitCode, timedOut, _, err := classifyRun(pastDeadline, pastDeadline, sigkilled, false, false)
	if !errors.Is(err, context.DeadlineExceeded) || err.Error() != "sandbox: execution cancelled: context deadline exceeded" {
		t.Fatalf("caller-deadline row: err = %v, want the shared cancelled error wrapping DeadlineExceeded", err)
	}
	if exitCode != 0 || timedOut {
		t.Errorf("caller-deadline row: (exit=%d, timedOut=%t), want (0, false)", exitCode, timedOut)
	}

	// A caller cancel wins over EVERYTHING — even over a quota breach and
	// a genuine exit.
	exitCode, _, _, err = classifyRun(cancelledCtx, fresh, nil, false, true)
	if !errors.Is(err, context.Canceled) || err.Error() != "sandbox: execution cancelled: context canceled" {
		t.Fatalf("cancel row: err = %v, want the shared cancelled error", err)
	}
	if exitCode != 0 {
		t.Errorf("cancel row: exitCode = %d, want 0 (the run's outcome is not the caller's)", exitCode)
	}

	rows := []struct {
		name          string
		ctx           context.Context
		runCtx        context.Context
		runErr        error
		idleKilled    bool
		quota         bool
		wantExit      int
		wantTimedOut  bool
		wantQuota     bool
		wantErr       bool
		wantCancelled bool
	}{
		{"cancel with quota breach wins", cancelledCtx, fresh, nil, false, true, 0, false, false, true, true},
		{"cancel over genuine exit wins", cancelledCtx, fresh, nil, false, false, 0, false, false, true, true},
		{"quota breach overrides own exit", fresh, fresh, nil, false, true, -1, false, true, false, false},
		{"clean exit is 0", fresh, fresh, nil, false, false, 0, false, false, false, false},
		{"own exit code 3 is preserved", fresh, fresh, code3, false, false, 3, false, false, false, false},
		{"own exit beats an idle kill", fresh, fresh, nil, true, false, 0, false, false, false, false},
		{"signal-killed exit falls to idle kill", fresh, fresh, sigkilled, true, false, -1, true, false, false, false},
		{"run deadline with signal kill is a timeout", fresh, pastDeadline, sigkilled, false, false, -1, true, false, false, false},
		{"plain error is infra", fresh, fresh, errors.New("boom"), false, false, 0, false, false, true, false},
		// A signal nobody in the kit sent (no idle kill, no quota, run
		// deadline live) is infrastructure here; HostExec, whose child IS
		// the command, maps it afterwards via runHooks.commandExit.
		{"signal-killed exit with no flags is infra", fresh, fresh, sigkilled, false, false, 0, false, false, true, false},
		{"idle kill alone with no runErr is still own exit", fresh, fresh, nil, true, false, 0, false, false, false, false},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			exitCode, timedOut, quota, err := classifyRun(row.ctx, row.runCtx, row.runErr, row.idleKilled, row.quota)
			if exitCode != row.wantExit || timedOut != row.wantTimedOut || quota != row.wantQuota {
				t.Fatalf("got (exit=%d, timedOut=%t, quota=%t), want (%d, %t, %t)", exitCode, timedOut, quota, row.wantExit, row.wantTimedOut, row.wantQuota)
			}
			if (err != nil) != row.wantErr {
				t.Fatalf("err = %v, wantErr = %t", err, row.wantErr)
			}
			if row.wantCancelled && !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want errors.Is(context.Canceled)", err)
			}
		})
	}
}

// TestCancelledRunErrIsTheSingleMessage pins the shared constructor: the
// error text is exactly the documented one, built from the context's own
// Err.
func TestCancelledRunErrIsTheSingleMessage(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Minute))
	defer cancel()
	err := cancelledRunErr(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want errors.Is(context.DeadlineExceeded)", err)
	}
	if got := err.Error(); got != "sandbox: execution cancelled: context deadline exceeded" {
		t.Fatalf("Error() = %q, want the documented cancelled message", got)
	}
}

// TestRunSupervisedDeadlinePolicy pins the unified run-deadline policy
// (llmkit-bk8.1.7/S1-7): timeout <= 0 means "no run deadline" — only the
// caller's ctx bounds the run — and a watchdog configured on that path
// still gets a real cancel func, instead of dereferencing a nil cancel on
// the watchdog goroutine (which would crash the whole test binary — a
// panic there cannot be recovered here). This is reachable only from
// HostExec: NewCLI and NewBwrap now refuse a <= 0 WithTimeout at
// construction, so runSupervised never sees a non-positive timeout from
// them. RETIRES the old two-flavor policy test (a bare runSpec with
// timeout=0 and no flag set producing an immediately-expired
// context.WithTimeout deadline): that shape modeled a CLI/Bwrap state
// that construction no longer allows to exist (the base-parity
// observation at history://ArchR1b.OracleS0B2 row 78 — bwrap/CLI
// WithTimeout(<=0) failing fast — is retired by the same refusal).
func TestRunSupervisedDeadlinePolicy(t *testing.T) {
	sleeper := runHooks{
		buildCmd: func(_ string, runCtx context.Context) (*exec.Cmd, error) {
			return exec.CommandContext(runCtx, "sleep", "30"), nil
		},
	}
	start := time.Now()
	res, err := runSupervised(context.Background(), runSpec{
		spec:        Spec{Workspace: t.TempDir(), Cmd: []string{"sleep", "30"}},
		timeout:     0,
		idleTimeout: 300 * time.Millisecond,
		hooks:       sleeper,
	})
	if err != nil {
		t.Fatalf("err = %v, want nil (an idle-kill shape)", err)
	}
	if !res.TimedOut || res.ExitCode != -1 {
		t.Errorf("res = %+v, want the idle kill: TimedOut with ExitCode -1", res)
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Errorf("took %v, want the idle watchdog to kill within a few idle windows", d)
	}
}

// TestRunSupervisedEndedCallerCtxPrecedesAdmission pins that a caller ctx
// that has already ended when Exec is called is the cancelled error on
// every real backend, ahead of backend admission and of any write: no
// ErrBwrapNoCapMethod (whose remediation, "run uncapped", answers a
// question the caller never asked), no cap detection, and no WriteFiles in
// the caller's Workspace. Bwrap is the zero value with cap detection
// stubbed to "none" and counted, so no row touches the host's bwrap,
// systemd, cgroups, or a container runtime.
func TestRunSupervisedEndedCallerCtxPrecedesAdmission(t *testing.T) {
	stubCapMethod(t, bwrapCapNone)
	stubbed := detectCapMethod
	detections := 0
	detectCapMethod = func(ctx context.Context, expandSupport func(context.Context) bool) bwrapCapMethod {
		detections++
		return stubbed(ctx, expandSupport)
	}
	t.Cleanup(func() { detectCapMethod = stubbed })

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()
	ctxs := []struct {
		name string
		ctx  context.Context
		want error
	}{
		{"cancelled", cancelled, context.Canceled},
		{"expired deadline", expired, context.DeadlineExceeded},
	}
	for _, c := range ctxs {
		for _, backend := range []string{backendCLI, backendBwrap, backendHost} {
			t.Run(c.name+"/"+backend, func(t *testing.T) {
				detections = 0
				ws := t.TempDir()
				_, err := execFor(t, backend).Exec(c.ctx, Spec{
					Workspace:  ws,
					Cmd:        []string{"true"},
					WriteFiles: map[string][]byte{"marker": []byte("x")},
				})
				if err == nil || !errors.Is(err, c.want) || !strings.HasPrefix(err.Error(), "sandbox: execution cancelled") || errors.Is(err, ErrBwrapNoCapMethod) {
					t.Errorf("err = %v, want the cancelled error wrapping %v", err, c.want)
				}
				if _, statErr := os.Stat(filepath.Join(ws, "marker")); !errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("marker stat = %v, want ENOENT: WriteFiles landed in the caller's Workspace", statErr)
				}
				if detections != 0 {
					t.Errorf("cap detection ran %d time(s), want 0: admission ran for an ended caller ctx", detections)
				}
			})
		}
	}
}

// TestRunSupervisedCallerCtxEndingDuringAdmission pins that a caller ctx
// that ends WHILE Bwrap's admission runs is the cancelled error too: not
// the admission's own outcome (ErrBwrapNoCapMethod here; a real host's
// probes fail under the dying ctx and degrade to a cgroup write that
// fails), and no WriteFiles in the caller's Workspace. Bwrap is the zero
// value with cap detection stubbed to end the ctx, then report "none".
// CLI and HostExec have no admission hook, so no such window.
func TestRunSupervisedCallerCtxEndingDuringAdmission(t *testing.T) {
	rows := []struct {
		name string
		// ctx returns the caller ctx and the end the stub drives inside
		// admission.
		ctx  func(t *testing.T) (context.Context, func(context.Context))
		want error
	}{
		{"cancelled", func(t *testing.T) (context.Context, func(context.Context)) {
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			return ctx, func(context.Context) { cancel() }
		}, context.Canceled},
		{"deadline", func(t *testing.T) (context.Context, func(context.Context)) {
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			t.Cleanup(cancel)
			return ctx, func(ctx context.Context) { <-ctx.Done() }
		}, context.DeadlineExceeded},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			ctx, end := row.ctx(t)
			prev := detectCapMethod
			t.Cleanup(func() { detectCapMethod = prev })
			detections := 0
			detectCapMethod = func(ctx context.Context, _ func(context.Context) bool) bwrapCapMethod {
				detections++
				end(ctx)
				return bwrapCapNone
			}
			ws := t.TempDir()
			_, err := (&Bwrap{}).Exec(ctx, Spec{
				Workspace:  ws,
				Cmd:        []string{"true"},
				WriteFiles: map[string][]byte{"marker": []byte("x")},
			})
			if detections != 1 {
				t.Fatalf("cap detection ran %d time(s), want 1: the ctx must end inside admission", detections)
			}
			if err == nil || !errors.Is(err, row.want) || !strings.HasPrefix(err.Error(), "sandbox: execution cancelled") || errors.Is(err, ErrBwrapNoCapMethod) {
				t.Errorf("err = %v, want the cancelled error wrapping %v", err, row.want)
			}
			if _, statErr := os.Stat(filepath.Join(ws, "marker")); !errors.Is(statErr, os.ErrNotExist) {
				t.Errorf("marker stat = %v, want ENOENT: WriteFiles landed in the caller's Workspace", statErr)
			}
		})
	}
}

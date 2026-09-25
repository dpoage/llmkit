//go:build integration

// The real-backend conformance legs: the SAME shared case table (see
// conformance_test.go) run through the actual Bwrap and CLI backends. Each
// leg auto-skips when its backend is unavailable — bwrap/userns for the
// Bwrap leg, a container runtime plus a pullable test image for the CLI leg
// (same ensureImage pattern as integration_test.go).
//
// Both backends are built with a small per-stream output cap so the
// stdout_truncated_over_cap case exercises the real truncation path.

package sandbox

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// newConformanceBwrap builds a Bwrap backend for the conformance leg,
// skipping when the backend is unusable on this host.
func newConformanceBwrap(t *testing.T) *Bwrap {
	t.Helper()
	s := newTestBwrap(t, WithMaxOutputBytes(64))
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// newConformanceCLI builds a CLI backend for the conformance leg, skipping
// when no runtime is detected or the image cannot be pulled/used.
func newConformanceCLI(t *testing.T) *CLI {
	t.Helper()
	rt, ok := Detect()
	if !ok {
		t.Skip("no container runtime detected; skipping CLI conformance leg")
	}
	base := []Option{
		WithRuntime(rt),
		WithImage(testImage),
		WithCPUs(1),
		WithMemoryMB(256),
		WithPidsLimit(128),
		WithTimeout(30 * time.Second),
		WithMaxOutputBytes(64),
	}
	s, err := NewCLI(base...)
	if err != nil {
		t.Skipf("NewCLI: %v", err)
	}
	ensureImage(t, s)
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// runConformance runs the whole shared case table through one backend —
// the real-backend legs call this once per constructed sandbox.
func runConformance(t *testing.T, sb Sandbox, repoDir string) {
	t.Helper()
	for _, tc := range conformanceCases() {
		t.Run(tc.name, func(t *testing.T) {
			runConformanceCase(t, sb, repoDir, tc)
		})
	}
}

func TestConformanceBwrap(t *testing.T) {
	s := newConformanceBwrap(t)
	runConformance(t, s, t.TempDir())
}

func TestConformanceCLI(t *testing.T) {
	s := newConformanceCLI(t)
	runConformance(t, s, t.TempDir())
}

// TestConformanceCLI_CallerDeadlineReapsContainer pins that a caller
// DEADLINE ends a CLI run like a caller cancel: Exec returns the cancelled
// error wrapping context.DeadlineExceeded, and the container is gone when
// Exec returns (killing the runtime client alone leaves it running). The
// command carries a unique token so the check ignores containers other
// runs on this host may own.
func TestConformanceCLI_CallerDeadlineReapsContainer(t *testing.T) {
	s := newConformanceCLI(t)
	token := "llmkit-deadline-reap-" + randToken()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, err := s.Exec(ctx, Spec{RepoDir: t.TempDir(), Cmd: []string{"sh", "-c", "sleep 60 # " + token}})
	// Checked first, so a leak is reported (and cleaned up) whatever Exec
	// returned.
	out, psErr := exec.Command(s.runtime, "ps", "-a", "--no-trunc", "--filter", "name=llmkit-", "--format", "{{.Names}} {{.Command}}").Output()
	if psErr != nil {
		t.Fatalf("%s ps: %v", s.runtime, psErr)
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, token) {
			name := strings.Fields(line)[0]
			_ = exec.Command(s.runtime, "rm", "-f", name).Run()
			t.Errorf("container %s outlived the caller's deadline (removed by the test): %q", name, line)
		}
	}
	if !errors.Is(err, context.DeadlineExceeded) || !strings.HasPrefix(err.Error(), "sandbox: execution cancelled") {
		t.Errorf("err = %v, want the cancelled error wrapping context.DeadlineExceeded", err)
	}
}

// TestIntegrationCLISetupCmdsOnDash pins that the SetupCmds wrapper script
// runs Spec.Cmd under dash, the /bin/sh of Debian and Ubuntu images (and of
// ubuntu-latest CI runners). dash's exec builtin takes no options: handing
// it a "--" end-of-options marker exits 127 there ("exec: --: not found")
// while busybox and bash accept it, so the alpine-based suite cannot see it.
func TestIntegrationCLISetupCmdsOnDash(t *testing.T) {
	s := newTestCLI(t, WithImage("docker.io/library/debian:bookworm-slim"))
	t.Cleanup(func() { _ = s.Close() })
	sh, err := s.Exec(context.Background(), Spec{RepoDir: t.TempDir(), Cmd: []string{"readlink", "-f", "/bin/sh"}})
	if err != nil || sh.ExitCode != 0 || strings.TrimSpace(sh.Stdout) != "/usr/bin/dash" {
		t.Fatalf("image /bin/sh = %q (exit %d, err %v), want /usr/bin/dash: the row needs a dash image", sh.Stdout, sh.ExitCode, err)
	}
	res, err := s.Exec(context.Background(), Spec{
		RepoDir:   t.TempDir(),
		SetupCmds: [][]string{{"true"}},
		Cmd:       []string{"echo", "hi"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 0 || res.Stdout != "hi\n" {
		t.Errorf("res = {ExitCode:%d Stdout:%q Stderr:%q}, want exit 0 and stdout \"hi\\n\"", res.ExitCode, res.Stdout, res.Stderr)
	}
}

//go:build integration

package sandbox

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestBwrapSystemdRunAcceptsRenderedCPUs (llmkit-bk8.1.45): WithCPUs values
// that made systemd-run exit 1 (float noise or more than two decimals of
// percent) run to exit 0 through the real systemd-run --user --scope wrapper.
// Its skips (bwrap unusable, no host sh, cap method not systemd-run) are
// probed without the value under test; NewBwrap refusing a row's value
// fails that row.
func TestBwrapSystemdRunAcceptsRenderedCPUs(t *testing.T) {
	if ok, reason := DetectBwrap(); !ok {
		t.Skipf("bwrap unavailable: %s", reason)
	}
	probe, err := NewBwrap()
	if err != nil {
		t.Fatalf("NewBwrap(): %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	m := detectCapMethod(ctx, probe.expandSupport.supported)
	cancel()
	_ = probe.Close()
	if m != bwrapCapSystemdRun {
		t.Skipf("host cap method is %v, not systemd-run", m)
	}
	shMounts, sh := shForTest(t)
	for _, cpus := range append([]float64{minCPUs}, p63FractionalCPUs...) {
		t.Run(strconv.FormatFloat(cpus, 'f', -1, 64), func(t *testing.T) {
			s, err := NewBwrap(WithCPUs(cpus), WithMemoryMB(256), WithPidsLimit(64), WithTimeout(30*time.Second))
			if err != nil {
				t.Fatalf("NewBwrap(WithCPUs(%v)): %v, want admitted", cpus, err)
			}
			t.Cleanup(func() { _ = s.Close() })
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			res, err := s.Exec(ctx, Spec{
				RepoDir:  t.TempDir(),
				Timeout:  20 * time.Second,
				ROMounts: shMounts,
				Cmd:      []string{sh, "-c", "exit 0"},
			})
			if err != nil {
				t.Fatalf("Exec: %v", err)
			}
			if res.ExitCode != 0 {
				t.Errorf("ExitCode = %d, want 0 (stderr %q)", res.ExitCode, res.Stderr)
			}
		})
	}
}

// TestCLIFloorCPUsCapsFinitely pins the floor row against the real
// container runtime: the smallest admitted WithCPUs value yields a finite
// cpu.max rather than "max 100000". Its skips (no
// runtime detected, test container not runnable) are probed on a backend
// built without WithCPUs; NewCLI refusing the floor fails the test.
func TestCLIFloorCPUsCapsFinitely(t *testing.T) {
	rt, ok := Detect()
	if !ok {
		t.Skip("no container runtime detected")
	}
	probe, err := NewCLI(WithRuntime(rt), WithImage(testImage))
	if err != nil {
		t.Fatalf("NewCLI(): %v", err)
	}
	t.Cleanup(func() { _ = probe.Close() })
	ensureImage(t, probe)
	s, err := NewCLI(WithRuntime(rt), WithImage(testImage), WithCPUs(minCPUs), WithMemoryMB(256), WithPidsLimit(128), WithTimeout(30*time.Second))
	if err != nil {
		t.Fatalf("NewCLI(WithCPUs(%v)): %v, want admitted", minCPUs, err)
	}
	t.Cleanup(func() { _ = s.Close() })
	res, err := s.Exec(context.Background(), Spec{
		RepoDir: t.TempDir(),
		Cmd:     []string{"cat", "/sys/fs/cgroup/cpu.max"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if got := strings.TrimSpace(res.Stdout); res.ExitCode != 0 || got != "1000 100000" {
		t.Errorf("exit %d, cpu.max = %q, want exit 0 and %q", res.ExitCode, got, "1000 100000")
	}
}

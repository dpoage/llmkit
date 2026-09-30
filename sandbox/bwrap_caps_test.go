package sandbox

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestSystemdRunWrapArgs(t *testing.T) {
	args := systemdRunWrapArgs("/usr/bin/bwrap", []string{"--unshare-all", "true"}, 1.5, 512, 128)

	if args[0] != "systemd-run" {
		t.Fatalf("expected systemd-run as argv[0], got %q", args[0])
	}
	mustContainSeq(t, args, "--user")
	mustContainSeq(t, args, "--scope")
	mustContainSeq(t, args, "-p", "MemoryMax=512M")
	mustContainSeq(t, args, "-p", "CPUQuota=150%")
	mustContainSeq(t, args, "-p", "TasksMax=128")
	// llmkit-bk8.1.16: the wrapper must pass --expand-environment=no so the
	// untrusted argv reaches bwrap byte-for-byte instead of having ${NAME}
	// expanded from the HOST environment by systemd-run.
	mustContainSeq(t, args, "--expand-environment=no")
	mustContainSeq(t, args, "--", "/usr/bin/bwrap")

	// The bwrap argv must be preserved verbatim at the tail.
	tail := args[len(args)-2:]
	if !slices.Equal(tail, []string{"--unshare-all", "true"}) {
		t.Errorf("bwrap argv not preserved at tail: %q", tail)
	}
}

// systemdCPUQuotaPattern is systemd's measured CPUQuota= grammar on this
// lane's reference host (systemd 262): a percentage with at most two
// decimals ("33.3%" and "7.25%" parse; "0.333%" and "7.125%" are Invalid
// argument). systemdMaxCPUQuotaPercent is the measured ceiling
// (21474836.47% accepted, 21474836.48% "Numerical result out of range").
var systemdCPUQuotaPattern = regexp.MustCompile(`^[0-9]+(\.[0-9]{1,2})?%$`)

const systemdMaxCPUQuotaPercent = 21474836.47

// renderedCPUQuota returns the CPUQuota= property systemdRunWrapArgs
// renders for cpus.
func renderedCPUQuota(t *testing.T, cpus float64) string {
	t.Helper()
	args := systemdRunWrapArgs("/usr/bin/bwrap", []string{"true"}, cpus, 0, 0)
	for i, a := range args {
		if a == "-p" && i+1 < len(args) && strings.HasPrefix(args[i+1], "CPUQuota=") {
			return strings.TrimPrefix(args[i+1], "CPUQuota=")
		}
	}
	t.Fatalf("no CPUQuota property rendered for cpus=%v: %q", cpus, args)
	return ""
}

// TestSystemdRunWrapArgsRendersAcceptedCPUQuota (llmkit-bk8.1.45): every
// WithCPUs value a constructor admits renders a CPUQuota systemd accepts.
// The rows are float noise (1.1*100 = 110.00000000000001) and more than
// two decimals (0.333), plus the floor and the ceiling. The strings are
// asserted directly because CI has no systemd user session; the
// integration row runs them for real.
func TestSystemdRunWrapArgsRendersAcceptedCPUQuota(t *testing.T) {
	rows := []struct {
		cpus float64
		want string
	}{
		{minCPUs, "1%"},
		{0.07, "7%"},
		{0.29, "29%"},
		{0.333, "33.3%"},
		{0.0123, "1.23%"},
		{1.1, "110%"},
		{1.5, "150%"},
		{2.3, "230%"},
		{maxCPUs, "21474836.47%"},
	}
	for _, r := range rows {
		got := renderedCPUQuota(t, r.cpus)
		if got != r.want {
			t.Errorf("WithCPUs(%v) renders CPUQuota=%s, want %s", r.cpus, got, r.want)
		}
	}
}

// TestSystemdRunWrapArgsCPUQuotaProperty sweeps admitted values and checks
// the two rendered-string rules the measured systemd enforces plus the
// rounding contract: the rendered cap is within one render step
// (0.0001 CPU) of the request, so it is never looser than asked by more
// than that.
func TestSystemdRunWrapArgsCPUQuotaProperty(t *testing.T) {
	var values []float64
	for k := 10; k <= 5000; k++ { // 0.010 .. 5.000 in 0.001 steps
		values = append(values, float64(k)/1000)
	}
	values = append(values, 0.0101, 0.0149, 0.0151, 1.00005, 1.00004, 99.9999, 1e3, 1e5, maxCPUs)
	for _, cpus := range values {
		got := renderedCPUQuota(t, cpus)
		if !systemdCPUQuotaPattern.MatchString(got) {
			t.Errorf("WithCPUs(%v) renders CPUQuota=%s, which systemd rejects (more than two decimals of percent)", cpus, got)
			continue
		}
		pct, err := strconv.ParseFloat(strings.TrimSuffix(got, "%"), 64)
		if err != nil {
			t.Fatalf("parse %q: %v", got, err)
		}
		if pct > systemdMaxCPUQuotaPercent {
			t.Errorf("WithCPUs(%v) renders CPUQuota=%s, above systemd's %v%% maximum", cpus, got, systemdMaxCPUQuotaPercent)
		}
		if diff := pct/100 - cpus; diff > 0.0001 || diff < -0.0001 {
			t.Errorf("WithCPUs(%v) renders %s: %v CPUs, off by %v (more than one 0.0001 step)", cpus, got, pct/100, diff)
		}
	}
}

// TestCgroupV2LimitsRendersRequestedCPUQuota pins the cgroup v2 render of
// the same values: cpu.max quota is the request rounded to nearest
// microsecond of the 100000us period, not truncated (0.29*100000 is
// 28999.999999999996 in float64, which truncation would take to 28999).
func TestCgroupV2LimitsRendersRequestedCPUQuota(t *testing.T) {
	rows := []struct {
		cpus float64
		want string
	}{
		{minCPUs, "1000 100000"},
		{0.07, "7000 100000"},
		{0.29, "29000 100000"},
		{0.333, "33300 100000"},
		{1.1, "110000 100000"},
		{2.3, "230000 100000"},
		{maxCPUs, "21474836470 100000"},
	}
	for _, r := range rows {
		_, got, _, _, cpuOK, _ := cgroupV2Limits(r.cpus, 0, 0)
		if !cpuOK || got != r.want {
			t.Errorf("cgroupV2Limits(%v) cpu.max = %q (ok=%v), want %q", r.cpus, got, cpuOK, r.want)
		}
	}
}

func TestSystemdRunWrapArgsOmitsUnsetLimits(t *testing.T) {
	args := systemdRunWrapArgs("/usr/bin/bwrap", []string{"true"}, 0, 0, 0)
	for _, prop := range []string{"MemoryMax", "CPUQuota", "TasksMax"} {
		for _, a := range args {
			if strings.HasPrefix(a, prop) {
				t.Errorf("expected %s to be omitted when unset; args=%q", prop, args)
			}
		}
	}
}

func TestCgroupV2Limits(t *testing.T) {
	memory, cpuMax, pids, memOK, cpuOK, pidsOK := cgroupV2Limits(1.5, 512, 128)
	if !memOK || memory != "536870912" {
		t.Errorf("memory = %q ok=%v, want 536870912 bytes", memory, memOK)
	}
	if !cpuOK || cpuMax != "150000 100000" {
		t.Errorf("cpuMax = %q ok=%v, want '150000 100000'", cpuMax, cpuOK)
	}
	if !pidsOK || pids != "128" {
		t.Errorf("pids = %q ok=%v, want 128", pids, pidsOK)
	}
}

func TestCgroupV2LimitsOmitsUnset(t *testing.T) {
	_, _, _, memOK, cpuOK, pidsOK := cgroupV2Limits(0, 0, 0)
	if memOK || cpuOK || pidsOK {
		t.Errorf("expected all limits omitted for zero inputs, got mem=%v cpu=%v pids=%v", memOK, cpuOK, pidsOK)
	}
}

func TestDelegatedCgroupV2Dir(t *testing.T) {
	// Best-effort: only assert the function does not panic and, when it does
	// report a dir, that dir path is non-empty and rooted under cgroupV2Root.
	// The real availability varies by host (CI containers commonly run
	// without a delegated cgroup v2 subtree), so this cannot assert a fixed
	// outcome.
	dir, ok := delegatedCgroupV2Dir()
	if ok && !strings.HasPrefix(dir, cgroupV2Root) {
		t.Errorf("delegated dir %q must be rooted under %q", dir, cgroupV2Root)
	}
}

func TestSystemdRunUserAvailableDoesNotHang(t *testing.T) {
	// Whatever the host's real answer is, the probe must return within its
	// own timeout instead of blocking the test.
	ctx, cancel := context.WithTimeout(context.Background(), systemdRunProbeTimeout*2)
	defer cancel()
	done := make(chan struct{})
	go func() {
		systemdRunUserAvailable(ctx)
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("systemdRunUserAvailable did not return within 2x its own timeout")
	}
}

func TestDetectBwrapCapMethodNeverPanics(t *testing.T) {
	// detectBwrapCapMethod must always resolve to one of the three defined
	// methods and never block indefinitely, regardless of host capability.
	ctx, cancel := context.WithTimeout(context.Background(), systemdRunProbeTimeout*2)
	defer cancel()
	method := detectBwrapCapMethod(ctx, systemdRunExpandSupport)
	if method != bwrapCapNone && method != bwrapCapSystemdRun && method != bwrapCapCgroupV2 {
		t.Errorf("unexpected cap method %v", method)
	}
}

// stubCapMethod forces detectCapMethod to report m for the test's duration.
func stubCapMethod(t *testing.T, m bwrapCapMethod) {
	t.Helper()
	prev := detectCapMethod
	detectCapMethod = func(context.Context, func(context.Context) bool) bwrapCapMethod { return m }
	t.Cleanup(func() { detectCapMethod = prev })
}

// TestDetectBwrapCapMethodFailsClosedWithoutExpandSupport pins the 1.16
// fail-closed path: when the host's systemd-run does NOT support
// --expand-environment=no (the bk8.1.16 fix flag), the systemd-run method
// must never resolve — detection degrades to the cgroup v2 subtree or none.
// Enforcement may be absent on such a host, never "enforcement plus a
// host-env leak". systemd-run itself is stubbed AVAILABLE, so only the
// support answer can rule the method out; the supported row proves the
// stub reaches detection.
func TestDetectBwrapCapMethodFailsClosedWithoutExpandSupport(t *testing.T) {
	prev := systemdRunUserAvailable
	systemdRunUserAvailable = func(context.Context) bool { return true }
	t.Cleanup(func() { systemdRunUserAvailable = prev })

	if got := detectBwrapCapMethod(context.Background(), func(context.Context) bool { return true }); got != bwrapCapSystemdRun {
		t.Fatalf("with expand support, detectBwrapCapMethod = %d, want systemd-run (the stubs must reach detection)", got)
	}
	switch got := detectBwrapCapMethod(context.Background(), func(context.Context) bool { return false }); got {
	case bwrapCapCgroupV2, bwrapCapNone:
		// ok — degraded, not leaked
	default:
		t.Fatalf("detectBwrapCapMethod = %d, want cgroup v2 or none when the expand fix is unsupported", got)
	}
}

// TestSystemdRunExpandSupportProbeParsesVersion pins the support probe: a
// "systemd <major>" first line is accepted at >= 254, and every other
// answer — older systemd, malformed or empty output, a failing or missing
// systemd-run — fails closed.
func TestSystemdRunExpandSupportProbeParsesVersion(t *testing.T) {
	prev := systemdRunVersion
	t.Cleanup(func() { systemdRunVersion = prev })
	rows := []struct {
		name string
		out  string
		err  error
		want bool
	}{
		{"254 is the first supported", "systemd 254 (254.5-1)\n+PAM +AUDIT\n", nil, true},
		{"newer systemd", "systemd 261 (261.2-1-arch)\n", nil, true},
		{"253 is too old", "systemd 253 (253.1-1)\n", nil, false},
		{"rc suffix is not an integer", "systemd 254rc1 (254-rc1)\n", nil, false},
		{"garbage", "hello world\n", nil, false},
		{"empty", "", nil, false},
		{"nonzero exit", "systemd 261 (261.2)\n", errors.New("exit status 1"), false},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			systemdRunVersion = func(context.Context) ([]byte, error) { return []byte(row.out), row.err }
			if got := systemdRunExpandSupport(context.Background()); got != row.want {
				t.Fatalf("systemdRunExpandSupport(%q, %v) = %t, want %t", row.out, row.err, got, row.want)
			}
		})
	}
	t.Run("unreachable systemd-run", func(t *testing.T) {
		systemdRunVersion = prev
		t.Setenv("PATH", t.TempDir())
		if systemdRunExpandSupport(context.Background()) {
			t.Fatal("probe must fail closed when systemd-run is unreachable")
		}
	})
}

// TestExpandSupportCacheProbesOncePerInstance pins the per-instance cache:
// the version probe runs once for any number of Execs, but an answer the
// caller's ended ctx cut short is not remembered.
func TestExpandSupportCacheProbesOncePerInstance(t *testing.T) {
	prev := systemdRunVersion
	t.Cleanup(func() { systemdRunVersion = prev })
	calls := 0
	systemdRunVersion = func(ctx context.Context) ([]byte, error) {
		calls++
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return []byte("systemd 261 (261.2)\n"), nil
	}

	var c expandSupportCache
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if c.supported(cancelled) {
		t.Fatal("a probe cut short by the caller must answer no")
	}
	for range 3 {
		if !c.supported(context.Background()) {
			t.Fatal("supported = false, want the probed yes")
		}
	}
	if calls != 2 {
		t.Fatalf("version probe ran %d times, want 2 (the cancelled attempt, then once for every live call)", calls)
	}
}

// TestBwrapExec_NoCapMethod_IsSentinel pins the exported contract: when no
// enforcement mechanism exists and uncapped runs are not allowed, Exec's error
// matches ErrBwrapNoCapMethod via errors.Is, so callers can attach their own
// remediation without string matching.
func TestBwrapExec_NoCapMethod_IsSentinel(t *testing.T) {
	stubCapMethod(t, bwrapCapNone)

	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "f.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Bwrap{bwrapPath: "/nonexistent/bwrap", defaults: defaults{defaultTimeout: time.Second}}
	_, err := s.Exec(context.Background(), Spec{RepoDir: repo, Cmd: []string{"true"}})
	if !errors.Is(err, ErrBwrapNoCapMethod) {
		t.Fatalf("Exec error = %v, want errors.Is ErrBwrapNoCapMethod", err)
	}

	s.capPolicy = CapBestEffort
	_, err = s.Exec(context.Background(), Spec{RepoDir: repo, Cmd: []string{"true"}})
	if errors.Is(err, ErrBwrapNoCapMethod) {
		t.Fatalf("with allow-uncapped, Exec must not return ErrBwrapNoCapMethod; got %v", err)
	}
}

// TestDelegatedCgroupV2DirRefusesWritableScopeWithoutLimitFiles pins bead
// llmkit-bk8.1.19: a scope this process can mkdir in is not a delegation. A
// child of a scope whose parent enables no controllers has no memory.max /
// cpu.max / pids.max, so reporting the scope would make Describe say
// "enforced" while every Exec died at its first limit write. The fake root
// is a plain writable directory tree — exactly the "mkdir succeeds, no limit
// files appear" shape of an undelegated scope.
func TestDelegatedCgroupV2DirRefusesWritableScopeWithoutLimitFiles(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "cgroup.controllers"), []byte("memory pids\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "scope"), 0o755); err != nil {
		t.Fatal(err)
	}
	self := filepath.Join(t.TempDir(), "cgroup")
	if err := os.WriteFile(self, []byte("0::/scope\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if dir, ok := delegatedCgroupV2DirAt(root, self); ok {
		t.Fatalf("delegatedCgroupV2DirAt = (%q, true) for a writable scope whose children get no limit files; want false", dir)
	}
	if entries, _ := os.ReadDir(filepath.Join(root, "scope")); len(entries) != 0 {
		t.Errorf("the probe left %d entries in the scope, want none", len(entries))
	}
}

package sandbox

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestDetectBwrapReasonsAreActionable(t *testing.T) {
	ok, reason := DetectBwrap()
	if runtime.GOOS != "linux" {
		if ok {
			t.Fatalf("DetectBwrap must never report ok on %s", runtime.GOOS)
		}
		if reason == "" {
			t.Error("expected a non-empty reason on a non-Linux host")
		}
		return
	}
	// On Linux the outcome depends on whether this host actually has bwrap
	// and usable userns; either way a false result must explain why.
	if !ok && reason == "" {
		t.Error("DetectBwrap reported unavailable with no reason")
	}
}

// TestDetectBwrapTrueWhenUsable is the regression the reviewer's finding
// exposed: TestDetectBwrapReasonsAreActionable above accepts ok==false
// unconditionally on Linux ("either way a false result must explain why"),
// which let a probeBwrapUserns bug that made EVERY host report unavailable
// ship unnoticed (the probe ran `bwrap --unshare-all --die-with-parent
// /bin/true` against bwrap's empty tmpfs root, where /bin/true never
// exists). This test independently confirms userns is usable (bwrap on
// PATH + the kernel's own /proc/sys/user/max_user_namespaces > 0) and then
// asserts DetectBwrap agrees — so a probe that spuriously reports
// unavailable on a genuinely usable host fails loudly instead of blending
// into "either way" silence.
func TestDetectBwrapTrueWhenUsable(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("bwrap is Linux-only")
	}
	if _, err := exec.LookPath("bwrap"); err != nil {
		t.Skip("bwrap not found on PATH")
	}
	raw, err := os.ReadFile("/proc/sys/user/max_user_namespaces")
	if err != nil {
		t.Skipf("cannot read /proc/sys/user/max_user_namespaces: %v", err)
	}
	max, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || max <= 0 {
		t.Skip("unprivileged user namespaces are disabled on this host (max_user_namespaces <= 0)")
	}
	ok, reason := DetectBwrap()
	if !ok {
		t.Fatalf("DetectBwrap reported unavailable on a host with bwrap on PATH and max_user_namespaces=%d: %s", max, reason)
	}
}

func TestNewBwrapFailsFastWhenUnavailable(t *testing.T) {
	ok, _ := DetectBwrap()
	if ok {
		t.Skip("bwrap is available on this host; NewBwrap success path is covered by the integration suite")
	}
	if _, err := NewBwrap(); err == nil {
		t.Fatal("expected NewBwrap to fail when DetectBwrap reports unavailable")
	}
}

// TestBwrapOptionsApplyDefaults pins the option→defaults mapping through
// the REAL constructor: NewBwrap must land every shared knob on its
// embedded defaults, so deleting any single application in
// options.applyDefaults (or any base value in baseDefaults) fails here
// instead of silently dropping the knob — a dropped WithIdleTimeout, for
// instance, turns idle kills into unnoticed absolute-timeout kills.
func TestBwrapOptionsApplyDefaults(t *testing.T) {
	if ok, _ := DetectBwrap(); !ok {
		t.Skip("bwrap unavailable; the application pin needs the real NewBwrap")
	}
	s, err := NewBwrap(
		WithCPUs(3), WithMemoryMB(1024), WithPidsLimit(64),
		WithNetwork(NetworkHost), WithCapPolicy(CapBestEffort),
		WithScratchSizeMB(256), WithWorkspaceGrowthCeilingMB(1024),
		WithIdleTimeout(30*time.Second), WithTimeout(7*time.Minute),
		WithMaxOutputBytes(4096),
	)
	if err != nil {
		t.Fatalf("NewBwrap: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	if s.defaultCPUs != 3 || s.defaultMemory != 1024 || s.pidsLimit != 64 {
		t.Errorf("defaultCPUs/defaultMemory/pidsLimit = %v %v %v, want 3 1024 64", s.defaultCPUs, s.defaultMemory, s.pidsLimit)
	}
	if s.defaultNetwork != NetworkHost {
		t.Errorf("defaultNetwork = %q, want %q", s.defaultNetwork, NetworkHost)
	}
	if s.capPolicy != CapBestEffort {
		t.Errorf("capPolicy = %v, want CapBestEffort", s.capPolicy)
	}
	if s.defaultScratchSizeMB != 256 {
		t.Errorf("defaultScratchSizeMB = %d, want 256", s.defaultScratchSizeMB)
	}
	if want := int64(1024) * 1024 * 1024; s.defaultGrowthCeilingBytes != want {
		t.Errorf("defaultGrowthCeilingBytes = %d, want %d (1024 MB in bytes)", s.defaultGrowthCeilingBytes, want)
	}
	if s.defaultIdleTimeout != 30*time.Second {
		t.Errorf("defaultIdleTimeout = %v, want 30s", s.defaultIdleTimeout)
	}
	if s.defaultTimeout != 7*time.Minute {
		t.Errorf("defaultTimeout = %v, want 7m", s.defaultTimeout)
	}
	if s.maxOutputBytes != 4096 {
		t.Errorf("maxOutputBytes = %d, want 4096", s.maxOutputBytes)
	}

	// The zero-option leg pins baseDefaults itself: every shared knob must
	// start at the documented out-of-the-box posture.
	base, err := NewBwrap()
	if err != nil {
		t.Fatalf("NewBwrap(): %v", err)
	}
	t.Cleanup(func() { _ = base.Close() })
	if want := baseDefaults(); base.defaults != want {
		t.Errorf("zero-option defaults = %+v, want baseDefaults %+v", base.defaults, want)
	}
}

// TestBwrapConstructorRefusesCLIOnlyOptions pins the one-Option-type
// contract from the other side: NewBwrap must reject every CLI-only option
// with an error NAMING it, instead of silently ignoring the knob.
func TestBwrapConstructorRefusesCLIOnlyOptions(t *testing.T) {
	if ok, _ := DetectBwrap(); !ok {
		t.Skip("bwrap unavailable; constructor refusal is host-independent but DetectBwrap runs first")
	}
	for _, opt := range []Option{WithRuntime("podman"), WithImage("img")} {
		_, err := NewBwrap(opt)
		if err == nil {
			t.Errorf("NewBwrap accepted a CLI-only option without error")
			continue
		}
		if !strings.Contains(err.Error(), "option With") || !strings.Contains(err.Error(), "bwrap") {
			t.Errorf("error %v must name the option and the bwrap backend", err)
		}
	}
}

func TestBwrapResolveParamsRejectsBadNetwork(t *testing.T) {
	s := &Bwrap{defaults: defaults{defaultNetwork: NetworkNone}}
	_, err := s.resolveBwrapParams(Spec{Cmd: []string{"true"}, Network: NetworkBridge})
	var ue *UnsupportedSpecError
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *UnsupportedSpecError", err)
	}
	if ue.Backend != "bwrap" || ue.Field != "Network" || ue.Value != string(NetworkBridge) {
		t.Errorf("UnsupportedSpecError = %+v, want backend=bwrap field=Network value=bridge", ue)
	}
	p, err := s.resolveBwrapParams(Spec{Cmd: []string{"true"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.network != NetworkNone {
		t.Errorf("expected default network NetworkNone, got %q", p.network)
	}
}

// TestBwrapConstructorRejectsBridgeNetworkDefault pins the construction-time
// half of the Network row: WithNetwork(NetworkBridge) — a mode bwrap could
// NEVER honor on any host — fails NewBwrap with an error naming the mode,
// not a constructor that silently stores it and only fails per-Exec. This
// runs before DetectBwrap, so it is host-independent.
func TestBwrapConstructorRejectsBridgeNetworkDefault(t *testing.T) {
	_, err := NewBwrap(WithNetwork(NetworkBridge))
	var ue *UnsupportedSpecError
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *UnsupportedSpecError naming the network mode", err)
	}
	if ue.Backend != "bwrap" || ue.Field != "Network" {
		t.Errorf("UnsupportedSpecError = %+v, want backend=bwrap field=Network", ue)
	}
}

func TestBwrapExecRejectsEmptyCmd(t *testing.T) {
	s := &Bwrap{}
	if _, err := s.Exec(nil, Spec{}); err == nil { //nolint:staticcheck // nil ctx is fine; Exec must fail before using it.
		t.Error("expected Exec to reject an empty Spec.Cmd before touching ctx")
	}
}

func TestBwrapExecRejectsMountCollisionBeforeAnyWork(t *testing.T) {
	s := &Bwrap{}
	_, err := s.Exec(nil, Spec{ //nolint:staticcheck
		Cmd:      []string{"true"},
		ROMounts: []ROMount{{HostPath: "/host", ContainerPath: "/usr"}},
	})
	if err == nil {
		t.Error("expected Exec to reject a mount colliding with the fixed allowlist")
	}
}

// TestBwrapExecAsksTheInstanceExpandSupportCache pins the Exec→probe
// wiring: the cap-method detection Exec runs is handed this instance's
// cached systemd-run --expand-environment=no answer. A pre-254 systemd
// answers no (so the host-env-expanding systemd-run is never chosen), and
// the version probe runs once for the instance, not once per Exec.
func TestBwrapExecAsksTheInstanceExpandSupportCache(t *testing.T) {
	prevVersion := systemdRunVersion
	t.Cleanup(func() { systemdRunVersion = prevVersion })
	probes := 0
	systemdRunVersion = func(context.Context) ([]byte, error) {
		probes++
		return []byte("systemd 253 (253.1)\n"), nil
	}
	prevDetect := detectCapMethod
	t.Cleanup(func() { detectCapMethod = prevDetect })
	var answers []bool
	detectCapMethod = func(ctx context.Context, expandSupport func(context.Context) bool) bwrapCapMethod {
		answers = append(answers, expandSupport(ctx))
		return bwrapCapNone
	}

	s := &Bwrap{}
	for range 2 {
		if _, err := s.Exec(context.Background(), Spec{Workspace: t.TempDir(), Cmd: []string{"true"}}); !errors.Is(err, ErrBwrapNoCapMethod) {
			t.Fatalf("Exec error = %v, want ErrBwrapNoCapMethod from the stubbed detection", err)
		}
	}
	if !slices.Equal(answers, []bool{false, false}) || probes != 1 {
		t.Fatalf("expandSupport answers = %v with %d version probe(s), want [false false] from one probe (systemd 253 lacks --expand-environment=no)", answers, probes)
	}
}

// --- /proc-based CPU sampling (activeFallback) --------------------------

func TestProcTreeCPUTicksSelf(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("procfs is Linux-only")
	}
	ticks, ok := procTreeCPUTicks(os.Getpid())
	if !ok {
		t.Fatal("expected to read this process's own /proc/<pid>/stat")
	}
	if ticks < 0 {
		t.Errorf("ticks = %d, want >= 0", ticks)
	}
}

func TestProcTreeCPUTicksUnknownPid(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("procfs is Linux-only")
	}
	// A very large, almost certainly unused PID should not resolve.
	if _, ok := procTreeCPUTicks(1 << 30); ok {
		t.Error("expected ok=false for a pid that does not exist")
	}
}

func TestProcStatTicksParsesOwnProcess(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("procfs is Linux-only")
	}
	// Exercise the real parser against this process's own stat line; comm
	// rarely has spaces in the test binary's name, but the parser must not
	// depend on that — it locates fields from the trailing ')' regardless.
	if _, ok := procStatTicks(os.Getpid()); !ok {
		t.Fatal("expected to parse this process's own /proc/<pid>/stat")
	}
}

func TestSplitFields(t *testing.T) {
	got := splitFields("  1  2\t3\n")
	want := []string{"1", "2", "3"}
	if len(got) != len(want) {
		t.Fatalf("splitFields = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("splitFields[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestLastIndexByte(t *testing.T) {
	if got := lastIndexByte("a)b)c", ')'); got != 3 {
		t.Errorf("lastIndexByte = %d, want 3", got)
	}
	if got := lastIndexByte("abc", ')'); got != -1 {
		t.Errorf("lastIndexByte = %d, want -1", got)
	}
}

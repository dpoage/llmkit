package sandbox

import (
	"errors"
	"os"
	"os/exec"
	"runtime"
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

func TestBwrapOptionsApplyDefaults(t *testing.T) {
	o := newOptions([]Option{
		WithCPUs(3), WithMemoryMB(1024), WithPidsLimit(64),
		WithNetwork(NetworkHost), WithCapPolicy(CapBestEffort),
		WithScratchSizeMB(256), WithWorkspaceGrowthCeilingMB(1024),
		WithIdleTimeout(30 * time.Second),
	})
	s := &Bwrap{}
	if o.has("WithCPUs") {
		s.defaultCPUs = o.cpus
	}
	if o.has("WithMemoryMB") {
		s.defaultMemory = o.memoryMB
	}
	if o.has("WithPidsLimit") {
		s.pidsLimit = o.pidsLimit
	}
	if o.has("WithNetwork") {
		s.defaultNetwork = o.network
	}
	if o.has("WithCapPolicy") {
		s.capPolicy = o.capPolicy
	}
	if o.has("WithScratchSizeMB") {
		s.defaultScratchSizeMB = o.scratchSizeMB
	}
	if o.has("WithWorkspaceGrowthCeilingMB") {
		s.defaultGrowthCeilingBytes = int64(o.growthCeilingMB) * 1024 * 1024
	}
	if o.has("WithIdleTimeout") {
		s.defaultIdleTimeout = o.idleTimeout
	}

	cpus, mem, pids := s.Limits()
	if cpus != 3 || mem != 1024 || pids != 64 {
		t.Errorf("Limits() = %v %v %v, want 3 1024 64", cpus, mem, pids)
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
	s := &Bwrap{defaultNetwork: NetworkNone}
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

// TestBwrapExec_RefusesImage pins the Spec honesty table's Image row for the
// bwrap backend: a non-empty Image is a typed Exec refusal, never a silent
// ignore.
func TestBwrapExec_RefusesImage(t *testing.T) {
	s := &Bwrap{}
	_, err := s.resolveBwrapParams(Spec{Cmd: []string{"true"}, Image: "quay.io/example/img:latest"})
	var ue *UnsupportedSpecError
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *UnsupportedSpecError", err)
	}
	if ue.Backend != "bwrap" || ue.Field != "Image" {
		t.Errorf("UnsupportedSpecError = %+v, want backend=bwrap field=Image", ue)
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

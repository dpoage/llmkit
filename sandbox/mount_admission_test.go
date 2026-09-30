package sandbox

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
)

// mountSpec is a well-formed Spec carrying the given read-only mounts.
func mountSpec(ro ...ROMount) Spec {
	s := baseValidSpec()
	s.ROMounts = ro
	return s
}

// TestValidateSpecMountAdmission pins that, after filepath.Clean, the Spec's
// ContainerPaths and the backend's rendered mounts are pairwise distinct and
// none is "/" or a destination the backend renders itself, and the CLI
// refuses what its `-v host:ctr:opts` syntax cannot express — each with the
// error class the refusal belongs to.
func TestValidateSpecMountAdmission(t *testing.T) {
	toolchain := []ROMount{{HostPath: "/host/node", ContainerPath: hostToolchainMountRoot + "/node", Shared: true}}
	type want int
	const (
		admit want = iota
		invalid
		unsupported
	)
	rows := []struct {
		name     string
		backend  string
		spec     Spec
		rendered []ROMount
		want     want
		field    string // expected Field of the refusal
	}{
		{"trailing-slash dup, cli", backendCLI, mountSpec(ROMount{HostPath: "/h/a", ContainerPath: "/opt/a"}, ROMount{HostPath: "/h/b", ContainerPath: "/opt/a/"}), nil, invalid, "ROMounts"},
		{"dot-spelling dup, bwrap", backendBwrap, mountSpec(ROMount{HostPath: "/h/a", ContainerPath: "/opt/a"}, ROMount{HostPath: "/h/b", ContainerPath: "/opt/./a"}), nil, invalid, "ROMounts"},
		{"dup across ro/rw spelled differently, mock", backendMock, func() Spec {
			s := mountSpec(ROMount{HostPath: "/h/a", ContainerPath: "/opt/a"})
			s.RWMounts = []ROMount{{HostPath: "/h/b", ContainerPath: "/opt//a/"}}
			return s
		}(), nil, invalid, "RWMounts"},
		{"root, cli", backendCLI, mountSpec(ROMount{HostPath: "/h", ContainerPath: "/"}), nil, invalid, "ROMounts"},
		{"root spelled //, mock", backendMock, mountSpec(ROMount{HostPath: "/h", ContainerPath: "//"}), nil, invalid, "ROMounts"},
		{"workspace with slash, cli", backendCLI, mountSpec(ROMount{HostPath: "/h", ContainerPath: "/workspace/"}), nil, invalid, "ROMounts"},
		{"workspace, bwrap", backendBwrap, mountSpec(ROMount{HostPath: "/h", ContainerPath: WorkspaceMount}), nil, invalid, "ROMounts"},
		{"workspace, mock", backendMock, mountSpec(ROMount{HostPath: "/h", ContainerPath: "/workspace/./"}), nil, invalid, "ROMounts"},
		{"cli /tmp", backendCLI, mountSpec(ROMount{HostPath: "/h", ContainerPath: "/tmp"}), nil, unsupported, "ROMounts"},
		{"cli /tmp/ in RWMounts", backendCLI, func() Spec {
			s := baseValidSpec()
			s.RWMounts = []ROMount{{HostPath: "/h", ContainerPath: "/tmp/"}}
			return s
		}(), nil, unsupported, "RWMounts"},
		{"mock /tmp admitted (mock renders nothing)", backendMock, mountSpec(ROMount{HostPath: "/h", ContainerPath: "/tmp"}), nil, admit, ""},
		{"toolchain collision, cli", backendCLI, mountSpec(ROMount{HostPath: "/h", ContainerPath: hostToolchainMountRoot + "/node"}), toolchain, unsupported, "ROMounts"},
		{"toolchain collision spelled with slash, cli RW", backendCLI, func() Spec {
			s := baseValidSpec()
			s.RWMounts = []ROMount{{HostPath: "/h", ContainerPath: hostToolchainMountRoot + "/node/"}}
			return s
		}(), toolchain, unsupported, "RWMounts"},
		{"toolchain collision, bwrap", backendBwrap, mountSpec(ROMount{HostPath: "/h", ContainerPath: hostToolchainMountRoot + "/node"}), toolchain, unsupported, "ROMounts"},
		{"no collision with toolchain, cli", backendCLI, mountSpec(ROMount{HostPath: "/h", ContainerPath: hostToolchainMountRoot + "/other"}), toolchain, admit, ""},
		{"host with ':' in HostPath, cli", backendCLI, mountSpec(ROMount{HostPath: "/h/a:b", ContainerPath: "/c"}), nil, unsupported, "ROMounts"},
		{"':' in ContainerPath, cli RW", backendCLI, func() Spec {
			s := baseValidSpec()
			s.RWMounts = []ROMount{{HostPath: "/h", ContainerPath: "/c:d"}}
			return s
		}(), nil, unsupported, "RWMounts"},
		{"':' in HostPath admitted on bwrap", backendBwrap, mountSpec(ROMount{HostPath: "/h/a:b", ContainerPath: "/c"}), nil, admit, ""},
		{"':' in HostPath admitted on mock", backendMock, mountSpec(ROMount{HostPath: "/h/a:b", ContainerPath: "/c"}), nil, admit, ""},
		{"comma and space are expressible on cli", backendCLI, mountSpec(ROMount{HostPath: "/h/a,b c", ContainerPath: "/c,d e"}), nil, admit, ""},
		{"clean distinct paths, cli", backendCLI, mountSpec(ROMount{HostPath: "/h/a", ContainerPath: "/opt/a"}, ROMount{HostPath: "/h/b", ContainerPath: "/opt/b"}), toolchain, admit, ""},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			err := validateSpec(r.backend, r.spec, r.rendered)
			var inv *InvalidSpecError
			var uns *UnsupportedSpecError
			switch r.want {
			case admit:
				if err != nil {
					t.Fatalf("validateSpec = %v, want nil", err)
				}
			case invalid:
				if !errors.As(err, &inv) || inv.Field != r.field {
					t.Fatalf("validateSpec = %v, want InvalidSpecError{Field:%s}", err, r.field)
				}
			case unsupported:
				if !errors.As(err, &uns) || uns.Backend != r.backend || uns.Field != r.field {
					t.Fatalf("validateSpec = %v, want UnsupportedSpecError{Backend:%s Field:%s}", err, r.backend, r.field)
				}
			}
		})
	}
}

// TestNewCLIRefusesColonToolchainMount: a host-toolchain mount the CLI's -v
// syntax cannot express is refused at construction, naming the option —
// before the runtime lookup (the runtime named here does not exist).
func TestNewCLIRefusesColonToolchainMount(t *testing.T) {
	for _, m := range []ROMount{
		{HostPath: "/host/a:b", ContainerPath: hostToolchainMountRoot + "/a", Shared: true},
		{HostPath: "/host/a", ContainerPath: hostToolchainMountRoot + "/a:b", Shared: true},
	} {
		_, err := NewCLI(WithRuntime("llmkit-no-such-runtime"), WithImage("img"),
			WithHostToolchains(ToolchainResolution{mounts: []ROMount{m}}))
		if err == nil || !strings.Contains(err.Error(), "WithHostToolchains") || strings.Contains(err.Error(), "not found on PATH") {
			t.Errorf("NewCLI(%+v) err = %v, want a WithHostToolchains ':' refusal", m, err)
		}
	}
	// The same colon-free mount passes this check and reaches the runtime lookup.
	_, err := NewCLI(WithRuntime("llmkit-no-such-runtime"), WithImage("img"),
		WithHostToolchains(ToolchainResolution{mounts: []ROMount{{HostPath: "/host/a", ContainerPath: hostToolchainMountRoot + "/a"}}}))
	if err == nil || !strings.Contains(err.Error(), "not found on PATH") {
		t.Errorf("colon-free toolchain: err = %v, want the runtime-lookup error", err)
	}
}

// stubRuntime writes a container "runtime" that exits 97 at once, for tests
// asserting that Exec refuses a Spec before any runtime contact: if the
// refusal regresses, the run fails fast instead of launching anything.
func stubRuntime(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "stub-runtime")
	if err := os.WriteFile(p, []byte("#!/bin/sh\nexit 97\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestCLIExecRefusesSpecMountAtToolchainPath: the CLI's Exec admission sees
// the toolchain mounts its constructor took, so a Spec mount at a toolchain
// ContainerPath is refused before any runtime contact (stubRuntime).
func TestCLIExecRefusesSpecMountAtToolchainPath(t *testing.T) {
	res := ToolchainResolution{mounts: []ROMount{{HostPath: "/host/node", ContainerPath: hostToolchainMountRoot + "/node", Shared: true}}}
	s, err := NewCLI(WithRuntime(stubRuntime(t)), WithImage("img"), WithHostToolchains(res))
	if err != nil {
		t.Fatalf("NewCLI: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	_, err = s.Exec(context.Background(), mountSpec(ROMount{HostPath: "/h", ContainerPath: hostToolchainMountRoot + "/node"}))
	var uns *UnsupportedSpecError
	if !errors.As(err, &uns) || uns.Backend != backendCLI || uns.Field != "ROMounts" {
		t.Fatalf("Exec err = %v, want cli UnsupportedSpecError{ROMounts}", err)
	}
}

// fakeBaselineMkdir puts an executable `mkdir` in a fresh temp dir at the
// front of PATH. The dir is outside defaultContainerPath, so NewBwrap's
// POSIX-baseline resolution is non-empty and mounts it (one bind).
func fakeBaselineMkdir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "bin")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mkdir"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	root, err := filepath.EvalSymlinks(filepath.Dir(dir))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func skipWithoutBwrap(t *testing.T) {
	t.Helper()
	if ok, reason := DetectBwrap(); !ok {
		t.Skipf("bwrap unusable: %s", reason)
	}
}

// TestNewBwrapDoesNotWriteIntoCallerToolchainMounts pins that concurrent
// NewBwrap calls sharing one ToolchainResolution whose mounts slice has
// spare capacity, on a host with a non-empty bwrap baseline, leave the
// caller's slice — length, capacity region, and elements — unchanged and are
// -race clean, and each backend's own list carries the baseline bind.
func TestNewBwrapDoesNotWriteIntoCallerToolchainMounts(t *testing.T) {
	skipWithoutBwrap(t)
	baselineRoot := fakeBaselineMkdir(t)

	backing := make([]ROMount, 3, 4)
	for i := range backing {
		backing[i] = ROMount{HostPath: "/host/tc", ContainerPath: hostToolchainMountRoot + "/tc" + string(rune('a'+i)), Shared: true}
	}
	want := append([]ROMount(nil), backing...)
	res := ToolchainResolution{mounts: backing}

	const n = 8
	backends := make([]*Bwrap, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b, err := NewBwrap(WithHostToolchains(res))
			if err != nil {
				t.Errorf("NewBwrap: %v", err)
				return
			}
			backends[i] = b
		}()
	}
	wg.Wait()
	if t.Failed() {
		return
	}

	if got := backing[:cap(backing)]; got[3] != (ROMount{}) || len(backing) != 3 {
		t.Errorf("caller's spare capacity was written: backing[:cap] = %+v", got)
	}
	for i, m := range want {
		if backing[i] != m {
			t.Errorf("caller's mount %d changed: %+v, want %+v", i, backing[i], m)
		}
	}
	for i, b := range backends {
		t.Cleanup(func() { _ = b.Close() })
		if len(b.toolchainBinds) <= 3 || !slices.Equal(b.toolchainBinds[:3], want) ||
			!slices.ContainsFunc(b.toolchainBinds[3:], func(m ROMount) bool { return m.HostPath == baselineRoot }) {
			t.Errorf("backend %d toolchainBinds = %+v, want the 3 toolchain mounts plus the baseline bind for %s", i, b.toolchainBinds, baselineRoot)
		}
	}
}

// TestNewCLIOwnsToolchainMounts: the CLI's toolchain list is its own copy —
// writing through it does not reach the caller's ToolchainResolution.
func TestNewCLIOwnsToolchainMounts(t *testing.T) {
	backing := []ROMount{{HostPath: "/host/a", ContainerPath: hostToolchainMountRoot + "/a", Shared: true}}
	s, err := NewCLI(WithRuntime(os.Args[0]), WithImage("img"), WithHostToolchains(ToolchainResolution{mounts: backing}))
	if err != nil {
		t.Fatalf("NewCLI: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	s.toolchainBinds[0].HostPath = "/mutated"
	if backing[0].HostPath != "/host/a" {
		t.Errorf("caller's mount rewritten through the backend's list: %+v", backing[0])
	}
}

// TestBwrapExecRefusesSpecMountAtBaselineOrToolchainPath: Bwrap's Exec
// admission sees both the toolchain and the baseline binds its constructor
// merged, refusing a Spec mount at either before any run.
func TestBwrapExecRefusesSpecMountAtBaselineOrToolchainPath(t *testing.T) {
	skipWithoutBwrap(t)
	baselineRoot := fakeBaselineMkdir(t)
	res := ToolchainResolution{mounts: []ROMount{{HostPath: "/host/tc", ContainerPath: hostToolchainMountRoot + "/tc", Shared: true}}}
	b, err := NewBwrap(WithHostToolchains(res))
	if err != nil {
		t.Fatalf("NewBwrap: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	for _, m := range b.toolchainBinds {
		_, err := b.Exec(context.Background(), mountSpec(ROMount{HostPath: "/h", ContainerPath: m.ContainerPath}))
		var uns *UnsupportedSpecError
		if !errors.As(err, &uns) || uns.Backend != backendBwrap || uns.Field != "ROMounts" {
			t.Errorf("Exec with a Spec mount at %q: err = %v, want bwrap UnsupportedSpecError{ROMounts}", m.ContainerPath, err)
		}
	}
	if !slices.ContainsFunc(b.toolchainBinds, func(m ROMount) bool { return m.HostPath == baselineRoot }) {
		t.Fatalf("toolchainBinds = %+v carry no baseline bind for %s; the test proved nothing", b.toolchainBinds, baselineRoot)
	}
}

// TestValidateSpecRefusesProvidedDestinations pins the reserved
// destinations: a Spec mount whose cleaned ContainerPath equals a
// destination the backend or its runtime provides itself is refused as
// that backend's UnsupportedSpecError. The literal sets are the frozen
// measured sets less /workspace (WorkspaceMount, refused on every backend
// as InvalidSpecError), and each backend's package var must equal its set,
// so adding an entry to a var or removing one turns this red.
// Only equality is refused: a nested path (/tmp/goroot is what the bwrap
// Go-toolchain integration test mounts) and an unrelated path are admitted.
func TestValidateSpecRefusesProvidedDestinations(t *testing.T) {
	provided := map[string][]string{
		backendCLI: {
			"/tmp", "/run", "/var/tmp", "/run/.containerenv", "/etc/hostname", "/etc/hosts",
			"/etc/resolv.conf", "/dev", "/dev/shm", "/dev/pts", "/dev/mqueue", "/dev/null",
			"/dev/zero", "/dev/full", "/dev/tty", "/dev/random", "/dev/urandom", "/proc", "/sys",
			"/sys/fs/cgroup", "/proc/acpi", "/proc/scsi", "/proc/kcore", "/proc/keys",
			"/proc/timer_list", "/proc/interrupts", "/proc/asound", "/proc/bus", "/proc/fs",
			"/proc/irq", "/proc/sys", "/proc/sysrq-trigger", "/sys/devices/virtual/powercap",
			"/sys/firmware",
		},
		backendBwrap: {
			"/proc", "/dev", "/dev/null", "/dev/zero", "/dev/full", "/dev/random",
			"/dev/urandom", "/dev/tty", "/dev/pts", "/tmp", "/etc/resolv.conf",
		},
	}
	vars := map[string][]string{backendCLI: cliProvidedDestinations, backendBwrap: bwrapProvidedDestinations}
	asSet := func(s []string) []string { return slices.Compact(slices.Sorted(slices.Values(s))) }
	for backend, dests := range provided {
		if got, want := asSet(vars[backend]), asSet(dests); !slices.Equal(got, want) {
			t.Errorf("%s provided destinations = %q, want exactly the frozen set %q", backend, got, want)
		}
		for _, dest := range dests {
			for _, spelled := range []string{dest, dest + "/", "/x/.." + dest} {
				for _, field := range []string{"ROMounts", "RWMounts"} {
					s := baseValidSpec()
					m := []ROMount{{HostPath: "/h", ContainerPath: spelled}}
					if field == "ROMounts" {
						s.ROMounts = m
					} else {
						s.RWMounts = m
					}
					err := validateSpec(backend, s, nil)
					var uns *UnsupportedSpecError
					if !errors.As(err, &uns) || uns.Backend != backend || uns.Field != field || uns.Value != spelled {
						t.Errorf("%s %s at %q: validateSpec = %v, want UnsupportedSpecError{%s %s %q}", backend, field, spelled, err, backend, field, spelled)
					}
				}
			}
		}
		for _, ok := range []string{"/opt/x", "/tmp/goroot", "/dev/x"} {
			if err := validateSpec(backend, mountSpec(ROMount{HostPath: "/h", ContainerPath: ok}), nil); err != nil {
				t.Errorf("%s mount at %q: validateSpec = %v, want nil", backend, ok, err)
			}
		}
	}
	if err := validateSpec(backendMock, mountSpec(ROMount{HostPath: "/h", ContainerPath: "/dev"}), nil); err != nil {
		t.Errorf("mock renders nothing, so /dev is admitted: validateSpec = %v", err)
	}
}

// TestValidateSpecCLIRefusesColonWorkspace: the CLI renders Spec.Workspace
// through `-v host:ctr:opts`, which cannot express ':', so an absolute
// Workspace containing one is refused before the run as the CLI's
// UnsupportedSpecError; the backends that can express it admit it, and a
// relative one stays the universal InvalidSpecError.
func TestValidateSpecCLIRefusesColonWorkspace(t *testing.T) {
	ws := func(p string) Spec {
		s := baseValidSpec()
		s.Workspace = p
		return s
	}
	var uns *UnsupportedSpecError
	if err := validateSpec(backendCLI, ws("/tmp/ws:x"), nil); !errors.As(err, &uns) || uns.Backend != backendCLI || uns.Field != "Workspace" || uns.Value != "/tmp/ws:x" {
		t.Errorf("cli Workspace /tmp/ws:x: validateSpec = %v, want UnsupportedSpecError{cli Workspace}", err)
	}
	for _, backend := range []string{backendBwrap, backendMock} {
		if err := validateSpec(backend, ws("/tmp/ws:x"), nil); err != nil {
			t.Errorf("%s Workspace /tmp/ws:x: validateSpec = %v, want nil", backend, err)
		}
	}
	var inv *InvalidSpecError
	if err := validateSpec(backendCLI, ws("ws:x"), nil); !errors.As(err, &inv) || inv.Field != "Workspace" {
		t.Errorf("cli relative Workspace ws:x: validateSpec = %v, want InvalidSpecError{Workspace}", err)
	}
}

// TestCLIExecRefusesRuntimeDestinationAndColonWorkspace drives the shipped
// CLI Exec (stubRuntime, never reached): a Spec mount at /dev (the
// runtime's own mount; podman exits 127 on it) and a Workspace containing
// ':' (podman exits 125 on it) are refused before any run.
func TestCLIExecRefusesRuntimeDestinationAndColonWorkspace(t *testing.T) {
	s, err := NewCLI(WithRuntime(stubRuntime(t)), WithImage("img"))
	if err != nil {
		t.Fatalf("NewCLI: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	var uns *UnsupportedSpecError
	if _, err := s.Exec(context.Background(), mountSpec(ROMount{HostPath: t.TempDir(), ContainerPath: "/dev"})); !errors.As(err, &uns) || uns.Field != "ROMounts" || uns.Value != "/dev" {
		t.Errorf("Exec with a mount at /dev: err = %v, want cli UnsupportedSpecError{ROMounts /dev}", err)
	}
	wsDir := filepath.Join(t.TempDir(), "ws:x")
	if err := os.Mkdir(wsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	spec := baseValidSpec()
	spec.Workspace = wsDir
	if _, err := s.Exec(context.Background(), spec); !errors.As(err, &uns) || uns.Field != "Workspace" {
		t.Errorf("Exec with Workspace %q: err = %v, want cli UnsupportedSpecError{Workspace}", wsDir, err)
	}
}

// TestBwrapExecRefusesSpecMountAtTmp drives the shipped Bwrap Exec: a Spec
// mount at /tmp would shadow bwrap's own --tmpfs /tmp (HOME), leaving it
// read-only, so it is refused before any run.
func TestBwrapExecRefusesSpecMountAtTmp(t *testing.T) {
	skipWithoutBwrap(t)
	b, err := NewBwrap()
	if err != nil {
		t.Fatalf("NewBwrap: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	_, err = b.Exec(context.Background(), mountSpec(ROMount{HostPath: t.TempDir(), ContainerPath: "/tmp"}))
	var uns *UnsupportedSpecError
	if !errors.As(err, &uns) || uns.Backend != backendBwrap || uns.Field != "ROMounts" || uns.Value != "/tmp" {
		t.Fatalf("Exec with a mount at /tmp: err = %v, want bwrap UnsupportedSpecError{ROMounts /tmp}", err)
	}
}

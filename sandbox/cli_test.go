package sandbox

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestNewCLIRequiresImage(t *testing.T) {
	if _, err := NewCLI(WithRuntime("podman")); err == nil {
		t.Fatal("expected error when no image is configured")
	}
}

func TestNewCLIUnknownRuntime(t *testing.T) {
	if _, err := NewCLI(WithRuntime("definitely-not-a-real-runtime-xyz"), WithImage("img")); err == nil {
		t.Fatal("expected error for runtime not on PATH")
	}
}

// TestNewCLIRefusesBwrapOnlyOptions pins the one-Option-type contract from
// the CLI side: every bwrap-only option must be rejected with an error
// NAMING the option, never silently ignored.
func TestNewCLIRefusesBwrapOnlyOptions(t *testing.T) {
	for _, opt := range []Option{
		WithCapPolicy(CapBestEffort),
		WithToolchainBinds([]ROMount{{HostPath: "/h", ContainerPath: "/c"}}),
		WithToolchainPath("/opt/kit/bin"),
	} {
		if _, err := NewCLI(WithRuntime("podman"), WithImage("img"), opt); err == nil {
			t.Error("NewCLI accepted a bwrap-only option without error")
		} else if !strings.Contains(err.Error(), "option With") || !strings.Contains(err.Error(), "cli") {
			t.Errorf("error %v must name the option and the cli backend", err)
		}
	}
}

func TestResolveParamsAppliesDefaultsAndOverrides(t *testing.T) {
	s := &CLI{
		runtime:        "podman",
		defaultImage:   "default-img",
		defaultCPUs:    2,
		defaultMemory:  2048,
		defaultNetwork: NetworkNone,
		pidsLimit:      256,
		maxOutputBytes: DefaultMaxOutputBytes,
	}

	// Empty spec -> backend defaults.
	p, err := s.resolveParams(Spec{Cmd: []string{"true"}})
	if err != nil {
		t.Fatalf("resolveParams: %v", err)
	}
	if p.image != "default-img" || p.cpus != 2 || p.memoryMB != 2048 || p.network != NetworkNone || p.pidsLimit != 256 {
		t.Fatalf("defaults not applied: %+v", p)
	}

	// Spec overrides: image (the container backend honors it) and network.
	p, err = s.resolveParams(Spec{
		Cmd:     []string{"true"},
		Image:   "custom",
		Network: NetworkHost,
		Env:     []string{"A=b"},
	})
	if err != nil {
		t.Fatalf("resolveParams: %v", err)
	}
	if p.image != "custom" || p.network != NetworkHost {
		t.Fatalf("overrides not applied: %+v", p)
	}
	if len(p.env) != 1 || p.env[0] != "A=b" {
		t.Fatalf("env not propagated: %+v", p.env)
	}
}

func TestExecRejectsEmptyCmd(t *testing.T) {
	s := &CLI{runtime: "podman", defaultImage: "img", maxOutputBytes: DefaultMaxOutputBytes}
	if _, err := s.Exec(context.Background(), Spec{RepoDir: t.TempDir()}); err == nil {
		t.Fatal("expected error for empty Cmd")
	}
}

func TestRandTokenUnique(t *testing.T) {
	seen := make(map[string]bool)
	for i := 0; i < 1000; i++ {
		tok := randToken()
		if len(tok) != 32 {
			t.Fatalf("token length = %d, want 32", len(tok))
		}
		if seen[tok] {
			t.Fatalf("duplicate token %q", tok)
		}
		seen[tok] = true
	}
}

// TestOptionsConfigureCLI pins the shared Option set's effect on the CLI
// backend's defaults, through the real constructor.
func TestOptionsConfigureCLI(t *testing.T) {
	if _, ok := Detect(); !ok {
		t.Skip("no container runtime detected; NewCLI needs one")
	}
	s, err := NewCLI(
		WithRuntime("podman"), WithImage("img"),
		WithCPUs(4), WithMemoryMB(1024), WithTimeout(5*time.Second),
		WithNetwork(NetworkBridge), WithPidsLimit(64), WithMaxOutputBytes(2048),
		WithScratchSizeMB(256), WithWorkspaceGrowthCeilingMB(1024),
	)
	if err != nil {
		t.Fatalf("NewCLI: %v", err)
	}
	if s.defaultCPUs != 4 || s.defaultMemory != 1024 || s.defaultTimeout != 5*time.Second {
		t.Fatalf("options not applied: %+v", s)
	}
	if s.defaultNetwork != NetworkBridge || s.pidsLimit != 64 || s.maxOutputBytes != 2048 {
		t.Fatalf("options not applied: %+v", s)
	}
	if s.defaultScratchSizeMB != 256 {
		t.Errorf("defaultScratchSizeMB = %d, want 256", s.defaultScratchSizeMB)
	}
	if want := int64(1024) * 1024 * 1024; s.defaultGrowthCeilingBytes != want {
		t.Errorf("defaultGrowthCeilingBytes = %d, want %d (1024 MB in bytes)", s.defaultGrowthCeilingBytes, want)
	}
}

//go:build integration

// Local-mounts integration test proves that a module whose dependency lives
// in a mounted sibling directory builds and tests offline when that sibling is
// exposed as a read-only mount at the path the replace directive references.
// Run with:
//
//	go test -tags integration ./sandbox/...
//
// Requires a container runtime with a Go toolchain image. Skipped automatically
// when no runtime is detected or the image cannot be used.
package sandbox

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

// goTestImage is a Go toolchain image used to actually build/test a module.
const goTestImage = "docker.io/library/golang:1.23-alpine"

// newGoMountCLI builds a CLI against the detected runtime with a Go image,
// skipping when none is available or the image cannot be used.
func newGoMountCLI(t *testing.T) *CLI {
	t.Helper()
	rt, ok := Detect()
	if !ok {
		t.Skip("no container runtime detected; skipping integration test")
	}
	base := []Option{
		WithRuntime(rt),
		WithImage(goTestImage),
		WithCPUs(2),
		WithMemoryMB(1024),
		WithPidsLimit(512),
		WithTimeout(120 * time.Second),
	}
	s, err := NewCLI(base...)
	if err != nil {
		t.Skipf("NewCLI: %v", err)
	}
	// Force the image to be available up front; skip if it can't be pulled.
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if _, err := s.Exec(ctx, Spec{RepoDir: t.TempDir(), Cmd: []string{"true"}, Network: NetworkBridge}); err != nil {
		t.Skipf("cannot run Go test image %q (pull failed?): %v", goTestImage, err)
	}
	return s
}

// TestLocalMountsSiblingDepBuildsOffline verifies the composable local-mount
// layer: a Go module that path-replaces a dependency with a sibling on disk
// builds and tests correctly in a network-none sandbox when the sibling is
// mounted read-only at the replaced path. The ROMount is built by hand
// (Shared=true: the host owns this tree during the test) rather than via a
// dependency resolver.
//
// Layout:
//
//	<tmpdir>/
//	  sibling/           <- the "local dep" (a tiny Go library)
//	    go.mod
//	    lib.go
//	  repo/              <- the module that uses the sibling
//	    go.mod           (replace directive: /sibling)
//	    main_test.go     (imports and tests the sibling lib)
func TestLocalMountsSiblingDepBuildsOffline(t *testing.T) {
	cli := newGoMountCLI(t)

	root := t.TempDir()

	// --- sibling library -------------------------------------------------------
	siblingDir := filepath.Join(root, "sibling")
	mustMkdir(t, siblingDir)
	mustWrite(t, filepath.Join(siblingDir, "go.mod"), "module example.com/sibling\n\ngo 1.21\n", 0o644)
	mustWrite(t, filepath.Join(siblingDir, "lib.go"), `package sibling

// Answer returns the answer to life, the universe and everything.
func Answer() int { return 42 }
`, 0o644)

	// --- module that uses the sibling -------------------------------------------
	repoDir := filepath.Join(root, "repo")
	mustMkdir(t, repoDir)
	mustWrite(t, filepath.Join(repoDir, "go.mod"), `module example.com/repo

go 1.21

require example.com/sibling v0.0.0

replace example.com/sibling => /sibling
`, 0o644)
	mustWrite(t, filepath.Join(repoDir, "main_test.go"), `package main_test

import (
	"testing"
	"example.com/sibling"
)

func TestAnswer(t *testing.T) {
	if got := sibling.Answer(); got != 42 {
		t.Fatalf("Answer() = %d, want 42", got)
	}
}
`, 0o644)

	// The container path for the sibling matches the replace directive above.
	siblingMount := ROMount{
		HostPath:      siblingDir,
		ContainerPath: "/sibling",
		Shared:        true,
	}

	// Run `go test ./...` in the module inside a network-none sandbox.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := cli.Exec(ctx, Spec{
		RepoDir:  repoDir,
		Cmd:      []string{"go", "test", "./..."},
		ROMounts: []ROMount{siblingMount},
		Network:  NetworkNone,
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if result.ExitCode != 0 {
		t.Fatalf("go test failed (exit %d):\nstdout: %s\nstderr: %s",
			result.ExitCode, result.Stdout, result.Stderr)
	}
}

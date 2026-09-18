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
	"testing"
	"time"
)

// newConformanceBwrap builds a Bwrap backend for the conformance leg,
// skipping when the backend is unusable on this host.
func newConformanceBwrap(t *testing.T) *Bwrap {
	t.Helper()
	s := newTestBwrap(t, WithBwrapMaxOutputBytes(64))
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
	s, err := NewCLI(rt, testImage,
		WithCPUs(1),
		WithMemoryMB(256),
		WithPidsLimit(128),
		WithTimeout(30*time.Second),
		WithMaxOutputBytes(64),
	)
	if err != nil {
		t.Skipf("NewCLI: %v", err)
	}
	ensureImage(t, s)
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestConformanceBwrap(t *testing.T) {
	s := newConformanceBwrap(t)
	runConformance(t, s, t.TempDir())
}

func TestConformanceCLI(t *testing.T) {
	s := newConformanceCLI(t)
	runConformance(t, s, t.TempDir())
}

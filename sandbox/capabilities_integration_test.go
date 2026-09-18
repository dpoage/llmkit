//go:build integration

package sandbox

import (
	"context"
	"testing"
)

// Real-backend probe test: the parameterized probe table is executed inside
// an actual sandbox (container CLI, and Bwrap when available) with the
// generic shell fixtures defined in capabilities_test.go, and the
// interpreted modes are asserted. Requires podman/docker on PATH and network
// access to pull the test image; the bwrap leg requires bwrap + unprivileged
// user namespaces. Run with: go test -tags integration ./sandbox/...

func TestProbeCapabilitiesIntegration(t *testing.T) {
	repoDir := t.TempDir()
	ctx := context.Background()

	t.Run("cli_generic_probe", func(t *testing.T) {
		s := newTestCLI(t)
		// The image string IS the container image here — ProbeCapabilities
		// threads it into every probe's Spec.Image — so pass a real,
		// pullable image, not a made-up cache-key label.
		image := testImage
		cs := ProbeCapabilities(ctx, s, image, repoDir, nil, nil, nil, testProbes)
		if !cs.Available("alpha", "one") || !cs.Available("alpha", "two") {
			t.Errorf("expected alpha one+two from a real `echo tok1; echo tok2` probe, got caps=%v", cs)
		}
		if !cs.Available("beta", "three") {
			t.Errorf("expected beta/three from a real `echo tok3` probe, got caps=%v", cs)
		}
	})

	t.Run("cli_failing_probe_best_effort", func(t *testing.T) {
		s := newTestCLI(t)
		image := testImage
		// A probe whose command exits non-zero with no tokens: best-effort
		// semantics mean an all-false entry, never an error.
		failing := ProbeEntry{Name: "failing", Probe: []string{"/bin/sh", "-c", "exit 9"}, Interpret: alphaProbe.Interpret}
		cs := ProbeCapabilities(ctx, s, image, repoDir, nil, nil, nil, []ProbeEntry{failing})
		if cs.Available("failing", "one") || cs.Available("failing", "two") {
			t.Errorf("a failing probe must read as unavailable, got caps=%v", cs)
		}
		if _, ok := cs["failing"]; !ok {
			t.Errorf("failing entry missing from the returned set: %v", cs)
		}
	})

	t.Run("bwrap_generic_probe", func(t *testing.T) {
		s := newTestBwrap(t)
		// Bwrap has no image concept; the image string only feeds the cache
		// key and Spec, both of which the backend ignores.
		const image = "kit-probe-integration-bwrap"
		InvalidateCapabilityCache(image)
		cs := ProbeCapabilities(ctx, s, image, repoDir, nil, nil, nil, testProbes)
		if !cs.Available("alpha", "one") || !cs.Available("alpha", "two") {
			t.Errorf("expected alpha one+two from a real bwrap probe, got caps=%v", cs)
		}
		if !cs.Available("beta", "three") {
			t.Errorf("expected beta/three from a real bwrap probe, got caps=%v", cs)
		}
	})
}

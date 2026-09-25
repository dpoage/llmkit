//go:build integration

package sandbox

import (
	"context"
	"errors"
	"testing"
)

// Real-backend probe test: the parameterized probe table is executed inside
// an actual sandbox (container CLI, and Bwrap when available) with the
// generic shell fixtures defined in capabilities_test.go, and the
// interpreted modes are asserted. Requires podman/docker on PATH and network
// access to pull the test image; the bwrap leg requires bwrap + unprivileged
// user namespaces. Run with: go test -tags integration ./sandbox/...

func TestProbeIntegration(t *testing.T) {
	repoDir := t.TempDir()
	ctx := context.Background()

	t.Run("cli_generic_probe", func(t *testing.T) {
		s := newTestCLI(t)
		// The CLI backend is the only one with an image concept: pass a
		// real, pullable image on base.Image — it reaches every probe's
		// Spec unchanged (Probe does no backend sniffing).
		base := Spec{RepoDir: repoDir, Image: testImage}
		cs, err := Probe(ctx, s, base, testProbes)
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if !cs.Available("alpha", "one") || !cs.Available("alpha", "two") {
			t.Errorf("expected alpha one+two from a real `echo tok1; echo tok2` probe, got caps=%v", cs)
		}
		if !cs.Available("beta", "three") {
			t.Errorf("expected beta/three from a real `echo tok3` probe, got caps=%v", cs)
		}
	})

	t.Run("cli_failing_probe_best_effort", func(t *testing.T) {
		s := newTestCLI(t)
		base := Spec{RepoDir: repoDir, Image: testImage}
		// A probe whose command exits non-zero with no tokens: best-effort
		// semantics mean an all-false entry, never an error.
		failing := ProbeEntry{Name: "failing", Probe: []string{"/bin/sh", "-c", "exit 9"}, Interpret: alphaProbe.Interpret}
		cs, err := Probe(ctx, s, base, []ProbeEntry{failing})
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if cs.Available("failing", "one") || cs.Available("failing", "two") {
			t.Errorf("a failing probe must read as unavailable, got caps=%v", cs)
		}
		if _, ok := cs["failing"]; !ok {
			t.Errorf("failing entry missing from the returned set: %v", cs)
		}
	})

	t.Run("bwrap_generic_probe", func(t *testing.T) {
		s := newTestBwrap(t)
		// Bwrap has no image concept: base.Image MUST be empty, or every
		// probe is refused with UnsupportedSpecError.
		base := Spec{RepoDir: repoDir}
		cs, err := Probe(ctx, s, base, testProbes)
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if !cs.Available("alpha", "one") || !cs.Available("alpha", "two") {
			t.Errorf("expected alpha one+two from a real bwrap probe, got caps=%v", cs)
		}
		if !cs.Available("beta", "three") {
			t.Errorf("expected beta/three from a real bwrap probe, got caps=%v", cs)
		}
	})

	t.Run("bwrap_nonempty_image_is_refused", func(t *testing.T) {
		s := newTestBwrap(t)
		base := Spec{RepoDir: repoDir, Image: "should-be-refused"}
		cs, err := Probe(ctx, s, base, testProbes)
		if cs != nil {
			t.Errorf("CapabilitySet = %v, want nil on a refusal", cs)
		}
		var unsupported *UnsupportedSpecError
		if !errors.As(err, &unsupported) {
			t.Fatalf("err = %v, want *UnsupportedSpecError (Bwrap has no image concept)", err)
		}
		if unsupported.Field != "Image" {
			t.Errorf("UnsupportedSpecError.Field = %q, want %q", unsupported.Field, "Image")
		}
	})
}

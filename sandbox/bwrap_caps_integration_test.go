//go:build integration

package sandbox

import (
	"strings"
	"testing"
)

// bwrap_caps_integration_test.go pins bead llmkit-bk8.1.16 end to end: under
// the systemd-run cap wrapper, every Spec.Cmd argv element reaches the inner
// command byte-for-byte. On a host where systemd-run expands ${NAME} from
// the HOST environment (systemd before the fix), the row would print the
// host value; with --expand-environment=no in place it prints the empty
// in-sandbox expansion (bwrap --clearenv), the literal, and the literal $$.

// TestBwrapCapWrapperPreservesArgvByteForByte skips unless this host's bwrap
// runs under the systemd-run cap method — the only wrapper that re-reads
// the argv.
func TestBwrapCapWrapperPreservesArgvByteForByte(t *testing.T) {
	if detectCapMethod(t.Context(), systemdRunExpandSupport) != bwrapCapSystemdRun {
		t.Skip("cap method is not systemd-run; nothing re-reads the argv here")
	}

	t.Setenv("ZZHOSTONLY", "host-secret-value")

	s := newTestBwrap(t)
	defer func() { _ = s.Close() }()

	res, err := s.Exec(t.Context(), Spec{
		RepoDir: t.TempDir(),
		Cmd:     []string{"/bin/sh", "-c", "printf '%s|%s|%s\\n' \"${ZZHOSTONLY}\" '$ZZHOSTONLY' '$$'"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	got := strings.TrimSuffix(res.Stdout, "\n")
	if got != "|$ZZHOSTONLY|$$" {
		t.Fatalf("stdout = %q, want |$ZZHOSTONLY|$$ — the wrapper must not expand ${NAME} from the host env, must not touch quoted literals, and must leave $$ alone", got)
	}
}

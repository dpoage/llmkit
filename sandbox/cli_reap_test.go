package sandbox

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestForceRemoveArgvFollowsRuntimeIdentity: forceRemove invokes the runtime
// the CLI was built with using the argv that runtime accepts, identifying
// it by the basename of its resolved path — a docker-named symlink to a
// podman binary is podman.
func TestForceRemoveArgvFollowsRuntimeIdentity(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based fake runtime assumes POSIX /bin/sh")
	}
	for _, tc := range []struct {
		name    string
		binary  string // file name of the script
		link    string // if set, the runtime name is a symlink to binary
		wantArg string
	}{
		{"podman", "podman", "", "rm -f --time 0 llmkit-x"},
		{"docker", "docker", "", "rm -f llmkit-x"},
		{"docker symlink to podman", "podman", "docker", "rm -f --time 0 llmkit-x"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			logFile := filepath.Join(dir, "argv")
			script := "#!/bin/sh\necho \"$@\" > " + logFile + "\n"
			if err := os.WriteFile(filepath.Join(dir, tc.binary), []byte(script), 0o755); err != nil {
				t.Fatal(err)
			}
			rt := tc.binary
			if tc.link != "" {
				if err := os.Symlink(tc.binary, filepath.Join(dir, tc.link)); err != nil {
					t.Fatal(err)
				}
				rt = tc.link
			}
			t.Setenv("PATH", dir)
			(&CLI{runtime: rt}).forceRemove("llmkit-x")
			got, err := os.ReadFile(logFile)
			if err != nil {
				t.Fatalf("the runtime was not invoked: %v", err)
			}
			if strings.TrimSpace(string(got)) != tc.wantArg {
				t.Errorf("argv = %q, want %q", strings.TrimSpace(string(got)), tc.wantArg)
			}
		})
	}
}

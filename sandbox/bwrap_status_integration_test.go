//go:build integration

package sandbox

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// TestBwrapExecFailureBeforeCommandStartIs125 pins bead llmkit-bk8.1.24
// against the real bwrap: a Spec whose ROMount/RWMount HostPath does not
// exist fails inside bwrap before the command starts and reports ExitCode
// 125 with a nil error (the CLI backend's measured answer for the same
// Spec), while a command that ran keeps its own exit code — including a 1
// that prints bwrap's stderr prefix itself.
func TestBwrapExecFailureBeforeCommandStartIs125(t *testing.T) {
	s := newTestBwrap(t, WithCapPolicy(CapBestEffort))
	defer func() { _ = s.Close() }()
	missing := filepath.Join(t.TempDir(), "absent")

	for _, tc := range []struct {
		name     string
		spec     Spec
		wantExit int
		wantErr  string
	}{
		{"missing ROMount HostPath", Spec{ROMounts: []ROMount{{HostPath: missing, ContainerPath: "/mnt/ro"}}, Cmd: []string{"/bin/sh", "-c", "exit 0"}}, 125, "Can't find source path"},
		{"missing RWMount HostPath", Spec{RWMounts: []ROMount{{HostPath: missing, ContainerPath: "/mnt/rw"}}, Cmd: []string{"/bin/sh", "-c", "exit 0"}}, 125, "Can't find source path"},
		{"command exit 1 with bwrap-looking stderr", Spec{Cmd: []string{"/bin/sh", "-c", "echo 'bwrap: x' >&2; exit 1"}}, 1, "bwrap: x"},
		{"command exit 3", Spec{Cmd: []string{"/bin/sh", "-c", "exit 3"}}, 3, ""},
		{"missing command", Spec{Cmd: []string{"/definitely/not/here"}}, 127, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.spec.Workspace = t.TempDir()
			res, err := s.Exec(context.Background(), tc.spec)
			if err != nil {
				t.Fatalf("Exec error = %v, want nil", err)
			}
			if res.ExitCode != tc.wantExit {
				t.Errorf("ExitCode = %d, want %d (stderr %q)", res.ExitCode, tc.wantExit, res.Stderr)
			}
			if !strings.Contains(res.Stderr, tc.wantErr) {
				t.Errorf("Stderr = %q, want it to contain %q", res.Stderr, tc.wantErr)
			}
		})
	}
}

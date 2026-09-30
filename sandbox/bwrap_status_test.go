package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fakeBwrap writes an executable stand-in for the bwrap binary: a shell
// script that reproduces one of bwrap's observable outcomes through its
// --json-status-fd descriptor (fd 3 — the descriptor Exec hands every bwrap
// child) and its exit status. The stand-in ignores its argv.
func fakeBwrap(t *testing.T, script string) *Bwrap {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("bwrap stand-in is a shell script")
	}
	stubCapMethod(t, bwrapCapNone)
	path := filepath.Join(t.TempDir(), "bwrap")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return &Bwrap{
		bwrapPath: path,
		capPolicy: CapBestEffort,
		defaults: defaults{
			defaultTimeout:       5 * time.Second,
			defaultScratchSizeMB: 1,
			maxOutputBytes:       1 << 16,
		},
	}
}

// TestBwrapExecAttributesFailureBeforeCommandStart pins bead
// llmkit-bk8.1.24: bwrap's --json-status-fd stream carries an
// {"exit-code":N} record exactly when the sandboxed command ran, and a
// non-zero bwrap exit without it is bwrap's own setup failure — reported as
// ExitCode 125 with a nil error and bwrap's message kept in Stderr, as the
// CLI backend reports the same Spec. A command's own exit codes, including a
// 1 whose stderr merely looks like bwrap's, are unchanged.
func TestBwrapExecAttributesFailureBeforeCommandStart(t *testing.T) {
	cases := []struct {
		name       string
		script     string
		wantExit   int
		wantStderr string
	}{
		{
			name:     "command exit code is unchanged",
			script:   `echo '{ "child-pid": 7 }' >&3; echo '{ "exit-code": 3 }' >&3; exit 3`,
			wantExit: 3,
		},
		{
			name:     "command exit 1 stays 1",
			script:   `echo '{ "child-pid": 7 }' >&3; echo '{ "exit-code": 1 }' >&3; exit 1`,
			wantExit: 1,
		},
		{
			name:       "command stderr that reads like bwrap's stays the command's exit 1",
			script:     `echo '{ "child-pid": 7 }' >&3; echo "bwrap: x" >&2; echo '{ "exit-code": 1 }' >&3; exit 1`,
			wantExit:   1,
			wantStderr: "bwrap: x",
		},
		{
			name:       "setup failure with a child-pid record only",
			script:     `echo '{ "child-pid": 7 }' >&3; echo "bwrap: Can't chdir to /workspace: Permission denied" >&2; exit 1`,
			wantExit:   125,
			wantStderr: "Can't chdir to /workspace",
		},
		{
			name:       "setup failure with no record at all",
			script:     `echo "bwrap: Can't find source path /nope: No such file or directory" >&2; exit 1`,
			wantExit:   125,
			wantStderr: "Can't find source path /nope",
		},
		{
			name:     "clean exit",
			script:   `echo '{ "exit-code": 0 }' >&3; exit 0`,
			wantExit: 0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := fakeBwrap(t, tc.script)
			res, err := s.Exec(context.Background(), Spec{Workspace: t.TempDir(), Cmd: []string{"true"}})
			if err != nil {
				t.Fatalf("Exec error = %v, want nil", err)
			}
			if res.ExitCode != tc.wantExit {
				t.Errorf("ExitCode = %d, want %d (stderr %q)", res.ExitCode, tc.wantExit, res.Stderr)
			}
			if !strings.Contains(res.Stderr, tc.wantStderr) {
				t.Errorf("Stderr = %q, want it to contain %q", res.Stderr, tc.wantStderr)
			}
		})
	}
}

// TestBwrapExecTimeoutKeepsClassificationWithoutStatusRecord pins the
// interaction with the supervisor's outcomes: a run killed by the deadline
// never wrote an exit-code record either, and stays a timeout (ExitCode -1,
// TimedOut) rather than becoming bwrap's 125.
func TestBwrapExecTimeoutKeepsClassificationWithoutStatusRecord(t *testing.T) {
	s := fakeBwrap(t, `echo '{ "child-pid": 7 }' >&3; exec sleep 30`)
	res, err := s.Exec(context.Background(), Spec{Workspace: t.TempDir(), Cmd: []string{"true"}, Timeout: 300 * time.Millisecond})
	if err != nil {
		t.Fatalf("Exec error = %v, want nil", err)
	}
	if !res.TimedOut || res.ExitCode != -1 {
		t.Fatalf("res = {TimedOut:%v ExitCode:%d}, want a timeout with ExitCode -1", res.TimedOut, res.ExitCode)
	}
}

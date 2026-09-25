package sandbox

import (
	"context"
	"runtime"
	"strings"
	"testing"
	"time"
)

// conformance_test.go holds ONE shared case table run through every
// available backend, asserting identical Result-SHAPE semantics: the exit
// code, the InfraKilled/KillReason pair, captured files, and stream
// truncation must mean the same thing on every backend a caller can hold.
//
// The Mock leg (untagged) is the always-on conformance check — and doubles
// as documentation of the contract a Mock user may script against and rely
// on. The real-backend legs live in conformance_integration_test.go
// (//go:build integration): running podman/docker and bwrap under -race on
// every untagged `go test` would be far too slow, so the Bwrap and CLI legs
// execute in the integration suite and auto-skip where their backend is
// missing. All legs share conformanceCases and runConformance below, so a
// case added to the table is exercised everywhere at once.
//
// The table is deliberately discriminating rather than illustrative: each
// check pins an exact shape (ExitCode == -1 and a non-empty KillReason on a
// timeout kill, Captured silently missing an unwritten file, ...) that a
// backend regression or a Mock drift would fail loudly.

// conformanceCase is one row of the shared table.
type conformanceCase struct {
	name string
	// spec is run verbatim except RepoDir, which the runner fills in.
	spec Spec
	// check asserts the Result shape. It runs only when Exec returned a nil
	// error (infrastructure failures fail the case before check).
	check func(t *testing.T, res Result)
}

func conformanceCases() []conformanceCase {
	return []conformanceCase{
		{
			name: "exit_zero_with_stdout",
			spec: Spec{Cmd: []string{"/bin/sh", "-c", "echo hello"}},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 0 {
					t.Errorf("ExitCode = %d, want 0", res.ExitCode)
				}
				if res.InfraKilled() {
					t.Errorf("InfraKilled = true, want false (res=%+v)", res)
				}
				if res.KillReason() != "" {
					t.Errorf("KillReason = %q, want empty for a run that exited on its own", res.KillReason())
				}
				if !strings.Contains(res.Stdout, "hello") {
					t.Errorf("Stdout = %q, want it to contain hello", res.Stdout)
				}
			},
		},
		{
			name: "nonzero_exit_preserved",
			// A non-zero exit of the sandboxed command is data, not a Go
			// error: Exec returns nil and the code lands in ExitCode.
			spec: Spec{Cmd: []string{"/bin/sh", "-c", "exit 3"}},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 3 {
					t.Errorf("ExitCode = %d, want 3", res.ExitCode)
				}
				if res.InfraKilled() {
					t.Errorf("InfraKilled = true, want false (res=%+v)", res)
				}
			},
		},
		{
			name: "timeout_kills_run",
			// A run killed by the sandbox must be unmistakable: TimedOut,
			// InfraKilled, a non-empty KillReason, and ExitCode -1 — never
			// "sleep exited 0".
			spec: Spec{
				Cmd:     []string{"/bin/sh", "-c", "sleep 30"},
				Timeout: time.Second,
			},
			check: func(t *testing.T, res Result) {
				if !res.TimedOut {
					t.Errorf("TimedOut = false, want true (res=%+v)", res)
				}
				if !res.InfraKilled() {
					t.Errorf("InfraKilled = false, want true for a timeout kill (res=%+v)", res)
				}
				if res.KillReason() == "" {
					t.Error("KillReason = empty, want non-empty for an infra kill")
				}
				if res.ExitCode != -1 {
					t.Errorf("ExitCode = %d, want -1 on a timeout kill", res.ExitCode)
				}
			},
		},
		{
			name: "write_and_capture_file",
			// The structured-output seam: WriteFiles injects in.txt, the
			// command turns it into out.txt, CaptureFiles reads it back.
			spec: Spec{
				Cmd:          []string{"/bin/sh", "-c", "cat in.txt > out.txt"},
				WriteFiles:   map[string][]byte{"in.txt": []byte("x")},
				CaptureFiles: []string{"out.txt"},
			},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 0 {
					t.Fatalf("ExitCode = %d, stderr=%q", res.ExitCode, res.Stderr)
				}
				if got, ok := res.Captured["out.txt"]; !ok {
					t.Errorf("Captured[out.txt] absent, want the written file (res=%+v)", res)
				} else if string(got) != "x" {
					t.Errorf("Captured[out.txt] = %q, want %q", got, "x")
				}
			},
		},
		{
			name: "missing_capture_file_absent",
			// CaptureFiles is best-effort: a file the command never wrote is
			// silently absent from Captured, never an error.
			spec: Spec{
				Cmd:          []string{"/bin/sh", "-c", "true"},
				CaptureFiles: []string{"absent.txt"},
			},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 0 {
					t.Fatalf("ExitCode = %d, stderr=%q", res.ExitCode, res.Stderr)
				}
				if _, ok := res.Captured["absent.txt"]; ok {
					t.Errorf("Captured[absent.txt] present, want silently absent (res=%+v)", res)
				}
			},
		},
		{
			name: "stdout_truncated_over_cap",
			// Output beyond the configured per-stream cap is discarded and
			// StdoutTruncated records it. The runner builds each real
			// backend with a small cap; the Mock leg scripts the truncated
			// shape directly.
			// Emits well past DefaultMaxOutputBytes (1 MiB) so the
			// untagged HostExec leg truncates on its fixed 1 MiB cap; the
			// integration runners build their backends with a 64-byte cap
			// and truncate there too.
			spec: Spec{Cmd: []string{"/bin/sh", "-c", "head -c 1200000 /dev/zero | tr '\\0' a"}},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 0 {
					t.Fatalf("ExitCode = %d, stderr=%q", res.ExitCode, res.Stderr)
				}
				if !res.StdoutTruncated {
					t.Errorf("StdoutTruncated = false, want true for %d bytes over the cap", len(res.Stdout))
				}
			},
		},
		{
			name: "missing_binary_path",
			// A command that cannot be LAUNCHED reports through the
			// shell's exit-code convention, not as an infrastructure
			// error: 127 when the path does not exist, with Exec
			// returning nil.
			spec: Spec{Cmd: []string{"/definitely/not/here"}},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 127 {
					t.Errorf("ExitCode = %d, want 127 for a missing binary (res=%+v)", res.ExitCode, res)
				}
				if res.InfraKilled() {
					t.Errorf("InfraKilled = true, want false for a launch failure (res=%+v)", res)
				}
			},
		},
		{
			name: "missing_binary_name",
			// A bare name not on PATH is also 127 — the shell convention
			// covers PATH lookup, not just absolute paths.
			spec: Spec{Cmd: []string{"definitely-not-here-xyz"}},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 127 {
					t.Errorf("ExitCode = %d, want 127 for a bare missing name (res=%+v)", res.ExitCode, res)
				}
				if res.InfraKilled() {
					t.Errorf("InfraKilled = true, want false for a launch failure (res=%+v)", res)
				}
			},
		},
		{
			name: "not_executable",
			// A found-but-not-executable command is 126. WriteFiles
			// injects the script 0644 (applyWriteFiles' fixed mode), so
			// the exec system call refuses it with EACCES.
			spec: Spec{
				Cmd:        []string{"./noexec"},
				WriteFiles: map[string][]byte{"noexec": []byte("#!/bin/sh\nexit 0\n")},
			},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 126 {
					t.Errorf("ExitCode = %d, want 126 for a non-executable command (res=%+v)", res.ExitCode, res)
				}
				if res.InfraKilled() {
					t.Errorf("InfraKilled = true, want false for a launch failure (res=%+v)", res)
				}
			},
		},
		{
			name: "signal_death",
			// A command killed by a signal reports 128+signo (SIGSEGV:
			// 139) as its exit code with a nil error — the command's own
			// verdict, never an infrastructure error and never the kit's
			// -1. The shape kills a nested shell first (a container's PID 1
			// ignores a signal it sends itself, so there the outer shell
			// survives and exits with the nested shell's 139), then the
			// outer shell itself (on HostExec and bwrap the supervised
			// command dies by SIGSEGV). ulimit -c 0 keeps core files out
			// of the test host.
			spec: Spec{Cmd: []string{"/bin/sh", "-c", `ulimit -c 0; sh -c 'kill -SEGV $$'; s=$?; kill -SEGV $$; exit $s`}},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 128+11 {
					t.Errorf("ExitCode = %d, want 139 (128+SIGSEGV) (res=%+v)", res.ExitCode, res)
				}
				if res.InfraKilled() {
					t.Errorf("InfraKilled = true, want false for the command's own signal death (res=%+v)", res)
				}
			},
		},
	}
}

// runConformanceCase runs one table row through a backend, failing the case
// on any Exec error (infrastructure failure) before the shape check.
func runConformanceCase(t *testing.T, sb Sandbox, repoDir string, tc conformanceCase) {
	t.Helper()
	spec := tc.spec
	spec.RepoDir = repoDir
	res, err := sb.Exec(context.Background(), spec)
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	tc.check(t, res)
}

// TestConformanceHostExec runs the whole shared table against the host
// backend — the untagged, always-on conformance leg. (The old Mock leg was
// tautological: it checked that a scripted Result echoes.) HostExec runs
// every row for real: launch failures exit 127/126 and a signal death exits
// 128+signo via hostCommandExit, and
// stdout_truncated_over_cap truncates on the backend's fixed
// DefaultMaxOutputBytes cap.
func TestConformanceHostExec(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based cases assume POSIX /bin/sh")
	}
	for _, tc := range conformanceCases() {
		t.Run(tc.name, func(t *testing.T) {
			runConformanceCase(t, NewHostExec(), t.TempDir(), tc)
		})
	}
}

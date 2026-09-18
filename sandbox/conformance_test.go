package sandbox

import (
	"context"
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
	// mockResult is what the Mock leg scripts for this case — the shape a
	// real backend is expected to produce for the same spec.
	mockResult Result
	// check asserts the Result shape. It runs only when Exec returned a nil
	// error (infrastructure failures fail the case before check).
	check func(t *testing.T, res Result)
}

func conformanceCases() []conformanceCase {
	return []conformanceCase{
		{
			name: "exit_zero_with_stdout",
			spec: Spec{Cmd: []string{"/bin/sh", "-c", "echo hello"}},
			mockResult: Result{
				ExitCode: 0,
				Stdout:   "hello\n",
			},
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
			spec:       Spec{Cmd: []string{"/bin/sh", "-c", "exit 3"}},
			mockResult: Result{ExitCode: 3},
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
			mockResult: Result{
				ExitCode: -1,
				TimedOut: true,
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
			mockResult: Result{
				ExitCode: 0,
				Captured: map[string][]byte{"out.txt": []byte("x")},
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
			mockResult: Result{ExitCode: 0},
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
			spec: Spec{Cmd: []string{"/bin/sh", "-c", "i=0; while [ $i -lt 400 ]; do echo 012345678901234567890123456789; i=$((i+1)); done"}},
			mockResult: Result{
				ExitCode:        0,
				Stdout:          strings.Repeat("012345678901234567890123456789\n", 4),
				StdoutTruncated: true,
			},
			check: func(t *testing.T, res Result) {
				if res.ExitCode != 0 {
					t.Fatalf("ExitCode = %d, stderr=%q", res.ExitCode, res.Stderr)
				}
				if !res.StdoutTruncated {
					t.Errorf("StdoutTruncated = false, want true for %d bytes over the cap", len(res.Stdout))
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

// runConformance runs the whole shared case table through one backend —
// the real-backend legs call this once per constructed sandbox.
func runConformance(t *testing.T, sb Sandbox, repoDir string) {
	t.Helper()
	for _, tc := range conformanceCases() {
		t.Run(tc.name, func(t *testing.T) {
			runConformanceCase(t, sb, repoDir, tc)
		})
	}
}

// TestConformanceMock runs the shared table against the Mock backend, one
// scripted Mock per case. It is the untagged, always-on conformance leg and
// documents the Result shapes a Mock user may script and rely on.
func TestConformanceMock(t *testing.T) {
	for _, tc := range conformanceCases() {
		t.Run(tc.name, func(t *testing.T) {
			m := NewMock(MockResponse{Result: tc.mockResult})
			runConformanceCase(t, m, t.TempDir(), tc)
		})
	}
}

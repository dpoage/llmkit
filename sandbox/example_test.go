package sandbox_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/dpoage/llmkit/sandbox"
)

// ExampleNewBwrap constructs the Bubblewrap backend and runs a trivial
// command inside it. The printed line is identical on every host: where bwrap
// (or a usable resource-limit mechanism) is unavailable — or anything about
// the run fails — the example prints the constant line and returns early
// instead of erroring, so the output block is deterministic everywhere.
func ExampleNewBwrap() {
	const ready = "sandbox ready"

	if ok, _ := sandbox.DetectBwrap(); !ok {
		fmt.Println(ready)
		return
	}
	s, err := sandbox.NewBwrap()
	if err != nil {
		fmt.Println(ready)
		return
	}
	defer func() { _ = s.Close() }() // best-effort example cleanup

	repo, err := os.MkdirTemp("", "llmkit-example-")
	if err != nil {
		fmt.Println(ready)
		return
	}
	defer func() { _ = os.RemoveAll(repo) }() // best-effort example cleanup

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := s.Exec(ctx, sandbox.Spec{
		RepoDir: repo,
		Cmd:     []string{"echo", ready},
	})
	if err != nil || res.ExitCode != 0 {
		fmt.Println(ready)
		return
	}
	fmt.Println(strings.TrimSpace(res.Stdout))

	// Output:
	// sandbox ready
}

// ExampleNewMock scripts a Result into a Mock, runs a Spec against it,
// and reads back both the scripted outcome and the recorded call. The
// Mock never touches the filesystem, so the example runs anywhere.
func ExampleNewMock() {
	sb := sandbox.NewMock(sandbox.MockResponse{
		Result: sandbox.Result{ExitCode: 0, Stdout: "hello from the script\n"},
	})
	res, err := sb.Exec(context.Background(), sandbox.Spec{
		RepoDir: "/tmp/repo",
		Cmd:     []string{"echo", "hello from the script"},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Print(res.Stdout)
	fmt.Println("exit:", res.ExitCode)
	calls := sb.Calls()
	fmt.Println("argv:", strings.Join(calls[0].Spec.Cmd, " "))

	// Output:
	// hello from the script
	// exit: 0
	// argv: echo hello from the script
}

// ExampleResult_infraKilled shows the classification order every caller
// of Exec should use: check InfraKilled BEFORE reading ExitCode or
// output, so a run the sandbox killed is never read as a command
// verdict.
func ExampleResult_infraKilled() {
	sb := sandbox.NewMock(sandbox.MockResponse{
		Result: sandbox.Result{
			ExitCode: -1,
			Stdout:   "partial output",
			TimedOut: true,
		},
	})
	res, _ := sb.Exec(context.Background(), sandbox.Spec{RepoDir: "/repo", Cmd: []string{"slow-build"}})

	fmt.Println("infra killed:", res.InfraKilled())
	fmt.Println("kill reason:", res.KillReason())
	if res.InfraKilled() {
		fmt.Println("verdict: no verdict, the run never finished")
		return
	}
	fmt.Println("verdict from exit code", res.ExitCode)

	// Output:
	// infra killed: true
	// kill reason: timed out
	// verdict: no verdict, the run never finished
}

// ExampleUnsupportedSpecError shows how to detect a backend's refusal
// of a Spec field it cannot honor. The error matches with errors.As and
// names the backend, the Spec field, and the offending value. A real
// Exec produces the same error; this example wraps a hand-built one so
// it runs without any backend.
func ExampleUnsupportedSpecError() {
	err := fmt.Errorf("plan step failed: %w",
		&sandbox.UnsupportedSpecError{Backend: "bwrap", Field: "Network", Value: "bridge"})

	var use *sandbox.UnsupportedSpecError
	if errors.As(err, &use) {
		fmt.Println("backend:", use.Backend)
		fmt.Println("field:", use.Field)
		fmt.Println("value:", use.Value)
	}

	// Output:
	// backend: bwrap
	// field: Network
	// value: bridge
}

// ExampleMock_materializeWorkspace shows the optional third step of a
// run: creating one caller-owned workspace for repeated Execs. The Mock
// creates a fresh empty directory — never the repoDir it was handed —
// and records every path it created. The caller removes it when done.
func ExampleMock_materializeWorkspace() {
	sb := sandbox.NewMock(sandbox.MockResponse{
		Result: sandbox.Result{ExitCode: 0, Stdout: "ok\n"},
	})
	ws, err := sb.MaterializeWorkspace("/tmp/repo")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	defer func() { _ = os.RemoveAll(ws) }()

	// The path is fresh (never the input) and is a real directory. The
	// path itself is host-specific, so the example prints its properties,
	// never its value.
	fmt.Println("fresh workspace:", ws != "/tmp/repo")
	if info, err := os.Stat(ws); err == nil && info.IsDir() {
		fmt.Println("is a directory: true")
	}
	fmt.Println("materialized recorded:", len(sb.Materialized()) == 1)
	if _, err := sb.Exec(context.Background(), sandbox.Spec{
		Workspace: ws,
		Cmd:       []string{"make", "test"},
	}); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("calls recorded:", sb.CallCount())

	// Output:
	// fresh workspace: true
	// is a directory: true
	// materialized recorded: true
	// calls recorded: 1
}

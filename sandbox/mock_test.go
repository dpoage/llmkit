package sandbox

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestMockQueueThenDefault(t *testing.T) {
	m := NewMock(MockResponse{Result: Result{ExitCode: 7}})
	m.EnqueueResponse(MockResponse{Result: Result{ExitCode: 0, Stdout: "first"}})
	m.EnqueueResponse(MockResponse{Err: errors.New("boom")})

	ctx := context.Background()

	r1, err := m.Exec(ctx, Spec{RepoDir: "/repo", Cmd: []string{"a"}})
	if err != nil || r1.Stdout != "first" {
		t.Fatalf("call 1: r=%+v err=%v", r1, err)
	}

	_, err = m.Exec(ctx, Spec{RepoDir: "/repo", Cmd: []string{"b"}})
	if err == nil || err.Error() != "boom" {
		t.Fatalf("call 2 expected boom error, got %v", err)
	}

	// Queue exhausted -> default.
	r3, err := m.Exec(ctx, Spec{RepoDir: "/repo", Cmd: []string{"c"}})
	if err != nil || r3.ExitCode != 7 {
		t.Fatalf("call 3 expected default exit 7, got r=%+v err=%v", r3, err)
	}

	if got := m.CallCount(); got != 3 {
		t.Fatalf("CallCount = %d, want 3", got)
	}
	calls := m.Calls()
	if len(calls) != 3 || calls[0].Spec.Cmd[0] != "a" || calls[2].Spec.Cmd[0] != "c" {
		t.Fatalf("recorded calls wrong: %+v", calls)
	}
}

func TestMockResponseFuncTakesPrecedence(t *testing.T) {
	m := NewMock(MockResponse{Result: Result{ExitCode: 99}})
	m.EnqueueResponse(MockResponse{Result: Result{ExitCode: 1}})
	m.ResponseFunc = func(_ context.Context, n int, spec Spec) (Result, error) {
		return Result{ExitCode: n, Stdout: spec.Cmd[0]}, nil
	}

	for i := 0; i < 3; i++ {
		r, err := m.Exec(context.Background(), Spec{RepoDir: "/repo", Cmd: []string{"cmd"}})
		if err != nil || r.ExitCode != i {
			t.Fatalf("call %d: r=%+v err=%v", i, r, err)
		}
	}
}

// TestMockHonoursCancelledContext pins the ctx contract: an already-done
// caller context returns the zero Result and the shared cancelled error,
// the call is still recorded (the caller DID attempt it), but nothing is
// consumed — the queue keeps its response and ResponseFunc is never
// consulted. A live context reaches ResponseFunc verbatim.
func TestMockHonoursCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	m := NewMock(MockResponse{Result: Result{ExitCode: 0}})
	m.EnqueueResponse(MockResponse{Result: Result{ExitCode: 1}})

	type marker struct{}
	seen := make(chan context.Context, 1)
	m.ResponseFunc = func(ctx context.Context, _ int, _ Spec) (Result, error) {
		seen <- ctx
		return Result{ExitCode: 2}, nil
	}

	res, err := m.Exec(ctx, Spec{RepoDir: "/repo", Cmd: []string{"cmd"}})
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want errors.Is(context.Canceled)", err)
	}
	if res.ExitCode != 0 || res.Stdout != "" || res.Stderr != "" || res.TimedOut || res.WorkspaceQuotaExceeded {
		t.Fatalf("res = %+v, want the zero Result on a cancelled context", res)
	}
	if got := m.CallCount(); got != 1 {
		t.Errorf("CallCount = %d, want 1 (the attempt is recorded)", got)
	}
	if len(m.queue) != 1 {
		t.Errorf("queue length = %d, want 1 (nothing consumed)", len(m.queue))
	}

	// A live context reaches ResponseFunc with the caller's values (the
	// ResponseFunc takes precedence over the queue, so this call's result
	// is its ExitCode 2).
	res, err = m.Exec(context.WithValue(context.Background(), marker{}, "v"), Spec{RepoDir: "/repo", Cmd: []string{"cmd"}})
	if err != nil || res.ExitCode != 2 {
		t.Fatalf("live ctx: r=%+v err=%v, want the ResponseFunc result", res, err)
	}
	got := <-seen
	if got.Value(marker{}) != "v" || got.Err() != nil {
		t.Errorf("ResponseFunc ctx = %v, want the caller's Exec context", got)
	}
}

// TestMockMaterializeWorkspaceNeverReturnsCallerRepo pins the 1.6 fix: the
// Mock's MaterializeWorkspace returns a FRESH EMPTY directory — never the
// caller's repoDir, whose contents a documented os.RemoveAll of the return
// value would otherwise destroy (that was the bug: following the documented
// contract deleted the caller's repository).
func TestMockMaterializeWorkspaceNeverReturnsCallerRepo(t *testing.T) {
	m := NewMock(MockResponse{Result: Result{ExitCode: 0}})

	repoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoDir, "precious.txt"), []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}

	p, err := m.MaterializeWorkspace(repoDir)
	if err != nil {
		t.Fatalf("MaterializeWorkspace: %v", err)
	}
	if p == repoDir {
		t.Fatal("MaterializeWorkspace returned the caller's repoDir — the documented RemoveAll would delete it")
	}
	if strings.HasPrefix(p, repoDir+string(filepath.Separator)) {
		t.Fatalf("MaterializeWorkspace = %q, must not live under the caller's repoDir", p)
	}
	info, err := os.Stat(p)
	if err != nil || !info.IsDir() {
		t.Fatalf("MaterializeWorkspace = %q (%v); want an existing directory", p, err)
	}
	if entries, _ := os.ReadDir(p); len(entries) != 0 {
		t.Fatalf("fresh workspace = %q has entries %v; want EMPTY (the Mock has nothing to copy)", p, entries)
	}
	if got := m.Materialized(); len(got) != 1 || got[0] != p {
		t.Fatalf("Materialized() = %v, want [%q]", got, p)
	}

	// Removing the returned workspace must leave the caller's repo intact.
	if err := os.RemoveAll(p); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(repoDir, "precious.txt")); err != nil {
		t.Fatalf("caller's repo was damaged: %v", err)
	}

	// repoDir need not exist: the second call still yields a fresh dir.
	second, err := m.MaterializeWorkspace("/does/not/exist")
	if err != nil {
		t.Fatalf("MaterializeWorkspace(nonexistent): %v", err)
	}
	defer func() { _ = os.RemoveAll(second) }()
	if second == p || second == "/does/not/exist" {
		t.Fatalf("second MaterializeWorkspace = %q; want another fresh directory", second)
	}
	if got := m.Materialized(); len(got) != 2 {
		t.Fatalf("Materialized() = %v, want two entries", got)
	}
}

func TestMockConcurrentSafe(t *testing.T) {
	m := NewMock(MockResponse{Result: Result{ExitCode: 0}})
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = m.Exec(context.Background(), Spec{RepoDir: "/repo", Cmd: []string{"x"}})
		}()
	}
	wg.Wait()
	if got := m.CallCount(); got != 50 {
		t.Fatalf("CallCount = %d, want 50", got)
	}
}

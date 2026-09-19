package sandbox

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func newHostExecRepoDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "marker.txt"), []byte("original\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestHostExec_RunsAgainstWorkspaceCopyNotLiveCheckout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	// Mutate the workspace copy; the live checkout must be untouched.
	res, err := h.Exec(context.Background(), Spec{
		RepoDir: repoDir,
		Cmd:     []string{"/bin/sh", "-c", "echo mutated > marker.txt; cat marker.txt"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("ExitCode = %d, want 0: stderr=%s", res.ExitCode, res.Stderr)
	}
	if !strings.Contains(res.Stdout, "mutated") {
		t.Errorf("Stdout = %q, want it to contain the workspace-copy write", res.Stdout)
	}

	live, err := os.ReadFile(filepath.Join(repoDir, "marker.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(live) != "original\n" {
		t.Errorf("live checkout was mutated: %q, want unchanged %q", live, "original\n")
	}
}

func TestHostExec_NonZeroExitIsNotAnError(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	res, err := h.Exec(context.Background(), Spec{
		RepoDir: repoDir,
		Cmd:     []string{"/bin/sh", "-c", "exit 3"},
	})
	if err != nil {
		t.Fatalf("Exec should not error on a non-zero exit, got %v", err)
	}
	if res.ExitCode != 3 {
		t.Errorf("ExitCode = %d, want 3", res.ExitCode)
	}
}

func TestHostExec_LaunchFailureIsAnError(t *testing.T) {
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	_, err := h.Exec(context.Background(), Spec{
		RepoDir: repoDir,
		Cmd:     []string{"definitely-not-a-real-binary-xyz"},
	})
	if err == nil {
		t.Fatal("want an infrastructure error for a binary that cannot be launched")
	}
}

func TestHostExec_EnvPropagates(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	res, err := h.Exec(context.Background(), Spec{
		RepoDir: repoDir,
		Cmd:     []string{"/bin/sh", "-c", "echo $LLMKIT_TEST_VAR"},
		Env:     []string{"LLMKIT_TEST_VAR=hello-hostexec"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !strings.Contains(res.Stdout, "hello-hostexec") {
		t.Errorf("Stdout = %q, want it to contain the injected env value", res.Stdout)
	}
}

func TestHostExec_TimeoutReportsTimedOut(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	res, err := h.Exec(context.Background(), Spec{
		RepoDir: repoDir,
		Cmd:     []string{"/bin/sh", "-c", "sleep 5"},
		Timeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !res.TimedOut {
		t.Error("want TimedOut=true")
	}
	if res.ExitCode != -1 {
		t.Errorf("ExitCode = %d, want -1 on timeout", res.ExitCode)
	}
}

func TestHostExec_EmptyCmdIsAnError(t *testing.T) {
	h := NewHostExec()
	if _, err := h.Exec(context.Background(), Spec{RepoDir: t.TempDir()}); err == nil {
		t.Fatal("want an error for an empty Cmd")
	}
}

func TestHostExec_WriteFilesInjected(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	res, err := h.Exec(context.Background(), Spec{
		RepoDir:    repoDir,
		Cmd:        []string{"/bin/sh", "-c", "cat injected.txt"},
		WriteFiles: map[string][]byte{"injected.txt": []byte("from write files\n")},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !strings.Contains(res.Stdout, "from write files") {
		t.Errorf("Stdout = %q, want the injected file's content", res.Stdout)
	}
	// The live checkout must never see the injected file either.
	if _, err := os.Stat(filepath.Join(repoDir, "injected.txt")); !os.IsNotExist(err) {
		t.Error("WriteFiles must never touch the live checkout")
	}
}

// wantUnsupportedSpec asserts err is a *UnsupportedSpecError naming the
// given backend and Spec field — the typed honor-or-refuse contract from
// the design record's field x backend table.
func wantUnsupportedSpec(t *testing.T, err error, backend, field string) {
	t.Helper()
	var ue *UnsupportedSpecError
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *UnsupportedSpecError (matchable via errors.As)", err)
	}
	if ue.Backend != backend || ue.Field != field {
		t.Errorf("UnsupportedSpecError = %+v, want backend=%q field=%q", ue, backend, field)
	}
}

// The isolation-shaped Spec fields HostExec cannot honor are REFUSED at
// Exec — never silently ignored, which would run a weaker posture than the
// Spec requested.
func TestHostExec_RefusesImage(t *testing.T) {
	h := NewHostExec()
	_, err := h.Exec(context.Background(), Spec{
		RepoDir: newHostExecRepoDir(t),
		Cmd:     []string{"true"},
		Image:   "quay.io/example/img:latest",
	})
	wantUnsupportedSpec(t, err, "host", "Image")
}

func TestHostExec_RefusesROMounts(t *testing.T) {
	h := NewHostExec()
	_, err := h.Exec(context.Background(), Spec{
		RepoDir: newHostExecRepoDir(t),
		Cmd:     []string{"true"},
		ROMounts: []ROMount{
			{HostPath: "/host/cache", ContainerPath: "/modcache", Shared: true},
		},
	})
	wantUnsupportedSpec(t, err, "host", "ROMounts")
}

func TestHostExec_RefusesRWMounts(t *testing.T) {
	h := NewHostExec()
	_, err := h.Exec(context.Background(), Spec{
		RepoDir: newHostExecRepoDir(t),
		Cmd:     []string{"true"},
		RWMounts: []ROMount{
			{HostPath: "/host/vendor", ContainerPath: "/vendor", Shared: true},
		},
	})
	wantUnsupportedSpec(t, err, "host", "RWMounts")
}

func TestHostExec_RefusesSetupCmds(t *testing.T) {
	h := NewHostExec()
	_, err := h.Exec(context.Background(), Spec{
		RepoDir:   newHostExecRepoDir(t),
		Cmd:       []string{"true"},
		SetupCmds: [][]string{{"npm", "ci", "--offline"}},
	})
	wantUnsupportedSpec(t, err, "host", "SetupCmds")
}

// TestHostExec_RefusesNetworkNoneAndBridge pins the Network row: HostExec
// cannot cut or namespace the host's network, so only "" (the default) and
// NetworkHost — both meaning "run with the host's network" — are honored.
func TestHostExec_RefusesNetworkNoneAndBridge(t *testing.T) {
	h := NewHostExec()
	for _, mode := range []NetworkMode{NetworkNone, NetworkBridge} {
		_, err := h.Exec(context.Background(), Spec{
			RepoDir: newHostExecRepoDir(t),
			Cmd:     []string{"true"},
			Network: mode,
		})
		wantUnsupportedSpec(t, err, "host", "Network")
	}
}

// TestHostExec_WorkspaceQuotaExceededAlwaysFalse pins the Result row:
// HostExec has no watchdog and no growth ceiling, so
// Result.WorkspaceQuotaExceeded is false even on a timed-out run.
func TestHostExec_WorkspaceQuotaExceededAlwaysFalse(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()
	res, err := h.Exec(context.Background(), Spec{
		RepoDir: repoDir,
		Cmd:     []string{"/bin/sh", "-c", "sleep 5"},
		Timeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !res.TimedOut {
		t.Fatal("precondition: want a timed-out run")
	}
	if res.WorkspaceQuotaExceeded {
		t.Error("WorkspaceQuotaExceeded must always be false on HostExec")
	}
}

// TestHostExec_MaterializeWorkspace pins the interface seam: a fresh,
// caller-owned full copy of repoDir (no WriteFiles applied), independent of
// any Exec's own temporary workspace.
func TestHostExec_MaterializeWorkspace(t *testing.T) {
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	ws, err := h.MaterializeWorkspace(repoDir)
	if err != nil {
		t.Fatalf("MaterializeWorkspace: %v", err)
	}
	defer os.RemoveAll(ws)
	if ws == repoDir {
		t.Fatal("MaterializeWorkspace returned the live checkout itself")
	}
	data, err := os.ReadFile(filepath.Join(ws, "marker.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "original\n" {
		t.Errorf("copied content = %q, want %q", data, "original\n")
	}

	// The returned workspace is writable and Exec accepts it via
	// Spec.Workspace, mutating only the copy.
	res, err := h.Exec(context.Background(), Spec{
		Workspace: ws,
		Cmd:       []string{"true"},
	})
	if err != nil {
		t.Fatalf("Exec on materialized workspace: %v", err)
	}
	if res.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", res.ExitCode)
	}
}

// TestHostExec_CaptureFilesReadBack pins the CaptureFiles row for the host
// backend: files the command wrote in the workspace are read back into
// Result.Captured, and a requested path the command never wrote is silently
// absent rather than failing the run.
func TestHostExec_CaptureFilesReadBack(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-based test assumes POSIX /bin/sh")
	}
	repoDir := newHostExecRepoDir(t)
	h := NewHostExec()

	res, err := h.Exec(context.Background(), Spec{
		RepoDir:      repoDir,
		Cmd:          []string{"/bin/sh", "-c", "echo result > out.txt"},
		CaptureFiles: []string{"out.txt", "never-written.txt"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if got := string(res.Captured["out.txt"]); got != "result\n" {
		t.Errorf("Captured[out.txt] = %q, want %q", got, "result\n")
	}
	if _, ok := res.Captured["never-written.txt"]; ok {
		t.Errorf("a file the command never wrote must be absent from Captured, got %v", res.Captured)
	}
}

package sandbox

import (
	"context"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestPrepareWorkspaceCopiesTreeAndAppliesWriteFiles(t *testing.T) {
	src := t.TempDir()

	// Lay out a small repo: a nested file and an executable.
	mustWrite(t, filepath.Join(src, "go.mod"), "module example\n", 0o644)
	mustMkdir(t, filepath.Join(src, "pkg"))
	mustWrite(t, filepath.Join(src, "pkg", "main.go"), "package pkg\n", 0o644)
	mustWrite(t, filepath.Join(src, "run.sh"), "#!/bin/sh\necho hi\n", 0o755)

	ws, err := prepareWorkspace(src, map[string][]byte{
		"repro/bug_test.go": []byte("package repro\n"),
		"go.mod":            []byte("module overridden\n"), // overwrites an existing file
	})
	if err != nil {
		t.Fatalf("prepareWorkspace: %v", err)
	}
	defer func() { _ = os.RemoveAll(ws) }()

	if ws == src {
		t.Fatal("workspace must be a separate directory from the source repo")
	}

	// Copied files present.
	assertFileContent(t, filepath.Join(ws, "pkg", "main.go"), "package pkg\n")

	// Executable bit preserved.
	if info, err := os.Stat(filepath.Join(ws, "run.sh")); err != nil {
		t.Fatalf("stat run.sh: %v", err)
	} else if info.Mode().Perm()&0o111 == 0 {
		t.Errorf("run.sh lost its executable bit: %v", info.Mode())
	}

	// WriteFiles injected, including nested dir creation.
	assertFileContent(t, filepath.Join(ws, "repro", "bug_test.go"), "package repro\n")

	// WriteFiles overwrote the copied file.
	assertFileContent(t, filepath.Join(ws, "go.mod"), "module overridden\n")

	// Original repo was not mutated.
	assertFileContent(t, filepath.Join(src, "go.mod"), "module example\n")
	if _, err := os.Stat(filepath.Join(src, "repro")); !os.IsNotExist(err) {
		t.Errorf("original repo was mutated: repro/ should not exist, err=%v", err)
	}
}

func TestPrepareWorkspaceRejectsNonDir(t *testing.T) {
	f := filepath.Join(t.TempDir(), "afile")
	mustWrite(t, f, "x", 0o644)
	if _, err := prepareWorkspace(f, nil); err == nil {
		t.Fatal("expected error for non-directory repo dir")
	}
}

func TestSanitizeRelPath(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		wantErr bool
		want    string
	}{
		{"simple", "a.txt", false, "a.txt"},
		{"nested", "a/b/c.txt", false, filepath.Join("a", "b", "c.txt")},
		{"dotslash", "./a.txt", false, "a.txt"},
		{"absolute", "/etc/passwd", true, ""},
		{"escape", "../outside", true, ""},
		{"escape-nested", "a/../../outside", true, ""},
		{"empty", "", true, ""},
		{"root", ".", true, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sanitizeRelPath(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for %q, got %q", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error for %q: %v", tc.in, err)
			}
			if got != tc.want {
				t.Errorf("sanitizeRelPath(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestValidateWorkspacePath locks the exported guard to sanitizeRelPath's rule:
// it must reject the exact keys applyWriteFiles would reject (absolute,
// escaping, empty, root) and accept legal relative ones, so the reproducer's
// pre-flight plan check can never diverge from the actual write.
func TestValidateWorkspacePath(t *testing.T) {
	reject := []string{"/tmp/repro_test.cpp", "/etc/passwd", "../outside", "a/../../outside", "", "."}
	for _, in := range reject {
		if err := ValidateWorkspacePath(in); err == nil {
			t.Errorf("ValidateWorkspacePath(%q) = nil, want error", in)
		}
	}
	accept := []string{"repro_test.cpp", "test/repro_test.cpp", "./a.txt"}
	for _, in := range accept {
		if err := ValidateWorkspacePath(in); err != nil {
			t.Errorf("ValidateWorkspacePath(%q) = %v, want nil", in, err)
		}
	}
}

func TestApplyWriteFilesRejectsEscape(t *testing.T) {
	ws := t.TempDir()
	err := applyWriteFiles(ws, map[string][]byte{"../escape.txt": []byte("nope")})
	if err == nil {
		t.Fatal("expected error for path escaping workspace")
	}
	if _, statErr := os.Stat(filepath.Join(filepath.Dir(ws), "escape.txt")); !os.IsNotExist(statErr) {
		t.Errorf("escape file should not have been written: %v", statErr)
	}
}

// TestApplyWriteFilesRefusesSymlinkedIntermediateDir verifies the
// write-side hardening: a symlinked directory component planted in the
// workspace by an earlier (untrusted) command must be refused, never walked
// through to write outside the workspace root.
func TestApplyWriteFilesRefusesSymlinkedIntermediateDir(t *testing.T) {
	ws := t.TempDir()
	outside := t.TempDir()

	// Simulate a prior untrusted call planting "sub" -> outside.
	if err := os.Symlink(outside, filepath.Join(ws, "sub")); err != nil {
		t.Fatal(err)
	}

	err := applyWriteFiles(ws, map[string][]byte{"sub/evil.txt": []byte("escaped")})
	if err == nil {
		t.Fatal("expected error for a symlinked intermediate directory")
	}
	if _, statErr := os.Stat(filepath.Join(outside, "evil.txt")); !os.IsNotExist(statErr) {
		t.Errorf("file must not have been written through the symlink: stat err = %v", statErr)
	}
}

// TestApplyWriteFilesRefusesSymlinkedLeaf verifies that a symlinked LEAF
// (planted at exactly the path a later WriteFiles entry targets) is refused
// rather than followed — the same escape shape as the intermediate-dir case,
// one level shallower.
func TestApplyWriteFilesRefusesSymlinkedLeaf(t *testing.T) {
	ws := t.TempDir()
	outsideFile := filepath.Join(t.TempDir(), "authorized_keys")
	mustWrite(t, outsideFile, "original\n", 0o644)

	// Simulate a prior untrusted call planting "leak" -> outsideFile.
	if err := os.Symlink(outsideFile, filepath.Join(ws, "leak")); err != nil {
		t.Fatal(err)
	}

	err := applyWriteFiles(ws, map[string][]byte{"leak": []byte("pwned\n")})
	if err == nil {
		t.Fatal("expected error for a symlinked leaf")
	}
	got, readErr := os.ReadFile(outsideFile)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if string(got) != "original\n" {
		t.Errorf("host file must not have been overwritten through the symlink: got %q", got)
	}
}

// TestApplyWriteFilesReuseOverwritesRegularFile verifies the hardening does
// NOT break the legitimate iteration-workspace reuse case: overwriting a plain
// (non-symlink) file left by an earlier call on the SAME workspace must still
// succeed.
func TestApplyWriteFilesReuseOverwritesRegularFile(t *testing.T) {
	ws := t.TempDir()
	if err := applyWriteFiles(ws, map[string][]byte{"x_test.go": []byte("v1")}); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := applyWriteFiles(ws, map[string][]byte{"x_test.go": []byte("v2")}); err != nil {
		t.Fatalf("second write (overwrite): %v", err)
	}
	got, err := os.ReadFile(filepath.Join(ws, "x_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "v2" {
		t.Errorf("content = %q, want %q (overwrite must take effect)", got, "v2")
	}
}

// --- helpers ---

func mustWrite(t *testing.T, path, content string, perm os.FileMode) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), perm); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

// writeFile is a tiny helper that creates parent dirs and writes a file.
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func assertFileContent(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if string(got) != want {
		t.Errorf("%s = %q, want %q", path, got, want)
	}
}

// TestPrepareWorkspaceSkipsGitignoredAndGit verifies the git-aware copy path:
// for a git work tree, gitignored content (a stale build tree) and the .git
// directory are NOT copied into the sandbox workspace, while tracked and
// untracked-but-not-ignored files are. This is the guard against a host
// CMakeCache.txt poisoning an in-sandbox rebuild.
func TestPrepareWorkspaceSkipsGitignoredAndGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	src := t.TempDir()
	mustWrite(t, filepath.Join(src, ".gitignore"), "/build\n", 0o644)
	mustWrite(t, filepath.Join(src, "main.go"), "package main\n", 0o644)
	mustMkdir(t, filepath.Join(src, "build"))
	mustWrite(t, filepath.Join(src, "build", "CMakeCache.txt"), "stale-host-path\n", 0o644)
	gitInit(t, src)
	// Written AFTER the commit so it is untracked-but-not-ignored: it must still
	// be copied (working-tree fidelity for uncommitted source).
	mustWrite(t, filepath.Join(src, "untracked.go"), "package main\n", 0o644)

	ws, err := prepareWorkspace(src, nil)
	if err != nil {
		t.Fatalf("prepareWorkspace: %v", err)
	}
	defer func() { _ = os.RemoveAll(ws) }()

	// Tracked + untracked-non-ignored are copied.
	assertFileContent(t, filepath.Join(ws, "main.go"), "package main\n")
	assertFileContent(t, filepath.Join(ws, "untracked.go"), "package main\n")
	// The gitignored build tree must NOT be copied (it would poison a rebuild).
	if _, err := os.Stat(filepath.Join(ws, "build")); !os.IsNotExist(err) {
		t.Errorf("gitignored build/ was copied into the workspace (err=%v)", err)
	}
	// .git metadata must NOT be copied.
	if _, err := os.Stat(filepath.Join(ws, ".git")); !os.IsNotExist(err) {
		t.Errorf(".git was copied into the workspace (err=%v)", err)
	}
}

// TestGitWorktreeFilesContract pins the exported listing contract: tracked
// files and untracked-not-ignored files are listed, gitignored files and .git
// metadata are not, and a non-git directory yields (nil, false, nil).
// Dependency-resolution callers consult this listing to learn what a sandbox
// run will materialize, so it must match copyWorkspace's input exactly.
func TestGitWorktreeFilesContract(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	// Non-git dir: isRepo=false, no error, no files.
	plain := t.TempDir()
	mustWrite(t, filepath.Join(plain, "file.txt"), "x\n", 0o644)
	files, isRepo, err := GitWorktreeFiles(plain)
	if err != nil || isRepo || files != nil {
		t.Fatalf("GitWorktreeFiles(non-git) = %v, %v, %v; want nil, false, nil", files, isRepo, err)
	}

	src := t.TempDir()
	mustWrite(t, filepath.Join(src, ".gitignore"), "/ignored-dir/\nignored.log\n", 0o644)
	mustWrite(t, filepath.Join(src, "tracked.go"), "package main\n", 0o644)
	mustMkdir(t, filepath.Join(src, "ignored-dir"))
	mustWrite(t, filepath.Join(src, "ignored-dir", "cache.bin"), "stale\n", 0o644)
	mustWrite(t, filepath.Join(src, "ignored.log"), "noise\n", 0o644)
	gitInit(t, src)
	// Written AFTER the commit so it is untracked-but-not-ignored: it must
	// still be listed (working-tree fidelity for uncommitted source).
	mustWrite(t, filepath.Join(src, "untracked.go"), "package main\n", 0o644)

	files, isRepo, err = GitWorktreeFiles(src)
	if err != nil {
		t.Fatalf("GitWorktreeFiles: %v", err)
	}
	if !isRepo {
		t.Fatal("isRepo = false for a git work tree")
	}
	got := make(map[string]bool, len(files))
	for _, f := range files {
		got[f] = true
	}
	for _, want := range []string{"tracked.go", "untracked.go", ".gitignore"} {
		if !got[want] {
			t.Errorf("%q missing from listing; got %v", want, files)
		}
	}
	for _, unwanted := range []string{"ignored-dir/cache.bin", "ignored.log", ".git/HEAD"} {
		if got[unwanted] {
			t.Errorf("%q listed but should be excluded; got %v", unwanted, files)
		}
	}
}

// gitInit makes dir a git repo and commits its current tracked contents.
func gitInit(t *testing.T, dir string) {
	t.Helper()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
		{"add", "-A"},
		{"commit", "-q", "-m", "init"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		// Neutralize the user's/CI's global+system git config so a global
		// commit.gpgsign, hooksPath hook, or credential prompt cannot make
		// `git commit` fail or hang. The repo-local identity below suffices.
		cmd.Env = append(os.Environ(),
			"GIT_CONFIG_GLOBAL=/dev/null",
			"GIT_CONFIG_SYSTEM=/dev/null",
			"GIT_TERMINAL_PROMPT=0",
		)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
}

// TestCopyWorkspaceRecreatesSymlink covers the git-aware copy's symlink branch:
// a tracked symlink must be recreated as a symlink (target not followed).
func TestCopyWorkspaceRecreatesSymlink(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	src := t.TempDir()
	mustWrite(t, filepath.Join(src, "real.txt"), "hi\n", 0o644)
	if err := os.Symlink("real.txt", filepath.Join(src, "link.txt")); err != nil {
		t.Fatalf("symlink: %v", err)
	}
	gitInit(t, src)
	dst := t.TempDir()

	if err := copyWorkspace(src, dst); err != nil {
		t.Fatalf("copyWorkspace: %v", err)
	}
	fi, err := os.Lstat(filepath.Join(dst, "link.txt"))
	if err != nil {
		t.Fatalf("symlink not copied: %v", err)
	}
	if fi.Mode()&fs.ModeSymlink == 0 {
		t.Errorf("link.txt copied as non-symlink: %v", fi.Mode())
	}
	if tgt, _ := os.Readlink(filepath.Join(dst, "link.txt")); tgt != "real.txt" {
		t.Errorf("symlink target = %q, want real.txt", tgt)
	}
}

// TestCopyFileList_RecursesSubmoduleDir covers the submodule path: git ls-files
// emits a submodule by its gitlink path, which resolves on disk to a directory.
// copyFileListSkipping must recurse-copy that working tree (so vendored-as-submodule
// deps reach the sandbox) while skipping the submodule's own .git gitfile.
func TestCopyFileList_RecursesSubmoduleDir(t *testing.T) {
	src := t.TempDir()
	mustMkdir(t, filepath.Join(src, "vendor", "gtest"))
	mustWrite(t, filepath.Join(src, "vendor", "gtest", "gtest.h"), "#pragma once\n", 0o644)
	// Submodules carry a .git FILE (a gitfile); copyTreeWith must skip it.
	mustWrite(t, filepath.Join(src, "vendor", "gtest", ".git"), "gitdir: ../../.git/modules/vendor/gtest\n", 0o644)
	dst := t.TempDir()

	dstInfo, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if err := copyFileListSkipping(src, dst, []string{filepath.Join("vendor", "gtest")}, dstInfo); err != nil {
		t.Fatalf("copyFileListSkipping: %v", err)
	}
	assertFileContent(t, filepath.Join(dst, "vendor", "gtest", "gtest.h"), "#pragma once\n")
	if _, err := os.Stat(filepath.Join(dst, "vendor", "gtest", ".git")); !os.IsNotExist(err) {
		t.Errorf("submodule .git gitfile was copied (err=%v)", err)
	}
}

// TestCopyTreeSkipsNestedGitInFallback covers the full-copy fallback (non-git
// src): a nested .git directory must be skipped so stale metadata never reaches
// the workspace even when the git-aware path is not taken.
func TestCopyTreeSkipsNestedGitInFallback(t *testing.T) {
	src := t.TempDir() // no top-level .git -> copyWorkspace falls back to copyTreeWith
	mustWrite(t, filepath.Join(src, "main.go"), "package main\n", 0o644)
	mustMkdir(t, filepath.Join(src, "sub", ".git"))
	mustWrite(t, filepath.Join(src, "sub", ".git", "config"), "x\n", 0o644)
	mustWrite(t, filepath.Join(src, "sub", "real.txt"), "keep\n", 0o644)
	dst := t.TempDir()

	if err := copyWorkspace(src, dst); err != nil {
		t.Fatalf("copyWorkspace: %v", err)
	}
	assertFileContent(t, filepath.Join(dst, "main.go"), "package main\n")
	assertFileContent(t, filepath.Join(dst, "sub", "real.txt"), "keep\n")
	if _, err := os.Stat(filepath.Join(dst, "sub", ".git")); !os.IsNotExist(err) {
		t.Errorf("nested .git dir was copied in fallback (err=%v)", err)
	}
}

// TestCopyWorkspaceSurfacesGitError: when src is a git work tree but `git
// ls-files` fails, copyWorkspace returns the error rather than silently
// full-copying gitignored artifacts (which would reintroduce the stale
// build-tree poisoning).
func TestCopyWorkspaceSurfacesGitError(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	src := t.TempDir()
	mustWrite(t, filepath.Join(src, "main.go"), "package main\n", 0o644)
	// A .git GITFILE pointing at a nonexistent gitdir: os.Stat(.git) succeeds so
	// the git-repo branch is taken, but `git ls-files` exits non-zero, driving
	// the (nil, true, err) return that copyWorkspace must propagate.
	mustWrite(t, filepath.Join(src, ".git"), "gitdir: /nonexistent-gitdir\n", 0o644)
	dst := t.TempDir()

	if err := copyWorkspace(src, dst); err == nil {
		t.Fatal("copyWorkspace must return an error when git ls-files fails on a work tree")
	}
}

// TestSanitizeCapturePaths mirrors the sanitizeRelPath contract that
// applyWriteFiles already relies on: escaping paths are rejected up front
// (before a sandbox run is spent), well-formed relative paths are cleaned and
// returned in order.
func TestSanitizeCapturePaths(t *testing.T) {
	t.Run("empty input", func(t *testing.T) {
		got, err := sanitizeCapturePaths(nil)
		if err != nil || got != nil {
			t.Fatalf("sanitizeCapturePaths(nil) = (%v, %v), want (nil, nil)", got, err)
		}
	})

	t.Run("valid relative paths are cleaned and ordered", func(t *testing.T) {
		got, err := sanitizeCapturePaths([]string{".repro-junit.xml", "./sub/../report.json"})
		if err != nil {
			t.Fatalf("sanitizeCapturePaths: %v", err)
		}
		want := []string{".repro-junit.xml", "report.json"}
		if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
			t.Fatalf("sanitizeCapturePaths = %v, want %v", got, want)
		}
	})

	t.Run("absolute path rejected", func(t *testing.T) {
		if _, err := sanitizeCapturePaths([]string{"/etc/passwd"}); err == nil {
			t.Fatal("expected error for absolute capture path")
		}
	})

	t.Run("escaping .. path rejected", func(t *testing.T) {
		if _, err := sanitizeCapturePaths([]string{"../outside.txt"}); err == nil {
			t.Fatal("expected error for escaping capture path")
		}
	})
}

// TestCaptureWorkspaceFiles covers the post-run read-back: present files are
// returned (capped at maxBytes), missing files are silently absent, and an
// empty paths list yields a nil map.
func TestCaptureWorkspaceFiles(t *testing.T) {
	ws := t.TempDir()
	mustWrite(t, filepath.Join(ws, "report.xml"), "<testsuites></testsuites>", 0o644)
	mustWrite(t, filepath.Join(ws, "big.txt"), "0123456789", 0o644)

	t.Run("nil paths yields nil", func(t *testing.T) {
		if got := captureWorkspaceFiles(ws, nil, 0); got != nil {
			t.Fatalf("captureWorkspaceFiles(nil paths) = %v, want nil", got)
		}
	})

	t.Run("present file captured, missing file silently absent", func(t *testing.T) {
		got := captureWorkspaceFiles(ws, []string{"report.xml", "does-not-exist.xml"}, 0)
		if string(got["report.xml"]) != "<testsuites></testsuites>" {
			t.Fatalf("got[report.xml] = %q", got["report.xml"])
		}
		if _, present := got["does-not-exist.xml"]; present {
			t.Fatal("missing file must be absent from the map, not present with empty/zero value")
		}
	})

	t.Run("capped at maxBytes", func(t *testing.T) {
		got := captureWorkspaceFiles(ws, []string{"big.txt"}, 4)
		if string(got["big.txt"]) != "0123" {
			t.Fatalf("got[big.txt] = %q, want capped to 4 bytes", got["big.txt"])
		}
	})

	t.Run("no captures found yields nil map", func(t *testing.T) {
		got := captureWorkspaceFiles(ws, []string{"nope.txt"}, 0)
		if got != nil {
			t.Fatalf("captureWorkspaceFiles with no hits = %v, want nil", got)
		}
	})

	t.Run("directory at the capture path is refused, not read as a file", func(t *testing.T) {
		mustMkdir(t, filepath.Join(ws, "a-directory"))
		got := captureWorkspaceFiles(ws, []string{"a-directory"}, 0)
		if _, present := got["a-directory"]; present {
			t.Fatal("a directory must be refused like a missing file, not captured")
		}
	})
}

// TestCaptureWorkspaceFiles_SymlinkEscapeRejected — a review finding:
// the sandboxed command has full write access to the workspace, so
// it is untrusted with respect to what it leaves behind. A symlink planted
// at the capture path pointing outside the workspace (a host path, or a
// device like /dev/zero) must be refused exactly like a missing file, never
// followed — this is the read-back counterpart of the write-side
// symlink-escape class (captureWorkspaceFiles /
// readCaptureFile).
func TestCaptureWorkspaceFiles_SymlinkEscapeRejected(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink semantics differ on windows")
	}

	t.Run("symlink to an absolute host path outside the workspace", func(t *testing.T) {
		ws := t.TempDir()
		outside := t.TempDir()
		secret := filepath.Join(outside, "secret.txt")
		mustWrite(t, secret, "host-only content", 0o644)
		if err := os.Symlink(secret, filepath.Join(ws, "report.xml")); err != nil {
			t.Fatalf("Symlink: %v", err)
		}

		got := captureWorkspaceFiles(ws, []string{"report.xml"}, 0)
		if _, present := got["report.xml"]; present {
			t.Fatal("a symlink escaping the workspace must be refused, not followed")
		}
	})

	t.Run("symlink to a device node is refused, not read unbounded", func(t *testing.T) {
		if _, err := os.Stat("/dev/zero"); err != nil {
			t.Skip("/dev/zero not available")
		}
		ws := t.TempDir()
		if err := os.Symlink("/dev/zero", filepath.Join(ws, "report.xml")); err != nil {
			t.Fatalf("Symlink: %v", err)
		}

		got := captureWorkspaceFiles(ws, []string{"report.xml"}, 1<<20)
		if _, present := got["report.xml"]; present {
			t.Fatal("a symlink to a device node must be refused, not read")
		}
	})

	t.Run("symlinked intermediate directory escaping the workspace", func(t *testing.T) {
		ws := t.TempDir()
		outside := t.TempDir()
		mustWrite(t, filepath.Join(outside, "report.xml"), "outside content", 0o644)
		if err := os.Symlink(outside, filepath.Join(ws, "sub")); err != nil {
			t.Fatalf("Symlink: %v", err)
		}

		got := captureWorkspaceFiles(ws, []string{"sub/report.xml"}, 0)
		if _, present := got["sub/report.xml"]; present {
			t.Fatal("a symlinked intermediate directory escaping the workspace must be refused")
		}
	})

	t.Run("symlink that stays inside the workspace is followed normally", func(t *testing.T) {
		ws := t.TempDir()
		mustWrite(t, filepath.Join(ws, "actual.xml"), "<testsuites></testsuites>", 0o644)
		if err := os.Symlink(filepath.Join(ws, "actual.xml"), filepath.Join(ws, "report.xml")); err != nil {
			t.Fatalf("Symlink: %v", err)
		}

		got := captureWorkspaceFiles(ws, []string{"report.xml"}, 0)
		if string(got["report.xml"]) != "<testsuites></testsuites>" {
			t.Fatalf("in-workspace symlink target must be captured normally, got %q", got["report.xml"])
		}
	})
}

// treeListing returns a sorted "<kind> <rel>" line per entry under root (d, f
// or l for directory, regular file or symlink), skipping any entry named .git
// because the workspace copy never carries VCS metadata.
func treeListing(root string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		if d.Name() == ".git" {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		kind := "f"
		switch {
		case d.IsDir():
			kind = "d"
		case d.Type()&fs.ModeSymlink != 0:
			kind = "l"
		}
		out = append(out, kind+" "+filepath.ToSlash(rel))
		return nil
	})
	sort.Strings(out)
	return out, err
}

// TestSelfCopyHelperProcess is the child half of runSelfCopyChild: it runs one
// workspace preparation with TMPDIR pointing into the repo being copied and
// prints the resulting workspace listing. It is skipped unless the parent set
// the mode variable, so the ordinary suite never runs it directly.
func TestSelfCopyHelperProcess(t *testing.T) {
	mode := os.Getenv("LLMKIT_SELFCOPY_MODE")
	if mode == "" {
		t.Skip("child half of runSelfCopyChild")
	}
	repo := os.Getenv("LLMKIT_SELFCOPY_REPO")
	var (
		ws       string
		listRoot string
		err      error
	)
	switch mode {
	case "prepare":
		ws, err = prepareWorkspace(repo, nil)
	case "cached":
		var c wsCache
		defer func() { _ = c.close() }()
		ws, _, err = prepareWorkspaceCached(&c, repo, nil)
	case "gitlink":
		// The destination sits inside a gitlink directory entry that
		// copyFileListSkipping copies recursively.
		if ws, err = os.MkdirTemp("", "llmkit-sandbox-"); err == nil {
			var wsInfo fs.FileInfo
			if wsInfo, err = os.Stat(ws); err == nil {
				err = copyFileListSkipping(repo, ws, []string{filepath.Join("vendor", "gtest")}, wsInfo)
			}
			listRoot = filepath.Join(ws, "vendor", "gtest")
		}
	default:
		t.Fatalf("unknown mode %q", mode)
	}
	if err != nil {
		if ws != "" {
			_ = os.RemoveAll(ws)
		}
		t.Fatalf("%s: %v", mode, err)
	}
	if listRoot == "" {
		listRoot = ws
	}
	lines, lerr := treeListing(listRoot)
	_ = os.RemoveAll(ws)
	if lerr != nil {
		t.Fatalf("list workspace: %v", lerr)
	}
	for _, l := range lines {
		_, _ = os.Stdout.WriteString("SELFCOPY " + l + "\n")
	}
}

// runSelfCopyChild re-executes the test binary as TestSelfCopyHelperProcess
// with TMPDIR=tmpdir and returns the workspace listing it printed. The child
// runs under a wall-clock bound: a copy that recurses into its own destination
// is killed rather than allowed to grow disk without limit.
func runSelfCopyChild(t *testing.T, mode, repo, tmpdir string) []string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestSelfCopyHelperProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(),
		"LLMKIT_SELFCOPY_MODE="+mode,
		"LLMKIT_SELFCOPY_REPO="+repo,
		"TMPDIR="+tmpdir,
	)
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("child %s did not terminate within 30s", mode)
	}
	if err != nil {
		t.Fatalf("child %s failed: %v\n%s", mode, err, out)
	}
	var got []string
	for _, line := range strings.Split(string(out), "\n") {
		if rest, ok := strings.CutPrefix(line, "SELFCOPY "); ok {
			got = append(got, rest)
		}
	}
	sort.Strings(got)
	return got
}

func diffListings(t *testing.T, got, want []string) {
	t.Helper()
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("workspace file set != repo file set\n got: %q\nwant: %q", got, want)
	}
}

// TestPrepareWorkspaceNeverCopiesItself pins that a workspace created inside
// the repo it copies (TMPDIR under RepoDir, directly or through a symlinked
// alias) contains exactly the repo's file set: no copy of itself, and no
// omission of a legitimate entry that merely shares the workspace name prefix.
// Covered on the non-git full-copy path, the git file-list path and the
// wsCache materialization path.
func TestPrepareWorkspaceNeverCopiesItself(t *testing.T) {
	type entry struct{ shape, mode string }
	entries := []entry{{"nongit", "prepare"}, {"git", "prepare"}, {"git", "cached"}}
	for _, e := range entries {
		for _, tmpKind := range []string{"inside", "alias"} {
			t.Run(e.shape+"/"+e.mode+"/"+tmpKind, func(t *testing.T) {
				if e.shape == "git" {
					if _, err := exec.LookPath("git"); err != nil {
						t.Skip("git not on PATH")
					}
				}
				if tmpKind == "alias" && runtime.GOOS == "windows" {
					t.Skip("symlink alias needs unix symlinks")
				}
				repo := t.TempDir()
				mustWrite(t, filepath.Join(repo, "go.mod"), "module example\n", 0o644)
				mustMkdir(t, filepath.Join(repo, "pkg"))
				mustWrite(t, filepath.Join(repo, "pkg", "main.go"), "package pkg\n", 0o644)
				mustMkdir(t, filepath.Join(repo, "tmp"))
				mustWrite(t, filepath.Join(repo, "tmp", "keep.txt"), "keep\n", 0o644)
				// Named like the workspace prefix but a real repo entry.
				mustMkdir(t, filepath.Join(repo, "llmkit-sandbox-notes"))
				mustWrite(t, filepath.Join(repo, "llmkit-sandbox-notes", "x.txt"), "notes\n", 0o644)
				if e.shape == "git" {
					gitInit(t, repo)
				}
				want, err := treeListing(repo)
				if err != nil {
					t.Fatal(err)
				}

				tmpdir := filepath.Join(repo, "tmp")
				if tmpKind == "alias" {
					tmpdir = filepath.Join(t.TempDir(), "alias")
					if err := os.Symlink(filepath.Join(repo, "tmp"), tmpdir); err != nil {
						t.Fatal(err)
					}
				}
				got := runSelfCopyChild(t, e.mode, repo, tmpdir)
				diffListings(t, got, want)
			})
		}
	}
}

// TestCopyFileListNeverCopiesGitlinkDestination pins the recursive gitlink
// copy: a destination inside a submodule working tree that copyFileListSkipping
// recurses into is left out of the copy.
func TestCopyFileListNeverCopiesGitlinkDestination(t *testing.T) {
	for _, tmpKind := range []string{"inside", "alias"} {
		t.Run(tmpKind, func(t *testing.T) {
			if tmpKind == "alias" && runtime.GOOS == "windows" {
				t.Skip("symlink alias needs unix symlinks")
			}
			repo := t.TempDir()
			gt := filepath.Join(repo, "vendor", "gtest")
			mustMkdir(t, filepath.Join(gt, "src"))
			mustMkdir(t, filepath.Join(gt, "tmp"))
			mustWrite(t, filepath.Join(gt, "CMakeLists.txt"), "project(gtest)\n", 0o644)
			mustWrite(t, filepath.Join(gt, ".git"), "gitdir: ../../.git/modules/gtest\n", 0o644)
			mustWrite(t, filepath.Join(gt, "src", "a.cc"), "int a;\n", 0o644)
			mustWrite(t, filepath.Join(gt, "tmp", "keep.txt"), "keep\n", 0o644)
			want, err := treeListing(gt)
			if err != nil {
				t.Fatal(err)
			}
			tmpdir := filepath.Join(gt, "tmp")
			if tmpKind == "alias" {
				tmpdir = filepath.Join(t.TempDir(), "alias")
				if err := os.Symlink(filepath.Join(gt, "tmp"), tmpdir); err != nil {
					t.Fatal(err)
				}
			}
			diffListings(t, runSelfCopyChild(t, "gitlink", repo, tmpdir), want)
		})
	}
}

// TestPrepareWorkspaceCachedNeverCopiesCacheDir pins the wsCache path when
// TMPDIR lies inside a directory the git copy takes as a whole working tree (a
// submodule, or an untracked nested repo): the workspace holds exactly the
// repo's file set from before the call, so neither the cache directory nor the
// pristine it holds appears in it.
func TestPrepareWorkspaceCachedNeverCopiesCacheDir(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	for _, shape := range []string{"submodule", "nestedrepo"} {
		for _, tmpKind := range []string{"inside", "alias"} {
			t.Run(shape+"/"+tmpKind, func(t *testing.T) {
				if tmpKind == "alias" && runtime.GOOS == "windows" {
					t.Skip("symlink alias needs unix symlinks")
				}
				repo := t.TempDir()
				mustWrite(t, filepath.Join(repo, "go.mod"), "module example\n", 0o644)
				mustMkdir(t, filepath.Join(repo, "llmkit-sandbox-notes"))
				mustWrite(t, filepath.Join(repo, "llmkit-sandbox-notes", "x.txt"), "notes\n", 0o644)
				gitInit(t, repo)
				var inner string
				switch shape {
				case "submodule":
					up := t.TempDir()
					mustWrite(t, filepath.Join(up, "lib.c"), "int lib;\n", 0o644)
					gitInit(t, up)
					gitRun(t, repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", up, "vendor/up")
					gitRun(t, repo, "commit", "-q", "-m", "add submodule")
					inner = filepath.Join(repo, "vendor", "up")
				case "nestedrepo":
					// Created after the outer commit, so the outer repo lists it
					// as one untracked directory.
					inner = filepath.Join(repo, "nested")
					mustMkdir(t, inner)
					mustWrite(t, filepath.Join(inner, "n.txt"), "n\n", 0o644)
					gitInit(t, inner)
				}
				mustMkdir(t, filepath.Join(inner, "tmp"))
				mustWrite(t, filepath.Join(inner, "tmp", "keep.txt"), "keep\n", 0o644)
				want, err := treeListing(repo)
				if err != nil {
					t.Fatal(err)
				}

				tmpdir := filepath.Join(inner, "tmp")
				if tmpKind == "alias" {
					tmpdir = filepath.Join(t.TempDir(), "alias")
					if err := os.Symlink(filepath.Join(inner, "tmp"), tmpdir); err != nil {
						t.Fatal(err)
					}
				}
				diffListings(t, runSelfCopyChild(t, "cached", repo, tmpdir), want)
			})
		}
	}
}

// gitRun runs one git command in dir with the same config isolation as
// gitInit.
func gitRun(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_SYSTEM=/dev/null",
		"GIT_TERMINAL_PROMPT=0",
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

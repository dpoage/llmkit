package fsroot

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// fixtureTree builds a small repo tree for path-containment tests and returns
// its root.
//
//	root/
//	  README.md         "hello world\nsecond line\n"
//	  main.go           "package main\nfunc main() { foo() }\n"
//	  pkg/util.go       "package pkg\nfunc foo() int { return 42 }\n"
//	  pkg/data.bin      <NUL bytes>
//	  empty/            (empty dir)
func fixtureTree(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "README.md"), "hello world\nsecond line\n")
	mustWrite(t, filepath.Join(root, "main.go"), "package main\nfunc main() { foo() }\n")
	mustMkdir(t, filepath.Join(root, "pkg"))
	mustWrite(t, filepath.Join(root, "pkg", "util.go"), "package pkg\nfunc foo() int { return 42 }\n")
	mustWrite(t, filepath.Join(root, "pkg", "data.bin"), "\x00\x01\x02foo\x00")
	mustMkdir(t, filepath.Join(root, "empty"))
	return root
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func TestFSRoot_Resolve_Traversal(t *testing.T) {
	root := fixtureTree(t)
	fr, err := NewFSRoot(root)
	if err != nil {
		t.Fatalf("NewFSRoot: %v", err)
	}

	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{"plain file", "README.md", false},
		{"nested file", "pkg/util.go", false},
		{"root via empty", "", false},
		{"dot", ".", false},
		{"forward-slash nested", "pkg/util.go", false},
		{"dotdot escape", "../secret", true},
		{"deep dotdot escape", "pkg/../../secret", true},
		{"absolute path", "/etc/passwd", true},
		{"dotdot only", "..", true},
		{"sneaky dotdot mid", "a/b/../../../x", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := fr.Resolve(tc.path)
			if tc.wantErr {
				if err == nil {
					t.Errorf("resolve(%q) = %q, want error", tc.path, got)
				} else if !errors.Is(err, ErrPathEscape) {
					t.Errorf("resolve(%q) error = %v, want ErrPathEscape", tc.path, err)
				}
				return
			}
			if err != nil {
				t.Errorf("resolve(%q) unexpected error: %v", tc.path, err)
			}
		})
	}
}

func TestFSRoot_SymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink semantics differ on windows")
	}
	root := fixtureTree(t)
	outside := t.TempDir()
	mustWrite(t, filepath.Join(outside, "secret.txt"), "top secret\n")

	// A symlink inside the root pointing at a file outside it.
	link := filepath.Join(root, "escape")
	if err := os.Symlink(filepath.Join(outside, "secret.txt"), link); err != nil {
		t.Fatalf("symlink: %v", err)
	}
	// A symlink inside the root pointing at a directory outside it.
	dirLink := filepath.Join(root, "outdir")
	if err := os.Symlink(outside, dirLink); err != nil {
		t.Fatalf("symlink dir: %v", err)
	}

	fr, err := NewFSRoot(root)
	if err != nil {
		t.Fatalf("NewFSRoot: %v", err)
	}

	if _, err := fr.Resolve("escape"); err == nil {
		t.Error("resolve via file symlink escaping root should fail")
	}
	if _, err := fr.Resolve("outdir/secret.txt"); err == nil {
		t.Error("resolve through dir symlink escaping root should fail")
	}
}

func TestEvalExistingPrefixPath_ExistingPrefixMissingTail(t *testing.T) {
	root := t.TempDir()
	mustMkdir(t, filepath.Join(root, "a", "b"))

	// Only root/a/b exists; the tail (c/d.txt) does not. The longest existing
	// prefix must resolve and the tail must be re-appended unchanged.
	got, err := evalExistingPrefixPath(filepath.Join(root, "a", "b", "c", "d.txt"))
	if err != nil {
		t.Fatalf("evalExistingPrefixPath: %v", err)
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatalf("EvalSymlinks(root): %v", err)
	}
	want := filepath.Join(resolvedRoot, "a", "b", "c", "d.txt")
	if got != want {
		t.Errorf("evalExistingPrefixPath = %q, want %q", got, want)
	}
}

func TestEvalExistingPrefixPath_FullyExisting(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "a")
	mustMkdir(t, dir)

	got, err := evalExistingPrefixPath(dir)
	if err != nil {
		t.Fatalf("evalExistingPrefixPath: %v", err)
	}
	want, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatalf("EvalSymlinks(dir): %v", err)
	}
	if got != want {
		t.Errorf("evalExistingPrefixPath = %q, want %q", got, want)
	}
}

func TestEvalExistingPrefixPath_SymlinkedPrefix(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink semantics differ on windows")
	}
	root := t.TempDir()
	real := filepath.Join(root, "real")
	mustMkdir(t, real)
	link := filepath.Join(root, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	// The symlinked directory exists, so it is resolved even though the final
	// component does not — this is what lets callers catch a symlinked
	// intermediate directory that escapes a containment root.
	got, err := evalExistingPrefixPath(filepath.Join(link, "missing.txt"))
	if err != nil {
		t.Fatalf("evalExistingPrefixPath: %v", err)
	}
	resolvedReal, err := filepath.EvalSymlinks(real)
	if err != nil {
		t.Fatalf("EvalSymlinks(real): %v", err)
	}
	want := filepath.Join(resolvedReal, "missing.txt")
	if got != want {
		t.Errorf("evalExistingPrefixPath = %q, want %q", got, want)
	}
}

// TestFSRoot_SiblingSymlinkBackIntoRoot pins the lexical ".." check: a
// symlink in the ROOT'S PARENT directory pointing back into the root makes an
// escaping literal path resolve back INSIDE the root, so the symlink check
// alone would pass it. Removing the lexical containment check must fail this
// test.
func TestFSRoot_SiblingSymlinkBackIntoRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink semantics differ on windows")
	}
	root := fixtureTree(t)
	// Sibling of the root, pointing back at a directory inside it.
	sibling := filepath.Join(filepath.Dir(root), "backlink")
	if err := os.Symlink(filepath.Join(root, "pkg"), sibling); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	fr, err := NewFSRoot(root)
	if err != nil {
		t.Fatalf("NewFSRoot: %v", err)
	}

	// The literal path climbs out of the root even though its
	// symlink-resolved form lands back inside it. It must be rejected.
	_, err = fr.Resolve("../backlink/util.go")
	if !errors.Is(err, ErrPathEscape) {
		t.Fatalf("resolve(../backlink/util.go) error = %v, want ErrPathEscape", err)
	}

	// The in-root route to the same target stays allowed.
	if _, err := fr.Resolve("pkg/util.go"); err != nil {
		t.Errorf("resolve(pkg/util.go): %v", err)
	}
}

package fsroot

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"
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

	link := filepath.Join(root, "escape")
	if err := os.Symlink(filepath.Join(outside, "secret.txt"), link); err != nil {
		t.Fatalf("symlink: %v", err)
	}
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
	} else if !errors.Is(err, ErrPathEscape) {
		t.Errorf("file symlink escape error = %v, want ErrPathEscape", err)
	}
	if _, err := fr.Resolve("outdir/secret.txt"); err == nil {
		t.Error("resolve through dir symlink escaping root should fail")
	} else if !errors.Is(err, ErrPathEscape) {
		t.Errorf("dir symlink escape error = %v, want ErrPathEscape", err)
	}
}

func TestEvalExistingPrefixPath_ExistingPrefixMissingTail(t *testing.T) {
	root := t.TempDir()
	mustMkdir(t, filepath.Join(root, "a", "b"))

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
	sibling := filepath.Join(filepath.Dir(root), "backlink")
	if err := os.Symlink(filepath.Join(root, "pkg"), sibling); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	fr, err := NewFSRoot(root)
	if err != nil {
		t.Fatalf("NewFSRoot: %v", err)
	}

	_, err = fr.Resolve("../backlink/util.go")
	if !errors.Is(err, ErrPathEscape) {
		t.Fatalf("resolve(../backlink/util.go) error = %v, want ErrPathEscape", err)
	}

	if _, err := fr.Resolve("pkg/util.go"); err != nil {
		t.Errorf("resolve(pkg/util.go): %v", err)
	}
}

// mustSymlink creates target->link or fails the test.
func mustSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		t.Fatalf("symlink %s -> %s: %v", link, target, err)
	}
}

// TestFSRoot_Resolve_DanglingSymlink pins fail-closed resolution for symlink
// chains whose final target does not exist on disk: rejected with
// ErrPathEscape when the chain physically lands outside the root, accepted —
// returning the lexical path — when it lands inside. A ".." in or after a
// link body must be applied to the physically resolved directory, the way
// the kernel does, never collapsed against the spelled-out string. Every
// link target below is deliberately non-existent; a link to an existing
// outside path is already rejected by the existing-prefix check and proves
// nothing about dangling handling. A ".." that follows a missing component
// fails closed by choice: the kernel could not open such a path either, so
// no workable path is lost.
func TestFSRoot_Resolve_DanglingSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink semantics differ on windows")
	}
	root := fixtureTree(t)
	outside := t.TempDir()

	// root/dangling -> <outside>/nonexistent-outside/x (absolute, missing).
	mustSymlink(t, filepath.Join(outside, "nonexistent-outside", "x"),
		filepath.Join(root, "dangling"))
	// root/dotdot-dangling -> ../nonexistent-outside (relative, escaping).
	mustSymlink(t, filepath.Join("..", "nonexistent-outside"),
		filepath.Join(root, "dotdot-dangling"))
	// root/d/dangling -> ../../nonexistent-outside (relative, escaping).
	mustMkdir(t, filepath.Join(root, "d"))
	mustSymlink(t, filepath.Join("..", "..", "nonexistent-outside"),
		filepath.Join(root, "d", "dangling"))
	// root/to-inside -> inside/notyet (relative, inside root, missing); the
	// inside/ directory exists so the resolved path is writable in place.
	mustMkdir(t, filepath.Join(root, "inside"))
	mustSymlink(t, filepath.Join("inside", "notyet"),
		filepath.Join(root, "to-inside"))
	// root/chain-a -> chain-b -> <outside>/nonexistent-outside (2-hop chain).
	mustSymlink(t, "chain-b", filepath.Join(root, "chain-a"))
	mustSymlink(t, filepath.Join(outside, "nonexistent-outside"),
		filepath.Join(root, "chain-b"))
	// root/loop -> loop (self-referential).
	mustSymlink(t, "loop", filepath.Join(root, "loop"))
	// Physical-".." fixtures: each rejected row's write would land OUTSIDE
	// the root if ".." were collapsed against the lexical spelling.
	// root/self -> "."; root/dd1 -> self/../pwned1.txt.
	mustSymlink(t, ".", filepath.Join(root, "self"))
	mustSymlink(t, "self/../pwned1.txt", filepath.Join(root, "dd1"))
	// root/dd2 -> <root>/self/../pwned2.txt (absolute body).
	mustSymlink(t, root+"/self/../pwned2.txt", filepath.Join(root, "dd2"))
	// root/dd3 -> ../pwned3.txt, reached through root/self.
	mustSymlink(t, "../pwned3.txt", filepath.Join(root, "dd3"))
	// root/sub -> <outside>/deep (existing dir); root/dd4 -> sub/../pwned4.txt.
	mustMkdir(t, filepath.Join(outside, "deep"))
	mustSymlink(t, filepath.Join(outside, "deep"), filepath.Join(root, "sub"))
	mustSymlink(t, "sub/../pwned4.txt", filepath.Join(root, "dd4"))
	// root/a/sub -> ".."; root/dd5 -> ../pwned.txt.
	mustMkdir(t, filepath.Join(root, "a"))
	mustSymlink(t, "..", filepath.Join(root, "a", "sub"))
	mustSymlink(t, "../pwned.txt", filepath.Join(root, "dd5"))
	// In-root fixtures: the target stays inside the root even though ".."
	// crosses a symlinked directory on the way.
	// root/in -> x/y (existing); root/back2 -> in/../../notyet2.
	mustMkdir(t, filepath.Join(root, "x", "y", "z"))
	mustSymlink(t, "x/y", filepath.Join(root, "in"))
	mustSymlink(t, "in/../../notyet2", filepath.Join(root, "back2"))
	// root/deep -> x/y/z (existing); root/x/y/z/dl -> ../../../notyet.txt.
	mustSymlink(t, "x/y/z", filepath.Join(root, "deep"))
	mustSymlink(t, "../../../notyet.txt", filepath.Join(root, "x", "y", "z", "dl"))
	// Hop-bound boundary: a straight 40-hop in-root dangling chain is
	// accepted (kernel MAXSYMLINKS parity), 41 hops rejected.
	for i := 1; i < 40; i++ {
		mustSymlink(t, fmt.Sprintf("hop%d", i+1), filepath.Join(root, fmt.Sprintf("hop%d", i)))
	}
	mustSymlink(t, "notyet40.txt", filepath.Join(root, "hop40"))
	for i := 1; i < 41; i++ {
		mustSymlink(t, fmt.Sprintf("over%d", i+1), filepath.Join(root, fmt.Sprintf("over%d", i)))
	}
	mustSymlink(t, "notyet41.txt", filepath.Join(root, "over41"))
	// root/ddgone -> gone/../x: ".." after a missing component fails closed.
	mustSymlink(t, "gone/../x", filepath.Join(root, "ddgone"))

	fr, err := NewFSRoot(root)
	if err != nil {
		t.Fatalf("NewFSRoot: %v", err)
	}

	canonicalRoot, rerr := filepath.EvalSymlinks(root)
	if rerr != nil {
		t.Fatalf("EvalSymlinks(root): %v", rerr)
	}
	parent := filepath.Dir(canonicalRoot)

	tests := []struct {
		name    string
		path    string
		wantErr bool
		// landing is where a kernel write through this path would physically
		// touch; on rejected rows nothing may ever appear there.
		landing string
	}{
		{"dangling to absolute outside", "dangling", true, ""},
		{"dangling to dotdot outside", "dotdot-dangling", true, ""},
		{"deep dangling to dotdot outside", "d/dangling", true, ""},
		{"dangling to missing inside target", "to-inside", false, ""},
		{"dangling then missing subpath", "dangling/sub/file", true, ""},
		{"two-link chain to outside", "chain-a", true, ""},
		{"dotdot after symlinked dir", "dd1", true, filepath.Join(parent, "pwned1.txt")},
		{"dotdot in absolute body after symlinked dir", "dd2", true, filepath.Join(parent, "pwned2.txt")},
		{"dotdot body via symlinked dir", "self/dd3", true, filepath.Join(parent, "pwned3.txt")},
		{"dotdot over symlinked outside dir", "dd4", true, filepath.Join(outside, "pwned4.txt")},
		{"dotdot over symlinked parent", "a/sub/dd5", true, filepath.Join(parent, "pwned.txt")},
		{"control plain dotdot body", "dd5", true, filepath.Join(parent, "pwned.txt")},
		{"dotdot back inside through symlinked dir", "back2", false, ""},
		{"deep dotdot back inside root", "deep/dl", false, ""},
		{"40-hop in-root dangling chain", "hop1", false, ""},
		{"41-hop in-root dangling chain", "over1", true, ""},
		{"dotdot after missing component", "ddgone", true, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := fr.Resolve(tc.path)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("resolve(%q) = %q, want error", tc.path, got)
				}
				if !errors.Is(err, ErrPathEscape) {
					t.Errorf("resolve(%q) error = %v, want ErrPathEscape", tc.path, err)
				}
				// Never write on an error row; the kernel-visible landing
				// spot must stay empty.
				if tc.landing != "" {
					if _, serr := os.Stat(tc.landing); !os.IsNotExist(serr) {
						t.Errorf("landing %q exists after rejected resolve(%q)", tc.landing, tc.path)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("resolve(%q) unexpected error: %v", tc.path, err)
			}
			// The lexical path is returned; NewFSRoot canonicalizes the
			// root through symlinks, so compare against the canonical
			// spelling, and the result must be writable in place.
			want := filepath.Join(canonicalRoot, tc.path)
			if got != want {
				t.Errorf("resolve(%q) = %q, want %q", tc.path, got, want)
			}
			if werr := os.WriteFile(got, []byte("x"), 0o644); werr != nil {
				t.Errorf("write via resolved(%q) = %q: %v", tc.path, got, werr)
			}
		})
	}

	// A symlink loop must be rejected promptly by the hop bound, not hang.
	start := time.Now()
	if _, err := fr.Resolve("loop"); err == nil {
		t.Error("resolve(loop) = nil error, want ErrPathEscape")
	} else if !errors.Is(err, ErrPathEscape) {
		t.Errorf("self-loop error = %v, want ErrPathEscape", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("self-loop resolution took %s, want < 5s", elapsed)
	}
}

// TestFSRoot_Resolve_UnresolvablePrefix pins fail-closed resolution when an
// existing prefix cannot be resolved for a non-symlink reason: a path
// traversing through a regular file (ENOTDIR) or into an unsearchable
// directory (EACCES). Resolve must refuse with ErrPathEscape rather than
// return the lexical path.
func TestFSRoot_Resolve_UnresolvablePrefix(t *testing.T) {
	root := fixtureTree(t)
	fr, err := NewFSRoot(root)
	if err != nil {
		t.Fatalf("NewFSRoot: %v", err)
	}

	t.Run("through regular file", func(t *testing.T) {
		_, err := fr.Resolve("README.md/child")
		if err == nil {
			t.Fatal("resolve(README.md/child) = nil error, want ErrPathEscape")
		}
		if !errors.Is(err, ErrPathEscape) {
			t.Errorf("resolve(README.md/child) error = %v, want ErrPathEscape", err)
		}
	})

	// A non-directory component inside a link body fails the same way: the
	// kernel and filepath.EvalSymlinks both refuse with ENOTDIR, so Resolve
	// must not return the link's lexical spelling. A real directory in the
	// same position stays accepted.
	t.Run("through non-directory in link body", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("symlink semantics differ on windows")
		}
		mustSymlink(t, "README.md", filepath.Join(root, "lf"))
		mustMkdir(t, filepath.Join(root, "dir"))

		tests := []struct {
			name    string
			body    string
			wantErr bool
		}{
			{"dotdot after regular file in body", "README.md/../x.txt", true},
			{"dot after regular file in body", "README.md/.", true},
			{"dotdot after linked regular file", "lf/../y.txt", true},
			{"control dotdot after real directory", "dir/../ok.txt", false},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				mustSymlink(t, tc.body, filepath.Join(root, "dl"))
				t.Cleanup(func() { _ = os.Remove(filepath.Join(root, "dl")) })

				got, err := fr.Resolve("dl")
				if tc.wantErr {
					if err == nil {
						t.Fatalf("resolve(dl -> %q) = %q, want ErrPathEscape", tc.body, got)
					}
					if !errors.Is(err, ErrPathEscape) {
						t.Errorf("resolve(dl -> %q) error = %v, want ErrPathEscape", tc.body, err)
					}
					if !errors.Is(err, syscall.ENOTDIR) {
						t.Errorf("resolve(dl -> %q) error = %v, want ENOTDIR", tc.body, err)
					}
					return
				}
				if err != nil {
					t.Fatalf("resolve(dl -> %q) unexpected error: %v", tc.body, err)
				}
				if werr := os.WriteFile(got, []byte("x"), 0o644); werr != nil {
					t.Errorf("write via resolved(dl -> %q) = %q: %v", tc.body, got, werr)
				}
			})
		}
	})

	t.Run("into unsearchable directory", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("file modes do not restrict access on windows")
		}
		if os.Geteuid() == 0 {
			t.Skip("running as root: directory permissions not enforced")
		}
		locked := filepath.Join(root, "locked")
		mustMkdir(t, locked)
		if err := os.Chmod(locked, 0); err != nil {
			t.Fatalf("chmod: %v", err)
		}
		t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })

		_, err := fr.Resolve("locked/x")
		if err == nil {
			t.Fatal("resolve(locked/x) = nil error, want ErrPathEscape")
		}
		if !errors.Is(err, ErrPathEscape) {
			t.Errorf("resolve(locked/x) error = %v, want ErrPathEscape", err)
		}
	})
}

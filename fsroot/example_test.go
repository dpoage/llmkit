package fsroot_test

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dpoage/llmkit/fsroot"
)

// ExampleNewFSRoot anchors path resolution at a trusted root and shows
// which inputs Resolve accepts and which it refuses.
func ExampleNewFSRoot() {
	dir, err := os.MkdirTemp("", "fsroot-example-")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	defer func() { _ = os.RemoveAll(dir) }()

	if err := os.MkdirAll(filepath.Join(dir, "pkg"), 0o755); err != nil {
		fmt.Println("error:", err)
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "pkg", "util.go"), []byte("package pkg\n"), 0o644); err != nil {
		fmt.Println("error:", err)
		return
	}

	fsr, err := fsroot.NewFSRoot(dir)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// A good relative path resolves to an absolute path under the root.
	// NewFSRoot canonicalizes the root through symlinks, so mirror that
	// here before trimming it back off for display.
	abs, err := fsr.Resolve("pkg/util.go")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	resolvedRoot := dir
	if r, rerr := filepath.EvalSymlinks(dir); rerr == nil {
		resolvedRoot = r
	}
	rel, err := filepath.Rel(resolvedRoot, abs)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("resolved:", rel)

	// ".." escapes the root lexically; an absolute path is refused
	// outright. Both fail with fsroot.ErrPathEscape, matched with
	// errors.Is.
	_, err = fsr.Resolve("../secrets")
	fmt.Println(".. escape refused:", errors.Is(err, fsroot.ErrPathEscape))
	_, err = fsr.Resolve("/etc/passwd")
	fmt.Println("absolute refused:", errors.Is(err, fsroot.ErrPathEscape))

	// Output:
	// resolved: pkg/util.go
	// .. escape refused: true
	// absolute refused: true
}

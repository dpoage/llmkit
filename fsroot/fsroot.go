// Package fsroot provides tool-anchored path containment for agent tools.
// An FSRoot anchors resolution at a single directory and maps
// root-relative paths to absolute on-disk paths, rejecting absolute inputs
// and any path that escapes the root lexically ("..") or through a symlink
// that points outside it — the latter detected by resolving the longest
// existing prefix of the path.
//
// Threat-model note: this package guards path RESOLUTION anchored at a
// trusted root. It is deliberately not unified with llmkit/sandbox's
// workspace write hardening (sanitizeRelPath/secureJoinForWrite,
// O_NOFOLLOW), which defends a different threat model: post-exec writes
// into an untrusted scratch workspace. The two boundaries share no code.
package fsroot

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

// ErrPathEscape is returned, wrapped via fmt.Errorf with %w, when a
// requested path resolves outside the root.
var ErrPathEscape = errors.New("path escapes the repository root")

// FSRoot anchors tool path resolution at a single directory and resolves
// tool-supplied, root-relative paths to absolute on-disk paths while
// guaranteeing they cannot escape the root — including via "..", absolute
// inputs, or symlinks that point outside the tree.
//
// It is the kit's shared containment primitive: any tool that takes
// root-relative paths constructs one FSRoot and delegates resolution to it,
// instead of duplicating the guard per tool.
type FSRoot struct {
	// root is the cleaned, symlink-resolved absolute repository root. All
	// resolved paths must remain within it.
	root string
}

// NewFSRoot resolves dir to a canonical absolute root. It follows symlinks on
// the root itself (so the containment check compares like with like) but does
// not require dir to exist as a directory beyond being resolvable.
func NewFSRoot(dir string) (*FSRoot, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve root %q: %w", dir, err)
	}
	// Canonicalize through any symlinks so containment compares like-with-like.
	// Unresolvable roots fall back to the cleaned absolute path; access will fail naturally.
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		abs = resolved
	}
	return &FSRoot{root: filepath.Clean(abs)}, nil
}

// Resolve maps a model-supplied, repo-relative path to an absolute on-disk path
// inside the root, rejecting absolute inputs and any path that escapes the root
// either lexically ("..") or after symlink resolution.
//
// An empty path resolves to the root itself (useful for list_dir of the repo
// root).
func (r *FSRoot) Resolve(rel string) (string, error) {
	// Normalize separators so callers may use forward slashes regardless of OS.
	rel = filepath.FromSlash(rel)

	// Absolute inputs are rejected: tools take repo-relative paths only.
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("%w: absolute paths are not allowed (%q)", ErrPathEscape, rel)
	}

	joined := filepath.Join(r.root, rel)
	cleaned := filepath.Clean(joined)

	// Lexical containment: cleaned must be the root or under root+sep.
	if !r.contains(cleaned) {
		return "", fmt.Errorf("%w: %q", ErrPathEscape, rel)
	}

	// Symlink containment: resolve the longest existing prefix and ensure it
	// still lands inside the root. Non-existent tails are fine; validation runs
	// only over what exists.
	if resolved, err := evalExistingPrefixPath(cleaned); err == nil {
		if !r.contains(resolved) {
			return "", fmt.Errorf("%w: %q resolves outside the root via symlink", ErrPathEscape, rel)
		}
	}

	return cleaned, nil
}

// contains reports whether p is the root itself or lies beneath it.
func (r *FSRoot) contains(p string) bool {
	if p == r.root {
		return true
	}
	return strings.HasPrefix(p, r.root+string(filepath.Separator))
}

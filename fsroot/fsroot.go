// Package fsroot provides tool-anchored path containment. An FSRoot
// resolves root-relative paths to absolute on-disk paths, rejecting absolute
// inputs and any path that escapes the root lexically ("..") or via a symlink
// pointing outside it (detected by resolving the longest existing prefix,
// following dangling links to their final target, and failing closed when a
// prefix cannot be resolved).
//
// Threat-model note: this package guards path RESOLUTION anchored at a
// trusted root. It is deliberately separate from sandbox's workspace write
// hardening (sanitizeRelPath/secureJoinForWrite, O_NOFOLLOW), which defends a
// different threat model: post-exec writes into an untrusted scratch workspace.
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

// FSRoot resolves root-relative paths to absolute on-disk paths under a
// trusted directory, guaranteeing they cannot escape the root via "..",
// absolute inputs, or symlinks pointing outside the tree.
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

// Resolve maps a repo-relative path to an absolute on-disk path inside the
// root, rejecting absolute inputs and any path that escapes the root either
// lexically ("..") or after symlink resolution. Dangling symlinks are followed
// to their final target — with ".." applied to the physically resolved
// directory, as the kernel does — before the containment decision; any
// prefix that cannot be resolved (non-directory component, unsearchable
// directory, symlink loop) is rejected with ErrPathEscape rather than taken
// at its lexical spelling.
//
// An empty path resolves to the root itself.
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

	// Symlink containment: resolve the longest existing prefix — dangling
	// links included, bounded by the kernel's symlink budget — and require
	// the physical result to stay inside the root.
	resolved, err := evalExistingPrefixPath(cleaned)
	if err != nil {
		return "", fmt.Errorf("%w: %q: %w", ErrPathEscape, rel, err)
	}
	if !r.contains(resolved) {
		return "", fmt.Errorf("%w: %q resolves outside the root via symlink", ErrPathEscape, rel)
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

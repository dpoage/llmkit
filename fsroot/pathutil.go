package fsroot

import "path/filepath"

// EvalExistingPrefixPath resolves symlinks on the longest prefix of p that
// exists on disk, then re-appends the non-existent tail. This lets callers
// validate containment even when the final path component does not yet exist,
// while still catching a symlinked intermediate directory that escapes a root.
//
// It is the single authoritative implementation shared by FSRoot's
// in-root traversal guard and by external traversal guards over other
// configured roots (e.g. dependency-source directories). Keeping one copy
// ensures any hardening applied to this path hits every security boundary
// simultaneously.
func EvalExistingPrefixPath(p string) (string, error) {
	// Walk from the full path up toward the filesystem root, finding the longest
	// prefix that EvalSymlinks can resolve.
	tail := ""
	cur := p
	for {
		if resolved, err := filepath.EvalSymlinks(cur); err == nil {
			if tail == "" {
				return resolved, nil
			}
			return filepath.Join(resolved, tail), nil
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			// Reached the filesystem root without resolving anything.
			return p, nil
		}
		tail = filepath.Join(filepath.Base(cur), tail)
		cur = parent
	}
}

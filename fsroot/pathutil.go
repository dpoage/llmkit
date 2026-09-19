package fsroot

import "path/filepath"

// evalExistingPrefixPath resolves symlinks on the longest prefix of p that
// exists on disk, then re-appends the non-existent tail. FSRoot.Resolve uses
// it to validate containment even when the final path component does not yet
// exist, while still catching a symlinked intermediate directory that escapes
// the root. Keeping one copy ensures any hardening applied to this walk is
// the one containment check relies on.
func evalExistingPrefixPath(p string) (string, error) {
	// Walk upward for the longest prefix EvalSymlinks can resolve.
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

package fsroot

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// maxSymlinkHops bounds symlink hops within one evalExistingPrefixPath call.
// It matches the kernel's per-lookup budget (MAXSYMLINKS), so every chain the
// kernel could resolve is accepted and anything longer (a loop or an over-long
// chain) fails closed.
const maxSymlinkHops = 40

// evalExistingPrefixPath resolves the longest existing prefix of p (absolute),
// applying each component the way the kernel does, and re-appends the
// non-existent tail. FSRoot.Resolve uses it to validate containment even when
// the final component does not yet exist.
//
// dest is the resolved prefix: absolute, clean, symlink-free, so ".." always
// pops a physical directory, never a spelling that may hide a symlink. A
// symlink is replaced by its body — relative bodies apply against dest, the
// link's physical directory — and its components are walked the same way,
// never lexically joined or cleaned. The first missing component starts the
// tail; the rest are appended verbatim, except "..", which fails closed
// (the kernel could not resolve such a path either). Components that cannot
// be inspected (ENOTDIR, unsearchable directory, ENAMETOOLONG) and
// non-final non-directory components fail closed, as the kernel's walk does.
func evalExistingPrefixPath(p string) (string, error) {
	dest := filepath.VolumeName(p) + "/"
	q := strings.TrimPrefix(filepath.ToSlash(p), dest)
	hops := 0
	tail := ""
	for q != "" {
		var comp string
		if i := strings.IndexByte(q, '/'); i >= 0 {
			comp, q = q[:i], q[i+1:]
		} else {
			comp, q = q, ""
		}
		switch comp {
		case "", ".":
			// No-ops, skipped wherever they appear.
		case "..":
			if tail != "" {
				return "", fmt.Errorf("resolve %q: \"..\" after missing component", p)
			}
			// Popping the physical prefix; at the volume root ".." stays,
			// as in the kernel.
			if d := filepath.Dir(dest); d != dest {
				dest = d
			}
		default:
			if tail != "" {
				tail += "/" + comp
				continue
			}
			cand := filepath.Join(dest, comp)
			fi, err := os.Lstat(cand)
			if err != nil {
				if os.IsNotExist(err) {
					// First missing component: the tail starts here and is
					// never inspected further.
					tail = comp
					continue
				}
				// Uninspectable (ENOTDIR, an unsearchable directory,
				// ENAMETOOLONG, ...): fail closed rather than trust the
				// lexical spelling.
				return "", fmt.Errorf("resolve %q: %w", cand, err)
			}
			if fi.Mode()&os.ModeSymlink == 0 {
				// A non-directory has nothing under it: when components
				// still follow, fail ENOTDIR exactly where filepath's walk
				// does.
				if !fi.IsDir() && q != "" {
					return "", fmt.Errorf("resolve %q: %w", cand, syscall.ENOTDIR)
				}
				dest = cand
				continue
			}
			hops++
			if hops > maxSymlinkHops {
				return "", fmt.Errorf("resolve %q: symlink loop or over-long chain", p)
			}
			target, err := os.Readlink(cand)
			if err != nil {
				return "", fmt.Errorf("resolve %q: %w", cand, err)
			}
			// Substitute the body for the link as components to walk: a
			// relative body continues from dest (the link's physical
			// directory, which exists because Lstat did), an absolute one
			// restarts from the volume root. The body is never joined or
			// cleaned — its ".." and deeper links must be applied
			// physically, left to right.
			t := filepath.ToSlash(target)
			if filepath.IsAbs(t) {
				dest = filepath.VolumeName(t) + "/"
				t = strings.TrimPrefix(t, dest)
			}
			switch {
			case t == "":
			case q == "":
				q = t
			default:
				q = t + "/" + q
			}
		}
	}
	return filepath.Join(dest, tail), nil
}

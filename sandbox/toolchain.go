package sandbox

// toolchain.go implements host toolchain provisioning: resolving named
// toolchains (or explicit host directories) into read-only bind mounts, a
// PATH prefix, and provenance fingerprints, so a sandbox image that lacks a
// toolchain (node, python, cargo, ...) can still run it via the host's own
// installation.
//
// # Security posture
//
// Same as any other ROMount (see the package doc and ROMount.Shared): only
// PUBLIC, READ-ONLY content is exposed — the resolved binary and the
// directory that holds its runtime dependencies. NEVER a secret or
// credential directory. Operators are exposing their PATH's resolution of
// the name (or an explicit directory) to untrusted, model-driven code —
// audit accordingly, exactly as for any operator-opted-in mount.
//
// # Resolution algorithm
//
//  1. Each requested entry is either a bare name (resolved via the HOST's
//     `command -v <name>`, i.e. Go's exec.LookPath) or an absolute directory
//     path (used directly, no lookup — lets an operator pin an exact
//     toolchain install outside PATH, e.g. a specific nix store path).
//  2. A resolved executable's symlink chain is followed to its final target
//     (filepath.EvalSymlinks), so nix/asdf/nvm shim layouts resolve to the
//     real toolchain directory rather than a one-file shim.
//  3. The mounted root is the resolved target's containing directory, or
//     that directory's parent when the containing directory is named "bin"
//     — this pulls in sibling lib/ and share/ the runtime needs, in one
//     mount. The ascent only fires when the parent is narrow (a
//     version-manager's own versioned dir, a nix store path); see
//     isOverbroadToolchainRoot for the guard.
//  4. A provenance fingerprint (resolved host path + `<name> --version`,
//     run on the HOST, not in any sandbox) is recorded per toolchain.
//     Hermeticity is knowingly traded for provisioning correctness; the
//     fingerprint is what keeps a verdict attributable to the exact host
//     toolchain build that produced it.
//
// # Per-backend rendering
//
// ResolveHostToolchains is the single implementation of this algorithm, so
// resolution/fingerprinting/PATH handling never drifts between backends.
// Callers pass its ToolchainResolution to WithHostToolchains on NewCLI or
// NewBwrap. The CLI backend renders each mount as `-v host:ctr:ro` and the
// PATH prefix as an `--env PATH=` entry ahead of Spec.Env; the bwrap
// backend renders `--ro-bind host ctr` and `--setenv PATH`.

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// hostToolchainMountRoot is the fixed container path prefix under which every
// resolved host toolchain is mounted, one subdirectory per requested entry,
// so toolchain ContainerPaths cannot collide with each other or with
// conventional dependency-cache mount paths (/modcache, /pipcache, ...).
const hostToolchainMountRoot = "/opt/llmkit-toolchains"

// defaultContainerPath is appended after any resolved toolchain bin
// directories when building the container's PATH override. It mirrors a
// standard Linux distribution's default PATH so images that already ship
// their own toolchains keep working exactly as before when no host
// toolchains are configured or resolve nothing.
const defaultContainerPath = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

// toolchainVersionProbeTimeout bounds the HOST-side `<bin> --version` probe
// used to build a fingerprint. Runs directly on the host (not inside any
// sandbox), so it must not be allowed to hang the caller on a broken
// binary.
const toolchainVersionProbeTimeout = 3 * time.Second

// ToolchainFingerprint records provenance for one resolved host toolchain
// mount: the host path actually mounted and the toolchain's reported version
// string. Recorded in run metadata so a verdict can be attributed to the
// exact host toolchain build that produced it.
type ToolchainFingerprint struct {
	// Name is the requested entry, trimmed: a bare name (e.g. "node") or an
	// explicit directory's absolute path.
	Name string
	// Path is the resolved host directory that was mounted read-only.
	Path string
	// Version is the toolchain's self-reported version string. Best-effort:
	// empty when the `--version` probe failed or does not apply.
	Version string
}

// ToolchainResolution is the result of resolving a set of named host
// toolchains (or explicit directories) for sandbox provisioning.
type ToolchainResolution struct {
	// mounts are read-only bind mounts to add to a sandbox Spec (or a
	// backend's own bind-mount list). Each is Shared=true: host-owned
	// toolchain installs must never be SELinux :Z relabeled (see
	// ROMount.Shared).
	mounts []ROMount
	// pathPrepend is a ":"-joined, request-ordered list of in-container
	// directories (the containing directory of each resolved executable,
	// rewritten under its mount's ContainerPath) to place at the front of
	// PATH inside the container. Empty when no entry resolved to an
	// executable (e.g. every entry was an explicit non-executable dir).
	pathPrepend string
	// Fingerprints records provenance for each successfully resolved
	// toolchain, in request order. An entry that did not resolve has no
	// fingerprint; it is listed in Unresolved instead.
	Fingerprints []ToolchainFingerprint
	// Unresolved lists, in request order, every trimmed non-empty entry
	// that did not resolve on this host (not on PATH, a dangling symlink,
	// or a named directory that does not exist). A blank entry (empty or
	// whitespace-only) is silently skipped — never resolved, never
	// unresolved. Nil when every entry resolved.
	Unresolved []string
}

// ResolveHostToolchains resolves each entry in names into a read-only bind
// mount and a provenance fingerprint, plus an in-container PATH entry when
// the entry resolved to an executable (a bare name; an explicit directory
// contributes no PATH entry).
//
// An entry is either:
//   - a bare name (e.g. "node"): resolved via the host's PATH
//     (exec.LookPath), then its symlink closure is followed to the real
//     toolchain directory (see the file doc's resolution algorithm).
//   - an absolute directory path (starts with "/"): used directly as the
//     mounted root, no PATH lookup or symlink following.
//
// Resolution is best-effort per entry: a name that cannot be resolved on
// the host (not on PATH, dangling symlink, or a named directory that does
// not exist) is skipped and its trimmed name recorded in Unresolved, in
// request order — never an error return. A misconfigured entry degrades
// (the resulting CapabilitySet probe will report that ecosystem
// unavailable); it does not abort the run. Duplicate ContainerPaths are
// de-duplicated; only the first is kept, and a deduplicated entry is NOT
// reported in Unresolved (it did resolve).
func ResolveHostToolchains(names []string) ToolchainResolution {
	var res ToolchainResolution
	var pathDirs []string
	seenContainerPaths := make(map[string]bool, len(names))

	for _, raw := range names {
		name := strings.TrimSpace(raw)
		if name == "" {
			continue
		}
		root, execPath, err := resolveToolchainRoot(name)
		if err != nil {
			res.Unresolved = append(res.Unresolved, name)
			continue
		}
		ctrPath := hostToolchainMountRoot + "/" + sanitizeToolchainSegment(name)
		if seenContainerPaths[ctrPath] {
			continue
		}
		seenContainerPaths[ctrPath] = true

		res.mounts = append(res.mounts, ROMount{
			HostPath:      root,
			ContainerPath: ctrPath,
			Shared:        true, // host-owned toolchain install; never :Z relabeled
		})

		if execPath != "" {
			binDir := ctrPath
			if rel, rerr := filepath.Rel(root, filepath.Dir(execPath)); rerr == nil && rel != "." {
				binDir = filepath.Join(ctrPath, rel)
			}
			pathDirs = append(pathDirs, binDir)
		}

		res.Fingerprints = append(res.Fingerprints, ToolchainFingerprint{
			Name:    name,
			Path:    root,
			Version: probeToolchainVersion(name, execPath),
		})
	}

	res.pathPrepend = strings.Join(pathDirs, ":")
	return res
}

// resolveToolchainRoot resolves name to (mountedRootDir, resolvedExecPath).
// For an absolute-path entry, execPath is "" (no binary identified; the
// directory itself is mounted verbatim) and root is name, validated to exist
// and be a directory. For a bare name, root is the containing directory of
// the symlink-closure-resolved executable (see the file doc's step 3), and
// execPath is that resolved executable path.
func resolveToolchainRoot(name string) (root, execPath string, err error) {
	if filepath.IsAbs(name) {
		info, statErr := os.Stat(name)
		if statErr != nil || !info.IsDir() {
			return "", "", fmt.Errorf("sandbox: host toolchain dir %q not found: %w", name, statErr)
		}
		return name, "", nil
	}

	found, lookErr := exec.LookPath(name)
	if lookErr != nil {
		return "", "", fmt.Errorf("sandbox: host toolchain %q not on PATH: %w", name, lookErr)
	}
	resolved, evalErr := filepath.EvalSymlinks(found)
	if evalErr != nil {
		return "", "", fmt.Errorf("sandbox: resolve symlink closure for %q: %w", name, evalErr)
	}
	dir := filepath.Dir(resolved)
	root = dir
	if filepath.Base(dir) == "bin" {
		// Pull in the toolchain root (sibling lib/, share/, ...) alongside
		// the bin/ directory, but only when that root is narrow (a
		// version-manager's own versioned directory, a nix store path, ...).
		// A $HOME/bin/node or ~/.local/bin/node layout would otherwise
		// ascend to $HOME or ~/.local and RO-mount the user's entire home
		// directory (SSH keys, git credentials, unrelated dotfiles) into
		// whatever untrusted, model-driven code the sandbox runs. See
		// isOverbroadToolchainRoot.
		if candidate := filepath.Dir(dir); !isOverbroadToolchainRoot(candidate) {
			root = candidate
		}
	}
	return root, resolved, nil
}

// isOverbroadToolchainRoot reports whether dir is a shared, multi-purpose
// directory that must never be RO-mounted wholesale as a "toolchain root":
// the user's home directory itself, or a broad catch-all subdirectory like
// ~/.local that holds far more than one toolchain. A $HOME/bin/node or
// ~/.local/bin/node layout ascends exactly here without this guard.
// Narrow, single-purpose version-manager directories
// (~/.nvm/versions/node/vX, ~/.asdf/installs/..., a nix store path) are NOT
// caught by this — they are exactly the layout the ascent exists to support.
func isOverbroadToolchainRoot(dir string) bool {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return false
	}
	home = filepath.Clean(home)
	dir = filepath.Clean(dir)
	if dir == home {
		return true
	}
	for _, broad := range []string{".local", ".config", ".cache", ".ssh"} {
		if dir == filepath.Join(home, broad) {
			return true
		}
	}
	return false
}

// sanitizeToolchainSegment reduces name to a single, safe path component
// for use under hostToolchainMountRoot: the base name only, so an absolute
// directory entry (e.g. "/nix/store/xxx-nodejs-18") mounts at a flat
// "/opt/llmkit-toolchains/xxx-nodejs-18" rather than a nested, traversal-prone
// path. A bare name is used as-is (it is already a single component).
func sanitizeToolchainSegment(name string) string {
	base := filepath.Base(name)
	if base == "" || base == "." || base == "/" || base == string(filepath.Separator) {
		base = "toolchain"
	}
	return base
}

// probeToolchainVersion best-effort runs `<bin> --version` on the HOST —
// not inside any sandbox; this inspects the host toolchain being mounted,
// before any container exists — and returns its first output line,
// trimmed. Returns "" on any failure (binary rejects --version, times
// out, etc.); the fingerprint remains useful with just the resolved path.
func probeToolchainVersion(name, execPath string) string {
	bin := execPath
	if bin == "" {
		bin = name
	}
	ctx, cancel := context.WithTimeout(context.Background(), toolchainVersionProbeTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, bin, "--version").Output()
	if err != nil {
		return ""
	}
	line, _, _ := strings.Cut(string(out), "\n")
	return strings.TrimSpace(line)
}

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
//  1. An absolute entry names a directory, used directly with no lookup
//     (lets an operator pin an exact toolchain install outside PATH, e.g. a
//     specific nix store path); one whose cleaned path is not an existing
//     directory is unresolved. Any other entry is resolved to an
//     executable: a bare name through the HOST's PATH with Go's
//     exec.LookPath, a relative path containing a slash as
//     ResolveHostToolchains states.
//  2. A resolved executable's symlink chain is followed to its final target
//     (filepath.EvalSymlinks), so nix/asdf/nvm shim layouts resolve to the
//     real toolchain directory rather than a one-file shim.
//  3. For an entry resolved to an executable, the mounted root is the
//     resolved target's containing directory, or that directory's parent
//     when the containing directory is named "bin" and the parent is not
//     one of the directories isOverbroadToolchainRoot refuses; mounting the
//     parent brings sibling directories (lib/, share/, ...) along in one
//     mount. An executable entry whose resolved target sits directly in one
//     of those refused directories is unresolved. An absolute directory entry is
//     itself the mounted root, after filepath.Clean.
//  4. A provenance fingerprint (resolved host path + `<name> --version`,
//     run on the HOST, not in any sandbox) is recorded per mount.
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
	"slices"
	"strings"
	"syscall"
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
	// Name is the requested entry, trimmed, as written (a bare name such as
	// "node", a relative path, or an absolute directory).
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
	// Fingerprints records provenance for each toolchain mounted, in
	// request order. An entry that did not resolve has no fingerprint (it
	// is listed in Unresolved instead), nor does one dropped as a duplicate
	// ContainerPath (see ResolveHostToolchains).
	Fingerprints []ToolchainFingerprint
	// Unresolved lists, in request order, every trimmed non-empty entry
	// that did not resolve on this host (see ResolveHostToolchains). A
	// blank entry (empty or whitespace-only) is silently skipped — never
	// resolved, never unresolved. Nil when every entry resolved.
	Unresolved []string
}

// ResolveHostToolchains resolves the entries in names into read-only bind
// mounts, each with a provenance fingerprint, plus an in-container PATH
// entry for each mount whose entry resolved to an executable (a
// non-absolute entry; an explicit directory contributes no PATH entry).
//
// An entry is one of:
//   - a bare name (e.g. "node"): resolved via the host's PATH
//     (exec.LookPath), then its symlink closure is followed to the real
//     toolchain directory (see the file doc's resolution algorithm).
//   - a relative path containing a slash (e.g. "./node", "bin/node",
//     "../tools/bin/node"): resolved to the absolute, symlink-free path of
//     the file the kernel would execute for it from the process's working
//     directory at the time of the call (exec.LookPath, no PATH search),
//     then handled like a bare name's executable.
//   - an absolute directory path (starts with "/"): cleaned
//     (filepath.Clean), then used directly as the mounted root, no PATH
//     lookup or symlink following.
//
// Resolution is best-effort per entry and never returns an error. Each of
// these entries is skipped and its trimmed name recorded in Unresolved, in
// request order: a non-absolute entry exec.LookPath does not find (not on
// PATH, a dangling symlink, a relative path that does not exist); a
// non-absolute entry whose resolved executable sits directly in /, /usr,
// /usr/local, /opt, $HOME or $HOME/.local, .config, .cache or .ssh (each
// compared symlink-resolved); an absolute entry whose cleaned path is not
// an existing directory. A misconfigured entry degrades
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
// directory itself is mounted verbatim) and root is name after
// filepath.Clean; that cleaned path is the one checked to exist and be a
// directory. Any other entry is looked up with exec.LookPath; the result's
// symlinks are resolved (filepath.EvalSymlinks) before a result that is
// still relative is joined onto syscall.Getwd, which is how a relative
// entry reaches the path ResolveHostToolchains states. root is then the
// containing directory of that resolved executable (see the file doc's
// step 3), and execPath is the resolved executable path; both are
// absolute. An entry whose executable sits directly in an
// overbroad directory (see isOverbroadToolchainRoot) is an error, since no
// narrower mount exists.
func resolveToolchainRoot(name string) (root, execPath string, err error) {
	if filepath.IsAbs(name) {
		root = filepath.Clean(name)
		info, statErr := os.Stat(root)
		if statErr != nil || !info.IsDir() {
			return "", "", fmt.Errorf("sandbox: host toolchain dir %q not found: %w", root, statErr)
		}
		return root, "", nil
	}

	found, lookErr := exec.LookPath(name)
	if lookErr != nil {
		return "", "", fmt.Errorf("sandbox: host toolchain %q not on PATH: %w", name, lookErr)
	}
	resolved, evalErr := filepath.EvalSymlinks(found)
	if evalErr != nil {
		return "", "", fmt.Errorf("sandbox: resolve symlink closure for %q: %w", name, evalErr)
	}
	if !filepath.IsAbs(resolved) {
		// EvalSymlinks walked the relative path from the working directory
		// as the kernel does; the relative result is symlink-free, with any
		// ".." only leading, so it is joined onto the kernel's working
		// directory. Not os.Getwd: that may answer with $PWD, a path
		// reaching the same directory through a symlink, whose lexical
		// parent is a different directory.
		wd, wdErr := syscall.Getwd()
		if wdErr != nil {
			return "", "", fmt.Errorf("sandbox: working directory for host toolchain %q: %w", name, wdErr)
		}
		resolved = filepath.Join(wd, resolved)
	}
	dir := filepath.Dir(resolved)
	if isOverbroadToolchainRoot(dir) {
		return "", "", fmt.Errorf("sandbox: host toolchain %q resolves directly into the shared directory %q", name, dir)
	}
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

// sharedToolchainRoots are the system-wide directories isOverbroadToolchainRoot
// refuses as a toolchain root.
var sharedToolchainRoots = []string{"/", "/usr", "/usr/local", "/opt"}

// isOverbroadToolchainRoot reports whether dir is a shared, multi-purpose
// directory that must never be RO-mounted wholesale as a "toolchain root":
// a system-wide root ("/", /usr, /usr/local, /opt), the user's home
// directory itself, or a broad catch-all subdirectory like ~/.local that
// holds far more than one toolchain. A $HOME/bin/node or ~/.local/bin/node
// layout ascends to the latter, /usr/bin/python3 and /usr/local/bin/node to
// the former, without this guard; the caller then mounts the bin directory
// alone. dir and every listed directory are compared both as written and
// symlink-resolved, so a root reached through a symlink (a symlinked
// $HOME, /usr/local -> /var/usrlocal) is still caught.
// Narrow, single-purpose version-manager directories
// (~/.nvm/versions/node/vX, ~/.asdf/installs/..., /opt/node-vX, a nix store
// path) are NOT caught by this — they are exactly the layout the ascent
// exists to support.
func isOverbroadToolchainRoot(dir string) bool {
	listed := slices.Clone(sharedToolchainRoots)
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		listed = append(listed, home)
		for _, broad := range []string{".local", ".config", ".cache", ".ssh"} {
			listed = append(listed, filepath.Join(home, broad))
		}
	}
	candidates := symlinkForms(dir)
	for _, l := range listed {
		for _, form := range symlinkForms(l) {
			if slices.Contains(candidates, form) {
				return true
			}
		}
	}
	return false
}

// symlinkForms returns p cleaned and, when it resolves, p with every
// symlink resolved.
func symlinkForms(p string) []string {
	forms := []string{filepath.Clean(p)}
	if resolved, err := filepath.EvalSymlinks(p); err == nil && resolved != forms[0] {
		forms = append(forms, resolved)
	}
	return forms
}

// sanitizeToolchainSegment reduces name to a single, safe path component
// for use under hostToolchainMountRoot: the base name of the cleaned entry,
// so an absolute directory entry (e.g. "/nix/store/xxx-nodejs-18") mounts at
// a flat "/opt/llmkit-toolchains/xxx-nodejs-18" rather than a nested,
// traversal-prone path. A bare name is used as-is (it is already a single
// component). An entry whose cleaned base is not a real name ("/", ".",
// "..", e.g. "/x/..") mounts at "toolchain", so the result is always a
// direct child of hostToolchainMountRoot — never the root itself or its
// parent.
func sanitizeToolchainSegment(name string) string {
	base := filepath.Base(filepath.Clean(name))
	if base == "" || base == "." || base == ".." || base == "/" || base == string(filepath.Separator) {
		base = "toolchain"
	}
	return base
}

// checkCLIToolchainMounts refuses a host-toolchain mount the CLI backend
// cannot render: a HostPath or ContainerPath the runtime's `-v host:ctr:opts`
// syntax cannot express (cliPathExpressible) would make the runtime abort
// every run with an "incorrect volume format" error. Known at construction,
// so NewCLI refuses it there.
func checkCLIToolchainMounts(mounts []ROMount) error {
	for _, m := range mounts {
		for _, p := range []string{m.HostPath, m.ContainerPath} {
			if !cliPathExpressible(p) {
				return fmt.Errorf("sandbox: cli backend cannot render WithHostToolchains mount %q -> %q: %q contains ':', which the container runtime's -v syntax cannot express", m.HostPath, m.ContainerPath, p)
			}
		}
	}
	return nil
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

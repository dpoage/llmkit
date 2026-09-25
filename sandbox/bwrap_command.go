package sandbox

import (
	"path/filepath"
	"strconv"
	"strings"
)

// bwrapParams is the fully-resolved set of inputs to a single bwrap run,
// after backend defaults have been applied to a Spec. It mirrors runParams
// (command.go) so the two backends stay structurally comparable, but carries
// no containerName/image — bwrap has neither (Exec refuses a non-empty
// Spec.Image).
type bwrapParams struct {
	// workspace: host path of the prepared rw workspace bound at /workspace
	// inside the sandbox.
	workspace string
	network   NetworkMode
	env       []string
	cmd       []string
	// roMounts: extra read-only binds (e.g. a dependency cache), rendered
	// after the fixed toolchain allowlist. Shared is a no-op for bwrap (no
	// SELinux relabeling) and ignored.
	roMounts []ROMount
	// rwMounts: extra writable binds (the trusted dependency-prefetch
	// step's cache dirs, operator-configured writable mounts).
	rwMounts []ROMount
	// setupCmds: optional in-sandbox commands run before cmd. When non-empty
	// the backend wraps execution in /bin/sh, exactly like buildRunArgs.
	setupCmds [][]string
	// toolchainBinds: additional read-only binds resolved by the
	// host-toolchain resolver (WithHostToolchains) for toolchains beyond the
	// fixed allowlist below. Nil when no host toolchains are configured; the
	// fixed allowlist alone is sufficient for base functionality.
	toolchainBinds []ROMount
	// toolchainPathPrepend: ResolveHostToolchains' pathPrepend for the same
	// resolution that produced toolchainBinds — in-sandbox directories placed
	// at the front of PATH so resolved toolchain binaries are reachable.
	// Empty when no toolchains were configured/resolved.
	toolchainPathPrepend string
	// baselinePathAppend: ":"-joined in-sandbox directories of the resolved
	// POSIX baseline utilities (resolveBwrapBaseline), appended AFTER
	// defaultContainerPath — and after any caller-supplied PATH in env — so
	// core utilities stay reachable without ever shadowing allowlist
	// binaries or operator toolchains. Empty on FHS hosts.
	baselinePathAppend string
	// scratchSizeBytes: size of the writable tmpfs scratch space, applied
	// via bwrap's --size flag to BOTH the tmpfs root ("/") and /tmp. Always
	// positive: the constructor refuses a <= 0 WithScratchSizeMB override,
	// so buildBwrapArgs applies no fallback here.
	scratchSizeBytes int64
}

// fixedROAllowlist is the minimal, hardcoded set of host directories bound
// read-only into every bwrap sandbox so a bare-bones toolchain (a shell, libc,
// dynamic linker, core utilities) is reachable. This is intentionally narrow:
// broader binds (a whole $HOME, /root, or /etc) can exfiltrate host secrets
// through workspace -> transcript -> LLM even under network=none, since the
// untrusted command can read anything bound in and the sandbox output is fed
// back to a model. Toolchain-specific binds (go, node, python, ...) are
// layered on top via bwrapParams.toolchainBinds, never by widening this list.
//
// Bound with --ro-bind-try (not --ro-bind): non-FHS hosts (NixOS, Guix)
// genuinely lack /lib, /sbin, or even /bin as real paths — a strict
// --ro-bind on an absent path makes bwrap exit 1 before the sandboxed
// command ever runs. A missing entry here just means that slice contributes
// nothing; it is still narrower than the container backend's baked image.
//
// /etc/resolv.conf is deliberately absent here: it is added only when the
// network is enabled (see buildBwrapArgs), matching the container backend's
// contract that DNS resolution is unavailable under network=none.
//
// /nix/store, /gnu/store, and /etc/static exist for store-based distros
// (NixOS, Guix), where the FHS paths above are symlink farms into the store:
// /bin/sh -> /nix/store/...-bash/bin/sh. Binding /bin alone carries the
// symlink into the sandbox but not its target, so exec fails with
// "execvp /bin/sh: No such file or directory". Binding the store roots
// read-only resolves that class wholesale (shells, env, and the rpath lib
// closures of any store-resolved toolchain) without widening the
// secret-exfiltration surface: both stores are world-readable by design on
// their distros, so the sandboxed code gains no read access it would not
// already have running unsandboxed as the same user. /etc/static is NixOS's
// symlink-farm indirection into /nix/store (e.g.
// /etc/ssl/certs -> /etc/static/ssl/certs), needed so the /etc/ssl bind above
// resolves. On FHS hosts none of the three exist and the --ro-bind-try is a
// no-op.
//
// /etc/alternatives is Debian/Ubuntu's update-alternatives indirection:
// /usr/bin/awk -> /etc/alternatives/awk -> /usr/bin/gawk (likewise which,
// editor, pager, java, python-config, ...). Without it those symlinks dangle
// inside the sandbox and every alternatives-managed tool fails with
// "not found". The directory holds only dpkg-managed symlinks (no secrets),
// so binding it read-only widens nothing that /usr does not already expose.
var fixedROAllowlist = []string{
	"/usr",
	"/lib",
	"/lib64",
	"/bin",
	"/sbin",
	"/etc/ssl",
	"/etc/static",
	"/nix/store",
	"/gnu/store",
	"/etc/alternatives",
}

// buildBwrapArgs constructs the argv passed to the bwrap binary for a single
// run. It is a pure function so the security-relevant flag construction can
// be exercised in unit tests without bubblewrap or a Linux host.
//
// Security posture encoded here (defense in depth for untrusted, model-driven
// code — mirrors buildRunArgs' contract for the container backend):
//   - --unshare-all             : unshare every namespace (user, pid, net,
//     ipc, uts, cgroup) by default; --share-net is added back ONLY when the
//     resolved network mode explicitly enables it (never the "none" default).
//   - --die-with-parent         : bwrap's child is killed if the sandboxing
//     process dies, so a crashed harness can never leave an orphaned
//     sandboxed process running.
//   - --new-session             : the sandboxed process gets its own session,
//     so it cannot inject input into (or steal signals from) the calling
//     process's controlling terminal.
//   - --clearenv + --setenv     : the sandbox starts with NO inherited host
//     environment; every variable it sees is explicit, mirroring --env on the
//     container backend. HOME=/tmp, USER=sandbox, and LOGNAME=sandbox are set
//     first as the sandbox's defaults, but p.env is rendered afterward and
//     DOES win on a repeat --setenv of any of them (bwrap's env map is
//     last-write-wins, exactly like buildRunArgs' --env HOME=/tmp — an
//     operator explicitly setting HOME/USER/LOGNAME in Spec.Env overrides the
//     default, not the other way around). USER/LOGNAME exist because the
//     tmpfs root has no /etc/passwd: bazel's client launcher hard-fails with
//     "FATAL: $USER is not set" without them, and POSIX tools generally
//     expect USER to be set.
//   - --size N --tmpfs / FIRST  : the root filesystem is an empty, SIZED
//     tmpfs, established BEFORE any subpath (--proc, --dev, --tmpfs /tmp,
//     the allowlist, workspace) is bound — bwrap applies mount operations in
//     argv order within one shared mount namespace, so mounting "/" AFTER
//     something is already mounted at a subpath (e.g. /tmp) shadows that
//     subpath's mount entirely: the new root's own (empty) /tmp directory
//     wins, silently making the "earlier" /tmp completely inaccessible.
//     The size is REQUIRED here too: every directory bwrap does not bind
//     something else over remains part of this writable tmpfs, so an unsized
//     root is just as real a RAM-DoS surface as an unsized /tmp when the
//     allow-uncapped override is set.
//   - --proc /proc, --dev /dev  : minimal, namespace-scoped pseudo-filesystems
//     (no host /proc or /dev is ever bound), layered onto the tmpfs root.
//   - --size N --tmpfs /tmp     : writable scratch space for language
//     toolchain caches. bwrap's --size flag applies to the SINGLE --tmpfs
//     invocation immediately following it, never cumulatively — hence it is
//     repeated before each of the two --tmpfs flags above and below, both
//     driven by the SAME p.scratchSizeBytes (the WithScratchSizeMB value;
//     always positive — see bwrapParams.scratchSizeBytes).
//   - --ro-bind-try allowlist   : ONLY the fixed allowlist (fixedROAllowlist)
//     plus any resolved toolchain/extra RO mounts are bound in, read-only —
//     best-effort (--ro-bind-try) since non-FHS hosts genuinely lack some
//     allowlist paths, see fixedROAllowlist's doc. No wholesale $HOME,
//     /root, or /etc bind ever happens.
//   - --bind ws /workspace      : the workspace copy is the ONLY writable
//     mount. The original repo is never mounted.
//   - --chdir /workspace        : run from the workspace.
//
// Resource caps (memory/CPU/pids) have no bwrap flag equivalent — bwrap has
// no cgroups of its own — so they are applied by the caller wrapping this
// argv in a systemd-run --user --scope or cgroup v2 invocation; see
// bwrap_caps.go.
func buildBwrapArgs(p bwrapParams) []string {
	scratchSize := strconv.FormatInt(p.scratchSizeBytes, 10)

	args := []string{
		"--unshare-all",
		"--die-with-parent",
		"--new-session",
		"--clearenv",
		"--setenv", "HOME", "/tmp",
		"--setenv", "USER", "sandbox",
		"--setenv", "LOGNAME", "sandbox",
	}

	// Network defaults to unshared (set by --unshare-all above). Only an
	// explicitly enabling network mode restores it — "none" (the default)
	// and the empty string both stay unshared.
	if bwrapNetworkEnabled(p.network) {
		args = append(args, "--share-net")
	}

	// tmpfs root MUST be established before anything else is bound under it
	// (see the doc above) — everything not explicitly bound below is then
	// absent, not merely read-only. This is the bwrap analogue of
	// --read-only + --tmpfs /tmp on the container backend, except there is
	// no underlying image filesystem to fall back to at all. --size applies
	// to the SINGLE --tmpfs that immediately follows it, so it must be
	// repeated here rather than hoisted once for both tmpfs mounts.
	args = append(args, "--size", scratchSize, "--tmpfs", "/")

	args = append(args, "--proc", "/proc", "--dev", "/dev")
	args = append(args, "--size", scratchSize, "--tmpfs", "/tmp")

	for _, host := range fixedROAllowlist {
		args = append(args, "--ro-bind-try", host, host)
	}
	if bwrapNetworkEnabled(p.network) {
		// DNS resolution is meaningless (and unreachable) under the unshared
		// network namespace, so /etc/resolv.conf is bound only when network
		// access was explicitly requested.
		args = append(args, "--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf")
	}

	// Toolchain binds resolved beyond the fixed allowlist (the host-toolchain
	// resolver) are rendered next, still read-only.
	for _, m := range p.toolchainBinds {
		args = append(args, "--ro-bind", m.HostPath, m.ContainerPath)
	}

	// Operator-supplied extra mounts, same ordering contract as buildRunArgs:
	// read-only mounts first, in caller order, always read-only regardless
	// of Shared (bwrap has no SELinux relabeling concept, so Shared is a
	// no-op here — plain ro binds either way).
	for _, m := range p.roMounts {
		args = append(args, "--ro-bind", m.HostPath, m.ContainerPath)
	}
	for _, m := range p.rwMounts {
		args = append(args, "--bind", m.HostPath, m.ContainerPath)
	}

	// The writable workspace copy is the only writable mount, bound last so
	// it always wins if an operator-supplied mount collides on
	// ContainerPath (validateMounts already rejects intra-set duplicates;
	// this is defense in depth against a mount aimed at /workspace itself).
	args = append(args, "--bind", p.workspace, WorkspaceMount)
	args = append(args, "--chdir", WorkspaceMount)

	// PATH must be set explicitly: --clearenv wipes it along with everything
	// else, so without this every in-sandbox command would silently fall
	// back to the shell's compiled-in default path, which never includes
	// resolved toolchain directories. toolchainPathPrepend (from the
	// host-toolchain resolver's pathPrepend) goes first so resolved
	// toolchains shadow any same-named binary under the fixed allowlist;
	// defaultContainerPath (same constant the container backend's toolchain
	// wiring uses) is always the tail so plain allowlisted binaries stay
	// reachable even with no toolchains configured; baselinePathAppend comes
	// last so POSIX baseline utilities are a pure fallback, never shadowing
	// either. p.env is applied after this, so an operator who sets PATH
	// explicitly in Spec.Env still wins — but see the env loop below: the
	// baseline is re-appended to a caller PATH too.
	path := defaultContainerPath
	if p.toolchainPathPrepend != "" {
		path = p.toolchainPathPrepend + ":" + defaultContainerPath
	}
	path = appendBaselinePath(path, p.baselinePathAppend)
	args = append(args, "--setenv", "PATH", path)

	// --clearenv leaves the sandbox with no environment at all; every
	// variable it sees must be set explicitly here, mirroring --env on the
	// container backend. HOME/USER/LOGNAME's defaults (set above) ARE
	// overridable here: an operator entry for any of them in p.env renders
	// a second --setenv, and bwrap's env map is last-write-wins, so it
	// wins — same contract as buildRunArgs' --env HOME=/tmp. An entry
	// without "=" cannot reach this loop: validateSpec refuses it before the
	// run.
	//
	// PATH is the one variable that is rewritten rather than passed
	// verbatim: the POSIX baseline (see baselinePathAppend) is appended to
	// a caller-supplied PATH as well. Container images make core utilities
	// reachable under any PATH the caller constructs because every internal
	// constructor ends with the defaultContainerPath tail and images
	// populate those directories; on store-based hosts those directories
	// hold no utilities, so a caller-supplied value like the capability
	// prober's "<toolchains>:<default>" would silently lose mkdir/grep/...
	// without this append. The baseline is a strict suffix — a caller PATH
	// still shadows everything in it.
	for _, e := range p.env {
		// Every entry here has an "="; validateSpec refused any without.
		key, value, _ := strings.Cut(e, "=")
		if key == "PATH" {
			value = appendBaselinePath(value, p.baselinePathAppend)
		}
		args = append(args, "--setenv", key, value)
	}

	// The command ALWAYS execs through /bin/sh with a setup script; with no
	// SetupCmds the script is exactly `exec "$@"`, so the wrapper is
	// behavior-neutral for a runnable command — but it is what turns a
	// MISSING or NON-EXECUTABLE Spec.Cmd into the shell's exit 127/126
	// instead of bwrap's own execvp failure (which exits 1), matching the
	// container backend's launch-failure contract.
	script := buildSetupScript(p.setupCmds)
	args = append(args, "/bin/sh", "-c", script, "sh")
	args = append(args, p.cmd...)
	return args
}

// bwrapNetworkEnabled reports whether the resolved network mode is an
// explicit opt-in to network access. Only NetworkHost enables it;
// NetworkNone (the resolved default) keeps the network namespace unshared.
// Any other mode is refused before this point by resolveNetworkMode — bwrap
// has no equivalent of the container backend's bridge mode, since
// --unshare-all either shares the host's single network namespace wholesale
// or not at all.
func bwrapNetworkEnabled(network NetworkMode) bool {
	return network == NetworkHost
}

// bwrapAllowlistContainerPaths returns the fixed allowlist's container paths,
// used by validateBwrapMounts to reject an operator-supplied mount that would
// collide with (and silently shadow or be shadowed by) a fixed bind.
func bwrapAllowlistContainerPaths() map[string]bool {
	seen := make(map[string]bool, len(fixedROAllowlist)+1)
	for _, p := range fixedROAllowlist {
		seen[p] = true
	}
	seen[WorkspaceMount] = true
	return seen
}

// validateBwrapMounts extends validateMounts (command.go) with the one rule
// specific to bwrap's tmpfs-root model: no caller-supplied mount may target
// a path already claimed by the fixed allowlist or the workspace, since
// bwrap binds are applied in a fixed order and a collision would silently
// shadow (or be shadowed by) sandbox-critical content. The universal shape
// checks live in validateMounts (called first by validateSpec); the
// collision is a THIS-BACKEND refusal: *UnsupportedSpecError.
func validateBwrapMounts(ro, rw []ROMount) error {
	reserved := bwrapAllowlistContainerPaths()
	check := func(mounts []ROMount, field string) error {
		for _, m := range mounts {
			if reserved[filepath.Clean(m.ContainerPath)] {
				return &UnsupportedSpecError{Backend: backendBwrap, Field: field, Value: m.ContainerPath}
			}
		}
		return nil
	}
	if err := check(ro, "ROMounts"); err != nil {
		return err
	}
	return check(rw, "RWMounts")
}

// appendBaselinePath appends the POSIX-baseline directories to a PATH value
// unless they are already its suffix (a repeated Spec.Env PATH entry, or a
// caller that composed the effective PATH itself). When no baseline
// resolved on this host (the FHS case), the incoming value is returned
// untouched, keeping standard-distro argv byte-identical.
func appendBaselinePath(current, baseline string) string {
	if baseline == "" {
		return current
	}
	if current == "" {
		return baseline
	}
	if current == baseline || strings.HasSuffix(current, ":"+baseline) {
		return current
	}
	return current + ":" + baseline
}

// Package sandbox provides an isolated execution layer for running
// untrusted, model-generated commands against snapshots of arbitrary
// repositories, through one interface:
//
//	Sandbox: Exec(ctx, Spec) (Result, error)
//
// # Backends
//
//   - NewBwrap (*Bwrap): an unprivileged Bubblewrap sandbox on Linux. The
//     command runs against host toolchains directly (no image): bwrap
//     unshares everything, mounts a sized tmpfs root, and binds only a
//     narrow read-only allowlist back into view. DetectBwrap reports whether
//     the host can run it; NewBwrap fails fast with an actionable reason
//     when it cannot — including when no resource-limit mechanism (a
//     systemd-run user scope or a delegated cgroup v2 subtree) is available,
//     which WithBwrapAllowUncapped deliberately opts out of.
//     DescribeBwrapCapMethod reports which mechanism (if any) enforces
//     memory/CPU/pids limits.
//   - NewCLI (*CLI): a container-runtime backend (podman, then docker) driven
//     through its CLI, so the implementation stays daemon-API-free and
//     standard-library only. Detect reports the first available runtime.
//   - NewHostExec (*HostExec): the attended escape hatch. It provides NO
//     isolation — no container, no network policy, no resource caps; the
//     command runs as the calling OS user. It exists only for attended,
//     operator-opted-in paths where a human is watching the exact command
//     they chose to run. Nothing in this package, its defaults, or its
//     examples constructs one.
//   - NewMock (*Mock): a scriptable Sandbox for tests; it enqueues responses
//     and records the calls it receives, so callers can be tested without a
//     real container runtime or bwrap.
//
// # Capability probes
//
// Before planning work against an unfamiliar sandbox, a caller can measure
// what it can actually run: ProbeCapabilities executes the caller's
// []ProbeEntry table (each entry names a capability, supplies the argv that
// measures it, and interprets the probe's exit code/stdout into named
// modes) inside the sandbox and returns the result as a CapabilitySet,
// cached per process. The kit ships no probe entries — what to probe and
// how to interpret it is entirely the caller's knowledge; this package
// only runs the argv and caches the answers.
//
// # Backend conformance
//
// Every backend must give a Result the same shape semantics (exit codes,
// InfraKilled/KillReason, Captured, truncation flags); conformance_test.go
// runs one shared case table through Mock and, under the integration tag,
// through Bwrap and CLI to pin that contract.
//
// # Security posture
//
// Because the commands executed here originate from an LLM and run against
// untrusted code, the package is built around defense in depth: the original
// repository is never mounted read-write, the fresh workspace copy is the
// only writable surface, the container drops all capabilities and gains no
// new privileges, the root filesystem is read-only where practical, and the
// network defaults to "none". Read-only bind mounts (Spec.ROMounts) are the
// one window into host content: because untrusted code can read whatever is
// mounted, callers MUST only mount public/cache content (e.g. a module cache
// of public package source) and NEVER secrets, credentials, or private trees.
//
// Workspaces are materialized from the repository snapshot into a fresh
// temporary directory (never the live checkout), and everything written into
// a workspace is symlink-hardened: paths are resolved and refused if they
// escape the workspace root, so content planted by an earlier untrusted run
// cannot redirect a later write or read onto the host.
//
// Every run is supervised by a shared idle watchdog with two independent
// kill conditions: an idle window (no observable progress) and a
// workspace-growth ceiling (a disk-filler bound). Either kill surfaces in
// Result.TimedOut / Result.WorkspaceQuotaExceeded; callers MUST consult
// Result.InfraKilled before classifying a Result, so a run killed by the
// sandbox is never read as "the command completed and said X". Captured
// stdout/stderr are capped at DefaultMaxOutputBytes per stream (head and
// tail retained), so a chatty run cannot exhaust memory.
//
// # Error contract
//
// Exec returns a Go error only for infrastructure failures — a missing
// runtime, a failed workspace copy, an inability to launch. A non-zero exit
// code of the sandboxed command itself is reported in Result.ExitCode, not
// as an error.
//
// Path containment for agent tools (resolving and confining tool-supplied
// paths under a trusted root) is a separate concern and lives in the sibling
// fsroot package: this package confines the executed command's filesystem
// view; fsroot confines the caller's file tools.
package sandbox

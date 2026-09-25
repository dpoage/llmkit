// Package sandbox runs untrusted, model-generated commands against
// repository snapshots, inside an isolated backend. Every backend
// implements one interface:
//
//	Sandbox: Exec(ctx, Spec) (Result, error); MaterializeWorkspace(repoDir) (string, error)
//
// # Getting started
//
// A minimal run makes the calls below, in order:
//
//  1. Detect or DetectBwrap reports whether a real backend exists on
//     this host.
//  2. NewCLI, NewBwrap, NewHostExec, or NewMock constructs a backend.
//     NewMock always works; the real backends need their host
//     prerequisites.
//  3. MaterializeWorkspace (optional) creates one caller-owned
//     workspace for repeated Execs. Without it, Exec copies a fresh
//     workspace per call.
//  4. Exec runs the Spec to completion and returns a Result.
//  5. The caller checks Result.InfraKilled BEFORE reading ExitCode or
//     output. A run the sandbox killed must never read as a command
//     verdict.
//
// The Example functions in this package show each step. See
// docs/sandbox.md for the threat model and the backend comparison
// tables.
//
// # Backends
//
//   - NewBwrap (*Bwrap): an unprivileged Bubblewrap sandbox on Linux.
//     Bwrap unshares every namespace, mounts a sized tmpfs root, and
//     binds a narrow read-only allowlist back into view. The command
//     runs against host toolchains directly; there is no image.
//     DetectBwrap reports whether the host can run bwrap, with an
//     actionable reason when it cannot. NewBwrap fails fast with the
//     same reasons. Bwrap.Exec returns ErrBwrapNoCapMethod when no
//     resource-limit mechanism exists: enforcing memory/CPU/pids
//     limits needs a systemd-run user scope or a delegated cgroup v2
//     subtree. WithCapPolicy(CapBestEffort) deliberately opts out of
//     that requirement. DescribeBwrapCapMethod reports which mechanism
//     (if any) enforces the limits.
//   - NewCLI (*CLI): a container-runtime backend, podman first and
//     docker second. It drives the runtime CLI only, so it needs no
//     daemon API and only the standard library. Detect reports the
//     first available runtime. WithImage is required.
//   - NewHostExec (*HostExec): the attended escape hatch. It provides
//     NO isolation: no container, no network policy, no resource caps.
//     The command runs as the calling OS user. It exists only for
//     attended, operator-opted-in paths where a human watches the exact
//     command they chose to run. Nothing in this package, its defaults,
//     or its examples constructs one.
//   - NewMock (*Mock): a scriptable Sandbox for tests. The caller
//     enqueues scripted responses and reads back the recorded calls, so
//     caller code runs its tests without a real runtime or bwrap. The Mock
//     runs nothing but is not a law-free zone: it refuses a malformed Spec
//     exactly like every backend, and its MaterializeWorkspace creates a
//     real empty directory.
//
// # Specs and options
//
// Spec fields are honest per backend. A Spec that is malformed for every
// backend — empty Cmd, neither RepoDir nor Workspace, a relative
// Workspace, an escaping WriteFiles key or CaptureFiles entry, an
// empty/relative mount path, a duplicate ContainerPath, or an Env entry
// without "=" or with an empty key — is refused at Exec with an
// InvalidSpecError on EVERY backend, the Mock included. A well-formed Spec
// field a backend cannot honor is refused at Exec with an
// UnsupportedSpecError naming the
// backend, field, and value. No backend silently drops or substitutes a
// requested posture. Spec.Network is a
// typed NetworkMode: none, host, or bridge. Only the container backend
// honors bridge. Spec.Image is honored only by the container backend;
// the imageless backends refuse a non-empty Image.
//
// Fields that only some backends honor as knobs (runtime, image,
// resource caps, scratch size, growth ceiling, idle window) are backend
// options. One shared Option type serves both NewCLI and NewBwrap. A
// constructor refuses a misdirected option with a plain error naming
// the option and the backend. A default network mode the backend can
// never honor comes back as an UnsupportedSpecError.
//
// # Capability probes
//
// Before planning work against an unfamiliar sandbox, a caller can
// measure what the sandbox can actually run. Probe(ctx, sb, base, probes)
// runs the caller's []ProbeEntry table against sb and returns a
// CapabilitySet. Each entry names a capability, supplies the argv that
// measures it, and interprets the probe's exit code and stdout into named
// modes. The kit ships no probe entries: what to probe, and how to read
// the answer, is the caller's knowledge. Probe keeps no cache — every
// call re-runs every entry — and returns a refused base Spec
// (InvalidSpecError, UnsupportedSpecError) as an error instead of a
// guessed-at set.
//
// # Backend conformance
//
// Every backend gives a Result the same shape semantics: exit codes,
// InfraKilled and KillReason, Captured, and the truncation flags.
// conformance_test.go runs one shared case table through HostExec on
// every test run, and through Bwrap and CLI under the integration tag,
// to pin that contract.
//
// # Security posture
//
// The commands executed here come from an LLM and run against
// untrusted code, so the package builds defense in depth:
//
//   - The original repository is never mounted read-write.
//   - The fresh workspace copy is the only writable surface by
//     default; Spec.RWMounts are explicit exceptions.
//   - The container drops all capabilities and gains no new
//     privileges.
//   - The CLI backend runs on a read-only root filesystem; the Bwrap
//     backend runs on a fresh, sized tmpfs root.
//   - The network defaults to "none".
//
// Read-only bind mounts (Spec.ROMounts) are the one window into host
// content. Untrusted code can read whatever a caller mounts, so callers
// MUST mount only public or cache content (for example, a module cache
// of public package source) and NEVER secrets, credentials, or private
// trees.
//
// Exec materializes every workspace from the repository snapshot into
// a fresh temporary directory, never the live checkout, and refuses a
// Spec before any write when it is malformed (InvalidSpecError) or not
// honorable by the backend (UnsupportedSpecError). A caller can
// instead pass its own directory as Spec.Workspace. Everything
// written into a workspace is symlink-hardened: Exec resolves paths
// and refuses them if they escape the workspace root. Content planted
// by an earlier untrusted run therefore cannot redirect a later write
// or read onto the host.
//
// A shared idle watchdog supervises every CLI and Bwrap run — HostExec
// has no watchdog — with two independent kill conditions: an idle
// window (no observable progress) and a workspace-growth ceiling (a
// disk-filler bound). Either kill surfaces in Result.TimedOut or
// Result.WorkspaceQuotaExceeded. Callers MUST consult
// Result.InfraKilled before classifying a Result, so a run the sandbox
// killed is never read as "the command completed and said X". Captured
// stdout and stderr are capped at DefaultMaxOutputBytes per stream,
// with head and tail retained, so a chatty run cannot exhaust memory.
//
// # Error contract
//
// Exec returns a Go error in exactly three cases: a refused Spec — an
// InvalidSpecError (malformed for every backend: fix the Spec) or an
// UnsupportedSpecError (this backend cannot honor a field: pick another
// backend or drop the field) — before anything is written or launched; a
// caller context that ended (cancel or deadline), as the "sandbox:
// execution cancelled" error wrapping ctx.Err(); and an infrastructure
// failure (a missing runtime, a failed workspace copy, a runtime or
// wrapper that could not be started). Everything the sandboxed command
// itself does is reported in the Result instead: its exit code, 127/126
// when it cannot be launched, and 128+signo when a signal killed it.
//
// Path containment for agent tools — resolving and confining
// tool-supplied paths under a trusted root — is a separate concern. It
// lives in the sibling fsroot package: this package confines the
// executed command's filesystem view; fsroot confines the caller's file
// tools.
package sandbox

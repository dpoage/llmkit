package sandbox

import (
	"context"
	"fmt"
	"time"
)

// DefaultMaxOutputBytes is the per-stream cap applied to captured stdout and
// stderr. Output beyond this size is discarded and Result records that it was
// truncated.
const DefaultMaxOutputBytes = 1 << 20 // 1 MiB

// NetworkMode selects a run's network posture. The zero value ("") means
// "the backend default" and is resolved explicitly at Exec; every non-empty
// mode must be in the backend's supported set, which each Exec enforces.
type NetworkMode string

const (
	// NetworkNone disables all network egress. It is the CLI and Bwrap
	// backends' default.
	NetworkNone NetworkMode = "none"
	// NetworkHost shares the host's own network namespace: full access,
	// no namespace isolation. The HostExec backend's default is the
	// equivalent posture (a bare host process has only host networking).
	NetworkHost NetworkMode = "host"
	// NetworkBridge attaches the run to the container runtime's default
	// bridge network. Only the CLI backend honors it — bwrap either
	// unshares the single host namespace or shares it wholesale, and a
	// bare host process has no namespace to bridge to — so the other
	// backends refuse it at Exec.
	NetworkBridge NetworkMode = "bridge"
)

// UnsupportedSpecError is returned by Exec when a backend cannot honor a
// non-empty per-call Spec field, and by NewCLI/NewBwrap when a WithNetwork
// default names a mode that backend could never honor. A backend never
// silently runs with a different posture than the Spec requested: refusing
// the run IS the contract. Match with errors.As; Backend is "cli", "bwrap",
// or "host".
type UnsupportedSpecError struct {
	Backend string // "cli", "bwrap", or "host"
	Field   string // Spec field name, e.g. "Network", "Image"
	Value   string // the unsupported value as written
}

func (e *UnsupportedSpecError) Error() string {
	return fmt.Sprintf("sandbox: %s backend cannot honor Spec.%s = %q", e.Backend, e.Field, e.Value)
}

// resolveNetworkMode merges the backend default with the Spec's per-call
// mode and validates the result against the backend's supported set. An
// empty requested mode resolves to def; when both are empty the package
// default NetworkNone applies (the constructors set it explicitly, so this
// fallback only covers a zero-value backend struct). Anything outside
// supported is refused with an UnsupportedSpecError — never silently
// substituted.
func resolveNetworkMode(backend string, def, requested NetworkMode, supported ...NetworkMode) (NetworkMode, error) {
	mode := requested
	if mode == "" {
		mode = def
	}
	if mode == "" {
		mode = NetworkNone
	}
	for _, s := range supported {
		if mode == s {
			return mode, nil
		}
	}
	return "", &UnsupportedSpecError{Backend: backend, Field: "Network", Value: string(mode)}
}

// Spec describes a single sandboxed execution.
//
// The Mock backend records every field verbatim and runs nothing; the
// honor notes below cover the three real backends (CLI, Bwrap,
// HostExec).
type Spec struct {
	// RepoDir is the host path to the repository snapshot to run against. It is
	// copied into a fresh temporary workspace before execution; the original is
	// never mounted writable (and is not mutated). Required.
	RepoDir string

	// Workspace, when non-empty, overrides the fresh-copy-per-Exec default:
	// Exec uses this host directory as the workspace directly instead of
	// copying RepoDir into a new temp dir. Exec does not remove it afterward;
	// the caller owns its entire lifecycle (creation via MaterializeWorkspace
	// and removal). WriteFiles still apply onto it, so repeated Execs
	// against the same Workspace accumulate and overwrite files exactly like
	// repeated writes to a real working tree. The container backends require
	// an absolute path; HostExec uses the path verbatim as the working
	// directory.
	//
	// TRUST: only pass a directory the harness itself created (e.g. via
	// MaterializeWorkspace). Exec does no provenance check, so an arbitrary
	// caller-supplied path is trusted verbatim as writable model-code
	// execution surface. RepoDir is ignored when Workspace is set.
	Workspace string

	// Cmd is the command (argv) executed inside the container's workspace.
	// Required and non-empty.
	Cmd []string

	// Env is a list of KEY=VALUE environment variables to set inside the
	// container, in the same form as os.Environ.
	Env []string

	// Image overrides the backend's default image for this execution. When
	// empty, the backend's configured default image is used. The Bwrap and
	// HostExec backends have no image concept, so they refuse a non-empty
	// Image at Exec instead of silently ignoring it.
	Image string

	// Timeout bounds the execution wall-clock time as a HARD ceiling. When
	// <= 0 the backend's default timeout is used (CLI and Bwrap default to
	// 10m; HostExec has no default, so only the caller's context can cancel).
	// On expiry the backend kills the command and sets Result.TimedOut.
	Timeout time.Duration

	// Network selects the run's network posture (see NetworkMode). The zero
	// value resolves to the backend default — NetworkNone on the CLI and
	// Bwrap backends; the host network on HostExec. A mode the backend
	// cannot honor is refused at Exec with an UnsupportedSpecError, never
	// silently substituted.
	Network NetworkMode

	// WriteFiles are files to write into the workspace before execution, keyed
	// by path relative to the workspace root. This is how reproduction tests
	// are injected into the snapshot. Parent directories are created as needed.
	// Paths that escape the workspace (absolute, or containing "..") are
	// rejected as an error.
	WriteFiles map[string][]byte

	// ROMounts are additional host directories bind-mounted read-only into
	// the sandbox, in addition to the writable workspace. They exist so a
	// dependency cache (e.g. a Go module cache) can be available to an
	// otherwise network-none run without copying it into the workspace. The
	// CLI and Bwrap backends honor them; HostExec refuses a non-empty
	// ROMounts with an UnsupportedSpecError.
	//
	// A mount is NEVER writable. Both paths must be absolute; empty paths
	// and ContainerPaths duplicate across ROMounts and RWMounts are rejected
	// as an error by Exec. On the CLI backend each mount renders as
	// `-v host:ctr:ro,Z` unless ROMount.Shared is true.
	//
	// SECURITY: a read-only mount exposes host content to untrusted,
	// model-driven code. Callers must only mount public/cache content and
	// never secrets or private trees. See the package doc.
	ROMounts []ROMount

	// RWMounts are host directories bind-mounted WRITABLE into the sandbox.
	// They exist for two callers:
	//
	//   - A trusted dependency-prefetch step (e.g. `go mod download`) that
	//     populates a caller-managed module cache on the host, later exposed
	//     to the untrusted network-none run read-only via ROMounts.
	//   - An operator's explicitly opted-in writable entry, for tools that
	//     unconditionally mutate a mounted directory at analysis or build
	//     time. The motivating case is `bazel vendor`, which refreshes its
	//     bazel-external symlinks, .marker files, and disk cache inside the
	//     vendor dir; a read-only mount there aborts the run.
	//
	// The CLI and Bwrap backends honor RWMounts; HostExec refuses a
	// non-empty RWMounts with an UnsupportedSpecError.
	//
	// SECURITY: a writable mount is strictly more dangerous than ROMounts —
	// untrusted, model-driven code can corrupt whatever is mounted. Do NOT
	// use RWMounts for a normal model-driven run's own scratch space: the
	// writable workspace copy is that surface. Scope an operator-opted-in
	// entry to a caller-owned directory (e.g. a dedicated vendor or
	// disk-cache dir the operator manages for the tool). The accepted
	// tradeoff: a compromised run can corrupt that directory, and the blast
	// radius stops there — the worst case is the tool's own next sandbox run
	// reading corrupted vendored state, not a wider compromise. On the CLI
	// backend Shared=true suppresses the SELinux :Z relabel, exactly like
	// ROMounts: host-owned trees the host also manages must keep their
	// context. Absolute-path and uniqueness validation match ROMounts, and
	// ContainerPaths must be unique across ROMounts and RWMounts combined.
	RWMounts []ROMount

	// SetupCmds are optional ordered commands executed inside the sandbox,
	// in the same network-none run, in the workspace directory, BEFORE Cmd.
	// They exist so non-Go ecosystems (npm, pip, cargo, etc.) can perform
	// offline package installation from a pre-mounted cache without
	// altering the main command. The CLI and Bwrap backends honor them;
	// HostExec refuses a non-empty SetupCmds with an UnsupportedSpecError
	// (a bare host process has no exit-125 environment-error contract to
	// keep).
	//
	// Examples: ["npm","ci","--offline"] or ["pip","install","--no-index","--find-links=/pipcache","."]
	//
	// When SetupCmds is non-empty, both container backends wrap the
	// execution in /bin/sh: each command is shell-quoted and chained with
	// "|| exit 125", so any setup failure exits with code 125. Exit 125 is
	// intentional: verdict classification treats container exit 125/126/127
	// as an environment error, NOT a demonstrated result — a failed
	// "npm ci --offline" must never be misread as a successful repro. The
	// original Cmd is exec'd (via sh's exec builtin), so it retains its own
	// exit code and signal mask.
	//
	// Requires /bin/sh in the container image (or in the bwrap allowlist).
	// Go-only images set no SetupCmds, so existing images and behavior are
	// untouched.
	SetupCmds [][]string

	// CaptureFiles are workspace-relative paths to read back from the
	// workspace after the command finishes, in addition to stdout/stderr.
	// All three real backends honor them. This is the seam structured-output
	// ecosystems use: e.g. a pytest run asked to emit
	// `--junitxml=report.xml` writes machine-readable results to a file
	// rather than (only) stdout, and the caller needs that file's bytes,
	// not just the exit code. Each path is validated with the same rule as
	// WriteFiles keys (no escaping the workspace); a path the command never
	// wrote is silently absent from Result.Captured rather than failing the
	// run.
	CaptureFiles []string
}

// ROMount is a single read-only bind mount of a host directory into the
// container. It is never writable.
type ROMount struct {
	// HostPath is the absolute host path to expose. Required.
	HostPath string
	// ContainerPath is the absolute path the mount appears at inside the
	// sandbox. Required, and unique across the Spec's ROMounts and RWMounts
	// combined.
	ContainerPath string
	// Shared, when true, suppresses the SELinux :Z relabel suffix on this
	// mount. Use Shared=true for host directories that are NOT owned exclusively
	// by the sandboxing tool — in particular, the user's shared Go module cache
	// (~/go/pkg/mod). On SELinux-enforcing hosts (Fedora, RHEL — rootless
	// podman's home turf) :Z recursively relabels the target to a
	// container-PRIVATE MCS label. That is correct for tool-owned dirs (it
	// isolates them), but catastrophic for a shared cache: it is slow on
	// multi-GB trees, breaks the host go toolchain, and breaks any other
	// container concurrently sharing the same cache.
	//
	// When Shared=true the mount is rendered :ro with NO label suffix. This
	// means the container accesses the directory under its existing SELinux
	// context. Under a strict enforcing policy the container may get EACCES if
	// that policy does not allow the container domain to read the host user's
	// home content. That is the correct conservative failure — a permission
	// error is loud and actionable, whereas :Z silently corrupts shared state.
	// Users who hit EACCES can opt in to :z (lowercase, shared relabel) by
	// labeling the tree as shared at the host level, or by moving the
	// directory under a tool-managed cache root that is safe to relabel
	// (Shared=false, gets :Z).
	//
	// Tool-managed dirs (dependency caches, prefetch RW targets) leave Shared
	// false so they receive :Z isolation.
	Shared bool
}

// Result is the faithful outcome of a sandboxed execution.
//
// A non-zero ExitCode is NOT reported as a Go error: callers interpret exit
// codes themselves (a failing repro test is expected to exit non-zero). Only
// infrastructure failures — a missing runtime, a failed workspace copy, an
// inability to launch the container — are returned as errors from Exec.
type Result struct {
	// ExitCode is the process exit code of the sandboxed command. A
	// watchdog kill (timeout or growth ceiling) reports -1; every other
	// value is the command's own exit code. Set by Exec on every backend;
	// a Mock script supplies whatever value its caller scripted.
	ExitCode int

	// Stdout and Stderr are the captured output streams, each capped at the
	// backend's max output size. Truncation is recorded in the Truncated flags
	// and a trailing marker is appended to the captured text.
	Stdout string
	Stderr string

	// StdoutTruncated / StderrTruncated report whether the corresponding stream
	// exceeded the cap and was truncated.
	StdoutTruncated bool
	StderrTruncated bool

	// Duration is the measured wall-clock time of the execution: process
	// launch through exit. It covers container create + the command itself;
	// a single `podman run` invocation does not expose a way to separate
	// container-create time from command time, so this stays one number. See
	// PrepDuration for the (separable) pre-container workspace-preparation
	// cost.
	Duration time.Duration

	// TimedOut is true when the execution was killed because it exceeded the
	// effective timeout OR because the idle watchdog observed no progress for
	// the backend's WithIdleTimeout window. It is left false when
	// WorkspaceQuotaExceeded is true (see below) — the two are mutually
	// exclusive, distinct kill reasons.
	TimedOut bool

	// WorkspaceQuotaExceeded is true when Exec killed the run because the
	// workspace's NET regular-file size grew by more than the backend's
	// configured growth-ceiling bytes since the run started (a
	// write-then-delete churn nets out and never trips this). The CLI and
	// Bwrap watchdogs set it; HostExec never does (no watchdog), and a Mock
	// script supplies it verbatim.
	//
	// This is deliberately NOT reported as TimedOut. A run that is actively
	// filling disk reads as "making progress" to the idle check and would
	// otherwise run undetected until the absolute Timeout. Callers that
	// only check TimedOut must not mistake a disk-filler for a genuine
	// stall or a legitimate long-running build.
	//
	// ExitCode is -1, exactly like a TimedOut kill. If the process exited
	// on its own after breaching the ceiling, the breach still overrides
	// its own exit code: the ceiling is a measured invariant, not a racing
	// heuristic.
	WorkspaceQuotaExceeded bool

	// PrepDuration is the wall-clock time spent preparing the workspace
	// BEFORE the command ran: ensuring the pristine workspace copy
	// (materializing on a cache miss, reusing it on a hit), cloning it into
	// a fresh per-run workspace, and applying WriteFiles. It is disjoint
	// from Duration. The three real backends set it; a Mock script supplies
	// whatever its caller scripted.
	PrepDuration time.Duration

	// WorkspaceCacheHit reports whether the backend's pristine-workspace
	// cache already held a pristine copy matching this repo's current HEAD
	// + working-tree state, so this Exec skipped materialization and only
	// cloned it. CLI and Bwrap set it; always false on HostExec (no cache)
	// and whenever RepoDir is not a git work tree, since the cache is
	// bypassed entirely there. A Mock script supplies it verbatim.
	WorkspaceCacheHit bool

	// Captured holds the workspace-relative files named in Spec.CaptureFiles,
	// keyed by their (cleaned) relative path, each capped at the backend's max
	// output size like Stdout/Stderr. A key is absent when the command never
	// wrote that file — CaptureFiles is a best-effort "grab it if the tool
	// produced it" contract, not a manifest every run must satisfy. Nil when
	// Spec.CaptureFiles was empty or nothing was captured.
	Captured map[string][]byte
}

// InfraKilled reports whether Exec killed this run for an infrastructure
// reason — the absolute/idle timeout OR the workspace-growth ceiling —
// rather than the command exiting (successfully or not) on
// its own. Callers that classify a Result into a verdict MUST check this
// BEFORE interpreting ExitCode/output: a run killed by us must never be
// read as "the command completed and its output says X" (e.g.
// misclassified as not-demonstrated or a rejected fix) regardless of which
// specific kill reason fired. This is the shared seam every caller's
// verdict-classification code uses so a new kill reason only needs to be
// taught here once, not re-derived at every call site.
func (r Result) InfraKilled() bool {
	return r.TimedOut || r.WorkspaceQuotaExceeded
}

// KillReason returns a short, human-readable label naming why Exec killed
// this run when InfraKilled is true, for verdict/summary messages that want
// to name the specific cause rather than a generic "timed out" — in
// particular so a WorkspaceQuotaExceeded kill (a disk-filler) reads
// distinctly from a genuine idle-stall/absolute-timeout kill instead of
// both collapsing into the same message. Returns "" when InfraKilled is
// false.
func (r Result) KillReason() string {
	switch {
	case r.WorkspaceQuotaExceeded:
		return "workspace growth exceeded the configured ceiling (see WithWorkspaceGrowthCeilingMB)"
	case r.TimedOut:
		return "timed out"
	default:
		return ""
	}
}

// Sandbox is an isolated command executor. Implementations must be safe for
// concurrent use by multiple goroutines.
type Sandbox interface {
	// Exec runs spec to completion (or until the timeout / ctx cancellation)
	// and returns the captured Result. It returns a non-nil error only for
	// infrastructure failures; a non-zero exit code is reported via
	// Result.ExitCode, not as an error.
	Exec(ctx context.Context, spec Spec) (Result, error)

	// MaterializeWorkspace clones repoDir into a fresh, caller-owned
	// workspace directory and returns its path, writing nothing into the
	// clone itself. It is the public seam behind Spec.Workspace: a caller
	// that wants to write into and run repeated Execs against ONE persistent
	// workspace materializes it once here, then passes the returned path as
	// Spec.Workspace on each Exec instead of letting Exec copy a fresh one
	// every time. The caller owns the returned directory's entire lifecycle:
	// Exec(Workspace: ...) never removes it, so the caller MUST os.RemoveAll
	// it when done (typically via defer at the scope that bounds all the
	// iteration's Execs).
	MaterializeWorkspace(repoDir string) (string, error)
}

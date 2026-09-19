package sandbox

import (
	"fmt"
	"slices"
	"time"
)

// Option configures a sandbox backend. One Option type serves both
// option-taking constructors (NewCLI, NewBwrap); each constructor refuses
// with an error naming any option its backend cannot honor, so a
// misdirected option is a construction-time error instead of a silently
// ignored knob. No option takes a bare bool: modes are named types
// (NetworkMode, CapPolicy).
type Option func(*options)

// options is the accumulated state a sequence of Options produces. It is
// unexported: callers configure exclusively through the With* functions.
type options struct {
	// Shared knobs (CLI + Bwrap).
	cpus            float64
	memoryMB        int
	timeout         time.Duration
	idleTimeout     time.Duration
	network         NetworkMode
	pidsLimit       int
	maxOutputBytes  int
	scratchSizeMB   int
	growthCeilingMB int

	// CLI-only.
	runtime string
	image   string

	// Bwrap-only.
	capPolicy            CapPolicy
	toolchainBinds       []ROMount
	toolchainPathPrepend string

	// requested names each applied option in application order, so a
	// constructor can refuse options its backend does not understand
	// (naming them) and apply only what was actually passed.
	requested []string
}

// CapPolicy names what a Bwrap run does when the host offers no
// resource-limit enforcement mechanism (neither systemd-run --user --scope
// nor a delegated cgroup v2 subtree).
type CapPolicy int

const (
	// CapRequired (the zero value, and the default) fails the run with
	// ErrBwrapNoCapMethod when no enforcement mechanism exists: requested
	// resource caps are a guarantee, not a suggestion.
	CapRequired CapPolicy = iota
	// CapBestEffort runs the command WITHOUT enforced memory/CPU/pids
	// limits when no mechanism is available, instead of failing. An
	// explicit, named opt-in to the weaker posture.
	CapBestEffort
)

// cliOnlyOptions names the options only the container-runtime backend
// understands; NewBwrap refuses each of them.
var cliOnlyOptions = map[string]bool{"WithRuntime": true, "WithImage": true}

// bwrapOnlyOptions names the options only the bwrap backend understands;
// NewCLI refuses each of them.
var bwrapOnlyOptions = map[string]bool{
	"WithCapPolicy":      true,
	"WithToolchainBinds": true,
	"WithToolchainPath":  true,
}

// cliNetworks / bwrapNetworks are the network modes each backend can honor,
// used both to validate a WithNetwork default at construction and to resolve
// a per-call Spec.Network at Exec.
var (
	cliNetworks   = []NetworkMode{NetworkNone, NetworkHost, NetworkBridge}
	bwrapNetworks = []NetworkMode{NetworkNone, NetworkHost}
)

func newOptions(opts []Option) options {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

func (o *options) track(name string) { o.requested = append(o.requested, name) }

// has reports whether the named option was applied.
func (o *options) has(name string) bool { return slices.Contains(o.requested, name) }

// checkSupported returns an error naming the first applied option the given
// backend cannot honor.
func (o *options) checkSupported(backend string, unsupported map[string]bool) error {
	for _, name := range o.requested {
		if unsupported[name] {
			return fmt.Errorf("sandbox: option %s is not supported by the %s backend", name, backend)
		}
	}
	return nil
}

// validateNetworkDefault rejects a WithNetwork value the backend could never
// honor — at construction, rather than on every later Exec. The refusal is
// the same typed UnsupportedSpecError Exec uses, so a misdirected network
// posture is matched the same way at both boundaries.
func validateNetworkDefault(backend string, m NetworkMode, supported []NetworkMode) error {
	if m == "" {
		return nil
	}
	for _, s := range supported {
		if m == s {
			return nil
		}
	}
	return &UnsupportedSpecError{Backend: backend, Field: "Network", Value: string(m)}
}

// defaults is the shared per-run default state both option-taking backends
// apply: every Spec field that resolves to a backend default when the Spec
// leaves it unset, plus the knobs with no per-call counterpart. CLI and
// Bwrap embed it, so their field names — and the option mapping below —
// exist exactly once.
type defaults struct {
	// defaultTimeout is the absolute wall-clock ceiling applied when a
	// Spec leaves Timeout unset.
	defaultTimeout time.Duration
	// defaultIdleTimeout is the inactivity window applied to every run.
	// Zero disables the idle watchdog (absolute timeout only).
	defaultIdleTimeout time.Duration
	defaultNetwork     NetworkMode
	defaultCPUs        float64
	defaultMemory      int
	pidsLimit          int
	maxOutputBytes     int
	// defaultScratchSizeMB is the size (MB) of the writable tmpfs scratch
	// space (/tmp, plus the tmpfs root under bwrap). <= 0 is treated as
	// unset and falls back to fallbackScratchSizeMB in the argv builders.
	defaultScratchSizeMB int
	// defaultGrowthCeilingBytes bounds NET workspace growth (the fsSize
	// delta, not cumulative bytes written — see workspaceProgress) since a
	// run starts, tolerated by the shared idle watchdog before killing the
	// run with the distinct Result.WorkspaceQuotaExceeded reason.
	// <= 0 disables the ceiling.
	defaultGrowthCeilingBytes int64
}

// baseDefaults returns the out-of-the-box posture every backend starts
// from, before any With* override is applied.
func baseDefaults() defaults {
	return defaults{
		defaultCPUs:               2,
		defaultMemory:             2048,
		defaultTimeout:            10 * time.Minute,
		defaultNetwork:            NetworkNone,
		pidsLimit:                 256,
		maxOutputBytes:            DefaultMaxOutputBytes,
		defaultScratchSizeMB:      fallbackScratchSizeMB,
		defaultGrowthCeilingBytes: defaultWorkspaceGrowthCeilingBytes,
	}
}

// applyDefaults copies each applied option's value over the base defaults.
// This is the SINGLE site where an option reaches a backend's defaults: a
// knob added to options without a line here is a compile-visible no-op the
// option-application tests catch, and no constructor carries its own
// hand-copied application block that can silently drift.
func (o *options) applyDefaults(d *defaults) {
	if o.has("WithCPUs") {
		d.defaultCPUs = o.cpus
	}
	if o.has("WithMemoryMB") {
		d.defaultMemory = o.memoryMB
	}
	if o.has("WithTimeout") {
		d.defaultTimeout = o.timeout
	}
	if o.has("WithIdleTimeout") {
		d.defaultIdleTimeout = o.idleTimeout
	}
	if o.has("WithNetwork") {
		d.defaultNetwork = o.network
	}
	if o.has("WithPidsLimit") {
		d.pidsLimit = o.pidsLimit
	}
	if o.has("WithMaxOutputBytes") {
		d.maxOutputBytes = o.maxOutputBytes
	}
	if o.has("WithScratchSizeMB") {
		d.defaultScratchSizeMB = o.scratchSizeMB
	}
	if o.has("WithWorkspaceGrowthCeilingMB") {
		d.defaultGrowthCeilingBytes = int64(o.growthCeilingMB) * 1024 * 1024
	}
}

// WithRuntime sets the container runtime binary (podman or docker); empty
// auto-detects. CLI-only: NewBwrap refuses it.
func WithRuntime(name string) Option {
	return func(o *options) { o.runtime = name; o.track("WithRuntime") }
}

// WithImage sets the default container image used when a Spec leaves Image
// unset; required (NewCLI errors without it). CLI-only: NewBwrap refuses it.
func WithImage(image string) Option {
	return func(o *options) { o.image = image; o.track("WithImage") }
}

// WithCPUs sets the default CPU limit applied to every run.
func WithCPUs(c float64) Option {
	return func(o *options) { o.cpus = c; o.track("WithCPUs") }
}

// WithMemoryMB sets the default memory limit (MB) applied to every run.
func WithMemoryMB(m int) Option {
	return func(o *options) { o.memoryMB = m; o.track("WithMemoryMB") }
}

// WithTimeout sets the default execution timeout applied when a Spec leaves
// Timeout unset.
func WithTimeout(d time.Duration) Option {
	return func(o *options) { o.timeout = d; o.track("WithTimeout") }
}

// WithIdleTimeout sets the default idle (no-progress) window applied to every
// run. A run is cancelled only after this long with no observable progress;
// the absolute WithTimeout remains a hard ceiling. Zero disables the
// watchdog (the HostExec backend has no watchdog and takes no options).
func WithIdleTimeout(d time.Duration) Option {
	return func(o *options) { o.idleTimeout = d; o.track("WithIdleTimeout") }
}

// WithNetwork sets the default network mode applied when a Spec leaves
// Network unset. The package default is NetworkNone; a mode the backend
// cannot honor is rejected by the constructor.
func WithNetwork(n NetworkMode) Option {
	return func(o *options) { o.network = n; o.track("WithNetwork") }
}

// WithPidsLimit sets the process-count cap. A value <= 0 disables the cap.
func WithPidsLimit(n int) Option {
	return func(o *options) { o.pidsLimit = n; o.track("WithPidsLimit") }
}

// WithMaxOutputBytes overrides the per-stream output cap.
func WithMaxOutputBytes(n int) Option {
	return func(o *options) { o.maxOutputBytes = n; o.track("WithMaxOutputBytes") }
}

// WithScratchSizeMB sets the size (MB) of the writable tmpfs scratch space
// (/tmp, plus the tmpfs root under bwrap). Values <= 0 fall back to
// fallbackScratchSizeMB.
func WithScratchSizeMB(mb int) Option {
	return func(o *options) { o.scratchSizeMB = mb; o.track("WithScratchSizeMB") }
}

// WithWorkspaceGrowthCeilingMB sets the workspace-growth ceiling (MB of NET
// workspace-size growth, not cumulative bytes written) the shared idle
// watchdog enforces independent of idle-stall detection: a run whose
// workspace grows past this is killed with Result.WorkspaceQuotaExceeded.
// <= 0 disables the ceiling entirely.
func WithWorkspaceGrowthCeilingMB(mb int) Option {
	return func(o *options) { o.growthCeilingMB = mb; o.track("WithWorkspaceGrowthCeilingMB") }
}

// WithCapPolicy sets what a Bwrap run does when the host offers no
// resource-limit enforcement mechanism: CapRequired (the default) fails with
// ErrBwrapNoCapMethod; CapBestEffort runs uncapped. Bwrap-only: NewCLI
// refuses it.
func WithCapPolicy(p CapPolicy) Option {
	return func(o *options) { o.capPolicy = p; o.track("WithCapPolicy") }
}

// WithToolchainBinds adds extra read-only binds (beyond fixedROAllowlist) to
// every Bwrap run, resolved by the host-toolchain resolver. Bwrap-only:
// NewCLI refuses it.
func WithToolchainBinds(mounts []ROMount) Option {
	return func(o *options) { o.toolchainBinds = mounts; o.track("WithToolchainBinds") }
}

// WithToolchainPath sets the PATH prefix (ResolveHostToolchains' PathPrepend)
// paired with WithToolchainBinds, so resolved toolchain binaries are actually
// reachable via PATH inside the sandbox rather than merely bind-mounted.
// Callers pass both options from the same ToolchainResolution. Bwrap-only:
// NewCLI refuses it.
func WithToolchainPath(prepend string) Option {
	return func(o *options) { o.toolchainPathPrepend = prepend; o.track("WithToolchainPath") }
}

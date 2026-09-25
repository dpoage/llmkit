package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// candidateRuntimes is the auto-detect search order for the container runtime
// CLI. Podman is preferred (rootless, daemonless), with docker as a fallback.
var candidateRuntimes = []string{"podman", "docker"}

// Detect reports the first available container runtime CLI on PATH and whether
// one was found. Callers (and tests) use it to skip gracefully when no runtime
// is installed.
func Detect() (runtime string, ok bool) {
	for _, rt := range candidateRuntimes {
		if _, err := exec.LookPath(rt); err == nil {
			return rt, true
		}
	}
	return "", false
}

// CLI is a Sandbox backed by a container runtime CLI (podman or docker),
// safe for concurrent use.
type CLI struct {
	runtime      string
	defaultImage string
	// defaults holds the shared per-run default state, configured via Option.
	defaults
	// toolchainBinds and toolchainPathPrepend are WithHostToolchains'
	// resolved read-only mounts and PATH prefix, rendered by
	// resolveParams/buildRunArgs alongside Spec.ROMounts. Nil/empty when
	// no host toolchains are configured.
	toolchainBinds       []ROMount
	toolchainPathPrepend string
	// wsCache is the pristine-materialization cache; zero value is ready to use.
	wsCache wsCache
}

// NewCLI constructs a CLI sandbox backed by a container runtime (podman, then
// docker; WithRuntime overrides the auto-detect order). WithImage is
// required. The bwrap-only options (WithCapPolicy) are refused here with an
// error naming the option, a numeric option out of range is refused before
// any runtime lookup, and a WithNetwork mode the runtime could never honor
// fails at construction.
func NewCLI(opts ...Option) (*CLI, error) {
	o := newOptions(opts)
	if err := o.checkSupported(backendCLI, bwrapOnlyOptions); err != nil {
		return nil, err
	}
	if err := o.checkNumericOptions(backendCLI); err != nil {
		return nil, err
	}
	if err := validateNetworkDefault(backendCLI, o.network, cliNetworks); err != nil {
		return nil, err
	}

	runtime := o.runtime
	if runtime == "" {
		detected, ok := Detect()
		if !ok {
			return nil, errors.New("sandbox: no container runtime found on PATH (tried podman, docker)")
		}
		runtime = detected
	} else if _, err := exec.LookPath(runtime); err != nil {
		return nil, fmt.Errorf("sandbox: container runtime %q not found on PATH: %w", runtime, err)
	}

	if o.image == "" {
		return nil, errors.New("sandbox: a default image is required (WithImage)")
	}

	s := &CLI{
		runtime:      runtime,
		defaultImage: o.image,
	}
	s.defaults = baseDefaults()
	o.applyDefaults(&s.defaults)
	if o.has("WithHostToolchains") {
		s.toolchainBinds = o.toolchainBinds
		s.toolchainPathPrepend = o.toolchainPathPrepend
	}
	// Best-effort hygiene: purge any workspace-cache parent dirs a previous,
	// non-Closed CLI instance (or a crashed process) left behind. See
	// purgeStaleWorkspaceCaches.
	purgeStaleWorkspaceCaches()
	return s, nil
}

// Close removes this CLI instance's workspace-cache parent directory (see
// wsCache), if one was ever materialized. wsCache's mutex still guards
// concurrent access, so Close atomically claims the parent-dir path before
// removing it; an Exec racing Close afterward simply re-lazily-inits a fresh
// cache dir. That mutex protects the CACHE STATE ONLY (which pristine is
// current, where the parent dir lives) — it says nothing about an in-flight
// Exec's OWN per-run workspace clone, which Close never touches. Safety for a
// concurrent Exec instead comes from construction-site scoping: a caller that
// defers Close only does so after every Exec using that *CLI has returned, so
// Close and Close-observing Execs are never concurrent by construction, not
// because the mutex would arbitrate a race if they were. Safe to call on a
// nil receiver and multiple times.
//
// Callers that hold a *CLI across a natural scope (a single command's RunE, a
// function that both builds and exhausts the sandbox) should defer Close.
// Where no such scope exists (e.g. a sandbox handed off to a longer-lived
// consumer), the 24h purge in NewCLI is the backstop.
func (s *CLI) Close() error {
	if s == nil {
		return nil
	}
	return s.wsCache.close()
}

// MaterializeWorkspace clones the pristine-workspace cache for repoDir (see
// wsCache) into a fresh, caller-owned workspace directory and returns its
// path with no files written into it beyond the clone itself. It is the
// public seam behind Spec.Workspace: a caller that wants to write into and
// run repeated Execs against ONE persistent workspace (for example,
// iterating across several sandbox runs before settling on a final
// plan) materializes it once here, then passes the returned path as
// Spec.Workspace on each Exec instead of letting Exec copy a fresh one every
// time.
//
// The caller owns the returned directory's entire lifecycle: MaterializeWorkspace
// applies no WriteFiles and Exec(Workspace: ...) never removes it, so the
// caller MUST os.RemoveAll it when done (typically via defer at the scope that
// bounds all the iteration's Execs).
func (s *CLI) MaterializeWorkspace(repoDir string) (string, error) {
	ws, _, err := s.prepareWorkspace(repoDir, nil)
	return ws, err
}

// resolveParams applies backend defaults to a Spec, producing the concrete
// runParams for the run (workspace and containerName are filled in by Exec).
// The network mode is resolved and validated here: a mode the backend cannot
// honor refuses the run with an UnsupportedSpecError instead of passing
// through to the runtime flag. Host-toolchain binds/PATH prefix (see
// WithHostToolchains) are copied straight through — this backend renders no
// PATH override of its own when none are configured.
func (s *CLI) resolveParams(spec Spec) (runParams, error) {
	network, err := resolveNetworkMode(backendCLI, s.defaultNetwork, spec.Network, cliNetworks...)
	if err != nil {
		return runParams{}, err
	}
	p := runParams{
		image:                s.defaultImage,
		network:              network,
		cpus:                 s.defaultCPUs,
		memoryMB:             s.defaultMemory,
		pidsLimit:            s.pidsLimit,
		scratchSizeMB:        s.defaultScratchSizeMB,
		env:                  spec.Env,
		cmd:                  spec.Cmd,
		roMounts:             spec.ROMounts,
		rwMounts:             spec.RWMounts,
		setupCmds:            spec.SetupCmds,
		toolchainBinds:       s.toolchainBinds,
		toolchainPathPrepend: s.toolchainPathPrepend,
	}
	if spec.Image != "" {
		p.image = spec.Image
	}
	return p, nil
}

// Exec implements Sandbox. Its error contract is the Sandbox interface's
// (the package doc's "Error contract"): an error only for a refused Spec
// (InvalidSpecError, UnsupportedSpecError), a caller ctx that ended (the
// "sandbox: execution cancelled" error), or an infrastructure failure; a
// non-zero exit code is reported in Result.ExitCode.
func (s *CLI) Exec(ctx context.Context, spec Spec) (Result, error) {
	if err := validateSpec(backendCLI, spec); err != nil {
		return Result{}, err
	}
	timeout := spec.Timeout
	if timeout <= 0 {
		timeout = s.defaultTimeout
	}
	// resolveParams is the "Spec + defaults -> argv inputs" stage; the
	// per-backend Spec admission already ran in validateSpec.
	p, err := s.resolveParams(spec)
	if err != nil {
		return Result{}, err
	}
	// The container name is fixed BEFORE the run: cpuBusy reads it from
	// the watchdog goroutine and reap from the supervisor.
	name := "llmkit-" + randToken()
	p.containerName = name
	return runSupervised(ctx, runSpec{
		spec:           spec,
		timeout:        timeout,
		idleTimeout:    s.defaultIdleTimeout,
		growthCeiling:  s.defaultGrowthCeilingBytes,
		maxOutputBytes: s.maxOutputBytes,
		hooks: runHooks{
			prepareWorkspace: func(repoDir string) (string, bool, error) {
				return s.prepareWorkspace(repoDir, spec.WriteFiles)
			},
			buildCmd: func(ws string, runCtx context.Context) (*exec.Cmd, error) {
				p.workspace = ws
				return exec.CommandContext(runCtx, s.runtime, buildRunArgs(p)...), nil
			},
			cpuBusy: func() bool { return s.containerCPUBusy(name) },
			reap:    func(*exec.Cmd) { s.forceRemove(name) },
		},
	})
}

// forceRemove best-effort removes a container by name, used to guarantee
// cleanup of a container that outran its timeout.
func (s *CLI) forceRemove(name string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, s.runtime, removeArgs(name)...)
	_ = cmd.Run()
}

// cpuBusyThreshold is the container CPU percentage above which a run counts as
// making progress even when it writes nothing — e.g. a compiler working on one
// large generated source file. Below it, CPU is treated as idle.
const cpuBusyThreshold = 1.0

// cpuProbeTimeout bounds a single CPU probe; `stats --no-stream` samples for
// about a second, so this stays generous but finite.
const cpuProbeTimeout = 5 * time.Second

// containerCPUBusy reports whether the named container is currently consuming
// CPU above cpuBusyThreshold. It is a BEST-EFFORT progress signal: any failure
// (runtime quirk, container already gone, unparsable output) returns false so
// the watchdog falls back to the output/filesystem signals. It can only PREVENT
// a false idle-kill, never cause one. `--format {{.CPUPerc}}` (e.g. "12.34%")
// is supported by both podman and docker.
func (s *CLI) containerCPUBusy(name string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), cpuProbeTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, s.runtime, "stats", "--no-stream", "--format", "{{.CPUPerc}}", name).Output()
	if err != nil {
		return false
	}
	field := strings.TrimSuffix(strings.TrimSpace(string(out)), "%")
	pct, err := strconv.ParseFloat(field, 64)
	if err != nil {
		return false
	}
	return pct > cpuBusyThreshold
}

var _ Sandbox = (*CLI)(nil)

package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// bwrapProbeTimeout bounds DetectBwrap's userns capability probe. A bwrap
// running /bin/true inside an unshared user namespace exits almost instantly;
// this stays generous without letting a wedged host hang detection.
const bwrapProbeTimeout = 5 * time.Second

// DetectBwrap reports whether the bwrap backend is usable on this host and,
// when it is not, an actionable reason. Three independent conditions gate
// usability, checked in the order a user would want to fix them:
//  1. the host must be Linux (bwrap depends on Linux-only namespace syscalls);
//  2. the bwrap binary must be on PATH;
//  3. unprivileged user namespaces must actually work — some distributions
//     ship bwrap but disable unprivileged userns via sysctl
//     (kernel.unprivileged_userns_clone=0) or an AppArmor profile (Ubuntu
//     24.04's default "restrict unprivileged user namespaces"), in which case
//     bwrap is present but every real run would fail. This is probed directly
//     (attempting the actual unshare) rather than by reading a specific
//     sysctl, since the gate mechanism differs across distributions (sysctl
//     vs AppArmor vs SELinux) and a direct probe is the one check that agrees
//     with reality on all of them.
func DetectBwrap() (ok bool, reason string) {
	if runtime.GOOS != "linux" {
		return false, fmt.Sprintf("bwrap backend requires Linux (running on %s); use the container CLI backend (NewCLI, podman/docker) instead", runtime.GOOS)
	}
	path, err := exec.LookPath("bwrap")
	if err != nil {
		return false, "bwrap not found on PATH; install bubblewrap (e.g. `apt install bubblewrap` / `dnf install bubblewrap` / `pacman -S bubblewrap`)"
	}
	if err := probeBwrapUserns(path); err != nil {
		return false, fmt.Sprintf("unprivileged user namespaces are unavailable (%v); check kernel.unprivileged_userns_clone, an AppArmor userns-restriction profile, or run as a user with CAP_SYS_ADMIN — or use the container CLI backend (NewCLI, podman/docker) instead", err)
	}
	return true, ""
}

// probeBwrapUserns attempts the smallest possible real bwrap run — unshare
// every namespace and immediately exit — to prove unprivileged user
// namespaces actually work end to end, rather than inferring it from a
// sysctl file whose name and meaning vary across distributions.
//
// The probe needs SOME executable reachable inside the sandbox but must not
// assume fixedROAllowlist's paths exist: on non-FHS hosts (NixOS, Guix) /bin
// and /sbin are absent entirely, and even `true` lives under /nix/store or
// /run rather than /bin. Since this ephemeral, immediately-exiting process
// runs nothing untrusted and only answers "does unshare(2) actually work
// here", binding the entire host root read-only (rather than the real run's
// narrow allowlist) is safe and portable across distro layouts.
//
// INVARIANT: --ro-bind / / below is acceptable ONLY because this probe
// always execs a fixed, hardcoded, trusted command (`true`, resolved by this
// function itself) — it must NEVER be reused to run caller-supplied or
// model-generated code. Every other bwrap invocation in this package goes
// through buildBwrapArgs' narrow fixedROAllowlist instead; this is the one
// deliberate exception.
func probeBwrapUserns(bwrapPath string) error {
	truePath, err := exec.LookPath("true")
	if err != nil {
		// Belt-and-suspenders: every POSIX system ships a `true` somewhere
		// on PATH, but fallback in case PATH is unusually stripped down.
		truePath = "/bin/true"
	}
	ctx, cancel := context.WithTimeout(context.Background(), bwrapProbeTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bwrapPath, "--unshare-all", "--die-with-parent", "--ro-bind", "/", "/", truePath)
	if out, err := cmd.CombinedOutput(); err != nil {
		if len(out) > 0 {
			return fmt.Errorf("%s: %s", err, string(out))
		}
		return err
	}
	return nil
}

// Bwrap is a Sandbox backed by the host's bubblewrap binary: an
// unprivileged-user-namespace sandbox that runs directly on host toolchains
// instead of a baked container image (see bwrap_command.go for the security
// posture). It is safe for concurrent use: each Exec prepares its own
// workspace and launches its own bwrap process.
//
// A non-empty Spec.Image is REFUSED at Exec (UnsupportedSpecError) rather
// than silently ignored. To run with host toolchains, construct the backend
// with WithHostToolchains(res): every Exec on it sees the toolchains.
type Bwrap struct {
	bwrapPath string
	// defaults: shared per-run state (CPU/memory/pids caps, absolute+idle
	// timeouts, network mode, output cap, scratch size, workspace-growth
	// ceiling). Backed by options.baseDefaults, configured via the one
	// shared Option type.
	defaults
	// capPolicy: CapRequired (default) fails loudly with ErrBwrapNoCapMethod
	// when neither systemd-run --user --scope nor a delegated cgroup v2
	// subtree is available; CapBestEffort (WithCapPolicy) runs uncapped.
	capPolicy CapPolicy
	// toolchainBinds: extra read-only binds (beyond fixedROAllowlist)
	// resolved by the host-toolchain resolver, applied to every run.
	toolchainBinds []ROMount
	// toolchainPathPrepend: ResolveHostToolchains' pathPrepend for the same
	// resolution that produced toolchainBinds; see
	// bwrapParams.toolchainPathPrepend for how it reaches buildBwrapArgs.
	toolchainPathPrepend string
	// baselinePathAppend: ":"-joined in-sandbox directories of the resolved
	// POSIX baseline utilities, appended AFTER defaultContainerPath so it
	// never shadows allowlist binaries or operator toolchains. Empty on FHS
	// hosts. The matching binds are merged into toolchainBinds at
	// construction.
	baselinePathAppend string
	// wsCache: pristine-materialization cache backing prepareWorkspace,
	// shared with the CLI backend via prepareWorkspaceCached (workspace.go).
	wsCache wsCache
	// expandSupport: caches whether this host's systemd-run accepts
	// --expand-environment=no, so the version probe runs once per instance
	// rather than on every Exec.
	expandSupport expandSupportCache
}

// NewBwrap constructs a Bwrap sandbox. It fails fast with the same
// actionable reasons as DetectBwrap when the backend is not usable on this
// host, so a misconfigured backend choice is caught at construction time
// rather than on the first real run. The CLI-only options (WithRuntime,
// WithImage) are refused here with an error naming the option; a numeric
// option out of range is refused before any runtime lookup; a WithNetwork
// mode bwrap could never honor fails at construction.
func NewBwrap(opts ...Option) (*Bwrap, error) {
	o := newOptions(opts)
	if err := o.checkSupported(backendBwrap, cliOnlyOptions); err != nil {
		return nil, err
	}
	if err := o.checkNumericOptions(backendBwrap); err != nil {
		return nil, err
	}
	if err := validateNetworkDefault(backendBwrap, o.network, bwrapNetworks); err != nil {
		return nil, err
	}
	ok, reason := DetectBwrap()
	if !ok {
		return nil, errors.New("sandbox: " + reason)
	}
	path, err := exec.LookPath("bwrap")
	if err != nil {
		return nil, fmt.Errorf("sandbox: bwrap not found on PATH: %w", err)
	}
	s := &Bwrap{bwrapPath: path}
	s.defaults = baseDefaults()
	o.applyDefaults(&s.defaults)
	if o.has("WithCapPolicy") {
		s.capPolicy = o.capPolicy
	}
	if o.has("WithHostToolchains") {
		s.toolchainBinds = o.toolchainBinds
		s.toolchainPathPrepend = o.toolchainPathPrepend
	}

	// POSIX baseline provisioning: container images guarantee a shell + core
	// utilities structurally; the bwrap tmpfs root does not. On store-based
	// distros (NixOS, Guix) defaultContainerPath's FHS dirs hold only sh and
	// env, so caller machinery (`mkdir -p` setup commands, agent-planned
	// `sh -c` scripts) fails with "mkdir: command not found". Resolve the
	// baseline from the host once per construction; on FHS hosts this is a
	// no-op (empty baseline). Operator toolchainBinds (WithHostToolchains)
	// win any ContainerPath collision — pinning e.g. "bash" overrides the
	// baseline resolution of the same name.
	baseMounts, basePath := resolveBwrapBaseline(exec.LookPath, filepath.EvalSymlinks)
	s.baselinePathAppend = basePath
	seen := make(map[string]bool, len(s.toolchainBinds))
	for _, m := range s.toolchainBinds {
		seen[m.ContainerPath] = true
	}
	for _, m := range baseMounts {
		if !seen[m.ContainerPath] {
			s.toolchainBinds = append(s.toolchainBinds, m)
		}
	}

	purgeStaleWorkspaceCaches()
	return s, nil
}

// bwrapBaselineUtilities names the POSIX utilities every sandbox run may
// assume, mirroring what any container base image ships. One name per host
// PACKAGE is enough on store-based distros (resolving "mkdir" mounts the
// whole coreutils applet directory; "find" brings xargs's findutils dir —
// xargs is still listed for hosts that split them), and resolveBwrapBaseline
// deduplicates by resolved directory. sh is NOT listed: it is reached by the
// literal /bin/sh path through the fixed allowlist + store binds, not via
// PATH.
var bwrapBaselineUtilities = []string{
	"mkdir", // coreutils: cp, rm, cat, mv, touch, dirname, tee, ... (one package dir)
	"grep",
	"sed",
	"awk",
	"find",
	"xargs",
	"diff",
	"tar",
	"gzip",
	"bash",
	// which is NOT a coreutils applet (GNU which on NixOS, debianutils on
	// Debian, a busybox applet on Alpine) but every container base image
	// ships it, and a missing which makes agents fall back to `command -v`
	// after a spurious environment_error.
	"which",
}

// resolveBwrapBaseline resolves bwrapBaselineUtilities into read-only binds
// and a ":"-joined PATH APPEND for the bwrap sandbox. Filtering happens
// before the toolchain resolver runs:
//
//   - a utility whose symlink-resolved home is already a defaultContainerPath
//     directory is skipped — it is reachable through the fixed allowlist
//     binds, so FHS hosts resolve an EMPTY baseline and keep byte-identical
//     sandbox argv (no extra mounts, no PATH suffix, no --version probes);
//   - utilities sharing one resolved directory (nix coreutils-full's
//     multi-call applets, findutils' find+xargs) collapse to the first name,
//     so the resolver mounts each host package once.
//
// Survivors go through ResolveHostToolchains — the single mount-shaping
// implementation (see toolchain.go's backend contract) — with fingerprints
// discarded: baseline utilities are plumbing, not verdict-relevant
// toolchains. Best-effort throughout: an unresolvable utility is silently
// skipped (recorded in ToolchainResolution.Unresolved, which this function
// discards along with the fingerprints) — the run then fails with the same
// command-not-found it would have hit anyway.
//
// lookPath and evalSymlinks are injected for testability (exec.LookPath and
// filepath.EvalSymlinks in production).
func resolveBwrapBaseline(lookPath func(string) (string, error), evalSymlinks func(string) (string, error)) ([]ROMount, string) {
	names := filterBwrapBaseline(bwrapBaselineUtilities, lookPath, evalSymlinks)
	if len(names) == 0 {
		return nil, ""
	}
	res := ResolveHostToolchains(names)
	return res.mounts, res.pathPrepend
}

// filterBwrapBaseline applies resolveBwrapBaseline's filtering rules (see
// its doc) to names: drop unresolvable utilities, drop utilities already
// reachable through a defaultContainerPath directory, and collapse
// utilities sharing one resolved home directory to the first name. Pure with
// respect to its injected lookups, so the FHS-no-op and
// store-layout-dedupe guarantees are unit-testable without a store-based host.
func filterBwrapBaseline(names []string, lookPath func(string) (string, error), evalSymlinks func(string) (string, error)) []string {
	defaultDirs := make(map[string]bool)
	for _, d := range strings.Split(defaultContainerPath, ":") {
		defaultDirs[d] = true
	}
	seenDirs := make(map[string]bool)
	var kept []string
	for _, name := range names {
		p, err := lookPath(name)
		if err != nil {
			continue
		}
		resolved, err := evalSymlinks(p)
		if err != nil {
			continue
		}
		dir := filepath.Dir(resolved)
		if defaultDirs[dir] || seenDirs[dir] {
			continue
		}
		seenDirs[dir] = true
		kept = append(kept, name)
	}
	return kept
}

// Close removes this Bwrap instance's workspace-cache parent directory, if
// one was ever materialized. See CLI.Close's doc comment for the concurrency
// contract; the same reasoning applies. Safe to call on a nil receiver and
// multiple times.
func (s *Bwrap) Close() error {
	if s == nil {
		return nil
	}
	return s.wsCache.close()
}

// MaterializeWorkspace clones the pristine-workspace cache for repoDir into
// a fresh, caller-owned workspace directory. See CLI.MaterializeWorkspace's
// doc comment; the contract is identical.
func (s *Bwrap) MaterializeWorkspace(repoDir string) (string, error) {
	ws, _, err := prepareWorkspaceCached(&s.wsCache, repoDir, nil)
	return ws, err
}

// resolveBwrapParams applies backend defaults to a Spec, producing the
// concrete bwrapParams for the run (workspace is filled in by Exec). The
// network mode must be one bwrap can honor. A non-empty Spec.Image never
// reaches here: validateSpec refuses it.
func (s *Bwrap) resolveBwrapParams(spec Spec) (bwrapParams, error) {
	network, err := resolveNetworkMode(backendBwrap, s.defaultNetwork, spec.Network, bwrapNetworks...)
	if err != nil {
		return bwrapParams{}, err
	}
	return bwrapParams{
		network:              network,
		env:                  spec.Env,
		cmd:                  spec.Cmd,
		roMounts:             spec.ROMounts,
		rwMounts:             spec.RWMounts,
		setupCmds:            spec.SetupCmds,
		toolchainBinds:       s.toolchainBinds,
		toolchainPathPrepend: s.toolchainPathPrepend,
		baselinePathAppend:   s.baselinePathAppend,
		scratchSizeBytes:     int64(s.defaultScratchSizeMB) * 1024 * 1024,
	}, nil
}

// Exec implements Sandbox. Its error contract is the Sandbox interface's:
// an error only for a refused Spec (InvalidSpecError, UnsupportedSpecError),
// a caller ctx that ended (the "sandbox: execution cancelled" error), or an
// infrastructure failure (ErrBwrapNoCapMethod among them); a non-zero exit
// code is reported in Result.ExitCode. Process supervision (deadline,
// watchdog, growth ceiling, outcome precedence, reap discipline) lives in
// run.go's runSupervised; this body keeps only the bwrap-specific pieces:
// Spec admission, cap-method resolution (after the Workspace checks, before
// any write), and the resource-cap wrapper (with --die-with-parent + a
// process-group kill replacing container rm — bwrap has no daemon-tracked
// object for a "docker rm -f" equivalent to reap).
func (s *Bwrap) Exec(ctx context.Context, spec Spec) (Result, error) {
	if err := validateSpec(backendBwrap, spec); err != nil {
		return Result{}, err
	}
	// resolveBwrapParams is the "Spec + defaults -> argv inputs" stage;
	// per-backend Spec admission already ran in validateSpec.
	p, err := s.resolveBwrapParams(spec)
	if err != nil {
		return Result{}, err
	}

	cpus := s.defaultCPUs
	memoryMB := s.defaultMemory

	// Spec.Timeout <= 0 resolves to the backend default here; the supervisor
	// applies the resolved value as the run deadline verbatim.
	timeout := spec.Timeout
	if timeout <= 0 {
		timeout = s.defaultTimeout
	}

	// Cap method is resolved by the admit hook — after the supervisor's
	// Workspace checks (so a bad Workspace is refused as such on every
	// host) and before any write. The cap wrapper (systemd-run scope,
	// cgroup v2 subtree, or none) wraps the bwrap argv and is built inside
	// buildCmd — it needs the workspace-filled argv. Its cleanup always
	// runs, whether or not the run started.
	var capMethod bwrapCapMethod
	var wrap resourceCapWrap
	defer func() {
		if wrap.cleanup != nil {
			wrap.cleanup()
		}
	}()

	// pidForCPU feeds the watchdog's /proc-tree CPU probe; stored right
	// after Start.
	var pidForCPU atomic.Int64
	return runSupervised(ctx, runSpec{
		spec:           spec,
		timeout:        timeout,
		idleTimeout:    s.defaultIdleTimeout,
		growthCeiling:  s.defaultGrowthCeilingBytes,
		maxOutputBytes: s.maxOutputBytes,
		hooks: runHooks{
			admit: func(ctx context.Context) error {
				capMethod = detectCapMethod(ctx, s.expandSupport.supported)
				if capMethod == bwrapCapNone && s.capPolicy != CapBestEffort {
					return ErrBwrapNoCapMethod
				}
				return nil
			},
			prepareWorkspace: func(repoDir string) (string, bool, error) {
				return prepareWorkspaceCached(&s.wsCache, repoDir, spec.WriteFiles)
			},
			buildCmd: func(ws string, runCtx context.Context) (*exec.Cmd, error) {
				p.workspace = ws
				w, err := s.newResourceCapWrap(capMethod, buildBwrapArgs(p), cpus, memoryMB, s.pidsLimit)
				if err != nil {
					return nil, err
				}
				wrap = w
				cmd := exec.CommandContext(runCtx, w.name, w.args...)
				setBwrapProcAttr(cmd)
				return cmd, nil
			},
			afterStart: func(cmd *exec.Cmd) error {
				if cmd.Process != nil {
					pidForCPU.Store(int64(cmd.Process.Pid))
					if joinErr := wrap.joinCgroup(cmd.Process.Pid); joinErr != nil {
						return joinErr
					}
				}
				return nil
			},
			cpuBusy: func() bool {
				pid := pidForCPU.Load()
				if pid == 0 {
					return false
				}
				return procTreeCPUBusy(int(pid))
			},
			reap: func(cmd *exec.Cmd) { killBwrapProcessGroup(cmd) },
		},
	})
}

// resourceCapWrap is the resolved (binary, args) to exec for a run's cap
// method, plus the hooks Exec drives at the right point in the process
// lifecycle: joinCgroup (called once, right after Start, with the started
// pid) and cleanup (always called via defer, whether or not the run
// succeeded).
type resourceCapWrap struct {
	name       string
	args       []string
	joinCgroup func(pid int) error
	cleanup    func()
}

// newResourceCapWrap builds the resourceCapWrap for the resolved cap method:
//   - systemd-run: the bwrap argv is wrapped in a transient --user --scope
//     unit at exec time; MemoryMax/CPUQuota/TasksMax are systemd unit
//     properties, so no post-Start action is needed and --collect lets
//     systemd reap the transient unit itself (cleanup is a no-op).
//   - cgroup v2: a delegated subtree is created with the resolved limits
//     BEFORE Start (cgroup v2 requires controllers configured before
//     population), bwrap execs directly (no wrapper binary), and joinCgroup
//     moves the just-started pid into the subtree by writing it to
//     cgroup.procs — cgroup v2 membership propagates to every process the
//     joined pid subsequently forks/execs, so bwrap and everything it runs
//     inside the sandbox is covered. cleanup removes the (by-then-empty)
//     subtree.
//   - none (only reachable with CapBestEffort): bwrap execs directly with
//     no wrapper and no cap enforcement.
func (s *Bwrap) newResourceCapWrap(method bwrapCapMethod, bwrapArgv []string, cpus float64, memoryMB, pidsLimit int) (resourceCapWrap, error) {
	noop := func(int) error { return nil }
	switch method {
	case bwrapCapSystemdRun:
		full := systemdRunWrapArgs(s.bwrapPath, bwrapArgv, cpus, memoryMB, pidsLimit)
		return resourceCapWrap{name: full[0], args: full[1:], joinCgroup: noop, cleanup: func() {}}, nil
	case bwrapCapCgroupV2:
		parent, ok := delegatedCgroupV2Dir()
		if !ok {
			// Lost the delegated subtree between detection and use (e.g. a
			// concurrent process reconfigured cgroups); fail rather than
			// silently run uncapped.
			return resourceCapWrap{}, ErrBwrapNoCapMethod
		}
		dir := filepath.Join(parent, "llmkit-"+randToken())
		if err := os.Mkdir(dir, 0o755); err != nil {
			return resourceCapWrap{}, fmt.Errorf("sandbox: create cgroup subtree: %w", err)
		}
		if err := writeCgroupLimits(dir, cpus, memoryMB, pidsLimit); err != nil {
			_ = os.Remove(dir)
			return resourceCapWrap{}, err
		}
		return resourceCapWrap{
			name: s.bwrapPath,
			args: bwrapArgv,
			joinCgroup: func(pid int) error {
				if err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte(strconv.Itoa(pid)), 0o644); err != nil {
					return fmt.Errorf("sandbox: join cgroup subtree: %w", err)
				}
				return nil
			},
			cleanup: func() { _ = os.Remove(dir) },
		}, nil
	default:
		return resourceCapWrap{name: s.bwrapPath, args: bwrapArgv, joinCgroup: noop, cleanup: func() {}}, nil
	}
}

// writeCgroupLimits writes the resolved memory.max/cpu.max/pids.max files
// into dir. A limit that resolves to "omit" (see cgroupV2Limits) is left
// unwritten, so that controller inherits the parent's limit rather than
// being reset to "max".
func writeCgroupLimits(dir string, cpus float64, memoryMB, pidsLimit int) error {
	memory, cpuMax, pids, memOK, cpuOK, pidsOK := cgroupV2Limits(cpus, memoryMB, pidsLimit)
	if memOK {
		if err := os.WriteFile(filepath.Join(dir, "memory.max"), []byte(memory), 0o644); err != nil {
			return fmt.Errorf("sandbox: write memory.max: %w", err)
		}
	}
	if cpuOK {
		if err := os.WriteFile(filepath.Join(dir, "cpu.max"), []byte(cpuMax), 0o644); err != nil {
			return fmt.Errorf("sandbox: write cpu.max: %w", err)
		}
	}
	if pidsOK {
		if err := os.WriteFile(filepath.Join(dir, "pids.max"), []byte(pids), 0o644); err != nil {
			return fmt.Errorf("sandbox: write pids.max: %w", err)
		}
	}
	return nil
}

// procTreeCPUBusy is the bwrap backend's activeFallback progress signal:
// it reports whether the process tree rooted at pid is currently consuming
// CPU above cpuBusyThreshold, sampled via /proc — the host-process analogue
// of containerCPUBusy's `runtime stats` probe, since bwrap has no container
// object to query.
func procTreeCPUBusy(pid int) bool {
	before, ok := procTreeCPUTicks(pid)
	if !ok {
		return false
	}
	time.Sleep(cpuSampleWindow)
	after, ok := procTreeCPUTicks(pid)
	if !ok {
		return false
	}
	deltaTicks := after - before
	if deltaTicks <= 0 {
		return false
	}
	// Convert ticks busy over the sample window into a CPU percentage
	// (100% == one full core continuously busy for the whole window) and
	// compare against the same threshold containerCPUBusy uses, so the two
	// backends' idle-watchdog "still working" semantics agree.
	ticksPerSec := float64(clockTicksPerSec)
	pct := (float64(deltaTicks) / ticksPerSec) / cpuSampleWindow.Seconds() * 100
	return pct > cpuBusyThreshold
}

// cpuSampleWindow is how long procTreeCPUBusy waits between its two /proc
// samples. Short enough to keep an idle-tick's added latency negligible,
// long enough that scheduler jitter doesn't dominate the measurement.
const cpuSampleWindow = 200 * time.Millisecond

// clockTicksPerSec is USER_HZ, the unit /proc/<pid>/stat's utime/stime
// fields are expressed in on Linux. 100 on every architecture Go supports.
const clockTicksPerSec = 100

// procTreeCPUTicks sums utime+stime (in clock ticks) across pid and every
// live descendant, discovered via /proc/<pid>/task/<tid>/children (Linux
// 3.5+, present on any kernel modern enough to run unprivileged userns
// sandboxes). Best-effort: any failure (process exited mid-walk, kernel
// lacks the children file) returns ok=false so the caller falls back to
// treating the tick as idle, exactly like containerCPUBusy's error handling.
func procTreeCPUTicks(root int) (ticks int64, ok bool) {
	rootTicks, rootOK := procStatTicks(root)
	if !rootOK {
		// The root pid must resolve — otherwise this is "process does not
		// exist", not "process with zero CPU usage", and the caller must
		// treat it as unknown (ok=false), not zero ticks.
		return 0, false
	}
	seen := map[int]bool{root: true}
	total := rootTicks
	var walk func(pid int)
	walk = func(pid int) {
		for _, c := range procChildren(pid) {
			if seen[c] {
				continue
			}
			seen[c] = true
			if t, statOK := procStatTicks(c); statOK {
				total += t
			}
			walk(c)
		}
	}
	walk(root)
	return total, true
}

// procStatTicks reads utime (field 14) and stime (field 15) from
// /proc/<pid>/stat. The comm field (2nd, parenthesized) may itself contain
// spaces, so fields are counted from the END of the line rather than
// splitting naively on whitespace from the start.
func procStatTicks(pid int) (int64, bool) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, false
	}
	line := string(data)
	closeParen := lastIndexByte(line, ')')
	if closeParen < 0 || closeParen+2 >= len(line) {
		return 0, false
	}
	rest := splitFields(line[closeParen+2:])
	// rest[0] is field 3 (state); utime is field 14 (rest[11]); stime is
	// field 15 (rest[12]).
	if len(rest) < 13 {
		return 0, false
	}
	utime, err1 := strconv.ParseInt(rest[11], 10, 64)
	stime, err2 := strconv.ParseInt(rest[12], 10, 64)
	if err1 != nil || err2 != nil {
		return 0, false
	}
	return utime + stime, true
}

// procChildren returns pid's direct child pids via the Linux 3.5+
// /proc/<pid>/task/<tid>/children interface, aggregated across every thread
// of pid (a multi-threaded process can parent children from any thread).
// Best-effort: a missing/unreadable children file for one thread is skipped
// rather than failing the whole call, since a thread can legitimately exit
// mid-walk.
func procChildren(pid int) []int {
	taskDir := fmt.Sprintf("/proc/%d/task", pid)
	entries, err := os.ReadDir(taskDir)
	if err != nil {
		return nil
	}
	var out []int
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(taskDir, e.Name(), "children"))
		if err != nil {
			continue
		}
		for _, f := range splitFields(string(data)) {
			if n, err := strconv.Atoi(f); err == nil {
				out = append(out, n)
			}
		}
	}
	return out
}

func lastIndexByte(s string, b byte) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == b {
			return i
		}
	}
	return -1
}

func splitFields(s string) []string {
	var fields []string
	start := -1
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' || s[i] == '\n' || s[i] == '\t' {
			if start >= 0 {
				fields = append(fields, s[start:i])
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		fields = append(fields, s[start:])
	}
	return fields
}

var _ Sandbox = (*Bwrap)(nil)

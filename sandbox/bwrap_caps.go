package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// bwrap has no cgroups (and no daemon) of its own to enforce resource limits,
// unlike the container backend's --memory/--cpus/--pids-limit flags (see
// buildRunArgs). Two host mechanisms can still enforce them:
//
//   - systemd-run --user --scope wraps the bwrap invocation in a transient
//     systemd unit and applies MemoryMax/CPUQuota/TasksMax via the user's
//     systemd instance — no privilege beyond a running `systemd --user`
//     session and a login manager that set up the delegate hierarchy (true
//     for any modern systemd-logind desktop/server).
//   - cgroup v2 direct: write memory.max/cpu.max/pids.max into a delegated
//     subtree of the CALLING process's own cgroup (found via
//     /proc/self/cgroup) and move the bwrap pid into it after launch.
//
// When NEITHER is available, resource limits would otherwise be silently
// dropped; the run fails with an actionable error unless the operator opts
// into WithCapPolicy(CapBestEffort).

// bwrapCapMethod names the resource-limit enforcement mechanism a Bwrap
// backend resolved for the current host.
type bwrapCapMethod int

const (
	// bwrapCapNone means neither systemd-run --user --scope nor a delegated
	// cgroup v2 subtree is usable on this host.
	bwrapCapNone bwrapCapMethod = iota
	bwrapCapSystemdRun
	bwrapCapCgroupV2
)

// systemdRunProbeTimeout bounds the one-time check that the user's systemd
// instance is actually reachable (not merely that the binaries exist on
// PATH) — a headless host with systemd-run installed but no user session bus
// running must fail this probe, not silently be believed capable.
const systemdRunProbeTimeout = 5 * time.Second

// detectBwrapCapMethod picks the best available resource-limit mechanism.
// Every step is best-effort and cheap; failures fall through to the next
// method rather than erroring, since the ultimate "nothing worked" case is
// handled by the caller (Exec), which decides whether that is fatal based on
// capPolicy. expandSupport answers whether this host's systemd-run accepts
// --expand-environment=no (systemdRunExpandSupport, or a Bwrap instance's
// cached answer).
func detectBwrapCapMethod(ctx context.Context, expandSupport func(context.Context) bool) bwrapCapMethod {
	if systemdRunUserAvailable(ctx) && expandSupport(ctx) {
		return bwrapCapSystemdRun
	}
	if _, ok := delegatedCgroupV2Dir(); ok {
		return bwrapCapCgroupV2
	}
	return bwrapCapNone
}

// systemdRunMinNoExpandVersion is the first systemd whose systemd-run
// understands --expand-environment=no (v254). Passing the flag to an older
// systemd-run fails the exec outright — preferred over silently launching
// with host-env expansion.
const systemdRunMinNoExpandVersion = 254

// systemdRunVersionWaitDelay bounds how long the version probe waits for
// its output pipe to close after systemd-run exits or is killed: a
// systemd-run that leaves a child holding stdout open must not stretch the
// probe past its own timeout.
const systemdRunVersionWaitDelay = time.Second

// systemdRunVersion runs `systemd-run --version` and returns its stdout.
// Test seam: tests override it to feed systemdRunExpandSupport canned output.
var systemdRunVersion = func(ctx context.Context) ([]byte, error) {
	probeCtx, cancel := context.WithTimeout(ctx, systemdRunProbeTimeout)
	defer cancel()
	cmd := exec.CommandContext(probeCtx, "systemd-run", "--version")
	cmd.WaitDelay = systemdRunVersionWaitDelay
	return cmd.Output()
}

// systemdRunExpandSupport reports whether this host's systemd-run accepts
// --expand-environment=no. Without that flag, systemd-run expands ${NAME}
// references inside the wrapped argv from the HOST environment before bwrap
// starts (measured on systemd 261: with ZZHOSTONLY set on the host, sh -c
// 'echo brace=${ZZHOSTONLY}' printed the host value inside the sandbox) — a
// host-env leak a model-driven Cmd can read through. A NO answer — including
// a probe that fails, times out, or prints anything but a parseable
// "systemd <major>" first line — fails CLOSED: detectBwrapCapMethod never
// resolves the systemd-run method, degrading to a delegated cgroup v2
// subtree (or bwrapCapNone, honoring WithCapPolicy) — enforcement may be
// absent, but never "enforcement plus a leak".
func systemdRunExpandSupport(ctx context.Context) bool {
	out, err := systemdRunVersion(ctx)
	return err == nil && systemdRunSupportsNoExpand(out)
}

// systemdRunSupportsNoExpand parses `systemd-run --version` output: the
// first line must be "systemd <major> ..." with an integer major of at
// least systemdRunMinNoExpandVersion. Any other shape is a NO.
func systemdRunSupportsNoExpand(versionOutput []byte) bool {
	fields := strings.Fields(strings.SplitN(string(versionOutput), "\n", 2)[0])
	if len(fields) < 2 || fields[0] != "systemd" {
		return false
	}
	major, err := strconv.Atoi(fields[1])
	if err != nil {
		return false
	}
	return major >= systemdRunMinNoExpandVersion
}

// expandSupportCache remembers one Bwrap instance's systemdRunExpandSupport
// answer, so the version probe runs once per instance instead of on every
// Exec. Only an answer the probe reached with the caller's ctx still live is
// cached: a probe cut short by the caller is asked again next time. A probe
// that fails or times out on its own is a cached NO for the instance's
// lifetime — fail-closed, and bounded to one probe timeout per instance
// (the probe runs under mu, so re-probing on every Exec would serialize
// concurrent Execs behind a hung systemd-run one timeout at a time). A fresh
// Bwrap probes again. The zero value is ready to use.
type expandSupportCache struct {
	mu    sync.Mutex
	known bool
	ok    bool
}

// supported returns the cached answer, probing on first use.
func (c *expandSupportCache) supported(ctx context.Context) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.known {
		return c.ok
	}
	ok := systemdRunExpandSupport(ctx)
	if ctx.Err() == nil {
		c.known, c.ok = true, ok
	}
	return ok
}

// DescribeBwrapCapMethod reports the resource-limit enforcement mechanism
// this host currently supports for the bwrap backend, for doctor's advisory
// reporting: "enforced" (systemd-run --user --scope or cgroup v2 available,
// so runs get their configured caps) or a reason why neither is available
// (runs would fail unless the allow-uncapped option is set — see
// ErrBwrapNoCapMethod). The label names the mechanism only; remediation
// guidance belongs to the caller, which wraps ErrBwrapNoCapMethod with its
// own operator-facing hint.
func DescribeBwrapCapMethod(ctx context.Context) (label string, enforced bool) {
	switch detectBwrapCapMethod(ctx, systemdRunExpandSupport) {
	case bwrapCapSystemdRun:
		return "systemd-run --user --scope", true
	case bwrapCapCgroupV2:
		return "delegated cgroup v2 subtree", true
	default:
		return fmt.Sprintf("none (neither systemd-run --user --scope on systemd >= %d nor a delegated cgroup v2 subtree)", systemdRunMinNoExpandVersion), false
	}
}

// systemdRunUserAvailable reports whether `systemd-run --user --scope` can
// actually launch a transient unit on this host: both systemd-run and
// systemctl must be on PATH, and the user's systemd instance must answer a
// trivial query within systemdRunProbeTimeout. A binary merely existing on
// PATH is not sufficient — a container or minimal host can have the client
// tools installed with no systemd PID 1 and no user session bus behind them.
// Test seam: tests override it so detection does not depend on the host.
var systemdRunUserAvailable = func(ctx context.Context) bool {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		return false
	}
	if _, err := exec.LookPath("systemctl"); err != nil {
		return false
	}
	probeCtx, cancel := context.WithTimeout(ctx, systemdRunProbeTimeout)
	defer cancel()
	cmd := exec.CommandContext(probeCtx, "systemctl", "--user", "show-environment")
	return cmd.Run() == nil
}

// cgroupV2Root is the standard cgroup v2 mount point.
const cgroupV2Root = "/sys/fs/cgroup"

// delegatedCgroupV2Dir returns the directory of the calling process's own
// cgroup v2 membership, if the host mounts cgroup v2 (cgroup.controllers
// present at cgroupV2Root) AND that directory is writable by this process —
// the signal that the cgroup was delegated to the (possibly unprivileged)
// user. A subdirectory created there inherits delegation and can set
// memory.max/cpu.max/pids.max for its own descendants.
func delegatedCgroupV2Dir() (string, bool) {
	if _, err := os.Stat(cgroupV2Root + "/cgroup.controllers"); err != nil {
		return "", false
	}
	self, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", false
	}
	// cgroup v2 has exactly one line: "0::<path>".
	line := strings.TrimSpace(string(self))
	parts := strings.SplitN(line, ":", 3)
	if len(parts) != 3 {
		return "", false
	}
	dir := cgroupV2Root + parts[2]
	// Writability is the actual delegation signal: probe with a throwaway
	// subdirectory rather than trusting ownership bits, since ACLs/systemd
	// delegation can grant write access without matching Unix ownership.
	probe := dir + "/.llmkit-cgroup-probe"
	if err := os.Mkdir(probe, 0o755); err != nil {
		return "", false
	}
	_ = os.Remove(probe)
	return dir, true
}

// ErrBwrapNoCapMethod is returned (possibly wrapped) by Bwrap.Exec when
// resource limits were requested (the normal case) but neither enforcement
// mechanism is available and the operator has not opted into running
// uncapped via WithCapPolicy(CapBestEffort). Callers match it with
// errors.Is to attach their own remediation (e.g. a config key) instead of
// string matching.
var ErrBwrapNoCapMethod = errors.New("sandbox: bwrap backend found no resource-limit mechanism (systemd-run --user --scope on systemd >= " + strconv.Itoa(systemdRunMinNoExpandVersion) + ", which is required for --expand-environment=no, or a delegated cgroup v2 subtree); set WithCapPolicy(CapBestEffort) to run without enforced memory/CPU/pids limits")

// systemdRunWrapArgs prepends a systemd-run --user --scope invocation (with
// MemoryMax/CPUQuota/TasksMax properties) around the given bwrap binary +
// args, returning the full argv to exec. cpus <= 0 / memoryMB <= 0 / pids <=
// 0 omit the corresponding property, matching buildRunArgs' "omit when
// unset" convention for --memory/--cpus/--pids-limit.
func systemdRunWrapArgs(bwrapPath string, bwrapArgs []string, cpus float64, memoryMB, pidsLimit int) []string {
	// --expand-environment=no: bwrapArgs is UNTRUSTED Spec.Cmd content and
	// must reach bwrap byte-for-byte. Without the flag, systemd-run expands
	// ${NAME} in the command line from the HOST environment (see
	// systemdRunExpandSupport). The support probe behind
	// detectBwrapCapMethod guarantees the flag is only ever passed to a
	// systemd-run that accepts it.
	args := []string{"systemd-run", "--user", "--scope", "--quiet", "--collect", "--expand-environment=no"}
	if memoryMB > 0 {
		args = append(args, "-p", fmt.Sprintf("MemoryMax=%dM", memoryMB))
	}
	if cpus > 0 {
		// CPUQuota is a percentage of one core; 1.5 cores -> "150%".
		pct := cpus * 100
		args = append(args, "-p", fmt.Sprintf("CPUQuota=%s%%", strconv.FormatFloat(pct, 'f', -1, 64)))
	}
	if pidsLimit > 0 {
		args = append(args, "-p", fmt.Sprintf("TasksMax=%d", pidsLimit))
	}
	args = append(args, "--", bwrapPath)
	args = append(args, bwrapArgs...)
	return args
}

// cgroupV2Limits renders the raw file contents for a delegated cgroup v2
// subtree's memory.max, cpu.max, and pids.max controllers. cpu.max's format
// is "<quota> <period>" in microseconds; a 100000us (100ms) period with
// quota = cpus*period gives the same fractional-core semantics as the
// container backend's --cpus. Zero/negative inputs are reported as omit=true
// so the caller writes nothing for that controller (leaving it at the
// parent's inherited limit, mirroring "omit when unset" elsewhere).
func cgroupV2Limits(cpus float64, memoryMB, pidsLimit int) (memory, cpuMax, pids string, memOK, cpuOK, pidsOK bool) {
	if memoryMB > 0 {
		memory = strconv.FormatInt(int64(memoryMB)*1024*1024, 10)
		memOK = true
	}
	if cpus > 0 {
		const periodUS = 100000
		quota := int64(cpus * float64(periodUS))
		cpuMax = fmt.Sprintf("%d %d", quota, periodUS)
		cpuOK = true
	}
	if pidsLimit > 0 {
		pids = strconv.Itoa(pidsLimit)
		pidsOK = true
	}
	return
}

// detectCapMethod is the seam Exec uses to probe the host's enforcement
// mechanism (with the instance's cached expand-support answer); tests
// override it to force the "none" path deterministically.
var detectCapMethod = detectBwrapCapMethod

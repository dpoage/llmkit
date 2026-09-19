package sandbox

// watchdog.go holds the run-health machinery shared by every backend that
// execs a sandboxed command: the activity fingerprint (progressSnapshot), the
// idle-stall and workspace-growth-ceiling watchdog (watchIdle plus its
// definitive post-run completion check checkGrowthCeiling), the workspace-size
// fingerprint walk (workspaceProgress), and the temp-dir hygiene backstops
// (staleWorkspaceCacheAge, purgeStaleWorkspaceCaches) and random token
// (randToken) used for collision-resistant sandbox resource naming.

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// staleWorkspaceCacheAge bounds how long an orphaned workspace-cache parent
// dir (see wsCache, purgeStaleWorkspaceCaches) is allowed to linger in the OS
// temp dir before a later process reclaims it.
const staleWorkspaceCacheAge = 24 * time.Hour

// purgeStaleWorkspaceCaches best-effort removes llmkit-wscache-* directories
// in the OS temp dir older than staleWorkspaceCacheAge. It exists because not
// every backend construction site has a natural defer-Close scope (see the
// backends' Close doc comments), and any process crash bypasses a deferred
// Close regardless — so this purge, run once per backend construction, is the
// backstop that actually bounds disk usage. All failures are swallowed: this
// is hygiene, never a reason to fail sandbox construction.
func purgeStaleWorkspaceCaches() {
	matches, err := filepath.Glob(filepath.Join(os.TempDir(), "llmkit-wscache-*"))
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-staleWorkspaceCacheAge)
	for _, dir := range matches {
		info, statErr := os.Stat(dir)
		if statErr != nil || info.ModTime().After(cutoff) {
			continue
		}
		_ = os.RemoveAll(dir)
	}
}

// randToken returns a 128-bit random hex string used to give each sandbox
// resource (container name, cgroup subtree) a unique, collision-resistant
// name (so containers can be reaped by name on timeout).
func randToken() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failure is catastrophic and effectively never happens;
		// fall back to a time-based token so naming still works.
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// progressSnapshot is a language-agnostic activity fingerprint of a running
// sandbox execution. It is comparable; ANY field change between successive
// samples counts as progress and resets the idle watchdog's clock.
type progressSnapshot struct {
	outputBytes  int64 // total bytes written to stdout+stderr (incl. discarded over cap)
	fsSize       int64 // sum of regular-file sizes under the workspace
	fsCount      int64 // number of entries under the workspace
	fsMaxModNano int64 // newest mtime under the workspace, unix nanoseconds
}

// defaultWorkspaceGrowthCeilingBytes bounds cumulative workspace growth
// (the NET size delta since a run started, sampled via workspaceProgress —
// write-then-delete churn nets out and never trips it) before the shared
// idle watchdog kills the run with the distinct Result.WorkspaceQuotaExceeded
// reason, when no operator override is configured. Deliberately
// generous — this exists to catch a runaway/malicious disk-filler, not to
// constrain a legitimate build's disk usage (a full toolchain build plus
// test artifacts can easily reach several hundred MB); 2 GiB comfortably
// clears that bar while still bounding an unbounded write loop's blast
// radius well short of exhausting a typical CI/dev host's disk.
const defaultWorkspaceGrowthCeilingBytes int64 = 2 * 1024 * 1024 * 1024 // 2 GiB

// idlePollInterval derives how often the watchdog samples progress from the
// idle window: frequent enough to notice a stall promptly, but bounded so the
// workspace walk stays cheap. Clamped to [1s, 30s].
func idlePollInterval(idleTimeout time.Duration) time.Duration {
	d := idleTimeout / 4
	if d < time.Second {
		d = time.Second
	}
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	return d
}

// growthPollInterval is the sampling cadence for the workspace-growth
// ceiling, fixed and independent of idleTimeout — see
// effectivePollInterval for why it must NOT be derived from
// idlePollInterval.
//
// Deriving the growth-check cadence from idlePollInterval(idleTimeout) let a
// breach overshoot the ceiling by tens of GB under the SHIPPED DEFAULT
// (a 120s idle window -> idlePollInterval = 30s; a host writing at
// ~2 GB/s overshoots the default 2 GiB ceiling by ~4-60 GiB inside one
// window before the tick that would have caught it). 1s is a DELIBERATE, hardcoded constant —
// not idlePollInterval(0)'s [1s,30s]-clamped floor, which is an ACCIDENT of
// a disabled idle window, not a chosen cadence for this knob. It is tight
// enough to bound worst-case overshoot at multi-GB/s NVMe throughput to a
// few GiB against a multi-GiB default ceiling, and cheap enough that one
// extra workspaceProgress walk per second is negligible next to the disk
// I/O it bounds (a warm-cache walk over 50k files measured ~76ms — well
// under the 1s budget).
const growthPollInterval = 1 * time.Second

// effectivePollInterval derives watchIdle's actual sampling cadence: the
// TIGHTER of idlePollInterval(idleTimeout) (the existing idle-stall
// cadence) and growthPollInterval, whenever the growth ceiling is active —
// so growth detection is never slower than its own dedicated cadence just
// because the operator's idle-stall window happens to be long (or
// disabled). When idleTimeout <= 0, idlePollInterval(0)'s floor is an
// artifact of a DISABLED window, never a deliberate growth-sampling choice,
// so growthPollInterval alone governs in that case. When the growth
// ceiling is disabled (growthCeilingBytes <= 0), this returns EXACTLY
// idlePollInterval(idleTimeout), preserving byte-identical behavior for
// idle-only configurations (including the pre-existing 1s floor at
// idleTimeout<=0 when no ceiling is configured at all).
func effectivePollInterval(idleTimeout time.Duration, growthCeilingBytes int64) time.Duration {
	if growthCeilingBytes <= 0 {
		return idlePollInterval(idleTimeout)
	}
	if idleTimeout <= 0 {
		return growthPollInterval
	}
	if pollEvery := idlePollInterval(idleTimeout); pollEvery < growthPollInterval {
		return pollEvery
	}
	return growthPollInterval
}

// watchdogLimits bundles watchIdle's two independent kill conditions so its
// parameter list doesn't grow unbounded as more are added:
//   - idleTimeout: kill after this long with NO observable progress (the
//     original idle-stall detector). <= 0 disables it.
//   - growthCeilingBytes: kill as soon as the workspace has grown by more
//     than this many bytes since watchIdle started sampling, REGARDLESS of
//     whether the fingerprint otherwise reads as "making progress" — a
//     process that only fills disk resets the idle clock forever under the
//     plain progress definition, so this check runs
//     independently of it. <= 0 disables it.
//
// At least one must be positive for watchIdle to do anything; both may be
// active simultaneously (whichever fires first wins).
type watchdogLimits struct {
	idleTimeout        time.Duration
	growthCeilingBytes int64
}

// watchdogArgs bundles every input watchIdle needs. The shared
// growth-ceiling baseline lives here so the tick loop AND Exec's post-run
// check see the same value (see checkGrowthCeiling's doc) — bundling avoids
// the parameter list growing without bound as future kill conditions are
// added.
type watchdogArgs struct {
	done <-chan struct{}
	// fingerprint is the cheap per-tick progress/growth signal (output
	// bytes + workspace filesystem state).
	fingerprint func() progressSnapshot
	// activeFallback is consulted ONLY when fingerprint is unchanged AND
	// the growth ceiling is not implicated, so its cost (a container-CPU
	// probe) is paid just on otherwise-idle ticks. May be nil.
	activeFallback func() bool
	limits         watchdogLimits
	// base is the fingerprint taken once, before the run starts (by the
	// caller — see checkGrowthCeiling), and is the SAME baseline both the
	// tick loop's growth check and Exec's post-run growth check measure
	// against, so the two never disagree about what counts as "growth
	// since the run started".
	base      progressSnapshot
	pollEvery time.Duration
	// killed is set ONLY on a plain idle-stall kill (never on a
	// growth-ceiling kill — see quotaExceeded).
	killed *atomic.Bool
	// quotaExceeded is set on a growth-ceiling kill, whether detected here
	// (a tick observes the breach and cancels the run) or by Exec's
	// post-run checkGrowthCeiling call (the run exited on its own before
	// any tick could observe the breach).
	quotaExceeded *atomic.Bool
	cancel        func()
}

// watchIdle samples progress every a.pollEvery and cancels the run when
// either of a.limits' two independent conditions trips: no progress for
// a.limits.idleTimeout, or NET workspace growth past
// a.limits.growthCeilingBytes since a.base. a.fingerprint is the cheap
// signal; a.activeFallback is consulted ONLY when the fingerprint is
// unchanged AND the growth ceiling is not implicated, so its cost (a
// container-CPU probe) is paid just on otherwise-idle ticks. The growth
// check reuses the SAME per-tick a.fingerprint() call the idle-stall check
// already makes (both derive from one workspaceProgress walk) rather than
// sampling the filesystem twice.
//
// On a growth-ceiling breach it sets a.quotaExceeded (NEVER a.killed —
// that flag is reserved for a plain idle-stall kill, see watchdogArgs) and
// calls a.cancel; on an idle-stall timeout it sets a.killed and calls
// a.cancel. Either flag is set BEFORE cancel runs, so it is visible
// (through the atomic barrier) by the time the cancelled command returns.
// It returns when the run finishes (a.done closed) or after it fires; when
// both of a.limits' fields are <= 0 it returns immediately without
// sampling. This is the PROACTIVE half of growth-ceiling enforcement — a
// run that breaches the ceiling and exits before the next tick escapes
// this loop entirely; Exec's unconditional post-run checkGrowthCeiling
// call is what catches that case.
func watchIdle(a watchdogArgs) {
	if a.limits.idleTimeout <= 0 && a.limits.growthCeilingBytes <= 0 {
		return
	}
	last := a.base
	lastChange := time.Now()
	t := time.NewTicker(a.pollEvery)
	defer t.Stop()
	for {
		select {
		case <-a.done:
			return
		case now := <-t.C:
			cur := a.fingerprint()

			// Growth ceiling is checked FIRST and independent of the
			// idle-stall logic below: unlike output/CPU activity, ongoing
			// workspace growth must never be treated as a reason to let the
			// run continue — that is exactly the disk-filler behavior this
			// ceiling exists to catch.
			if a.limits.growthCeilingBytes > 0 && cur.fsSize-a.base.fsSize > a.limits.growthCeilingBytes {
				a.quotaExceeded.Store(true)
				a.cancel()
				return
			}

			if a.limits.idleTimeout <= 0 {
				continue
			}
			if cur != last {
				last = cur
				lastChange = now
				continue
			}
			// Cheap signals flat: consult the costlier fallback before deciding.
			if a.activeFallback != nil && a.activeFallback() {
				lastChange = now
				continue
			}
			if now.Sub(lastChange) >= a.limits.idleTimeout {
				a.killed.Store(true)
				a.cancel()
				return
			}
		}
	}
}

// checkGrowthCeiling performs the DEFINITIVE post-run workspace-growth
// check. watchIdle's tick loop only evaluates growth periodically (every
// effectivePollInterval); a run that breaches the ceiling and exits before
// the NEXT tick fires — a single large burst write, or any process fast
// enough to finish inside one poll window — would otherwise escape
// classification entirely, regardless of its own exit code. Exec calls this
// exactly once, unconditionally, immediately after the command exits
// (success, failure, or signal) and BEFORE consulting runErr —
// growth-ceiling enforcement is a measured, absolute invariant on final
// disk usage, not a race-prone liveness heuristic like idle-stall
// detection, so it does NOT participate in the "a genuine exit code wins
// over a racing watchdog" precedence rule Exec applies to TimedOut: a
// breach always overrides whatever exit code the process itself reported.
//
// fingerprint may be nil (when neither idleTimeout nor the growth ceiling
// was configured, Exec never allocates one); growthCeilingBytes<=0 is
// checked FIRST so a nil fingerprint is never dereferenced in that case.
// A no-op, without sampling the filesystem again, when a tick already
// caught the breach (quotaExceeded already true).
func checkGrowthCeiling(fingerprint func() progressSnapshot, base progressSnapshot, growthCeilingBytes int64, quotaExceeded *atomic.Bool) {
	if growthCeilingBytes <= 0 || quotaExceeded.Load() {
		return
	}
	if final := fingerprint(); final.fsSize-base.fsSize > growthCeilingBytes {
		quotaExceeded.Store(true)
	}
}

// workspaceProgress returns an aggregate fingerprint of dir: total regular-file
// byte count, entry count, and newest mtime (unix nanos). It is best-effort
// (unreadable entries are skipped) and walks the whole tree so it captures
// activity anywhere a build/test writes — caches, compiled artifacts, generated
// files — independent of language. Cost is bounded by the poll interval; the
// workspace is a single repo copy.
func workspaceProgress(dir string) (size, count, maxModNano int64) {
	_ = filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		count++
		if d.Type().IsRegular() {
			size += info.Size()
		}
		if m := info.ModTime().UnixNano(); m > maxModNano {
			maxModNano = m
		}
		return nil
	})
	return
}

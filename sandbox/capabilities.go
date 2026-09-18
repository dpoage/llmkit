package sandbox

// capabilities.go implements parameterized sandbox capability probing.
//
// A caller that plans work inside a sandbox without knowing what that
// sandbox can actually run supplies its own probe table — a []ProbeEntry
// naming each capability of interest, the argv that measures it, and an
// Interpret function mapping the probe result to named modes. This file
// executes those probes inside the sandbox and caches the results. The kit
// itself ships no probe entries: WHAT to probe is entirely the caller's
// knowledge, and no ecosystem or toolchain vocabulary appears here.
//
// # Caching
//
// ProbeCapabilities caches per process, keyed on the image, the probe-set
// identity, and the mounts/env combination (see ProbeCapabilities for the
// exact key composition and why each part is there). A sync.Map carries the
// cache, and concurrent callers that compute the same key share a single
// probe run instead of racing duplicate ones. A failed probe records the
// best-effort unavailable result, so repeated calls for a broken setup
// never re-execute it.

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"
)

// ProbeResult is the slice of a sandbox Result an Interpret function may
// look at: the probe's exit code and captured stdout. ProbeCapabilities
// fills it from the Result the backend returned.
type ProbeResult struct {
	ExitCode int
	Stdout   string
}

// ProbeEntry describes one capability probe.
type ProbeEntry struct {
	// Name is the caller's key for this entry in the returned CapabilitySet.
	Name string
	// Probe is the argv run inside the sandbox under the backend's default
	// (no) network. Best-effort: an Exec error or timeout yields an
	// all-false result for this entry.
	Probe []string
	// Interpret maps the probe's ProbeResult to a map[mode]available. It is
	// called even on non-zero exit codes so it can decide per mode. A nil
	// return is treated as all-false.
	Interpret func(ProbeResult) map[string]bool
}

// CapabilitySet maps probe name → mode name → available. It is returned by
// ProbeCapabilities for the caller to gate work on.
//
// A missing key is equivalent to false (unavailable). CapabilitySet is
// read-only after construction; it is safe to share across goroutines.
type CapabilitySet map[string]map[string]bool

// Available reports whether the named mode is available under the named
// probe. A missing probe or mode returns false (unavailable).
func (cs CapabilitySet) Available(name, mode string) bool {
	if cs == nil {
		return false
	}
	return cs[name][mode]
}

// probeTimeout is the per-probe wall-clock ceiling. Probes are meant to be
// cheap; 30 s is generous.
const probeTimeout = 30 * time.Second

// capCache is the global probe cache keyed by the composite string built in
// ProbeCapabilities. Values are *capFlight.
var capCache sync.Map

// capFlight is one in-flight-or-complete probe computation for a cache key.
// Concurrent callers that compute the same key share a single flight: the
// first to claim the key runs the probes and publishes the CapabilitySet on
// done; the rest block on done and read the published set. That is what
// makes N concurrent ProbeCapabilities calls for the same inputs execute the
// probe argv once, not N times. On a panic inside the probe, done is still
// closed and waiters receive a nil CapabilitySet (all-unavailable); the
// panic itself propagates in the caller that ran the probe.
type capFlight struct {
	done chan struct{}
	cs   CapabilitySet
}

// wait blocks until the flight's probe completes and returns its result.
func (f *capFlight) wait() CapabilitySet {
	<-f.done
	return f.cs
}

// ProbeCapabilities runs the caller's probes against image once per process
// per distinct cache key and returns a CapabilitySet keyed by each entry's
// Name. The probe is best-effort: an Exec error or timeout marks all modes
// of that entry unavailable, but the call never returns an error. The result
// is cached, so subsequent calls with the same key are free.
//
// The cache key is image + "|" + probe-set identity + "|" +
// mountsEnvCacheKey(mounts, rwMounts, env). Every part is load-bearing:
//
//   - The probe-set identity (each entry's Name, in order) is in the key
//     because the shape of a CapabilitySet is exactly the shape the caller's
//     probe list declares. Two callers probing the same image with different
//     probe lists must never share an entry: a shared entry would hand one
//     caller a result shaped by the other's table, and a mode absent from
//     that table reads as "unavailable". Names are joined in order rather
//     than sorted: a reordered list re-probes (merely redundant) where a
//     shared entry could return wrong-shaped data.
//   - mountsEnvCacheKey makes the key order-independent in the mounts/env
//     the caller threads through (see its doc), so assembly order never
//     splits one logical configuration into two cache entries.
//
// Concurrent calls that compute the same key share one probe run (see
// capFlight): the first caller to claim the key executes it, and the rest
// block until that shared run completes — under the FIRST claimant's ctx,
// bounded by the probeTimeout ceiling — and its result is what gets cached.
// If the probe panics inside a caller-supplied Interpret, the panic
// propagates in the claimant while waiters receive the flight's nil
// (all-unavailable) set, which then stays cached until
// InvalidateCapabilityCache clears it.
//
// image is the container image Spec.Image on the CLI backend, so it must be
// a real, pullable image there; on the Bwrap and HostExec backends there is
// no image concept and it is only a cache-key label (Spec.Image is
// ignored). A made-up label on the CLI backend makes every probe Exec fail
// at the runtime, yielding all-false.
//
// repoDir gives the probe's Spec a valid RepoDir; it is only read (copied
// into the probe's fresh workspace, never written). An empty repoDir — or a
// nil sb — short-circuits: every entry is returned all-false and nothing is
// executed.
//
// mounts and env are attached to every probe's Spec so whatever the caller
// makes visible to a real run (a host toolchain or cache mounted read-only,
// a PATH adjustment) is visible to the probe exactly the same way — this is
// what makes a mounted tool show up as available in the returned set. Pass
// nil for both when there is nothing to thread through.
//
// rwMounts is the writable counterpart: an entry the caller mounts writable
// is threaded into the probe's Spec.RWMounts too, so a probe command that
// itself needs to write to a mounted directory (mirroring what the real run
// requires) must not read as unavailable purely because the probe
// under-provisioned the mount. Pass nil when there is nothing writable to
// thread through.
//
// The probe runs under the backend's default network mode ("none") with a
// probeTimeout ceiling so it cannot stall the caller.
func ProbeCapabilities(ctx context.Context, sb Sandbox, image, repoDir string, mounts, rwMounts []ROMount, env []string, probes []ProbeEntry) CapabilitySet {
	if sb == nil || repoDir == "" {
		// No sandbox or nothing to copy a workspace from — return an
		// all-false set for every entry without executing anything.
		cs := make(CapabilitySet, len(probes))
		for _, e := range probes {
			cs[e.Name] = allFalse(e)
		}
		return cs
	}

	key := image + "|" + probeSetKey(probes) + "|" + mountsEnvCacheKey(mounts, rwMounts, env)
	mine := &capFlight{done: make(chan struct{})}
	loaded, _ := capCache.LoadOrStore(key, mine)
	if loaded == mine {
		// This caller claimed the key: run the probes once, publish the
		// result, then release the waiters. close runs even on panic so a
		// panicking Interpret cannot block every waiter forever (see
		// capFlight). The winner returns its own result directly — waiting
		// on its own done channel would deadlock it.
		defer close(mine.done)
		mine.cs = runProbes(ctx, sb, image, repoDir, mounts, rwMounts, env, probes)
		return mine.cs
	}
	return loaded.(*capFlight).wait()
}

// probeSetKey identifies a probe list by its entry Names IN ORDER — the
// probe-set identity fragment of the cache key (see ProbeCapabilities for
// why it must be in the key, and why order is preserved rather than sorted).
func probeSetKey(probes []ProbeEntry) string {
	names := make([]string, len(probes))
	for i, e := range probes {
		names[i] = e.Name
	}
	return strings.Join(names, ",")
}

// mountsEnvCacheKey builds a deterministic cache key fragment from mounts,
// rwMounts, and env so ProbeCapabilities never returns a stale result for a
// different set of mounts or environment: the same probe argv can observe
// different inputs — a mounted tool, a writable-vs-read-only mount, a PATH
// change — and reach a different answer. Order-independent: mounts,
// rwMounts, and env are sorted before joining, since callers may assemble
// them from map iteration or independent resolution steps in varying order.
// rw entries carry a "rw:" prefix (distinct from ro's "m:") so a mount that
// is read-only under one configuration and writable under another never
// collides on the same key.
func mountsEnvCacheKey(mounts, rwMounts []ROMount, env []string) string {
	parts := make([]string, 0, len(mounts)+len(rwMounts)+len(env))
	for _, m := range mounts {
		parts = append(parts, "m:"+m.HostPath+"->"+m.ContainerPath)
	}
	for _, m := range rwMounts {
		parts = append(parts, "rw:"+m.HostPath+"->"+m.ContainerPath)
	}
	parts = append(parts, env...)
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

// runProbes executes every entry's probe and assembles the CapabilitySet.
// mounts, rwMounts, and env are attached to every probe's Spec (see
// ProbeCapabilities).
func runProbes(ctx context.Context, sb Sandbox, image, repoDir string, mounts, rwMounts []ROMount, env []string, probes []ProbeEntry) CapabilitySet {
	cs := make(CapabilitySet, len(probes))
	for _, e := range probes {
		spec := Spec{
			RepoDir:  repoDir,
			Cmd:      e.Probe,
			Image:    image,
			Timeout:  probeTimeout,
			ROMounts: mounts,
			RWMounts: rwMounts,
			Env:      env,
			// Network defaults to "none" in every backend; no override needed.
		}
		result, err := sb.Exec(ctx, spec)
		if err != nil {
			// Infrastructure failure → all modes unavailable for this entry.
			cs[e.Name] = allFalse(e)
			continue
		}
		modes := e.Interpret(ProbeResult{ExitCode: result.ExitCode, Stdout: result.Stdout})
		if modes == nil {
			modes = allFalse(e)
		}
		cs[e.Name] = modes
	}
	return cs
}

// allFalse returns a modes map with every mode for e set to false. Used when
// a probe fails at the infrastructure level. The false map is built by
// running Interpret on a synthetic non-zero-exit result so the mode set
// stays consistent with the entry's own declaration; an Interpret that
// returns nil there yields an empty map.
func allFalse(e ProbeEntry) map[string]bool {
	modes := e.Interpret(ProbeResult{ExitCode: 1})
	if modes == nil {
		return map[string]bool{}
	}
	for k := range modes {
		modes[k] = false
	}
	return modes
}

// InvalidateCapabilityCache removes every cached result for image, forcing
// the next ProbeCapabilities call to re-probe. Intended for tests that need
// a clean slate.
//
// The cache key is composed as image+"|"+probeSetKey+"|"+mountsEnvCacheKey
// (see ProbeCapabilities), so a plain capCache.Delete(image) would silently
// no-op against every real entry — this walks the cache and deletes every
// entry whose key starts with image+"|", regardless of which probe set or
// mounts/env combination produced it.
func InvalidateCapabilityCache(image string) {
	prefix := image + "|"
	capCache.Range(func(k, _ any) bool {
		if key, ok := k.(string); ok && strings.HasPrefix(key, prefix) {
			capCache.Delete(k)
		}
		return true
	})
}

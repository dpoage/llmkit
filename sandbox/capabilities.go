package sandbox

// capabilities.go implements parameterized sandbox capability probing.
//
// A caller that plans work inside a sandbox without knowing what that
// sandbox can actually run supplies its own probe table — a []ProbeEntry
// naming each capability of interest, the argv that measures it, and an
// Interpret function mapping the probe result to named modes. Probe
// executes those probes inside the sandbox and returns a CapabilitySet.
// The kit itself ships no probe entries: WHAT to probe is entirely the
// caller's knowledge, and no ecosystem or toolchain vocabulary appears
// here.

import (
	"context"
	"errors"
	"time"
)

// ProbeResult is the slice of a sandbox Result an Interpret function may
// look at: the probe's exit code and captured stdout. Probe fills it from
// the Result the backend returned.
type ProbeResult struct {
	ExitCode int
	Stdout   string
}

// ProbeEntry describes one capability probe.
type ProbeEntry struct {
	// Name is the caller's key for this entry in the returned CapabilitySet.
	Name string
	// Probe is the argv run inside the sandbox under the base Spec's
	// network posture. The Probe function's doc states what Probe does
	// with each Exec outcome, a timeout included.
	Probe []string
	// Interpret maps the probe's ProbeResult to a map[mode]available. It is
	// called even on non-zero exit codes so it can decide per mode. A nil
	// return is treated as all-false.
	Interpret func(ProbeResult) map[string]bool
}

// CapabilitySet maps probe name → mode name → available. It is returned by
// Probe for the caller to gate work on.
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

// probeTimeout is the per-probe Timeout that Probe sets when base.Timeout
// is <= 0 (see Probe).
const probeTimeout = 30 * time.Second

// Probe runs each of probes against sb, in order, using base as the
// template Spec for every probe's Exec call, and returns a CapabilitySet
// keyed by each entry's Name.
//
// For every entry, Probe copies base, sets Cmd to entry.Probe, and sets
// Timeout. If base.Timeout is <= 0, Timeout is 30 seconds; otherwise
// base.Timeout is kept unchanged. Every other base field — RepoDir,
// Workspace, Image, Env, Network, ROMounts, RWMounts, SetupCmds,
// WriteFiles, CaptureFiles — reaches sb.Exec exactly as given: Probe does
// no backend sniffing and strips nothing. In particular base.Image must be
// empty on backends that have no image concept (Bwrap, HostExec) — sb.Exec
// refuses a non-empty Image there with UnsupportedSpecError, exactly like
// any other Exec call.
//
// sb must be non-nil: a nil Sandbox interface value is an error. Probe does
// not detect a typed nil pointer, and the outcome depends on the backend:
// if the Spec passes the backend's validation, sb.Exec panics on a nil
// *CLI, *Bwrap, or *Mock.
//
// Probe handles the outcome of each Exec call in one of three ways:
//
//   - Exec returns a Result and no error. Probe passes the Result's
//     ExitCode and Stdout to entry.Interpret, whatever the exit code.
//     This is the single invariant Interpret can rely on: every
//     non-error Exec result reaches Interpret with the backend's
//     reported ExitCode and Stdout, including a Timeout kill where
//     ExitCode is -1 and Stdout is whatever the backend captured.
//     Backend-specific timeout and kill semantics — what is killed,
//     how long Exec waits — are each backend's own contract, not
//     Probe's. If Interpret returns nil, Probe records the entry as in
//     the last case.
//   - Exec returns a refusal: an error that errors.As matches
//     *InvalidSpecError or *UnsupportedSpecError. Probe returns (nil, err)
//     at once, and the entries after the refused one do not run. A refusal
//     caused by base repeats for every entry. A refusal caused by one entry
//     (an empty Probe argv gives InvalidSpecError{Field: "Cmd"}) is a bug
//     in the probe table.
//   - Exec returns any other error: an infrastructure failure, or the
//     caller's ctx ending (cancelled or past its deadline). Probe records
//     every mode of that entry as false and continues with the remaining
//     entries.
//
// A panic inside a caller-supplied Interpret is a caller bug and propagates
// to Probe's caller.
//
// Probe holds no cache, no singleflight, and no process-global state: every
// call re-runs every entry's probe. Callers that probe the same sandbox
// repeatedly own their own memoization.
func Probe(ctx context.Context, sb Sandbox, base Spec, probes []ProbeEntry) (CapabilitySet, error) {
	if sb == nil {
		return nil, errors.New("sandbox: Probe requires a non-nil Sandbox")
	}

	cs := make(CapabilitySet, len(probes))
	for _, e := range probes {
		spec := base
		spec.Cmd = e.Probe
		if base.Timeout <= 0 {
			spec.Timeout = probeTimeout
		}

		result, err := sb.Exec(ctx, spec)
		if err != nil {
			var invalid *InvalidSpecError
			var unsupported *UnsupportedSpecError
			if errors.As(err, &invalid) || errors.As(err, &unsupported) {
				// Refusal: see Probe.
				return nil, err
			}
			// Any other Exec error: best-effort, see Probe.
			cs[e.Name] = allFalse(e)
			continue
		}

		modes := e.Interpret(ProbeResult{ExitCode: result.ExitCode, Stdout: result.Stdout})
		if modes == nil {
			modes = allFalse(e)
		}
		cs[e.Name] = modes
	}
	return cs, nil
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

package sandbox

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// Kit-neutral probe fixtures. Each entry follows the same token protocol the
// probe seam is built for: the probe argv emits one token per available mode
// on stdout, and Interpret maps tokens to modes. The argv is a real shell
// script so the integration-tagged tests can execute it through the Bwrap
// and CLI backends; the Mock tests only ever see scripted stdout.

// alphaProbe declares two modes ("one", "two") over two tokens.
var alphaProbe = ProbeEntry{
	Name:  "alpha",
	Probe: []string{"/bin/sh", "-c", "echo tok1; echo tok2"},
	Interpret: func(r ProbeResult) map[string]bool {
		modes := map[string]bool{"one": false, "two": false}
		for _, line := range strings.Split(r.Stdout, "\n") {
			switch strings.TrimSpace(line) {
			case "tok1":
				modes["one"] = true
			case "tok2":
				modes["two"] = true
			}
		}
		return modes
	},
}

// betaProbe declares one mode ("three") over one token.
var betaProbe = ProbeEntry{
	Name:  "beta",
	Probe: []string{"/bin/sh", "-c", "echo tok3"},
	Interpret: func(r ProbeResult) map[string]bool {
		modes := map[string]bool{"three": false}
		for _, line := range strings.Split(r.Stdout, "\n") {
			if strings.TrimSpace(line) == "tok3" {
				modes["three"] = true
			}
		}
		return modes
	},
}

// testProbes is the two-entry list the multi-entry tests run with.
var testProbes = []ProbeEntry{alphaProbe, betaProbe}

// nilInterpretProbe's Interpret deliberately returns nil when the probe
// emitted nothing — the shape Probe must treat as all-false rather than
// storing a nil modes map in the set.
var nilInterpretProbe = ProbeEntry{
	Name:  "nilinterp",
	Probe: []string{"/bin/sh", "-c", "exit 0"},
	Interpret: func(r ProbeResult) map[string]bool {
		if r.Stdout == "" {
			return nil
		}
		return map[string]bool{"ghost": true}
	},
}

// errProbeSim is a sentinel error for probe-failure tests: an ordinary
// infrastructure failure, never a refusal.
var errProbeSim = probeSimErr("simulated exec error")

type probeSimErr string

func (e probeSimErr) Error() string { return string(e) }

// TestCapabilitySetAvailable tests the CapabilitySet.Available helper.
func TestCapabilitySetAvailable(t *testing.T) {
	cs := CapabilitySet{
		"alpha": {"one": true},
	}
	if !cs.Available("alpha", "one") {
		t.Error("Available(alpha, one) = false, want true")
	}
	if cs.Available("alpha", "missing") {
		t.Error("Available(alpha, missing) = true, want false")
	}
	if cs.Available("missing", "one") {
		t.Error("Available(missing, one) = true, want false")
	}
	var nilCS CapabilitySet
	if nilCS.Available("alpha", "one") {
		t.Error("nil.Available = true, want false")
	}
}

// TestProbe_NilSandboxIsError pins the nil-Sandbox contract: Probe never
// dereferences a nil sb (which would panic inside sb.Exec) and instead
// reports it as an ordinary error, with a nil CapabilitySet.
func TestProbe_NilSandboxIsError(t *testing.T) {
	cs, err := Probe(context.Background(), nil, Spec{RepoDir: t.TempDir()}, testProbes)
	if err == nil {
		t.Fatal("Probe(nil sb) returned a nil error, want non-nil")
	}
	if cs != nil {
		t.Errorf("Probe(nil sb) CapabilitySet = %v, want nil", cs)
	}
}

// TestProbe_RefusalsSurface pins S2-1: a refusal (InvalidSpecError or
// UnsupportedSpecError) from sb.Exec is returned as Probe's own error
// immediately — no further entries run — while any other Exec error keeps
// the best-effort all-false behavior and probing continues through every
// entry.
//
// Mutant: delete the errors.As refusal branch (fold refusals into the
// best-effort all-false path like any other Exec error) → every subtest
// here goes red: the first two expect (nil, non-nil-err) and CallCount==1,
// which a fully-best-effort Probe would never produce.
func TestProbe_RefusalsSurface(t *testing.T) {
	base := Spec{RepoDir: t.TempDir()}

	t.Run("unsupported_spec_error_returned", func(t *testing.T) {
		refusal := &UnsupportedSpecError{Backend: backendMock, Field: "Image", Value: "x"}
		mock := NewMock(MockResponse{Err: refusal})
		cs, err := Probe(context.Background(), mock, base, testProbes)
		if cs != nil {
			t.Errorf("CapabilitySet = %v, want nil on a refusal", cs)
		}
		var got *UnsupportedSpecError
		if !errors.As(err, &got) {
			t.Fatalf("err = %v, want errors.As match for *UnsupportedSpecError", err)
		}
		if got != refusal {
			t.Errorf("errors.As unwrapped %v, want the exact scripted refusal %v", got, refusal)
		}
		if n := mock.CallCount(); n != 1 {
			t.Errorf("Exec ran %d times, want 1 (refusal stops probing the remaining entries)", n)
		}
	})

	t.Run("invalid_spec_error_returned", func(t *testing.T) {
		refusal := &InvalidSpecError{Field: "Cmd", Reason: "x"}
		mock := NewMock(MockResponse{Err: refusal})
		cs, err := Probe(context.Background(), mock, base, testProbes)
		if cs != nil {
			t.Errorf("CapabilitySet = %v, want nil on a refusal", cs)
		}
		var got *InvalidSpecError
		if !errors.As(err, &got) {
			t.Fatalf("err = %v, want errors.As match for *InvalidSpecError", err)
		}
		if got != refusal {
			t.Errorf("errors.As unwrapped %v, want the exact scripted refusal %v", got, refusal)
		}
		if n := mock.CallCount(); n != 1 {
			t.Errorf("Exec ran %d times, want 1 (refusal stops probing the remaining entries)", n)
		}
	})

	t.Run("plain_error_is_all_false_and_probing_continues", func(t *testing.T) {
		mock := NewMock(MockResponse{Err: errProbeSim})
		cs, err := Probe(context.Background(), mock, base, testProbes)
		if err != nil {
			t.Fatalf("err = %v, want nil (a plain Exec error is best-effort, not a refusal)", err)
		}
		for _, e := range testProbes {
			modes, ok := cs[e.Name]
			if !ok {
				t.Errorf("entry %q missing from the returned set: %v", e.Name, cs)
				continue
			}
			for mode, avail := range modes {
				if avail {
					t.Errorf("entry %q mode %q = true after an Exec error, want false", e.Name, mode)
				}
			}
		}
		if n := mock.CallCount(); n != len(testProbes) {
			t.Errorf("Exec ran %d times, want %d (a non-refusal error must not stop later entries)", n, len(testProbes))
		}
	})
}

// TestProbe_TokenDrivenModes exercises the Interpret/allFalse/best-effort
// contract that predates this round: token-driven mode availability,
// partial availability, non-zero exits, and a nil Interpret result.
func TestProbe_TokenDrivenModes(t *testing.T) {
	t.Run("both_modes_available_when_both_tokens_emitted", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\n"}})
		cs, err := Probe(context.Background(), mock, Spec{RepoDir: t.TempDir()}, testProbes)
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if !cs.Available("alpha", "one") {
			t.Errorf("want alpha/one available, got %v", cs)
		}
		if !cs.Available("alpha", "two") {
			t.Errorf("want alpha/two available, got %v", cs)
		}
		if cs.Available("beta", "three") {
			t.Errorf("want beta/three unavailable (the scripted stdout carries no tok3), got %v", cs)
		}
	})

	t.Run("one_mode_when_only_first_token", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\n"}})
		cs, err := Probe(context.Background(), mock, Spec{RepoDir: t.TempDir()}, testProbes)
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if !cs.Available("alpha", "one") {
			t.Errorf("want alpha/one available, got %v", cs)
		}
		if cs.Available("alpha", "two") {
			t.Errorf("want alpha/two unavailable when only tok1 is emitted, got %v", cs)
		}
	})

	t.Run("all_unavailable_on_nonzero_exit_no_tokens", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 127, Stdout: ""}})
		cs, err := Probe(context.Background(), mock, Spec{RepoDir: t.TempDir()}, testProbes)
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if cs.Available("alpha", "one") || cs.Available("alpha", "two") || cs.Available("beta", "three") {
			t.Errorf("want every mode unavailable on non-zero exit with no tokens, got %v", cs)
		}
		for _, e := range testProbes {
			if _, ok := cs[e.Name]; !ok {
				t.Errorf("entry %q missing from the returned set: %v", e.Name, cs)
			}
		}
	})

	t.Run("sandbox_kill_reaches_interpret", func(t *testing.T) {
		// A timed-out probe is a Result, not an Exec error: its partial stdout reaches Interpret.
		mock := NewMock(MockResponse{Result: Result{ExitCode: -1, TimedOut: true, Stdout: "tok1\n"}})
		cs, err := Probe(context.Background(), mock, Spec{RepoDir: t.TempDir()}, testProbes)
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		if !cs.Available("alpha", "one") {
			t.Errorf("want alpha/one available from the killed probe's partial stdout, got %v", cs)
		}
	})

	t.Run("nil_interpret_result_is_all_false", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: ""}})
		cs, err := Probe(context.Background(), mock, Spec{RepoDir: t.TempDir()}, []ProbeEntry{nilInterpretProbe})
		if err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		modes, ok := cs["nilinterp"]
		if !ok {
			t.Fatalf("entry nilinterp missing from the returned set: %v", cs)
		}
		if len(modes) != 0 {
			t.Errorf("nil Interpret result must yield an empty (all-false) modes map, got %v", modes)
		}
		if cs.Available("nilinterp", "ghost") {
			t.Error("Available(nilinterp, ghost) = true, want false")
		}
	})
}

// TestProbe_NoCaching pins S2-2: Probe holds no state across calls. Two
// calls with equal inputs on one Mock must each execute every entry — the
// second call is not served from any memoized result — and both must
// return an equal CapabilitySet since nothing about the scripted response
// changed between calls.
//
// Mutant: memoize on the base Spec (or the probe names) inside Probe →
// the second call's CallCount would stay len(testProbes) instead of
// doubling, and this test goes red.
func TestProbe_NoCaching(t *testing.T) {
	mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
	base := Spec{RepoDir: t.TempDir()}

	cs1, err := Probe(context.Background(), mock, base, testProbes)
	if err != nil {
		t.Fatalf("first Probe error: %v", err)
	}
	cs2, err := Probe(context.Background(), mock, base, testProbes)
	if err != nil {
		t.Fatalf("second Probe error: %v", err)
	}

	if n := mock.CallCount(); n != 2*len(testProbes) {
		t.Errorf("Exec ran %d times across two identical calls, want %d (2x len(testProbes) — no memoization)", n, 2*len(testProbes))
	}
	if !reflect.DeepEqual(cs1, cs2) {
		t.Errorf("two calls with equal inputs and an unchanged scripted response returned different sets: cs1=%v cs2=%v", cs1, cs2)
	}
}

// TestProbe_NoBackendSniffing pins S2-3: Probe never inspects sb's concrete
// type. base.Image (and every other base field) reaches sb.Exec verbatim
// regardless of what sb is or how it is wrapped.
func TestProbe_NoBackendSniffing(t *testing.T) {
	t.Run("image_reaches_exec_unchanged", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		base := Spec{RepoDir: t.TempDir(), Image: "img:tag"}
		if _, err := Probe(context.Background(), mock, base, testProbes); err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		for _, c := range mock.Calls() {
			if c.Spec.Image != "img:tag" {
				t.Errorf("recorded Spec.Image = %q, want the caller's %q unchanged", c.Spec.Image, "img:tag")
			}
		}
	})

	t.Run("observe_wrapped_sees_the_same_specs_as_bare", func(t *testing.T) {
		mockDirect := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		mockWrapped := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		obs := llmkit.ObserverFunc(func(context.Context, llmkit.Event) {})
		wrapped := Observe(mockWrapped, obs)

		base := Spec{
			RepoDir: t.TempDir(),
			Image:   "img:tag",
			Env:     []string{"A=1"},
		}
		if _, err := Probe(context.Background(), mockDirect, base, testProbes); err != nil {
			t.Fatalf("Probe(direct) error: %v", err)
		}
		if _, err := Probe(context.Background(), wrapped, base, testProbes); err != nil {
			t.Fatalf("Probe(wrapped) error: %v", err)
		}

		direct := mockDirect.Calls()
		via := mockWrapped.Calls()
		if len(direct) != len(via) {
			t.Fatalf("recorded %d calls direct, %d via Observe, want equal", len(direct), len(via))
		}
		for i := range direct {
			if !reflect.DeepEqual(direct[i].Spec, via[i].Spec) {
				t.Errorf("call %d: direct Spec = %+v, via Observe = %+v, want identical", i, direct[i].Spec, via[i].Spec)
			}
		}
	})
}

// TestProbe_PerEntrySpecShape pins S2-4: every field of base reaches
// sb.Exec unchanged except Cmd (set to the entry's Probe) and Timeout (set
// to probeTimeout only when base.Timeout is <= 0). A panicking Interpret
// propagates to Probe's own caller.
//
// Mutant: build the per-entry Spec field by field instead of copying base
// wholesale (e.g. omit ROMounts) → the equality check against "base with
// only Cmd/Timeout changed" goes red.
func TestProbe_PerEntrySpecShape(t *testing.T) {
	t.Run("every_field_but_cmd_and_timeout_is_preserved", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		base := Spec{
			RepoDir:   t.TempDir(),
			Workspace: "",
			Image:     "img:tag",
			Env:       []string{"A=1", "B=2"},
			Network:   NetworkNone,
			ROMounts: []ROMount{
				{HostPath: "/host/ro", ContainerPath: "/opt/ro"},
			},
			RWMounts: []ROMount{
				{HostPath: "/host/rw", ContainerPath: "/opt/rw"},
			},
			SetupCmds:    [][]string{{"true"}},
			CaptureFiles: []string{"out.txt"},
			Timeout:      0,
		}
		if _, err := Probe(context.Background(), mock, base, testProbes); err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		calls := mock.Calls()
		if len(calls) != len(testProbes) {
			t.Fatalf("recorded %d calls, want %d", len(calls), len(testProbes))
		}
		for i, e := range testProbes {
			want := base
			want.Cmd = e.Probe
			want.Timeout = probeTimeout
			if !reflect.DeepEqual(calls[i].Spec, want) {
				t.Errorf("entry %q: recorded Spec = %+v, want %+v (base unchanged except Cmd/Timeout)", e.Name, calls[i].Spec, want)
			}
		}
	})

	t.Run("nonzero_base_timeout_is_kept", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		base := Spec{RepoDir: t.TempDir(), Timeout: 5 * time.Second}
		if _, err := Probe(context.Background(), mock, base, testProbes); err != nil {
			t.Fatalf("Probe error: %v", err)
		}
		for _, c := range mock.Calls() {
			if c.Spec.Timeout != 5*time.Second {
				t.Errorf("recorded Spec.Timeout = %v, want the caller's 5s kept unchanged", c.Spec.Timeout)
			}
		}
	})

	t.Run("panicking_interpret_propagates", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "irrelevant\n"}})
		panicking := ProbeEntry{
			Name:  "boom",
			Probe: []string{"echo", "irrelevant"},
			Interpret: func(ProbeResult) map[string]bool {
				panic("buggy Interpret")
			},
		}
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected the panicking Interpret's panic to propagate to Probe's caller")
				}
				if r != "buggy Interpret" {
					t.Errorf("recovered panic value = %v, want %q", r, "buggy Interpret")
				}
			}()
			_, _ = Probe(context.Background(), mock, Spec{RepoDir: t.TempDir()}, []ProbeEntry{panicking})
		}()
	})
}

package sandbox

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"
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
// emitted nothing — the shape runProbes must treat as all-false rather than
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

// errProbeSim is a sentinel error for probe-failure tests.
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

// TestProbeCapabilitiesMock tests ProbeCapabilities against NewMock,
// covering: token-driven mode availability, partial availability, non-zero
// exits, Exec errors, nil Interpret results, cache hits, and the
// sb==nil / repoDir=="" short-circuits (which must not execute anything).
func TestProbeCapabilitiesMock(t *testing.T) {
	t.Run("both_modes_available_when_both_tokens_emitted", func(t *testing.T) {
		const image = "test-image-both-tokens"
		InvalidateCapabilityCache(image)
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\n"}})
		cs := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
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
		const image = "test-image-one-token"
		InvalidateCapabilityCache(image)
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\n"}})
		cs := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
		if !cs.Available("alpha", "one") {
			t.Errorf("want alpha/one available, got %v", cs)
		}
		if cs.Available("alpha", "two") {
			t.Errorf("want alpha/two unavailable when only tok1 is emitted, got %v", cs)
		}
	})

	t.Run("all_unavailable_on_nonzero_exit_no_tokens", func(t *testing.T) {
		const image = "test-image-nonzero"
		InvalidateCapabilityCache(image)
		mock := NewMock(MockResponse{Result: Result{ExitCode: 127, Stdout: ""}})
		cs := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
		if cs.Available("alpha", "one") || cs.Available("alpha", "two") {
			t.Errorf("want alpha modes unavailable on non-zero exit with no tokens, got %v", cs)
		}
		if cs.Available("beta", "three") {
			t.Errorf("want beta/three unavailable on non-zero exit with no tokens, got %v", cs)
		}
		// Every declared entry must still be present (all-false), so callers
		// can enumerate modes without a nil/missing-key dance.
		for _, e := range testProbes {
			if _, ok := cs[e.Name]; !ok {
				t.Errorf("entry %q missing from the returned set: %v", e.Name, cs)
			}
		}
	})

	t.Run("all_unavailable_on_exec_error", func(t *testing.T) {
		const image = "test-image-exec-error"
		InvalidateCapabilityCache(image)
		mock := NewMock(MockResponse{Err: errProbeSim})
		cs := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
		if cs.Available("alpha", "one") || cs.Available("alpha", "two") {
			t.Errorf("want alpha modes unavailable on exec error, got %v", cs)
		}
		if cs.Available("beta", "three") {
			t.Errorf("want beta/three unavailable on exec error, got %v", cs)
		}
		for _, e := range testProbes {
			if _, ok := cs[e.Name]; !ok {
				t.Errorf("entry %q missing from the returned set after exec error: %v", e.Name, cs)
			}
		}
	})

	t.Run("nil_interpret_result_is_all_false", func(t *testing.T) {
		const image = "test-image-nil-interpret"
		InvalidateCapabilityCache(image)
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: ""}})
		cs := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, []ProbeEntry{nilInterpretProbe})
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

	t.Run("cache_hit_does_not_reexec", func(t *testing.T) {
		const image = "test-image-cache-hit"
		InvalidateCapabilityCache(image)
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		cs1 := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
		cs2 := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
		if cs1["alpha"]["one"] != cs2["alpha"]["one"] || cs1["beta"]["three"] != cs2["beta"]["three"] {
			t.Errorf("cache hit must return the same result: cs1=%v cs2=%v", cs1, cs2)
		}
		if n := mock.CallCount(); n != len(testProbes) {
			t.Errorf("second call must be served from cache: Exec ran %d times, want %d", n, len(testProbes))
		}
	})

	t.Run("nil_sandbox_returns_unavailable", func(t *testing.T) {
		// A real repoDir here: this subtest pins the sb==nil clause alone
		// (the empty-repoDir clause has its own subtest below).
		cs := ProbeCapabilities(context.Background(), nil, "any", t.TempDir(), nil, nil, nil, testProbes)
		for _, e := range testProbes {
			if _, ok := cs[e.Name]; !ok {
				t.Errorf("entry %q missing from the returned set: %v", e.Name, cs)
			}
			for mode := range cs[e.Name] {
				if cs[e.Name][mode] {
					t.Errorf("nil sandbox must yield all-false, got %s/%s=true (%v)", e.Name, mode, cs)
				}
			}
		}
	})

	t.Run("empty_repoDir_returns_unavailable_without_exec", func(t *testing.T) {
		mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
		cs := ProbeCapabilities(context.Background(), mock, "any", "", nil, nil, nil, testProbes)
		if cs.Available("alpha", "one") || cs.Available("beta", "three") {
			t.Errorf("empty repoDir must yield all-false, got %v", cs)
		}
		if n := mock.CallCount(); n != 0 {
			t.Errorf("empty repoDir must short-circuit without probing: Exec ran %d times, want 0", n)
		}
	})
}

// TestProbeCapabilities_MountsThreadedIntoEveryProbe verifies that mounts
// and env are attached to every probe's Spec, so whatever the caller makes
// visible to a real run is visible to the probe the same way.
func TestProbeCapabilities_MountsThreadedIntoEveryProbe(t *testing.T) {
	const image = "test-image-toolchain-mount"
	InvalidateCapabilityCache(image)
	mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\n"}})
	mounts := []ROMount{{HostPath: "/host/tools", ContainerPath: "/opt/kit/tools", Shared: true}}
	env := []string{"PATH=/opt/kit/tools/bin:/usr/bin"}

	cs := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), mounts, nil, env, testProbes)
	if !cs.Available("alpha", "one") {
		t.Errorf("want alpha/one available once the mocked probe reports it, got %v", cs)
	}

	if n := mock.CallCount(); n != len(testProbes) {
		t.Fatalf("Exec ran %d times, want %d (one per probe entry)", n, len(testProbes))
	}
	for _, c := range mock.Calls() {
		if len(c.Spec.ROMounts) != 1 || c.Spec.ROMounts[0].HostPath != "/host/tools" {
			t.Errorf("probe call missing the mounted tools: %+v", c.Spec.ROMounts)
		}
		found := false
		for _, e := range c.Spec.Env {
			if e == env[0] {
				found = true
			}
		}
		if !found {
			t.Errorf("probe call missing the threaded env, got %v", c.Spec.Env)
		}
	}
}

// TestProbeCapabilities_CacheKeyDependsOnMounts verifies that two calls for
// the SAME image but DIFFERENT mounts do not share a cache entry — otherwise
// a probe run before a directory was mounted would poison the result for
// every later call with the mount attached.
func TestProbeCapabilities_CacheKeyDependsOnMounts(t *testing.T) {
	const image = "test-image-mount-cache-key"
	InvalidateCapabilityCache(image)
	mock := NewMock(MockResponse{Result: Result{ExitCode: 1}})
	mock.ResponseFunc = func(_ int, spec Spec) (Result, error) {
		if len(spec.ROMounts) > 0 {
			return Result{ExitCode: 0, Stdout: "tok1\ntok2\n"}, nil
		}
		return Result{ExitCode: 1}, nil
	}

	alphaOnly := []ProbeEntry{alphaProbe}
	without := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, alphaOnly)
	if without.Available("alpha", "one") {
		t.Fatalf("without a mount, alpha/one should be unavailable, got %v", without)
	}

	mounts := []ROMount{{HostPath: "/host/tools", ContainerPath: "/opt/kit/tools", Shared: true}}
	with := ProbeCapabilities(context.Background(), mock, image, t.TempDir(), mounts, nil, nil, alphaOnly)
	if !with.Available("alpha", "one") {
		t.Errorf("with the mount, alpha/one should be available (must not reuse the mount-less cache entry), got %v", with)
	}
}

// TestProbeCapabilities_CacheKeyOrderIndependent verifies that the mounts /
// rwMounts / env fragments of the cache key are ORDER-INDEPENDENT: the same
// logical configuration assembled in a different order must hit the same
// cache entry (no second probe run), while a mount moved from read-only to
// writable is a genuinely different configuration and must re-probe.
func TestProbeCapabilities_CacheKeyOrderIndependent(t *testing.T) {
	const image = "test-image-order-independent"
	InvalidateCapabilityCache(image)
	result := Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}

	mountsAB := []ROMount{
		{HostPath: "/host/a", ContainerPath: "/opt/a", Shared: true},
		{HostPath: "/host/b", ContainerPath: "/opt/b"},
	}
	mountsBA := []ROMount{
		{HostPath: "/host/b", ContainerPath: "/opt/b"},
		{HostPath: "/host/a", ContainerPath: "/opt/a", Shared: true},
	}
	rwAB := []ROMount{
		{HostPath: "/host/rw1", ContainerPath: "/kit/rw1"},
		{HostPath: "/host/rw2", ContainerPath: "/kit/rw2"},
	}
	rwBA := []ROMount{
		{HostPath: "/host/rw2", ContainerPath: "/kit/rw2"},
		{HostPath: "/host/rw1", ContainerPath: "/kit/rw1"},
	}
	envAB := []string{"A=1", "B=2"}
	envBA := []string{"B=2", "A=1"}

	first := NewMock(MockResponse{Result: result})
	cs1 := ProbeCapabilities(context.Background(), first, image, t.TempDir(), mountsAB, rwAB, envAB, testProbes)
	if !cs1.Available("beta", "three") {
		t.Fatalf("precondition: first probe should report beta/three available, got %v", cs1)
	}

	// Same configuration, every slice reversed: must be a cache hit.
	second := NewMock(MockResponse{Result: Result{ExitCode: 1}})
	cs2 := ProbeCapabilities(context.Background(), second, image, t.TempDir(), mountsBA, rwBA, envBA, testProbes)
	if !cs2.Available("beta", "three") {
		t.Errorf("reordered assembly must hit the same cache entry, got a fresh (different) result: %v", cs2)
	}
	if n := second.CallCount(); n != 0 {
		t.Errorf("reordered assembly must not re-probe: Exec ran %d times, want 0", n)
	}

	// The same directory moved from a read-only mount to a writable one is a
	// different configuration: must re-probe, not reuse.
	rwMoved := NewMock(MockResponse{Result: result})
	cs3 := ProbeCapabilities(context.Background(), rwMoved, image, t.TempDir(), mountsBA, append(rwBA, ROMount{HostPath: "/host/a", ContainerPath: "/opt/a"}), envBA, testProbes)
	if n := rwMoved.CallCount(); n != len(testProbes) {
		t.Errorf("rw-vs-ro change must re-probe: Exec ran %d times, want %d", n, len(testProbes))
	}
	if !cs3.Available("alpha", "two") {
		t.Errorf("fresh probe should report alpha/two available, got %v", cs3)
	}
}

// TestProbeCapabilities_CacheKeyDependsOnProbeSet verifies the probe-set
// identity fragment of the cache key: two callers probing the SAME image
// with different probe lists must never share an entry — a shared entry
// would hand one caller a result shaped by the other's table (missing modes
// read as unavailable). Names in ORDER are the identity: a reordered list is
// a different key (re-probing is merely redundant; sharing could be wrong).
func TestProbeCapabilities_CacheKeyDependsOnProbeSet(t *testing.T) {
	const image = "test-image-probe-set-key"
	InvalidateCapabilityCache(image)
	result := Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}

	// Different name sets: [alpha] then [alpha beta] must re-probe.
	first := NewMock(MockResponse{Result: result})
	cs1 := ProbeCapabilities(context.Background(), first, image, t.TempDir(), nil, nil, nil, []ProbeEntry{alphaProbe})
	if !cs1.Available("alpha", "one") {
		t.Fatalf("precondition: alpha/one available, got %v", cs1)
	}
	second := NewMock(MockResponse{Result: result})
	cs2 := ProbeCapabilities(context.Background(), second, image, t.TempDir(), nil, nil, nil, testProbes)
	if !cs2.Available("beta", "three") {
		t.Errorf("a different probe list must probe fresh (got a result missing beta/three: %v)", cs2)
	}
	if n := second.CallCount(); n != len(testProbes) {
		t.Errorf("different probe list must not share the [alpha]-only cache entry: Exec ran %d times, want %d", n, len(testProbes))
	}

	// Same names, different order: different key, fresh probe.
	reordered := NewMock(MockResponse{Result: result})
	ProbeCapabilities(context.Background(), reordered, image, t.TempDir(), nil, nil, nil, []ProbeEntry{betaProbe, alphaProbe})
	if n := reordered.CallCount(); n != len(testProbes) {
		t.Errorf("reordered probe list must re-probe: Exec ran %d times, want %d", n, len(testProbes))
	}

	// Same list twice: cached, no re-probe (identity is stable).
	again := NewMock(MockResponse{Result: result})
	ProbeCapabilities(context.Background(), again, image, t.TempDir(), nil, nil, nil, []ProbeEntry{betaProbe, alphaProbe})
	if n := again.CallCount(); n != 0 {
		t.Errorf("identical probe list must hit the cache: Exec ran %d times, want 0", n)
	}
}

// TestProbeCapabilities_ConcurrentProbeOnce verifies the probe-once
// guarantee under concurrency: N goroutines calling ProbeCapabilities with
// the same inputs share one probe run (capFlight), and all receive the same
// result.
func TestProbeCapabilities_ConcurrentProbeOnce(t *testing.T) {
	const image = "test-image-concurrent"
	InvalidateCapabilityCache(image)
	mock := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})
	// Widen the race window so the test actually exercises the sharing path
	// rather than serial probe/hit sequences.
	mock.ResponseFunc = func(_ int, _ Spec) (Result, error) {
		time.Sleep(20 * time.Millisecond)
		return Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}, nil
	}

	const callers = 8
	results := make([]CapabilitySet, callers)
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = ProbeCapabilities(context.Background(), mock, image, t.TempDir(), nil, nil, nil, testProbes)
		}(i)
	}
	wg.Wait()

	if n := mock.CallCount(); n != len(testProbes) {
		t.Errorf("concurrent callers must share one probe run: Exec ran %d times, want %d", n, len(testProbes))
	}
	for i, cs := range results {
		if !cs.Available("alpha", "one") || !cs.Available("beta", "three") {
			t.Errorf("caller %d got an all-unavailable result: %v", i, cs)
		}
	}
}

// TestInvalidateCapabilityCache_DeletesComposedKey verifies that
// InvalidateCapabilityCache deletes every composed entry for an image —
// across mounts/env variants — not just a bare-image key that would never
// match. ProbeCapabilities keys its cache on
// image+"|"+probeSetKey+"|"+mountsEnvCacheKey(...), so a naive
// capCache.Delete(image) would silently no-op against every real entry.
func TestInvalidateCapabilityCache_DeletesComposedKey(t *testing.T) {
	const image = "test-image-invalidate-composed-key"
	InvalidateCapabilityCache(image) // clean slate regardless of prior test order

	mockA := NewMock(MockResponse{Result: Result{ExitCode: 1}})
	mockB := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "tok1\ntok2\ntok3\n"}})

	// Two different mount sets against the SAME image populate two distinct
	// composed cache keys.
	without := ProbeCapabilities(context.Background(), mockA, image, t.TempDir(), nil, nil, nil, testProbes)
	if without.Available("beta", "three") {
		t.Fatalf("precondition: expected beta/three unavailable without a mount, got %v", without)
	}
	mounts := []ROMount{{HostPath: "/host/tools", ContainerPath: "/opt/kit/tools", Shared: true}}
	with := ProbeCapabilities(context.Background(), mockB, image, t.TempDir(), mounts, nil, nil, testProbes)
	if !with.Available("beta", "three") {
		t.Fatalf("precondition: expected beta/three available with a mount, got %v", with)
	}

	InvalidateCapabilityCache(image)

	// After invalidation, BOTH composed entries must be gone — re-probing
	// the previously-available "with mounts" case with an always-unavailable
	// mock must reflect the fresh probe, not a stale cached true.
	reprobed := ProbeCapabilities(context.Background(), mockA, image, t.TempDir(), mounts, nil, nil, testProbes)
	if reprobed.Available("beta", "three") {
		t.Errorf("stale cache entry survived InvalidateCapabilityCache: got %v after re-probing with an always-unavailable mock", reprobed)
	}
	if n := mockA.CallCount(); n != 2*len(testProbes) {
		// mockA ran the first (mount-less) probe of both entries AND the
		// post-invalidation re-probe of both: 2 rounds x len(testProbes).
		t.Errorf("re-probe after invalidation must actually execute: Exec ran %d times, want %d", n, 2*len(testProbes))
	}
}

// TestProbeCapabilitiesPanickingInterpretEvictsCache pins the panic
// contract of a shared flight, in two race-free phases:
//
// Phase A (single goroutine): a panicking Interpret is a caller bug, so the
// claimant's panic propagates (after recover+re-panic in
// ProbeCapabilities) AND the flight's poisoned nil entry is EVICTED — a
// fresh ProbeCapabilities for the same key re-probes for real instead of
// serving the cached nil forever.
//
// Phase B: a caller that joins an already-panicked flight — the cache
// holds a done-closed flight whose cs is nil, exactly what a panicked
// claimant leaves behind for waiters that loaded the entry before the
// panic — must receive the nil (all-unavailable) set, never hang.
func TestProbeCapabilitiesPanickingInterpretEvictsCache(t *testing.T) {
	const image = "test-image-panicking-interpret"
	InvalidateCapabilityCache(image) // clean slate regardless of prior test order

	panicking := ProbeEntry{
		Name:  "boom",
		Probe: []string{"echo", "irrelevant"},
		Interpret: func(ProbeResult) map[string]bool {
			panic("buggy Interpret")
		},
	}
	reprober := NewMock(MockResponse{Result: Result{ExitCode: 0, Stdout: "irrelevant\n"}})

	// Phase A: the panic propagates to the claiming caller.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected the panicking Interpret's panic to propagate in the claimant")
			}
		}()
		ProbeCapabilities(context.Background(), NewMock(MockResponse{Result: Result{ExitCode: 0}}), image, t.TempDir(), nil, nil, nil, []ProbeEntry{panicking})
	}()

	// Phase A: the eviction — a fresh call for the SAME key (the probe-set
	// identity is the entry's NAME, so a healthy Interpret named "boom"
	// addresses the identical cache entry) must actually re-probe — its
	// mock records the Exec — and return a healthy set, proving the
	// poisoned nil entry was never served from the cache.
	healthy := ProbeEntry{
		Name:  "boom",
		Probe: []string{"echo", "irrelevant"},
		Interpret: func(ProbeResult) map[string]bool {
			return map[string]bool{"ok": true}
		},
	}
	cs := ProbeCapabilities(context.Background(), reprober, image, t.TempDir(), nil, nil, nil, []ProbeEntry{healthy})
	if cs == nil {
		t.Fatal("re-probe after a panicking flight got the poisoned nil set from the cache")
	}
	if !cs.Available("boom", "ok") {
		t.Errorf("re-probed set should carry the healthy Interpret's verdict, got %v", cs)
	}
	if n := reprober.CallCount(); n != 1 {
		t.Errorf("re-probe must execute the probe again after eviction: Exec ran %d times, want 1", n)
	}
	InvalidateCapabilityCache(image) // remove Phase A's healthy entry before Phase B

	// Phase B: a caller joining a flight that a panicked claimant left
	// behind (done closed, cs nil — stored here directly, so the join is
	// deterministic: no claimant is running that could evict it) receives
	// the nil set and does not hang.
	flight := &capFlight{done: make(chan struct{})}
	key := image + "|" + probeSetKey([]ProbeEntry{panicking}) + "|" + mountsEnvCacheKey(nil, nil, nil)
	capCache.Store(key, flight)
	t.Cleanup(func() { capCache.Delete(key) })

	waiterDone := make(chan CapabilitySet, 1)
	go func() {
		waiterDone <- ProbeCapabilities(context.Background(), NewMock(MockResponse{}), image, t.TempDir(), nil, nil, nil, []ProbeEntry{panicking})
	}()
	// A panicked claimant's cleanup: close(done) with cs still nil (the
	// defer ordering in ProbeCapabilities guarantees waiters are released
	// even though the claimant re-panicked).
	close(flight.done)
	select {
	case cs := <-waiterDone:
		if cs != nil {
			t.Errorf("waiter of a panicked flight should receive the nil (all-unavailable) set, got %v", cs)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waiter blocked forever on a panicked flight")
	}
}

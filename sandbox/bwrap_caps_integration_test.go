//go:build integration

package sandbox

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// bwrap_caps_integration_test.go pins bead llmkit-bk8.1.16 end to end: under
// the systemd-run cap wrapper, every Spec.Cmd argv element reaches the inner
// command byte-for-byte. On a host where systemd-run expands ${NAME} from
// the HOST environment (systemd before the fix), the row would print the
// host value; with --expand-environment=no in place it prints the empty
// in-sandbox expansion (bwrap --clearenv), the literal, and the literal $$.

// TestBwrapCapWrapperPreservesArgvByteForByte skips unless this host's bwrap
// runs under the systemd-run cap method — the only wrapper that re-reads
// the argv.
func TestBwrapCapWrapperPreservesArgvByteForByte(t *testing.T) {
	if detectCapMethod(t.Context(), systemdRunExpandSupport) != bwrapCapSystemdRun {
		t.Skip("cap method is not systemd-run; nothing re-reads the argv here")
	}

	t.Setenv("ZZHOSTONLY", "host-secret-value")

	s := newTestBwrap(t)
	defer func() { _ = s.Close() }()

	res, err := s.Exec(t.Context(), Spec{
		RepoDir: t.TempDir(),
		Cmd:     []string{"/bin/sh", "-c", "printf '%s|%s|%s\\n' \"${ZZHOSTONLY}\" '$ZZHOSTONLY' '$$'"},
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	got := strings.TrimSuffix(res.Stdout, "\n")
	if got != "|$ZZHOSTONLY|$$" {
		t.Fatalf("stdout = %q, want |$ZZHOSTONLY|$$ — the wrapper must not expand ${NAME} from the host env, must not touch quoted literals, and must leave $$ alone", got)
	}
}

// envIChildArg is the positional argument that tells the re-executed test
// binary it is the env -i child of TestBwrapCapDetectionAgreesWithExecUnderEnvI.
const envIChildArg = "llmkit-envi-child"

// TestBwrapCapDetectionAgreesWithExecUnderEnvI pins bead llmkit-bk8.1.19: in
// a session with no user-bus variables (env -i PATH=/usr/bin:/bin HOME=...
// GOPATH=... GOCACHE=... GOMODCACHE=...), DescribeBwrapCapMethod and Exec
// give one answer. Where Describe reports no method, a CapStrict Exec
// returns ErrBwrapNoCapMethod with nothing written to the caller's
// Workspace and a CapBestEffort Exec runs the command; where it reports a
// method, a CapStrict Exec runs the command. The test re-executes its own
// binary under env -i and runs the assertions there.
func TestBwrapCapDetectionAgreesWithExecUnderEnvI(t *testing.T) {
	if flag.Arg(0) == envIChildArg {
		envIChild(t)
		return
	}
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	env := []string{"PATH=/usr/bin:/bin"}
	for _, k := range []string{"HOME", "GOPATH", "GOCACHE", "GOMODCACHE"} {
		if v, ok := os.LookupEnv(k); ok {
			env = append(env, k+"="+v)
		}
	}
	cmd := exec.Command(bin, "-test.v", "-test.count=1", "-test.run=^TestBwrapCapDetectionAgreesWithExecUnderEnvI$", envIChildArg)
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if strings.Contains(string(out), "--- SKIP") {
		t.Skipf("env -i child skipped:\n%s", out)
	}
	if err != nil {
		t.Fatalf("env -i child failed: %v\n%s", err, out)
	}
}

func envIChild(t *testing.T) {
	ctx := context.Background()
	label, enforced := DescribeBwrapCapMethod(ctx)
	t.Logf("DescribeBwrapCapMethod under env -i = (%q, %v)", label, enforced)

	strict := newTestBwrap(t)
	defer func() { _ = strict.Close() }()
	ws := t.TempDir()
	spec := Spec{
		Workspace:  ws,
		WriteFiles: map[string][]byte{"marker.txt": []byte("written")},
		Cmd:        []string{"/bin/sh", "-c", "true"},
	}
	res, err := strict.Exec(ctx, spec)
	if enforced {
		if err != nil || res.ExitCode != 0 {
			t.Fatalf("Describe reports %q enforced but CapStrict Exec = (exit %d, err %v): detection and Exec disagree", label, res.ExitCode, err)
		}
		return
	}
	if !errors.Is(err, ErrBwrapNoCapMethod) {
		t.Fatalf("Describe reports no method but CapStrict Exec err = %v, want errors.Is ErrBwrapNoCapMethod", err)
	}
	if _, statErr := os.Stat(filepath.Join(ws, "marker.txt")); !os.IsNotExist(statErr) {
		t.Errorf("the refused CapStrict Exec wrote into the caller's Workspace (stat err %v)", statErr)
	}
	loose := newTestBwrap(t, WithCapPolicy(CapBestEffort))
	defer func() { _ = loose.Close() }()
	res, err = loose.Exec(ctx, spec)
	if err != nil || res.ExitCode != 0 {
		t.Fatalf("CapBestEffort Exec = (exit %d, err %v, stderr %q), want the command to run with exit 0", res.ExitCode, err, res.Stderr)
	}
}

// TestDelegatedCgroupV2DirAcceptsDelegatedAncestor is the positive half of
// the 1.19 rule, against the real cgroup tree: an ancestor of this
// process's cgroup that enables the limit controllers for its children and
// lets this user create one is reported, and the probe leaves nothing
// behind. Skips on hosts with no such ancestor.
func TestDelegatedCgroupV2DirAcceptsDelegatedAncestor(t *testing.T) {
	selfData, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		t.Skipf("no /proc/self/cgroup: %v", err)
	}
	_, path, ok := strings.Cut(strings.TrimSpace(string(selfData)), "::")
	if !ok {
		t.Skip("not a cgroup v2 host")
	}
	for path != "/" && path != "" {
		path = filepath.Dir(path)
		dir := cgroupV2Root + path
		probe := filepath.Join(dir, ".llmkit-test-ancestor-probe")
		if os.Mkdir(probe, 0o755) != nil {
			continue
		}
		delegated := cgroupLimitsWritable(probe)
		_ = os.Remove(probe)
		if !delegated {
			continue
		}
		fake := filepath.Join(t.TempDir(), "cgroup")
		if err := os.WriteFile(fake, []byte("0::"+path+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		before := childEntries(t, dir, cgroupProbeNamePrefix)
		got, ok := delegatedCgroupV2DirAt(cgroupV2Root, fake)
		if !ok || got != dir {
			t.Fatalf("delegatedCgroupV2DirAt(%q) = (%q, %v), want (%q, true)", path, got, ok, dir)
		}
		assertNoNewChildren(t, dir, before, cgroupProbeNamePrefix)
		return
	}
	t.Skip("no ancestor of this process's cgroup is delegated for memory, cpu and pids")
}

// cgroupProbeNamePrefix matches every name delegatedCgroupV2DirAt has used
// for its throwaway child, fixed or per call.
const cgroupProbeNamePrefix = ".llmkit-cgroup-probe"

// delegatedAncestor returns the directory of the nearest ancestor of this
// process's cgroup whose children get writable memory, cpu and pids limit
// files, and a self-cgroup file naming it. It skips the test when there is
// none.
func delegatedAncestor(t *testing.T) (dir, selfFile string) {
	t.Helper()
	selfData, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		t.Skipf("no /proc/self/cgroup: %v", err)
	}
	_, path, ok := strings.Cut(strings.TrimSpace(string(selfData)), "::")
	if !ok {
		t.Skip("not a cgroup v2 host")
	}
	for path != "/" && path != "" {
		path = filepath.Dir(path)
		dir := cgroupV2Root + path
		probe := filepath.Join(dir, ".llmkit-test-ancestor-probe-"+randToken())
		if os.Mkdir(probe, 0o755) != nil {
			continue
		}
		delegated := cgroupLimitsWritable(probe)
		if err := os.Remove(probe); err != nil {
			t.Fatalf("remove %s: %v", probe, err)
		}
		if !delegated {
			continue
		}
		selfFile := filepath.Join(t.TempDir(), "cgroup")
		if err := os.WriteFile(selfFile, []byte("0::"+path+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir, selfFile
	}
	t.Skip("no ancestor of this process's cgroup is delegated for memory, cpu and pids")
	return "", ""
}

// childEntries returns the names in dir that start with one of prefixes.
func childEntries(t *testing.T, dir string, prefixes ...string) map[string]bool {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, e := range entries {
		for _, p := range prefixes {
			if strings.HasPrefix(e.Name(), p) {
				names[e.Name()] = true
			}
		}
	}
	return names
}

// assertNoNewChildren fails the test if dir holds an entry starting with one
// of prefixes that was not in before. It re-reads dir for up to a second
// before failing, so a probe another process removes in that time is not
// counted.
func assertNoNewChildren(t *testing.T, dir string, before map[string]bool, prefixes ...string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		var left []string
		for name := range childEntries(t, dir, prefixes...) {
			if !before[name] {
				left = append(left, name)
			}
		}
		if len(left) == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Errorf("left behind in %s: %v", dir, left)
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestDelegatedCgroupV2DirConcurrentCallsAllReportDelegated pins bead
// llmkit-bk8.7.23 at the function level: 16 detections at once against one
// delegated cgroup (a delegated ancestor stands in for this process's own
// cgroup) all report it, 40 rounds over, and leave no probe dir behind. With
// one fixed probe name the concurrent callers collide on it and most of them
// report no delegation.
func TestDelegatedCgroupV2DirConcurrentCallsAllReportDelegated(t *testing.T) {
	dir, self := delegatedAncestor(t)
	before := childEntries(t, dir, cgroupProbeNamePrefix)
	const workers, rounds = 16, 40
	var mu sync.Mutex
	notDelegated := 0
	for range rounds {
		start := make(chan struct{})
		var wg sync.WaitGroup
		for range workers {
			wg.Go(func() {
				<-start
				if got, ok := delegatedCgroupV2DirAt(cgroupV2Root, self); !ok || got != dir {
					mu.Lock()
					notDelegated++
					mu.Unlock()
				}
			})
		}
		close(start)
		wg.Wait()
	}
	t.Logf("%d/%d concurrent detections on %s reported no delegation", notDelegated, workers*rounds, dir)
	if notDelegated != 0 {
		t.Errorf("%d/%d concurrent detections on the delegated %s reported no delegation, want 0", notDelegated, workers*rounds, dir)
	}
	assertNoNewChildren(t, dir, before, cgroupProbeNamePrefix)
}

// TestBwrapConcurrentExecsKeepCgroupV2Caps pins bead llmkit-bk8.7.23 end to
// end: on a host whose own cgroup is delegated (emulated: the self-cgroup
// file names a delegated ancestor and systemd-run is reported unavailable),
// 16 concurrent Execs on one Bwrap all resolve the cgroup v2 method, as a
// sequential Exec does. Under CapRequired none returns ErrBwrapNoCapMethod;
// under CapBestEffort none runs uncapped.
func TestBwrapConcurrentExecsKeepCgroupV2Caps(t *testing.T) {
	dir, self := delegatedAncestor(t)
	prevSelf, prevSystemd, prevDetect := selfCgroupFile, systemdRunUserAvailable, detectCapMethod
	t.Cleanup(func() {
		selfCgroupFile, systemdRunUserAvailable, detectCapMethod = prevSelf, prevSystemd, prevDetect
	})
	selfCgroupFile = self
	systemdRunUserAvailable = func(context.Context) bool { return false }
	var mu sync.Mutex
	methods := map[bwrapCapMethod]int{}
	detectCapMethod = func(ctx context.Context, expandSupport func(context.Context) bool) bwrapCapMethod {
		m := prevDetect(ctx, expandSupport)
		mu.Lock()
		methods[m]++
		mu.Unlock()
		return m
	}
	prefixes := []string{cgroupProbeNamePrefix, "llmkit-"}
	before := childEntries(t, dir, prefixes...)
	spec := func() Spec { return Spec{RepoDir: t.TempDir(), Cmd: []string{"/bin/sh", "-c", "true"}} }

	seq := newTestBwrap(t)
	defer func() { _ = seq.Close() }()
	if res, err := seq.Exec(context.Background(), spec()); err != nil || res.ExitCode != 0 || methods[bwrapCapCgroupV2] != 1 {
		t.Skipf("a sequential Exec does not get cgroup v2 caps under %s here (exit %d, err %v, methods %v)", dir, res.ExitCode, err, methods)
	}

	const workers, rounds = 16, 5
	for _, policy := range []CapPolicy{CapRequired, CapBestEffort} {
		b := newTestBwrap(t, WithCapPolicy(policy))
		clear(methods)
		var failed []string
		for range rounds {
			start := make(chan struct{})
			var wg sync.WaitGroup
			for range workers {
				wg.Go(func() {
					<-start
					res, err := b.Exec(context.Background(), spec())
					if err != nil || res.ExitCode != 0 {
						mu.Lock()
						failed = append(failed, fmt.Sprintf("exit %d err %v", res.ExitCode, err))
						mu.Unlock()
					}
				})
			}
			close(start)
			wg.Wait()
		}
		_ = b.Close()
		t.Logf("policy %d: detections %v, failed Execs %d/%d", policy, methods, len(failed), workers*rounds)
		if methods[bwrapCapCgroupV2] != workers*rounds || len(failed) != 0 {
			t.Errorf("policy %d: %d/%d concurrent Execs resolved cgroup v2 (detections %v), %d failed %v; want every one capped by cgroup v2 like the sequential Exec",
				policy, methods[bwrapCapCgroupV2], workers*rounds, methods, len(failed), failed)
		}
	}
	assertNoNewChildren(t, dir, before, prefixes...)
}

// killedProberChildArg is the positional argument that tells the re-executed
// test binary it is the prober TestDelegatedCgroupV2DirSurvivesKilledProber
// kills; the next argument is the self-cgroup file to probe with.
const killedProberChildArg = "llmkit-killed-prober-child"

// TestDelegatedCgroupV2DirSurvivesKilledProber pins bead llmkit-bk8.7.23's
// crash case: a prober SIGKILLed between creating its probe dir and removing
// it leaves that dir behind, and a later detection on the same cgroup still
// reports the delegation. The test removes the dir the killed prober left.
func TestDelegatedCgroupV2DirSurvivesKilledProber(t *testing.T) {
	if flag.Arg(0) == killedProberChildArg {
		for deadline := time.Now().Add(20 * time.Second); time.Now().Before(deadline); {
			delegatedCgroupV2DirAt(cgroupV2Root, flag.Arg(1))
		}
		return
	}
	dir, self := delegatedAncestor(t)
	before := childEntries(t, dir, cgroupProbeNamePrefix)
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	child := exec.Command(bin, "-test.count=1", "-test.run=^TestDelegatedCgroupV2DirSurvivesKilledProber$", killedProberChildArg, self)
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	pid := child.Process.Pid
	exited := false
	t.Cleanup(func() {
		if !exited {
			_ = child.Process.Kill()
			_ = child.Wait()
		}
	})

	// Stop the prober at arbitrary points until one lands between its mkdir
	// and its remove, then kill it there.
	var left []string
	for deadline := time.Now().Add(15 * time.Second); len(left) == 0; {
		if time.Now().After(deadline) {
			t.Fatal("never stopped the prober with its probe dir open")
		}
		time.Sleep(time.Millisecond)
		if err := syscall.Kill(pid, syscall.SIGSTOP); err != nil {
			t.Fatal(err)
		}
		waitProcStopped(t, pid)
		for name := range childEntries(t, dir, cgroupProbeNamePrefix) {
			if !before[name] {
				left = append(left, name)
			}
		}
		if len(left) == 0 {
			if err := syscall.Kill(pid, syscall.SIGCONT); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	_ = child.Wait()
	exited = true
	for _, name := range left {
		path := filepath.Join(dir, name)
		t.Cleanup(func() { _ = os.Remove(path) })
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("the killed prober's probe dir %s is gone (%v); the kill did not land mid-probe", path, err)
		}
		t.Logf("killed prober left %s", path)
	}

	if got, ok := delegatedCgroupV2DirAt(cgroupV2Root, self); !ok || got != dir {
		t.Errorf("after a prober was killed mid-probe, delegatedCgroupV2DirAt = (%q, %v), want (%q, true)", got, ok, dir)
	}
	for _, name := range left {
		if err := os.Remove(filepath.Join(dir, name)); err != nil {
			t.Errorf("remove the killed prober's probe dir: %v", err)
		}
	}
	assertNoNewChildren(t, dir, before, cgroupProbeNamePrefix)
}

// waitProcStopped waits until /proc reports every thread of pid stopped, so
// the prober cannot run another syscall before it is inspected.
func waitProcStopped(t *testing.T, pid int) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(100 * time.Microsecond) {
		stats, err := filepath.Glob(fmt.Sprintf("/proc/%d/task/*/stat", pid))
		if err != nil || len(stats) == 0 {
			t.Fatalf("list threads of pid %d: %v", pid, err)
		}
		stopped := true
		for _, f := range stats {
			stat, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			// The state is the field after the parenthesized command name.
			i := strings.LastIndexByte(string(stat), ')')
			if i < 0 || !strings.HasPrefix(string(stat[i:]), ") T") {
				stopped = false
				break
			}
		}
		if stopped {
			return
		}
	}
	t.Fatalf("pid %d did not stop", pid)
}

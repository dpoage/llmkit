//go:build linux

package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

const (
	killHelperEnv  = "LLMKIT_S1_HELPER"
	killMarkEnv    = "LLMKIT_S1_MARK"
	killHelperRepo = "LLMKIT_S1_REPO"
	killHelperMark = "LLMKIT_S1_MARKVAL"
	// killCPUBound is the CPU time a kill whose children reads all fail
	// may use over its whole walk budget.
	killCPUBound = 200 * time.Millisecond
)

// TestHostExecKillHelperProcess is the re-exec target of the kill tests, not
// a test: without killHelperEnv it returns at once.
func TestHostExecKillHelperProcess(t *testing.T) {
	mode := os.Getenv(killHelperEnv)
	if mode == "" {
		return
	}
	// A helper that outlives its test is a leak, not a service.
	time.AfterFunc(2*time.Minute, func() { os.Exit(0) })
	switch mode {
	case "forker":
		// Fork from several locked OS threads, none of them the
		// thread-group leader, and keep forking while the kill runs.
		for range 4 {
			go func() {
				runtime.LockOSThread()
				for range 60 {
					_ = exec.Command("sleep", "30").Start()
					time.Sleep(20 * time.Millisecond)
				}
				select {}
			}()
		}
		select {}
	case "hostexec":
		_, _ = NewHostExec().Exec(context.Background(), Spec{
			RepoDir: os.Getenv(killHelperRepo),
			Cmd:     []string{"sleep", "30"},
			Env:     []string{killMarkEnv + "=" + os.Getenv(killHelperMark)},
		})
		os.Exit(0)
	}
}

func newKillMark(t *testing.T) string {
	t.Helper()
	mark := fmt.Sprintf("%d-%s", os.Getpid(), randToken())
	t.Cleanup(func() {
		for pid := range markedProcs(mark) {
			_ = syscall.Kill(pid, syscall.SIGKILL)
		}
	})
	return mark
}

// markedProcs maps the pid of every live (not zombie) process whose
// environment carries the mark to its state letter.
func markedProcs(mark string) map[int]byte {
	want := killMarkEnv + "=" + mark
	out := map[int]byte{}
	ents, _ := os.ReadDir("/proc")
	for _, e := range ents {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		env, err := os.ReadFile("/proc/" + e.Name() + "/environ")
		if err != nil {
			continue
		}
		hit := false
		for _, kv := range strings.Split(string(env), "\x00") {
			if kv == want {
				hit = true
				break
			}
		}
		if !hit {
			continue
		}
		state, _, err := readProcStat("/proc/" + e.Name() + "/stat")
		if err != nil || state == 'Z' || state == 'X' {
			continue
		}
		out[pid] = state
	}
	return out
}

func pidDead(pid int) bool {
	state, _, err := readProcStat("/proc/" + strconv.Itoa(pid) + "/stat")
	return err != nil || state == 'Z' || state == 'X'
}

// killRun runs script on a fresh HostExec under the given Spec.Timeout
// and/or a caller deadline (zero = none) and reports how long Exec took to
// return after the earlier of the two fired. The workspace is prepared
// first, so the clock does not include the copy.
func killRun(t *testing.T, mark, script string, timeout, callerDeadline time.Duration) (Result, time.Duration, error) {
	t.Helper()
	ctx := context.Background()
	kill := timeout
	if callerDeadline > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, callerDeadline)
		defer cancel()
		if kill == 0 || callerDeadline < kill {
			kill = callerDeadline
		}
	}
	spec := Spec{
		Workspace: t.TempDir(),
		Cmd:       []string{"/bin/sh", "-c", script},
		Timeout:   timeout,
		Env:       []string{killMarkEnv + "=" + mark},
	}
	start := time.Now()
	res, err := NewHostExec().Exec(ctx, spec)
	return res, time.Since(start) - kill, err
}

// withProcOps swaps the walk's OS seam for one test. The tests using it are
// not parallel, and Exec has returned (its Cancel goroutine finished) before
// the restore.
func withProcOps(t *testing.T, mutate func(*procOps)) {
	t.Helper()
	old := hostProcOps
	mutate(&hostProcOps)
	t.Cleanup(func() { hostProcOps = old })
}

// execWithin runs f and fails the test, leaving the goroutine to the mark
// cleanup, when it does not return within d.
func execWithin(t *testing.T, d time.Duration, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { defer close(done); f() }()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("Exec did not return within %v", d)
	}
}

func TestHostExecKillsForkedTreeOnTimeout(t *testing.T) {
	mark := newKillMark(t)
	res, sinceKill, err := killRun(t, mark, "echo tok1; (sleep 3; echo tok2); sleep 1", 1500*time.Millisecond, 0)
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !res.TimedOut || res.ExitCode != -1 {
		t.Errorf("res = %+v, want TimedOut with ExitCode -1", res)
	}
	if res.Stdout != "tok1\n" {
		t.Errorf("Stdout = %q, want %q: a descendant wrote after the kill", res.Stdout, "tok1\n")
	}
	if sinceKill > 2*time.Second {
		t.Errorf("Exec returned %v after the kill, want <= 2s", sinceKill)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("descendants alive at return: %v", left)
	}
}

func TestHostExecKillsTreeOnCallerEnd(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cause func(t *testing.T, mark, pidFile string) (time.Duration, error)
	}{
		{"deadline", func(t *testing.T, mark, pidFile string) (time.Duration, error) {
			_, since, err := killRun(t, mark, "sleep 30 & echo $! > "+pidFile+"; wait", 0, 500*time.Millisecond)
			return since, err
		}},
		{"cancel", func(t *testing.T, mark, pidFile string) (time.Duration, error) {
			ctx, cancel := context.WithCancel(context.Background())
			spec := Spec{
				Workspace: t.TempDir(),
				Cmd:       []string{"/bin/sh", "-c", "(sleep 30) & echo $! > " + pidFile + "; wait"},
				Env:       []string{killMarkEnv + "=" + mark},
			}
			var killedAt time.Time
			go func() {
				time.Sleep(500 * time.Millisecond)
				killedAt = time.Now()
				cancel()
			}()
			_, err := NewHostExec().Exec(ctx, spec)
			return time.Since(killedAt), err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mark := newKillMark(t)
			pidFile := filepath.Join(t.TempDir(), "pid")
			since, err := tc.cause(t, mark, pidFile)
			if err == nil || !strings.HasPrefix(err.Error(), "sandbox: execution cancelled") {
				t.Fatalf("err = %v, want the cancelled error", err)
			}
			if since > 2*time.Second {
				t.Errorf("Exec returned %v after the kill, want <= 2s", since)
			}
			b, rerr := os.ReadFile(pidFile)
			if rerr != nil {
				t.Fatalf("pid file: %v", rerr)
			}
			pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
			if pid <= 0 || !pidDead(pid) {
				t.Errorf("grandchild %d alive at return", pid)
			}
			if left := markedProcs(mark); len(left) != 0 {
				t.Errorf("descendants alive at return: %v", left)
			}
		})
	}
}

// TestHostExecKillTimingFiveRuns measures Exec's return from the kill
// instant on five runs and logs each duration.
func TestHostExecKillTimingFiveRuns(t *testing.T) {
	var ds []time.Duration
	for range 5 {
		mark := newKillMark(t)
		_, since, err := killRun(t, mark, "(sleep 5) & wait", 0, 500*time.Millisecond)
		if err == nil {
			t.Fatal("want the cancelled error")
		}
		ds = append(ds, since)
		if since > 2*time.Second {
			t.Errorf("Exec returned %v after the kill, want <= 2s", since)
		}
	}
	t.Logf("return-after-kill durations: %v", ds)
}

// TestHostExecKillsForkBombLiteDescendant: a descendant forking children
// continuously while the kill runs leaves no marked process. The fork loop
// is bounded and, unlike the other rows, not held to a time bound; it holds
// no pipe to Exec, so Exec's return does not wait for it.
func TestHostExecKillsForkBombLiteDescendant(t *testing.T) {
	mark := newKillMark(t)
	_, _, err := killRun(t, mark,
		"sh -c 'i=0; while [ $i -lt 250 ]; do sleep 30 & i=$((i+1)); sleep 0.008; done; wait' >/dev/null 2>&1 & wait",
		time.Second, 0)
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("%d marked processes alive at return", len(left))
	}
}

// TestHostExecKillsMultithreadedForker: the descendant forks from several
// OS threads, none the thread-group leader; children hang off the tasks'
// own children files.
func TestHostExecKillsMultithreadedForker(t *testing.T) {
	mark := newKillMark(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewHostExec().Exec(context.Background(), Spec{
		Workspace: t.TempDir(),
		Cmd:       []string{exe, "-test.run=^TestHostExecKillHelperProcess$"},
		Timeout:   time.Second,
		Env:       []string{killMarkEnv + "=" + mark, killHelperEnv + "=forker"},
	})
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("%d marked processes alive at return", len(left))
	}
}

// TestHostExecKillLeavesRedirectedDescendantDead: a descendant that holds
// no pipe to Exec is dead, not merely signalled, when Exec returns. It is
// checked by pid, since a dying process no longer shows its environment.
func TestHostExecKillLeavesRedirectedDescendantDead(t *testing.T) {
	for i := range 4 {
		mark := newKillMark(t)
		pidFile := filepath.Join(t.TempDir(), "pid")
		_, _, err := killRun(t, mark, "sleep 30 >/dev/null 2>&1 & echo $! > "+pidFile+"; sleep 30", 500*time.Millisecond, 0)
		if err != nil {
			t.Fatalf("run %d: Exec: %v", i, err)
		}
		b, rerr := os.ReadFile(pidFile)
		pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
		if rerr != nil || pid <= 0 || !pidDead(pid) {
			t.Fatalf("run %d: descendant %d (err %v) alive at return", i, pid, rerr)
		}
	}
}

// TestKillHeldWaitsForEachProcessToBeGone: after signalling, killHeld
// returns only once every held pidfd reports its process gone — here a pipe
// that becomes readable 200ms later stands in for a slow-dying process —
// and gives up after killReapBudget on one that never does.
func TestKillHeldWaitsForEachProcessToBeGone(t *testing.T) {
	pipe := func() (r, w int) {
		var fds [2]int
		if err := unix.Pipe2(fds[:], unix.O_CLOEXEC); err != nil {
			t.Fatal(err)
		}
		return fds[0], fds[1]
	}
	t.Run("readable later", func(t *testing.T) {
		r, w := pipe()
		defer func() { _ = unix.Close(w) }()
		go func() {
			time.Sleep(200 * time.Millisecond)
			_, _ = unix.Write(w, []byte{1})
		}()
		start := time.Now()
		killHeld([]heldProc{{pid: 0, fd: r}}, -1, hostProcOps, nil)
		if d := time.Since(start); d < 150*time.Millisecond {
			t.Errorf("killHeld returned after %v, before the process reported gone", d)
		}
	})
	t.Run("never readable", func(t *testing.T) {
		r, w := pipe()
		defer func() { _ = unix.Close(w) }()
		start := time.Now()
		killHeld([]heldProc{{pid: 0, fd: r}}, -1, hostProcOps, nil)
		if d := time.Since(start); d > killReapBudget+time.Second {
			t.Errorf("killHeld waited %v, want it bounded by %v", d, killReapBudget)
		}
	})
}

// TestHostExecKillNeverSignalsANonDescendant: a children source that names
// a process outside the tree gets that process neither stopped nor killed.
func TestHostExecKillNeverSignalsANonDescendant(t *testing.T) {
	mark := newKillMark(t)
	stranger := exec.Command("sleep", "60")
	if err := stranger.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = stranger.Process.Kill(); _ = stranger.Wait() })
	withProcOps(t, func(o *procOps) {
		o.children = func(pid int) ([]int, error) {
			kids, err := readProcChildren(pid)
			return append(kids, stranger.Process.Pid), err
		}
	})
	res, _, err := killRun(t, mark, "sleep 30 >/dev/null 2>&1 & wait", 500*time.Millisecond, 0)
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	state, _, serr := readProcStat("/proc/" + strconv.Itoa(stranger.Process.Pid) + "/stat")
	if serr != nil || state != 'S' {
		t.Errorf("stranger state = %q err = %v, want S: it was signalled", state, serr)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("descendants alive at return: %v", left)
	}
}

// TestHostExecKillFindsChildShownOnLaterRead: a child the root reports only
// on its second read is still found, killed.
func TestHostExecKillFindsChildShownOnLaterRead(t *testing.T) {
	mark := newKillMark(t)
	var mu sync.Mutex
	firstPid, reads := 0, 0
	withProcOps(t, func(o *procOps) {
		o.children = func(pid int) ([]int, error) {
			kids, err := readProcChildren(pid)
			mu.Lock()
			defer mu.Unlock()
			if firstPid == 0 {
				firstPid = pid
			}
			if pid == firstPid {
				reads++
				if reads == 1 {
					return nil, err
				}
			}
			return kids, err
		}
	})
	res, _, err := killRun(t, mark, "sleep 30 >/dev/null 2>&1 & wait", 500*time.Millisecond, 0)
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("child shown on the second read survived: %v", left)
	}
}

// TestHostExecKillBeforeRootIsArmedStillKillsTree: a deadline that fires
// between Start and the root's pidfd being taken is not lost; the tree is
// killed as soon as the root is held.
func TestHostExecKillBeforeRootIsArmedStillKillsTree(t *testing.T) {
	mark := newKillMark(t)
	var slowOnce sync.Once
	withProcOps(t, func(o *procOps) {
		open := o.pidfdOpen
		o.pidfdOpen = func(pid int) (int, error) {
			slowOnce.Do(func() { time.Sleep(400 * time.Millisecond) })
			return open(pid)
		}
	})
	var res Result
	var err error
	execWithin(t, 10*time.Second, func() {
		res, _, err = killRun(t, mark, "sleep 30 >/dev/null 2>&1 & wait", 200*time.Millisecond, 0)
	})
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("descendant survived a kill that preceded arming: %v", left)
	}
}

// TestHostExecKillFailureNeverWorseThanBase: when the walk cannot hold a
// descendant, or the root at all, the command is still killed, Exec
// returns, and no process is left stopped.
func TestHostExecKillFailureNeverWorseThanBase(t *testing.T) {
	failure := errors.New("injected pidfd_open failure")
	for _, tc := range []struct {
		name string
		open func(calls *atomic.Int32) func(pid int) (int, error)
	}{
		{"candidates fail", func(calls *atomic.Int32) func(int) (int, error) {
			return func(pid int) (int, error) {
				if calls.Add(1) > 1 {
					return -1, failure
				}
				return unix.PidfdOpen(pid, 0)
			}
		}},
		{"root fails", func(calls *atomic.Int32) func(int) (int, error) {
			return func(int) (int, error) { calls.Add(1); return -1, failure }
		}},
		{"root gets ENOSYS", func(calls *atomic.Int32) func(int) (int, error) {
			return func(int) (int, error) { calls.Add(1); return -1, unix.ENOSYS }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mark := newKillMark(t)
			var calls atomic.Int32
			withProcOps(t, func(o *procOps) { o.pidfdOpen = tc.open(&calls) })
			var res Result
			var err error
			execWithin(t, 10*time.Second, func() {
				res, _, err = killRun(t, mark, "sleep 30 >/dev/null 2>&1 & echo up; sleep 30 >/dev/null 2>&1; true", 500*time.Millisecond, 0)
			})
			if err != nil || !res.TimedOut {
				t.Fatalf("res = %+v err = %v, want a timeout", res, err)
			}
			for pid, state := range markedProcs(mark) {
				if state == 'T' || state == 't' {
					t.Errorf("process %d left stopped", pid)
				}
			}
			if calls.Load() == 0 {
				t.Error("the injected open was never used")
			}
		})
	}
}

// TestHostExecKillRestopsRootAfterSIGCONT: a descendant that resumes the
// frozen root with SIGCONT — once, or in a finite burst — the moment the
// root shows stopped does not shield the root's other children.
func TestHostExecKillRestopsRootAfterSIGCONT(t *testing.T) {
	for _, tc := range []struct{ name, onStopped string }{
		{"single", "kill -CONT $r"},
		{"burst", "j=0; while [ $j -lt 20000 ]; do kill -CONT $r; j=$((j+1)); done"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mark := newKillMark(t)
			script := "r=$$; " +
				"( while :; do read -r _ _ s _ < /proc/$r/stat; if [ \"$s\" = T ]; then " + tc.onStopped + "; exit; fi; done ) & " +
				"i=0; while [ $i -lt 100 ]; do sleep 30 >/dev/null 2>&1 & i=$((i+1)); done; wait"
			var res Result
			var since time.Duration
			var err error
			execWithin(t, 10*time.Second, func() {
				res, since, err = killRun(t, mark, script, time.Second, 0)
			})
			if err != nil || !res.TimedOut {
				t.Fatalf("res = %+v err = %v, want a timeout", res, err)
			}
			t.Logf("returned %v after the kill", since)
			if left := markedProcs(mark); len(left) != 0 {
				t.Errorf("%d marked processes alive at return", len(left))
			}
			if since > 2*time.Second {
				t.Errorf("Exec returned %v after the kill, want <= 2s", since)
			}
		})
	}
}

// TestHostExecKillRestopEndsTheWalkEarly: a root resumed once by SIGCONT
// during the kill is stopped again, so the walk ends as soon as the tree
// is frozen rather than running out its budget with the root running.
func TestHostExecKillRestopEndsTheWalkEarly(t *testing.T) {
	mark := newKillMark(t)
	sent := filepath.Join(t.TempDir(), "sent")
	script := "r=$$; " +
		"( while :; do read -r _ _ s _ < /proc/$r/stat; if [ \"$s\" = T ]; then kill -CONT $r; : > " + sent + "; exit; fi; done ) & " +
		"i=0; while [ $i -lt 20 ]; do sleep 30 >/dev/null 2>&1 & i=$((i+1)); done; wait"
	var res Result
	var since time.Duration
	var err error
	execWithin(t, 10*time.Second, func() {
		res, since, err = killRun(t, mark, script, 500*time.Millisecond, 0)
	})
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	if _, serr := os.Stat(sent); serr != nil {
		t.Fatalf("the root was never resumed: %v", serr)
	}
	t.Logf("returned %v after the kill", since)
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("%d marked processes alive at return", len(left))
	}
	if since > killWalkBudget/2 {
		t.Errorf("Exec returned %v after the kill, want <= %v: the walk ran out its budget", since, killWalkBudget/2)
	}
}

// TestHostExecKillRootSignalFailureKillsRoot: when signalling the root
// through its pidfd fails (ENOSYS, as under a seccomp filter), the root is
// still killed, Exec returns promptly, and nothing is left stopped.
func TestHostExecKillRootSignalFailureKillsRoot(t *testing.T) {
	mark := newKillMark(t)
	pidFile := filepath.Join(t.TempDir(), "pid")
	var calls atomic.Int32
	withProcOps(t, func(o *procOps) {
		o.pidfdSignal = func(int, unix.Signal) error { calls.Add(1); return unix.ENOSYS }
	})
	var res Result
	var since time.Duration
	var err error
	execWithin(t, 10*time.Second, func() {
		res, since, err = killRun(t, mark, "echo $$ > "+pidFile+"; exec sleep 30", 500*time.Millisecond, 0)
	})
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	t.Logf("returned %v after the kill", since)
	if since > 2*time.Second {
		t.Errorf("Exec returned %v after the kill, want <= 2s", since)
	}
	b, rerr := os.ReadFile(pidFile)
	pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
	if rerr != nil || pid <= 0 || !pidDead(pid) {
		t.Errorf("root %d (err %v) alive at return", pid, rerr)
	}
	for pid, state := range markedProcs(mark) {
		if state == 'T' || state == 't' {
			t.Errorf("process %d left stopped", pid)
		}
	}
	if calls.Load() == 0 {
		t.Error("the injected signal was never used")
	}
}

// TestHostExecKillSignalsLeavesFirst: the SIGKILLs go out in the reverse
// of the order the walk froze the processes in, so a process is killed
// only after every descendant found below it.
func TestHostExecKillSignalsLeavesFirst(t *testing.T) {
	mark := newKillMark(t)
	var mu sync.Mutex
	var stops, kills []int
	stopped := map[int]bool{}
	withProcOps(t, func(o *procOps) {
		send := o.pidfdSignal
		o.pidfdSignal = func(fd int, sig unix.Signal) error {
			mu.Lock()
			switch sig {
			case unix.SIGSTOP:
				if !stopped[fd] {
					stopped[fd] = true
					stops = append(stops, fd)
				}
			case unix.SIGKILL:
				kills = append(kills, fd)
			}
			mu.Unlock()
			return send(fd, sig)
		}
	})
	res, _, err := killRun(t, mark, "sh -c 'sleep 30 >/dev/null 2>&1 & wait' & wait", 500*time.Millisecond, 0)
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(stops) < 3 {
		t.Fatalf("froze %d processes, want the 3-level tree", len(stops))
	}
	want := make([]int, len(stops))
	for i, fd := range stops {
		want[len(stops)-1-i] = fd
	}
	if !equalInts(kills, want) {
		t.Errorf("SIGKILL order (by pidfd) %v, want the reverse of the freeze order %v", kills, stops)
	}
}

// TestHostExecKillTerminalSignalReachesGroup: the run stays in the caller's
// process group, so a signal to that group reaches it (Ctrl-C).
func TestHostExecKillTerminalSignalReachesGroup(t *testing.T) {
	mark := newKillMark(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	helper := exec.Command(exe, "-test.run=^TestHostExecKillHelperProcess$")
	helper.Env = append(os.Environ(), killHelperEnv+"=hostexec", killHelperRepo+"="+t.TempDir(), killHelperMark+"="+mark)
	helper.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := helper.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = helper.Process.Kill(); _ = helper.Wait() })
	deadline := time.Now().Add(10 * time.Second)
	for len(markedProcs(mark)) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("run process never appeared")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := syscall.Kill(-helper.Process.Pid, syscall.SIGINT); err != nil {
		t.Fatal(err)
	}
	_ = helper.Wait()
	for time.Now().Before(deadline) && len(markedProcs(mark)) != 0 {
		time.Sleep(10 * time.Millisecond)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("run process alive after SIGINT to the caller's group: %v", left)
	}
}

// TestHostExecKillReachesChildOfUnstoppableParent: a descendant whose
// parent cannot be stopped during the kill — python3's posix_spawn parent,
// in uninterruptible sleep while its child blocks opening a FIFO that has
// no writer — is dead when Exec returns.
func TestHostExecKillReachesChildOfUnstoppableParent(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	mark := newKillMark(t)
	fifo := filepath.Join(t.TempDir(), "fifo")
	// Redirected: a surviving child then shows as alive at return rather
	// than holding Exec's stdout open.
	script := `python3 -c 'import os, sys; os.mkfifo(sys.argv[1]); ` +
		`os.posix_spawn("/bin/sleep", ["sleep", "30"], os.environ, ` +
		`file_actions=[(os.POSIX_SPAWN_OPEN, 0, sys.argv[1], os.O_RDONLY, 0)])' ` + fifo + ` >/dev/null 2>&1`
	type result struct {
		res   Result
		since time.Duration
		err   error
	}
	done := make(chan result, 1)
	go func() {
		res, since, err := killRun(t, mark, script, time.Second, 0)
		done <- result{res, since, err}
	}()
	// Wait for the shape: a marked process in D with a marked child.
	parent, child := 0, 0
	for child == 0 {
		select {
		case r := <-done:
			t.Fatalf("Exec returned (%+v, %v) before the spawner blocked", r.res, r.err)
		case <-time.After(5 * time.Millisecond):
		}
		procs := markedProcs(mark)
		for pid, state := range procs {
			if state != 'D' {
				continue
			}
			for c := range procs {
				if ppid, err := procParent(c); err == nil && ppid == pid {
					parent, child = pid, c
				}
			}
		}
	}
	var r result
	select {
	case r = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Exec did not return within 10s")
	}
	if r.err != nil || !r.res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", r.res, r.err)
	}
	t.Logf("spawner %d (D), child %d; returned %v after the kill", parent, child, r.since)
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("marked processes alive at return: %v", left)
	}
	if r.since > 2*time.Second {
		t.Errorf("Exec returned %v after the kill, want <= 2s", r.since)
	}
}

// TestHostExecKillRewalksARootResumedAfterItsRead: when the root is resumed
// right after the read that would confirm the walk, and forks again, the
// walk does not end on that read: the new child is found and killed.
func TestHostExecKillRewalksARootResumedAfterItsRead(t *testing.T) {
	mark := newKillMark(t)
	var mu sync.Mutex
	rootPid, reads := 0, 0
	var resumed atomic.Bool
	withProcOps(t, func(o *procOps) {
		o.children = func(pid int) ([]int, error) {
			kids, err := readProcChildren(pid)
			mu.Lock()
			if rootPid == 0 {
				rootPid = pid
			}
			if pid == rootPid {
				reads++
			}
			second := pid == rootPid && reads == 2
			mu.Unlock()
			if !second || err != nil {
				return kids, err
			}
			// Resume the root and return only once it has started a
			// `sleep 30` this read does not report.
			known := map[int]bool{}
			for _, k := range kids {
				known[k] = true
			}
			_ = unix.Kill(pid, unix.SIGCONT)
			for end := time.Now().Add(time.Second); time.Now().Before(end); time.Sleep(time.Millisecond) {
				now, _ := readProcChildren(pid)
				for _, k := range now {
					cmdline, _ := os.ReadFile("/proc/" + strconv.Itoa(k) + "/cmdline")
					if !known[k] && string(cmdline) == "sleep\x0030\x00" {
						resumed.Store(true)
						return kids, err
					}
				}
			}
			return kids, err
		}
	})
	// The root paces its forks with a busy loop, not a child: a child it
	// waited on would be stopped too, and the resumed root would not fork.
	res, _, err := killRun(t, mark,
		"i=0; while [ $i -lt 200 ]; do sleep 30 >/dev/null 2>&1 & i=$((i+1)); "+
			"j=0; while [ $j -lt 5000 ]; do j=$((j+1)); done; done; wait",
		500*time.Millisecond, 0)
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	if !resumed.Load() {
		t.Fatal("the resumed root never forked a new sleep")
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("%d marked processes alive at return", len(left))
	}
}

// TestHostExecKillRereadsAfterAFailedChildrenRead: a children read that
// fails does not count as a read: when the read of the root's child fails
// twice, the walk reads it again and kills that child's children.
func TestHostExecKillRereadsAfterAFailedChildrenRead(t *testing.T) {
	mark := newKillMark(t)
	var mu sync.Mutex
	var order []int // distinct pids read, in first-read order
	fails := 0
	withProcOps(t, func(o *procOps) {
		o.children = func(pid int) ([]int, error) {
			mu.Lock()
			defer mu.Unlock()
			if !slices.Contains(order, pid) {
				order = append(order, pid)
			}
			if len(order) >= 2 && pid == order[1] && fails < 2 {
				fails++
				return nil, unix.EIO
			}
			return readProcChildren(pid)
		}
	})
	res, _, err := killRun(t, mark,
		"sh -c 'sleep 30 >/dev/null 2>&1 & sleep 30 >/dev/null 2>&1 & wait' & wait",
		500*time.Millisecond, 0)
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	mu.Lock()
	injected := fails
	mu.Unlock()
	if injected != 2 {
		t.Fatalf("injected %d read failures, want 2", injected)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("%d marked processes alive at return", len(left))
	}
}

// processCPU is the user plus system time this process has used.
func processCPU() time.Duration {
	var ru unix.Rusage
	_ = unix.Getrusage(unix.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

// TestHostExecKillBacksOffWhileChildrenReadsFail: when every children read
// fails, the walk runs until its budget ends without spinning: the CPU
// time the process uses from the kill's first read to Exec's return stays
// a small fraction of that wall time.
func TestHostExecKillBacksOffWhileChildrenReadsFail(t *testing.T) {
	mark := newKillMark(t)
	var first sync.Once
	var cpuAtKill time.Duration
	var reads atomic.Int32
	withProcOps(t, func(o *procOps) {
		o.children = func(int) ([]int, error) {
			first.Do(func() { cpuAtKill = processCPU() })
			reads.Add(1)
			return nil, unix.EIO
		}
	})
	var res Result
	var since time.Duration
	var err error
	execWithin(t, 10*time.Second, func() {
		res, since, err = killRun(t, mark, "exec sleep 30", 500*time.Millisecond, 0)
	})
	cpu := processCPU() - cpuAtKill
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	t.Logf("returned %v after the kill; %d children reads; %v CPU from the first read", since, reads.Load(), cpu)
	if reads.Load() == 0 {
		t.Fatal("the injected children source was never used")
	}
	if since > 2*time.Second {
		t.Errorf("Exec returned %v after the kill, want <= 2s", since)
	}
	if cpu > killCPUBound {
		t.Errorf("the kill used %v CPU over a %v walk, want <= %v: it spins", cpu, killWalkBudget, killCPUBound)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("marked processes alive at return: %v", left)
	}
}

// TestHostExecKillRootSIGKILLFailureKillsRoot: when the root's SIGSTOP
// through its pidfd succeeds but its SIGKILL fails (other than the root
// being gone), the root is still killed, Exec returns promptly, and
// nothing is left stopped.
func TestHostExecKillRootSIGKILLFailureKillsRoot(t *testing.T) {
	mark := newKillMark(t)
	pidFile := filepath.Join(t.TempDir(), "pid")
	var rootFd atomic.Int32
	rootFd.Store(-1)
	var injected atomic.Int32
	withProcOps(t, func(o *procOps) {
		send := o.pidfdSignal
		o.pidfdSignal = func(fd int, sig unix.Signal) error {
			// The walk's first signal is the root's SIGSTOP.
			rootFd.CompareAndSwap(-1, int32(fd))
			if sig == unix.SIGKILL && int32(fd) == rootFd.Load() {
				injected.Add(1)
				return unix.EPERM
			}
			return send(fd, sig)
		}
	})
	var res Result
	var since time.Duration
	var err error
	execWithin(t, 10*time.Second, func() {
		res, since, err = killRun(t, mark, "echo $$ > "+pidFile+"; sleep 30 >/dev/null 2>&1 & wait", 500*time.Millisecond, 0)
	})
	if err != nil || !res.TimedOut {
		t.Fatalf("res = %+v err = %v, want a timeout", res, err)
	}
	t.Logf("returned %v after the kill", since)
	if since > 2*time.Second {
		t.Errorf("Exec returned %v after the kill, want <= 2s", since)
	}
	if injected.Load() == 0 {
		t.Fatal("the root's SIGKILL was never attempted")
	}
	b, rerr := os.ReadFile(pidFile)
	pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
	if rerr != nil || pid <= 0 || !pidDead(pid) {
		t.Errorf("root %d (err %v) alive at return", pid, rerr)
	}
	if left := markedProcs(mark); len(left) != 0 {
		t.Errorf("marked processes alive at return: %v", left)
	}
}

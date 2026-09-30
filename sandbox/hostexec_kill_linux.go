//go:build linux

package sandbox

// hostexec_kill_linux.go hides how HostExec kills a run: the whole tree of
// descendants of the command, not only the command itself, without ever
// signalling a process that is not one of them.
//
// A plain kill(pid) on a walked pid cannot promise that: between reading a
// pid out of /proc and signalling it, the process can exit and the pid be
// reused. So the walk holds every process by pidfd — a handle that names
// one process for its whole life — and signals only through pidfds:
//
//  1. The root's pidfd is opened right after Start (afterStart), while the
//     unreaped child still owns its pid.
//  2. The root is SIGSTOPped, which freezes it: a stopped process can
//     neither fork nor be waited on, so its children stay its children.
//  3. For each held process P: wait until every task of P is stopped —
//     re-sending SIGSTOP through P's pidfd while it is not, since a tree
//     member may have sent it SIGCONT — then read the children of every
//     task of P, and count the read only if P's pidfd still reports P alive
//     afterwards. A P that does not stop in time (one in uninterruptible
//     sleep, as a vfork parent is) is read all the same, but that read
//     never lets the walk end. For each candidate: open its pidfd, and only
//     then read its parent from /proc/<pid>/stat; accept it only if that
//     parent is P and P is still alive. An accepted candidate is SIGSTOPped
//     and joins the walk.
//  4. Passes repeat until one finds nothing new, read every live held
//     process while it was stopped, and ends with all of them still
//     stopped; at least two run, the second being the confirming read of
//     the frozen tree. After a pass that found nothing new and had a
//     children read fail, the walk sleeps, twice as long each time up to
//     killReadBackoffMax, so a read that keeps failing does not spin it.
//  5. Every held pidfd is SIGKILLed, leaves first, then waited for (bounded)
//     until it reports the process gone.
//
// Step 5 runs on every exit path of the walk, so a returning kill never
// leaves a process stopped, and a walk error skips only the process it
// concerned. When the SIGSTOP that starts the walk, or the root's SIGKILL,
// fails through the root's pidfd for any reason but the root being gone,
// the root is killed with cmd.Process.Kill instead.

import (
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/unix"
)

const (
	// killWalkBudget bounds the freeze-and-walk phase; when it runs out the
	// walk stops looking and kills what it holds.
	killWalkBudget = time.Second
	// killStopWait bounds how long one process may take to reach a stopped
	// state after its SIGSTOP (a process in uninterruptible sleep does not
	// stop until the syscall returns).
	killStopWait = 250 * time.Millisecond
	// killReapBudget bounds, in total, the wait for every SIGKILLed process
	// to report gone.
	killReapBudget = time.Second
	// killReadBackoffMin and killReadBackoffMax bound the sleep after a
	// pass that had a children read fail and found nothing new.
	killReadBackoffMin = time.Millisecond
	killReadBackoffMax = 50 * time.Millisecond
)

// procOps is the walk's view of the OS: where children come from, how a
// pidfd is opened, and how a signal is sent through one. It is a package
// variable so a hermetic test can inject an unrelated pid, a child that
// appears on a later read, a failing pidfd_open or pidfd_send_signal, or
// record the signals sent.
type procOps struct {
	// children lists the pids the kernel reports as children of pid,
	// across all of pid's tasks.
	children func(pid int) ([]int, error)
	// pidfdOpen opens a pidfd for pid.
	pidfdOpen func(pid int) (int, error)
	// pidfdSignal sends sig to the process behind fd.
	pidfdSignal func(fd int, sig unix.Signal) error
}

var hostProcOps = procOps{
	children:    readProcChildren,
	pidfdOpen:   func(pid int) (int, error) { return unix.PidfdOpen(pid, 0) },
	pidfdSignal: func(fd int, sig unix.Signal) error { return unix.PidfdSendSignal(fd, sig, nil, 0) },
}

// hostKiller is one run's kill state: the root's pidfd (opened in
// afterStart), and the Cancel that walks and kills the tree. Cancel runs on
// exec's watch goroutine, afterStart and close on the supervisor's, so mu
// guards everything.
type hostKiller struct {
	ops procOps

	mu      sync.Mutex
	cmd     *exec.Cmd
	armed   bool // afterStart has run
	pending bool // Cancel fired before afterStart; afterStart kills
	rootPid int
	rootFd  int // -1: no pidfd, fall back to cmd.Process.Kill
}

func newHostKiller() *hostKiller {
	return &hostKiller{ops: hostProcOps, rootFd: -1}
}

// install makes cmd's context-end kill the tree.
func (k *hostKiller) install(cmd *exec.Cmd) {
	k.cmd = cmd
	cmd.Cancel = k.cancel
}

// afterStart takes the root's pidfd while the unreaped child still owns its
// pid; a pidfd_open failure leaves the root to cmd.Process.Kill. It never
// fails the run. A kill that arrived before this point runs now.
func (k *hostKiller) afterStart(cmd *exec.Cmd) error {
	fd, err := k.ops.pidfdOpen(cmd.Process.Pid)
	k.mu.Lock()
	defer k.mu.Unlock()
	if err == nil {
		k.rootPid, k.rootFd = cmd.Process.Pid, fd
	}
	k.armed = true
	if k.pending {
		k.pending = false
		_ = k.killLocked()
	}
	return nil
}

// close releases the root's pidfd once the run is over.
func (k *hostKiller) close() {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.rootFd >= 0 {
		_ = unix.Close(k.rootFd)
		k.rootFd = -1
	}
}

func (k *hostKiller) cancel() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if !k.armed {
		k.pending = true
		return nil
	}
	return k.killLocked()
}

func (k *hostKiller) killLocked() error {
	if k.rootFd < 0 {
		return k.cmd.Process.Kill()
	}
	killTree(k.rootPid, k.rootFd, k.ops, k.cmd.Process.Kill)
	return nil
}

type heldProc struct {
	pid int
	fd  int
}

// killTree freezes and kills root and every descendant it can verify.
// killRoot kills the root without its pidfd; it runs when the SIGSTOP or the
// SIGKILL sent to the root through rootFd fails with an error other than
// ESRCH (root gone). The SIGSTOP re-sent in waitStopped does not fall back.
// killTree does not close rootFd.
func killTree(rootPid, rootFd int, ops procOps, killRoot func() error) {
	if !pidfdAlive(rootFd) {
		return
	}
	if !signalRoot(rootFd, unix.SIGSTOP, ops, killRoot) {
		return
	}
	held := []heldProc{{rootPid, rootFd}}
	defer func() { killHeld(held, rootFd, ops, killRoot) }()

	seen := map[int]bool{rootPid: true}
	deadline := time.Now().Add(killWalkBudget)
	backoff := killReadBackoffMin
	for pass := 1; time.Now().Before(deadline); pass++ {
		// settled: every live held process was read while stopped.
		found, settled, readFailed := false, true, false
		// held grows while the pass runs: a process accepted here is
		// already stopped and is walked in the same pass.
		for i := 0; i < len(held); i++ {
			p := held[i]
			if !pidfdAlive(p.fd) {
				// Exited: its children are no longer its own.
				continue
			}
			// One that does not stop (uninterruptible sleep) is read
			// anyway: its children can still be held and stopped.
			if !waitStopped(p, deadline, ops) {
				settled = false
			}
			kids, err := ops.children(p.pid)
			if err != nil {
				settled, readFailed = false, true
				continue
			}
			if !pidfdAlive(p.fd) {
				continue
			}
			for _, c := range kids {
				if seen[c] {
					continue
				}
				fd, err := ops.pidfdOpen(c)
				if err != nil {
					continue
				}
				ppid, err := procParent(c)
				if err != nil || ppid != p.pid || !pidfdAlive(p.fd) {
					_ = unix.Close(fd)
					continue
				}
				seen[c] = true
				if err := ops.pidfdSignal(fd, unix.SIGSTOP); err != nil {
					_ = unix.Close(fd)
					continue
				}
				held = append(held, heldProc{c, fd})
				found = true
			}
		}
		if !found && settled && pass >= 2 && heldStillStopped(held) {
			break
		}
		if readFailed && !found {
			time.Sleep(min(backoff, time.Until(deadline)))
			backoff = min(2*backoff, killReadBackoffMax)
		}
	}
}

// signalRoot sends sig to the root through its pidfd. When that fails for
// any reason but the root being gone, it kills the root with killRoot
// instead. It reports whether sig was sent.
func signalRoot(rootFd int, sig unix.Signal, ops procOps, killRoot func() error) bool {
	err := ops.pidfdSignal(rootFd, sig)
	if err != nil && !errors.Is(err, unix.ESRCH) {
		_ = killRoot()
	}
	return err == nil
}

// heldStillStopped reports whether every live held process is still
// stopped: one resumed since its read may have forked since.
func heldStillStopped(held []heldProc) bool {
	for _, h := range held {
		if _, ok := allStopped(h.pid); !ok && pidfdAlive(h.fd) {
			return false
		}
	}
	return true
}

// killHeld SIGKILLs every held process, last discovered first, waits for
// each to report gone, and closes every fd but the root's. The root is
// signalled through signalRoot.
func killHeld(held []heldProc, rootFd int, ops procOps, killRoot func() error) {
	for i := len(held) - 1; i >= 0; i-- {
		if held[i].fd == rootFd {
			signalRoot(rootFd, unix.SIGKILL, ops, killRoot)
		} else {
			_ = ops.pidfdSignal(held[i].fd, unix.SIGKILL)
		}
	}
	deadline := time.Now().Add(killReapBudget)
	for i := len(held) - 1; i >= 0; i-- {
		waitPidfdGone(held[i].fd, deadline)
	}
	for _, h := range held {
		if h.fd != rootFd {
			_ = unix.Close(h.fd)
		}
	}
}

// pidfdAlive reports whether the process behind fd has not exited (a
// zombie has). A poll failure reads as not alive: the caller then trusts
// nothing it read.
func pidfdAlive(fd int) bool {
	pfd := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
	for {
		n, err := unix.Poll(pfd, 0)
		if errors.Is(err, unix.EINTR) {
			continue
		}
		return err == nil && n == 0
	}
}

func waitPidfdGone(fd int, deadline time.Time) {
	pfd := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
	for {
		ms := int(time.Until(deadline) / time.Millisecond)
		if ms < 0 {
			ms = 0
		}
		if _, err := unix.Poll(pfd, ms); !errors.Is(err, unix.EINTR) {
			return
		}
	}
}

// waitStopped waits until every task of p is in a stopped state (or dead),
// the task list being the same on two consecutive reads, so no thread of p
// can still run or create another. While p is not stopped it re-sends
// SIGSTOP through p's pidfd: a SIGCONT from another tree member resumes a
// frozen process. It reports false on timeout or when p is gone.
func waitStopped(p heldProc, deadline time.Time, ops procOps) bool {
	if limit := time.Now().Add(killStopWait); limit.Before(deadline) {
		deadline = limit
	}
	for {
		tids, ok := allStopped(p.pid)
		if ok {
			if again, err := procTasks(p.pid); err == nil && equalInts(tids, again) {
				return true
			}
		} else {
			_ = ops.pidfdSignal(p.fd, unix.SIGSTOP)
		}
		if !time.Now().Before(deadline) {
			return false
		}
		time.Sleep(200 * time.Microsecond)
	}
}

func allStopped(pid int) ([]int, bool) {
	tids, err := procTasks(pid)
	if err != nil {
		return nil, false
	}
	for _, tid := range tids {
		state, _, err := readProcStat("/proc/" + strconv.Itoa(pid) + "/task/" + strconv.Itoa(tid) + "/stat")
		if err != nil {
			return nil, false
		}
		switch state {
		case 'T', 't', 'Z', 'X':
		default:
			return nil, false
		}
	}
	return tids, true
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func procTasks(pid int) ([]int, error) {
	ents, err := os.ReadDir("/proc/" + strconv.Itoa(pid) + "/task")
	if err != nil {
		return nil, err
	}
	tids := make([]int, 0, len(ents))
	for _, e := range ents {
		if tid, err := strconv.Atoi(e.Name()); err == nil {
			tids = append(tids, tid)
		}
	}
	return tids, nil
}

// readProcChildren returns the children of every task of pid. A task that
// exits between the listing and its read is skipped: its children belong to
// the remaining tasks.
func readProcChildren(pid int) ([]int, error) {
	tids, err := procTasks(pid)
	if err != nil {
		return nil, err
	}
	var kids []int
	for _, tid := range tids {
		b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/task/" + strconv.Itoa(tid) + "/children")
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, err
		}
		for _, f := range strings.Fields(string(b)) {
			if c, err := strconv.Atoi(f); err == nil {
				kids = append(kids, c)
			}
		}
	}
	return kids, nil
}

func procParent(pid int) (int, error) {
	_, ppid, err := readProcStat("/proc/" + strconv.Itoa(pid) + "/stat")
	return ppid, err
}

// readProcStat returns the state letter and parent pid from a stat file.
// The command name (field 2) may contain spaces and parentheses, so fields
// are counted from the LAST ')'.
func readProcStat(path string) (state byte, ppid int, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, err
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 || i+2 >= len(s) {
		return 0, 0, errors.New("sandbox: malformed proc stat")
	}
	f := strings.Fields(s[i+1:])
	if len(f) < 2 || len(f[0]) != 1 {
		return 0, 0, errors.New("sandbox: malformed proc stat")
	}
	ppid, err = strconv.Atoi(f[1])
	return f[0][0], ppid, err
}

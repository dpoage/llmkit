package sqlite_test

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/store/sqlite"
)

const helperEnv = "LLMKIT_STORE_TEST_HELPER"

// TestMain turns the test binary into the writer process of the two-process
// leg when helperEnv is set.
func TestMain(m *testing.M) {
	if os.Getenv(helperEnv) == "1" {
		os.Exit(helperWriter())
	}
	os.Exit(m.Run())
}

// helperWriter opens the database named by os.Args[1], waits for a line on
// stdin, then writes os.Args[3] events for run os.Args[2].
func helperWriter() int {
	path, run := os.Args[1], llmkit.RunID(os.Args[2])
	n, _ := strconv.Atoi(os.Args[3])
	if _, err := bufio.NewReader(os.Stdin).ReadString('\n'); err != nil {
		fmt.Println("no start signal:", err)
		return 2
	}
	var mu sync.Mutex
	var errs []error
	s, err := sqlite.Open(path, func(err error) {
		mu.Lock()
		errs = append(errs, err)
		mu.Unlock()
	})
	if err != nil {
		fmt.Println("Open:", err)
		return 1
	}
	ctx := context.Background()
	start := newEv(run, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "two-process"}
	s.Observe(ctx, start)
	for i := 1; i < n; i++ {
		e := newEv(run, llmkit.KindToolRun)
		e.Step = i
		e.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c" + strconv.Itoa(i), Name: "echo", Arguments: []byte(`{}`)}, Result: "r"}
		s.Observe(ctx, e)
	}
	if err := s.Close(); err != nil {
		fmt.Println("Close:", err)
		return 1
	}
	mu.Lock()
	defer mu.Unlock()
	if len(errs) > 0 {
		fmt.Printf("onErr called %d times, first: %v\n", len(errs), errs[0])
		return 1
	}
	return 0
}

// prefixChecker wraps a Store and, after each Observe returns, requires the
// run's stored events to be exactly the prefix observed so far.
type prefixChecker struct {
	t     *testing.T
	store *sqlite.Store

	mu       sync.Mutex
	observed map[llmkit.RunID][]llmkit.Event
}

func (p *prefixChecker) Observe(ctx context.Context, ev llmkit.Event) {
	p.store.Observe(ctx, ev)
	p.mu.Lock()
	p.observed[ev.RunID] = append(p.observed[ev.RunID], ev)
	want := append([]llmkit.Event(nil), p.observed[ev.RunID]...)
	p.mu.Unlock()
	got, err := p.store.Events(ctx, ev.RunID)
	if err != nil {
		p.t.Errorf("run %s after %s event %d: Events: %v", ev.RunID, ev.Kind, len(want), err)
		return
	}
	if len(got) != len(want) {
		p.t.Errorf("run %s after %s event %d: Events returned %d events, want the %d-event prefix", ev.RunID, ev.Kind, len(want), len(got), len(want))
		return
	}
	for i := range want {
		if g, w := mustJSON(p.t, got[i]), canon(p.t, want[i]); g != w {
			p.t.Errorf("run %s after %s event %d: event %d is not the observed one", ev.RunID, ev.Kind, len(want), i)
			return
		}
	}
}

func TestP2_ConcurrentRunsSeePrefixAfterEveryObserve(t *testing.T) {
	const n = 10
	s, log, _ := openStore(t)
	chk := &prefixChecker{t: t, store: s, observed: map[llmkit.RunID][]llmkit.Event{}}
	r := agent.NewRunner(loopClient(4), []agent.Tool{markerTool()}, "sys", agent.WithObserver(chk))

	var wg sync.WaitGroup
	ids := make([]llmkit.RunID, n)
	for i := range n {
		ids[i] = llmkit.RunID(fmt.Sprintf("run-%02d", i))
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := r.Run(context.Background(), fmt.Sprintf("task %d", i), agent.WithRunID(ids[i])); err != nil {
				t.Errorf("run %d: %v", i, err)
			}
		}()
	}
	wg.Wait()
	noErrors(t, log)

	for _, id := range ids {
		got, err := s.Events(context.Background(), id)
		if err != nil {
			t.Fatalf("Events(%s): %v", id, err)
		}
		if got[0].Kind != llmkit.KindStart || got[len(got)-1].Kind != llmkit.KindFinalize {
			t.Errorf("run %s is not complete: first %s, last %s", id, got[0].Kind, got[len(got)-1].Kind)
		}
		if want := len(chk.observed[id]); len(got) != want {
			t.Errorf("run %s holds %d events, observed %d", id, len(got), want)
		}
	}
}

func mustJSON(t testing.TB, ev llmkit.Event) string {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Errorf("marshal: %v", err)
	}
	return string(b)
}

// TestP2_TwoProcessesOpenOneNewPath starts two writer processes together on
// a path that does not exist yet.
func TestP2_TwoProcessesOpenOneNewPath(t *testing.T) {
	const (
		trials  = 30
		perProc = 300
	)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for trial := range trials {
		path := filepath.Join(t.TempDir(), "shared.db")
		type proc struct {
			cmd   *exec.Cmd
			stdin io.WriteCloser
			out   *safeBuf
		}
		procs := make([]proc, 2)
		for i := range procs {
			cmd := exec.Command(exe, path, fmt.Sprintf("proc-%d", i), strconv.Itoa(perProc))
			cmd.Env = append(os.Environ(), helperEnv+"=1")
			out := &safeBuf{}
			cmd.Stdout, cmd.Stderr = out, out
			stdin, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			procs[i] = proc{cmd, stdin, out}
		}
		for _, p := range procs {
			_, _ = io.WriteString(p.stdin, "go\n")
			_ = p.stdin.Close()
		}
		for i, p := range procs {
			if err := p.cmd.Wait(); err != nil {
				t.Fatalf("trial %d: process %d failed: %v\n%s", trial, i, err, p.out.String())
			}
		}
		checkTwoProcessFile(t, trial, path, perProc)
	}
}

// checkTwoProcessFile requires both processes' runs to be stored whole and in order.
func checkTwoProcessFile(t *testing.T, trial int, path string, perProc int) {
	t.Helper()
	s, err := sqlite.Open(path, nil)
	if err != nil {
		t.Fatalf("trial %d: reopen: %v", trial, err)
	}
	defer func() { _ = s.Close() }()
	total := 0
	for i := range 2 {
		evs, err := s.Events(context.Background(), llmkit.RunID(fmt.Sprintf("proc-%d", i)))
		if err != nil {
			t.Fatalf("trial %d: Events(proc-%d): %v", trial, i, err)
		}
		total += len(evs)
		for j, ev := range evs {
			if ev.Step != j {
				t.Fatalf("trial %d: proc-%d event %d has step %d", trial, i, j, ev.Step)
			}
		}
	}
	if total != 2*perProc {
		t.Fatalf("trial %d: %d events stored, want %d", trial, total, 2*perProc)
	}
}

type safeBuf struct {
	mu sync.Mutex
	b  []byte
}

func (s *safeBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.b = append(s.b, p...)
	return len(p), nil
}

func (s *safeBuf) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return string(s.b)
}

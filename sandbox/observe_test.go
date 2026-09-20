package sandbox_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/sandbox"
)

// execLog is an llmkit.Observer collecting events for assertions. It is
// safe for concurrent use.
type execLog struct {
	mu  sync.Mutex
	evs []llmkit.Event
}

func (l *execLog) Observe(_ context.Context, ev llmkit.Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.evs = append(l.evs, ev)
}

func (l *execLog) events() []llmkit.Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]llmkit.Event(nil), l.evs...)
}

func TestObserveExecEmitsOneEvent(t *testing.T) {
	log := &execLog{}
	m := sandbox.NewMock(sandbox.MockResponse{
		Result: sandbox.Result{
			ExitCode: 0,
			Stdout:   "hello\n",
			Stderr:   "warn\n",
			Duration: 5 * time.Millisecond,
		},
	})
	sb := sandbox.Observe(m, log)

	runID := llmkit.NewRunID()
	ctx := llmkit.WithRun(context.Background(), runID)

	res, err := sb.Exec(ctx, sandbox.Spec{Cmd: []string{"echo", "hello"}})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 0 || res.Stdout != "hello\n" {
		t.Fatalf("res = %+v, want the scripted result", res)
	}

	evs := log.events()
	if len(evs) != 1 {
		t.Fatalf("got %d events, want 1", len(evs))
	}
	ev := evs[0]
	if ev.Kind != llmkit.KindExec {
		t.Errorf("Kind = %q, want %q", ev.Kind, llmkit.KindExec)
	}
	if ev.RunID != runID {
		t.Errorf("RunID = %q, want %q", ev.RunID, runID)
	}
	if ev.SchemaVersion != llmkit.EventSchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", ev.SchemaVersion, llmkit.EventSchemaVersion)
	}
	x := ev.Exec
	if x == nil {
		t.Fatalf("Exec payload is nil")
	}
	if x.Backend != "mock" {
		t.Errorf("Backend = %q, want %q", x.Backend, "mock")
	}
	if len(x.Command) != 2 || x.Command[0] != "echo" || x.Command[1] != "hello" {
		t.Errorf("Command = %v, want [echo hello]", x.Command)
	}
	if x.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", x.ExitCode)
	}
	if x.StdoutBytes != int64(len("hello\n")) {
		t.Errorf("StdoutBytes = %d, want %d", x.StdoutBytes, len("hello\n"))
	}
	if x.StderrBytes != int64(len("warn\n")) {
		t.Errorf("StderrBytes = %d, want %d", x.StderrBytes, len("warn\n"))
	}
	if x.Truncated {
		t.Errorf("Truncated = true, want false")
	}
	if ev.Duration != 5*time.Millisecond {
		t.Errorf("Duration = %v, want the Result's 5ms", ev.Duration)
	}
	if x.Err != "" {
		t.Errorf("Err = %q, want empty on success", x.Err)
	}
}

func TestObserveNonZeroExitIsNotAnError(t *testing.T) {
	log := &execLog{}
	m := sandbox.NewMock(sandbox.MockResponse{Result: sandbox.Result{ExitCode: 7, Stdout: "boom"}})
	sb := sandbox.Observe(m, log)

	res, err := sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"false"}})
	if err != nil {
		t.Fatalf("Exec err = %v, want nil (a non-zero exit is the command's verdict)", err)
	}
	if res.ExitCode != 7 {
		t.Fatalf("ExitCode = %d, want 7", res.ExitCode)
	}

	x := log.events()[0].Exec
	if x.ExitCode != 7 {
		t.Errorf("ExitCode = %d, want 7", x.ExitCode)
	}
	if x.Err != "" {
		t.Errorf("Err = %q, want empty — a non-zero exit is not an infrastructure error", x.Err)
	}
}

func TestObserveInfraErrorStillEmits(t *testing.T) {
	boom := errors.New("boom")
	log := &execLog{}
	m := sandbox.NewMock(sandbox.MockResponse{})
	m.EnqueueResponse(sandbox.MockResponse{Err: boom})
	sb := sandbox.Observe(m, log)

	res, err := sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}})
	if !errors.Is(err, boom) {
		t.Fatalf("Exec err = %v, want the scripted error", err)
	}
	if res.ExitCode != 0 || res.Stdout != "" {
		t.Fatalf("res = %+v, want the zero Result on infrastructure failure", res)
	}

	evs := log.events()
	if len(evs) != 1 {
		t.Fatalf("got %d events, want 1 (a failed call still emits)", len(evs))
	}
	ev := evs[0]
	x := ev.Exec
	if x.ExitCode != -1 {
		t.Errorf("ExitCode = %d, want -1 (the process never ran to an exit)", x.ExitCode)
	}
	if x.Err != boom.Error() {
		t.Errorf("Err = %q, want %q", x.Err, boom.Error())
	}
	if ev.Duration != 0 {
		t.Errorf("Duration = %v, want 0 (no Result measured one)", ev.Duration)
	}
	if x.StdoutBytes != 0 || x.StderrBytes != 0 {
		t.Errorf("bytes = %d/%d, want 0/0", x.StdoutBytes, x.StderrBytes)
	}
}

func TestObserveTruncatedReflectsEitherStream(t *testing.T) {
	cases := []struct {
		name string
		res  sandbox.Result
		want bool
	}{
		{"neither", sandbox.Result{ExitCode: 0}, false},
		{"stdout", sandbox.Result{ExitCode: 0, StdoutTruncated: true}, true},
		{"stderr", sandbox.Result{ExitCode: 0, StderrTruncated: true}, true},
		{"both", sandbox.Result{ExitCode: 0, StdoutTruncated: true, StderrTruncated: true}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			log := &execLog{}
			sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{Result: tc.res}), log)
			if _, err := sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}}); err != nil {
				t.Fatalf("Exec: %v", err)
			}
			if got := log.events()[0].Exec.Truncated; got != tc.want {
				t.Errorf("Truncated = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestObserveWatchdogKillCarriesMinusOne(t *testing.T) {
	// A watchdog kill returns a Result (no error) whose ExitCode is -1; the
	// event must carry that -1 so it never reads as a real exit status.
	log := &execLog{}
	sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{Result: sandbox.Result{ExitCode: -1, TimedOut: true}}), log)

	res, err := sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !res.InfraKilled() {
		t.Fatalf("InfraKilled = false, want true")
	}
	if got := log.events()[0].Exec.ExitCode; got != -1 {
		t.Errorf("ExitCode = %d, want -1", got)
	}
}

func TestObserveMaterializeWorkspacePassesThrough(t *testing.T) {
	log := &execLog{}
	sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{}), log)

	got, err := sb.MaterializeWorkspace("/repo")
	if err != nil || got != "/repo" {
		t.Fatalf("MaterializeWorkspace = %q, %v; want /repo, nil", got, err)
	}
	if n := len(log.events()); n != 0 {
		t.Errorf("got %d events from MaterializeWorkspace, want 0", n)
	}
}

func TestObserveNilObserverReturnsInput(t *testing.T) {
	m := sandbox.NewMock(sandbox.MockResponse{})
	if got := sandbox.Observe(m, nil); got != sandbox.Sandbox(m) {
		t.Fatalf("Observe(m, nil) returned a new wrapper, want m itself")
	}
}

func TestObserveObserverPanicPropagates(t *testing.T) {
	sb := sandbox.Observe(
		sandbox.NewMock(sandbox.MockResponse{}),
		llmkit.ObserverFunc(func(context.Context, llmkit.Event) { panic("observer bug") }),
	)

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Exec recovered an observer panic, want it to propagate")
		}
	}()
	_, _ = sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}})
}

func TestObserveConcurrentExecs(t *testing.T) {
	log := &execLog{}
	sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{Result: sandbox.Result{ExitCode: 0}}), log)

	ctx := llmkit.WithRun(context.Background(), llmkit.NewRunID())
	const n = 32
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := sb.Exec(ctx, sandbox.Spec{Cmd: []string{"cmd"}}); err != nil {
				t.Errorf("Exec: %v", err)
			}
		}()
	}
	wg.Wait()

	if got := len(log.events()); got != n {
		t.Errorf("got %d events, want %d", got, n)
	}
}

// ExampleObserve wraps a Mock backend with an observer and reads the one
// Exec event back. Real use points Observe at NewCLI, NewBwrap, or
// NewHostExec; the Mock keeps the example hermetic.
func ExampleObserve() {
	log := &execLog{}
	sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{
		Result: sandbox.Result{ExitCode: 0, Stdout: "hello\n"},
	}), log)

	_, _ = sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"echo", "hello"}})

	for _, ev := range log.events() {
		x := ev.Exec
		fmt.Printf("%s backend=%s cmd=%v exit=%d stdout=%dB truncated=%v\n",
			ev.Kind, x.Backend, x.Command, x.ExitCode, x.StdoutBytes, x.Truncated)
	}
	// Output:
	// exec backend=mock cmd=[echo hello] exit=0 stdout=6B truncated=false
}

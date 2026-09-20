package sandbox_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
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

func TestObserveCommandIsCopiedNotAliased(t *testing.T) {
	log := &execLog{}
	sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{}), log)

	cmd := []string{"go", "test", "./a"}
	_, _ = sb.Exec(context.Background(), sandbox.Spec{Cmd: cmd})

	// A sink retains events past the call; the caller owns the argv and
	// may reuse its backing array. The event must hold its own copy.
	cmd[2] = "./MUTATED"
	x := log.events()[0].Exec
	if len(x.Command) != 3 || x.Command[2] != "./a" {
		t.Errorf("Command = %v, want the argv at Exec time [go test ./a]", x.Command)
	}
}

func TestObservePopulatedResultWithErrorPassesThrough(t *testing.T) {
	boom := errors.New("boom")
	want := sandbox.Result{
		ExitCode:          -1,
		Stdout:            "partial output",
		Stderr:            "warn",
		Duration:          7 * time.Millisecond,
		StdoutTruncated:   true,
		WorkspaceCacheHit: true,
	}
	log := &execLog{}
	m := sandbox.NewMock(sandbox.MockResponse{})
	m.EnqueueResponse(sandbox.MockResponse{Result: want, Err: boom})
	sb := sandbox.Observe(m, log)

	res, err := sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}})
	if err != boom {
		t.Fatalf("err = %v, want the identical scripted error", err)
	}
	if !reflect.DeepEqual(res, want) {
		t.Fatalf("res = %+v, want the scripted Result verbatim alongside the error", res)
	}

	ev := log.events()[0]
	x := ev.Exec
	if x.ExitCode != -1 {
		t.Errorf("ExitCode = %d, want -1 (an infrastructure error, whatever the Result carried)", x.ExitCode)
	}
	if x.StdoutBytes != int64(len("partial output")) || x.StderrBytes != int64(len("warn")) {
		t.Errorf("bytes = %d/%d, want the populated Result's capture sizes", x.StdoutBytes, x.StderrBytes)
	}
	if ev.Duration != 7*time.Millisecond {
		t.Errorf("Duration = %v, want the populated Result's 7ms", ev.Duration)
	}
	if x.Err != boom.Error() {
		t.Errorf("Err = %q, want %q", x.Err, boom.Error())
	}
}

func TestObserveHostExecRefusalJoinsBackendName(t *testing.T) {
	log := &execLog{}
	sb := sandbox.Observe(sandbox.NewHostExec(), log)

	_, err := sb.Exec(context.Background(), sandbox.Spec{
		RepoDir: t.TempDir(),
		Cmd:     []string{"true"},
		Image:   "alpine",
	})
	var use *sandbox.UnsupportedSpecError
	if !errors.As(err, &use) {
		t.Fatalf("err = %v, want *UnsupportedSpecError", err)
	}
	if use.Backend != "host" {
		t.Fatalf("UnsupportedSpecError.Backend = %q, want host", use.Backend)
	}

	x := log.events()[0].Exec
	if x.Backend != use.Backend {
		t.Errorf("event Backend = %q, want %q (the same name the real refusal used)", x.Backend, use.Backend)
	}
}

func TestObserveUnknownBackendNamedByGoType(t *testing.T) {
	log := &execLog{}
	sb := sandbox.Observe(customSandbox{}, log)

	_, _ = sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}})
	if got := log.events()[0].Exec.Backend; got != "sandbox_test.customSandbox" {
		t.Errorf("Backend = %q, want the Go type name", got)
	}
}

func TestObserveNestedWrapperReportsInnerBackend(t *testing.T) {
	log := &execLog{}
	sb := sandbox.Observe(sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{}), log), log)

	_, _ = sb.Exec(context.Background(), sandbox.Spec{Cmd: []string{"cmd"}})

	// One event per layer: [0] is the inner wrapper's (its Backend is
	// trivially "mock"); [1] is the OUTER wrapper's, which must report
	// through the nested case instead of leaking its own Go type.
	evs := log.events()
	if len(evs) != 2 {
		t.Fatalf("got %d events, want 2 (one per wrapper layer)", len(evs))
	}
	if got := evs[1].Exec.Backend; got != "mock" {
		t.Errorf("outer wrapper's Backend = %q, want mock (a wrapped backend reports its inner backend)", got)
	}
}

func TestObservePassesCallerCtxToObserver(t *testing.T) {
	type marker struct{}
	var got any
	obs := llmkit.ObserverFunc(func(ctx context.Context, _ llmkit.Event) {
		got = ctx.Value(marker{})
	})
	sb := sandbox.Observe(sandbox.NewMock(sandbox.MockResponse{}), obs)

	ctx := context.WithValue(context.Background(), marker{}, "present")
	_, _ = sb.Exec(ctx, sandbox.Spec{Cmd: []string{"cmd"}})
	if got != "present" {
		t.Errorf("observer saw ctx value %v, want \"present\" — the observer must receive the caller's context", got)
	}
}

// customSandbox is an out-of-package Sandbox implementation, so the
// backend name falls back to the Go type.
type customSandbox struct{}

func (customSandbox) Exec(context.Context, sandbox.Spec) (sandbox.Result, error) {
	return sandbox.Result{ExitCode: 0}, nil
}

func (customSandbox) MaterializeWorkspace(repoDir string) (string, error) {
	return repoDir, nil
}

package sqlite_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/store/sqlite"
)

func TestSource_UnknownRunWrapsErrUnknownRun(t *testing.T) {
	s, _, _ := openStore(t)
	s.Observe(context.Background(), startEvent("known"))
	for _, run := range []llmkit.RunID{"missing", ""} {
		if _, err := s.Events(context.Background(), run); !errors.Is(err, llmkit.ErrUnknownRun) {
			t.Errorf("Events(%q) = %v, want an error wrapping ErrUnknownRun", run, err)
		}
	}
}

func TestSource_TwoStartsForOneRunIsAnError(t *testing.T) {
	s, log, _ := openStore(t)
	s.Observe(context.Background(), startEvent("twice"))
	s.Observe(context.Background(), startEvent("twice"))
	noErrors(t, log)
	if _, err := s.Events(context.Background(), "twice"); !errors.Is(err, llmkit.ErrUnknownRun) {
		t.Fatalf("Events over two Start events = %v, want an error wrapping ErrUnknownRun", err)
	}
}

// Every event is stored: no Start, after Finalize, empty run id.
func TestSource_AdmitsEveryEvent(t *testing.T) {
	s, log, path := openStore(t)
	ctx := context.Background()
	noStart := newEv("no-start", llmkit.KindToolRun)
	noStart.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c", Name: "n"}}
	start := startEvent("fin")
	fin := newEv("fin", llmkit.KindFinalize)
	fin.Finalize = &llmkit.FinalizeEvent{Status: llmkit.RunCompleted}
	late := newEv("fin", llmkit.KindSteer)
	late.Steer = &llmkit.SteerEvent{Message: llmkit.TextMessage(llmkit.RoleUser, "late")}
	anon := newEv("", llmkit.KindExec)
	anon.Exec = &llmkit.ExecEvent{Backend: "host", Command: []string{"true"}}
	for _, e := range []llmkit.Event{noStart, start, fin, late, anon} {
		s.Observe(ctx, e)
	}
	noErrors(t, log)

	got, err := s.Events(ctx, "fin")
	if err != nil {
		t.Fatal(err)
	}
	assertIdentical(t, got, []llmkit.Event{start, fin, late})
	got, err = s.Events(ctx, "no-start")
	if err != nil {
		t.Fatal(err)
	}
	assertIdentical(t, got, []llmkit.Event{noStart})
	if _, err := s.Events(ctx, ""); !errors.Is(err, llmkit.ErrUnknownRun) {
		t.Fatalf("Events of the empty run id = %v", err)
	}
	if n := rowCount(t, openRaw(t, path), "events"); n != 5 {
		t.Fatalf("%d events stored, want 5 (the empty-run event included)", n)
	}
}

// The Runner's Finalize arrives on a canceled context when the run was canceled.
func TestObserve_CanceledContextStillWrites(t *testing.T) {
	s, log, _ := openStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.Observe(ctx, startEvent("canceled"))
	noErrors(t, log)
	if _, err := s.Events(context.Background(), "canceled"); err != nil {
		t.Fatal(err)
	}
}

func TestRunner_CanceledRunIsFullyRecorded(t *testing.T) {
	s, log, _ := openStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	tool := fnTool{name: "echo", run: func(context.Context, json.RawMessage) ([]llmkit.Block, error) {
		cancel()
		return []llmkit.Block{llmkit.Text("done")}, nil
	}}
	r := agent.NewRunner(loopClient(3), []agent.Tool{tool}, "sys", agent.WithObserver(s))
	out, _ := r.Run(ctx, "cancel me")
	noErrors(t, log)
	got, err := s.Events(context.Background(), out.Transcript.Record[0].RunID)
	if err != nil {
		t.Fatal(err)
	}
	assertIdentical(t, got, out.Transcript.Record)
	if last := got[len(got)-1]; last.Kind != llmkit.KindFinalize || last.Finalize.Status != llmkit.RunCanceled {
		t.Fatalf("last event %s %+v, want a canceled Finalize", last.Kind, last.Finalize)
	}
}

func TestEvents_CallerOwnsTheResult(t *testing.T) {
	s, _, _ := openStore(t)
	s.Observe(context.Background(), startEvent("own"))
	a, _ := s.Events(context.Background(), "own")
	a[0].Start.Task = "mutated"
	b, _ := s.Events(context.Background(), "own")
	if b[0].Start.Task != "t" {
		t.Fatalf("a second Events saw the caller's mutation: %q", b[0].Start.Task)
	}
}

func TestOpen_PathWithURICharacters(t *testing.T) {
	for _, name := range []string{"with space.db", "q?mark.db", "hash#tag.db", "pct%41.db"} {
		path := filepath.Join(t.TempDir(), name)
		s, err := sqlite.Open(path, nil)
		if err != nil {
			t.Fatalf("Open(%q): %v", name, err)
		}
		s.Observe(context.Background(), startEvent("uri"))
		if _, err := s.Events(context.Background(), "uri"); err != nil {
			t.Fatalf("%q: %v", name, err)
		}
		_ = s.Close()
		if _, err := sqlite.Open(path, nil); err != nil {
			t.Fatalf("re-Open(%q): %v", name, err)
		}
	}
}

func TestOpen_MissingDirectoryFails(t *testing.T) {
	if s, err := sqlite.Open(filepath.Join(t.TempDir(), "no", "such", "dir.db"), nil); err == nil {
		_ = s.Close()
		t.Fatal("Open under a missing directory succeeded")
	}
}

// Hand-built events need not be well formed; the store keeps them as they are.
func TestSource_StoresMalformedEventsAsGiven(t *testing.T) {
	s, log, path := openStore(t)
	const run = "run-odd"
	req := llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "one"), llmkit.TextMessage(llmkit.RoleUser, "two")}}

	both := newEv(run, llmkit.KindCompletion)
	both.Completion = &llmkit.CompletionEvent{Request: req, Response: llmkit.Response{Text: "c"}}
	both.Attempt = &llmkit.AttemptEvent{Attempt: 1, Request: req}
	every := newEv(run, llmkit.KindStart)
	every.Start = &llmkit.StartEvent{Task: "t"}
	every.Steer = &llmkit.SteerEvent{Message: llmkit.TextMessage(llmkit.RoleUser, "s")}
	every.Embed = &llmkit.EmbedEvent{Model: "m"}
	noKind := llmkit.Event{RunID: run}
	zeroTime := newEv(run, llmkit.KindSteer)
	zeroTime.Time = time.Time{}
	zeroTime.Steer = &llmkit.SteerEvent{}
	far := newEv(run, llmkit.KindSteer)
	far.Time = time.Date(9999, 12, 31, 23, 59, 59, 999999999, time.UTC)
	far.Steer = &llmkit.SteerEvent{}

	evs := []llmkit.Event{both, every, noKind, zeroTime, far}
	observeAll(s, evs)
	noErrors(t, log)
	got, err := s.Events(context.Background(), run)
	if err != nil {
		t.Fatal(err)
	}
	assertIdentical(t, got, evs)

	rows, err := openRaw(t, path).Query("SELECT time_ns FROM events ORDER BY id")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	var ns []int64
	for rows.Next() {
		var n int64
		_ = rows.Scan(&n)
		ns = append(ns, n)
	}
	if ns[3] != math.MinInt64 || ns[4] != math.MaxInt64 {
		t.Fatalf("time_ns of the year-1 and year-9999 events = %d, %d; want them saturated", ns[3], ns[4])
	}
}

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// composeLog is one ordered log shared by every registration in a test:
// entries are "<registration>:<fire point>", in the order the callbacks ran.
type composeLog struct {
	mu      sync.Mutex
	entries []string
}

func (l *composeLog) add(s string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries = append(l.entries, s)
}

// composeHooks builds one registration that logs every fire point except the
// ones in nilFields, which stay nil. Delta is never set here: a delta stream's
// length is the client's business, so it is pinned separately.
func composeHooks(log *composeLog, idx int, nilFields ...string) Hooks {
	tag := func(name string) string { return fmt.Sprintf("%d:%s", idx, name) }
	h := Hooks{
		BeforeCompletion: func(_ context.Context, step int, _ *llmkit.Request) {
			log.add(tag(fmt.Sprintf("before%d", step)))
		},
		AfterCompletion: func(_ context.Context, step int, _ *llmkit.Request, _ *llmkit.Response, _ error) {
			log.add(tag(fmt.Sprintf("after%d", step)))
		},
		ToolStart: func(_ context.Context, ev ToolEvent) { log.add(tag("start:" + ev.Call.Name)) },
		ToolEnd:   func(_ context.Context, ev ToolEvent) { log.add(tag("end:" + ev.Call.Name)) },
		ToolHealth: func(_ context.Context, tool string, _ *ToolHealthError) {
			log.add(tag("health:" + tool))
		},
	}
	for _, f := range nilFields {
		switch f {
		case "BeforeCompletion":
			h.BeforeCompletion = nil
		case "AfterCompletion":
			h.AfterCompletion = nil
		case "ToolStart":
			h.ToolStart = nil
		case "ToolEnd":
			h.ToolEnd = nil
		case "ToolHealth":
			h.ToolHealth = nil
		default:
			panic("composeHooks: unknown field " + f)
		}
	}
	return h
}

// TestWithHooks_RegistrationsCompose pins the composing rule over a scripted
// two-turn tool run (turn 1: a healthy call, then a call that fails with a
// *ToolHealthError; turn 2: the answer): with k registrations, every non-nil
// callback fires once per fire point, in registration order, and a nil field
// in one registration never suppresses another registration's callback.
func TestWithHooks_RegistrationsCompose(t *testing.T) {
	// nilFields[k-1][i] lists the fields registration i leaves nil.
	cases := []struct {
		name string
		nils [][]string
	}{
		{"one", [][]string{nil}},
		{"two", [][]string{nil, nil}},
		{"three-middle-sparse", [][]string{nil, {"BeforeCompletion", "ToolStart", "ToolHealth"}, nil}},
		{"three-first-and-last-sparse", [][]string{{"AfterCompletion", "ToolEnd"}, nil, {"BeforeCompletion", "ToolStart"}}},
	}
	// The fire points of the scripted run, in order, with the field each is.
	type point struct{ field, label string }
	points := []point{
		{"BeforeCompletion", "before1"}, {"AfterCompletion", "after1"},
		{"ToolStart", "start:echo"}, {"ToolEnd", "end:echo"},
		{"ToolStart", "start:broken"}, {"ToolHealth", "health:broken"}, {"ToolEnd", "end:broken"},
		{"BeforeCompletion", "before2"}, {"AfterCompletion", "after2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			log := &composeLog{}
			opts := make([]Option, 0, len(tc.nils))
			for i, nils := range tc.nils {
				opts = append(opts, WithHooks(composeHooks(log, i, nils...)))
			}
			fc := newFakeClient(
				toolCallsResp(
					llmkit.ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{}`)},
					llmkit.ToolCall{ID: "c2", Name: "broken", Arguments: json.RawMessage(`{}`)},
				),
				textResp("done", 5, 2),
			)
			tools := []Tool{
				echoTool{name: "echo"},
				healthEchoTool{name: "broken", health: &ToolHealthError{Reason: "down", Err: errors.New("infra")}},
			}
			r := NewRunner(fc, tools, "sys", opts...)
			if _, err := r.Run(context.Background(), "task"); err != nil {
				t.Fatalf("Run: %v", err)
			}

			var want []string
			for _, p := range points {
				for i, nils := range tc.nils {
					if !slices.Contains(nils, p.field) {
						want = append(want, fmt.Sprintf("%d:%s", i, p.label))
					}
				}
			}
			if !slices.Equal(log.entries, want) {
				t.Errorf("hook log =\n  %v\nwant\n  %v", log.entries, want)
			}
		})
	}
}

// TestWithHooks_DeltaFansOutToEveryRegistration pins that a Delta callback
// registered later still receives every fragment, and that each fragment
// reaches the registrations in registration order (the log is the repeating
// pattern 0,1,..,k-1). A registration without Delta neither suppresses nor
// is called.
func TestWithHooks_DeltaFansOutToEveryRegistration(t *testing.T) {
	for _, k := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("k=%d", k), func(t *testing.T) {
			var mu sync.Mutex
			var order []int
			opts := []Option{WithHooks(Hooks{})} // a registration with no Delta
			for i := range k {
				opts = append(opts, WithHooks(Hooks{Delta: func(_ context.Context, _ int, _ llmkit.Delta) {
					mu.Lock()
					defer mu.Unlock()
					order = append(order, i)
				}}))
			}
			r := NewRunner(newFakeClient(textResp("hello streamed world", 1, 1)), nil, "sys", opts...)
			if _, err := r.Run(context.Background(), "task"); err != nil {
				t.Fatalf("Run: %v", err)
			}
			if len(order) == 0 || len(order)%k != 0 {
				t.Fatalf("delta log %v: want a non-empty multiple of k=%d entries", order, k)
			}
			for j, got := range order {
				if got != j%k {
					t.Fatalf("delta log %v: entry %d = registration %d, want %d", order, j, got, j%k)
				}
			}
		})
	}
	// A registration without Delta between two that set it: the fan-out
	// skips it and still reaches every later registration.
	t.Run("nil-middle", func(t *testing.T) {
		var mu sync.Mutex
		var order []int
		reg := func(i int) Hooks {
			return Hooks{Delta: func(_ context.Context, _ int, _ llmkit.Delta) {
				mu.Lock()
				defer mu.Unlock()
				order = append(order, i)
			}}
		}
		r := NewRunner(newFakeClient(textResp("hello streamed world", 1, 1)), nil, "sys",
			WithHooks(reg(0)), WithHooks(Hooks{}), WithHooks(reg(1)))
		if _, err := r.Run(context.Background(), "task"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if len(order) == 0 {
			t.Fatal("no delta fired")
		}
		var n [2]int
		for j, got := range order {
			if want := j % 2; got != want {
				t.Fatalf("delta log %v: entry %d = registration %d, want %d — the nil-middle registration is skipped, not stopped at", order, j, got, want)
			}
			n[got]++
		}
		if n[0] == 0 || n[0] != n[1] {
			t.Fatalf("delta log %v: registration 0 fired %d times, registration 1 %d — every fragment must reach both registrations, across the nil-middle one", order, n[0], n[1])
		}
	})
}

// TestWithHooks_FirstPanicStopsLaterRegistrations pins the panic contract: the
// first panic propagates and later registrations' callbacks for that event
// do not fire.
func TestWithHooks_FirstPanicStopsLaterRegistrations(t *testing.T) {
	var laterFired bool
	r := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys",
		WithHooks(Hooks{BeforeCompletion: func(context.Context, int, *llmkit.Request) { panic("hook bug") }}),
		WithHooks(Hooks{BeforeCompletion: func(context.Context, int, *llmkit.Request) { laterFired = true }}))
	defer func() {
		if v := recover(); v != "hook bug" {
			t.Errorf("recovered %v, want the first hook's panic value", v)
		}
		if laterFired {
			t.Error("a later registration's callback fired after the first one panicked")
		}
	}()
	_, _ = r.Run(context.Background(), "task")
	t.Fatal("Run returned; want the hook panic to propagate")
}

// eventSequence renders events as JSON lines: the exact-sequence comparison
// key for sinks (Time included — every sink sees the same emitted value).
func eventSequence(t *testing.T, evs []llmkit.Event) []string {
	t.Helper()
	out := make([]string, len(evs))
	for i, ev := range evs {
		b, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("marshal event %d: %v", i, err)
		}
		out[i] = string(b)
	}
	return out
}

// TestWithObserver_SinksComposeInRegistrationOrder pins the composing rule for
// k ∈ {1,2} sinks over a two-turn tool run: each sink receives the
// transcript's exact event sequence, and for every event the sinks are called
// in registration order (nil skipped, per llmkit.Observers).
func TestWithObserver_SinksComposeInRegistrationOrder(t *testing.T) {
	for _, k := range []int{1, 2} {
		t.Run(fmt.Sprintf("k=%d", k), func(t *testing.T) {
			var mu sync.Mutex
			var order []int
			got := make([][]llmkit.Event, k)
			opts := []Option{WithObserver(nil)} // nil interface: skipped, not a panic
			for i := range k {
				opts = append(opts, WithObserver(llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
					mu.Lock()
					defer mu.Unlock()
					got[i] = append(got[i], ev)
					order = append(order, i)
				})))
			}
			fc := newFakeClient(toolResp("c1", "echo", `{}`, 5, 2), textResp("done", 5, 2))
			r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", opts...)
			out, err := r.Run(context.Background(), "task")
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			want := eventSequence(t, out.Transcript.Record)
			if len(want) == 0 {
				t.Fatal("transcript recorded no events")
			}
			for i := range k {
				if seq := eventSequence(t, got[i]); !slices.Equal(seq, want) {
					t.Errorf("sink %d saw\n  %v\nwant the transcript's\n  %v", i, seq, want)
				}
			}
			for j, s := range order {
				if s != j%k {
					t.Fatalf("sink call order %v: call %d went to sink %d, want %d", order, j, s, j%k)
				}
			}
		})
	}
}

// spendLedger is a side observer of the kind bk8.9.9 names: it counts
// completions and sums their usage.
type spendLedger struct {
	mu          sync.Mutex
	completions int
	tokens      int64
}

func (s *spendLedger) Observe(_ context.Context, ev llmkit.Event) {
	if ev.Kind != llmkit.KindCompletion || ev.Completion == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.completions++
	s.tokens += ev.Completion.Response.Usage.InputTokens + ev.Completion.Response.Usage.OutputTokens
}

// TestWithObserver_TranscriptSinkAndSpendLedgerCompose is the bk8.9.9 probe:
// a JSONL transcript sink and a spend ledger both see the run whichever order
// they are registered in — one transcript file and one counted completion.
func TestWithObserver_TranscriptSinkAndSpendLedgerCompose(t *testing.T) {
	orders := []struct {
		name       string
		jsonlFirst bool
	}{
		{name: "jsonl-then-spend", jsonlFirst: true},
		{name: "spend-then-jsonl", jsonlFirst: false},
	}
	for _, tc := range orders {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			ledger := &spendLedger{}
			jsonl := WithObserver(JSONL(dir, func(err error) { t.Errorf("sink error: %v", err) }))
			spend := WithObserver(ledger)
			opts := []Option{spend, jsonl}
			if tc.jsonlFirst {
				opts = []Option{jsonl, spend}
			}
			r := NewRunner(newFakeClient(textResp("done", 7, 3)), nil, "sys", opts...)
			if _, err := r.Run(context.Background(), "task"); err != nil {
				t.Fatalf("Run: %v", err)
			}
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatalf("ReadDir: %v", err)
			}
			if len(entries) != 1 {
				t.Errorf("transcript files = %d, want 1", len(entries))
			}
			if ledger.completions != 1 || ledger.tokens != 10 {
				t.Errorf("spend ledger = %d completions / %d tokens, want 1 / 10", ledger.completions, ledger.tokens)
			}
		})
	}
}

// TestWithObserver_TranscriptReceivesEachEventFirst pins the head of the
// chain: the in-memory transcript receives each event before any WithObserver
// sink, so an event whose first sink panics is already in the run's
// Outcome.Transcript (em.tr — run hands that same pointer out), while a later
// sink never sees it. The run is driven through begin/run because the sink's
// panic propagates out of Run before it could return the Outcome.
func TestWithObserver_TranscriptReceivesEachEventFirst(t *testing.T) {
	var later []llmkit.EventKind
	r := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys",
		WithObserver(llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
			if ev.Kind == llmkit.KindCompletion {
				panic("sink bug")
			}
		})),
		WithObserver(llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
			later = append(later, ev.Kind)
		})))
	ctx, em := r.begin(context.Background(), runConfig{}, "task")
	if v := panicValue(func() { _, _ = r.run(ctx, em, nil, "task", nil, "", nil, nil) }); v != "sink bug" {
		t.Fatalf("recovered %v, want the first sink's panic value", v)
	}
	var kinds []llmkit.EventKind
	for _, ev := range em.tr.Record {
		kinds = append(kinds, ev.Kind)
	}
	if want := []llmkit.EventKind{llmkit.KindStart, llmkit.KindCompletion}; !slices.Equal(kinds, want) {
		t.Errorf("transcript kinds = %v, want %v: the transcript must receive the completion before the panicking sink", kinds, want)
	}
	if want := []llmkit.EventKind{llmkit.KindStart}; !slices.Equal(later, want) {
		t.Errorf("later sink kinds = %v, want %v: a panicking sink stops the event from reaching later sinks", later, want)
	}
}

// TestWithObserver_CompactionContextCarriesItsStep pins that an observer
// receives a compaction event on a context whose step (llmkit.StepFromContext)
// equals the event's Step — in the main loop and on the forced-finalization
// turn alike — so a sink that reads the turn from its context agrees with the
// event it is handed.
func TestWithObserver_CompactionContextCarriesItsStep(t *testing.T) {
	cases := []struct {
		name      string
		maxIter   int  // 0: the default, so the loop runs to the answer
		finalized bool // compaction fires inside the forced-finalization turn
	}{
		{name: "main-loop"},
		{name: "finalization-turn", maxIter: 5, finalized: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			steps := make([]scriptStep, 0, 6)
			for i := range 5 {
				steps = append(steps, toolResp(fmt.Sprintf("c%d", i), "big", `{}`, 10, 4))
			}
			steps = append(steps, textResp(`{}`, 5, 2))
			type seen struct{ ctxStep, evStep int }
			var got []seen
			r := NewRunner(newFakeClient(steps...), []Tool{bigResultTool{name: "big"}}, "sys",
				WithLimits(Limits{MaxIterations: tc.maxIter, HistoryTokenBudget: 3500}),
				WithObserver(llmkit.ObserverFunc(func(ctx context.Context, ev llmkit.Event) {
					if ev.Kind == llmkit.KindCompaction {
						got = append(got, seen{ctxStep: llmkit.StepFromContext(ctx), evStep: ev.Step})
					}
				})))
			var out map[string]any
			o, err := r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &out)
			if err != nil {
				t.Fatalf("RunJSON: %v", err)
			}
			if o.Finalized != tc.finalized {
				t.Fatalf("Finalized = %v, want %v", o.Finalized, tc.finalized)
			}
			// Five big results cross the 3500 budget before the sixth
			// completion; that turn's compaction is the only one.
			if want := []seen{{ctxStep: 6, evStep: 6}}; !slices.Equal(got, want) {
				t.Errorf("compaction (observer ctx step, event Step) = %v, want %v", got, want)
			}
		})
	}
}

package decide

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// memSource is a test-local sink that is also a Source: it keeps every
// event it observes and serves them back by run.
type memSource struct {
	mu     sync.Mutex
	events []llmkit.Event
}

func (m *memSource) Observe(_ context.Context, ev llmkit.Event) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, ev)
}

func (m *memSource) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []llmkit.Event
	for _, ev := range m.events {
		if ev.RunID == run {
			out = append(out, ev)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("memSource run %q: %w", run, llmkit.ErrUnknownRun)
	}
	return out, nil
}

func (m *memSource) snapshot() []llmkit.Event {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llmkit.Event(nil), m.events...)
}

// scriptDecider answers every Ask with its fixed (resp, err) and counts
// calls; identity is not implemented.
type scriptDecider struct {
	mu    sync.Mutex
	calls int
	resp  Response
	err   error
}

func (s *scriptDecider) Ask(context.Context, any, Questions) (Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	return s.resp, s.err
}

// identDecider is a scriptDecider that also names its backend.
type identDecider struct {
	scriptDecider
	id llmkit.Identity
}

func (d *identDecider) Identity() llmkit.Identity { return d.id }

func runCtx(run llmkit.RunID, step int) context.Context {
	return llmkit.WithStep(llmkit.WithRun(context.Background(), run), step)
}

func ids(qs []llmkit.DecisionQuestion) []string {
	out := make([]string, len(qs))
	for i, q := range qs {
		out[i] = q.ID
	}
	return out
}

func TestObserve_EmitsExactlyOneDecisionPerNormalizableAsk(t *testing.T) {
	boom := errors.New("backend exploded")
	okResp := checkResponse()
	okResp.Model = "m-reported"
	okResp.Usage = llmkit.Usage{InputTokens: 7, OutputTokens: 3}

	tests := []struct {
		name       string
		d          func() (Decider, *scriptDecider)
		qs         Questions
		wantEvents int
		check      func(t *testing.T, de *llmkit.DecisionEvent)
	}{
		{
			name: "success with identity",
			d: func() (Decider, *scriptDecider) {
				d := &identDecider{scriptDecider: scriptDecider{resp: okResp}, id: llmkit.Identity{Provider: "fakeprov", Model: "m-alias"}}
				return d, &d.scriptDecider
			},
			qs:         checkQuestions(),
			wantEvents: 1,
			check: func(t *testing.T, de *llmkit.DecisionEvent) {
				if de.Backend != "fakeprov" || de.Model != "m-reported" {
					t.Errorf("Backend/Model = %q/%q, want fakeprov/m-reported (response model wins on success)", de.Backend, de.Model)
				}
				if de.Err != "" {
					t.Errorf("Err = %q on success", de.Err)
				}
				if got := ids(de.Questions); !reflect.DeepEqual(got, []string{"c", "n", "s"}) {
					t.Errorf("Questions ids = %v, want sorted [c n s]", got)
				}
				if len(de.Answers) != 3 || de.Answers[0].ID != "c" || de.Answers[1].ID != "n" || de.Answers[2].ID != "s" {
					t.Errorf("Answers = %+v, want sorted c, n, s", de.Answers)
				}
				if de.Usage != okResp.Usage {
					t.Errorf("Usage = %+v, want %+v", de.Usage, okResp.Usage)
				}
				if de.Answers[1].Belief == nil || *de.Answers[1].Belief != 0.5 {
					t.Errorf("noul answer = %+v, want belief 0.5", de.Answers[1])
				}
				if string(de.State) != `"state"` {
					t.Errorf("State = %s, want the marshalled state", de.State)
				}
			},
		},
		{
			name: "success without response model falls back to identity model",
			d: func() (Decider, *scriptDecider) {
				r := checkResponse()
				d := &identDecider{scriptDecider: scriptDecider{resp: r}, id: llmkit.Identity{Provider: "fakeprov", Model: "m-alias"}}
				return d, &d.scriptDecider
			},
			qs:         checkQuestions(),
			wantEvents: 1,
			check: func(t *testing.T, de *llmkit.DecisionEvent) {
				if de.Model != "m-alias" {
					t.Errorf("Model = %q, want identity model m-alias", de.Model)
				}
			},
		},
		{
			name: "success without identity",
			d: func() (Decider, *scriptDecider) {
				d := &scriptDecider{resp: okResp}
				return d, d
			},
			qs:         checkQuestions(),
			wantEvents: 1,
			check: func(t *testing.T, de *llmkit.DecisionEvent) {
				if de.Backend != "" || de.Model != "m-reported" {
					t.Errorf("Backend/Model = %q/%q, want empty/m-reported", de.Backend, de.Model)
				}
			},
		},
		{
			name: "failure carries the error text and identity model",
			d: func() (Decider, *scriptDecider) {
				d := &identDecider{scriptDecider: scriptDecider{err: boom}, id: llmkit.Identity{Provider: "fakeprov", Model: "m-alias"}}
				return d, &d.scriptDecider
			},
			qs:         checkQuestions(),
			wantEvents: 1,
			check: func(t *testing.T, de *llmkit.DecisionEvent) {
				if de.Err != boom.Error() {
					t.Errorf("Err = %q, want %q", de.Err, boom.Error())
				}
				if de.Answers != nil {
					t.Errorf("Answers = %+v on failure, want nil", de.Answers)
				}
				if de.Backend != "fakeprov" || de.Model != "m-alias" {
					t.Errorf("Backend/Model = %q/%q, want fakeprov/m-alias", de.Backend, de.Model)
				}
				if got := ids(de.Questions); !reflect.DeepEqual(got, []string{"c", "n", "s"}) {
					t.Errorf("Questions ids = %v: a failed Ask still records what was asked", got)
				}
			},
		},
		{
			name: "Normalize refusal emits nothing and still reaches d",
			d: func() (Decider, *scriptDecider) {
				d := &scriptDecider{err: boom}
				return d, d
			},
			qs:         Questions{"n": Noul{Instructions: 42}},
			wantEvents: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, inner := tt.d()
			sink := &memSource{}
			run := llmkit.RunID("run-1")
			gotResp, gotErr := Observe(d, sink).Ask(runCtx(run, 4), "state", tt.qs)

			inner.mu.Lock()
			calls := inner.calls
			inner.mu.Unlock()
			if calls != 1 {
				t.Errorf("inner Ask calls = %d, want 1", calls)
			}
			wantResp, wantErr := inner.resp, inner.err
			if !reflect.DeepEqual(gotResp, wantResp) || gotErr != wantErr {
				t.Errorf("Ask = (%+v, %v), want d's (%+v, %v) exactly", gotResp, gotErr, wantResp, wantErr)
			}
			evs := sink.snapshot()
			if len(evs) != tt.wantEvents {
				t.Fatalf("events = %d, want %d", len(evs), tt.wantEvents)
			}
			if tt.wantEvents == 0 {
				return
			}
			ev := evs[0]
			if ev.Kind != llmkit.KindDecision || ev.Decision == nil {
				t.Fatalf("event = %+v, want a KindDecision payload", ev)
			}
			if ev.RunID != run || ev.Step != 4 {
				t.Errorf("RunID/Step = %q/%d, want %q/4 from ctx", ev.RunID, ev.Step, run)
			}
			tt.check(t, ev.Decision)
		})
	}
}

func TestObserve_NormalizeRefusalIsPassedThroughUntouched(t *testing.T) {
	// d's own error for the refused Ask is returned, not Normalize's.
	boom := errors.New("d's own refusal")
	d := &scriptDecider{err: boom}
	sink := &memSource{}
	_, err := Observe(d, sink).Ask(context.Background(), "s", Questions{})
	if err != boom {
		t.Errorf("err = %v, want d's error untouched", err)
	}
	if n := len(sink.snapshot()); n != 0 {
		t.Errorf("events = %d, want 0", n)
	}
}

func TestObserve_IdentityForwardedOnlyWhenInnerHasOne(t *testing.T) {
	sink := &memSource{}
	withID := Observe(&identDecider{id: llmkit.Identity{Provider: "p", Model: "m"}}, sink)
	ic, ok := withID.(interface{ Identity() llmkit.Identity })
	if !ok {
		t.Fatal("Observe(identified) does not implement Identity()")
	}
	if got := ic.Identity(); got != (llmkit.Identity{Provider: "p", Model: "m"}) {
		t.Errorf("Identity() = %+v, want inner's", got)
	}
	if _, ok := Observe(&scriptDecider{}, sink).(interface{ Identity() llmkit.Identity }); ok {
		t.Error("Observe(unidentified) implements Identity(); it must not invent one")
	}
}

func TestObserve_NilObserverReturnsDeciderUnchanged(t *testing.T) {
	d := &scriptDecider{}
	if got := Observe(d, nil); got != Decider(d) {
		t.Errorf("Observe(d, nil) = %v, want d itself", got)
	}
}

// record asks d once per (step, qs) through Observe and returns the
// recording sink and run.
func record(t *testing.T, d Decider, asks ...recordedAsk) (*memSource, llmkit.RunID) {
	t.Helper()
	sink := &memSource{}
	run := llmkit.RunID("rec-run")
	od := Observe(d, sink)
	for _, a := range asks {
		_, _ = od.Ask(runCtx(run, a.step), "recorded state", a.qs)
	}
	return sink, run
}

type recordedAsk struct {
	step int
	qs   Questions
}

// seqDecider returns its scripted outcomes in order.
type seqDecider struct {
	mu   sync.Mutex
	outs []seqOut
	i    int
}

type seqOut struct {
	resp Response
	err  error
}

func (s *seqDecider) Ask(context.Context, any, Questions) (Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	o := s.outs[s.i]
	s.i++
	return o.resp, o.err
}

func TestReplay_SameStepSameIDSetReturnsRecordedAnswersIncludingProbabilities(t *testing.T) {
	want := checkResponse()
	want.Model = "m-reported"
	want.Usage = llmkit.Usage{InputTokens: 11, OutputTokens: 2}
	sink, run := record(t, &scriptDecider{resp: want}, recordedAsk{3, checkQuestions()})

	d, err := Replay(sink, run)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	// A different state replays the same answer: State is not compared.
	got, err := d.Ask(runCtx(run, 3), map[string]any{"other": "state"}, checkQuestions())
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("replayed Response =\n%+v\nwant\n%+v", got, want)
	}
	if got.Choices["c"].Probabilities["yes"] != 0.9 || got.Scores["s"].Probabilities[1] != 0.5 {
		t.Errorf("probabilities lost in replay: %+v", got)
	}
}

func TestReplay_NoMatchDivergesNamingTheStep(t *testing.T) {
	sink, run := record(t, &scriptDecider{resp: checkResponse()}, recordedAsk{3, checkQuestions()})
	d, err := Replay(sink, run)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("other step", func(t *testing.T) {
		resp, err := d.Ask(runCtx(run, 5), "s", checkQuestions())
		if !errors.Is(err, ErrReplayDiverged) {
			t.Fatalf("err = %v, want ErrReplayDiverged", err)
		}
		if !strings.Contains(err.Error(), "step 5") {
			t.Errorf("err = %q, want it to name step 5", err)
		}
		if !reflect.DeepEqual(resp, Response{}) {
			t.Errorf("resp = %+v, want zero", resp)
		}
	})
	t.Run("same step, different id set", func(t *testing.T) {
		_, err := d.Ask(runCtx(run, 3), "s", Questions{"n": Noul{Instructions: "i"}})
		if !errors.Is(err, ErrReplayDiverged) {
			t.Fatalf("err = %v, want ErrReplayDiverged: the id sets differ", err)
		}
		if !strings.Contains(err.Error(), "step 3") {
			t.Errorf("err = %q, want it to name step 3", err)
		}
	})
	t.Run("record consumed", func(t *testing.T) {
		if _, err := d.Ask(runCtx(run, 3), "s", checkQuestions()); err != nil {
			t.Fatalf("first matching Ask: %v", err)
		}
		_, err := d.Ask(runCtx(run, 3), "s", checkQuestions())
		if !errors.Is(err, ErrReplayDiverged) {
			t.Fatalf("second Ask err = %v, want ErrReplayDiverged: one record answers one Ask", err)
		}
	})
}

func TestReplay_FailedRecordReturnsReplayedFailureWithRecordedText(t *testing.T) {
	sink, run := record(t, &scriptDecider{err: errors.New("upstream said no: quota")}, recordedAsk{2, checkQuestions()})
	d, err := Replay(sink, run)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := d.Ask(runCtx(run, 2), "s", checkQuestions())
	if !errors.Is(err, ErrReplayedFailure) {
		t.Fatalf("err = %v, want ErrReplayedFailure", err)
	}
	if !strings.Contains(err.Error(), "upstream said no: quota") {
		t.Errorf("err = %q, want the recorded text", err)
	}
	if errors.Is(err, ErrReplayDiverged) {
		t.Error("a replayed failure is not a divergence")
	}
	if !reflect.DeepEqual(resp, Response{}) {
		t.Errorf("resp = %+v, want zero", resp)
	}
}

func TestReplay_NormalizeRefusalIsInvalidRequestAndConsumesNothing(t *testing.T) {
	good := Questions{"n": Noul{Instructions: "i"}}
	want := Response{Model: "m", Nouls: map[string]float64{"n": 0.25}}
	sink, run := record(t, &scriptDecider{resp: want}, recordedAsk{1, good})
	d, err := Replay(sink, run)
	if err != nil {
		t.Fatal(err)
	}
	// Same step, same id set as the record, but the question is refused.
	_, err = d.Ask(runCtx(run, 1), "s", Questions{"n": Noul{Instructions: 42}})
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v, want ErrInvalidRequest", err)
	}
	var ae *llmkit.APIError
	if !errors.As(err, &ae) || ae.Provider != "decide" {
		t.Errorf("err = %#v, want Normalize's refusal unchanged (Provider decide)", err)
	}
	got, err := d.Ask(runCtx(run, 1), "s", good)
	if err != nil {
		t.Fatalf("Ask after a refusal: %v: the refusal consumed the record", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Response = %+v, want %+v", got, want)
	}
}

func TestReplay_SameStepDifferentIDSetsMatchOutOfOrder(t *testing.T) {
	qa := Questions{"a": Noul{Instructions: "i"}}
	qb := Questions{"b": Noul{Instructions: "i"}}
	ra := Response{Model: "m", Nouls: map[string]float64{"a": 0.1}}
	rb := Response{Model: "m", Nouls: map[string]float64{"b": 0.9}}
	sink, run := record(t, &seqDecider{outs: []seqOut{{resp: ra}, {resp: rb}}},
		recordedAsk{2, qa}, recordedAsk{2, qb})
	d, err := Replay(sink, run)
	if err != nil {
		t.Fatal(err)
	}
	gotB, err := d.Ask(runCtx(run, 2), "s", qb)
	if err != nil || !reflect.DeepEqual(gotB, rb) {
		t.Fatalf("Ask(b) = (%+v, %v), want %+v", gotB, err, rb)
	}
	gotA, err := d.Ask(runCtx(run, 2), "s", qa)
	if err != nil || !reflect.DeepEqual(gotA, ra) {
		t.Fatalf("Ask(a) = (%+v, %v), want %+v", gotA, err, ra)
	}
}

func TestReplay_SameIDSetMatchesInRecordedOrder(t *testing.T) {
	q := Questions{"n": Noul{Instructions: "i"}}
	r1 := Response{Model: "m", Nouls: map[string]float64{"n": 0.1}}
	r2 := Response{Model: "m", Nouls: map[string]float64{"n": 0.9}}
	sink, run := record(t, &seqDecider{outs: []seqOut{{resp: r1}, {resp: r2}}}, recordedAsk{1, q}, recordedAsk{1, q})
	d, _ := Replay(sink, run)
	for i, want := range []Response{r1, r2} {
		got, err := d.Ask(runCtx(run, 1), "s", q)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("Ask %d = (%+v, %v), want %+v", i, got, err, want)
		}
	}
}

func TestReplay_RecordThatFailsCheckDiverges(t *testing.T) {
	q := Questions{"n": Noul{Instructions: "i"}}
	ev := llmkit.NewEvent(runCtx("r", 1), llmkit.KindDecision)
	ev.Decision = &llmkit.DecisionEvent{Model: "m", Answers: nil} // no answer for "n"
	ev.Decision.Questions = []llmkit.DecisionQuestion{{ID: "n", Kind: "noul"}}
	src := &memSource{events: []llmkit.Event{ev}}
	d, err := Replay(src, "r")
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.Ask(runCtx("r", 1), "s", q)
	if !errors.Is(err, ErrReplayDiverged) {
		t.Fatalf("err = %v, want ErrReplayDiverged from the Check failure", err)
	}
}

func TestReplay_UnknownRunIsTheSourcesError(t *testing.T) {
	_, err := Replay(&memSource{}, "nope")
	if !errors.Is(err, llmkit.ErrUnknownRun) {
		t.Fatalf("err = %v, want ErrUnknownRun", err)
	}
}

func TestReplay_IgnoresNonDecisionEvents(t *testing.T) {
	q := Questions{"n": Noul{Instructions: "i"}}
	want := Response{Model: "m", Nouls: map[string]float64{"n": 0.4}}
	sink, run := record(t, &scriptDecider{resp: want}, recordedAsk{1, q})
	sink.Observe(context.Background(), llmkit.Event{Kind: llmkit.KindStart, RunID: run, Start: &llmkit.StartEvent{Task: "t"}})
	d, err := Replay(sink, run)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := d.Ask(runCtx(run, 1), "s", q); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Ask = (%+v, %v), want %+v", got, err, want)
	}
}

func TestReplay_ConcurrentAsksConsumeEachRecordOnce(t *testing.T) {
	const n = 16
	q := Questions{"n": Noul{Instructions: "i"}}
	outs := make([]seqOut, n)
	for i := range outs {
		outs[i] = seqOut{resp: Response{Model: "m", Nouls: map[string]float64{"n": float64(i) / n}}}
	}
	sink, run := record(t, &seqDecider{outs: outs}, func() []recordedAsk {
		a := make([]recordedAsk, n)
		for i := range a {
			a[i] = recordedAsk{1, q}
		}
		return a
	}()...)
	d, _ := Replay(sink, run)

	var mu sync.Mutex
	seen := map[float64]int{}
	var wg sync.WaitGroup
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := d.Ask(runCtx(run, 1), "s", q)
			if err != nil {
				t.Errorf("Ask: %v", err)
				return
			}
			mu.Lock()
			seen[got.Nouls["n"]]++
			mu.Unlock()
		}()
	}
	wg.Wait()
	if len(seen) != n {
		t.Errorf("distinct answers = %d, want %d: a record was served twice", len(seen), n)
	}
	if _, err := d.Ask(runCtx(run, 1), "s", q); !errors.Is(err, ErrReplayDiverged) {
		t.Errorf("Ask past the record: err = %v, want ErrReplayDiverged", err)
	}
}

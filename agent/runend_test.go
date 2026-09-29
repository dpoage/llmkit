package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// finalizeLog is an observer that keeps every Finalize event a Runner emits.
// It reads the durable-sink side of the chain, so it still sees the Finalize
// of a run that panicked and returned no Outcome.
type finalizeLog struct {
	mu     sync.Mutex
	events []llmkit.Event
}

func (l *finalizeLog) Observe(_ context.Context, ev llmkit.Event) {
	if ev.Kind != llmkit.KindFinalize {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, ev)
}

// only returns the single Finalize event, failing the test if the run emitted
// zero or several.
func (l *finalizeLog) only(t *testing.T) llmkit.Event {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.events) != 1 {
		t.Fatalf("Finalize events = %d, want exactly 1", len(l.events))
	}
	return l.events[0]
}

// echoOnce is the tool the run-end rows call once before their answer turn.
var echoOnce = []Tool{echoTool{name: "echo"}}

func echoCall() scriptStep { return toolResp("c1", "echo", `{"v":"x"}`, 1, 1) }

// TestRun_OutputCapStop pins TruncOutputCap: a main-loop turn whose max-tokens
// continuation also stopped at the cap ends the run cut, unless something
// queued continues the loop.
func TestRun_OutputCapStop(t *testing.T) {
	t.Run("both capped with nothing queued ends TruncOutputCap", func(t *testing.T) {
		fc := newFakeClient(maxTokensResp("first half ", 1, 1), maxTokensResp("second half", 1, 1))
		out, err := NewRunner(fc, nil, "sys").Run(context.Background(), "task")
		if ierr := incompleteErr(out, err, TruncOutputCap); ierr != nil {
			t.Fatal(ierr)
		}
		if fc.callCount() != 2 {
			t.Errorf("completions = %d, want 2 (main + continuation, no further turn)", fc.callCount())
		}
		if out.FinalText != "first half second half" {
			t.Errorf("FinalText = %q, want the stitched halves", out.FinalText)
		}
	})

	t.Run("a queued follow-up continues the loop and a later answer returns nil", func(t *testing.T) {
		fc := newFakeClient(maxTokensResp("cut", 1, 1), maxTokensResp("cut again", 1, 1), textResp("the answer", 1, 1))
		s := NewSteering()
		s.FollowUp(llmkit.Text("go on"))
		out, err := NewRunner(fc, nil, "sys").Run(context.Background(), "task", WithSteering(s))
		if err != nil {
			t.Fatalf("Run: %v, want nil: the follow-up continued the loop to a clean answer", err)
		}
		if out.TruncationReason != "" {
			t.Errorf("TruncationReason = %q, want empty", out.TruncationReason)
		}
		if out.FinalText != "the answer" {
			t.Errorf("FinalText = %q, want the later answer", out.FinalText)
		}
	})

	t.Run("a capped continuation with no visible text is cut, not nudged", func(t *testing.T) {
		fc := newFakeClient(maxTokensResp("<think>still thinking", 1, 1), maxTokensResp("", 1, 1), textResp("never asked", 1, 1))
		out, err := NewRunner(fc, nil, "sys").Run(context.Background(), "task")
		if ierr := incompleteErr(out, err, TruncOutputCap); ierr != nil {
			t.Fatal(ierr)
		}
		if fc.callCount() != 2 {
			t.Errorf("completions = %d, want 2 (a capped empty turn is not nudged)", fc.callCount())
		}
	})

	t.Run("a finalization turn whose continuation caps keeps the stop reason", func(t *testing.T) {
		fc := newFakeClient(
			echoCall(),
			maxTokensResp(`{"path":`, 1, 1),               // forced finalization turn, cut off
			maxTokensResp(`"a`, 1, 1),                     // its continuation, cut off again
			textResp(`{"path":"r.go","note":"ok"}`, 1, 1), // repair parses
		)
		r := NewRunner(fc, echoOnce, "sys", WithLimits(Limits{MaxIterations: 1}))
		var got item
		out, err := r.RunJSON(context.Background(), "task", nil, &got)
		if err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		if out.TruncationReason != TruncMaxIterations {
			t.Errorf("TruncationReason = %q, want %q (the finalization turn's cap does not replace the stop condition)", out.TruncationReason, TruncMaxIterations)
		}
		if !out.Finalized {
			t.Error("Finalized = false, want true")
		}
	})
}

// TestRun_NoAnswerStop pins TruncNoAnswer: a model that stays silent through
// every nudge ends the run unanswered, and an answer on the last nudge is a
// clean finish.
func TestRun_NoAnswerStop(t *testing.T) {
	t.Run("nudges exhausted ends TruncNoAnswer", func(t *testing.T) {
		fc := newFakeClient(textResp("", 1, 1), thinkOnlyResp("hmm", 1, 1), textResp("  ", 1, 1), textResp("never asked", 1, 1))
		out, err := NewRunner(fc, nil, "sys").Run(context.Background(), "task")
		if ierr := incompleteErr(out, err, TruncNoAnswer); ierr != nil {
			t.Fatal(ierr)
		}
		if fc.callCount() != maxEmptyTurnNudges+1 {
			t.Errorf("completions = %d, want %d (the turn plus one per nudge)", fc.callCount(), maxEmptyTurnNudges+1)
		}
	})

	t.Run("an answer on the last nudge is a clean finish", func(t *testing.T) {
		fc := newFakeClient(textResp("", 1, 1), textResp("", 1, 1), textResp("the answer", 1, 1))
		out, err := NewRunner(fc, nil, "sys").Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("Run: %v, want nil", err)
		}
		if out.TruncationReason != "" || out.FinalText != "the answer" {
			t.Errorf("outcome = (%q, %q), want a clean answer", out.TruncationReason, out.FinalText)
		}
	})
}

// finalizeRow is one run end of TestFinalize_StatusAndErr: run makes the
// Runner drive to that end, with log attached as the durable sink, and
// returns what Run/RunJSON returned.
type finalizeRow struct {
	name string
	run  func(t *testing.T, log *finalizeLog) error
	want llmkit.RunStatus
}

func logged(log *finalizeLog, opts ...Option) []Option {
	return append([]Option{WithObserver(log)}, opts...)
}

func runOnce(client llmkit.Client, tools []Tool, log *finalizeLog, opts ...Option) func(context.Context) error {
	return func(ctx context.Context) error {
		_, err := NewRunner(client, tools, "sys", logged(log, opts...)...).Run(ctx, "task")
		return err
	}
}

func runJSONOnce(client llmkit.Client, tools []Tool, log *finalizeLog, opts ...Option) func(context.Context) error {
	return func(ctx context.Context) error {
		var got item
		_, err := NewRunner(client, tools, "sys", logged(log, opts...)...).RunJSON(ctx, "task", nil, &got)
		return err
	}
}

// TestFinalize_StatusAndErr pins the Finalize event's Status and Err over
// every run end of Run and RunJSON: exactly one Finalize per run, Status by
// the precedence completed, incomplete, refused, canceled, failed, and Err
// the returned error's text (empty iff the run returned nil).
func TestFinalize_StatusAndErr(t *testing.T) {
	refusal := scriptStep{resp: llmkit.Response{Text: "I cannot", StopReason: llmkit.StopRefusal}}
	providerStop := scriptStep{resp: llmkit.Response{Text: "boom", StopReason: llmkit.StopError}}

	rows := []finalizeRow{
		{"Run completed", func(t *testing.T, log *finalizeLog) error {
			return runOnce(newFakeClient(textResp("done", 1, 1)), nil, log)(context.Background())
		}, llmkit.RunCompleted},
		{"RunJSON completed", func(t *testing.T, log *finalizeLog) error {
			return runJSONOnce(newFakeClient(textResp(`{"path":"a","note":"b"}`, 1, 1)), nil, log)(context.Background())
		}, llmkit.RunCompleted},
		{"Run incomplete", func(t *testing.T, log *finalizeLog) error {
			return runOnce(newFakeClient(echoCall(), textResp("never", 1, 1)), echoOnce, log,
				WithLimits(Limits{MaxIterations: 1}))(context.Background())
		}, llmkit.RunIncomplete},
		{"RunJSON truncated and unparseable", func(t *testing.T, log *finalizeLog) error {
			fc := newFakeClient(echoCall(), textResp("not json", 1, 1), textResp("still not json", 1, 1))
			return runJSONOnce(fc, echoOnce, log, WithLimits(Limits{MaxIterations: 1}))(context.Background())
		}, llmkit.RunIncomplete},
		{"Run refused", func(t *testing.T, log *finalizeLog) error {
			return runOnce(newFakeClient(refusal), nil, log)(context.Background())
		}, llmkit.RunRefused},
		{"Run refused on a provider error stop", func(t *testing.T, log *finalizeLog) error {
			return runOnce(newFakeClient(providerStop), nil, log)(context.Background())
		}, llmkit.RunRefused},
		{"Run refused while the run ctx is done: the typed error wins", func(t *testing.T, log *finalizeLog) error {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// Cancel after the refusal completion returned: the loop ends
			// at the typed stop while the run's own ctx is done.
			hooks := Hooks{AfterCompletion: func(context.Context, int, *llmkit.Request, *llmkit.Response, error) { cancel() }}
			err := runOnce(newFakeClient(refusal), nil, log, WithHooks(hooks))(ctx)
			if ctx.Err() == nil {
				t.Fatal("the run ctx is not done; the row does not exercise the precedence")
			}
			var stop *StopReasonError
			if !errors.As(err, &stop) {
				t.Fatalf("err = %v, want *StopReasonError", err)
			}
			return err
		}, llmkit.RunRefused},
		{"Run completion failure", func(t *testing.T, log *finalizeLog) error {
			return runOnce(newFakeClient(scriptStep{err: errors.New("upstream down")}), nil, log)(context.Background())
		}, llmkit.RunFailed},
		{"RunJSON completion failure", func(t *testing.T, log *finalizeLog) error {
			return runJSONOnce(newFakeClient(scriptStep{err: errors.New("upstream down")}), nil, log)(context.Background())
		}, llmkit.RunFailed},
		{"Run RequestPolicy abort", func(t *testing.T, log *finalizeLog) error {
			policy := RequestPolicyFunc(func(context.Context, int, *llmkit.Request) error { return errors.New("policy said no") })
			return runOnce(newFakeClient(textResp("done", 1, 1)), nil, log, WithRequestPolicy(policy))(context.Background())
		}, llmkit.RunFailed},
		{"Run ErrSteeringInUse", func(t *testing.T, log *finalizeLog) error {
			// The outer run holds the handle while its own hook starts the
			// second run on it; the second run is the one under test.
			s := NewSteering()
			var inner error
			hooks := Hooks{BeforeCompletion: func(ctx context.Context, _ int, _ *llmkit.Request) {
				if inner != nil {
					return
				}
				_, inner = NewRunner(newFakeClient(textResp("x", 1, 1)), nil, "sys", WithObserver(log)).
					Run(ctx, "two", WithSteering(s))
			}}
			outer := NewRunner(newFakeClient(textResp("done", 1, 1)), nil, "sys", WithHooks(hooks))
			if _, err := outer.Run(context.Background(), "one", WithSteering(s)); err != nil {
				t.Fatalf("outer run: %v", err)
			}
			if !errors.Is(inner, ErrSteeringInUse) {
				t.Fatalf("inner err = %v, want ErrSteeringInUse", inner)
			}
			return inner
		}, llmkit.RunFailed},
		{"RunJSON unparseable, not truncated", func(t *testing.T, log *finalizeLog) error {
			fc := newFakeClient(textResp("not json", 1, 1), textResp("still not json", 1, 1))
			err := runJSONOnce(fc, nil, log)(context.Background())
			if !errors.Is(err, ErrUnparseableOutput) {
				t.Fatalf("err = %v, want ErrUnparseableOutput", err)
			}
			var ie *IncompleteError
			if errors.As(err, &ie) {
				t.Fatalf("err = %v matches *IncompleteError; the row must not be a truncation", err)
			}
			return err
		}, llmkit.RunFailed},
		{"Run canceled: the run ctx is cancelled mid-run", func(t *testing.T, log *finalizeLog) error {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			hooks := Hooks{BeforeCompletion: func(context.Context, int, *llmkit.Request) { cancel() }}
			return runOnce(newFakeClient(textResp("done", 1, 1)), nil, log, WithHooks(hooks))(ctx)
		}, llmkit.RunCanceled},
		{"Run canceled: the run ctx's deadline expired", func(t *testing.T, log *finalizeLog) error {
			// An already-expired deadline is not a cancel call, and the
			// error it produces is DeadlineExceeded, not Canceled: the
			// status must key on the run ctx being done, not on the
			// error's identity.
			ctx, cancel := context.WithTimeout(context.Background(), 0)
			defer cancel()
			err := runOnce(newFakeClient(textResp("never asked", 1, 1)), nil, log)(ctx)
			if ctx.Err() == nil {
				t.Fatal("the run ctx is not done; the row does not exercise the precedence")
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("err = %v, want it to wrap DeadlineExceeded", err)
			}
			return err
		}, llmkit.RunCanceled},
		{"RunJSON canceled: the run ctx is cancelled mid-run", func(t *testing.T, log *finalizeLog) error {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			hooks := Hooks{BeforeCompletion: func(context.Context, int, *llmkit.Request) { cancel() }}
			return runJSONOnce(newFakeClient(textResp(`{"path":"a","note":"b"}`, 1, 1)), nil, log, WithHooks(hooks))(ctx)
		}, llmkit.RunCanceled},
		{"Run provider DeadlineExceeded with a live run ctx", func(t *testing.T, log *finalizeLog) error {
			err := runOnce(newFakeClient(scriptStep{err: context.DeadlineExceeded}), nil, log)(context.Background())
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("err = %v, want it to wrap DeadlineExceeded", err)
			}
			return err
		}, llmkit.RunFailed},
		{"Run failed: the provider returned context.Canceled while the run ctx is live", func(t *testing.T, log *finalizeLog) error {
			// A provider Canceled with a live run ctx is a completion
			// failure, not a run cancellation: the status must key on the
			// run ctx, not on the error matching context.Canceled.
			err := runOnce(newFakeClient(scriptStep{err: context.Canceled}), nil, log)(context.Background())
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want it to wrap context.Canceled", err)
			}
			return err
		}, llmkit.RunFailed},
		{"Run incomplete while the run ctx is done: the typed error wins", func(t *testing.T, log *finalizeLog) error {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// Cancel after the continuation completion returned: the loop
			// still ends at the cap, and Run converts it to IncompleteError.
			hooks := Hooks{AfterCompletion: func(_ context.Context, step int, _ *llmkit.Request, _ *llmkit.Response, _ error) {
				if step == 2 {
					cancel()
				}
			}}
			fc := newFakeClient(maxTokensResp("a", 1, 1), maxTokensResp("b", 1, 1))
			err := runOnce(fc, nil, log, WithHooks(hooks))(ctx)
			if ctx.Err() == nil {
				t.Fatal("the run ctx is not done; the row does not exercise the precedence")
			}
			var ie *IncompleteError
			if !errors.As(err, &ie) {
				t.Fatalf("err = %v, want *IncompleteError", err)
			}
			return err
		}, llmkit.RunIncomplete},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			log := &finalizeLog{}
			err := row.run(t, log)
			fin := log.only(t).Finalize
			if fin == nil {
				t.Fatal("Finalize event carries no payload")
			}
			if fin.Status != row.want {
				t.Errorf("Status = %q, want %q (err = %v)", fin.Status, row.want, err)
			}
			wantErr := ""
			if err != nil {
				wantErr = err.Error()
			}
			if fin.Err != wantErr {
				t.Errorf("Err = %q, want %q", fin.Err, wantErr)
			}
			if (row.want == llmkit.RunCompleted) != (fin.Err == "") {
				t.Errorf("Err = %q with status %q: Err must be empty exactly when the run completed", fin.Err, fin.Status)
			}
		})
	}
}

// panicUnmarshal is a RunJSON out whose decoder panics.
type panicUnmarshal struct{}

func (*panicUnmarshal) UnmarshalJSON([]byte) error { panic("unmarshal boom") }

// recovered runs f and returns the value it panicked with (nil if it did not).
func recovered(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

// TestFinalize_PanicClosesRecordWithZeroCounters pins the panic path of Run
// and RunJSON: one Finalize with Status panicked, Step 0, zero Usage, empty
// text and error, while the ORIGINAL panic value propagates. The RunJSON rows
// panic after the run holds an Outcome with counters, so the path must not be
// detected by a nil Outcome.
func TestFinalize_PanicClosesRecordWithZeroCounters(t *testing.T) {
	rows := []struct {
		name  string
		panic any
		run   func(log *finalizeLog, panicValue any)
	}{
		{"Run: a hook in the main loop", "hook boom", func(log *finalizeLog, v any) {
			hooks := Hooks{BeforeCompletion: func(_ context.Context, step int, _ *llmkit.Request) {
				if step == 2 {
					panic(v)
				}
			}}
			fc := newFakeClient(withCache(echoCall(), 4, 2), textResp("done", 10, 5))
			_, _ = NewRunner(fc, echoOnce, "sys", WithObserver(log), WithHooks(hooks)).Run(context.Background(), "task")
		}},
		{"RunJSON: a hook in the main loop", "loop hook boom", func(log *finalizeLog, v any) {
			hooks := Hooks{BeforeCompletion: func(_ context.Context, step int, _ *llmkit.Request) {
				if step == 2 {
					panic(v)
				}
			}}
			fc := newFakeClient(echoCall(), textResp(`{"path":"a","note":"b"}`, 10, 5))
			var got item
			_, _ = NewRunner(fc, echoOnce, "sys", WithObserver(log), WithHooks(hooks)).RunJSON(context.Background(), "task", nil, &got)
		}},
		{"RunJSON: a hook on the repair turn", "repair hook boom", func(log *finalizeLog, v any) {
			// Turn 1 answers unparseably; the repair completion is turn 2.
			hooks := Hooks{AfterCompletion: func(_ context.Context, step int, _ *llmkit.Request, _ *llmkit.Response, _ error) {
				if step == 2 {
					panic(v)
				}
			}}
			fc := newFakeClient(textResp("not json", 10, 5), textResp(`{"path":"a","note":"b"}`, 10, 5))
			var got item
			_, _ = NewRunner(fc, nil, "sys", WithObserver(log), WithHooks(hooks)).RunJSON(context.Background(), "task", nil, &got)
		}},
		{"RunJSON: out.UnmarshalJSON", "unmarshal boom", func(log *finalizeLog, _ any) {
			fc := newFakeClient(textResp(`{"a":1}`, 10, 5))
			_, _ = NewRunner(fc, nil, "sys", WithObserver(log)).RunJSON(context.Background(), "task", nil, &panicUnmarshal{})
		}},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			log := &finalizeLog{}
			got := recovered(func() { row.run(log, row.panic) })
			if got != row.panic {
				t.Fatalf("recovered panic value = %v, want the original %v", fmt.Sprint(got), row.panic)
			}
			ev := log.only(t)
			fin := ev.Finalize
			if fin == nil {
				t.Fatal("Finalize event carries no payload")
			}
			if fin.Status != llmkit.RunPanicked {
				t.Errorf("Status = %q, want %q", fin.Status, llmkit.RunPanicked)
			}
			if ev.Step != 0 {
				t.Errorf("Step = %d, want 0", ev.Step)
			}
			if fin.Usage != (llmkit.Usage{}) {
				t.Errorf("Usage = %+v, want zero", fin.Usage)
			}
			if fin.FinalText != "" || fin.Err != "" || fin.TruncationReason != "" || fin.Finalized {
				t.Errorf("Finalize = %+v, want empty text, error, truncation reason, and Finalized false", *fin)
			}
		})
	}
}

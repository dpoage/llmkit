package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNewRunner_RejectsEmptyToolName pins the construction guard: a tool whose
// Def().Name is "" cannot be addressed by the model, so NewRunner panics with
// a string naming the tool's index, wherever it sits and however many
// unnamed tools the list holds.
func TestNewRunner_RejectsEmptyToolName(t *testing.T) {
	tests := []struct {
		name  string
		tools []Tool
		want  string
	}{
		{"only tool", entryTools(""), "index 0"},
		{"first", entryTools("", "b", "c"), "index 0"},
		{"middle", entryTools("a", "", "c"), "index 1"},
		{"last", entryTools("a", "b", ""), "index 2"},
		{"two unnamed reports the first", entryTools("a", "", ""), "index 1"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := panicValue(func() { NewRunner(newFakeClient(), tc.tools, "sys") })
			if v == nil {
				t.Fatal("NewRunner did not panic on an empty tool name")
			}
			msg := fmt.Sprint(v)
			if !strings.Contains(msg, tc.want) || !strings.Contains(msg, "empty name") {
				t.Errorf("panic %q must name %q and say the name is empty", msg, tc.want)
			}
		})
	}
}

// TestTranscriptEvents_NeverEmptySuccess pins the Source contract on the
// in-memory transcript: a matching RunID over an empty Record fails with
// ErrUnknownRun instead of an empty slice and a nil error; a non-empty
// record returns a clone the caller owns.
func TestTranscriptEvents_NeverEmptySuccess(t *testing.T) {
	ctx := context.Background()
	for _, tr := range []*Transcript{
		{RunID: "a"},
		{RunID: "a", Record: []llmkit.Event{}},
	} {
		evs, err := tr.Events(ctx, "a")
		if !errors.Is(err, llmkit.ErrUnknownRun) {
			t.Errorf("Events on a named, empty transcript = (%v, %v), want an error wrapping ErrUnknownRun", evs, err)
		}
		if len(evs) != 0 {
			t.Errorf("Events returned %d events beside an error", len(evs))
		}
	}

	rec := []llmkit.Event{{Kind: llmkit.KindStart}, {Kind: llmkit.KindFinalize}}
	tr := &Transcript{RunID: "a", Record: rec}
	evs, err := tr.Events(ctx, "a")
	if err != nil || len(evs) != 2 {
		t.Fatalf("Events on a recorded run = (%d events, %v), want 2 events and nil", len(evs), err)
	}
	evs[0].Kind = llmkit.KindSteer
	if tr.Record[0].Kind != llmkit.KindStart {
		t.Error("Events returned the transcript's own backing array, not a copy")
	}
}

// stepProbeTool runs a nested Runner inside a parent's tool phase and reports
// the step its context carried in.
type stepProbeTool struct {
	inner     func(ctx context.Context)
	ctxStep   int
	innerDone bool
}

func (*stepProbeTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "nest", Description: "nest", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (p *stepProbeTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	p.ctxStep = llmkit.NewEvent(ctx, llmkit.KindFinalize).Step
	p.inner(ctx)
	p.innerDone = true
	return "ok", nil
}

// TestFinalize_PanickedStepIsZeroUnderAnyContext pins that a panicked Finalize
// reports Step 0 (which the JSON omits) for Run and RunJSON under a context
// armed with WithStep, and for a Runner nested in a parent's tool phase, which
// inherits the parent's step. A clean run on the same context still reports
// its completed turns.
func TestFinalize_PanickedStepIsZeroUnderAnyContext(t *testing.T) {
	boom := Hooks{BeforeCompletion: func(_ context.Context, step int, _ *llmkit.Request) {
		if step == 2 {
			panic("boom")
		}
	}}
	check := func(t *testing.T, ev llmkit.Event) {
		t.Helper()
		if ev.Finalize == nil || ev.Finalize.Status != llmkit.RunPanicked {
			t.Fatalf("Finalize = %+v, want Status panicked", ev.Finalize)
		}
		if ev.Step != 0 {
			t.Errorf("panicked Finalize Step = %d, want 0", ev.Step)
		}
		raw, err := json.Marshal(ev)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(raw), `"step"`) {
			t.Errorf("panicked Finalize JSON carries a step: %s", raw)
		}
	}

	for _, n := range []int{0, 1, 7} {
		ctx := llmkit.WithStep(context.Background(), n)
		t.Run(fmt.Sprintf("Run/WithStep(%d)", n), func(t *testing.T) {
			log := &finalizeLog{}
			fc := newFakeClient(echoCall(), textResp("done", 10, 5))
			if v := recovered(func() {
				_, _ = NewRunner(fc, echoOnce, "sys", WithObserver(log), WithHooks(boom)).Run(ctx, "task")
			}); v != "boom" {
				t.Fatalf("recovered %v, want the hook panic", v)
			}
			check(t, log.only(t))
		})
		t.Run(fmt.Sprintf("RunJSON/WithStep(%d)", n), func(t *testing.T) {
			log := &finalizeLog{}
			fc := newFakeClient(echoCall(), textResp(`{"path":"a","note":"b"}`, 10, 5))
			var got item
			if v := recovered(func() {
				_, _ = NewRunner(fc, echoOnce, "sys", WithObserver(log), WithHooks(boom)).RunJSON(ctx, "task", nil, &got)
			}); v != "boom" {
				t.Fatalf("recovered %v, want the hook panic", v)
			}
			check(t, log.only(t))
		})
	}

	t.Run("nested Runner inherits the parent's step", func(t *testing.T) {
		log := &finalizeLog{}
		probe := &stepProbeTool{}
		probe.inner = func(ctx context.Context) {
			child := NewRunner(newFakeClient(echoCall(), textResp("done", 10, 5)), echoOnce, "sys",
				WithObserver(log), WithHooks(boom))
			if v := recovered(func() { _, _ = child.Run(ctx, "sub") }); v != "boom" {
				t.Errorf("child recovered %v, want the hook panic", v)
			}
		}
		parent := NewRunner(newFakeClient(toolResp("n1", "nest", `{}`, 1, 1), textResp("done", 1, 1)), []Tool{probe}, "sys")
		if _, err := parent.Run(context.Background(), "task"); err != nil {
			t.Fatal(err)
		}
		if !probe.innerDone || probe.ctxStep < 1 {
			t.Fatalf("nested run did not execute under a parent step: done=%v ctxStep=%d", probe.innerDone, probe.ctxStep)
		}
		check(t, log.only(t))
	})

	t.Run("a clean run on an armed context still reports its completed turns", func(t *testing.T) {
		log := &finalizeLog{}
		fc := newFakeClient(echoCall(), textResp("done", 10, 5))
		ctx := llmkit.WithStep(context.Background(), 7)
		if _, err := NewRunner(fc, echoOnce, "sys", WithObserver(log)).Run(ctx, "task"); err != nil {
			t.Fatal(err)
		}
		ev := log.only(t)
		if ev.Finalize.Status != llmkit.RunCompleted || ev.Step != 2 {
			t.Errorf("clean Finalize = status %q step %d, want completed step 2", ev.Finalize.Status, ev.Step)
		}
	})
}

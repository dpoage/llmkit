package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// entryTool is a Tool whose identity is observable: its description is unique
// per instance, so a def and a dispatch target can be compared exactly.
type entryTool struct {
	name string
	id   int
}

func (t *entryTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: fmt.Sprintf("tool #%d", t.id), Parameters: []byte(`{"type":"object"}`)}
}

func (t *entryTool) Run(context.Context, json.RawMessage) (string, error) {
	return fmt.Sprintf("ran #%d", t.id), nil
}

func entryTools(names ...string) []Tool {
	tools := make([]Tool, len(names))
	for i, n := range names {
		tools[i] = &entryTool{name: n, id: i}
	}
	return tools
}

// panicValue runs f and returns what it panicked with, or nil.
func panicValue(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

// TestNewRunner_RejectsDuplicateAndNilTools pins T-1: a duplicate Def().Name
// or a nil entry is a construction panic whose value names the offender.
func TestNewRunner_RejectsDuplicateAndNilTools(t *testing.T) {
	nilAt1 := entryTools("a", "b", "c")
	nilAt1[1] = nil
	nilAt0 := []Tool{nil}
	nilLast := append(entryTools("a", "b"), nil)

	tests := []struct {
		name  string
		tools []Tool
		want  []string // substrings the panic value must contain
	}{
		{"adjacent duplicate", entryTools("a", "b", "b", "c"), []string{`"b"`, "indexes 1 and 2"}},
		{"non-adjacent duplicate", entryTools("a", "b", "c", "a"), []string{`"a"`, "indexes 0 and 3"}},
		{"duplicate far apart", entryTools("x", "b", "c", "d", "e", "x"), []string{`"x"`, "indexes 0 and 5"}},
		{"three-way duplicate reports first pair", entryTools("k", "k", "k"), []string{`"k"`, "indexes 0 and 1"}},
		{"nil in the middle", nilAt1, []string{"index 1", "nil"}},
		{"only tool is nil", nilAt0, []string{"index 0", "nil"}},
		{"nil last", nilLast, []string{"index 2", "nil"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := panicValue(func() { NewRunner(newFakeClient(), tc.tools, "sys") })
			if v == nil {
				t.Fatal("NewRunner did not panic")
			}
			msg := fmt.Sprint(v)
			for _, w := range tc.want {
				if !strings.Contains(msg, w) {
					t.Errorf("panic %q does not contain %q", msg, w)
				}
			}
		})
	}
}

// TestNewRunner_DuplicateFreeListsDispatchWhatTheyAdvertise pins the T-1
// invariant for every accepted list: each advertised def and the tool the
// Runner dispatches under that name are the same tool, in input order.
func TestNewRunner_DuplicateFreeListsDispatchWhatTheyAdvertise(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := range 200 {
		n := rng.IntN(8)
		names := make([]string, n)
		for i, p := range rng.Perm(n) {
			names[i] = fmt.Sprintf("tool_%d", p)
		}
		tools := entryTools(names...)
		r := NewRunner(newFakeClient(), tools, "sys")

		if len(r.tools.defs) != n {
			t.Fatalf("trial %d: %d defs advertised for %d tools", trial, len(r.tools.defs), n)
		}
		for i, def := range r.tools.defs {
			got, ok := r.tools.lookup(def.Name)
			if !ok {
				t.Fatalf("trial %d: advertised %q has no dispatch target", trial, def.Name)
			}
			if got != tools[i] {
				t.Fatalf("trial %d: def %d (%q) dispatches %v, want tools[%d]", trial, i, def.Name, got.Def(), i)
			}
			if !reflect.DeepEqual(def, tools[i].Def()) {
				t.Fatalf("trial %d: def %d = %+v, want %+v", trial, i, def, tools[i].Def())
			}
		}
	}
}

// TestRunJSON_RejectsUnfillableOutBeforeAnyCompletion pins T-2: an out that
// cannot receive the answer fails at entry, with no client call and no event.
func TestRunJSON_RejectsUnfillableOutBeforeAnyCompletion(t *testing.T) {
	var nilPtr *item
	tests := []struct {
		name string
		out  any
	}{
		{"untyped nil", nil},
		{"struct value", item{}},
		{"map value", map[string]any{}},
		{"string value", "x"},
		{"nil pointer", nilPtr},
		{"nil map pointer", (*map[string]any)(nil)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fc := newFakeClient(textResp(`{"path":"a","note":"b"}`, 1, 1))
			var starts int
			obs := llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
				if ev.Kind == llmkit.KindStart {
					starts++
				}
			})
			r := NewRunner(fc, nil, "sys", WithObserver(obs))

			out, err := r.RunJSON(context.Background(), "task", nil, tc.out)
			if err == nil {
				t.Fatal("RunJSON returned nil error")
			}
			if !strings.Contains(err.Error(), "non-nil pointer") {
				t.Errorf("error %q does not name the requirement", err)
			}
			if out != nil {
				t.Errorf("Outcome = %+v, want nil (no run happened)", out)
			}
			if n := fc.callCount(); n != 0 {
				t.Errorf("client saw %d completions, want 0", n)
			}
			if starts != 0 {
				t.Errorf("%d Start events, want 0", starts)
			}
		})
	}
}

// TestRunJSON_StartRecordsRawTask pins T-3: the Start event carries the
// caller's task, while the model's first user message still carries the JSON
// instruction appended to it.
func TestRunJSON_StartRecordsRawTask(t *testing.T) {
	fc := newFakeClient(textResp(`{"path":"a.go","note":"n"}`, 1, 1))
	r := NewRunner(fc, nil, "sys")

	const task = "summarize the report"
	var got item
	out, err := r.RunJSON(context.Background(), task, SchemaOf[item](), &got)
	if err != nil {
		t.Fatalf("RunJSON: %v", err)
	}

	evs, err := out.Transcript.Events(context.Background(), out.Transcript.RunID)
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if len(evs) == 0 || evs[0].Kind != llmkit.KindStart || evs[0].Start == nil {
		t.Fatalf("first event is not a Start: %+v", evs)
	}
	if got := evs[0].Start.Task; got != task {
		t.Errorf("Start.Task = %q, want the raw task %q", got, task)
	}

	if len(fc.requests) != 1 || len(fc.requests[0].Messages) == 0 {
		t.Fatalf("requests = %+v", fc.requests)
	}
	first := fc.requests[0].Messages[0]
	if first.Role != llmkit.RoleUser {
		t.Fatalf("first wire message role = %q, want user", first.Role)
	}
	if text := first.Text(); !strings.HasPrefix(text, task) || !strings.Contains(text, jsonInstruction(SchemaOf[item]())) {
		t.Errorf("first wire message = %q, want task followed by the JSON instruction", text)
	}
}

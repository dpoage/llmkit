package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

func deepArray(n int) json.RawMessage {
	return json.RawMessage(strings.Repeat("[", n) + strings.Repeat("]", n))
}

func closureStart(run llmkit.RunID) llmkit.Event {
	ev := llmkit.NewEvent(llmkit.WithRun(context.Background(), run), llmkit.KindStart)
	ev.Start = &llmkit.StartEvent{Task: "t"}
	return ev
}

func closureFinalize(run llmkit.RunID) llmkit.Event {
	ev := llmkit.NewEvent(llmkit.WithRun(context.Background(), run), llmkit.KindFinalize)
	ev.Finalize = &llmkit.FinalizeEvent{Status: llmkit.RunCompleted}
	return ev
}

func closureToolRun(run llmkit.RunID, mut func(*llmkit.ToolRunEvent)) llmkit.Event {
	ev := llmkit.NewEvent(llmkit.WithRun(context.Background(), run), llmkit.KindToolRun)
	ev.Step = 1
	ev.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c1", Name: "echo"}, Result: "ok"}
	if mut != nil {
		mut(ev.ToolRun)
	}
	return ev
}

func closureDecision(run llmkit.RunID, mut func(*llmkit.DecisionEvent)) llmkit.Event {
	ev := llmkit.NewEvent(llmkit.WithRun(context.Background(), run), llmkit.KindDecision)
	ev.Step = 1
	ev.Decision = &llmkit.DecisionEvent{Backend: "b"}
	if mut != nil {
		mut(ev.Decision)
	}
	return ev
}

func closureCompletion(run llmkit.RunID, mut func(*llmkit.CompletionEvent)) llmkit.Event {
	ev := llmkit.NewEvent(llmkit.WithRun(context.Background(), run), llmkit.KindCompletion)
	ev.Step = 1
	ev.Completion = &llmkit.CompletionEvent{}
	if mut != nil {
		mut(ev.Completion)
	}
	return ev
}

// canonOf is json.Marshal of a fresh value decoded from json.Marshal(ev).
func canonOf(t *testing.T, ev llmkit.Event) []byte {
	t.Helper()
	b, err := json.Marshal(&ev)
	if err != nil {
		t.Fatalf("canon marshal: %v", err)
	}
	var fresh llmkit.Event
	if err := json.Unmarshal(b, &fresh); err != nil {
		t.Fatalf("canon unmarshal: %v", err)
	}
	out, err := json.Marshal(&fresh)
	if err != nil {
		t.Fatalf("canon remarshal: %v", err)
	}
	return out
}

func requireIdentical(t *testing.T, label string, read, observed llmkit.Event) {
	t.Helper()
	got, err := json.Marshal(&read)
	if err != nil {
		t.Fatalf("%s: marshal read event: %v", label, err)
	}
	if want := canonOf(t, observed); !bytes.Equal(got, want) {
		t.Errorf("%s: read-back not json-identical (%d vs %d bytes)", label, len(got), len(want))
	}
}

type closureSink struct {
	sink *JSONLSink
	dir  string
	mu   sync.Mutex
	errs []error
}

func newClosureSink(t *testing.T) *closureSink {
	t.Helper()
	cs := &closureSink{dir: t.TempDir()}
	cs.sink = JSONL(cs.dir, func(err error) {
		cs.mu.Lock()
		defer cs.mu.Unlock()
		cs.errs = append(cs.errs, err)
	})
	return cs
}

func (cs *closureSink) errors() []error {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return append([]error(nil), cs.errs...)
}

// closureOutcome observes Start, probe, a marker event and Finalize on a
// fresh sink and checks the sink's contract: either probe's line is written
// and read back json-identical, or nothing is written for it, onErr names the
// run and kind exactly once, and every other event reads back.
func closureOutcome(t *testing.T, label string, probe func(run llmkit.RunID) llmkit.Event, wantWritten bool) {
	t.Helper()
	const run llmkit.RunID = "closure-run"
	cs := newClosureSink(t)
	ctx := context.Background()
	p := probe(run)
	marker := closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Result = "marker" })
	observed := []llmkit.Event{closureStart(run), p, marker, closureFinalize(run)}
	for _, ev := range observed {
		cs.sink.Observe(ctx, ev)
	}

	want := observed
	errs := cs.errors()
	if wantWritten {
		if len(errs) != 0 {
			t.Fatalf("%s: onErr called %d times, first: %v", label, len(errs), errs[0])
		}
	} else {
		if len(errs) != 1 {
			t.Fatalf("%s: onErr called %d times, want exactly 1: %v", label, len(errs), errs)
		}
		msg := strings.ReplaceAll(errs[0].Error(), filepath.Join(cs.dir, string(run)+".jsonl"), "<path>") // the path embeds the test name and run id
		if !strings.Contains(msg, string(run)) || !strings.Contains(msg, string(p.Kind)) {
			t.Errorf("%s: onErr message %q does not name run %q and kind %q", label, msg, run, p.Kind)
		}
		want = []llmkit.Event{observed[0], observed[2], observed[3]}
	}

	got, err := cs.sink.Events(ctx, run)
	if err != nil {
		t.Fatalf("%s: Events: %v", label, err)
	}
	if len(got) != len(want) {
		t.Fatalf("%s: Events returned %d events, want %d", label, len(got), len(want))
	}
	for i := range want {
		requireIdentical(t, fmt.Sprintf("%s: Events[%d]", label, i), got[i], want[i])
	}

	f, err := os.Open(filepath.Join(cs.dir, string(run)+".jsonl"))
	if err != nil {
		t.Fatalf("%s: open: %v", label, err)
	}
	t.Cleanup(func() { _ = f.Close() })
	tr, err := LoadJSONL(f)
	if err != nil {
		t.Fatalf("%s: LoadJSONL: %v", label, err)
	}
	if len(tr.Record) != len(want) {
		t.Fatalf("%s: LoadJSONL returned %d events, want %d", label, len(tr.Record), len(want))
	}
	for i := range want {
		requireIdentical(t, fmt.Sprintf("%s: LoadJSONL[%d]", label, i), tr.Record[i], want[i])
	}
}

// closureCarrier builds an event holding a nested JSON value of the given
// depth in one carrier field.
type closureCarrier struct {
	name  string
	build func(run llmkit.RunID, depth int) llmkit.Event
}

var closureCarriers = []closureCarrier{
	{"decision State", func(run llmkit.RunID, d int) llmkit.Event {
		return closureDecision(run, func(de *llmkit.DecisionEvent) { de.State = deepArray(d) })
	}},
	{"decision Questions[].Instructions", func(run llmkit.RunID, d int) llmkit.Event {
		return closureDecision(run, func(de *llmkit.DecisionEvent) {
			de.Questions = []llmkit.DecisionQuestion{{ID: "q", Kind: "noul", Instructions: deepArray(d)}}
		})
	}},
	{"tool_run Call.Arguments", func(run llmkit.RunID, d int) llmkit.Event {
		return closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Call.Arguments = deepArray(d) })
	}},
	{"tool_run DispatchedArguments", func(run llmkit.RunID, d int) llmkit.Event {
		return closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.DispatchedArguments = deepArray(d) })
	}},
	{"completion Response.ToolCalls[].Arguments", func(run llmkit.RunID, d int) llmkit.Event {
		return closureCompletion(run, func(ce *llmkit.CompletionEvent) {
			ce.Response.ToolCalls = []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: deepArray(d)}}
		})
	}},
	{"completion Request.Messages[].ToolCalls[].Arguments", func(run llmkit.RunID, d int) llmkit.Event {
		return closureCompletion(run, func(ce *llmkit.CompletionEvent) {
			ce.Request.Messages = []llmkit.Message{{
				Role:      llmkit.RoleAssistant,
				ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: deepArray(d)}},
			}}
		})
	}},
}

// lineDepthOffset measures how many wrapper levels sit above the carrier's
// value: the first field depth whose encoded line encoding/json refuses to
// scan is line depth 10001.
func lineDepthOffset(t *testing.T, c closureCarrier) int {
	t.Helper()
	for f := 9970; f <= 10000; f++ {
		ev := c.build("measure", f)
		line, err := json.Marshal(&ev)
		if err != nil {
			t.Fatalf("%s: measuring marshal at field depth %d: %v", c.name, f, err)
		}
		if !json.Valid(line) {
			return 10001 - f
		}
	}
	t.Fatalf("%s: no field depth up to 10000 pushes the line past the decode limit", c.name)
	return 0
}

func TestClosure_DepthWitnessesTakeWrittenOrRefused(t *testing.T) {
	for _, c := range closureCarriers {
		t.Run(c.name, func(t *testing.T) {
			off := lineDepthOffset(t, c)
			t.Logf("wrapper levels = %d", off)
			for _, line := range []int{9999, 10000, 10001, 10002} {
				t.Run(fmt.Sprintf("line depth %d", line), func(t *testing.T) {
					closureOutcome(t, "witness", func(run llmkit.RunID) llmkit.Event {
						return c.build(run, line-off)
					}, line <= 10000)
				})
			}
		})
	}
}

func TestClosure_BeyondValidJSONArgumentsRideBase64AndAreWritten(t *testing.T) {
	for _, d := range []int{10001, 20000} {
		closureOutcome(t, fmt.Sprintf("tool_run Call.Arguments field depth %d", d), func(run llmkit.RunID) llmkit.Event {
			return closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Call.Arguments = deepArray(d) })
		}, true)
	}
}

func TestClosure_StateTooDeepToMarshalIsRefusedNamingKind(t *testing.T) {
	closureOutcome(t, "decision State 10001", func(run llmkit.RunID) llmkit.Event {
		return closureDecision(run, func(de *llmkit.DecisionEvent) { de.State = deepArray(10001) })
	}, false)
}

func TestClosure_NaNConfidenceIsRefused(t *testing.T) {
	closureOutcome(t, "decision NaN confidence", func(run llmkit.RunID) llmkit.Event {
		return closureDecision(run, func(de *llmkit.DecisionEvent) {
			de.Answers = []llmkit.DecisionAnswer{{ID: "q", Choice: "a", Confidence: math.NaN()}}
		})
	}, false)
}

func schemaZero(ev llmkit.Event) llmkit.Event {
	ev.SchemaVersion = 0
	return ev
}

func toolRunWithDecision(run llmkit.RunID) llmkit.Event {
	ev := closureToolRun(run, nil)
	ev.Decision = &llmkit.DecisionEvent{Backend: "b"}
	return ev
}

func TestClosure_EventsNoReaderAcceptsAreRefused(t *testing.T) {
	for name, probe := range map[string]func(llmkit.RunID) llmkit.Event{
		"tool_run with schema_version 0":       func(run llmkit.RunID) llmkit.Event { return schemaZero(closureToolRun(run, nil)) },
		"tool_run carrying a Decision payload": toolRunWithDecision,
	} {
		closureOutcome(t, name, probe, false)
	}
}

func TestClosure_RefusedStartLeavesNoFileAndEveryLaterEventReports(t *testing.T) {
	ctx := context.Background()
	for name, tc := range map[string]struct {
		run   llmkit.RunID
		start func(llmkit.RunID) llmkit.Event
	}{
		"schema_version 0":     {"refused-start", func(run llmkit.RunID) llmkit.Event { return schemaZero(closureStart(run)) }},
		"invalid UTF-8 run id": {"r\xff", closureStart},
	} {
		t.Run(name, func(t *testing.T) {
			cs := newClosureSink(t)
			observed := []llmkit.Event{tc.start(tc.run), closureToolRun(tc.run, nil), closureFinalize(tc.run)}
			for _, ev := range observed {
				cs.sink.Observe(ctx, ev)
			}
			errs := cs.errors()
			if len(errs) != len(observed) {
				t.Fatalf("onErr called %d times, want %d (one per event): %v", len(errs), len(observed), errs)
			}
			for i, err := range errs {
				if msg := err.Error(); !strings.Contains(msg, "run "+string(tc.run)) || !strings.Contains(msg, "kind "+string(observed[i].Kind)) {
					t.Errorf("onErr[%d] %q does not name run %q and kind %q", i, msg, tc.run, observed[i].Kind)
				}
			}
			entries, err := os.ReadDir(cs.dir)
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range entries {
				t.Errorf("refused Start left %s behind", e.Name())
			}
			if _, err := cs.sink.Events(ctx, tc.run); !errors.Is(err, llmkit.ErrUnknownRun) {
				t.Errorf("Events = %v, want ErrUnknownRun", err)
			}
		})
	}
}

func TestClosure_RefusedFinalizeStillClosesTheRun(t *testing.T) {
	const run llmkit.RunID = "refused-finalize"
	ctx := context.Background()
	cs := newClosureSink(t)
	written := []llmkit.Event{closureStart(run), closureToolRun(run, nil)}
	for _, ev := range append(written, schemaZero(closureFinalize(run))) {
		cs.sink.Observe(ctx, ev)
	}
	errs := cs.errors()
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "run "+string(run)) || !strings.Contains(errs[0].Error(), "kind finalize") {
		t.Fatalf("onErr = %v, want one error naming run %q and kind finalize", errs, run)
	}
	got, err := cs.sink.Events(ctx, run)
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if len(got) != len(written) {
		t.Fatalf("Events returned %d events, want %d", len(got), len(written))
	}
	for i := range written {
		requireIdentical(t, fmt.Sprintf("Events[%d]", i), got[i], written[i])
	}
	cs.sink.Observe(ctx, closureToolRun(run, nil))
	if errs := cs.errors(); len(errs) != 2 || !strings.Contains(errs[1].Error(), "unknown or closed run") {
		t.Errorf("event after the refused Finalize: onErr = %v, want the run closed", errs)
	}
}

func TestClosure_BracketsInsideStringsAreNotDepth(t *testing.T) {
	for name, result := range map[string]string{
		"10001 [ in a string":        strings.Repeat("[", 10001),
		"escaped quote then 10001 [": `"` + strings.Repeat("[", 10001),
	} {
		closureOutcome(t, name, func(run llmkit.RunID) llmkit.Event {
			return closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Result = result })
		}, true)
	}
}

func TestClosure_HugeLinesAreWrittenAndLoad(t *testing.T) {
	const run llmkit.RunID = "huge-run"
	for _, size := range []int{16<<20 + 1, 64 << 20} {
		t.Run(fmt.Sprintf("%d bytes", size), func(t *testing.T) {
			ev := closureToolRun(run, nil)
			base, err := json.Marshal(&ev)
			if err != nil {
				t.Fatal(err)
			}
			ev.ToolRun.Result = strings.Repeat("a", size-len(base)+len("ok")) // replaces the 2-byte "ok"
			line, err := json.Marshal(&ev)
			if err != nil {
				t.Fatal(err)
			}
			if len(line) != size {
				t.Fatalf("line is %d bytes, want %d", len(line), size)
			}

			cs := newClosureSink(t)
			ctx := context.Background()
			cs.sink.Observe(ctx, closureStart(run))
			cs.sink.Observe(ctx, ev)
			cs.sink.Observe(ctx, closureFinalize(run))
			if errs := cs.errors(); len(errs) != 0 {
				t.Fatalf("onErr: %v", errs[0])
			}
			got, err := cs.sink.Events(ctx, run)
			if err != nil {
				t.Fatalf("Events: %v", err)
			}
			if len(got) != 3 {
				t.Fatalf("Events returned %d events, want 3", len(got))
			}
			// ASCII payload: canon(ev) == json.Marshal(ev), so compare lines.
			readLine, err := json.Marshal(&got[1])
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(readLine, line) {
				t.Errorf("huge line read back differs (%d vs %d bytes)", len(readLine), len(line))
			}
		})
	}
}

func TestClosure_SaveJSONLWritesPrefixAndNamesTheRefusedEvent(t *testing.T) {
	const run llmkit.RunID = "save-run"
	bad := map[string]llmkit.Event{
		"over-depth tool_run": closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Call.Arguments = deepArray(9999) }),
		"unmarshalable decision": closureDecision(run, func(de *llmkit.DecisionEvent) {
			de.Answers = []llmkit.DecisionAnswer{{ID: "q", Confidence: math.NaN()}}
		}),
		"State 10001 decision":                 closureDecision(run, func(de *llmkit.DecisionEvent) { de.State = deepArray(10001) }),
		"schema_version 0 tool_run":            schemaZero(closureToolRun(run, nil)),
		"tool_run carrying a Decision payload": toolRunWithDecision(run),
		"invalid UTF-8 run id":                 closureToolRun(run+"\xff", nil),
	}
	for name, badEv := range bad {
		t.Run(name, func(t *testing.T) {
			good := []llmkit.Event{closureStart(run), closureToolRun(run, nil)}
			tr := &Transcript{RunID: run, Record: append(append([]llmkit.Event(nil), good...), badEv, closureFinalize(run))}
			var buf bytes.Buffer
			err := tr.SaveJSONL(&buf)
			if err == nil {
				t.Fatal("SaveJSONL returned nil for a line it cannot make loadable")
			}
			for _, want := range []string{"event 2", string(run), string(badEv.Kind)} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not name %q", err, want)
				}
			}
			loaded, lerr := LoadJSONL(&buf)
			if lerr != nil {
				t.Fatalf("prefix does not load: %v", lerr)
			}
			if len(loaded.Record) != len(good) {
				t.Fatalf("prefix holds %d events, want %d", len(loaded.Record), len(good))
			}
			for i := range good {
				requireIdentical(t, fmt.Sprintf("prefix[%d]", i), loaded.Record[i], good[i])
			}
		})
	}
}

func TestClosure_SaveJSONLWritesEveryLoadableEvent(t *testing.T) {
	const run llmkit.RunID = "save-ok"
	rec := []llmkit.Event{
		closureStart(run),
		closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Call.Arguments = json.RawMessage(`{"p":`) }),
		closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Call.Arguments = deepArray(10000 - 3) }),
		closureDecision(run, func(de *llmkit.DecisionEvent) { de.State = deepArray(10000 - 2) }),
		closureFinalize(run),
	}
	var buf bytes.Buffer
	if err := (&Transcript{Record: rec}).SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	loaded, err := LoadJSONL(&buf)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	if len(loaded.Record) != len(rec) {
		t.Fatalf("loaded %d events, want %d", len(loaded.Record), len(rec))
	}
	for i := range rec {
		requireIdentical(t, fmt.Sprintf("event %d", i), loaded.Record[i], rec[i])
	}
}

func TestClosure_LoadJSONLKeepsLineNumberedErrors(t *testing.T) {
	good, err := json.Marshal(func() *llmkit.Event { ev := closureStart("r"); return &ev }())
	if err != nil {
		t.Fatal(err)
	}
	_, err = LoadJSONL(strings.NewReader(string(good) + "\n{not json}\n"))
	if err == nil || !strings.Contains(err.Error(), "decode transcript line 2") {
		t.Errorf("decode error = %v, want it to name line 2", err)
	}
	_, err = LoadJSONL(strings.NewReader(string(good) + "\n\n" + `{"kind":"tool_run","step":1,"tool_run":{"call":{"id":"c","name":"n"}}}` + "\n"))
	if err == nil || !strings.Contains(err.Error(), "transcript line 3") {
		t.Errorf("validate error = %v, want it to name line 3", err)
	}
}

func TestClosure_LoadJSONLReadsAFinalLineWithoutNewline(t *testing.T) {
	ev := closureStart("r")
	b, err := json.Marshal(&ev)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := LoadJSONL(bytes.NewReader(b))
	if err != nil || len(tr.Record) != 1 {
		t.Fatalf("LoadJSONL = %v, %v; want one event", tr, err)
	}
}

func TestClosure_FixtureReencodesByteIdentical(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "base_record_denial_parallel.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	tr, err := LoadJSONL(bytes.NewReader(want))
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	var got bytes.Buffer
	if err := tr.SaveJSONL(&got); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	if !bytes.Equal(got.Bytes(), want) {
		t.Errorf("fixture re-encodes differently:\n got %s\nwant %s", got.Bytes(), want)
	}
}

// closureArgsCorpus: arguments A that are not valid JSON, and valid ones.
var closureArgsCorpus = []struct {
	name  string
	args  string
	valid bool
}{
	{"truncated object", `{"p":`, false},
	{"invalid utf8", string([]byte{0xff, 0xfe, 'a'}), false},
	{"empty non-nil (omitted, as nil is)", "", false},
	{"string literal", `""`, true},
	{"object", `{"p":"a<b"}`, true},
}

func TestClosure_InvalidToolArgumentsRoundTripThroughEveryCarrier(t *testing.T) {
	carriers := map[string]func(run llmkit.RunID, a json.RawMessage) llmkit.Event{
		"tool_run": func(run llmkit.RunID, a json.RawMessage) llmkit.Event {
			return closureToolRun(run, func(tre *llmkit.ToolRunEvent) { tre.Call.Arguments = a })
		},
		"completion response": func(run llmkit.RunID, a json.RawMessage) llmkit.Event {
			return closureCompletion(run, func(ce *llmkit.CompletionEvent) {
				ce.Response.ToolCalls = []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: a}}
			})
		},
		"completion request": func(run llmkit.RunID, a json.RawMessage) llmkit.Event {
			return closureCompletion(run, func(ce *llmkit.CompletionEvent) {
				ce.Request.Messages = []llmkit.Message{{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: a}}}}
			})
		},
	}
	for cname, build := range carriers {
		for _, tc := range closureArgsCorpus {
			label := cname + "/" + tc.name
			closureOutcome(t, label, func(run llmkit.RunID) llmkit.Event {
				return build(run, json.RawMessage(tc.args))
			}, true)

			const run llmkit.RunID = "args-run"
			cs := newClosureSink(t)
			for _, ev := range []llmkit.Event{closureStart(run), build(run, json.RawMessage(tc.args)), closureFinalize(run)} {
				cs.sink.Observe(context.Background(), ev)
			}
			evs, err := cs.sink.Events(context.Background(), run)
			if err != nil {
				t.Fatalf("%s: Events: %v", label, err)
			}
			var read json.RawMessage
			switch cname {
			case "tool_run":
				read = evs[1].ToolRun.Call.Arguments
			case "completion response":
				read = evs[1].Completion.Response.ToolCalls[0].Arguments
			default:
				read = evs[1].Completion.Request.Messages[0].ToolCalls[0].Arguments
			}
			if !tc.valid && !bytes.Equal(read, []byte(tc.args)) {
				t.Errorf("%s: arguments read back %q, want byte-equal %q", label, read, tc.args)
			}
		}
	}
}

// argRecorder is a tool that keeps the bytes it was handed.
type argRecorder struct {
	mu   *sync.Mutex
	args *[][]byte
}

func (argRecorder) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "echo", Description: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (a argRecorder) Run(_ context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	*a.args = append(*a.args, bytes.Clone(args))
	return textBlocks("ran"), nil
}

func TestRunner_TruncatedToolArgumentsAreRecordedAndReplay(t *testing.T) {
	cs := newClosureSink(t)
	var mu sync.Mutex
	var got [][]byte
	tool := argRecorder{mu: &mu, args: &got}
	r := NewRunner(newFakeClient(
		toolResp("c1", "echo", `{"p":`, 10, 4),
		textResp("done", 8, 3),
	), []Tool{tool}, "sys", WithObserver(cs.sink))
	out, err := r.Run(context.Background(), "task", WithRunID("bad-args"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if errs := cs.errors(); len(errs) != 0 {
		t.Fatalf("sink reported %d errors, first: %v", len(errs), errs[0])
	}
	if len(got) != 1 || string(got[0]) != `{"p":` {
		t.Fatalf("tool received %q, want [{\"p\":]", got)
	}

	raw, err := os.ReadFile(filepath.Join(cs.dir, "bad-args.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[llmkit.EventKind]int{}
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var head struct {
			Kind llmkit.EventKind `json:"kind"`
		}
		if err := json.Unmarshal(line, &head); err != nil {
			t.Fatalf("line %q: %v", line, err)
		}
		kinds[head.Kind]++
	}
	if kinds[llmkit.KindCompletion] != 2 || kinds[llmkit.KindToolRun] != 1 {
		t.Fatalf("file holds %v, want 2 completion lines and 1 tool_run line", kinds)
	}

	rc, err := NewReplayClient(cs.sink, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	replayed, err := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("replay Run: %v", err)
	}
	if err := rc.Err(); err != nil {
		t.Fatalf("rc.Err() = %v, want nil", err)
	}
	var replayArgs []byte
	for _, ev := range replayed.Transcript.Record {
		if ev.Kind == llmkit.KindToolRun {
			replayArgs = ev.ToolRun.Call.Arguments
		}
	}
	if string(replayArgs) != `{"p":` {
		t.Errorf("replayed tool call carries %q, want {\"p\":", replayArgs)
	}
}

func TestReplay_VerifiesCompactionWithMalformedToolArguments(t *testing.T) {
	steps := overflowTurns(8)
	for i := range steps {
		steps[i].resp.ToolCalls[0].Arguments = json.RawMessage(`{bad`)
	}
	out, err := overflowRunner(newFakeClient(append(steps, textResp("done", 100, 10))...), 1500).Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if n := len(compactionEvents(out.Transcript.Record)); n < 1 {
		t.Fatalf("record holds %d compactions, want at least 1", n)
	}
	if _, err := json.Marshal(out.Messages); err != nil {
		t.Fatalf("history with malformed tool arguments does not encode: %v", err)
	}
	if err := seamReplay(t, out, 1500).Err(); err != nil {
		t.Errorf("same config: rc.Err = %v, want nil", err)
	}
	err = seamReplay(t, out, 0).Err()
	if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "threshold compaction") {
		t.Errorf("compaction off: rc.Err = %v, want ErrReplayDiverged naming the threshold compaction", err)
	}
}

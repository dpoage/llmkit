package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

type w0bErrTool struct {
	name string
	err  error
}

func (e w0bErrTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: e.name, Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (e w0bErrTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	return nil, e.err
}

type w0bPanicTool struct {
	name string
	val  any
}

func (p w0bPanicTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: p.name, Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (p w0bPanicTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	panic(p.val)
}

// w0bToolRunOf returns the run's single ToolRun event.
func w0bToolRunOf(t *testing.T, evs []llmkit.Event) *llmkit.ToolRunEvent {
	t.Helper()
	runs := rdToolRuns(evs)
	if len(runs) != 1 {
		t.Fatalf("ToolRun events = %d, want 1", len(runs))
	}
	return runs[0].ToolRun
}

// w0bLoadRun reads the run's JSONL file back through LoadJSONL.
func w0bLoadRun(t *testing.T, dir string, run llmkit.RunID) *Transcript {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, string(run)+".jsonl"))
	if err != nil {
		t.Fatalf("read jsonl: %v", err)
	}
	tr, err := LoadJSONL(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	return tr
}

// w0bCanon is common.md's canon: json.Marshal of the value decoded from its
// own JSON encoding.
func w0bCanon(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	cb, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("remarshal: %v", err)
	}
	return string(cb)
}

func TestW0b_ToolOutcomeInvalidUTF8Agrees(t *testing.T) {
	raw := "\xff\xfea"
	want := "\ufffd\ufffda"
	cases := []struct {
		name string
		tool Tool
		// wantBlocks are the blocks placed in history when the tool returned
		// some; wantContent is the event's Content (nil unless not one text block).
		wantResult  string
		wantBlocks  []llmkit.Block
		wantContent []llmkit.Block
		wantIsError bool
	}{
		{
			name:        "result text",
			tool:        blocksTool{"blob", []llmkit.Block{llmkit.Text(raw)}},
			wantResult:  want,
			wantBlocks:  []llmkit.Block{llmkit.Text(want)},
			wantIsError: false,
		},
		{
			name:        "result multi text",
			tool:        blocksTool{"blob", []llmkit.Block{llmkit.Text("x\xffy"), llmkit.Text("z\xfe")}},
			wantResult:  "x\ufffdyz\ufffd",
			wantBlocks:  []llmkit.Block{llmkit.Text("x\ufffdy"), llmkit.Text("z\ufffd")},
			wantContent: []llmkit.Block{llmkit.Text("x\ufffdy"), llmkit.Text("z\ufffd")},
			wantIsError: false,
		},
		{
			name:        "run error",
			tool:        w0bErrTool{"blob", errors.New("disk on fire \xff")},
			wantResult:  "ERROR: disk on fire \ufffd",
			wantIsError: true,
		},
		{
			name:        "panic",
			tool:        w0bPanicTool{"blob", "bang \xff"},
			wantResult:  "ERROR: tool blob panicked: bang \ufffd",
			wantIsError: true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			run := llmkit.RunID("w0b-clu-" + c.name)
			dir := t.TempDir()
			fc := newFakeClient(toolResp("c1", "blob", `{}`, 1, 1), textResp("done", 1, 1))
			r := NewRunner(fc, []Tool{c.tool}, "sys", WithObserver(JSONL(dir, func(e error) {
				t.Errorf("sink: %v", e)
			})))
			out, err := r.Run(context.Background(), "task", WithRunID(run))
			if err != nil {
				t.Fatalf("Run: %v", err)
			}

			hist := toolMessage(t, out.Messages)
			if got := hist.Text(); got != c.wantResult {
				t.Errorf("history text = %q, want %q", got, c.wantResult)
			}
			if !c.wantIsError && !reflect.DeepEqual(hist.Content, c.wantBlocks) {
				t.Errorf("history blocks = %+v, want %+v", hist.Content, c.wantBlocks)
			}

			ev := w0bToolRunOf(t, out.Transcript.Record)
			if ev.Result != c.wantResult {
				t.Errorf("ToolRun.Result = %q, want %q", ev.Result, c.wantResult)
			}
			if !reflect.DeepEqual(ev.Content, c.wantContent) {
				t.Errorf("ToolRun.Content = %+v, want %+v", ev.Content, c.wantContent)
			}
			if ev.IsError != c.wantIsError {
				t.Errorf("ToolRun.IsError = %v, want %v", ev.IsError, c.wantIsError)
			}

			loaded := w0bLoadRun(t, dir, run)
			lev := w0bToolRunOf(t, loaded.Record)
			if lev.Result != c.wantResult {
				t.Errorf("record Result = %q, want %q (history/record disagree)", lev.Result, c.wantResult)
			}
			if w0bCanon(t, lev) != w0bCanon(t, ev) {
				t.Errorf("record ToolRun = %s, want canon-equal to %s", w0bCanon(t, lev), w0bCanon(t, ev))
			}
		})
	}
}

func TestW0b_NextRequestDecodedEqualsBase(t *testing.T) {
	raw := "\xff\xfea"
	want := "\ufffd\ufffda"
	fc := newFakeClient(toolResp("c1", "blob", `{}`, 1, 1), textResp("done", 1, 1))
	_, err := NewRunner(fc, []Tool{blocksTool{"blob", []llmkit.Block{llmkit.Text(raw)}}}, "sys").
		Run(context.Background(), "task", WithRunID("w0b-wire"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fc.requests))
	}
	got := fc.requests[1]
	if tr := toolMessage(t, got.Messages); tr.Text() != want {
		t.Errorf("next request tool result = %q, want the normalized %q", tr.Text(), want)
	}

	// Base kept the raw bytes in history; rewrite them back into a copy of the
	// request and compare the DECODED forms: the model must see the same turn.
	base := got
	base.Messages = slicesCloneMessages(got.Messages)
	for i := range base.Messages {
		if base.Messages[i].Role == llmkit.RoleToolResult {
			base.Messages[i].Content = []llmkit.Block{llmkit.Text(raw)}
		}
	}
	gotDecoded := decodeJSONValue(t, got)
	baseDecoded := decodeJSONValue(t, base)
	if !reflect.DeepEqual(gotDecoded, baseDecoded) {
		t.Errorf("decoded next request differs from base's:\n got: %s\nbase: %s",
			w0bCanon(t, got), w0bCanon(t, base))
	}
}

func slicesCloneMessages(msgs []llmkit.Message) []llmkit.Message {
	out := make([]llmkit.Message, len(msgs))
	copy(out, msgs)
	return out
}

func decodeJSONValue(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return out
}

// w0bProbeBlob is the bead llmkit-clu probe payload: 18000 raw bytes whose
// 2000 invalid \xff bytes normalize to 22000 bytes.
func w0bProbeBlob() string {
	return strings.Repeat("v", 16000) + strings.Repeat("\xff", 2000)
}

func TestW0b_CompactionProbeInvalidUTF8Replays(t *testing.T) {
	blob := w0bProbeBlob()
	run := llmkit.RunID("w0b-probe")
	dir := t.TempDir()
	steps := []scriptStep{
		toolResp("c1", "big", `{"v":1}`, 100, 10),
		toolResp("c2", "big", `{"v":2}`, 100, 10),
		toolResp("c3", "big", `{"v":3}`, 100, 10),
		toolResp("c4", "big", `{"v":4}`, 100, 10),
		toolResp("c5", "big", `{"v":5}`, 100, 10),
		toolResp("c6", "big", `{"v":6}`, 100, 10),
		textResp("done", 100, 10),
	}
	fc := newFakeClient(steps...)
	limits := Limits{MaxIterations: 30, HistoryTokenBudget: 8000}
	r := NewRunner(fc, []Tool{bigEchoTool{name: "big", blob: blob}}, "sys",
		WithLimits(limits), WithObserver(JSONL(dir, func(e error) { t.Errorf("sink: %v", e) })))
	out, err := r.Run(context.Background(), "task", WithRunID(run))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q", out.FinalText)
	}

	// The first compaction stubs the first result and reports its NORMALIZED
	// size (22000), which is what the record carries too.
	var compacted []llmkit.CompactionEvent
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompaction {
			compacted = append(compacted, *ev.Compaction)
		}
	}
	if len(compacted) == 0 {
		t.Fatal("no compaction fired")
	}
	stub := fmt.Sprintf("[tool result pruned to save context: big, %d bytes]", 22000)
	found := false
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleToolResult && m.Text() == stub {
			found = true
		}
	}
	if !found {
		t.Errorf("history holds no %q stub (normalized size not counted)", stub)
	}

	// History text and record bytes agree for the full (unstubbed) results.
	loaded := w0bLoadRun(t, dir, run)
	var histFull, recFull string
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleToolResult && len(m.Text()) == 22000 {
			histFull = m.Text()
		}
	}
	for _, ev := range rdToolRuns(loaded.Record) {
		if len(ev.ToolRun.Result) == 22000 {
			recFull = ev.ToolRun.Result
		}
	}
	if histFull == "" || histFull != recFull {
		t.Errorf("history full result (%d bytes) vs record (%d bytes) disagree", len(histFull), len(recFull))
	}

	// The bead repro: replaying the JSONL record under the same limits used to
	// diverge at the compaction point (18000 vs 22000 bytes in the stub).
	src := JSONL(dir, nil)
	rc, err := NewReplayClient(src, run, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	rout, rerr := NewRunner(rc, rc.Tools(), "sys", WithLimits(limits)).
		Run(context.Background(), "task")
	if rerr != nil {
		t.Fatalf("replayed Run: %v", rerr)
	}
	if derr := rc.Err(); derr != nil {
		t.Fatalf("rc.Err: %v", derr)
	}
	if rout.FinalText != "done" {
		t.Errorf("replayed FinalText = %q", rout.FinalText)
	}
}

func TestW0b_SpillRefInvalidUTF8IsSinkFailure(t *testing.T) {
	const limit = 1024
	big := strings.Repeat("s", 5000)
	run := llmkit.RunID("w0b-spill")
	dir := t.TempDir()
	sink := spillSinkFunc(func(_ context.Context, _ llmkit.ToolCall, _ []llmkit.Block) (string, error) {
		return "a\xffb", nil
	})
	fc := newFakeClient(toolResp("c1", "blob", `{}`, 1, 1), textResp("done", 1, 1))
	r := NewRunner(fc, []Tool{blocksTool{"blob", []llmkit.Block{llmkit.Text(big)}}}, "sys",
		WithToolOutputLimit(limit, sink), WithObserver(JSONL(dir, func(e error) { t.Errorf("sink: %v", e) })))
	out, err := r.Run(context.Background(), "task", WithRunID(run))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	hist := toolMessage(t, out.Messages).Text()
	if !strings.Contains(hist, "[tool output truncated: ") || strings.Contains(hist, "[tool output spilled: ") {
		t.Errorf("history = %q, want the truncation stand-in", hist)
	}
	if !utf8.ValidString(hist) {
		t.Errorf("history holds invalid UTF-8")
	}
	ev := w0bToolRunOf(t, out.Transcript.Record)
	if ev.SpillRef != "" {
		t.Errorf("SpillRef = %q, want empty", ev.SpillRef)
	}
	if ev.SpillError == "" {
		t.Error("SpillError empty, want the unusable-ref failure")
	}
	if ev.OriginalBytes != len(big) {
		t.Errorf("OriginalBytes = %d, want %d", ev.OriginalBytes, len(big))
	}
	if ev.Result != hist {
		t.Errorf("ToolRun.Result != history text: %q vs %q", ev.Result, hist)
	}
	lev := w0bToolRunOf(t, w0bLoadRun(t, dir, run).Record)
	if lev.SpillRef != "" || lev.Result != hist {
		t.Errorf("record SpillRef = %q, Result matches history = %v", lev.SpillRef, lev.Result == hist)
	}
}

type spillSinkFunc func(ctx context.Context, call llmkit.ToolCall, full []llmkit.Block) (string, error)

func (f spillSinkFunc) Spill(ctx context.Context, call llmkit.ToolCall, full []llmkit.Block) (string, error) {
	return f(ctx, call, full)
}

func TestW0b_EmptyCallIDGetsSurrogate(t *testing.T) {
	run := llmkit.RunID("w0b-9y4")
	dir := t.TempDir()
	fc := newFakeClient(toolResp("", "blob", `{}`, 1, 1), textResp("done", 1, 1))
	r := NewRunner(fc, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys",
		WithObserver(JSONL(dir, func(e error) { t.Errorf("sink: %v", e) })))
	out, err := r.Run(context.Background(), "task", WithRunID(run))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	surrogate := "llmkit-w0b-9y4-s1-c0"
	var assistant, result llmkit.Message
	for _, m := range out.Messages {
		switch {
		case m.Role == llmkit.RoleAssistant && len(m.ToolCalls) > 0:
			assistant = m
		case m.Role == llmkit.RoleToolResult:
			result = m
		}
	}
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ID != surrogate {
		t.Errorf("assistant call ID = %+v, want %q", assistant.ToolCalls, surrogate)
	}
	if result.ToolCallID != surrogate {
		t.Errorf("tool result ToolCallID = %q, want %q", result.ToolCallID, surrogate)
	}

	var completion *llmkit.CompletionEvent
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompletion && ev.Completion.Err == "" && len(ev.Completion.Response.ToolCalls) > 0 {
			completion = ev.Completion
		}
	}
	if completion == nil || len(completion.Response.ToolCalls) != 1 || completion.Response.ToolCalls[0].ID != surrogate {
		t.Errorf("Completion event ToolCalls = %+v, want the surrogate %q", completion, surrogate)
	}
	trev := w0bToolRunOf(t, out.Transcript.Record)
	if trev.Call.ID != surrogate {
		t.Errorf("ToolRun Call.ID = %q, want %q", trev.Call.ID, surrogate)
	}

	rc, err := NewReplayClient(JSONL(dir, nil), run, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	rout, rerr := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
	if rerr != nil {
		t.Fatalf("replayed Run: %v", rerr)
	}
	if derr := rc.Err(); derr != nil {
		t.Fatalf("rc.Err: %v", derr)
	}
	rtrev := w0bToolRunOf(t, rout.Transcript.Record)
	if rtrev.Call.ID != surrogate {
		t.Errorf("replayed ToolRun Call.ID = %q, want the recorded surrogate once", rtrev.Call.ID)
	}
}

func TestW0b_EmptyCallIDDeterministicAndSkipsUsed(t *testing.T) {
	// Determinism: the same pinned run id and script produce the same surrogate.
	var ids []string
	for i := range 2 {
		fc := newFakeClient(toolResp("", "blob", `{}`, 1, 1), textResp("done", 1, 1))
		out, err := NewRunner(fc, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys").
			Run(context.Background(), "task", WithRunID("w0b-det"))
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		ids = append(ids, out.Messages[1].ToolCalls[0].ID)
	}
	if ids[0] != ids[1] || ids[0] != "llmkit-w0b-det-s1-c0" {
		t.Errorf("surrogates = %v, want equal llmkit-w0b-det-s1-c0", ids)
	}

	// A Continue seed that pinned the same RunID makes the candidate collide;
	// the skip suffix must fire deterministically.
	first := llmkit.RunID("w0b-dup")
	fc := newFakeClient(toolResp("", "blob", `{}`, 1, 1), textResp("done", 1, 1))
	outA, err := NewRunner(fc, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys").
		Run(context.Background(), "task", WithRunID(first))
	if err != nil {
		t.Fatalf("run A: %v", err)
	}
	if id := outA.Messages[1].ToolCalls[0].ID; id != "llmkit-w0b-dup-s1-c0" {
		t.Fatalf("run A surrogate = %q", id)
	}
	fcB := newFakeClient(toolResp("", "blob", `{}`, 1, 1), textResp("done", 1, 1))
	outB, err := NewRunner(fcB, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys").
		Run(context.Background(), "next task", Continue(outA), WithRunID(first))
	if err != nil {
		t.Fatalf("run B: %v", err)
	}
	var bAssistant llmkit.Message
	for _, m := range outB.Messages {
		if m.Role == llmkit.RoleAssistant && len(m.ToolCalls) > 0 && m.ToolCalls[0].ID != "llmkit-w0b-dup-s1-c0" {
			bAssistant = m
		}
	}
	if bAssistant.ToolCalls == nil || bAssistant.ToolCalls[0].ID != "llmkit-w0b-dup-s1-c0-k1" {
		t.Errorf("run B surrogate = %+v, want llmkit-w0b-dup-s1-c0-k1 (seed skip)", bAssistant.ToolCalls)
	}

	// A vendor ID of the same form and two empties in one response.
	fcC := newFakeClient(
		toolCallsResp(
			llmkit.ToolCall{ID: "llmkit-w0b-v-s1-c0", Name: "blob", Arguments: json.RawMessage(`{}`)},
			llmkit.ToolCall{ID: "", Name: "blob", Arguments: json.RawMessage(`{}`)},
			llmkit.ToolCall{ID: "", Name: "blob", Arguments: json.RawMessage(`{}`)},
		),
		textResp("done", 1, 1),
	)
	outC, err := NewRunner(fcC, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys").
		Run(context.Background(), "task", WithRunID("w0b-v"))
	if err != nil {
		t.Fatalf("run C: %v", err)
	}
	var got [3]string
	for _, m := range outC.Messages {
		if m.Role == llmkit.RoleAssistant {
			for i, tc := range m.ToolCalls {
				got[i] = tc.ID
			}
		}
	}
	want := [3]string{"llmkit-w0b-v-s1-c0", "llmkit-w0b-v-s1-c1", "llmkit-w0b-v-s1-c2"}
	if got != want {
		t.Errorf("IDs = %v, want %v (vendor form untouched, empties indexed)", got, want)
	}
}

func TestW0b_EventContentNotOneTextBlock(t *testing.T) {
	run := llmkit.RunID("w0b-r2")
	dir := t.TempDir()
	fc := newFakeClient(toolResp("c1", "blob", `{}`, 1, 1), textResp("done", 1, 1))
	blocks := []llmkit.Block{llmkit.Text("a"), llmkit.Text("b")}
	r := NewRunner(fc, []Tool{blocksTool{"blob", blocks}}, "sys",
		WithObserver(JSONL(dir, func(e error) { t.Errorf("sink: %v", e) })))
	out, err := r.Run(context.Background(), "task", WithRunID(run))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	ev := w0bToolRunOf(t, out.Transcript.Record)
	if ev.Result != "ab" {
		t.Errorf("Result = %q, want ab", ev.Result)
	}
	if !reflect.DeepEqual(ev.Content, blocks) {
		t.Errorf("Content = %+v, want the placed blocks %+v", ev.Content, blocks)
	}
	hist := toolMessage(t, out.Messages)
	if !reflect.DeepEqual(hist.Content, blocks) {
		t.Errorf("history blocks = %+v, want %+v", hist.Content, blocks)
	}
	lev := w0bToolRunOf(t, w0bLoadRun(t, dir, run).Record)
	if !reflect.DeepEqual(lev.Content, blocks) {
		t.Errorf("record Content = %+v, want %+v", lev.Content, blocks)
	}
	if w0bCanon(t, lev) != w0bCanon(t, ev) {
		t.Errorf("record ToolRun = %s, want canon-equal to %s", w0bCanon(t, lev), w0bCanon(t, ev))
	}
}

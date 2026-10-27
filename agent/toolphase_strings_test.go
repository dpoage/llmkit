package agent

import (
	"context"
	"encoding/json"
	"errors"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

type w0bBlockTool struct{ name string }

func (b w0bBlockTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: b.name, Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (b w0bBlockTool) Run(ctx context.Context, _ json.RawMessage) ([]llmkit.Block, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func w0bInvalidBlockFields(blocks []llmkit.Block) []string {
	var bad []string
	for _, b := range blocks {
		for name, s := range map[string]string{
			"Kind": string(b.Kind), "Text": b.Text, "MediaType": b.MediaType,
			"URL": b.URL, "Title": b.Title, "Provider": b.Provider,
		} {
			if !utf8.ValidString(s) {
				bad = append(bad, name)
			}
		}
	}
	return bad
}

func w0bWithoutCall(ev *llmkit.ToolRunEvent) llmkit.ToolRunEvent {
	cp := *ev
	cp.Call = llmkit.ToolCall{}
	return cp
}

// Call is left out of the comparison: it is the model's request, placed by
// response intake rather than the tool phase.
func TestW0b_ToolPhaseStringsSurviveRecord(t *testing.T) {
	big := strings.Repeat("s", 5000)
	cases := []struct {
		name  string
		tool  Tool
		caps  llmkit.Capabilities
		opts  []Option
		check func(t *testing.T, ev *llmkit.ToolRunEvent)
	}{
		{
			name: "deny reason",
			tool: blocksTool{"blob", textBlocks("ok")},
			opts: []Option{WithToolPolicy(ToolPolicyFunc(func(context.Context, *llmkit.ToolCall) error {
				return errors.New("no \xff way")
			}))},
			check: func(t *testing.T, ev *llmkit.ToolRunEvent) {
				if !ev.Denied || ev.DenyReason != "no \ufffd way" {
					t.Errorf("Denied=%v DenyReason=%q, want the normalized reason", ev.Denied, ev.DenyReason)
				}
			},
		},
		{
			name: "denied tool name",
			tool: blocksTool{"blob\xff", textBlocks("ok")},
			opts: []Option{WithToolPolicy(ToolPolicyFunc(func(context.Context, *llmkit.ToolCall) error {
				return errors.New("no")
			}))},
		},
		{
			name: "timeout tool name",
			tool: w0bBlockTool{"blob\xff"},
			opts: []Option{WithToolTimeout(20 * time.Millisecond)},
			check: func(t *testing.T, ev *llmkit.ToolRunEvent) {
				if want := "ERROR: tool blob\ufffd timed out after 20ms"; ev.Result != want {
					t.Errorf("Result = %q, want %q", ev.Result, want)
				}
			},
		},
		{
			name: "spill error",
			tool: blocksTool{"blob", textBlocks(big)},
			opts: []Option{WithToolOutputLimit(1024, spillSinkFunc(func(context.Context, llmkit.ToolCall, []llmkit.Block) (string, error) {
				return "", errors.New("sink \xff down")
			}))},
			check: func(t *testing.T, ev *llmkit.ToolRunEvent) {
				if ev.SpillError != "sink \ufffd down" {
					t.Errorf("SpillError = %q, want the normalized error", ev.SpillError)
				}
			},
		},
		{
			name: "document title",
			tool: blocksTool{"blob", []llmkit.Block{{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: []byte("%PDF-1.4"), Title: "t\xff"}}},
			caps: llmkit.Capabilities{ToolResultMedia: true},
			check: func(t *testing.T, ev *llmkit.ToolRunEvent) {
				if len(ev.Content) != 1 || ev.Content[0].Title != "t\ufffd" {
					t.Errorf("Content = %+v, want the document with Title %q", ev.Content, "t\ufffd")
				}
			},
		},
		{
			name: "block kind",
			tool: blocksTool{"blob", []llmkit.Block{{Kind: "k\xff", Text: "a"}, llmkit.Text("b")}},
		},
		{
			name: "block media type",
			tool: blocksTool{"blob", []llmkit.Block{{Kind: llmkit.BlockText, Text: "a", MediaType: "m\xff"}, llmkit.Text("b")}},
		},
		{
			name: "block url",
			tool: blocksTool{"blob", []llmkit.Block{{Kind: llmkit.BlockText, Text: "a", URL: "u\xff"}, llmkit.Text("b")}},
		},
		{
			name: "block provider",
			tool: blocksTool{"blob", []llmkit.Block{{Kind: llmkit.BlockText, Text: "a", Provider: "p\xff"}, llmkit.Text("b")}},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			run := llmkit.RunID("w0b-str-" + strings.ReplaceAll(c.name, " ", "-"))
			dir := t.TempDir()
			fc := newFakeClient(toolResp("c1", c.tool.Def().Name, `{}`, 1, 1), textResp("done", 1, 1))
			fc.caps = c.caps
			opts := append(slices.Clone(c.opts), WithObserver(JSONL(dir, func(e error) { t.Errorf("sink: %v", e) })))
			out, err := NewRunner(fc, []Tool{c.tool}, "sys", opts...).Run(context.Background(), "task", WithRunID(run))
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			hist := toolMessage(t, out.Messages)
			ev := w0bToolRunOf(t, out.Transcript.Record)
			lev := w0bToolRunOf(t, w0bLoadRun(t, dir, run).Record)

			if bad := w0bInvalidBlockFields(hist.Content); bad != nil {
				t.Errorf("history block fields not valid UTF-8: %v", bad)
			}
			if got, want := w0bWithoutCall(ev), w0bWithoutCall(lev); !reflect.DeepEqual(got, want) {
				t.Errorf("ToolRun event and record differ:\nevent:  %+v\nrecord: %+v", got, want)
			}
			switch {
			case lev.Denied:
				if want := renderToolDenied(lev.Call.Name, lev.DenyReason); hist.Text() != want {
					t.Errorf("history = %q, record renders %q", hist.Text(), want)
				}
			case lev.Content != nil:
				if !reflect.DeepEqual(hist.Content, lev.Content) {
					t.Errorf("history blocks = %+v, record Content = %+v", hist.Content, lev.Content)
				}
			default:
				if hist.Text() != lev.Result {
					t.Errorf("history = %q, record Result = %q", hist.Text(), lev.Result)
				}
			}
			if c.check != nil {
				c.check(t, ev)
			}
		})
	}
}

// A denial reason of 18000 raw bytes normalizes to 22000 before the record.
func TestW0b_DenialReasonInvalidUTF8CompactionReplays(t *testing.T) {
	run := llmkit.RunID("w0b-deny-probe")
	dir := t.TempDir()
	fc := newFakeClient(
		toolResp("c1", "big", `{"v":1}`, 100, 10),
		toolResp("c2", "big", `{"v":2}`, 100, 10),
		toolResp("c3", "big", `{"v":3}`, 100, 10),
		toolResp("c4", "big", `{"v":4}`, 100, 10),
		toolResp("c5", "big", `{"v":5}`, 100, 10),
		toolResp("c6", "big", `{"v":6}`, 100, 10),
		textResp("done", 100, 10),
	)
	policy := ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if call.ID == "c1" {
			return errors.New(w0bProbeBlob())
		}
		return nil
	})
	limits := Limits{MaxIterations: 30, HistoryTokenBudget: 8000}
	tools := []Tool{bigEchoTool{name: "big", blob: strings.Repeat("v", 18000)}}
	out, err := NewRunner(fc, tools, "sys", WithLimits(limits), WithToolPolicy(policy),
		WithObserver(JSONL(dir, func(e error) { t.Errorf("sink: %v", e) }))).
		Run(context.Background(), "task", WithRunID(run))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	compacted := false
	for _, ev := range out.Transcript.Record {
		compacted = compacted || ev.Kind == llmkit.KindCompaction
	}
	if !compacted {
		t.Fatal("no compaction fired")
	}

	rc, err := NewReplayClient(JSONL(dir, nil), run, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	rout, rerr := NewRunner(rc, rc.Tools(), "sys", WithLimits(limits), WithToolPolicy(rc.ToolPolicy())).
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

// jsonSafeUTF8 must equal encoding/json's own string round trip.
func TestW0b_JSONSafeUTF8MatchesEncodingJSON(t *testing.T) {
	viaJSON := func(s string) string {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("marshal %q: %v", s, err)
		}
		var out string
		if err := json.Unmarshal(b, &out); err != nil {
			t.Fatalf("unmarshal %s: %v", b, err)
		}
		return out
	}
	inputs := []string{
		"", "plain", "\xff", "\xff\xfea", "\x80", "\xe2\x82", "\xed\xa0\x80", "\xc0\x80",
		"\xf4\x90\x80\x80", "\xf0\x9f\x98", "a\xe2\x82\xacb", "\xef\xbf\xbd", "\xf8\x88\x80\x80\x80",
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for range 2000 {
		b := make([]byte, rng.IntN(12))
		for i := range b {
			b[i] = byte(rng.UintN(256))
		}
		inputs = append(inputs, string(b))
	}
	for _, s := range inputs {
		if got, want := jsonSafeUTF8(s), viaJSON(s); got != want {
			t.Errorf("jsonSafeUTF8(%q) = %q, encoding/json gives %q", s, got, want)
		}
	}
}

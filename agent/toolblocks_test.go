package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

// textBlocks wraps s as the single-text-block result a text-only test fake's
// Tool.Run returns.
func textBlocks(s string) []llmkit.Block { return []llmkit.Block{llmkit.Text(s)} }

// goldenTool is a text-only tool whose outcome is chosen by name: the fixture
// behind TestToolResult_TextOnlyRunMatchesBase.
type goldenTool struct{ name string }

func (g goldenTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: g.name, Description: "golden " + g.name, Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (g goldenTool) text(args json.RawMessage) (string, error) {
	switch g.name {
	case "ok":
		return "line one\nline two ✓ " + string(args), nil
	case "empty":
		return "", nil
	case "fail":
		return "", errors.New("disk on fire")
	case "boom":
		panic("tool exploded")
	}
	return "", fmt.Errorf("unexpected golden tool %q", g.name)
}

func (g goldenTool) Run(_ context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	out, err := g.text(args)
	return textBlocks(out), err
}

// goldenDump runs one fixed text-only conversation and renders everything a
// text-only tool result touches into a deterministic text: the history
// (Outcome.Messages), the ToolEnd hook lines, and the JSONL transcript with
// the time-varying fields zeroed. The result is compared byte-for-byte with
// testdata/text_only_run.golden, captured from the c381345 Runner whose
// Tool.Run returned a string.
func goldenDump(t *testing.T) []byte {
	t.Helper()
	calls := []llmkit.ToolCall{
		{ID: "c-ok", Name: "ok", Arguments: json.RawMessage(`{"v":1}`)},
		{ID: "c-empty", Name: "empty", Arguments: json.RawMessage(`{}`)},
		{ID: "c-fail", Name: "fail", Arguments: json.RawMessage(`{}`)},
		{ID: "c-boom", Name: "boom", Arguments: json.RawMessage(`{}`)},
		{ID: "c-gone", Name: "nosuch", Arguments: json.RawMessage(`{}`)},
		{ID: "c-deny", Name: "ok", Arguments: json.RawMessage(`{"deny":true}`)},
	}
	fc := newFakeClient(toolCallsResp(calls...), textResp("done", 4, 2))
	var hooks []string
	var mu sync.Mutex
	policy := ToolPolicyFunc(func(_ context.Context, c *llmkit.ToolCall) error {
		if strings.Contains(string(c.Arguments), "deny") {
			return errors.New("policy says no")
		}
		return nil
	})
	r := NewRunner(fc, []Tool{goldenTool{"ok"}, goldenTool{"empty"}, goldenTool{"fail"}, goldenTool{"boom"}}, "sys",
		WithToolPolicy(policy),
		WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) {
			mu.Lock()
			defer mu.Unlock()
			hooks = append(hooks, fmt.Sprintf("ToolEnd step=%d id=%s name=%s isErr=%v result=%q", ev.Step, ev.Call.ID, ev.Call.Name, ev.IsError, ev.Result))
		}}))
	out, err := r.Run(context.Background(), "task", WithRunID("golden-1"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	var b bytes.Buffer
	b.WriteString("== messages\n")
	for _, m := range out.Messages {
		j, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(j)
		b.WriteByte('\n')
	}
	b.WriteString("== hooks\n")
	for _, h := range hooks {
		b.WriteString(h)
		b.WriteByte('\n')
	}
	b.WriteString("== transcript\n")
	tr := NewTranscript()
	for _, ev := range out.Transcript.Record {
		ev.Time, ev.Duration, ev.SpanID = time.Time{}, 0, ""
		tr.Record = append(tr.Record, ev)
	}
	if err := tr.SaveJSONL(&b); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	return b.Bytes()
}

// TestToolResult_TextOnlyRunMatchesBase pins the preserve contract of the
// Tool.Run signature cutover: history, ToolEnd hook lines, ToolRun events and
// the JSONL transcript of a text-only run (success, empty result, tool
// error, panic, unknown tool, policy denial) are byte-identical to the
// bytes the string-returning Runner at c381345 produced.
func TestToolResult_TextOnlyRunMatchesBase(t *testing.T) {
	got := goldenDump(t)
	want, err := os.ReadFile(filepath.Join("testdata", "text_only_run.golden"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("text-only run drifted from the c381345 golden\n--- got\n%s\n--- want\n%s", got, want)
	}
}

// blocksTool returns a fixed block slice.
type blocksTool struct {
	name string
	out  []llmkit.Block
}

func (b blocksTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: b.name, Description: "blocks", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (b blocksTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) { return b.out, nil }

// toolRunText runs one call of a blocksTool and returns the tool-result
// message fed to the second request, the ToolEnd hook result, and the ToolRun
// event result.
func toolRunText(t *testing.T, out []llmkit.Block) (msg llmkit.Message, hook string, event string, transcript []byte) {
	t.Helper()
	fc := newFakeClient(toolResp("c1", "t", `{}`, 1, 1), textResp("done", 1, 1))
	var hookResult string
	r := NewRunner(fc, []Tool{blocksTool{"t", out}}, "sys", WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) {
		hookResult = ev.Result
	}}))
	o, err := r.Run(context.Background(), "task", WithRunID("p2"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fc.requests))
	}
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult {
			msg = m
		}
	}
	for _, ev := range o.Transcript.Record {
		if ev.Kind == llmkit.KindToolRun {
			event = ev.ToolRun.Result
		}
		ev.Time, ev.Duration, ev.SpanID = time.Time{}, 0, ""
		tr := NewTranscript()
		tr.Record = []llmkit.Event{ev}
		var b bytes.Buffer
		if err := tr.SaveJSONL(&b); err != nil {
			t.Fatal(err)
		}
		transcript = append(transcript, b.Bytes()...)
	}
	return msg, hookResult, event, transcript
}

// TestToolResult_EmptyResultIsTheEmptyTextResult pins that a tool returning
// nil or an empty slice is the same as a tool returning "" on the base:
// one empty text block in history, an empty hook and event Result, and
// identical transcript bytes.
func TestToolResult_EmptyResultIsTheEmptyTextResult(t *testing.T) {
	baseMsg, baseHook, baseEvent, baseTr := toolRunText(t, textBlocks(""))
	if !reflect.DeepEqual(baseMsg.Content, textBlocks("")) {
		t.Fatalf("reference history Content = %+v, want [Text(\"\")]", baseMsg.Content)
	}
	for name, out := range map[string][]llmkit.Block{"nil": nil, "empty slice": {}} {
		t.Run(name, func(t *testing.T) {
			msg, hook, event, tr := toolRunText(t, out)
			if !reflect.DeepEqual(msg.Content, textBlocks("")) {
				t.Errorf("history Content = %+v, want exactly [Text(\"\")]", msg.Content)
			}
			if msg.IsError || msg.ToolCallID != "c1" {
				t.Errorf("message = %+v, want a successful result for c1", msg)
			}
			if hook != baseHook || hook != "" {
				t.Errorf("ToolEnd Result = %q, want %q", hook, baseHook)
			}
			if event != baseEvent || event != "" {
				t.Errorf("ToolRun Result = %q, want %q", event, baseEvent)
			}
			if !bytes.Equal(tr, baseTr) {
				t.Errorf("transcript bytes differ from the \"\" case\n--- got\n%s\n--- want\n%s", tr, baseTr)
			}
		})
	}
}

// TestToolResult_BlocksPassVerbatim pins P3a: the blocks a tool returns are
// the tool-result message's Content exactly — same count, kinds, order —
// as the outgoing request carries them, while hooks and the ToolRun event see
// the concatenated BlockText texts.
func TestToolResult_BlocksPassVerbatim(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\nfake")
	out := []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", png), llmkit.Text("tion")}
	msg, hook, event, _ := toolRunText(t, out)
	if msg.ToolCallID != "c1" || msg.IsError {
		t.Fatalf("message = %+v, want a successful result for c1", msg)
	}
	if !reflect.DeepEqual(msg.Content, out) {
		t.Fatalf("Content = %+v, want the returned blocks verbatim: %+v", msg.Content, out)
	}
	if hook != "caption" || event != "caption" {
		t.Errorf("ToolEnd Result = %q, ToolRun Result = %q, want the concatenated text %q", hook, event, "caption")
	}
}

// TestToolResult_MediaRefusedPreWireOnProviderClient pins P3b: at this tip a
// real provider client's per-role block rule refuses a tool result that
// carries an image BEFORE any wire call, and the run ends with an error
// wrapping llmkit.ErrInvalidRequest — typed, not silent, and no second
// request reaches the server.
func TestToolResult_MediaRefusedPreWireOnProviderClient(t *testing.T) {
	var mu sync.Mutex
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		n := requests
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if n > 1 {
			t.Errorf("request %d reached the wire; the media tool result must be refused pre-wire", n)
		}
		turn := reasoningTurn{reasoning: "r", details: `[]`, toolArgs: `{}`, finish: "tool_calls"}
		_, _ = w.Write([]byte(turn.complete()))
	}))
	t.Cleanup(srv.Close)
	client, err := provider.New(context.Background(), provider.Spec{
		Type: provider.TypeOpenAICompatible, Model: "m", BaseURL: srv.URL, Secret: "k",
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	shot := blocksTool{"echo", []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", []byte("png"))}}
	_, err = NewRunner(client, []Tool{shot}, "sys").Run(context.Background(), "task")
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("Run error = %v, want one wrapping ErrInvalidRequest", err)
	}
	if requests != 1 {
		t.Errorf("wire requests = %d, want 1 (the tool turn only)", requests)
	}
}

// TestReplayTool_ServesRecordedResultAsTextBlock pins P4: the replay tool
// serves a recorded result unchanged, as one text block, and surfaces a
// recorded error as an error.
func TestReplayTool_ServesRecordedResultAsTextBlock(t *testing.T) {
	rec := NewTranscript()
	rec.RunID = "rp-1"
	call := llmkit.ToolCall{ID: "c1", Name: "ok", Arguments: json.RawMessage(`{"v":1}`)}
	rec.Record = append(rec.Record,
		llmkit.Event{Kind: llmkit.KindCompletion, RunID: "rp-1", Step: 1, SchemaVersion: 2,
			Completion: &llmkit.CompletionEvent{Response: llmkit.Response{ToolCalls: []llmkit.ToolCall{call}}}},
		llmkit.Event{Kind: llmkit.KindToolRun, RunID: "rp-1", Step: 1, SchemaVersion: 2,
			ToolRun: &llmkit.ToolRunEvent{Call: call, Result: "recorded ✓\n"}},
		llmkit.Event{Kind: llmkit.KindCompletion, RunID: "rp-1", Step: 2, SchemaVersion: 2,
			Completion: &llmkit.CompletionEvent{Response: llmkit.Response{Text: "done"}}},
	)
	rc, err := NewReplayClient(rec, "rp-1", llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	out, err := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task")
	if err != nil || rc.Err() != nil {
		t.Fatalf("Run = %v, rc.Err = %v", err, rc.Err())
	}
	var got llmkit.Message
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleToolResult {
			got = m
		}
	}
	if !reflect.DeepEqual(got.Content, textBlocks("recorded ✓\n")) {
		t.Errorf("replayed tool-result Content = %+v, want one text block with the recorded result", got.Content)
	}
}

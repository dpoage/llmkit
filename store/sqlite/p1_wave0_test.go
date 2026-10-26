package sqlite_test

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
)

func TestP1_TruncatedArgumentsInAllCarriers(t *testing.T) {
	t.Skip("needs W0a (ToolCall encoder, llmkit-9pk): json.Marshal rejects truncated arguments on the base; remove this skip after the feature-branch merge")
	s, log, path := openStore(t)
	const run = "run-trunc"
	bad := json.RawMessage(`{"p":`)

	start := newEv(run, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "t"}

	comp := newEv(run, llmkit.KindCompletion)
	comp.Completion = &llmkit.CompletionEvent{
		Request: llmkit.Request{Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "go"),
			{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: bad}}},
		}},
		Response: llmkit.Response{ToolCalls: []llmkit.ToolCall{{ID: "c2", Name: "echo", Arguments: bad}}, StopReason: llmkit.StopToolUse},
	}
	tool := newEv(run, llmkit.KindToolRun)
	tool.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c2", Name: "echo", Arguments: bad}, Result: "ok"}

	checkIdentity(t, s, log, run, []llmkit.Event{start, comp, tool})

	// The audit view keeps the bytes the model sent.
	got := query(t, openRaw(t, path), "Q2", sql.Named("run", run))
	requireRows(t, "Q2", got, []string{"0|echo|NULL|" + base64.StdEncoding.EncodeToString(bad) + "|0|0|NULL"}, false)
}

// bigText is at least n bytes of text that needs JSON escaping.
func bigText(n int) string {
	const unit = "line \"q\" <&> \\ \u2028\n"
	return strings.Repeat(unit, (n+len(unit)-1)/len(unit))
}

func TestP1_TwentyMiBToolResultAndRequest(t *testing.T) {
	s, log, _ := openStore(t)
	const run = "run-big"
	big := bigText(20 << 20)

	start := newEv(run, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "t"}
	tool := newEv(run, llmkit.KindToolRun)
	tool.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{}`)}, Result: big}
	comp := newEv(run, llmkit.KindCompletion)
	comp.Completion = &llmkit.CompletionEvent{
		Request: llmkit.Request{Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "go"),
			{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{}`)}}},
			{Role: llmkit.RoleToolResult, ToolCallID: "c1", Content: []llmkit.Block{llmkit.Text(big)}},
		}},
		Response: llmkit.Response{Text: "ok", StopReason: llmkit.StopEndTurn},
	}
	checkIdentity(t, s, log, run, []llmkit.Event{start, tool, comp})
}

// badBytes holds the edge cases of encoding/json's per-byte replacement rule.
const badBytes = "a\xff\xfe b\x80 c\xe2\x82 d\xed\xa0\x80 e\xc0\x80 f"

// poison sets every string field under v to badBytes.
func poison(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer:
		if !v.IsNil() {
			poison(v.Elem())
		}
	case reflect.Struct:
		for i := range v.NumField() {
			poison(v.Field(i))
		}
	case reflect.Slice:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return
		}
		for i := range v.Len() {
			poison(v.Index(i))
		}
	case reflect.String:
		if v.CanSet() {
			v.SetString(badBytes + v.String())
		}
	}
}

func TestP1_InvalidUTF8InEveryStringField(t *testing.T) {
	s, log, _ := openStore(t)
	evs := allKinds("run-utf8")
	for i := range evs {
		e := &evs[i]
		for _, p := range []any{e.Start, e.Completion, e.Attempt, e.ToolRun, e.Compaction, e.Steer, e.Finalize, e.Decision, e.Embed, e.Exec} {
			if p != nil && !reflect.ValueOf(p).IsNil() {
				poison(reflect.ValueOf(p))
			}
		}
	}
	if b, _ := json.Marshal(evs[4]); !strings.Contains(string(b), `\ufffd`) {
		t.Fatal("poisoned tool_run event holds no U+FFFD; the corpus would not test normalization")
	}
	checkIdentity(t, s, log, "run-utf8", evs)
}

// TestP1_InvalidUTF8EdgeBytesFollowEncodingJSON pins the normalization to
// encoding/json's own rule: one U+FFFD per byte that starts no valid sequence.
func TestP1_InvalidUTF8EdgeBytesFollowEncodingJSON(t *testing.T) {
	s, log, _ := openStore(t)
	cases := map[string]string{
		"ff fe a":             "\xff\xfea",
		"lone continuation":   "x\x80y",
		"truncated E2 82":     "x\xe2\x82y",
		"surrogate ED A0 80":  "x\xed\xa0\x80y",
		"overlong C0 80":      "x\xc0\x80y",
		"trailing lead byte":  "x\xe2",
		"valid then invalid":  "é\xffé",
		"collapsed-run guard": "\xff\xff\xff",
	}
	const run = "run-edge"
	evs := []llmkit.Event{}
	start := newEv(run, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "t"}
	evs = append(evs, start)
	var names []string
	for name, in := range cases {
		e := newEv(run, llmkit.KindToolRun)
		e.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c", Name: "echo", Arguments: json.RawMessage(`{}`)}, Result: in}
		evs = append(evs, e)
		names = append(names, name)
	}
	checkIdentity(t, s, log, run, evs)
	got, err := s.Events(context.Background(), run)
	if err != nil {
		t.Fatal(err)
	}
	for i, name := range names {
		var want string
		b, _ := json.Marshal(cases[name])
		if err := json.Unmarshal(b, &want); err != nil {
			t.Fatal(err)
		}
		if got[i+1].ToolRun.Result != want {
			t.Errorf("%s: read %q, want %q", name, got[i+1].ToolRun.Result, want)
		}
	}
	literal := map[string]string{
		"ff fe a":             "\ufffd\ufffda",
		"collapsed-run guard": "\ufffd\ufffd\ufffd",
		"surrogate ED A0 80":  "x\ufffd\ufffd\ufffdy",
		"overlong C0 80":      "x\ufffd\ufffdy",
	}
	for i, name := range names {
		if want, ok := literal[name]; ok && got[i+1].ToolRun.Result != want {
			t.Errorf("%s: read %q, want %q", name, got[i+1].ToolRun.Result, want)
		}
	}
}

func TestP1_EmptyToolCallIDs(t *testing.T) {
	s, log, _ := openStore(t)
	const run = "run-noid"
	start := newEv(run, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "t"}

	calls := []llmkit.ToolCall{
		{Name: "echo", Arguments: json.RawMessage(`{"n":1}`)},
		{Name: "echo", Arguments: json.RawMessage(`{"n":2}`)},
		{Name: "other", Arguments: json.RawMessage(`{"n":3}`)},
	}
	comp := newEv(run, llmkit.KindCompletion)
	comp.Completion = &llmkit.CompletionEvent{
		Request: llmkit.Request{Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "go"),
			{Role: llmkit.RoleAssistant, ToolCalls: calls},
			{Role: llmkit.RoleToolResult, Content: []llmkit.Block{llmkit.Text("first")}},
			{Role: llmkit.RoleToolResult, Content: []llmkit.Block{llmkit.Text("second")}},
			{Role: llmkit.RoleToolResult, Content: []llmkit.Block{llmkit.Text("first")}},
		}},
		Response: llmkit.Response{ToolCalls: calls, StopReason: llmkit.StopToolUse},
	}
	evs := []llmkit.Event{start, comp}
	for _, c := range calls {
		e := newEv(run, llmkit.KindToolRun)
		e.ToolRun = &llmkit.ToolRunEvent{Call: c, Result: "r-" + string(c.Arguments)}
		evs = append(evs, e)
	}
	checkIdentity(t, s, log, run, evs)
}

func TestP1_RunnerEmptyToolCallIDs(t *testing.T) {
	t.Skip("needs W0b (surrogate IDs at response intake, llmkit-9y4): the Runner panics on an empty tool-call ID on the base; remove this skip after the feature-branch merge")
	s, log, _ := openStore(t)
	client := funcClient(func(req llmkit.Request) (llmkit.Response, error) {
		if toolResults(req) >= 3 {
			return llmkit.Response{Text: "done", StopReason: llmkit.StopEndTurn}, nil
		}
		return llmkit.Response{StopReason: llmkit.StopToolUse, ToolCalls: []llmkit.ToolCall{
			{Name: "echo", Arguments: json.RawMessage(`{"n":1}`)},
			{Name: "echo", Arguments: json.RawMessage(`{"n":2}`)},
		}}, nil
	})
	r := agent.NewRunner(client, []agent.Tool{markerTool()}, "sys", agent.WithObserver(s))
	observed := run(t, r, "no ids")
	noErrors(t, log)
	got, err := s.Events(context.Background(), observed[0].RunID)
	if err != nil {
		t.Fatal(err)
	}
	assertIdentical(t, got, observed)
}

func nested(depth int, open, closer string) json.RawMessage {
	return json.RawMessage(strings.Repeat(open, depth) + "0" + strings.Repeat(closer, depth))
}

func TestP1_PayloadDepthUpToTheStoreLimit(t *testing.T) {
	s, log, _ := openStore(t)
	const run = "run-deep"
	start := newEv(run, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "t"}
	// A decision's document is its State depth plus 2.
	dec := newEv(run, llmkit.KindDecision)
	dec.Decision = &llmkit.DecisionEvent{Backend: "b", State: nested(998, "[", "]")}
	deepObj := newEv(run, llmkit.KindDecision)
	deepObj.Decision = &llmkit.DecisionEvent{Backend: "b", State: nested(998, `{"k":`, "}")}
	tool := newEv(run, llmkit.KindToolRun)
	tool.ToolRun = &llmkit.ToolRunEvent{
		Call:                llmkit.ToolCall{ID: "c", Name: "echo", Arguments: json.RawMessage(`{}`)},
		DispatchedArguments: nested(998, "[", "]"),
	}
	checkIdentity(t, s, log, run, []llmkit.Event{start, dec, deepObj, tool})
}

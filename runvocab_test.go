package llmkit_test

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
)

// agent.Transcript already has the method llmkit.Source names; the assertion
// pins that the root interface stays satisfiable by the in-memory sink.
var _ llmkit.Source = (*agent.Transcript)(nil)

// frozenAssistantMessage is a frozen copy of the agent Runner's unexported
// assistantMessage at base 3511f82 (agent/runner.go), the rule
// Response.Message must reproduce exactly. Do not "improve" it.
func frozenAssistantMessage(resp llmkit.Response) llmkit.Message {
	if len(resp.Blocks) > 0 {
		blocks := resp.Blocks
		if resp.Text != "" && !slices.ContainsFunc(blocks, func(b llmkit.Block) bool {
			return b.Kind == llmkit.BlockText
		}) {
			blocks = append(slices.Clone(blocks), llmkit.Block{Kind: llmkit.BlockText, Text: resp.Text})
		}
		return llmkit.Message{Role: llmkit.RoleAssistant, Content: blocks}
	}
	if resp.Text == "" {
		return llmkit.Message{Role: llmkit.RoleAssistant}
	}
	return llmkit.TextMessage(llmkit.RoleAssistant, resp.Text)
}

func TestResponseMessage_MatchesAgentHistoryRule(t *testing.T) {
	calls := []llmkit.ToolCall{{ID: "t1", Name: "search", Arguments: json.RawMessage(`{"q":"go"}`)}}
	think := llmkit.Block{Kind: llmkit.BlockThinking, Text: "hmm", Provider: "anthropic", Raw: json.RawMessage(`{"type":"thinking"}`)}
	cases := []struct {
		name string
		resp llmkit.Response
	}{
		{"text only, no blocks", llmkit.Response{Text: "hello"}},
		{"blocks with text", llmkit.Response{Text: "hello", Blocks: []llmkit.Block{llmkit.Text("hello")}}},
		{"thinking then text", llmkit.Response{
			Text:   "answer",
			Blocks: []llmkit.Block{think, llmkit.Text("answer")},
		}},
		{"thinking only, empty text", llmkit.Response{Blocks: []llmkit.Block{think}}},
		{"blocks lack text while Text set", llmkit.Response{Text: "surfaced", Blocks: []llmkit.Block{think}}},
		{"tool calls, no text", llmkit.Response{ToolCalls: calls}},
		{"tool calls with text", llmkit.Response{Text: "calling", ToolCalls: calls}},
		{"tool calls with thinking and text", llmkit.Response{
			Text:      "calling",
			Blocks:    []llmkit.Block{think, llmkit.Text("calling")},
			ToolCalls: calls,
		}},
		{"empty response", llmkit.Response{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := frozenAssistantMessage(tc.resp)
			want.ToolCalls = tc.resp.ToolCalls
			got := tc.resp.Message()
			if !reflect.DeepEqual(got, want) {
				t.Errorf("Message() = %+v\nwant %+v", got, want)
			}
		})
	}
}

// finalizeObject marshals ev and returns the raw keys of its "finalize" object.
func finalizeObject(t *testing.T, ev llmkit.Event) (line []byte, keys map[string]json.RawMessage) {
	t.Helper()
	line, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(line, &top); err != nil {
		t.Fatalf("Unmarshal top: %v", err)
	}
	if err := json.Unmarshal(top["finalize"], &keys); err != nil {
		t.Fatalf("Unmarshal finalize: %v", err)
	}
	return line, keys
}

func TestFinalizeEvent_StatusErrRoundTrip(t *testing.T) {
	statuses := []llmkit.RunStatus{
		llmkit.RunCompleted, llmkit.RunIncomplete, llmkit.RunRefused,
		llmkit.RunFailed, llmkit.RunCanceled, llmkit.RunPanicked,
	}
	for _, s := range statuses {
		t.Run(string(s), func(t *testing.T) {
			ev := llmkit.NewEvent(context.Background(), llmkit.KindFinalize)
			ev.Finalize = &llmkit.FinalizeEvent{Status: s, Err: "boom: " + string(s)}
			line, keys := finalizeObject(t, ev)
			for _, k := range []string{"status", "err"} {
				if _, ok := keys[k]; !ok {
					t.Errorf("wire key %q missing from %s", k, line)
				}
			}
			var gotStatus, gotErr string
			if err := json.Unmarshal(keys["status"], &gotStatus); err != nil || gotStatus != string(s) {
				t.Errorf("wire status = %q (%v), want %q", gotStatus, err, s)
			}
			if err := json.Unmarshal(keys["err"], &gotErr); err != nil || gotErr != "boom: "+string(s) {
				t.Errorf("wire err = %q (%v)", gotErr, err)
			}

			var back llmkit.Event
			if err := json.Unmarshal(line, &back); err != nil {
				t.Fatalf("Unmarshal: %v", err)
			}
			if err := back.Validate(); err != nil {
				t.Fatalf("Validate after round trip: %v", err)
			}
			if !reflect.DeepEqual(back, ev) {
				t.Errorf("round trip changed the event:\n got %+v\nwant %+v", back.Finalize, ev.Finalize)
			}
		})
	}
}

func TestFinalizeEvent_EmptyStatusErrOmitted(t *testing.T) {
	ev := llmkit.NewEvent(context.Background(), llmkit.KindFinalize)
	ev.Finalize = &llmkit.FinalizeEvent{}
	line, keys := finalizeObject(t, ev)
	for _, k := range []string{"status", "err"} {
		if _, ok := keys[k]; ok {
			t.Errorf("empty %s must be omitted from the wire, got %s", k, line)
		}
	}
}

// TestFinalizeEvent_PreStatusRecordDecodes pins that a line written before
// the status/err fields existed still decodes and validates, with an empty
// Status marking it as pre-round.
func TestFinalizeEvent_PreStatusRecordDecodes(t *testing.T) {
	const old = `{"kind":"finalize","run_id":"r1","step":3,"time":"2026-09-20T12:30:00Z","schema_version":2,` +
		`"finalize":{"truncation_reason":"max_steps","usage":{"input_tokens":5,"output_tokens":2},"final_text":"x"}}`
	var ev llmkit.Event
	if err := json.Unmarshal([]byte(old), &ev); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if err := ev.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if ev.Finalize.Status != "" || ev.Finalize.Err != "" {
		t.Errorf("pre-round record decoded Status=%q Err=%q, want both empty", ev.Finalize.Status, ev.Finalize.Err)
	}
	if ev.Finalize.TruncationReason != "max_steps" || ev.Finalize.FinalText != "x" {
		t.Errorf("existing fields lost: %+v", ev.Finalize)
	}
}

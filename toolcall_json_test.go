package llmkit

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type argsCase struct {
	name  string
	args  string
	valid bool
}

var toolCallArgsCorpus = []argsCase{
	{"truncated object", `{"p":`, false},
	{"truncated array", `{"a":[1,2`, false},
	{"bare word", `nul`, false},
	{"unclosed string", `"unclosed`, false},
	{"invalid utf8 bytes", string([]byte{0xff, 0xfe, 'a'}), false},
	{"control bytes", "\x00\x01", false},
	{"string literal", `""`, true},
	{"string", `"text"`, true},
	{"object", `{"p":1}`, true},
	{"object with whitespace", "{ \"p\" : 1 }", true},
	{"object with unicode escape", `{"e":"\u00e9"}`, true},
	{"empty object", `{}`, true},
	{"array", `[1,2,3]`, true},
	{"number", `42`, true},
	{"true", `true`, true},
	{"null literal", `null`, true},
	{"html chars", `{"q":"a<b>&c"}`, true},
	{"unicode", "{\"é\":\"你好\U0001F600\"}", true},
	{"u2028 u2029", "{\"s\":\"\u2028\u2029\"}", true},
	{"nested object and array", `{"a":{"b":[1,{"c":null}]}}`, true},
}

// callsInEvent returns every ToolCall an Event carries, one per carrier:
// Completion.Response.ToolCalls, Completion.Request.Messages[].ToolCalls and
// ToolRun.Call.
func callsInEvent(t *testing.T, ev *Event, which string) []ToolCall {
	t.Helper()
	var out []ToolCall
	if ev.Completion != nil {
		out = append(out, ev.Completion.Response.ToolCalls...)
		for _, m := range ev.Completion.Request.Messages {
			out = append(out, m.ToolCalls...)
		}
	}
	if ev.ToolRun != nil {
		out = append(out, ev.ToolRun.Call)
	}
	if len(out) != 1 {
		t.Fatalf("%s: event carries %d tool calls, want 1", which, len(out))
	}
	return out
}

// eventCarriers wraps tc in each of the three carriers an Event can hold it in.
func eventCarriers(tc ToolCall) []Event {
	return []Event{
		{Kind: KindCompletion, Completion: &CompletionEvent{Response: Response{ToolCalls: []ToolCall{tc}}}},
		{Kind: KindCompletion, Completion: &CompletionEvent{Request: Request{Messages: []Message{{Role: RoleAssistant, ToolCalls: []ToolCall{tc}}}}}},
		{Kind: KindToolRun, ToolRun: &ToolRunEvent{Call: tc}},
	}
}

// canon returns json.Marshal of a fresh value decoded from json.Marshal(ev).
func canon(t *testing.T, ev Event) []byte {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("canon marshal: %v", err)
	}
	var fresh Event
	if err := json.Unmarshal(b, &fresh); err != nil {
		t.Fatalf("canon unmarshal: %v", err)
	}
	out, err := json.Marshal(&fresh)
	if err != nil {
		t.Fatalf("canon remarshal: %v", err)
	}
	return out
}

func TestToolCallJSONCarriersRoundTrip(t *testing.T) {
	for _, tc := range toolCallArgsCorpus {
		for ci, ev := range eventCarriers(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(tc.args)}) {
			wire, err := json.Marshal(ev)
			if err != nil {
				t.Errorf("%s carrier %d: Marshal: %v", tc.name, ci, err)
				continue
			}
			var read Event
			if err := json.Unmarshal(wire, &read); err != nil {
				t.Errorf("%s carrier %d: Unmarshal: %v", tc.name, ci, err)
				continue
			}
			got := callsInEvent(t, &read, tc.name)[0].Arguments
			if !tc.valid {
				if !bytes.Equal(got, json.RawMessage(tc.args)) {
					t.Errorf("%s carrier %d: Arguments = %q, want byte-equal %q", tc.name, ci, got, tc.args)
				}
				continue
			}
			remarshaled, err := json.Marshal(&read)
			if err != nil {
				t.Errorf("%s carrier %d: remarshal: %v", tc.name, ci, err)
				continue
			}
			if want := canon(t, ev); !bytes.Equal(remarshaled, want) {
				t.Errorf("%s carrier %d: round trip not json-identical:\n got %s\nwant %s", tc.name, ci, remarshaled, want)
			}
		}
	}
}

func TestToolCallJSONWireShapes(t *testing.T) {
	// Invalid JSON arguments ride base64 in arguments_b64; arguments stays
	// absent.
	b, err := json.Marshal(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"p":`)})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var shape map[string]json.RawMessage
	if err := json.Unmarshal(b, &shape); err != nil {
		t.Fatalf("shape decode: %v", err)
	}
	if _, ok := shape["arguments"]; ok {
		t.Errorf("wire carries arguments for non-JSON args: %s", b)
	}
	want := base64.StdEncoding.EncodeToString([]byte(`{"p":`))
	var got string
	if err := json.Unmarshal(shape["arguments_b64"], &got); err != nil {
		t.Fatalf("arguments_b64 decode: %v", err)
	}
	if got != want {
		t.Errorf("arguments_b64 = %q, want %q", got, want)
	}

	// nil arguments are omitted, exactly as before.
	b, err = json.Marshal(ToolCall{ID: "c1", Name: "echo"})
	if err != nil {
		t.Fatalf("Marshal nil args: %v", err)
	}
	if string(b) != `{"id":"c1","name":"echo"}` {
		t.Errorf("nil args wire = %s", b)
	}

	// The empty-but-non-nil payload is non-JSON (json.Valid("") is false);
	// it round-trips through arguments_b64:"" instead of collapsing to nil.
	b, err = json.Marshal(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage{}})
	if err != nil {
		t.Fatalf("Marshal empty args: %v", err)
	}
	if string(b) != `{"id":"c1","name":"echo","arguments_b64":""}` {
		t.Errorf("empty args wire = %s", b)
	}
	var back ToolCall
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("Unmarshal empty args: %v", err)
	}
	if back.Arguments == nil || len(back.Arguments) != 0 {
		t.Errorf("empty args round trip = %v (nil=%t), want non-nil empty", back.Arguments, back.Arguments == nil)
	}

	// A JSON string literal stays in arguments, as at base.
	b, err = json.Marshal(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`""`)})
	if err != nil {
		t.Fatalf("Marshal string literal: %v", err)
	}
	if string(b) != `{"id":"c1","name":"echo","arguments":""}` {
		t.Errorf("string literal wire = %s", b)
	}
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("Unmarshal string literal: %v", err)
	}
	if string(back.Arguments) != `""` {
		t.Errorf("string literal round trip = %q, want %q", back.Arguments, `""`)
	}
}

// legacyToolCall is ToolCall as a pre-arguments_b64 reader sees it.
type legacyToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

func TestToolCallJSONLegacyReaderSeesNilArguments(t *testing.T) {
	b, err := json.Marshal(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"p":`)})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var old legacyToolCall
	if err := json.Unmarshal(b, &old); err != nil {
		t.Fatalf("legacy Unmarshal: %v", err)
	}
	if old.ID != "c1" || old.Name != "echo" {
		t.Errorf("legacy fields = %q %q", old.ID, old.Name)
	}
	if old.Arguments != nil {
		t.Errorf("legacy reader sees Arguments = %q, want nil", old.Arguments)
	}
}

func TestToolCallJSONRefusesCorruptWire(t *testing.T) {
	var tc ToolCall
	if err := json.Unmarshal([]byte(`{"id":"c","name":"n","arguments":{"a":1},"arguments_b64":"eA=="}`), &tc); err == nil {
		t.Error("both fields set: want error, got nil")
	}
	if err := json.Unmarshal([]byte(`{"id":"c","name":"n","arguments_b64":"!!!"}`), &tc); err == nil {
		t.Error("undecodable base64: want error, got nil")
	}
}

// goldenLines is the base (2d3c441) encoding of every valid-JSON corpus
// entry in each carrier, one line per (entry, carrier).
func goldenLines(t *testing.T) []byte {
	t.Helper()
	var out bytes.Buffer
	for _, tc := range toolCallArgsCorpus {
		if !tc.valid {
			continue
		}
		for _, ev := range eventCarriers(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(tc.args)}) {
			b, err := json.Marshal(ev)
			if err != nil {
				t.Fatalf("%s: Marshal: %v", tc.name, err)
			}
			out.Write(b)
			out.WriteByte('\n')
		}
	}
	return out.Bytes()
}

func TestToolCallJSONValidArgumentsMatchBaseEncoding(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "toolcall_valid_args_base.golden"))
	if err != nil {
		t.Fatal(err)
	}
	if got := goldenLines(t); !bytes.Equal(got, want) {
		t.Errorf("valid-argument encoding drifted from base 2d3c441\n--- got\n%s\n--- want\n%s", got, want)
	}
}

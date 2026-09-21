package llmkit

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// Compile-time: both forms satisfy Observer.
var (
	_ Observer = (Observer)(nil)
	_ Observer = ObserverFunc(nil)
)

var (
	goldRun    = RunID("1758366600000-deadbeef00112233")
	goldParent = RunID("1758366500000-cafebabefeedface")
	goldSpan   = SpanID("1758366600001-0123456789abcdef")
	goldTime   = time.Date(2026, 9, 20, 12, 30, 0, 0, time.UTC)
)

func f64(v float64) *float64 { return &v }

func ptrInt(v int) *int       { return &v }
func ptrInt64(v int64) *int64 { return &v }

// compactJSON compacts a JSON literal, failing the test on malformed input.
func compactJSON(t *testing.T, s string) string {
	t.Helper()
	var buf bytes.Buffer
	if err := json.Compact(&buf, []byte(s)); err != nil {
		t.Fatalf("golden literal is not valid JSON: %v", err)
	}
	return buf.String()
}

// goldRequest is a fully-populated Request, so the Completion golden pins the
// wire shape of every nested root type (Message, ToolCall, ToolDef, ...).
func goldRequest() Request {
	return Request{
		System: "sys",
		Messages: []Message{
			TextMessage(RoleUser, "find it"),
			{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{ID: "t1", Name: "search", Arguments: json.RawMessage(`{"q":"golang"}`)},
				},
			},
		},
		Tools: []ToolDef{
			{Name: "search", Description: "web search", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
		MaxTokens:          512,
		Temperature:        f64(0.5),
		Thinking:           &ThinkingConfig{BudgetTokens: 1024},
		ToolChoice:         ToolChoice{Mode: ToolChoiceTool, Name: "search"},
		StopSequences:      []string{"STOP"},
		TopP:               f64(0.9),
		TopK:               ptrInt(40),
		Seed:               ptrInt64(7),
		ResponseSchema:     json.RawMessage(`{"type":"object"}`),
		ResponseSchemaName: "answer",
	}
}

func goldResponse() Response {
	return Response{
		Text:   "looking",
		Blocks: []Block{Text("looking")},
		ToolCalls: []ToolCall{
			{ID: "t1", Name: "search", Arguments: json.RawMessage(`{"q":"golang"}`)},
		},
		Usage:      Usage{InputTokens: 100, OutputTokens: 20, CacheReadInputTokens: 40, CacheCreationInputTokens: 10},
		StopReason: StopEndTurn,
	}
}

// goldEvents returns one fully-populated event per kind, in Kind order.
func goldEvents() []struct {
	name string
	ev   Event
} {
	belief := 0.25
	score := 2.09
	return []struct {
		name string
		ev   Event
	}{
		{
			name: "start",
			ev: Event{
				Kind: KindStart, RunID: goldRun, ParentRunID: goldParent,
				Time: goldTime, SchemaVersion: EventSchemaVersion,
				Start: &StartEvent{Task: "summarize the ledger", Tools: []string{"search", "calc"}},
			},
		},
		{
			name: "completion",
			ev: Event{
				Kind: KindCompletion, RunID: goldRun, SpanID: goldSpan, Step: 2,
				Time: goldTime, Duration: 1500 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Completion: &CompletionEvent{
					Request:  goldRequest(),
					Response: goldResponse(),
					Provider: "anthropic",
					Model:    "claude-sonnet",
				},
			},
		},
		{
			name: "attempt",
			ev: Event{
				Kind: KindAttempt, RunID: goldRun, SpanID: goldSpan,
				Time: goldTime, Duration: 250 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Attempt: &AttemptEvent{
					Attempt: 1,
					Request: Request{Messages: []Message{TextMessage(RoleUser, "hi")}},
					// A failed attempt carries the zero Response; its Usage
					// field has no omitempty (a struct tag cannot omit a
					// struct), so the wire shows it as {"usage":{}}.
					Response:   Response{},
					Err:        "429 too many requests",
					StatusCode: 429,
					RetryAfter: 30 * time.Second,
					Provider:   "openai",
					Model:      "gpt",
				},
			},
		},
		{
			name: "tool_run",
			ev: Event{
				Kind: KindToolRun, RunID: goldRun, Step: 3,
				Time: goldTime, Duration: 200 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				ToolRun: &ToolRunEvent{
					Call:   ToolCall{ID: "t1", Name: "calc", Arguments: json.RawMessage(`{"x":2}`)},
					Result: `{"y":4}`,
				},
			},
		},
		{
			name: "tool_run_denied",
			ev: Event{
				Kind: KindToolRun, RunID: goldRun, Step: 3,
				Time: goldTime, SchemaVersion: EventSchemaVersion,
				ToolRun: &ToolRunEvent{
					Call:       ToolCall{ID: "t2", Name: "rm", Arguments: json.RawMessage(`{"path":"/"}`)},
					Denied:     true,
					DenyReason: "blocked by policy",
				},
			},
		},
		{
			name: "compaction",
			ev: Event{
				Kind: KindCompaction, RunID: goldRun, Step: 4,
				Time: goldTime, Duration: 5 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Compaction: &CompactionEvent{BeforeTokens: 9000, AfterTokens: 4000, Pruned: 6},
			},
		},
		{
			name: "steer",
			ev: Event{
				Kind: KindSteer, RunID: goldRun, Step: 5,
				Time: goldTime, SchemaVersion: EventSchemaVersion,
				Steer: &SteerEvent{Message: TextMessage(RoleUser, "focus on taxes"), FollowUp: true},
			},
		},
		{
			name: "finalize",
			ev: Event{
				Kind: KindFinalize, RunID: goldRun, ParentRunID: goldParent, Step: 8,
				Time: goldTime, Duration: time.Minute, SchemaVersion: EventSchemaVersion,
				Finalize: &FinalizeEvent{
					TruncationReason: "max_steps",
					Finalized:        true,
					Usage:            Usage{InputTokens: 1000, OutputTokens: 200},
					FinalText:        "Answer: 42.",
				},
			},
		},
		{
			name: "decision",
			ev: Event{
				Kind: KindDecision, RunID: goldRun,
				Time: goldTime, Duration: 800 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Decision: &DecisionEvent{
					Backend: "typesafe",
					Model:   "jev-1",
					State:   json.RawMessage(`{"host":"db-1"}`),
					Questions: []DecisionQuestion{
						{
							ID: "q1", Kind: "noul", Instructions: json.RawMessage(`{"prompt":"confident?"}`),
							True:  json.RawMessage(`"clearly risky"`),
							False: json.RawMessage(`"clearly safe"`),
						},
						{
							ID: "q2", Kind: "choice", Instructions: json.RawMessage(`{"prompt":"stay or go?"}`),
							Options: map[string]json.RawMessage{
								"go":   json.RawMessage(`"leave"`),
								"stay": json.RawMessage(`"remain"`),
							},
						},
						{
							ID: "q3", Kind: "score", Instructions: json.RawMessage(`{"prompt":"rate it"}`),
							Levels: []json.RawMessage{json.RawMessage(`"low"`), json.RawMessage(`"high"`)},
						},
						{ID: "q4", Kind: "noul", Instructions: json.RawMessage(`{"prompt":"safe?"}`)},
						{ID: "q5", Kind: "score", Instructions: json.RawMessage(`{"prompt":"how bad?"}`)},
					},
					Answers: []DecisionAnswer{
						{ID: "q1", Belief: &belief},
						{ID: "q2", Choice: "go", Probabilities: map[string]float64{"go": 0.7, "stay": 0.3}, Confidence: 0.8},
						{ID: "q3", Score: &score, Levels: []string{"low", "high"}, LevelProbabilities: []float64{0.2, 0.8}},
						{ID: "q4", Belief: f64(0)},
						{ID: "q5", Score: f64(0)},
					},
					Usage: Usage{InputTokens: 50, OutputTokens: 10},
				},
			},
		},
		{
			name: "embed",
			ev: Event{
				Kind: KindEmbed, RunID: goldRun,
				Time: goldTime, Duration: 120 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Embed: &EmbedEvent{Model: "text-embed-3", Inputs: 4, Dimensions: 1536, CacheHits: 1, Usage: Usage{InputTokens: 900}},
			},
		},
		{
			name: "exec",
			ev: Event{
				Kind: KindExec, RunID: goldRun,
				Time: goldTime, Duration: 2 * time.Second, SchemaVersion: EventSchemaVersion,
				Exec: &ExecEvent{
					Backend:     "docker",
					Command:     []string{"sh", "-c", "ls"},
					ExitCode:    2,
					StdoutBytes: 120,
					StderrBytes: 30,
					Truncated:   true,
				},
			},
		},
	}
}

// TestEventGoldenJSON pins each kind's wire shape against a golden literal,
// so a tag rename or field reorder fails loudly instead of silently changing
// every durable sink's format.
//
// Sinks write one COMPACT line per event (json.Encoder), so the pinned
// artifact is the compact wire line. The literals stay indented for
// readability; both sides are compacted before the byte comparison, which
// still pins key names, key order, and values — including json.RawMessage
// content, which encoding/json carries verbatim (compacted) on the wire.
func TestEventGoldenJSON(t *testing.T) {
	goldens := map[string]string{
		"start": `{
  "kind": "start",
  "run_id": "1758366600000-deadbeef00112233",
  "parent_run_id": "1758366500000-cafebabefeedface",
  "time": "2026-09-20T12:30:00Z",
  "schema_version": 1,
  "start": {
    "task": "summarize the ledger",
    "tools": [
      "search",
      "calc"
    ]
  }
}`,
		"completion": `{
  "kind": "completion",
  "run_id": "1758366600000-deadbeef00112233",
  "span_id": "1758366600001-0123456789abcdef",
  "step": 2,
  "time": "2026-09-20T12:30:00Z",
  "duration": 1500000000,
  "schema_version": 1,
  "completion": {
    "request": {
      "system": "sys",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "kind": "text",
              "text": "find it"
            }
          ]
        },
        {
          "role": "assistant",
          "tool_calls": [
            {
              "id": "t1",
              "name": "search",
              "arguments": {"q":"golang"}
            }
          ]
        }
      ],
      "tools": [
        {
          "name": "search",
          "description": "web search",
          "parameters": {"type":"object"}
        }
      ],
      "max_tokens": 512,
      "temperature": 0.5,
      "thinking": {
        "budget_tokens": 1024
      },
      "tool_choice": {
        "mode": "tool",
        "name": "search"
      },
      "stop_sequences": [
        "STOP"
      ],
      "top_p": 0.9,
      "top_k": 40,
      "seed": 7,
      "response_schema": {"type":"object"},
      "response_schema_name": "answer"
    },
    "response": {
      "text": "looking",
      "blocks": [
        {
          "kind": "text",
          "text": "looking"
        }
      ],
      "tool_calls": [
        {
          "id": "t1",
          "name": "search",
          "arguments": {"q":"golang"}
        }
      ],
      "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 10
      },
      "stop_reason": "end_turn"
    },
    "provider": "anthropic",
    "model": "claude-sonnet"
  }
}`,
		"attempt": `{
  "kind": "attempt",
  "run_id": "1758366600000-deadbeef00112233",
  "span_id": "1758366600001-0123456789abcdef",
  "time": "2026-09-20T12:30:00Z",
  "duration": 250000000,
  "schema_version": 1,
  "attempt": {
    "attempt": 1,
    "request": {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "kind": "text",
              "text": "hi"
            }
          ]
        }
      ],
      "tool_choice": {}
    },
    "response": {"usage":{}},
    "err": "429 too many requests",
    "status_code": 429,
    "retry_after": 30000000000,
    "provider": "openai",
    "model": "gpt"
  }
}`,
		"tool_run": `{
  "kind": "tool_run",
  "run_id": "1758366600000-deadbeef00112233",
  "step": 3,
  "time": "2026-09-20T12:30:00Z",
  "duration": 200000000,
  "schema_version": 1,
  "tool_run": {
    "call": {
      "id": "t1",
      "name": "calc",
      "arguments": {"x":2}
    },
    "result": "{\"y\":4}"
  }
}`,
		"tool_run_denied": `{
  "kind": "tool_run",
  "run_id": "1758366600000-deadbeef00112233",
  "step": 3,
  "time": "2026-09-20T12:30:00Z",
  "schema_version": 1,
  "tool_run": {
    "call": {
      "id": "t2",
      "name": "rm",
      "arguments": {"path":"/"}
    },
    "denied": true,
    "deny_reason": "blocked by policy"
  }
}`,
		"compaction": `{
  "kind": "compaction",
  "run_id": "1758366600000-deadbeef00112233",
  "step": 4,
  "time": "2026-09-20T12:30:00Z",
  "duration": 5000000,
  "schema_version": 1,
  "compaction": {
    "before_tokens": 9000,
    "after_tokens": 4000,
    "pruned": 6
  }
}`,
		"steer": `{
  "kind": "steer",
  "run_id": "1758366600000-deadbeef00112233",
  "step": 5,
  "time": "2026-09-20T12:30:00Z",
  "schema_version": 1,
  "steer": {
    "message": {
      "role": "user",
      "content": [
        {
          "kind": "text",
          "text": "focus on taxes"
        }
      ]
    },
    "follow_up": true
  }
}`,
		"finalize": `{
  "kind": "finalize",
  "run_id": "1758366600000-deadbeef00112233",
  "parent_run_id": "1758366500000-cafebabefeedface",
  "step": 8,
  "time": "2026-09-20T12:30:00Z",
  "duration": 60000000000,
  "schema_version": 1,
  "finalize": {
    "truncation_reason": "max_steps",
    "finalized": true,
    "usage": {
      "input_tokens": 1000,
      "output_tokens": 200
    },
    "final_text": "Answer: 42."
  }
}`,
		"decision": `{
  "kind": "decision",
  "run_id": "1758366600000-deadbeef00112233",
  "time": "2026-09-20T12:30:00Z",
  "duration": 800000000,
  "schema_version": 1,
  "decision": {
    "backend": "typesafe",
    "model": "jev-1",
    "state": {"host":"db-1"},
    "questions": [
      {
        "id": "q1",
        "kind": "noul",
        "instructions": {"prompt":"confident?"},
        "true": "clearly risky",
        "false": "clearly safe"
      },
      {
        "id": "q2",
        "kind": "choice",
        "instructions": {"prompt":"stay or go?"},
        "options": {
          "go": "leave",
          "stay": "remain"
        }
      },
      {
        "id": "q3",
        "kind": "score",
        "instructions": {"prompt":"rate it"},
        "levels": [
          "low",
          "high"
        ]
      },
      {
        "id": "q4",
        "kind": "noul",
        "instructions": {"prompt":"safe?"}
      },
      {
        "id": "q5",
        "kind": "score",
        "instructions": {"prompt":"how bad?"}
      }
    ],
    "answers": [
      {
        "id": "q1",
        "belief": 0.25
      },
      {
        "id": "q2",
        "choice": "go",
        "probabilities": {
          "go": 0.7,
          "stay": 0.3
        },
        "confidence": 0.8
      },
      {
        "id": "q3",
        "score": 2.09,
        "levels": [
          "low",
          "high"
        ],
        "level_probabilities": [
          0.2,
          0.8
        ]
      },
      {
        "id": "q4",
        "belief": 0
      },
      {
        "id": "q5",
        "score": 0
      }
    ],
    "usage": {
      "input_tokens": 50,
      "output_tokens": 10
    }
  }
}`,
		"embed": `{
  "kind": "embed",
  "run_id": "1758366600000-deadbeef00112233",
  "time": "2026-09-20T12:30:00Z",
  "duration": 120000000,
  "schema_version": 1,
  "embed": {
    "model": "text-embed-3",
    "inputs": 4,
    "dimensions": 1536,
    "cache_hits": 1,
    "usage": {
      "input_tokens": 900
    }
  }
}`,
		"exec": `{
  "kind": "exec",
  "run_id": "1758366600000-deadbeef00112233",
  "time": "2026-09-20T12:30:00Z",
  "duration": 2000000000,
  "schema_version": 1,
  "exec": {
    "backend": "docker",
    "command": [
      "sh",
      "-c",
      "ls"
    ],
    "exit_code": 2,
    "stdout_bytes": 120,
    "stderr_bytes": 30,
    "truncated": true
  }
}`,
	}

	seen := map[string]bool{}
	for _, tc := range goldEvents() {
		t.Run(tc.name, func(t *testing.T) {
			seen[tc.name] = true
			got, err := json.Marshal(tc.ev)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			want, ok := goldens[tc.name]
			if !ok {
				t.Fatalf("no golden literal for kind event %q", tc.name)
			}
			if string(got) != compactJSON(t, want) {
				t.Fatalf("wire shape changed for %s\n--- got (compact wire) ---\n%s\n--- want (golden literal) ---\n%s", tc.name, got, want)
			}
		})
	}
	for name := range goldens {
		if !seen[name] {
			t.Errorf("golden literal %q has no matching event in goldEvents()", name)
		}
	}
}

// TestEventJSONRoundTrip encodes then decodes an event of every kind and
// requires DeepEqual: the wire carries enough to reconstruct the event with
// no registry and no loss.
func TestEventJSONRoundTrip(t *testing.T) {
	for _, tc := range goldEvents() {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.ev)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var back Event
			if err := json.Unmarshal(data, &back); err != nil {
				t.Fatalf("unmarshal: %v\njson: %s", err, data)
			}
			if !reflect.DeepEqual(tc.ev, back) {
				orig, _ := json.Marshal(tc.ev)
				rt, _ := json.Marshal(back)
				t.Fatalf("round-trip changed the event\nwant re-encoded: %s\ngot re-encoded:  %s\nwant: %#v\ngot:  %#v", orig, rt, tc.ev, back)
			}
		})
	}
}

// TestSchemaVersionOnEveryEncodedEvent pins that the schema_version key is
// present on every kind, even when an emitter hand-builds a bare event.
func TestSchemaVersionOnEveryEncodedEvent(t *testing.T) {
	for _, tc := range goldEvents() {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.ev)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if !strings.Contains(string(data), `"schema_version":1`) {
				t.Fatalf("schema_version missing from %s event: %s", tc.name, data)
			}
			var back Event
			if err := json.Unmarshal(data, &back); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if back.SchemaVersion != EventSchemaVersion {
				t.Fatalf("SchemaVersion = %d, want %d", back.SchemaVersion, EventSchemaVersion)
			}
		})
	}
}

func TestObserversFanOutOrderSkipsNils(t *testing.T) {
	var order []string
	mk := func(name string) Observer {
		return ObserverFunc(func(context.Context, Event) { order = append(order, name) })
	}
	chain := Observers(mk("first"), nil, mk("second"))
	chain.Observe(context.Background(), Event{Kind: KindCompletion})
	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Fatalf("fan-out order = %v, want [first second] with the nil skipped", order)
	}
}

func TestObserversEmptyChainIsNoOp(t *testing.T) {
	for name, chain := range map[string]Observer{
		"no arguments": Observers(),
		"all nil":      Observers(nil, nil),
	} {
		chain.Observe(context.Background(), Event{Kind: KindExec}) // must not panic
		if chain == nil {
			t.Fatalf("%s: empty fan-out must be a usable observer, got nil", name)
		}
	}
}

func TestObserverPanicPropagates(t *testing.T) {
	afterCalled := false
	chain := Observers(
		ObserverFunc(func(context.Context, Event) { panic("sink bug") }),
		ObserverFunc(func(context.Context, Event) { afterCalled = true }),
	)
	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("observer panic was swallowed, want it to propagate")
			}
			if r != "sink bug" {
				t.Fatalf("recover() = %v, want the observer's own panic value", r)
			}
		}()
		chain.Observe(context.Background(), Event{Kind: KindExec})
	}()
	if afterCalled {
		t.Fatal("observers after the panicking one must not run")
	}
}

func TestObserverMustNotSeeMutatedEvent(t *testing.T) {
	// Documented contract: an Observer is a data sink and must not mutate the
	// event. The fan-out passes the event by value, so reassigning a payload
	// pointer inside one observer stays invisible to the emitter and to
	// later observers. (Mutating THROUGH the shared payload pointer would be
	// visible; that is prohibited by the Observer doc, not enforced here.)
	var seenBySecond Event
	ev := Event{Kind: KindExec, Exec: &ExecEvent{Backend: "docker"}}
	chain := Observers(
		ObserverFunc(func(_ context.Context, e Event) { e.Exec = &ExecEvent{Backend: "hijacked"} }),
		ObserverFunc(func(_ context.Context, e Event) { seenBySecond = e }),
	)
	chain.Observe(context.Background(), ev)
	if seenBySecond.Exec == nil || seenBySecond.Exec.Backend != "docker" {
		t.Fatalf("second observer saw %+v, want the emitter's original payload", seenBySecond.Exec)
	}
	if ev.Exec.Backend != "docker" {
		t.Fatalf("emitter's event was mutated: %+v", ev.Exec)
	}
}

// The decide package imports root, so this test file cannot import decide.
// The mirrors below restate exactly the decide vocabulary the conversion
// layer (llmkit-1sx.3) will translate: if DecisionEvent cannot carry a
// decide fact, TestDecisionMirrorRoundTrip fails instead of the conversion
// silently losing it.

type mirrorQuestion interface {
	mirrorID() string
	instructions() any
}

type mirrorNoul struct {
	ID           string
	Instructions any
	True, False  any
}

func (q mirrorNoul) mirrorID() string { return q.ID }

type mirrorChoice struct {
	ID           string
	Instructions any
	Options      map[string]any
}

func (q mirrorChoice) mirrorID() string { return q.ID }

type mirrorScore struct {
	ID           string
	Instructions any
	Levels       []any
}

func (q mirrorScore) mirrorID() string { return q.ID }

type mirrorChoiceAnswer struct {
	Choice        string
	Probabilities map[string]float64
	Confidence    float64
}

type mirrorScoreAnswer struct {
	Score         float64
	Levels        []string
	Probabilities []float64
	Confidence    float64
}

type mirrorResponse struct {
	Model   string
	Nouls   map[string]float64
	Choices map[string]mirrorChoiceAnswer
	Scores  map[string]mirrorScoreAnswer
	Usage   Usage
}

// rawToAny decodes raw with json.Number preserved so integers survive the
// round-trip exactly (the real conversion layer must do the same).
func rawToAny(t *testing.T, raw json.RawMessage) any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatalf("decode raw %s: %v", raw, err)
	}
	return v
}

func mirrorToDecisionEvent(t *testing.T, state any, qs []mirrorQuestion, resp mirrorResponse) Event {
	t.Helper()
	raw := func(v any) json.RawMessage {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal mirror value: %v", err)
		}
		return b
	}
	questions := make([]DecisionQuestion, 0, len(qs))
	for _, q := range qs {
		dq := DecisionQuestion{ID: q.mirrorID(), Kind: kindOf(q), Instructions: raw(q.instructions())}
		switch q := q.(type) {
		case mirrorNoul:
			dq.True, dq.False = raw(q.True), raw(q.False)
		case mirrorChoice:
			dq.Options = map[string]json.RawMessage{}
			for k, v := range q.Options {
				dq.Options[k] = raw(v)
			}
		case mirrorScore:
			for _, lv := range q.Levels {
				dq.Levels = append(dq.Levels, raw(lv))
			}
		}
		questions = append(questions, dq)
	}
	sort.Slice(questions, func(i, j int) bool { return questions[i].ID < questions[j].ID })

	answers := make([]DecisionAnswer, 0, len(qs))
	for _, q := range qs {
		id := q.mirrorID()
		switch q.(type) {
		case mirrorNoul:
			belief := resp.Nouls[id]
			answers = append(answers, DecisionAnswer{ID: id, Belief: &belief})
		case mirrorChoice:
			a := resp.Choices[id]
			answers = append(answers, DecisionAnswer{ID: id, Choice: a.Choice, Probabilities: a.Probabilities, Confidence: a.Confidence})
		case mirrorScore:
			a := resp.Scores[id]
			answers = append(answers, DecisionAnswer{ID: id, Score: &a.Score, Levels: a.Levels, LevelProbabilities: a.Probabilities, Confidence: a.Confidence})
		}
	}
	sort.Slice(answers, func(i, j int) bool { return answers[i].ID < answers[j].ID })

	return Event{
		Kind: KindDecision, Time: goldTime, SchemaVersion: EventSchemaVersion,
		Decision: &DecisionEvent{
			Backend: "typesafe", Model: resp.Model, State: raw(state),
			Questions: questions, Answers: answers, Usage: resp.Usage,
		},
	}
}

func kindOf(q mirrorQuestion) string {
	switch q.(type) {
	case mirrorNoul:
		return "noul"
	case mirrorChoice:
		return "choice"
	case mirrorScore:
		return "score"
	}
	panic("unknown mirror question")
}

func (q mirrorNoul) instructions() any   { return q.Instructions }
func (q mirrorChoice) instructions() any { return q.Instructions }
func (q mirrorScore) instructions() any  { return q.Instructions }

func decisionToMirror(t *testing.T, ev Event) (any, []mirrorQuestion, mirrorResponse) {
	t.Helper()
	d := ev.Decision
	state := rawToAny(t, d.State)
	var qs []mirrorQuestion
	for _, dq := range d.Questions {
		ins := rawToAny(t, dq.Instructions)
		switch dq.Kind {
		case "noul":
			qs = append(qs, mirrorNoul{ID: dq.ID, Instructions: ins, True: rawToAny(t, dq.True), False: rawToAny(t, dq.False)})
		case "choice":
			opts := map[string]any{}
			for k, v := range dq.Options {
				opts[k] = rawToAny(t, v)
			}
			qs = append(qs, mirrorChoice{ID: dq.ID, Instructions: ins, Options: opts})
		case "score":
			var levels []any
			for _, lv := range dq.Levels {
				levels = append(levels, rawToAny(t, lv))
			}
			qs = append(qs, mirrorScore{ID: dq.ID, Instructions: ins, Levels: levels})
		default:
			t.Fatalf("unknown question kind %q", dq.Kind)
		}
	}
	resp := mirrorResponse{Model: d.Model, Nouls: map[string]float64{}, Choices: map[string]mirrorChoiceAnswer{}, Scores: map[string]mirrorScoreAnswer{}, Usage: d.Usage}
	for _, da := range d.Answers {
		switch {
		case da.Belief != nil:
			resp.Nouls[da.ID] = *da.Belief
		case da.Score != nil:
			resp.Scores[da.ID] = mirrorScoreAnswer{Score: *da.Score, Levels: da.Levels, Probabilities: da.LevelProbabilities, Confidence: da.Confidence}
		case da.Choice != "":
			resp.Choices[da.ID] = mirrorChoiceAnswer{Choice: da.Choice, Probabilities: da.Probabilities, Confidence: da.Confidence}
		default:
			t.Fatalf("answer %q carries no kind group", da.ID)
		}
	}
	return state, qs, resp
}

// TestDecisionMirrorRoundTrip converts a hand-built decide-shaped Ask
// (noul with criteria, choice, score) to DecisionEvent, through the JSON
// wire, and back, requiring equality: the root vocabulary can carry every
// fact the decide conversion layer must round-trip.
func TestDecisionMirrorRoundTrip(t *testing.T) {
	origState := map[string]any{"host": "db-1", "errors": json.Number("42")}
	origQuestions := []mirrorQuestion{
		mirrorNoul{ID: "risky", Instructions: "is this risky?", True: "clearly risky", False: "clearly safe"},
		mirrorChoice{ID: "next_action", Instructions: "stay or go?", Options: map[string]any{"go": "leave", "stay": "remain"}},
		mirrorScore{ID: "severity", Instructions: "rate it", Levels: []any{"low", "mid", "high"}},
	}
	origResponse := mirrorResponse{
		Model: "jev-1.13.0",
		Nouls: map[string]float64{"risky": 0.25},
		Choices: map[string]mirrorChoiceAnswer{
			"next_action": {Choice: "go", Probabilities: map[string]float64{"go": 0.7, "stay": 0.3}, Confidence: 0.8},
		},
		Scores: map[string]mirrorScoreAnswer{
			"severity": {Score: 2.09, Levels: []string{"high"}, Probabilities: []float64{0.1, 0.4, 0.5}, Confidence: 0.6},
		},
		Usage: Usage{InputTokens: 540, OutputTokens: 80},
	}

	ev := mirrorToDecisionEvent(t, origState, origQuestions, origResponse)

	// Through the wire, like a durable sink and a replay source would.
	data, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Event
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v\njson: %s", err, data)
	}
	if !reflect.DeepEqual(ev, back) {
		t.Fatalf("decision event did not survive the wire\nwant: %#v\ngot:  %#v", ev, back)
	}

	backState, backQuestions, backResponse := decisionToMirror(t, back)
	if !reflect.DeepEqual(origState, backState) {
		t.Fatalf("state lost: orig %v back %v", origState, backState)
	}
	if !reflect.DeepEqual(origResponse, backResponse) {
		t.Fatalf("response lost:\norig: %+v\nback: %+v", origResponse, backResponse)
	}
	if len(backQuestions) != len(origQuestions) {
		t.Fatalf("question count %d, want %d", len(backQuestions), len(origQuestions))
	}
	// decisionToMirror returns questions in encoded (ID-sorted) order; the
	// originals are compared as a set with type-sensitive equality.
	byID := map[string]mirrorQuestion{}
	for _, q := range backQuestions {
		byID[q.mirrorID()] = q
	}
	for _, q := range origQuestions {
		got, ok := byID[q.mirrorID()]
		if !ok {
			t.Fatalf("question %q missing after round-trip", q.mirrorID())
		}
		if !reflect.DeepEqual(q, got) {
			t.Fatalf("question %q lost:\norig: %+v\nback: %+v", q.mirrorID(), q, got)
		}
	}
}

// TestEmptyPayloadRoundTrip pushes an all-zero payload of every kind
// through the wire: omitempty must collapse the empty halves (State,
// Answers, Finalized, Exec.Command, ...) to absent and decode back to the
// zero payload with DeepEqual, so sinks never see phantom distinctions.
func TestEmptyPayloadRoundTrip(t *testing.T) {
	kinds := map[EventKind]func(*Event){
		KindStart:      func(e *Event) { e.Start = &StartEvent{} },
		KindCompletion: func(e *Event) { e.Completion = &CompletionEvent{} },
		KindAttempt:    func(e *Event) { e.Attempt = &AttemptEvent{} },
		KindToolRun:    func(e *Event) { e.ToolRun = &ToolRunEvent{} },
		KindCompaction: func(e *Event) { e.Compaction = &CompactionEvent{} },
		KindSteer:      func(e *Event) { e.Steer = &SteerEvent{} },
		KindFinalize:   func(e *Event) { e.Finalize = &FinalizeEvent{} },
		KindDecision:   func(e *Event) { e.Decision = &DecisionEvent{} },
		KindEmbed:      func(e *Event) { e.Embed = &EmbedEvent{} },
		KindExec:       func(e *Event) { e.Exec = &ExecEvent{} },
	}
	if len(kinds) != 10 {
		t.Fatalf("%d kinds wired, want 10", len(kinds))
	}
	for kind, setPayload := range kinds {
		t.Run(string(kind), func(t *testing.T) {
			ev := NewEvent(context.Background(), kind)
			setPayload(&ev)
			data, err := json.Marshal(ev)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var back Event
			if err := json.Unmarshal(data, &back); err != nil {
				t.Fatalf("unmarshal: %v\njson: %s", err, data)
			}
			if !reflect.DeepEqual(ev, back) {
				t.Fatalf("empty %s payload did not round-trip\njson: %s\nwant: %#v\ngot:  %#v", kind, data, ev, back)
			}
		})
	}
}

// payloadSlots returns, in Kind-declaration order, one payload setter per
// Event payload field, so the Validate matrix can set any slot on any kind
// and name the offending field in assertions.
func payloadSlots() []struct {
	kind  EventKind
	field string
	set   func(*Event)
} {
	return []struct {
		kind  EventKind
		field string
		set   func(*Event)
	}{
		{KindStart, "Start", func(e *Event) { e.Start = &StartEvent{} }},
		{KindCompletion, "Completion", func(e *Event) { e.Completion = &CompletionEvent{} }},
		{KindAttempt, "Attempt", func(e *Event) { e.Attempt = &AttemptEvent{} }},
		{KindToolRun, "ToolRun", func(e *Event) { e.ToolRun = &ToolRunEvent{} }},
		{KindCompaction, "Compaction", func(e *Event) { e.Compaction = &CompactionEvent{} }},
		{KindSteer, "Steer", func(e *Event) { e.Steer = &SteerEvent{} }},
		{KindFinalize, "Finalize", func(e *Event) { e.Finalize = &FinalizeEvent{} }},
		{KindDecision, "Decision", func(e *Event) { e.Decision = &DecisionEvent{} }},
		{KindEmbed, "Embed", func(e *Event) { e.Embed = &EmbedEvent{} }},
		{KindExec, "Exec", func(e *Event) { e.Exec = &ExecEvent{} }},
	}
}

// TestEventValidate walks the full kind x payload matrix: the ten matching
// pairs pass, all 90 mismatches fail with an error naming the kind and the
// offending payload field, and the shape rules (zero payloads, two
// payloads, empty Kind, unknown Kind, SchemaVersion 0, a future
// SchemaVersion) each behave as documented.
func TestEventValidate(t *testing.T) {
	slots := payloadSlots()
	for _, kc := range slots {
		for _, sc := range slots {
			ev := NewEvent(context.Background(), kc.kind)
			sc.set(&ev)
			err := ev.Validate()
			if sc.kind == kc.kind {
				if err != nil {
					t.Errorf("%s event with its own %s payload: Validate = %v, want nil", kc.kind, sc.field, err)
				}
				continue
			}
			if err == nil {
				t.Errorf("%s event with foreign %s payload: Validate = nil, want an error", kc.kind, sc.field)
				continue
			}
			for _, want := range []string{string(kc.kind), sc.field} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("%s event with foreign %s payload: error %q does not name %q", kc.kind, sc.field, err, want)
				}
			}
		}
	}

	t.Run("zero payloads", func(t *testing.T) {
		ev := Event{Kind: KindToolRun, SchemaVersion: EventSchemaVersion}
		err := ev.Validate()
		if err == nil {
			t.Fatal("Validate = nil, want an error")
		}
		for _, want := range []string{string(KindToolRun), "ToolRun"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not name %q", err, want)
			}
		}
	})

	t.Run("two payloads", func(t *testing.T) {
		ev := Event{Kind: KindEmbed, SchemaVersion: EventSchemaVersion}
		ev.Embed = &EmbedEvent{}
		ev.Exec = &ExecEvent{}
		err := ev.Validate()
		if err == nil {
			t.Fatal("Validate = nil, want an error")
		}
		for _, want := range []string{string(KindEmbed), "Embed", "Exec"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not name %q", err, want)
			}
		}
	})

	t.Run("empty kind", func(t *testing.T) {
		ev := Event{Kind: "", SchemaVersion: EventSchemaVersion, Start: &StartEvent{}}
		if err := ev.Validate(); err == nil {
			t.Error("Validate on Kind \"\" = nil, want an error")
		}
	})

	t.Run("unknown kind", func(t *testing.T) {
		ev := Event{Kind: EventKind("span"), SchemaVersion: EventSchemaVersion, Start: &StartEvent{}}
		if err := ev.Validate(); err == nil {
			t.Error("Validate on an unknown kind = nil, want an error")
		}
	})

	t.Run("schema version zero", func(t *testing.T) {
		// A hand-built Event that skipped NewEvent carries 0 and must fail,
		// even with an otherwise-perfect kind/payload pair.
		ev := Event{Kind: KindSteer, SchemaVersion: 0, Steer: &SteerEvent{}}
		err := ev.Validate()
		if err == nil {
			t.Fatal("Validate on SchemaVersion 0 = nil, want an error")
		}
		for _, want := range []string{string(KindSteer), "SchemaVersion"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not name %q", err, want)
			}
		}
	})

	t.Run("future schema version passes", func(t *testing.T) {
		// Any non-zero value passes: a newer schema is a sink's business,
		// not Validate's.
		ev := Event{Kind: KindSteer, SchemaVersion: EventSchemaVersion + 1, Steer: &SteerEvent{}}
		if err := ev.Validate(); err != nil {
			t.Errorf("Validate on a future SchemaVersion = %v, want nil", err)
		}
	})
}

// TestGoldenEventsValidate proves NewEvent plus a payload assignment
// produces a Validate-clean event for every kind: the golden fixtures are
// built exactly that way, so the pair the sinks will rely on is the pair
// Validate accepts.
func TestGoldenEventsValidate(t *testing.T) {
	for _, g := range goldEvents() {
		if err := g.ev.Validate(); err != nil {
			t.Errorf("golden %s event: Validate = %v, want nil", g.name, err)
		}
	}
	for _, s := range payloadSlots() {
		ev := NewEvent(context.Background(), s.kind)
		s.set(&ev)
		if err := ev.Validate(); err != nil {
			t.Errorf("NewEvent(%s) + %s payload: Validate = %v, want nil", s.kind, s.field, err)
		}
	}
}

package llmkit

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
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
	return []struct {
		name string
		ev   Event
	}{
		{
			name: "start",
			ev: Event{
				Kind: KindStart, RunID: goldRun, ParentRunID: goldParent, Step: 1,
				Time: goldTime, SchemaVersion: EventSchemaVersion,
				Start: &Start{Task: "summarize the ledger", Continued: true, Tools: []string{"search", "calc"}},
			},
		},
		{
			name: "completion",
			ev: Event{
				Kind: KindCompletion, RunID: goldRun, Step: 2,
				Time: goldTime, Duration: 1500 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Completion: &Completion{
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
				Kind: KindAttempt, RunID: goldRun,
				Time: goldTime, Duration: 250 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				Attempt: &Attempt{
					Attempt: 1,
					Request: Request{Messages: []Message{TextMessage(RoleUser, "hi")}},
					// A failed attempt carries the zero Response; its Usage
					// field has no omitempty (a struct tag cannot omit a
					// struct), so the wire shows it as {"usage":{}}.
					Response: Response{},
					Err:      "429 too many requests",
					Provider: "openai",
					Model:    "gpt",
				},
			},
		},
		{
			name: "tool_run",
			ev: Event{
				Kind: KindToolRun, RunID: goldRun, Step: 3,
				Time: goldTime, Duration: 200 * time.Millisecond, SchemaVersion: EventSchemaVersion,
				ToolRun: &ToolRun{
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
				ToolRun: &ToolRun{
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
				Compaction: &Compaction{BeforeTokens: 9000, AfterTokens: 4000, Pruned: 6},
			},
		},
		{
			name: "steer",
			ev: Event{
				Kind: KindSteer, RunID: goldRun, Step: 5,
				Time: goldTime, SchemaVersion: EventSchemaVersion,
				Steer: &Steer{Message: TextMessage(RoleUser, "focus on taxes"), FollowUp: true},
			},
		},
		{
			name: "finalize",
			ev: Event{
				Kind: KindFinalize, RunID: goldRun, ParentRunID: goldParent,
				Time: goldTime, Duration: time.Minute, SchemaVersion: EventSchemaVersion,
				Finalize: &Finalize{
					TruncationReason: "max_steps",
					Iterations:       8,
					Usage:            Usage{InputTokens: 1000, OutputTokens: 200},
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
					Questions: []DecisionQuestion{
						{ID: "q1", Kind: "noul", Instructions: json.RawMessage(`{"prompt":"confident?"}`)},
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
					},
					Answers: []DecisionAnswer{
						{ID: "q1", Belief: &belief},
						{ID: "q2", Choice: "go", Probabilities: map[string]float64{"go": 0.7, "stay": 0.3}, Confidence: 0.8},
						{ID: "q3", Levels: []string{"high"}, LevelProbabilities: []float64{0.2, 0.8}},
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
				Embed: &Embed{Model: "text-embed-3", Inputs: 4, Dimensions: 1536, CacheHits: 1, Usage: Usage{InputTokens: 900}},
			},
		},
		{
			name: "exec",
			ev: Event{
				Kind: KindExec, RunID: goldRun,
				Time: goldTime, Duration: 2 * time.Second, SchemaVersion: EventSchemaVersion,
				Exec: &Exec{
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
  "step": 1,
  "time": "2026-09-20T12:30:00Z",
  "schema_version": 1,
  "start": {
    "task": "summarize the ledger",
    "continued": true,
    "tools": [
      "search",
      "calc"
    ]
  }
}`,
		"completion": `{
  "kind": "completion",
  "run_id": "1758366600000-deadbeef00112233",
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
  "time": "2026-09-20T12:30:00Z",
  "duration": 60000000000,
  "schema_version": 1,
  "finalize": {
    "truncation_reason": "max_steps",
    "iterations": 8,
    "usage": {
      "input_tokens": 1000,
      "output_tokens": 200
    }
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
    "questions": [
      {
        "id": "q1",
        "kind": "noul",
        "instructions": {"prompt":"confident?"}
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
        "levels": [
          "high"
        ],
        "level_probabilities": [
          0.2,
          0.8
        ]
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
	// event. The fan-out passes the event by value, so mutation of the
	// struct is impossible; this pins that a payload pointer swap inside one
	// observer stays invisible to the emitter and to later observers.
	var seenBySecond Event
	ev := Event{Kind: KindExec, Exec: &Exec{Backend: "docker"}}
	chain := Observers(
		ObserverFunc(func(_ context.Context, e Event) { e.Exec = &Exec{Backend: "hijacked"} }),
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

package sqlite_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/store/sqlite"
)

func observeAll(s *sqlite.Store, evs []llmkit.Event) {
	for _, ev := range evs {
		s.Observe(context.Background(), ev)
	}
}

// checkIdentity observes evs and reads the run back.
func checkIdentity(t testing.TB, s *sqlite.Store, log *errLog, run llmkit.RunID, evs []llmkit.Event) {
	t.Helper()
	observeAll(s, evs)
	noErrors(t, log)
	got, err := s.Events(context.Background(), run)
	if err != nil {
		t.Fatalf("Events(%q): %v", run, err)
	}
	assertIdentical(t, got, evs)
}

func TestP1_HandBuiltCorpusCoversEveryKind(t *testing.T) {
	s, log, _ := openStore(t)
	evs := allKinds("run-all")
	seen := map[llmkit.EventKind]bool{}
	for _, ev := range evs {
		seen[ev.Kind] = true
		if err := ev.Validate(); err != nil {
			t.Fatalf("corpus event invalid: %v", err)
		}
	}
	if len(seen) != 10 {
		t.Fatalf("corpus covers %d kinds, want 10", len(seen))
	}
	checkIdentity(t, s, log, "run-all", evs)
}

func TestP1_RunnerRuns(t *testing.T) {
	cases := map[string]func(t *testing.T, s *sqlite.Store) []llmkit.Event{
		"tools": func(t *testing.T, s *sqlite.Store) []llmkit.Event {
			r := agent.NewRunner(loopClient(3), []agent.Tool{markerTool()}, "be brief", agent.WithObserver(s))
			return run(t, r, "task one")
		},
		"parallel tools": func(t *testing.T, s *sqlite.Store) []llmkit.Event {
			client := funcClient(func(req llmkit.Request) (llmkit.Response, error) {
				if toolResults(req) >= 2 {
					return llmkit.Response{Text: "done", StopReason: llmkit.StopEndTurn}, nil
				}
				return llmkit.Response{StopReason: llmkit.StopToolUse, ToolCalls: []llmkit.ToolCall{
					{ID: "p1", Name: "echo", Arguments: json.RawMessage(`{"n":1}`)},
					{ID: "p2", Name: "echo", Arguments: json.RawMessage(`{"n":2}`)},
					{ID: "p3", Name: "echo", Arguments: json.RawMessage(`{"n":3}`)},
				}}, nil
			})
			r := agent.NewRunner(client, []agent.Tool{markerTool()}, "sys", agent.WithObserver(s), agent.WithParallelTools())
			return run(t, r, "fan out")
		},
		"denial": func(t *testing.T, s *sqlite.Store) []llmkit.Event {
			deny := agent.ToolPolicyFunc(func(context.Context, *llmkit.ToolCall) error { return errors.New("not today") })
			r := agent.NewRunner(loopClient(2), []agent.Tool{markerTool()}, "sys", agent.WithObserver(s), agent.WithToolPolicy(deny))
			return run(t, r, "try it")
		},
		"compaction": func(t *testing.T, s *sqlite.Store) []llmkit.Event {
			compactor := agent.CompactorFunc(func(_ context.Context, in agent.CompactInput) (agent.CompactOutput, error) {
				if len(in.Messages) <= 4 {
					return agent.CompactOutput{}, nil
				}
				keep := in.Messages[len(in.Messages)-2:]
				msgs := append([]llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "[conversation summary]\nearlier steps")}, keep...)
				return agent.CompactOutput{Messages: msgs, Pruned: len(in.Messages) - 2, Summary: "earlier steps"}, nil
			})
			r := agent.NewRunner(loopClient(6), []agent.Tool{markerTool()}, "sys",
				agent.WithObserver(s), agent.WithCompactor(compactor), agent.WithLimits(agent.Limits{HistoryTokenBudget: 1}))
			return run(t, r, "long task")
		},
		"steering": func(t *testing.T, s *sqlite.Store) []llmkit.Event {
			st := agent.NewSteering()
			st.Steer(llmkit.Text("change course"))
			st.FollowUp(llmkit.Text("one more thing"))
			r := agent.NewRunner(loopClient(1), []agent.Tool{markerTool()}, "sys", agent.WithObserver(s))
			return run(t, r, "steer me", agent.WithSteering(st))
		},
		"RunJSON": func(t *testing.T, s *sqlite.Store) []llmkit.Event {
			client := funcClient(func(llmkit.Request) (llmkit.Response, error) {
				return llmkit.Response{Text: `{"answer": 42}`, StopReason: llmkit.StopEndTurn}, nil
			})
			r := agent.NewRunner(client, nil, "sys", agent.WithObserver(s))
			var out struct{ Answer int }
			o, err := r.RunJSON(context.Background(), "compute", json.RawMessage(`{"type":"object"}`), &out)
			if err != nil {
				t.Fatalf("RunJSON: %v", err)
			}
			return o.Transcript.Record
		},
	}
	for name, build := range cases {
		t.Run(name, func(t *testing.T) {
			s, log, _ := openStore(t)
			observed := build(t, s)
			noErrors(t, log)
			if len(observed) == 0 {
				t.Fatal("run observed no events")
			}
			got, err := s.Events(context.Background(), observed[0].RunID)
			if err != nil {
				t.Fatalf("Events: %v", err)
			}
			assertIdentical(t, got, observed)
		})
	}
}

func TestP1_RunnerRunsCoverTheirKinds(t *testing.T) {
	s, _, _ := openStore(t)
	compactor := agent.CompactorFunc(func(_ context.Context, in agent.CompactInput) (agent.CompactOutput, error) {
		if len(in.Messages) <= 4 {
			return agent.CompactOutput{}, nil
		}
		keep := in.Messages[len(in.Messages)-2:]
		msgs := append([]llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "[conversation summary]\nx")}, keep...)
		return agent.CompactOutput{Messages: msgs, Pruned: len(in.Messages) - 2}, nil
	})
	st := agent.NewSteering()
	st.Steer(llmkit.Text("steer"))
	r := agent.NewRunner(loopClient(6), []agent.Tool{markerTool()}, "sys",
		agent.WithObserver(s), agent.WithCompactor(compactor), agent.WithLimits(agent.Limits{HistoryTokenBudget: 1}))
	kinds := map[llmkit.EventKind]bool{}
	for _, ev := range run(t, r, "t", agent.WithSteering(st)) {
		kinds[ev.Kind] = true
	}
	for _, k := range []llmkit.EventKind{llmkit.KindStart, llmkit.KindCompletion, llmkit.KindToolRun, llmkit.KindCompaction, llmkit.KindSteer, llmkit.KindFinalize} {
		if !kinds[k] {
			t.Errorf("compaction+steering run emitted no %s event; the identity cases above would not exercise it", k)
		}
	}
}

// exportRecipe is the documented export: Events then SaveJSONL.
func exportRecipe(t testing.TB, s *sqlite.Store, run llmkit.RunID) []byte {
	t.Helper()
	evs, err := s.Events(context.Background(), run)
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	var buf bytes.Buffer
	if err := (&agent.Transcript{Record: evs}).SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	return buf.Bytes()
}

// canonLines decodes and re-marshals each JSONL line.
func canonLines(t testing.TB, b []byte) []string {
	t.Helper()
	var out []string
	for _, line := range bytes.Split(bytes.TrimSuffix(b, []byte("\n")), []byte("\n")) {
		var ev llmkit.Event
		if err := json.Unmarshal(line, &ev); err != nil {
			t.Fatalf("decode line: %v", err)
		}
		re, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("re-marshal line: %v", err)
		}
		out = append(out, string(re))
	}
	return out
}

// TestP1_ExportRecipeMatchesSinkFile runs the store and the JSONL sink side
// by side off one Observers chain.
func TestP1_ExportRecipeMatchesSinkFile(t *testing.T) {
	cases := map[string]struct {
		toolText  string
		canonical bool
	}{
		"canonical":     {toolText: "plain result <b> & \u2028", canonical: true},
		"invalid utf-8": {toolText: "bad \xff\xfe a bytes", canonical: false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			s, log, _ := openStore(t)
			dir := t.TempDir()
			sink := agent.JSONL(dir, log.add)
			tool := fnTool{name: "echo", run: func(context.Context, json.RawMessage) ([]llmkit.Block, error) {
				return []llmkit.Block{llmkit.Text(tc.toolText)}, nil
			}}
			r := agent.NewRunner(loopClient(3), []agent.Tool{tool}, "sys", agent.WithObserver(llmkit.Observers(sink, s)))
			observed := run(t, r, "export me")
			noErrors(t, log)
			runID := observed[0].RunID

			file, err := os.ReadFile(filepath.Join(dir, string(runID)+".jsonl"))
			if err != nil {
				t.Fatalf("sink file: %v", err)
			}
			export := exportRecipe(t, s, runID)

			allCanon := true
			for _, ev := range observed {
				if b, _ := json.Marshal(ev); string(b) != canon(t, ev) {
					allCanon = false
				}
			}
			if allCanon != tc.canonical {
				t.Fatalf("every observed event canonical = %v, want %v (the case would not test what it names)", allCanon, tc.canonical)
			}
			if tc.canonical && !bytes.Equal(export, file) {
				t.Fatalf("export is not byte-equal to the sink file\nexport: %s\nfile:   %s", clip(string(export)), clip(string(file)))
			}
			if g, w := canonLines(t, export), canonLines(t, file); strings.Join(g, "\n") != strings.Join(w, "\n") {
				t.Fatalf("export lines differ from sink lines in canon form\nexport: %s\nfile:   %s", clip(strings.Join(g, "\n")), clip(strings.Join(w, "\n")))
			}
		})
	}
}

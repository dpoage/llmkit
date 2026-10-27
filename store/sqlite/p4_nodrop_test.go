package sqlite_test

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/store/sqlite"
)

func startEvent(run llmkit.RunID) llmkit.Event {
	e := newEv(run, llmkit.KindStart)
	e.Start = &llmkit.StartEvent{Task: "t"}
	return e
}

func rowCount(t testing.TB, db *sql.DB, table string) int {
	t.Helper()
	var n int
	if err := db.QueryRow("SELECT count(*) FROM " + table).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// refusedOnce requires exactly one onErr call naming the run and the kind.
func refusedOnce(t testing.TB, log *errLog, run string, kind llmkit.EventKind) {
	t.Helper()
	errs := log.all()
	if len(errs) != 1 {
		t.Fatalf("onErr called %d times, want exactly 1: %v", len(errs), errs)
	}
	if !contains(errs[0].Error(), run, string(kind)) {
		t.Fatalf("onErr error %q does not name run %q and kind %q", errs[0], run, kind)
	}
}

func TestP4_ObserveAfterClose(t *testing.T) {
	s, log, _ := openStore(t)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s.Observe(context.Background(), startEvent("run-closed"))
	refusedOnce(t, log, "run-closed", llmkit.KindStart)
	if _, err := s.Events(context.Background(), "run-closed"); err == nil {
		t.Fatal("Events after Close succeeded")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestP4_EventJSONRejects(t *testing.T) {
	cases := map[string]llmkit.Event{
		"NaN in a DecisionAnswer": func() llmkit.Event {
			e := newEv("run-nan", llmkit.KindDecision)
			e.Decision = &llmkit.DecisionEvent{Answers: []llmkit.DecisionAnswer{{ID: "q", Belief: f64(math.NaN())}}}
			return e
		}(),
		"invalid ToolDef.Parameters": func() llmkit.Event {
			e := newEv("run-nan", llmkit.KindCompletion)
			e.Completion = &llmkit.CompletionEvent{Request: llmkit.Request{
				Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "kept out of the messages table")},
				Tools:    []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(`{"type":`)}},
			}}
			return e
		}(),
		"invalid DispatchedArguments": func() llmkit.Event {
			e := newEv("run-nan", llmkit.KindToolRun)
			e.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: "c", Name: "n", Arguments: json.RawMessage(`{}`)}, DispatchedArguments: json.RawMessage(`{`)}
			return e
		}(),
	}
	for name, ev := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := json.Marshal(ev); err == nil {
				t.Fatal("corpus event marshals; the case would not test a refusal")
			}
			s, log, path := openStore(t)
			s.Observe(context.Background(), ev)
			refusedOnce(t, log, "run-nan", ev.Kind)
			db := openRaw(t, path)
			if n := rowCount(t, db, "events") + rowCount(t, db, "messages"); n != 0 {
				t.Fatalf("a refused event left %d rows", n)
			}
		})
	}
}

// docDepth measures a JSON document's nesting with encoding/json's tokenizer,
// independently of the store.
func docDepth(t testing.TB, b []byte) int {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(b))
	depth, deepest := 0, 0
	for {
		tok, err := dec.Token()
		if err != nil {
			break
		}
		if d, ok := tok.(json.Delim); ok {
			if d == '{' || d == '[' {
				depth++
				deepest = max(deepest, depth)
			} else {
				depth--
			}
		}
	}
	return deepest
}

func sqliteAccepts(t testing.TB, db *sql.DB, doc []byte) bool {
	t.Helper()
	var ok int
	if err := db.QueryRow("SELECT json_valid(?)", string(doc)).Scan(&ok); err != nil {
		t.Fatal(err)
	}
	return ok == 1
}

func decisionAtStateDepth(run llmkit.RunID, depth int) llmkit.Event {
	e := newEv(run, llmkit.KindDecision)
	e.Decision = &llmkit.DecisionEvent{
		Backend: "b", Model: "m", State: nested(depth, "[", "]"),
		Questions: []llmkit.DecisionQuestion{{ID: "q", Kind: "score"}},
		Answers:   []llmkit.DecisionAnswer{{ID: "q", Score: f64(0.5)}},
	}
	return e
}

// exerciseViews reads every row of every view and every cookbook query.
func exerciseViews(t testing.TB, db *sql.DB) {
	t.Helper()
	for _, view := range []string{"runs", "spend", "tool_calls", "decisions"} {
		rows, err := db.Query("SELECT * FROM " + view)
		if err != nil {
			t.Fatalf("view %s: %v", view, err)
		}
		cols, _ := rows.Columns()
		vals := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		for rows.Next() {
			if err := rows.Scan(ptrs...); err != nil {
				t.Fatalf("view %s: %v", view, err)
			}
		}
		if err := rows.Err(); err != nil {
			t.Fatalf("view %s: %v", view, err)
		}
		_ = rows.Close()
	}
	for _, q := range []string{"Q1", "Q2", "Q3", "Q4", "Q5", "Q7"} {
		rows, err := db.Query(sqlite.CookbookQuery(q), sql.Named("run", "run-depth"), sql.Named("since_ns", 0), sql.Named("threshold", 1.0))
		if err != nil {
			t.Fatalf("%s: %v", q, err)
		}
		for rows.Next() {
		}
		if err := rows.Err(); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
		_ = rows.Close()
	}
}

func TestP4_PayloadDepth(t *testing.T) {
	s, log, path := openStore(t)
	db := openRaw(t, path)
	const run = "run-depth"
	s.Observe(context.Background(), startEvent(run))

	// The document's depth is the State's plus a constant offset; measure it.
	probe, _ := json.Marshal(decisionAtStateDepth(run, 10))
	offset := docDepth(t, probe) - 10
	if offset != 2 {
		t.Fatalf("a decision's document is State depth + %d, want + 2", offset)
	}
	maxState := sqlite.MaxPayloadDepth - offset
	if !sqliteAccepts(t, db, mustMarshal(t, decisionAtStateDepth(run, maxState))) || sqliteAccepts(t, db, mustMarshal(t, decisionAtStateDepth(run, maxState+1))) {
		t.Fatalf("SQLite's own limit is not at State depth %d", maxState)
	}

	// Stored: State 998 (document 1000), and a tool run at the same depth.
	s.Observe(context.Background(), decisionAtStateDepth(run, maxState))
	tool := newEv(run, llmkit.KindToolRun)
	tool.ToolRun = &llmkit.ToolRunEvent{
		Call:                llmkit.ToolCall{ID: "c", Name: "n", Arguments: json.RawMessage(`{}`)},
		DispatchedArguments: nested(maxState, "[", "]"),
	}
	s.Observe(context.Background(), tool)
	noErrors(t, log)
	if got := rowCount(t, db, "events"); got != 3 {
		t.Fatalf("%d events stored, want 3", got)
	}
	exerciseViews(t, db)
	if _, err := s.Events(context.Background(), run); err != nil {
		t.Fatalf("Events over a run holding depth-1000 documents: %v", err)
	}

	// Refused, each with one onErr naming run and kind and no write: 999 and 9997.
	for _, state := range []int{maxState + 1, 9997} {
		before := len(log.all())
		e := decisionAtStateDepth(run, state)
		if d := docDepth(t, mustMarshal(t, e)); d != state+offset {
			t.Fatalf("test event is %d deep, want %d", d, state+offset)
		}
		s.Observe(context.Background(), e)
		errs := log.all()
		if len(errs) != before+1 || !contains(errs[before].Error(), run, string(llmkit.KindDecision)) {
			t.Fatalf("State %d: onErr calls %v (before: %d), want one naming run and kind", state, errs, before)
		}
	}
	if got := rowCount(t, db, "events"); got != 3 {
		t.Fatalf("%d events stored after the refusals, want 3", got)
	}
	exerciseViews(t, db)
}

func mustMarshal(t testing.TB, ev llmkit.Event) []byte {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestP4_DepthRefusalKeepsMessagesOut: an over-depth Completion writes no
// message rows either.
func TestP4_DepthRefusalKeepsMessagesOut(t *testing.T) {
	s, log, path := openStore(t)
	e := newEv("run-deepmsg", llmkit.KindCompletion)
	e.Completion = &llmkit.CompletionEvent{
		Request:  llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "m")}, ResponseSchema: nested(1000, "[", "]")},
		Response: llmkit.Response{Text: "ok"},
	}
	s.Observe(context.Background(), e)
	refusedOnce(t, log, "run-deepmsg", llmkit.KindCompletion)
	db := openRaw(t, path)
	if n := rowCount(t, db, "events") + rowCount(t, db, "messages"); n != 0 {
		t.Fatalf("a refused event left %d rows", n)
	}
}

func TestP4_WriteError(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("file permissions do not stop root")
	}
	path := filepath.Join(t.TempDir(), "ro.db")
	s, err := sqlite.Open(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	s.Observe(context.Background(), startEvent("run-ro"))
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o444); err != nil {
		t.Fatal(err)
	}

	log := &errLog{}
	ro, err := sqlite.Open(path, log.add)
	if err != nil {
		t.Fatalf("no-op re-Open of a read-only file: %v", err)
	}
	defer func() { _ = ro.Close() }()
	ro.Observe(context.Background(), startEvent("run-ro-two"))
	refusedOnce(t, log, "run-ro-two", llmkit.KindStart)
	evs, err := ro.Events(context.Background(), "run-ro")
	if err != nil || len(evs) != 1 {
		t.Fatalf("read of the earlier run: %d events, %v", len(evs), err)
	}
	if _, err := ro.Events(context.Background(), "run-ro-two"); err == nil {
		t.Fatal("the failed write is visible")
	}
}

func TestP4_ErrorTextNamesRunAndKindForEveryKind(t *testing.T) {
	s, log, _ := openStore(t)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	for _, ev := range allKinds("run-kinds") {
		s.Observe(context.Background(), ev)
	}
	errs := log.all()
	if len(errs) != 13 {
		t.Fatalf("%d onErr calls for 13 events", len(errs))
	}
	for i, ev := range allKinds("run-kinds") {
		if !strings.Contains(errs[i].Error(), string(ev.Kind)) || !strings.Contains(errs[i].Error(), "run-kinds") {
			t.Errorf("error %d %q does not name kind %s and the run", i, errs[i], ev.Kind)
		}
	}
}

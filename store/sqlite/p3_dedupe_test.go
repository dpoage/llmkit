package sqlite_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
)

// blockLoopClient is loopClient whose replies also carry their text as a block.
func blockLoopClient(steps int) funcClient {
	inner := loopClient(steps)
	return func(req llmkit.Request) (llmkit.Response, error) {
		resp, err := inner(req)
		if resp.Text != "" {
			resp.Blocks = []llmkit.Block{llmkit.Text(resp.Text)}
		}
		return resp, err
	}
}

// textColumns lists every TEXT column of every table.
func textColumns(t testing.TB, db *sql.DB) [][2]string {
	t.Helper()
	rows, err := db.Query(`SELECT m.name, p.name FROM sqlite_schema m, pragma_table_info(m.name) p WHERE m.type = 'table' AND p.type = 'TEXT'`)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	var cols [][2]string
	for rows.Next() {
		var c [2]string
		if err := rows.Scan(&c[0], &c[1]); err != nil {
			t.Fatal(err)
		}
		cols = append(cols, c)
	}
	if len(cols) == 0 {
		t.Fatal("no TEXT columns found")
	}
	return cols
}

// occurrences counts the true occurrences of marker across every text column.
func occurrences(t testing.TB, db *sql.DB, cols [][2]string, marker string) int {
	t.Helper()
	total := 0
	for _, c := range cols {
		var n sql.NullInt64
		q := fmt.Sprintf(`SELECT SUM((length(%[2]s) - length(replace(%[2]s, ?1, ''))) / length(?1)) FROM %[1]s`, c[0], c[1])
		if err := db.QueryRow(q, marker).Scan(&n); err != nil {
			t.Fatalf("count %q in %s.%s: %v", marker, c[0], c[1], err)
		}
		total += int(n.Int64)
	}
	return total
}

// markerCounts runs a steps-turn run and returns, per message role, the
// largest number of times any one of its markers occurs in the database.
func markerCounts(t *testing.T, steps int, client func(int) funcClient) map[llmkit.Role]int {
	t.Helper()
	s, log, path := openStore(t)
	r := agent.NewRunner(client(steps), []agent.Tool{markerTool()}, "sys prompt", agent.WithObserver(s), agent.WithLimits(agent.Limits{MaxIterations: steps + 5}))
	run(t, r, "task-mark-unique")
	noErrors(t, log)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	db := openRaw(t, path)
	cols := textColumns(t, db)

	worst := map[llmkit.Role]int{}
	note := func(role llmkit.Role, marker string) {
		n := occurrences(t, db, cols, marker)
		if n == 0 {
			t.Fatalf("marker %q is nowhere in the database", marker)
		}
		worst[role] = max(worst[role], n)
	}
	note(llmkit.RoleUser, "task-mark-unique")
	for i := range steps {
		note(llmkit.RoleAssistant, fmt.Sprintf("assistant-%03d-mark", i))
		note(llmkit.RoleToolResult, fmt.Sprintf("result-%03d-mark", i))
	}
	return worst
}

func openRaw(t testing.TB, path string) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=busy_timeout(5000)")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// TestP3_MessageOccurrencesDoNotGrowWithSteps: a message is stored once as a
// body and again in the event that produced it (the task in Start, an
// assistant message in its Response, a tool result in its ToolRun), so k is
// 2 per role, 3 for an assistant reply whose text rides in a block too.
func TestP3_MessageOccurrencesDoNotGrowWithSteps(t *testing.T) {
	variants := map[string]struct {
		client func(int) funcClient
		k      map[llmkit.Role]int
	}{
		"text":        {loopClient, map[llmkit.Role]int{llmkit.RoleUser: 2, llmkit.RoleAssistant: 2, llmkit.RoleToolResult: 2}},
		"text+blocks": {blockLoopClient, map[llmkit.Role]int{llmkit.RoleUser: 2, llmkit.RoleAssistant: 3, llmkit.RoleToolResult: 2}},
	}
	for name, v := range variants {
		t.Run(name, func(t *testing.T) {
			at20 := markerCounts(t, 20, v.client)
			at60 := markerCounts(t, 60, v.client)
			for role, k := range v.k {
				if at20[role] > k || at60[role] > k {
					t.Errorf("%s messages: worst marker count %d at 20 steps, %d at 60, want at most %d", role, at20[role], at60[role], k)
				}
				if at20[role] != at60[role] {
					t.Errorf("%s messages: worst marker count differs between 20 steps (%d) and 60 steps (%d)", role, at20[role], at60[role])
				}
			}
			t.Logf("k per role at 20 steps: %v; at 60 steps: %v", at20, at60)
		})
	}
}

// TestP3_RepeatedMessageIsOneBody checks the table the count rests on.
func TestP3_RepeatedMessageIsOneBody(t *testing.T) {
	s, log, path := openStore(t)
	const run = "run-dedupe"
	same := llmkit.TextMessage(llmkit.RoleUser, "same body")
	for step := range 5 {
		e := newEv(run, llmkit.KindCompletion)
		e.Step = step
		e.Completion = &llmkit.CompletionEvent{Request: llmkit.Request{Messages: []llmkit.Message{same, same}}}
		s.Observe(context.Background(), e)
	}
	noErrors(t, log)
	var n int
	if err := openRaw(t, path).QueryRow("SELECT count(*) FROM messages").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("%d message rows for one distinct message, want 1", n)
	}
	var body string
	_ = openRaw(t, path).QueryRow("SELECT body FROM messages").Scan(&body)
	var m llmkit.Message
	if err := json.Unmarshal([]byte(body), &m); err != nil || m.Text() != "same body" {
		t.Fatalf("stored body %q: %v", body, err)
	}
	if strings.Count(body, "same body") != 1 {
		t.Fatalf("body %q", body)
	}
}

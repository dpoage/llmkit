package sqlite_test

// Package examples for docs/store.md; each one runs under go test and the
// docs quote them verbatim.

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/store/sqlite"
	_ "modernc.org/sqlite" // registers the "sqlite" driver the view query opens
)

// scriptedClient stands in for a provider: one tool call, then the answer.
func scriptedClient() *agent.ReplayClient {
	served := llmkit.Identity{Provider: "fake", Model: "scripted"}
	return agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls: []llmkit.ToolCall{{
				ID:        "call-1",
				Name:      "add",
				Arguments: json.RawMessage(`{"a":41,"b":58}`),
			}},
			StopReason: llmkit.StopToolUse,
			Served:     served,
			Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 4},
		},
		{
			Text:       "41 plus 58 is 99.",
			StopReason: llmkit.StopEndTurn,
			Served:     served,
			Usage:      llmkit.Usage{InputTokens: 20, OutputTokens: 6},
		},
	}, llmkit.Capabilities{})
}

func addTool() agent.Tool {
	return agent.Func("add", "adds two numbers", func(_ context.Context, p struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}) (string, error) {
		return fmt.Sprintf("%g", p.A+p.B), nil
	})
}

// recordScriptedRun records one scripted two-step run into a fresh store and
// closes it; done removes the database.
func recordScriptedRun() (dbPath string, id llmkit.RunID, done func()) {
	dir, err := os.MkdirTemp("", "llmkit-store-example-")
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(dir, "run.db")
	id = llmkit.RunID("example-run")
	st, err := sqlite.Open(path, func(err error) { log.Print(err) })
	if err != nil {
		log.Fatal(err)
	}
	runner := agent.NewRunner(scriptedClient(), []agent.Tool{addTool()},
		"You are a helpful assistant.", agent.WithObserver(st))
	if _, err := runner.Run(context.Background(), "What is 41 plus 58?", agent.WithRunID(id)); err != nil {
		log.Fatal(err)
	}
	if err := st.Close(); err != nil {
		log.Fatal(err)
	}
	return path, id, func() { _ = os.RemoveAll(dir) }
}

// ExampleOpen records a run: the store is the Runner's only durable sink,
// and Observe is synchronous — when Run returns the record is complete.
func ExampleOpen() {
	dir, err := os.MkdirTemp("", "llmkit-store-example-")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	st, err := sqlite.Open(filepath.Join(dir, "run.db"), func(err error) { log.Print(err) })
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = st.Close() }()

	client := scriptedClient()
	add := addTool()
	runner := agent.NewRunner(client, []agent.Tool{add}, "You are a helpful assistant.", agent.WithObserver(st))
	outcome, err := runner.Run(context.Background(), "What is 41 plus 58?", agent.WithRunID("example-run"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(outcome.FinalText)
	// Output:
	// 41 plus 58 is 99.
}

// ExampleStore_Events reads a recorded run back in emission order; each
// event is json-identical to the one observed.
func ExampleStore_Events() {
	dbPath, id, done := recordScriptedRun()
	defer done()

	st, err := sqlite.Open(dbPath, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = st.Close() }()

	evs, err := st.Events(context.Background(), id)
	if err != nil {
		log.Fatal(err)
	}
	for _, ev := range evs {
		fmt.Println(ev.Kind)
	}
	// Output:
	// start
	// completion
	// tool_run
	// completion
	// finalize
}

// Example_exportRecipe exports a stored run as a JSONL transcript: Events
// plus Transcript.SaveJSONL. The store has no Export method; this recipe is
// the export, and its file is byte-equal to the agent.JSONL sink's.
func Example_exportRecipe() {
	dbPath, id, done := recordScriptedRun()
	defer done()

	st, err := sqlite.Open(dbPath, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = st.Close() }()

	evs, err := st.Events(context.Background(), id)
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.CreateTemp("", "llmkit-export-*.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.Remove(f.Name()) }()
	if err := (&agent.Transcript{Record: evs}).SaveJSONL(f); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}

	r, err := os.Open(f.Name())
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	lines := 0
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		lines++
	}
	if err := sc.Err(); err != nil {
		log.Fatal(err)
	}
	fmt.Println(lines, "events exported to JSONL")
	// Output:
	// 5 events exported to JSONL
}

// Example_spendView reads the spend view through database/sql: the views are
// plain SQL, so any SQLite client (3.38.0 or newer) can audit a stored run.
func Example_spendView() {
	dbPath, id, done := recordScriptedRun()
	defer done()

	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = db.Close() }()

	rows, err := db.Query(`SELECT step, kind, provider, model, input_tokens, output_tokens
		FROM spend WHERE run_id = ? ORDER BY step`, string(id))
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var step int
		var kind, provider, model string
		var in, out int64
		if err := rows.Scan(&step, &kind, &provider, &model, &in, &out); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("step %d %s %s/%s in=%d out=%d\n", step, kind, provider, model, in, out)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
	// Output:
	// step 1 completion fake/scripted in=10 out=4
	// step 2 completion fake/scripted in=20 out=6
}

// Example_budgetRestart pre-charges a BudgetPool with a stored run's usage
// after a restart, so one ceiling spans the original run and its resumption.
func Example_budgetRestart() {
	dbPath, id, done := recordScriptedRun()
	defer done()

	st, err := sqlite.Open(dbPath, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = st.Close() }()

	evs, err := st.Events(context.Background(), id)
	if err != nil {
		log.Fatal(err)
	}
	var stored llmkit.Usage
	for _, ev := range evs {
		if ev.Kind == llmkit.KindFinalize && ev.Finalize != nil {
			stored = ev.Finalize.Usage
		}
	}
	pool := agent.NewBudgetPool(1000)
	pool.Add(stored.ChargeableTokens(1))
	fmt.Printf("charged %d tokens; %d remain\n", stored.ChargeableTokens(1), pool.Remaining())
	// Output:
	// charged 40 tokens; 960 remain
}

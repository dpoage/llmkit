# The SQLite store

`store/sqlite` keeps a run's whole event stream in one SQLite database file.
A `*sqlite.Store` is an `llmkit.Observer` you register with the agent Runner
and an `llmkit.Source` you replay from. Read-only SQL views answer audit
questions — spend, tool calls, decisions, lineage — from any SQL client
that runs SQLite 3.38.0 or newer.

The store hides its storage format. You see `Open`, `Observe`, `Events`,
`Close`, the `ErrSchemaTooNew` error, and the four documented views. The
tables, the schema version, and the write discipline are the package's own.

## Choosing the sink

One Runner keeps one durable history (see
[design](design.md#observability-and-replay)). When the store is that
history, it is the single source of truth: do not register a JSONL sink
beside it. JSONL is an export you derive later, never a second record.
`WithObserver` appends, so side observers — a spend ledger, metrics — still
compose with the store.

`examples/agent` picks the sink with a flag. `--store <db>` records through
the store and prints the run id:

```bash
# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
go run ./examples/agent --store run.db
```

The store registers like any observer, and `Observe` is synchronous — when
`Run` returns, the record is complete:

```go
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
```

This is `ExampleOpen` from the package examples
(`store/sqlite/example_test.go`), which run under `go test ./store/sqlite`;
the fake client there stands in for a provider.

## Open, observe, read

`sqlite.Open(path, onErr)` opens or creates the database at `path` (the
directory must exist) and applies every pending schema migration. Several
processes may open one file.

`Observe` stores one event and returns once it is committed, so `Events`
and other connections see it on return. Observe never fails the run: an
event the store cannot keep makes exactly one `onErr` call naming the run
id and kind. Refusals are listed under [Limits](#limits).

`Events(ctx, run)` returns the run's events in emission order, each
json-identical to the event observed. It fails with an error wrapping
`llmkit.ErrUnknownRun` for an empty id, a run with no events, and a run
holding more than one Start. `ExampleStore_Events` in the package examples
reads a recorded run back and prints one line per kind.

## The views

The views are the read-side contract: their columns are documented and
stable. The tables are not interface. Query them with any SQL client that
can read this file (see the version floor under [Limits](#limits)); the
package examples' `Example_spendView` does it through `database/sql`:

```go
	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = db.Close() }()
```

| View | One row per | Columns |
|---|---|---|
| `runs` | Start event | `run_id, parent_run_id, task, started_ns, finished_ns, status, truncation_reason, steps, finalized, final_text, err, input_tokens, output_tokens` |
| `spend` | Completion or Decision event | `event_id, run_id, step, kind, time_ns, day, provider, model, purpose, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, reasoning_tokens` |
| `tool_calls` | ToolRun event | `event_id, run_id, step, time_ns, duration_ns, call_id, name, arguments, arguments_b64, result, is_error, denied, deny_reason` |
| `decisions` | answer of a Decision event | `event_id, run_id, step, time_ns, backend, model, err, question_id, question_kind, belief, score, choice, confidence` |

Reading rules:

- `runs` takes the run's last Finalize when there is one; `status` is NULL
  while the run has none (still running, or the process died).
- `time_ns` columns are the event's instant as UTC Unix nanoseconds; `day`
  in `spend` is that instant's UTC day. Every "since T" filter compares
  instants, whatever zone the event's time text carries.
- `tool_calls.arguments` is the call's JSON; when the model sent bytes that
  are not JSON, `arguments` is NULL and `arguments_b64` holds them base64-
  encoded. `denied` and `deny_reason` mark policy denials.
- `decisions` yields one row per answered question; a failed Ask with no
  answers is one row with NULL answer columns and `err` set.

## The cookbook

These are the audit queries. Each is one SELECT over the views; each named
parameter binds as `:since_ns` (UTC Unix nanoseconds), `:run` (a run id),
or `:threshold` (a number). Q6 is the exception: it reads the `events`
table — a table outside the documented view surface — because it is the
store's own Source query.

```sql
-- Q1: spend per provider, model and UTC day since T. Index: events_kind_time.
SELECT day, provider, model,
       COUNT(*)                   AS calls,
       SUM(input_tokens)          AS input_tokens,
       SUM(output_tokens)         AS output_tokens,
       SUM(cache_read_tokens)     AS cache_read_tokens,
       SUM(cache_creation_tokens) AS cache_creation_tokens,
       SUM(reasoning_tokens)      AS reasoning_tokens
FROM spend
WHERE time_ns >= :since_ns
GROUP BY day, provider, model
ORDER BY day, provider, model;

-- Q2: the tool calls of a run in order. Index: events_run_kind.
SELECT step, name, arguments, arguments_b64, is_error, denied, deny_reason
FROM tool_calls
WHERE run_id = :run
ORDER BY event_id;

-- Q3: denied tool calls across runs since T. Index: events_kind_time.
SELECT time_ns, run_id, step, name, arguments, arguments_b64, deny_reason
FROM tool_calls
WHERE denied = 1 AND time_ns >= :since_ns
ORDER BY time_ns;

-- Q4: decisions whose score or belief is below a threshold since T. Index: events_kind_time.
SELECT time_ns, run_id, question_id, question_kind, score, belief
FROM decisions
WHERE time_ns >= :since_ns AND (score < :threshold OR belief < :threshold)
ORDER BY time_ns;

-- Q5: lineage of a run: its ancestors (events_run_kind) and descendants (events_parent).
-- CROSS JOIN keeps the walk driving and runs searched by index; a plain JOIN lets the planner scan.
WITH RECURSIVE
up(run_id, parent_run_id) AS (
    SELECT run_id, parent_run_id FROM runs WHERE run_id = :run
    UNION
    SELECT r.run_id, r.parent_run_id FROM up CROSS JOIN runs r ON r.run_id = up.parent_run_id
),
down(run_id, parent_run_id) AS (
    SELECT run_id, parent_run_id FROM runs WHERE parent_run_id = :run
    UNION
    SELECT r.run_id, r.parent_run_id FROM down CROSS JOIN runs r ON r.parent_run_id = down.run_id
)
SELECT 'ancestor' AS relation, run_id, parent_run_id FROM up WHERE run_id <> :run
UNION ALL
SELECT 'descendant', run_id, parent_run_id FROM down WHERE run_id <> :run;

-- Q6: a run's events in emission order. Index: events_run.
SELECT id, kind, payload
FROM events
WHERE run_id = :run
ORDER BY id;

-- Q7: runs that ended failed, panicked or incomplete since T. Index: events_kind_time.
SELECT run_id, parent_run_id, task, status, finished_ns, err
FROM runs
WHERE status IN ('failed', 'panicked', 'incomplete') AND finished_ns >= :since_ns
ORDER BY finished_ns;
```

`examples/audit` runs exactly this SQL (its `cookbook.sql` is a
byte-identical copy of `store/sqlite/cookbook.sql`) and prints one
tab-separated row per result row:

```bash
go run ./examples/audit run.db Q1 --since 0
go run ./examples/audit run.db Q2 --run <run-id>
go run ./examples/audit run.db Q5 --run <run-id>
go run ./examples/audit run.db Q7 --since 2026-10-26T08:30:00Z
```

`--since` takes an RFC3339 instant, or `0` for the epoch (no lower bound).
Rows are filtered by instant, so events recorded with different UTC
offsets land on the instant they happened, not where their time text
sorts.

## Export to JSONL

The store has no Export method. Export is `Events` plus the agent
package's `Transcript.SaveJSONL` — the same JSONL line format every other
record uses:

```go
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
```

If every string in the run is valid UTF-8, the exported file is byte-equal
to the `agent.JSONL` sink's file for the same run. If a string holds
invalid UTF-8, the two files differ in bytes but decode to the same
events. `Example_exportRecipe` in the package examples is the whole
recipe. Use it for fixtures and for tools that already read JSONL; the
store stays the only record.

## Replay from the store

The store is an `llmkit.Source`, so the replay client consumes it like any
record. `examples/replay` replays a stored run offline — no provider, no
credentials, no network:

```bash
go run ./examples/replay --store run.db --run <run-id>
```

The run id is the one `examples/agent --store` printed. Replay serves the
recorded completions and tool results through `agent.NewReplayClient` and
reports divergence exactly as JSONL replay does; see
[transcripts, observers, and replay](agent-loop.md#transcripts-observers-and-replay).

## Resuming an interrupted run

A resume replays the stored run and continues live at the record's end. A
tool call the record holds without a result may run twice (at-least-once);
a call with a recorded result never runs again. A crashed run resumes from
the store. A JSONL record whose final line is torn is refused until you
remove that line (llmkit-lz5f). The per-run token limit does not
accumulate across a chained resume, a resume of a resume (llmkit-j58a).
The resume recipe is in [the agent loop](agent-loop.md).

## Budgets across restarts

A token ceiling that must survive a restart is a `BudgetPool` pre-charged
from the stored run's usage. Open the store, sum the `Usage` of the run's
Completion events, and `Add` the sum's `ChargeableTokens` to a fresh pool
before the first new run. The store keeps each completion when it happens,
so the sum also covers a process that died before Finalize; the run's
`completion` rows in the `spend` view carry the same usage.
`Example_budgetRestart` in the package examples shows the whole
recipe. A resume does not re-charge the replayed prefix: the stored usage
already covers it. The pool gates one process; a cross-process ceiling is
not provided (bead llmkit-2or.5).

## Limits

- **Payload depth.** SQLite's JSON functions parse at most 1000 nesting
  levels. An event whose encoded payload nests deeper is refused through
  `onErr` (naming run id and kind) and nothing is written. The JSONL sink
  accepts up to encoding/json's 10 000-level decode limit, so a record the
  sink keeps can still be refused here.
- **Schema version.** `Open` on a database a newer version of this package
  wrote fails with `ErrSchemaTooNew` and leaves the file untouched.
  Migrations are forward-only; rolling back means restoring a copy of the
  database made before the upgrade.
- **Client version.** The views use SQLite's `->>` operator, so reading
  them takes SQLite 3.38.0 or newer. SQLite 3.37 fails every query on this
  file with `malformed database schema` (Ubuntu 22.04 ships 3.37, so
  distro clients can be too old).
- **Disk growth.** Message bodies are deduplicated by content hash, but
  each request still stores one hash reference per history message, so the
  stored hash lists grow with the square of the step count: measured 1.71,
  4.76, 14.8 and 51.1 MB at 100, 200, 400 and 800 steps with 2 KiB tool
  results. Dedupe cuts the JSONL equivalent (11.8 MB to 736 MB for the
  same runs), not the square term. Do not plan for linear disk.
- **Foreign files.** `Open` changes a SQLite file it did not create
  (llmkit-rqmm). If the file is at `user_version` 0 and none of its tables
  collides with the store's, `Open` succeeds: it writes the store schema
  into the file (the `events` and `messages` tables, their indexes, the
  views, `user_version` 1) and switches the file to WAL mode. If a table
  collides, `Open` fails, but only after it switched the file to WAL mode.
  Back the file up before you point `Open` at a database of unknown origin.

## Where to go next

The [agent loop](agent-loop.md) page covers the event table and replay;
the [design page](design.md) records why the store is one SQLite file; the
API reference is
[pkg.go.dev/github.com/dpoage/llmkit/store/sqlite](https://pkg.go.dev/github.com/dpoage/llmkit/store/sqlite).

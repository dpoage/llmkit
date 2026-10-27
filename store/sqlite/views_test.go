package sqlite_test

import (
	"database/sql"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// viewRows returns the column names of a view and its rows as sorted text.
func viewRows(t *testing.T, db *sql.DB, view string) ([]string, []string) {
	t.Helper()
	rows, err := db.Query("SELECT * FROM " + view)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	cols, _ := rows.Columns()
	var out []string
	for rows.Next() {
		vals := make([]sql.NullString, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			t.Fatal(err)
		}
		row := make([]string, len(cols))
		for i, v := range vals {
			row[i] = "NULL"
			if v.Valid {
				row[i] = v.String
			}
		}
		out = append(out, strings.Join(row, "|"))
	}
	slices.Sort(out)
	return cols, out
}

// TestViews_ColumnsAndValues pins the documented view columns, which callers
// outside this package query.
func TestViews_ColumnsAndValues(t *testing.T) {
	s, log, path := openStore(t)
	evs := allKinds("run-all")
	observeAll(s, evs)
	noErrors(t, log)
	db := openRaw(t, path)
	ns := func(i int) int64 { return evs[i].Time.UnixNano() }

	// Event order in allKinds: 0 start, 1 completion, 2 failed completion, 3 attempt,
	// 4 tool run, 5 denied tool run, 6 compaction, 7 steer, 8 decision, 9 failed decision,
	// 10 embed, 11 exec, 12 finalize.
	cases := []struct {
		view string
		cols string
		rows []string
	}{
		{"runs",
			"run_id parent_run_id task started_ns finished_ns status truncation_reason steps finalized final_text err input_tokens output_tokens",
			[]string{fmt.Sprintf("run-all|parent-run|do the thing|%d|%d|incomplete|max_iterations|12|1|done|stopped|500|40", ns(0), ns(12))}},
		{"spend",
			"event_id run_id step kind time_ns day provider model purpose input_tokens output_tokens cache_read_tokens cache_creation_tokens reasoning_tokens",
			[]string{
				fmt.Sprintf("2|run-all|1|completion|%d|2026-10-26|anthropic|claude-x|compaction|120|30|50|10|5", ns(1)),
				fmt.Sprintf("3|run-all|2|completion|%d|2026-10-26|p|m|NULL|0|0|0|0|0", ns(2)),
				fmt.Sprintf("9|run-all|8|decision|%d|2026-10-26|systemone|m1|NULL|10|2|0|0|0", ns(8)),
				fmt.Sprintf("10|run-all|9|decision|%d|2026-10-26|systemone|m1|NULL|0|0|0|0|0", ns(9)),
			}},
		{"tool_calls",
			"event_id run_id step time_ns duration_ns call_id name arguments arguments_b64 result is_error denied deny_reason",
			[]string{
				fmt.Sprintf("5|run-all|4|%d|4000000|c1|echo|{\"a\":1}|NULL|[image: image/png, 5 bytes] done|1|0|NULL", ns(4)),
				fmt.Sprintf("6|run-all|5|%d|5000000|c2|rm|{\"path\":\"/\"}|NULL|NULL|0|1|no rm", ns(5)),
			}},
		{"decisions",
			"event_id run_id step time_ns backend model err question_id question_kind belief score choice confidence",
			[]string{
				fmt.Sprintf("9|run-all|8|%d|systemone|m1|NULL|q1|noul|0.2|NULL|NULL|NULL", ns(8)),
				fmt.Sprintf("9|run-all|8|%d|systemone|m1|NULL|q2|choice|NULL|NULL|a|0.4", ns(8)),
				fmt.Sprintf("9|run-all|8|%d|systemone|m1|NULL|q3|score|NULL|0|NULL|NULL", ns(8)),
				fmt.Sprintf("10|run-all|9|%d|systemone|m1|ask failed|NULL|NULL|NULL|NULL|NULL|NULL", ns(9)),
			}},
	}
	for _, c := range cases {
		t.Run(c.view, func(t *testing.T) {
			cols, rows := viewRows(t, db, c.view)
			if got := strings.Join(cols, " "); got != c.cols {
				t.Fatalf("columns\n got: %s\nwant: %s", got, c.cols)
			}
			want := slices.Sorted(slices.Values(c.rows))
			if !slices.Equal(rows, want) {
				t.Fatalf("rows differ\n got: %q\nwant: %q", rows, want)
			}
		})
	}
}

// The runs view holds a row for a run that has started and has not ended.
func TestViews_RunningRunHasNoStatus(t *testing.T) {
	s, _, path := openStore(t)
	s.Observe(t.Context(), startEvent("live"))
	var status, finished sql.NullString
	if err := openRaw(t, path).QueryRow("SELECT status, finished_ns FROM runs WHERE run_id = 'live'").Scan(&status, &finished); err != nil {
		t.Fatal(err)
	}
	if status.Valid || finished.Valid {
		t.Fatalf("running run has status %v, finished %v", status, finished)
	}
}

func TestViews_OneRowPerStartUsesLastFinalize(t *testing.T) {
	s, log, path := openStore(t)
	observeAll(s, allKinds("run-two"))
	last := newEv("run-two", llmkit.KindFinalize)
	last.Finalize = &llmkit.FinalizeEvent{Status: llmkit.RunFailed, Finalized: true, Err: "late failure"}
	s.Observe(t.Context(), last)
	noErrors(t, log)

	db := openRaw(t, path)
	var n int
	if err := db.QueryRow("SELECT count(*) FROM runs WHERE run_id = 'run-two'").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("runs holds %d rows for a run with two Finalize events, want 1", n)
	}
	var status, errTx string
	if err := db.QueryRow("SELECT status, err FROM runs WHERE run_id = 'run-two'").Scan(&status, &errTx); err != nil {
		t.Fatal(err)
	}
	if status != "failed" || errTx != "late failure" {
		t.Fatalf("runs reports status %q err %q, want the last Finalize's %q %q", status, errTx, "failed", "late failure")
	}
}

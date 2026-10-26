package sqlite_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/store/sqlite"
)

// sinceT is the "since T" boundary of the audit dataset.
var sinceT = time.Date(2026, 10, 2, 0, 0, 0, 0, time.UTC)

// instants straddle sinceT with mixed UTC offsets and sub-second precision.
var instants = []struct {
	name string
	t    time.Time
}{
	{"+14:00: text sorts after T, instant before", time.Date(2026, 10, 2, 9, 0, 0, 0, time.FixedZone("", 14*3600))},
	{"1ns before T", sinceT.Add(-time.Nanosecond)},
	{"exactly T", sinceT},
	{"1ns after T", sinceT.Add(time.Nanosecond)},
	{"-12:00: text sorts before T, instant after", time.Date(2026, 10, 1, 23, 59, 59, 999999999, time.FixedZone("", -12*3600))},
}

type spendRec struct {
	t               time.Time
	provider, model string
	in, out         int64
}

type deniedRec struct {
	t         time.Time
	run, args string
	reason    string
}

type decisionRec struct {
	t      time.Time
	run, q string
	kind   string
	score  string
	belief string
}

type finishedRec struct {
	t                          time.Time
	run, parent, status, errTx string
}

// auditData is the stored corpus and what it must answer.
type auditData struct {
	db       *sql.DB
	spend    []spendRec
	denied   []deniedRec
	decided  []decisionRec
	finished []finishedRec
	rootKind []llmkit.EventKind
}

func toolRun(run llmkit.RunID, step int, name, args string, mod func(*llmkit.ToolRunEvent)) llmkit.Event {
	e := newEv(run, llmkit.KindToolRun)
	e.Step = step
	e.ToolRun = &llmkit.ToolRunEvent{Call: llmkit.ToolCall{ID: fmt.Sprintf("%s-%d", name, step), Name: name, Arguments: json.RawMessage(args)}, Result: "r"}
	if mod != nil {
		mod(e.ToolRun)
	}
	return e
}

func auditDataset(t *testing.T) *auditData {
	t.Helper()
	s, log, path := openStore(t)
	d := &auditData{}
	day1 := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	put := func(e llmkit.Event, tm time.Time) {
		e.Time = tm
		if e.RunID == "r-root" {
			d.rootKind = append(d.rootKind, e.Kind)
		}
		s.Observe(context.Background(), e)
	}
	start := func(run, parent llmkit.RunID, tm time.Time) {
		e := newEv(run, llmkit.KindStart)
		e.ParentRunID = parent
		e.Start = &llmkit.StartEvent{Task: "task of " + string(run)}
		put(e, tm)
	}
	finish := func(run, parent llmkit.RunID, tm time.Time, status llmkit.RunStatus, errText string) {
		e := newEv(run, llmkit.KindFinalize)
		e.ParentRunID = parent
		e.Step = 3
		e.Finalize = &llmkit.FinalizeEvent{Status: status, Err: errText, FinalText: "text of " + string(run), Usage: llmkit.Usage{InputTokens: 9, OutputTokens: 4}}
		put(e, tm)
		d.finished = append(d.finished, finishedRec{tm, string(run), string(parent), string(status), errText})
	}
	complete := func(run llmkit.RunID, tm time.Time, provider, model string, in, out int64) {
		e := newEv(run, llmkit.KindCompletion)
		e.Completion = &llmkit.CompletionEvent{
			Request:  llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "q")}},
			Response: llmkit.Response{Text: "a", Usage: llmkit.Usage{InputTokens: in, OutputTokens: out}},
			Provider: provider, Model: model,
		}
		put(e, tm)
		d.spend = append(d.spend, spendRec{tm, provider, model, in, out})
	}
	decide := func(run llmkit.RunID, tm time.Time, answers ...llmkit.DecisionAnswer) {
		e := newEv(run, llmkit.KindDecision)
		dec := &llmkit.DecisionEvent{Backend: "systemone", Model: "dm", Usage: llmkit.Usage{InputTokens: 10, OutputTokens: 1}, Answers: answers}
		for _, a := range answers {
			kind := "score"
			rec := decisionRec{t: tm, run: string(run), q: a.ID, score: "NULL", belief: "NULL"}
			switch {
			case a.Belief != nil:
				kind, rec.belief = "noul", fmt.Sprint(*a.Belief)
			case a.Score != nil:
				rec.score = fmt.Sprint(*a.Score)
			default:
				kind = "choice"
			}
			rec.kind = kind
			dec.Questions = append(dec.Questions, llmkit.DecisionQuestion{ID: a.ID, Kind: kind})
			d.decided = append(d.decided, rec)
		}
		e.Decision = dec
		put(e, tm)
		d.spend = append(d.spend, spendRec{tm, "systemone", "dm", 10, 1})
	}

	// Lineage: root <- child <- grand, root <- sibling; old ended before T;
	// live never finalizes; loop is its own parent.
	start("r-root", "", day1)
	start("r-child", "r-root", day1)
	start("r-grand", "r-child", day1)
	start("r-sibling", "r-root", day1)
	start("r-old", "", day1)
	start("r-live", "", day1)
	start("r-loop", "r-loop", day1)
	finish("r-old", "", day1.Add(time.Hour), llmkit.RunFailed, "old")
	finish("r-root", "", day1.Add(2*time.Hour), llmkit.RunCompleted, "")
	finish("r-child", "r-root", sinceT.Add(time.Hour), llmkit.RunFailed, "boom")
	finish("r-grand", "r-child", sinceT.Add(2*time.Hour), llmkit.RunPanicked, "")
	finish("r-sibling", "r-root", sinceT.Add(3*time.Hour), llmkit.RunIncomplete, "")
	finish("r-loop", "r-loop", sinceT.Add(4*time.Hour), llmkit.RunCompleted, "")

	// r-root's tool calls, in order.
	put(toolRun("r-root", 1, "read", `{"p":"a"}`, nil), day1)
	put(toolRun("r-root", 1, "write", `{"p":"b"}`, func(r *llmkit.ToolRunEvent) { r.IsError = true; r.Result = "ERROR: x" }), day1)
	denyRoot := toolRun("r-root", 2, "rm", `{"p":"/"}`, func(r *llmkit.ToolRunEvent) { r.Denied = true; r.DenyReason = "no rm"; r.Result = "" })
	put(denyRoot, day1)
	d.denied = append(d.denied, deniedRec{day1, "r-root", `{"p":"/"}`, "no rm"})

	// One denied call, one completion and one decision at each instant.
	owners := []llmkit.RunID{"r-child", "r-grand", "r-sibling", "r-root", "r-live"}
	for i, in := range instants {
		args := fmt.Sprintf(`{"i":%d}`, i)
		put(toolRun(owners[i], 9, "rm", args, func(r *llmkit.ToolRunEvent) { r.Denied = true; r.DenyReason = in.name; r.Result = "" }), in.t)
		d.denied = append(d.denied, deniedRec{in.t, string(owners[i]), args, in.name})
		complete(owners[i], in.t, "anthropic", "claude-x", 100, 10)
		decide(owners[i], in.t, llmkit.DecisionAnswer{ID: fmt.Sprintf("q%d", i), Score: f64(0.1)})
	}

	complete("r-root", day1, "openai", "gpt-y", 50, 5)
	complete("r-root", sinceT.Add(5*time.Hour), "openai", "gpt-y", 70, 7)
	complete("r-root", sinceT.Add(30*time.Hour), "anthropic", "claude-x", 11, 1)
	put(toolRun("r-root", 4, "read", `{}`, nil), sinceT.Add(5*time.Hour))
	decide("r-root", sinceT.Add(6*time.Hour),
		llmkit.DecisionAnswer{ID: "hi", Score: f64(0.9)},
		llmkit.DecisionAnswer{ID: "b", Belief: f64(0.3)},
		llmkit.DecisionAnswer{ID: "c", Choice: "a"})
	failed := newEv("r-root", llmkit.KindDecision)
	failed.Decision = &llmkit.DecisionEvent{Backend: "systemone", Err: "ask failed"}
	put(failed, sinceT.Add(7*time.Hour))
	d.spend = append(d.spend, spendRec{sinceT.Add(7 * time.Hour), "systemone", "", 0, 0})

	noErrors(t, log)
	d.db = openRaw(t, path)
	return d
}

func query(t testing.TB, db *sql.DB, name string, args ...sql.NamedArg) [][]string {
	t.Helper()
	all := map[string]any{"run": "", "since_ns": sinceT.UnixNano(), "threshold": 0.5}
	for _, a := range args {
		all[a.Name] = a.Value
	}
	var bound []any
	for k, v := range all {
		bound = append(bound, sql.Named(k, v))
	}
	rows, err := db.Query(sqlite.CookbookQuery(name), bound...)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	defer func() { _ = rows.Close() }()
	cols, _ := rows.Columns()
	var out [][]string
	for rows.Next() {
		vals := make([]sql.NullString, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		row := make([]string, len(cols))
		for i, v := range vals {
			row[i] = "NULL"
			if v.Valid {
				row[i] = v.String
			}
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return out
}

func joined(rows [][]string) []string {
	out := make([]string, len(rows))
	for i, r := range rows {
		out[i] = strings.Join(r, "|")
	}
	return out
}

func requireRows(t testing.TB, name string, got [][]string, want []string, unordered bool) {
	t.Helper()
	g := joined(got)
	if unordered {
		sort.Strings(g)
		sort.Strings(want)
	}
	if !slices.Equal(g, want) {
		t.Fatalf("%s rows differ\n got: %q\nwant: %q", name, g, want)
	}
}

func nullable(s string) string {
	if s == "" {
		return "NULL"
	}
	return s
}

func TestP6_CookbookResults(t *testing.T) {
	d := auditDataset(t)
	db := d.db
	since := func(tm time.Time) bool { return !tm.Before(sinceT) }

	spendRows := func(sinceNS int64) []string {
		type agg struct{ calls, in, out int64 }
		groups := map[string]*agg{}
		for _, r := range d.spend {
			if r.t.UnixNano() < sinceNS {
				continue
			}
			k := strings.Join([]string{r.t.UTC().Format("2006-01-02"), r.provider, nullable(r.model)}, "|")
			if groups[k] == nil {
				groups[k] = &agg{}
			}
			groups[k].calls++
			groups[k].in += r.in
			groups[k].out += r.out
		}
		var rows []string
		for k, a := range groups {
			rows = append(rows, fmt.Sprintf("%s|%d|%d|%d", k, a.calls, a.in, a.out))
		}
		sort.Strings(rows)
		return rows
	}
	for name, sinceNS := range map[string]int64{"since T": sinceT.UnixNano(), "since the beginning": 0} {
		t.Run("Q1 "+name, func(t *testing.T) {
			var got []string
			for _, r := range query(t, db, "Q1", sql.Named("since_ns", sinceNS)) {
				got = append(got, strings.Join(r[:6], "|"))
			}
			if want := spendRows(sinceNS); !slices.Equal(got, want) {
				t.Fatalf("Q1 differs\n got: %q\nwant: %q", got, want)
			}
		})
	}

	t.Run("Q2", func(t *testing.T) {
		requireRows(t, "Q2", query(t, db, "Q2", sql.Named("run", "r-root")), []string{
			`1|read|{"p":"a"}|0|0|NULL`,
			`1|write|{"p":"b"}|1|0|NULL`,
			`2|rm|{"p":"/"}|0|1|no rm`,
			`9|rm|{"i":3}|0|1|1ns after T`,
			`4|read|{}|0|0|NULL`,
		}, false)
		requireRows(t, "Q2 unknown run", query(t, db, "Q2", sql.Named("run", "nope")), nil, false)
	})

	t.Run("Q3", func(t *testing.T) {
		var recs []deniedRec
		for _, r := range d.denied {
			if since(r.t) {
				recs = append(recs, r)
			}
		}
		sort.SliceStable(recs, func(i, j int) bool { return recs[i].t.Before(recs[j].t) })
		var want []string
		for _, r := range recs {
			want = append(want, fmt.Sprintf("%d|%s|9|rm|%s|%s", r.t.UnixNano(), r.run, r.args, r.reason))
		}
		if len(want) != 3 {
			t.Fatalf("dataset error: %d denied calls since T, want 3", len(want))
		}
		requireRows(t, "Q3", query(t, db, "Q3"), want, false)
	})

	t.Run("Q4", func(t *testing.T) {
		var recs []decisionRec
		for _, r := range d.decided {
			if since(r.t) && (r.score != "NULL" && r.score < "0.5" || r.belief != "NULL" && r.belief < "0.5") {
				recs = append(recs, r)
			}
		}
		sort.SliceStable(recs, func(i, j int) bool { return recs[i].t.Before(recs[j].t) })
		var want []string
		for _, r := range recs {
			want = append(want, fmt.Sprintf("%d|%s|%s|%s|%s|%s", r.t.UnixNano(), r.run, r.q, r.kind, r.score, r.belief))
		}
		if len(want) != 4 {
			t.Fatalf("dataset error: %d decisions below the threshold since T, want 4", len(want))
		}
		requireRows(t, "Q4", query(t, db, "Q4"), want, false)
	})

	t.Run("Q5", func(t *testing.T) {
		for run, want := range map[string][]string{
			"r-grand": {"ancestor|r-child|r-root", "ancestor|r-root|NULL"},
			"r-child": {"ancestor|r-root|NULL", "descendant|r-grand|r-child"},
			"r-root":  {"descendant|r-child|r-root", "descendant|r-sibling|r-root", "descendant|r-grand|r-child"},
			"r-live":  nil,
			"r-loop":  nil,
		} {
			requireRows(t, "Q5 "+run, query(t, db, "Q5", sql.Named("run", run)), want, true)
		}
	})

	t.Run("Q6", func(t *testing.T) {
		var got []llmkit.EventKind
		for _, r := range query(t, db, "Q6", sql.Named("run", "r-root")) {
			got = append(got, llmkit.EventKind(r[1]))
		}
		if !slices.Equal(got, d.rootKind) {
			t.Fatalf("Q6 kinds %v, want emission order %v", got, d.rootKind)
		}
	})

	t.Run("Q7", func(t *testing.T) {
		var recs []finishedRec
		for _, r := range d.finished {
			if since(r.t) && slices.Contains([]string{"failed", "panicked", "incomplete"}, r.status) {
				recs = append(recs, r)
			}
		}
		sort.SliceStable(recs, func(i, j int) bool { return recs[i].t.Before(recs[j].t) })
		var want []string
		for _, r := range recs {
			want = append(want, fmt.Sprintf("%s|%s|task of %s|%s|%d|%s", r.run, nullable(r.parent), r.run, r.status, r.t.UnixNano(), nullable(r.errTx)))
		}
		if len(want) != 3 {
			t.Fatalf("dataset error: %d failed runs since T, want 3", len(want))
		}
		requireRows(t, "Q7", query(t, db, "Q7"), want, false)
	})
}

// scanOfEvents matches a plan line that reads a whole table: a SCAN that is
// neither a CTE nor a table-valued function.
var scanOfEvents = regexp.MustCompile(`^SCAN (\w+)( |$)`)

func plan(t testing.TB, db *sql.DB, name string) []string {
	t.Helper()
	rows, err := db.Query("EXPLAIN QUERY PLAN "+sqlite.CookbookQuery(name),
		sql.Named("run", "r"), sql.Named("since_ns", 0), sql.Named("threshold", 0.5))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	defer func() { _ = rows.Close() }()
	var details []string
	for rows.Next() {
		var id, parent, unused int
		var detail string
		if err := rows.Scan(&id, &parent, &unused, &detail); err != nil {
			t.Fatal(err)
		}
		details = append(details, detail)
	}
	return details
}

func TestP6_QueryPlansUseTheirIndexes(t *testing.T) {
	_, _, path := openStore(t)
	db := openRaw(t, path)
	want := map[string][]string{
		"Q1": {"events_kind_time"},
		"Q2": {"events_run_kind"},
		"Q3": {"events_kind_time"},
		"Q4": {"events_kind_time"},
		"Q5": {"events_run_kind", "events_parent"},
		"Q6": {"events_run"},
		"Q7": {"events_kind_time", "events_run_kind"},
	}
	for _, name := range []string{"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"} {
		t.Run(name, func(t *testing.T) {
			details := plan(t, db, name)
			text := strings.Join(details, "\n")
			for _, index := range want[name] {
				if !regexp.MustCompile(`SEARCH \w+ USING (COVERING )?INDEX ` + index + `\b`).MatchString(text) {
					t.Errorf("plan does not search %s:\n%s", index, text)
				}
			}
			for _, line := range details {
				m := scanOfEvents.FindStringSubmatch(line)
				if m == nil || strings.Contains(line, "VIRTUAL TABLE") || m[1] == "up" || m[1] == "down" {
					continue
				}
				t.Errorf("plan scans a table: %q\n%s", line, text)
			}
		})
	}
}

func TestP6_CookbookIsOneSelectEach(t *testing.T) {
	for _, name := range []string{"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"} {
		q := sqlite.CookbookQuery(name)
		body := regexp.MustCompile(`(?m)^--.*$`).ReplaceAllString(q, "")
		if n := strings.Count(body, ";"); n != 1 || !strings.HasSuffix(strings.TrimSpace(body), ";") {
			t.Errorf("%s holds %d statements, want exactly one", name, n)
		}
		if first := strings.Fields(body)[0]; first != "SELECT" && first != "WITH" {
			t.Errorf("%s starts with %s, want SELECT", name, first)
		}
	}
}

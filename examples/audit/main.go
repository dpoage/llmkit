// Command audit answers the cookbook queries of docs/store.md against a
// store/sqlite database: spend per model, a run's tool calls, denied calls,
// low decisions, lineage, the raw event stream, and runs by end status. It
// reads the store's documented SQL views through database/sql and never
// writes. The queries are the cookbook verbatim (examples/audit/cookbook.sql
// is a byte-identical copy of store/sqlite/cookbook.sql).
//
// Usage:
//
//	go run ./examples/audit <db> Q1 --since 0
//	go run ./examples/audit <db> Q2 --run <run-id>
//
// <query> is Q1..Q7 of the cookbook (docs/store.md):
//
//	Q1 spend per provider, model and day    --since
//	Q2 a run's tool calls in order          --run
//	Q3 denied tool calls                    --since
//	Q4 decisions below a threshold          --since --threshold
//	Q5 lineage of a run                     --run
//	Q6 a run's events in emission order     --run
//	Q7 runs failed, panicked, incomplete    --since
//
// --since is an RFC3339 instant or 0 for the epoch (no lower bound); rows
// print tab-separated. A query error prints the SQLite error and exits 1.
package main

import (
	"bufio"
	"database/sql"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

//go:embed cookbook.sql
var cookbookSQL string

var cookbookHeader = regexp.MustCompile(`(?m)^-- (Q\d+):.*$`)

func cookbookQuery(name string) (string, error) {
	locs := cookbookHeader.FindAllStringSubmatchIndex(cookbookSQL, -1)
	for i, loc := range locs {
		if cookbookSQL[loc[2]:loc[3]] != name {
			continue
		}
		end := len(cookbookSQL)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		return strings.TrimSpace(cookbookSQL[loc[0]:end]), nil
	}
	return "", fmt.Errorf("unknown query %s (want Q1..Q7)", name)
}

// needs names the flags each query cannot run without.
var needs = map[string]string{
	"Q2": "run", "Q4": "threshold", "Q5": "run", "Q6": "run",
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "audit:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	var since, runID, thresholdRaw string
	flags := flag.NewFlagSet("audit", flag.ContinueOnError)
	flags.StringVar(&since, "since", "0", "RFC3339 instant; 0 means the epoch (no lower bound)")
	flags.StringVar(&runID, "run", "", "run id (required by Q2, Q5 and Q6)")
	flags.StringVar(&thresholdRaw, "threshold", "", "threshold (required by Q4)")
	// Flags may follow the positional arguments (`audit <db> Q1 --since 0`),
	// so consume one positional per pass and re-parse the rest.
	var positional []string
	for {
		if err := flags.Parse(args); err != nil {
			return err
		}
		rest := flags.Args()
		if len(rest) == 0 {
			break
		}
		positional = append(positional, rest[0])
		args = rest[1:]
	}
	args = positional
	if len(args) != 2 {
		return errors.New(usage)
	}
	dbPath, name := args[0], args[1]

	query, err := cookbookQuery(name)
	if err != nil {
		return err
	}
	if need, ok := needs[name]; ok {
		missing := map[string]bool{"run": runID == "", "threshold": thresholdRaw == ""}[need]
		if missing {
			return fmt.Errorf("%s needs --%s", name, need)
		}
	}

	sinceNS := int64(0)
	if since != "" && since != "0" {
		t, err := time.Parse(time.RFC3339, since)
		if err != nil {
			return fmt.Errorf("--since: %w", err)
		}
		sinceNS = t.UnixNano()
	}
	var threshold float64
	if thresholdRaw != "" {
		if threshold, err = strconv.ParseFloat(thresholdRaw, 64); err != nil {
			return fmt.Errorf("--threshold: %w", err)
		}
	}

	// A reader must not create the database a mistyped path names.
	if _, err := os.Stat(dbPath); err != nil {
		return fmt.Errorf("store %s: %w", dbPath, err)
	}

	db, err := sql.Open("sqlite", dsn(dbPath))
	if err != nil {
		return err
	}
	defer func() { _ = db.Close() }()
	rows, err := db.Query(query,
		sql.Named("run", runID),
		sql.Named("since_ns", sinceNS),
		sql.Named("threshold", threshold),
	)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	defer func() { _ = rows.Close() }()

	cols, err := rows.Columns()
	if err != nil {
		return err
	}
	vals := make([]sql.NullString, len(cols))
	ptrs := make([]any, len(cols))
	for i := range vals {
		ptrs[i] = &vals[i]
	}
	w := bufio.NewWriter(os.Stdout)
	defer func() { _ = w.Flush() }()
	for rows.Next() {
		if err := rows.Scan(ptrs...); err != nil {
			return err
		}
		fields := make([]string, len(vals))
		for i, v := range vals {
			if v.Valid {
				fields[i] = v.String
			}
		}
		if _, err := fmt.Fprintln(w, strings.Join(fields, "\t")); err != nil {
			return err
		}
	}
	return rows.Err()
}

// dsn names exactly the path, read-only (SQLite never creates or writes the
// file), with the store's busy_timeout discipline.
func dsn(path string) string {
	esc := strings.NewReplacer("%", "%25", "?", "%3f", "#", "%23").Replace(path)
	for strings.HasPrefix(esc, "/") {
		esc = "%2F" + esc[1:]
	}
	q := url.Values{"mode": {"ro"}, "_pragma": {"busy_timeout(10000)"}}
	return "file:" + esc + "?" + q.Encode()
}

const usage = `usage: go run ./examples/audit <db> <query> [--since when] [--run id] [--threshold n]

<query> is Q1..Q7 of the cookbook (docs/store.md); run against a database
examples/agent --store wrote`

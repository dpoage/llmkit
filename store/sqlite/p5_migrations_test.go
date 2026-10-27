package sqlite_test

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/dpoage/llmkit/store/sqlite"
)

func embeddedMigrationCount(t testing.TB) int {
	t.Helper()
	files, err := filepath.Glob("migrations/*.sql")
	if err != nil || len(files) == 0 {
		t.Fatalf("no embedded migrations: %v", err)
	}
	return len(files)
}

func userVersion(t testing.TB, db *sql.DB) int {
	t.Helper()
	var v int
	if err := db.QueryRow("PRAGMA user_version").Scan(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

func schemaText(t testing.TB, db *sql.DB) string {
	t.Helper()
	rows, err := db.Query("SELECT sql FROM sqlite_schema ORDER BY type, name")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	var b strings.Builder
	for rows.Next() {
		var s sql.NullString
		if err := rows.Scan(&s); err != nil {
			t.Fatal(err)
		}
		b.WriteString(s.String + "\n")
	}
	return b.String()
}

func TestP5_FreshOpenLeavesLatestVersionAndReOpenIsANoOp(t *testing.T) {
	s, log, path := openStore(t)
	db := openRaw(t, path)
	if got, want := userVersion(t, db), embeddedMigrationCount(t); got != want {
		t.Fatalf("user_version = %d after a fresh Open, want %d", got, want)
	}
	before := schemaText(t, db)
	for _, table := range []string{"events", "messages"} {
		if !strings.Contains(before, "CREATE TABLE "+table) {
			t.Errorf("schema lacks table %s", table)
		}
	}
	s.Observe(context.Background(), startEvent("run-reopen"))
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	s2, err := sqlite.Open(path, log.add)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	defer func() { _ = s2.Close() }()
	if after := schemaText(t, db); after != before {
		t.Fatalf("schema changed on re-Open\nbefore: %s\nafter: %s", before, after)
	}
	if _, err := s2.Events(context.Background(), "run-reopen"); err != nil {
		t.Fatalf("data lost on re-Open: %v", err)
	}
	noErrors(t, log)
}

func migrationFS(files map[string]string) fstest.MapFS {
	fsys := fstest.MapFS{}
	for name, body := range files {
		fsys[name] = &fstest.MapFile{Data: []byte(body)}
	}
	return fsys
}

func TestP5_EveryMigrationAppliedOnce(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.db")
	init, err := os.ReadFile("migrations/0001_init.sql")
	if err != nil {
		t.Fatal(err)
	}
	v1 := map[string]string{
		"0001_log.sql": string(init) + "\nCREATE TABLE applied (n INTEGER); INSERT INTO applied VALUES (1);",
	}
	v3 := map[string]string{
		"0001_log.sql":   v1["0001_log.sql"],
		"0002_more.sql":  "INSERT INTO applied VALUES (2);",
		"0003_third.sql": "INSERT INTO applied VALUES (3);",
	}
	applied := func() string {
		var got []string
		rows, err := openRaw(t, path).Query("SELECT n FROM applied ORDER BY rowid")
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var n string
			_ = rows.Scan(&n)
			got = append(got, n)
		}
		return strings.Join(got, ",")
	}
	for _, step := range []struct {
		fs          fstest.MapFS
		wantVersion int
		wantApplied string
	}{
		{migrationFS(v1), 1, "1"},
		{migrationFS(v1), 1, "1"},
		{migrationFS(v3), 3, "1,2,3"},
		{migrationFS(v3), 3, "1,2,3"},
	} {
		s, err := sqlite.OpenFS(path, nil, step.fs)
		if err != nil {
			t.Fatal(err)
		}
		_ = s.Close()
		if got := userVersion(t, openRaw(t, path)); got != step.wantVersion {
			t.Fatalf("user_version = %d, want %d", got, step.wantVersion)
		}
		if got := applied(); got != step.wantApplied {
			t.Fatalf("applied migrations = %q, want %q", got, step.wantApplied)
		}
	}
}

func TestP5_NewerSchemaFailsAndLeavesFileByteIdentical(t *testing.T) {
	path := filepath.Join(t.TempDir(), "newer.db")
	raw, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	newer := embeddedMigrationCount(t) + 7
	for _, stmt := range []string{"CREATE TABLE keep (x)", "INSERT INTO keep VALUES (1)", "PRAGMA user_version = " + strconv.Itoa(newer)} {
		if _, err := raw.Exec(stmt); err != nil {
			t.Fatal(err)
		}
	}
	var mode string
	if err := raw.QueryRow("PRAGMA journal_mode").Scan(&mode); err != nil || mode == "wal" {
		t.Fatalf("witness must be a rollback-journal file, journal_mode = %q (%v)", mode, err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}
	before := fileDigest(t, path)

	_, err = sqlite.Open(path, nil)
	if !errors.Is(err, sqlite.ErrSchemaTooNew) {
		t.Fatalf("Open of a version-%d database: %v, want ErrSchemaTooNew", newer, err)
	}
	if after := fileDigest(t, path); after != before {
		t.Fatalf("Open changed a newer database file")
	}
	entries, _ := os.ReadDir(filepath.Dir(path))
	if len(entries) != 1 {
		t.Fatalf("Open left %d files beside the database, want none", len(entries)-1)
	}
}

func fileDigest(t testing.TB, path string) [32]byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sha256.Sum256(b)
}

// openQuiet converts an Open panic into an error, so the refusal tests below
// report it and still run their file and lock probes.
func openQuiet(path string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("Open panicked: %v", r)
		}
	}()
	s, oerr := sqlite.Open(path, nil)
	if oerr == nil {
		_ = s.Close()
	}
	return oerr
}

// beginImmediateSucceeds proves a refused Open left no transaction or
// connection behind: another connection can take the write lock at once.
func beginImmediateSucceeds(t testing.TB, path string) {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=busy_timeout(1000)")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = db.Close() }()
	if _, err := db.Exec("BEGIN IMMEDIATE"); err != nil {
		t.Fatalf("a second connection cannot take the write lock after Open refused: %v", err)
	}
	if _, err := db.Exec("ROLLBACK"); err != nil {
		t.Fatal(err)
	}
}

func noSidecarFiles(t testing.TB, path string) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("Open left %d files beside the database, want none", len(entries)-1)
	}
}

func TestP5_PendingMigrationsRunInOneTransaction(t *testing.T) {
	cases := map[string]map[string]string{
		"third statement of one file fails": {
			"0001_init.sql": "CREATE TABLE a (x); CREATE TABLE b (y); CREATE TABLE (broken;",
		},
		"second file fails after the first applied": {
			"0001_init.sql": "CREATE TABLE a (x);",
			"0002_next.sql": "CREATE TABLE b (y); INSERT INTO missing VALUES (1);",
		},
	}
	for name, files := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tx.db")
			if s, err := sqlite.OpenFS(path, nil, migrationFS(files)); err == nil {
				_ = s.Close()
				t.Fatal("Open with a failing migration succeeded")
			}
			db := openRaw(t, path)
			if got := schemaText(t, db); strings.TrimSpace(got) != "" {
				t.Fatalf("failed migrations left schema behind: %s", got)
			}
			if got := userVersion(t, db); got != 0 {
				t.Fatalf("user_version = %d after a failed migration, want 0", got)
			}
		})
	}
}

func TestP5_MigrationFilesMustBeContiguous(t *testing.T) {
	for name, files := range map[string]map[string]string{
		"gap":         {"0001_a.sql": "SELECT 1;", "0003_c.sql": "SELECT 1;"},
		"bad name":    {"init.sql": "SELECT 1;"},
		"none":        {},
		"starts at 2": {"0002_b.sql": "SELECT 1;"},
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "x.db")
			if s, err := sqlite.OpenFS(path, nil, migrationFS(files)); err == nil {
				_ = s.Close()
				t.Fatal("Open accepted a malformed migration set")
			}
			if _, err := os.Stat(path); err == nil {
				t.Fatal("Open created the database before validating the migrations")
			}
		})
	}
}

func TestP5_NegativeUserVersionRefusedBeforeAnyWrite(t *testing.T) {
	for _, tc := range []struct {
		version int
		wal     bool
	}{
		{-1, false},
		{-1, true},
		{-2147483648, false},
		{-2147483648, true},
	} {
		t.Run(fmt.Sprintf("v%d/wal=%v", tc.version, tc.wal), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "neg.db")
			raw := openRaw(t, path)
			if tc.wal {
				var mode string
				if err := raw.QueryRow("PRAGMA journal_mode = WAL").Scan(&mode); err != nil || !strings.EqualFold(mode, "wal") {
					t.Fatalf("witness must be a WAL file, journal_mode = %q (%v)", mode, err)
				}
			}
			if _, err := raw.Exec("PRAGMA user_version = " + strconv.Itoa(tc.version)); err != nil {
				t.Fatal(err)
			}
			if err := raw.Close(); err != nil {
				t.Fatal(err)
			}
			before := fileDigest(t, path)

			err := openQuiet(path)
			if err == nil {
				t.Fatal("Open of a negative-version database succeeded")
			}
			if got := err.Error(); !strings.Contains(got, strconv.Itoa(tc.version)) {
				t.Errorf("error does not name the version %d: %s", tc.version, got)
			}
			if errors.Is(err, sqlite.ErrSchemaTooNew) {
				t.Errorf("negative version reported as too new: %s", err)
			}
			if after := fileDigest(t, path); after != before {
				t.Error("Open modified a refused database")
			}
			beginImmediateSucceeds(t, path)
			noSidecarFiles(t, path)
		})
	}
}

func TestP5_DoubleSlashAbsolutePathOpensTheNamedFile(t *testing.T) {
	dir := t.TempDir()
	named := "/" + filepath.Join(dir, "dbl.db")
	s, err := sqlite.Open(named, nil)
	if err != nil {
		t.Fatalf("Open %s: %v", named, err)
	}
	s.Observe(context.Background(), startEvent("run-dbl"))
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if len(names) != 1 || names[0] != "dbl.db" {
		t.Fatalf("Open created %v, want only dbl.db", names)
	}
	s2, err := sqlite.Open(named, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s2.Close() }()
	got, err := s2.Events(context.Background(), "run-dbl")
	if err != nil || len(got) != 1 {
		t.Fatalf("Events after re-Open: %d events, err %v", len(got), err)
	}
}

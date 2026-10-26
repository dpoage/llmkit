package sqlite

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"regexp"
	"sort"
	"strconv"
)

//go:embed migrations/*.sql
var migrationFiles embed.FS

// beginWrite takes the write lock up front, so a concurrent writer waits on
// busy_timeout instead of failing a read-to-write upgrade.
const beginWrite = "BEGIN IMMEDIATE"

// ErrSchemaTooNew is returned by [Open] for a database that a newer version
// of this package wrote. The file is left untouched.
var ErrSchemaTooNew = errors.New("sqlite: database schema is newer than this version supports")

type migration struct {
	version int
	file    string
	sql     string
}

var migrationName = regexp.MustCompile(`^(\d{4})_[a-z0-9_]+\.sql$`)

// loadMigrations reads NNNN_name.sql files from the root of fsys. The
// versions must run 1..N with no gap or repeat.
func loadMigrations(fsys fs.FS) ([]migration, error) {
	names, err := fs.Glob(fsys, "*.sql")
	if err != nil {
		return nil, err
	}
	var ms []migration
	for _, name := range names {
		m := migrationName.FindStringSubmatch(name)
		if m == nil {
			return nil, fmt.Errorf("sqlite: migration file %q is not named NNNN_name.sql", name)
		}
		v, _ := strconv.Atoi(m[1])
		body, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, err
		}
		ms = append(ms, migration{version: v, file: name, sql: string(body)})
	}
	sort.Slice(ms, func(i, j int) bool { return ms[i].version < ms[j].version })
	for i, m := range ms {
		if m.version != i+1 {
			return nil, fmt.Errorf("sqlite: migration %q: want version %d (versions run 1..N without gaps or repeats)", m.file, i+1)
		}
	}
	if len(ms) == 0 {
		return nil, errors.New("sqlite: no migrations")
	}
	return ms, nil
}

func userVersion(ctx context.Context, c *sql.Conn) (int, error) {
	var v int
	err := c.QueryRowContext(ctx, "PRAGMA user_version").Scan(&v)
	return v, err
}

// migrate applies every migration above the file's user_version in one
// transaction, re-reading the version once it holds the write lock so a
// concurrent opener's work is not repeated.
func migrate(ctx context.Context, c *sql.Conn, ms []migration) (err error) {
	latest := ms[len(ms)-1].version
	if _, err = c.ExecContext(ctx, beginWrite); err != nil {
		return fmt.Errorf("sqlite: begin migration: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_, _ = c.ExecContext(context.WithoutCancel(ctx), "ROLLBACK")
		}
	}()
	cur, err := userVersion(ctx, c)
	if err != nil {
		return fmt.Errorf("sqlite: read schema version: %w", err)
	}
	if cur > latest {
		return tooNew(cur, latest)
	}
	if cur < 0 {
		return negativeVersion(cur)
	}
	if cur == latest {
		_, err = c.ExecContext(ctx, "COMMIT")
		committed = err == nil
		return err
	}
	for _, m := range ms[cur:] {
		if _, err = c.ExecContext(ctx, m.sql); err != nil {
			return fmt.Errorf("sqlite: migration %s: %w", m.file, err)
		}
	}
	if _, err = c.ExecContext(ctx, "PRAGMA user_version = "+strconv.Itoa(latest)); err != nil {
		return fmt.Errorf("sqlite: set schema version: %w", err)
	}
	if _, err = c.ExecContext(ctx, "COMMIT"); err != nil {
		return fmt.Errorf("sqlite: commit migration: %w", err)
	}
	committed = true
	return nil
}

func tooNew(have, latest int) error {
	return fmt.Errorf("%w (file version %d, this build supports %d)", ErrSchemaTooNew, have, latest)
}

// negativeVersion names a user_version no build of this package wrote: it is
// a signed 32-bit integer, this package only ever sets 0..latest, and
// ms[cur:] below cur=0 panics.
func negativeVersion(v int) error {
	return fmt.Errorf("sqlite: database schema version %d is negative", v)
}

var errForeignSchema = errors.New("sqlite: database at user_version 0 already holds tables or views; it was not created by this package")

// schemaEmpty reports whether the database holds no user objects; SQLite's
// own (sqlite_sequence, autoindexes) are prefixed sqlite_.
func schemaEmpty(ctx context.Context, c *sql.Conn) (bool, error) {
	var empty bool
	err := c.QueryRowContext(ctx,
		"SELECT NOT EXISTS (SELECT 1 FROM sqlite_schema WHERE substr(name, 1, 7) <> 'sqlite_')").Scan(&empty)
	return empty, err
}

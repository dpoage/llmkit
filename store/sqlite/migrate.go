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
	defer func() {
		if err != nil {
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
	if cur == latest {
		_, err = c.ExecContext(ctx, "COMMIT")
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
	return nil
}

func tooNew(have, latest int) error {
	return fmt.Errorf("%w (file version %d, this build supports %d)", ErrSchemaTooNew, have, latest)
}

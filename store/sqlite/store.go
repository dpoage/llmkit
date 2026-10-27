// Package sqlite is a durable llmkit event log in one SQLite file. A [Store]
// is both an [llmkit.Observer] (register it with the agent Runner) and an
// [llmkit.Source] (replay from it), and its read-only SQL views (runs, spend,
// tool_calls, decisions) serve audit queries from any SQLite client.
//
// Observe is synchronous: when it returns the event is committed and visible
// to Events and to other connections. Observe never fails its caller; an event
// the store cannot keep is reported through the onErr passed to [Open].
// Several processes may open one file.
package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/dpoage/llmkit"
	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// busyTimeoutMS bounds how long a writer waits on another connection's write lock.
const busyTimeoutMS = 10000

const (
	insertMessage = "INSERT OR IGNORE INTO messages (hash, body) VALUES (?, ?)"
	insertEvent   = "INSERT INTO events (run_id, parent_run_id, span_id, kind, step, time_ns, duration_ns, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
	// fetchMessages takes a JSON array of hashes.
	fetchMessages = "SELECT hash, body FROM messages WHERE hash IN (SELECT value FROM json_each(?))"
)

var queryEvents = cookbookQuery("Q6")

var (
	_ llmkit.Observer = (*Store)(nil)
	_ llmkit.Source   = (*Store)(nil)
)

// Store is a SQLite-backed event log. It is safe for concurrent use.
type Store struct {
	db    *sql.DB
	onErr func(error)

	mu     sync.Mutex // serializes writes; guards closed and the write connection
	w      *sql.Conn
	insMsg *sql.Stmt
	insEv  *sql.Stmt
	closed bool
}

// Open opens the database at path, creating it and applying every pending
// schema migration. path is a file path; the directory must exist.
//
// onErr receives one error for each event the store does not keep (see
// [Store.Observe]) and may be nil. It is never called while the store holds a
// lock, so it may call back into the store.
//
// Open fails with an error wrapping [ErrSchemaTooNew], leaving the file
// untouched, when a newer version of this package wrote it. A negative schema
// version is refused with an error before any write as well; it does not
// wrap ErrSchemaTooNew.
func Open(path string, onErr func(error)) (*Store, error) {
	return openFS(path, onErr, migrationsFS())
}

func migrationsFS() fs.FS {
	sub, err := fs.Sub(migrationFiles, "migrations")
	if err != nil {
		panic(err)
	}
	return sub
}

// openFS is Open with the migrations supplied; tests substitute their own.
func openFS(path string, onErr func(error), migrations fs.FS) (*Store, error) {
	ms, err := loadMigrations(migrations)
	if err != nil {
		return nil, err
	}
	if onErr == nil {
		onErr = func(error) {}
	}
	db, err := sql.Open("sqlite", dsn(path))
	if err != nil {
		return nil, fmt.Errorf("sqlite: open %s: %w", path, err)
	}
	ctx := context.Background()
	w, err := db.Conn(ctx)
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("sqlite: open %s: %w", path, err)
	}
	s := &Store{db: db, onErr: onErr, w: w}
	if err := s.init(ctx, ms); err != nil {
		_ = s.close()
		return nil, fmt.Errorf("sqlite: open %s: %w", path, err)
	}
	return s, nil
}

// init checks the schema version before it runs any statement that writes
// the file, then configures the write connection and migrates.
func (s *Store) init(ctx context.Context, ms []migration) error {
	latest := ms[len(ms)-1].version
	v, err := userVersion(ctx, s.w)
	if err != nil {
		return fmt.Errorf("read schema version: %w", err)
	}
	if v > latest {
		return tooNew(v, latest)
	}
	if v < 0 {
		return negativeVersion(v)
	}
	if err := enterWAL(ctx, s.w); err != nil {
		return fmt.Errorf("enter WAL mode: %w", err)
	}
	if _, err := s.w.ExecContext(ctx, "PRAGMA synchronous = NORMAL"); err != nil {
		return err
	}
	if v < latest {
		if err := migrate(ctx, s.w, ms); err != nil {
			return err
		}
	}
	if s.insMsg, err = s.w.PrepareContext(ctx, insertMessage); err != nil {
		return err
	}
	s.insEv, err = s.w.PrepareContext(ctx, insertEvent)
	return err
}

// dsn names exactly path: a literal // after file: would read as a URI
// authority, so leading slashes are percent-encoded.
func dsn(path string) string {
	esc := strings.NewReplacer("%", "%25", "?", "%3f", "#", "%23").Replace(path)
	for strings.HasPrefix(esc, "/") {
		esc = "%2F" + esc[1:]
	}
	q := url.Values{"_pragma": {fmt.Sprintf("busy_timeout(%d)", busyTimeoutMS)}}
	return "file:" + esc + "?" + q.Encode()
}

// enterWAL switches the file to WAL unless it already is. SQLite does not run
// the busy handler for the exclusive lock this needs, so a concurrent opener
// gets SQLITE_BUSY at once; retry until busy_timeout's worth of time has
// passed, and take the mode another opener set meanwhile.
func enterWAL(ctx context.Context, c *sql.Conn) error {
	deadline := time.Now().Add(busyTimeoutMS * time.Millisecond)
	for {
		var mode string
		err := c.QueryRowContext(ctx, "PRAGMA journal_mode").Scan(&mode)
		if err == nil && strings.EqualFold(mode, "wal") {
			return nil
		}
		if err == nil {
			err = c.QueryRowContext(ctx, "PRAGMA journal_mode = WAL").Scan(&mode)
			if err == nil && !strings.EqualFold(mode, "wal") {
				return fmt.Errorf("journal_mode is %q", mode)
			}
		}
		if err == nil {
			return nil
		}
		if !isBusy(err) || time.Now().After(deadline) {
			return err
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// isBusy reports SQLITE_BUSY and SQLITE_LOCKED, including their extended codes.
func isBusy(err error) bool {
	var coded interface{ Code() int }
	if !errors.As(err, &coded) {
		return false
	}
	switch coded.Code() & 0xff {
	case 5, 6:
		return true
	}
	return false
}

// Observe stores ev, implementing [llmkit.Observer]. It returns once the
// event is committed. Every event is stored, whatever its run id or position
// in the run, except these, each of which makes one onErr call naming the run
// id and kind and writes nothing: the store is closed; encoding/json rejects
// the event; the encoded payload nests deeper than 1000 levels, the most
// SQLite's JSON functions parse; Events would not decode what would be
// written; the write fails. ctx cancelation does not stop the write.
func (s *Store) Observe(_ context.Context, ev llmkit.Event) {
	if err := s.observe(ev); err != nil {
		s.onErr(fmt.Errorf("sqlite: %s event of run %q not stored: %w", ev.Kind, ev.RunID, err))
	}
}

var errClosed = errors.New("store is closed")

func (s *Store) observe(ev llmkit.Event) error {
	enc, err := encode(ev)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errClosed
	}
	ctx := context.Background()
	if _, err := s.w.ExecContext(ctx, beginWrite); err != nil {
		return err
	}
	if err := s.write(ctx, ev, enc); err != nil {
		_, _ = s.w.ExecContext(ctx, "ROLLBACK")
		return err
	}
	return nil
}

func (s *Store) write(ctx context.Context, ev llmkit.Event, enc encoded) error {
	for _, m := range enc.messages {
		if _, err := s.insMsg.ExecContext(ctx, m.hash, string(m.body)); err != nil {
			return err
		}
	}
	if _, err := s.insEv.ExecContext(ctx,
		nullable(ev.RunID), nullable(ev.ParentRunID), nullable(ev.SpanID), string(ev.Kind),
		ev.Step, unixNanos(ev.Time), int64(ev.Duration), string(enc.payload)); err != nil {
		return err
	}
	_, err := s.w.ExecContext(ctx, "COMMIT")
	return err
}

// Events returns the events of run in emission order, implementing
// [llmkit.Source]; each is json-identical to the event [Store.Observe]
// received. It fails with an error wrapping [llmkit.ErrUnknownRun] for an
// empty id, a run with no events, and a run holding more than one Start event.
func (s *Store) Events(ctx context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	if run == "" {
		return nil, fmt.Errorf("sqlite: empty run id: %w", llmkit.ErrUnknownRun)
	}
	stored, hashes, err := s.load(ctx, run)
	if err != nil {
		return nil, err
	}
	if len(stored) == 0 {
		return nil, fmt.Errorf("sqlite: run %q: %w", run, llmkit.ErrUnknownRun)
	}
	bodies, err := s.bodies(ctx, hashes)
	if err != nil {
		return nil, err
	}
	evs := make([]llmkit.Event, len(stored))
	for i, se := range stored {
		if evs[i], err = se.assemble(bodies); err != nil {
			return nil, fmt.Errorf("sqlite: run %q event %d: %w", run, i, err)
		}
	}
	return evs, nil
}

// load reads and parses the run's payloads in order, and returns the distinct
// message hashes they name.
func (s *Store) load(ctx context.Context, run llmkit.RunID) ([]storedEvent, []string, error) {
	rows, err := s.db.QueryContext(ctx, queryEvents, sql.Named("run", string(run)))
	if err != nil {
		return nil, nil, fmt.Errorf("sqlite: read run %q: %w", run, err)
	}
	defer func() { _ = rows.Close() }()
	var (
		stored []storedEvent
		seen   = map[string]bool{}
		hashes []string
		starts int
	)
	for rows.Next() {
		var (
			id      int64
			kind    string
			payload []byte
		)
		if err := rows.Scan(&id, &kind, &payload); err != nil {
			return nil, nil, fmt.Errorf("sqlite: read run %q: %w", run, err)
		}
		se, err := decode(payload)
		if err != nil {
			return nil, nil, fmt.Errorf("sqlite: run %q event %d: %w", run, id, err)
		}
		if starts += btoi(kind == string(llmkit.KindStart)); starts > 1 {
			return nil, nil, fmt.Errorf("sqlite: run %q holds more than one Start event: %w", run, llmkit.ErrUnknownRun)
		}
		for _, h := range se.hashes() {
			if !seen[h] {
				seen[h] = true
				hashes = append(hashes, h)
			}
		}
		stored = append(stored, se)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("sqlite: read run %q: %w", run, err)
	}
	return stored, hashes, nil
}

func (s *Store) bodies(ctx context.Context, hashes []string) (map[string][]byte, error) {
	bodies := make(map[string][]byte, len(hashes))
	if len(hashes) == 0 {
		return bodies, nil
	}
	arr, err := json.Marshal(hashes)
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, fetchMessages, string(arr))
	if err != nil {
		return nil, fmt.Errorf("sqlite: read messages: %w", err)
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var (
			hash string
			body []byte
		)
		if err := rows.Scan(&hash, &body); err != nil {
			return nil, fmt.Errorf("sqlite: read messages: %w", err)
		}
		bodies[hash] = body
	}
	return bodies, rows.Err()
}

// Close releases the database. It is idempotent. Observe after Close reports
// through onErr; Events after Close returns an error.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	return s.close()
}

func (s *Store) close() error {
	s.closed = true
	if s.insMsg != nil {
		_ = s.insMsg.Close()
	}
	if s.insEv != nil {
		_ = s.insEv.Close()
	}
	return errors.Join(s.w.Close(), s.db.Close())
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

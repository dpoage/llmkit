package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
)

// Transcript is the ordered record of a single [Runner.Run]: the [llmkit]
// events the Runner emitted, in emission order. It is always present on
// [Outcome.Transcript] and is the mandatory first link of the Runner's
// observer chain — a view of the run's event stream, never a second store:
// the same events a durable sink set with [WithObserver] receives.
//
// A Transcript is not safe for concurrent mutation, but a single run appends
// to it sequentially. It satisfies [llmkit.Observer] (it never fails and
// never blocks the run) and [Source] (replay reads it back).
type Transcript struct {
	// Record is the ordered event list.
	Record []llmkit.Event
	// RunID is the id of the recorded run — minted by the Runner, or the
	// caller's id via [WithRunID]. Empty on a hand-built transcript.
	RunID llmkit.RunID
	// ParentRunID is the id of the run this one continued from ([Continue]);
	// empty otherwise.
	ParentRunID llmkit.RunID
}

// NewTranscript returns an empty transcript.
func NewTranscript() *Transcript { return &Transcript{} }

// Observe appends ev to the record. It satisfies [llmkit.Observer]; the
// Runner places the transcript first in its observer chain (see
// [llmkit.Observers]).
func (t *Transcript) Observe(_ context.Context, ev llmkit.Event) {
	t.Record = append(t.Record, ev)
}

// Events returns a COPY of the recorded events of run, in emission order —
// the caller owns the returned slice. It satisfies [Source]. A transcript
// records exactly one run: any other id (including an empty one, unless the
// transcript itself is unnamed) fails with an error wrapping
// [ErrUnknownRun].
func (t *Transcript) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	if t.RunID == "" || run != t.RunID {
		// An unnamed (hand-built) transcript records no run at all.
		return nil, fmt.Errorf("agent: transcript records run %q, not %q: %w", t.RunID, run, ErrUnknownRun)
	}
	return slices.Clone(t.Record), nil
}

// line) to w.
func (t *Transcript) SaveJSONL(w io.Writer) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for i := range t.Record {
		if err := enc.Encode(&t.Record[i]); err != nil {
			// Best-effort: flush any successfully-encoded prefix to w so a
			// partial transcript is at least recoverable on the next load.
			// We discard the flush error — the encode error is the one to
			// surface, and the trailing bw.Flush in the success path
			// remains the authoritative one.
			_ = bw.Flush()
			return fmt.Errorf("agent: encode transcript event %d: %w", i, err)
		}
	}
	return bw.Flush()
}

// LoadJSONL reads a JSONL transcript (as written by [Transcript.SaveJSONL])
// from r.
//
// Every line must carry schema_version ([llmkit.EventSchemaVersion]): a line
// without one is a pre-schema recording, which is unsupported — the load
// fails with an error naming the line instead of silently loading zero
// values. The transcript's RunID/ParentRunID are recovered from the decoded
// events; a file mixing two run ids is an error naming the offending line.
func LoadJSONL(r io.Reader) (*Transcript, error) {
	t := NewTranscript()
	sc := bufio.NewScanner(r)
	// Transcript lines can be large (full conversations); raise the line cap.
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := sc.Bytes()
		if len(raw) == 0 {
			continue
		}
		var ev llmkit.Event
		if err := json.Unmarshal(raw, &ev); err != nil {
			return nil, fmt.Errorf("agent: decode transcript line %d: %w", line, err)
		}
		if ev.SchemaVersion == 0 {
			return nil, fmt.Errorf("agent: transcript line %d: missing schema_version: pre-schema recordings are unsupported", line)
		}
		if ev.RunID != "" {
			if t.RunID == "" {
				t.RunID = ev.RunID
			} else if t.RunID != ev.RunID {
				return nil, fmt.Errorf("agent: transcript line %d: run id %q mixes with %q", line, ev.RunID, t.RunID)
			}
		}
		if ev.ParentRunID != "" && t.ParentRunID == "" {
			t.ParentRunID = ev.ParentRunID
		}
		t.Record = append(t.Record, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("agent: read transcript: %w", err)
	}
	return t, nil
}

// JSONLSink is a durable event sink that streams a run's events to one JSONL
// file per run under a directory. Construct it with [JSONL] and install it
// with [WithObserver].
//
// Identity is the RunID alone: the file is "<dir>/<RunID>.jsonl", opened
// exclusively (O_EXCL) by the run's Start event and closed when its Finalize
// arrives — one RunID is one file, created exactly once. The admission state
// machine, per RunID:
//
//	Start, no entry:    create O_EXCL. Success → live entry, event written.
//	                    File exists (leftover from a prior process) → onErr
//	                    ("leftover file …; run refused") and a disabled
//	                    entry: the rest of this run's events drop silently,
//	                    reported once. Any other open error → same.
//	Start, entry live:  onErr ("duplicate Start for live run"), the entry is
//	                    poisoned (disabled, fd closed). Further events for
//	                    the id drop silently — the file keeps the first
//	                    run's prefix; nothing is ever merged.
//	Non-Start, no entry: onErr ("event for unknown or closed run") and drop.
//	                    Every occurrence reports — a harness bug is loud.
//	Non-Start, live:    written.
//	Finalize:           written if live; the fd closes and the entry is
//	                    deleted unconditionally (live, disabled, poisoned).
//	                    A later event for the id takes the unknown-run path.
//
// Everything is best-effort and never fails a run: refusals and write
// failures flow to the onErr callback given at construction, path-qualified
// so an operator can tell WHICH file stopped being written. onErr may be
// nil. An encode failure drops the single line and keeps the run live.
//
// A JSONLSink is safe for concurrent use: events from concurrent runs on one
// [Runner] multiplex by [llmkit.RunID], each run writing under its own lock.
// It satisfies [llmkit.Observer] and [Source].
type JSONLSink struct {
	dir   string
	onErr func(error)

	mu   sync.Mutex
	runs map[llmkit.RunID]*jsonlRun
}

// jsonlRun is one run's admission state inside a [JSONLSink].
type jsonlRun struct {
	mu       sync.Mutex
	path     string
	file     *os.File
	enc      *json.Encoder
	disabled bool // refused (leftover/duplicate) or poisoned: events drop silently
}

// JSONL returns a durable event sink that appends one JSON line per event to
// "<RunID>.jsonl" under dir, creating the directory when the run's Start
// event opens its file. Pass it to [WithObserver]; a Runner accepts at most
// one durable sink, and a later WithObserver wins (last-wins).
//
// onErr, when non-nil, receives every refusal or write failure with the run
// and file path attached (see [JSONLSink] for the admission table). The sink
// never fails the run.
func JSONL(dir string, onErr func(error)) *JSONLSink {
	return &JSONLSink{dir: dir, onErr: onErr, runs: map[llmkit.RunID]*jsonlRun{}}
}

// Observe routes ev through the admission state machine documented on
// [JSONLSink].
func (s *JSONLSink) Observe(_ context.Context, ev llmkit.Event) {
	switch ev.Kind {
	case llmkit.KindStart:
		s.startRun(ev)
	case llmkit.KindFinalize:
		s.finalizeRun(ev)
	default:
		s.appendRun(ev)
	}
}

// startRun admits a run's Start event (see the table on [JSONLSink]).
func (s *JSONLSink) startRun(ev llmkit.Event) {
	path := s.pathFor(ev)
	if !safeRunID(ev.RunID) {
		s.fail(ev.RunID, path, errors.New(`run id is not a safe filename component (empty, "." or "..", a path separator, or a NUL)`))
		return
	}
	s.mu.Lock()
	if live, exists := s.runs[ev.RunID]; exists {
		s.mu.Unlock()
		// Duplicate Start: report, poison the existing entry — both runs
		// drop from here; the file keeps the first run's prefix.
		s.fail(ev.RunID, path, errors.New("duplicate Start for live run"))
		live.mu.Lock()
		live.disabled = true
		if live.file != nil {
			_ = live.file.Close()
			live.file = nil
			live.enc = nil
		}
		live.mu.Unlock()
		return
	}
	run := &jsonlRun{path: path}
	s.runs[ev.RunID] = run
	s.mu.Unlock()

	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		s.disable(ev.RunID, run, fmt.Errorf("mkdir %s: %w", s.dir, err))
		return
	}
	f, err := os.OpenFile(run.path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			s.disable(ev.RunID, run, fmt.Errorf("leftover file for run at %s; run refused: %w", run.path, err))
		} else {
			s.disable(ev.RunID, run, fmt.Errorf("open %s: %w", run.path, err))
		}
		return
	}
	// Re-verify admission: a concurrent duplicate Start may have poisoned or
	// replaced this entry while the open was in flight. Close the orphan.
	s.mu.Lock()
	admitted := s.runs[ev.RunID] == run
	s.mu.Unlock()
	if !admitted {
		_ = f.Close()
		return
	}
	run.mu.Lock()
	run.file = f
	run.enc = json.NewEncoder(f)
	err = run.enc.Encode(&ev)
	run.mu.Unlock()
	if err != nil {
		// Drop the line, keep the run live (a later event might encode).
		s.fail(ev.RunID, run.path, fmt.Errorf("encode: %w", err))
	}
}

// appendRun writes a non-Start, non-Finalize event (see the table on
// [JSONLSink]).
func (s *JSONLSink) appendRun(ev llmkit.Event) {
	s.mu.Lock()
	run := s.runs[ev.RunID]
	s.mu.Unlock()
	if run == nil {
		s.fail(ev.RunID, s.pathFor(ev), errors.New("event for unknown or closed run dropped"))
		return
	}
	run.mu.Lock()
	defer run.mu.Unlock()
	if run.disabled || run.enc == nil {
		return // refused or poisoned runs drop silently
	}
	if err := run.enc.Encode(&ev); err != nil {
		// Drop the line, keep the run live.
		s.fail(ev.RunID, run.path, fmt.Errorf("encode: %w", err))
	}
}

// finalizeRun writes the Finalize if the run is live, closes the fd, and
// deletes the entry unconditionally (see the table on [JSONLSink]).
func (s *JSONLSink) finalizeRun(ev llmkit.Event) {
	s.mu.Lock()
	run := s.runs[ev.RunID]
	s.mu.Unlock()
	if run == nil {
		s.fail(ev.RunID, s.pathFor(ev), errors.New("event for unknown or closed run dropped"))
		return
	}
	run.mu.Lock()
	if !run.disabled && run.enc != nil {
		if err := run.enc.Encode(&ev); err != nil {
			s.fail(ev.RunID, run.path, fmt.Errorf("encode: %w", err))
		}
	}
	if run.file != nil {
		_ = run.file.Close()
		run.file = nil
		run.enc = nil
	}
	run.mu.Unlock()
	s.mu.Lock()
	delete(s.runs, ev.RunID)
	s.mu.Unlock()
}

// disable retires a refused run's entry: onErr fires once here, and every
// later event of the run drops silently.
func (s *JSONLSink) disable(run llmkit.RunID, r *jsonlRun, err error) {
	r.mu.Lock()
	r.disabled = true
	if r.file != nil {
		_ = r.file.Close()
		r.file = nil
		r.enc = nil
	}
	r.mu.Unlock()
	s.fail(run, r.path, err)
}

// safeRunID reports whether id can be a JSONL filename component: non-empty,
// not "." or "..", no path separators, no NUL.
func safeRunID(id llmkit.RunID) bool {
	s := string(id)
	if s == "" || s == "." || s == ".." {
		return false
	}
	if strings.ContainsAny(s, `/\`) || strings.ContainsRune(s, 0) {
		return false
	}
	return true
}

// Events returns the recorded events of run by reading the run's file back —
// the read side of [JSONL]. It satisfies [Source], so [NewReplayClient]
// replays from the sink exactly as from a [Transcript]; the caller owns the
// returned slice.
//
// The exact path "<dir>/<RunID>.jsonl" is opened (an id that is not a safe
// filename component is refused with [ErrUnknownRun] before any open); a
// missing file wraps [ErrUnknownRun]. Every decoded event must carry run, a
// file mixing run ids is an error naming file and line, an empty record
// wraps [ErrUnknownRun], and a record holding more than one Start is an
// error naming the run — one RunID is one run.
func (s *JSONLSink) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	if !safeRunID(run) {
		return nil, fmt.Errorf("agent: run id %q is not a safe filename component: %w", run, ErrUnknownRun)
	}
	path := filepath.Join(s.dir, string(run)+".jsonl")
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("agent: no JSONL transcript for run %s at %s: %w", run, path, ErrUnknownRun)
		}
		return nil, fmt.Errorf("agent: open %s: %w", path, err)
	}
	t, err := LoadJSONL(f)
	_ = f.Close()
	if err != nil {
		return nil, fmt.Errorf("agent: %s: %w", path, err)
	}
	if len(t.Record) == 0 {
		return nil, fmt.Errorf("agent: %s holds no events for run %s: %w", path, run, ErrUnknownRun)
	}
	for i, ev := range t.Record {
		if ev.RunID != run {
			return nil, fmt.Errorf("agent: %s line %d: event run id %q does not match %q", path, i+1, ev.RunID, run)
		}
	}
	starts := 0
	for _, ev := range t.Record {
		if ev.Kind == llmkit.KindStart {
			starts++
		}
	}
	if starts > 1 {
		return nil, fmt.Errorf("agent: record for run %s holds %d runs", run, starts)
	}
	return slices.Clone(t.Record), nil
}

// pathFor is a run's file name: "<RunID>.jsonl". Events that arrive without
// a RunID (a sink driven outside a Runner) fall back to the literal name
// "run" for the error message; the Start-admission guard rejects the empty
// id before any file is touched.
func (s *JSONLSink) pathFor(ev llmkit.Event) string {
	id := string(ev.RunID)
	if id == "" {
		id = "run"
	}
	return filepath.Join(s.dir, id+".jsonl")
}

// fail reports a sink refusal or failure through onErr, run- and
// path-qualified so an operator can tell WHICH autosave file stopped being
// written.
func (s *JSONLSink) fail(run llmkit.RunID, path string, err error) {
	if s.onErr != nil {
		s.onErr(fmt.Errorf("agent: jsonl transcript %s (run %s): %w", path, run, err))
	}
}

// argsEqual compares two raw JSON argument payloads semantically — as parsed
// JSON, so key order and insignificant whitespace never split a replay. A
// payload that fails to parse compares byte-equal instead.
func argsEqual(a, b json.RawMessage) bool {
	if string(a) == string(b) {
		return true
	}
	var av, bv any
	if json.Unmarshal(a, &av) != nil || json.Unmarshal(b, &bv) != nil {
		return string(a) == string(b)
	}
	return reflect.DeepEqual(av, bv)
}

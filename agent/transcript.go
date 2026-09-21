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
// Every line must be a well-formed [llmkit.Event] — [llmkit.Event.Validate]
// checks the shape: a declared Kind carrying exactly its own payload, and a
// non-zero schema_version. A line without one is a pre-schema recording,
// which is unsupported — the load fails with an error naming the line
// instead of silently loading zero values. The transcript's
// RunID/ParentRunID are recovered from the decoded events; a file mixing
// two run ids is an error naming the offending line.
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
		if err := ev.Validate(); err != nil {
			return nil, fmt.Errorf("agent: transcript line %d: %w", line, err)
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
//	Start, no entry:    create the directory, create the file O_EXCL and
//	                    write the Start line, all under the sink lock: the
//	                    live entry is published only once its file exists
//	                    and holds that line — unless the encode itself
//	                    failed, the one way a created file can end up
//	                    without it. A failed create instead installs a
//	                    retired entry — onErr once ("leftover file …; run
//	                    refused" when the file already exists, "mkdir …" or
//	                    "open …" otherwise) and the rest of this run's
//	                    events drop silently.
//	Start, entry exists: onErr ("duplicate Start for run …; already
//	                    recorded or refused") and the entry is poisoned: a
//	                    live run's fd closes, an already-retired one stays
//	                    retired, and from there BOTH runs' events drop
//	                    silently. The file keeps the first run's prefix —
//	                    and because that file now exists, every later Start
//	                    for the id is refused as a leftover, so no second
//	                    run can write into it.
//	Non-Start, no entry: onErr ("event for unknown or closed run") and drop.
//	                    Every occurrence reports — a harness bug is loud.
//	Non-Start, live:    written.
//	Finalize:           the entry leaves the table and its fd closes,
//	                    unconditionally (live, refused, poisoned); the
//	                    Finalize line is written first when the run was
//	                    live. A later event for the id takes the unknown-run
//	                    path.
//
// Everything is best-effort and never fails a run: refusals and write
// failures flow to the onErr callback given at construction, path-qualified
// so an operator can tell WHICH file stopped being written. An encode
// failure drops the single line and keeps the run live. onErr may be nil,
// and never runs under a sink lock — it may even call back into the sink.
//
// A JSONLSink is safe for concurrent use: events from concurrent runs on one
// [Runner] multiplex by [llmkit.RunID], each run writing under its own lock.
// Start admission — create and first line included — is serialized on the
// sink lock, so a duplicate RunID can never interleave into a half-open
// entry. It satisfies [llmkit.Observer] and [Source].
type JSONLSink struct {
	dir   string
	onErr func(error)

	mu   sync.Mutex
	runs map[llmkit.RunID]*jsonlRun
}

// jsonlRun is one run's admission state inside a [JSONLSink]. An entry is
// published fully formed and only ever holds one of two states: live (file
// and enc non-nil, the run's Start line already on disk) or retired (both
// nil — refused at Start, poisoned by a duplicate, or finalized). A nil enc
// is exactly "this run's events drop silently".
type jsonlRun struct {
	mu   sync.Mutex
	path string
	file *os.File
	enc  *json.Encoder
}

// write encodes ev into the run's file. A retired run has no encoder and
// drops its events silently.
func (r *jsonlRun) write(ev *llmkit.Event) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.enc == nil {
		return nil
	}
	return r.enc.Encode(ev)
}

// writeAndRetire writes a live run's last line and retires it in one step,
// so nothing can be appended between the two.
func (r *jsonlRun) writeAndRetire(ev *llmkit.Event) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var err error
	if r.enc != nil {
		err = r.enc.Encode(ev)
	}
	r.retireLocked()
	return err
}

// retire closes the run's fd, if it still has one, and drops the entry out
// of the live state: every later event for it is discarded silently.
func (r *jsonlRun) retire() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.retireLocked()
}

// retireLocked is [jsonlRun.retire] with r.mu already held.
func (r *jsonlRun) retireLocked() {
	if r.file != nil {
		_ = r.file.Close()
		r.file = nil
	}
	r.enc = nil
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
//
// Admission, the exclusive create and the Start line are one step under the
// sink lock: nothing can observe an entry before its file exists and carries
// that line, so a concurrent duplicate can neither poison a half-open entry
// — which would strand an empty file — nor win the id's path for a second
// run. The refusal, if any, is reported after the lock is dropped; onErr
// never runs under a sink lock.
func (s *JSONLSink) startRun(ev llmkit.Event) {
	path := s.pathFor(ev)
	if !safeRunID(ev.RunID) {
		s.fail(ev.RunID, path, errors.New(`run id is not a safe filename component (empty, "." or "..", a path separator, or a NUL)`))
		return
	}
	if err := s.admit(ev, path); err != nil {
		s.fail(ev.RunID, path, err)
	}
}

// admit is startRun's critical section: it runs the whole admission under
// the sink lock and returns the refusal or write failure to report once the
// lock is dropped.
func (s *JSONLSink) admit(ev llmkit.Event, path string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if prior, exists := s.runs[ev.RunID]; exists {
		// Duplicate Start: poison the entry. Both runs drop from here and
		// the file keeps the first run's prefix. The entry need not be
		// live — a refused or already-poisoned run holds one too, until
		// its Finalize — so the refusal does not claim it is.
		prior.retire()
		return fmt.Errorf("duplicate Start for run %s; already recorded or refused", ev.RunID)
	}
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		s.runs[ev.RunID] = &jsonlRun{path: path} // retired: reported once, then silent
		return fmt.Errorf("mkdir %s: %w", s.dir, err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		s.runs[ev.RunID] = &jsonlRun{path: path}
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("leftover file for run at %s; run refused: %w", path, err)
		}
		return fmt.Errorf("open %s: %w", path, err)
	}
	run := &jsonlRun{path: path, file: f, enc: json.NewEncoder(f)}
	encErr := run.enc.Encode(&ev)
	s.runs[ev.RunID] = run
	if encErr != nil {
		// Drop the line, keep the run live (a later event might encode).
		return fmt.Errorf("encode: %w", encErr)
	}
	return nil
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
	if err := run.write(&ev); err != nil {
		// Drop the line, keep the run live.
		s.fail(ev.RunID, run.path, fmt.Errorf("encode: %w", err))
	}
}

// finalizeRun writes the Finalize if the run is live, then retires it: the
// entry leaves the table and its fd closes unconditionally (see the table on
// [JSONLSink]). Taking the entry out under the same lock that found it means
// no second Finalize — and no duplicate Start — can reach a run that is
// already finalizing.
func (s *JSONLSink) finalizeRun(ev llmkit.Event) {
	s.mu.Lock()
	run := s.runs[ev.RunID]
	delete(s.runs, ev.RunID)
	s.mu.Unlock()
	if run == nil {
		s.fail(ev.RunID, s.pathFor(ev), errors.New("event for unknown or closed run dropped"))
		return
	}
	if err := run.writeAndRetire(&ev); err != nil {
		s.fail(ev.RunID, run.path, fmt.Errorf("encode: %w", err))
	}
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
//
// Read a run once it has finalized. The sink streams a line per event, and
// nothing synchronizes this read against a write in flight, so a read that
// races an append can decode a torn last line and fail. A finalized run's
// file is complete and closed.
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

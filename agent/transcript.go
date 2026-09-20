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
	if run != t.RunID {
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
// One RunID is one file, created exactly once: the run's first event opens
// "<RunID>-<task-slug>.jsonl" (the name derives from the Start event, so a
// caller that pinned the id with [WithRunID] can compute it) exclusively
// (O_EXCL), and the file closes when the run's Finalize event arrives. A
// RunID whose file already exists on disk — a leftover from a prior process,
// or a duplicate id — is refused: the failure is reported through onErr
// naming the run and the path, and every event of that run is dropped. The
// same refusal applies to a second in-process run presenting an
// already-open RunID, to a RunID that is empty or not a safe filename
// component (a path separator or ".."), and to any event arriving after the
// run's Finalize closed its file.
//
// Streaming is best-effort and never fails a run: a directory or file open
// failure disables that run's remaining writes and reports through onErr; an
// encode failure drops the single line. Every failure is path-qualified so
// an operator can tell WHICH file stopped being written. onErr may be nil.
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

// jsonlRun is one live run's open file state inside a [JSONLSink].
type jsonlRun struct {
	mu       sync.Mutex
	path     string
	file     *os.File
	enc      *json.Encoder
	disabled bool // an open failure retired this run's writes
}

// JSONL returns a durable event sink that appends one JSON line per event to
// "<RunID>-<task-slug>.jsonl" under dir, creating the directory on demand at
// the first event. Pass it to [WithObserver]; a Runner accepts at most one
// durable sink, and a later WithObserver wins (last-wins).
//
// onErr, when non-nil, receives every refusal or write failure with the run
// and file path attached (see [JSONLSink]). The sink never fails the run.
func JSONL(dir string, onErr func(error)) *JSONLSink {
	return &JSONLSink{dir: dir, onErr: onErr}
}

// Observe appends ev to its run's JSONL file. It satisfies
// [llmkit.Observer]. The Start event opens the run record (it derives the
// filename); the Finalize event closes the file and retires the run's entry.
func (s *JSONLSink) Observe(_ context.Context, ev llmkit.Event) {
	s.mu.Lock()
	run := s.runs[ev.RunID]
	s.mu.Unlock()
	if run == nil {
		// First sight of this run in the sink — normally its Start. A
		// non-Start first event (a sink misused outside a Runner, or an event
		// arriving after Finalize retired the entry) re-runs the exclusive
		// open: the finished run's file is still on disk, so the open fails
		// with ErrExist and the event is refused and reported.
		if !openRun(s, ev) {
			return // refused: every event of this run is dropped
		}
		s.mu.Lock()
		run = s.runs[ev.RunID]
		s.mu.Unlock()
		if run == nil {
			return
		}
	}
	run.mu.Lock()
	defer run.mu.Unlock()
	if run.disabled {
		return
	}
	if run.file == nil {
		// First append: the run's one and only open, exclusive (O_EXCL). A
		// file already on disk — leftover from a prior process, or events
		// arriving after Finalize — refuses the run.
		if err := os.MkdirAll(s.dir, 0o755); err != nil {
			run.disabled = true
			s.fail(ev.RunID, run.path, fmt.Errorf("mkdir %s: %w", s.dir, err))
			return
		}
		f, err := os.OpenFile(run.path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err != nil {
			run.disabled = true
			s.fail(ev.RunID, run.path, fmt.Errorf("open %s: %w", run.path, err))
			return
		}
		run.file = f
		run.enc = json.NewEncoder(f)
	}
	if err := run.enc.Encode(&ev); err != nil {
		// Leave the file open (a later event might still succeed); drop this
		// line, matching autosave's discard-on-error contract.
		s.fail(ev.RunID, run.path, fmt.Errorf("encode: %w", err))
		return
	}
	if ev.Kind == llmkit.KindFinalize {
		// The run is over: close its file and retire the entry. A post-close
		// event for the same RunID re-runs the O_EXCL open path, fails with
		// ErrExist (the file is still on disk), and is reported — a harness
		// bug, not something to swallow.
		_ = run.file.Close()
		run.file = nil
		run.enc = nil
		s.mu.Lock()
		delete(s.runs, ev.RunID)
		s.mu.Unlock()
	}
}

// openRun admits a run's first event in this sink: it validates the RunID as
// a filename component, refuses a RunID that is already live, and creates the
// run's entry with its path — derived from the Start event, or, for a
// non-Start first event (post-Finalize arrival, misuse outside a Runner),
// located among the run's existing files so the exclusive open refuses
// against the record that already exists. It reports every refusal through
// onErr and returns false when the run's events must be dropped.
func openRun(s *JSONLSink, ev llmkit.Event) bool {
	id := ev.RunID
	path := s.pathFor(ev)
	if !safeRunID(id) {
		s.fail(id, path, errors.New(`run id is not a safe filename component (empty, a path separator, or ..)`))
		return false
	}
	if ev.Kind != llmkit.KindStart {
		// Not the run's opening event: the record must already exist. Locate
		// it so the exclusive open refuses against the RIGHT file.
		matches, err := filepath.Glob(filepath.Join(s.dir, string(id)+"-*.jsonl"))
		if err != nil || len(matches) != 1 {
			if err == nil && len(matches) > 1 {
				s.fail(id, "", fmt.Errorf("no single record for run: %d candidates (%v)", len(matches), matches))
			} else {
				s.fail(id, path, fmt.Errorf("no record for run: %w", ErrUnknownRun))
			}
			return false
		}
		path = matches[0]
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.runs == nil {
		s.runs = map[llmkit.RunID]*jsonlRun{}
	}
	if _, live := s.runs[id]; live {
		s.fail(id, path, errors.New("run id already has an open record in this sink"))
		return false
	}
	s.runs[id] = &jsonlRun{path: path}
	return true
}

// safeRunID reports whether id can be a JSONL filename component: non-empty,
// no path separators, never ".." .
func safeRunID(id llmkit.RunID) bool {
	s := string(id)
	if s == "" || s == ".." || strings.Contains(s, "/") || strings.Contains(s, string(os.PathSeparator)) {
		return false
	}
	return true
}

// Events returns the recorded events of run by reading its JSONL file back —
// the read side of [JSONL]. It satisfies [Source], so [NewReplayClient] and
// [ReplayTools] replay from the sink exactly as from a [Transcript]; the
// caller owns the returned slice.
//
// Only files named "<RunID>-*.jsonl" are considered, and only events whose
// RunID matches are returned. A directory with no file for run yields an
// error wrapping [ErrUnknownRun]; more than one candidate file is an error
// naming them — one RunID is one file.
func (s *JSONLSink) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	if run == "" {
		return nil, fmt.Errorf("agent: JSONL source needs a run id: %w", ErrUnknownRun)
	}
	matches, err := filepath.Glob(filepath.Join(s.dir, string(run)+"-*.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("agent: scan %s for run %s: %w", s.dir, run, err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("agent: no JSONL transcript for run %s under %s: %w", run, s.dir, ErrUnknownRun)
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("agent: %d JSONL transcripts match run %s under %s (%v): one run must be one file", len(matches), run, s.dir, matches)
	}
	f, err := os.Open(matches[0])
	if err != nil {
		return nil, fmt.Errorf("agent: open %s: %w", matches[0], err)
	}
	t, err := LoadJSONL(f)
	_ = f.Close()
	if err != nil {
		return nil, fmt.Errorf("agent: %s: %w", matches[0], err)
	}
	evs := make([]llmkit.Event, 0, len(t.Record))
	for _, ev := range t.Record {
		if ev.RunID == run {
			evs = append(evs, ev)
		}
	}
	return evs, nil
}

// pathFor derives a run's file name from its Start event:
// "<RunID>-<task-slug>.jsonl". Events that arrive without a RunID (a sink
// driven outside a Runner) fall back to the literal name "run"; a first
// event that is not the Start has no task to slug, so it does too.
func (s *JSONLSink) pathFor(ev llmkit.Event) string {
	id := string(ev.RunID)
	if id == "" {
		id = "run"
	}
	task := ""
	if ev.Start != nil {
		task = ev.Start.Task
	}
	return filepath.Join(s.dir, id+"-"+slug(task)+".jsonl")
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

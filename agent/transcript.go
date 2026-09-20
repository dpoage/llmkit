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
	"strconv"
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

// Events returns the recorded events of run in emission order. It satisfies
// [Source]. A transcript records exactly one run, so run must match
// [Transcript.RunID] (or be empty); any other id yields no events and no
// error.
func (t *Transcript) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	if run != "" && run != t.RunID {
		return nil, nil
	}
	return t.Record, nil
}

// SaveJSONL writes the transcript as JSON Lines (one [llmkit.Event] per
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
// values.
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
// The filename is "<RunID>-<task-slug>.jsonl", derived from the run's Start
// event, so a caller that pinned the run id with [WithRunID] can recover the
// exact file later by name. The first open of a run's file is exclusive
// (O_EXCL): two runs that somehow derive the same path never interleave two
// JSONL streams into one file — the loser appends a numeric suffix before
// the extension. A reopen for a run whose file was already closed appends
// (O_APPEND), so late events of a run land in the same file.
//
// Streaming is best-effort and never fails a run: a directory or file open
// failure disables that run's remaining writes; an encode failure drops the
// single line. Every failure is reported through the onErr callback given at
// construction, path-qualified so an operator can tell WHICH file stopped
// being written. onErr may be nil.
//
// A run's file closes when its Finalize event arrives. A JSONLSink is safe
// for concurrent use: events from concurrent runs on one [Runner] multiplex
// by [llmkit.RunID]. It satisfies [llmkit.Observer] and [Source].
type JSONLSink struct {
	dir   string
	onErr func(error)

	mu   sync.Mutex
	runs map[llmkit.RunID]*jsonlRun
}

// jsonlRun is one run's open file state inside a [JSONLSink].
type jsonlRun struct {
	path     string
	file     *os.File
	enc      *json.Encoder
	opened   bool // a file was created for this run at least once
	disabled bool // an open failure retired this run's writes
}

// JSONL returns a durable event sink that appends one JSON line per event to
// "<RunID>-<task-slug>.jsonl" under dir, creating the directory on demand at
// the first event. Pass it to [WithObserver]; a Runner accepts at most one
// durable sink, and a later WithObserver wins (last-wins).
//
// onErr, when non-nil, receives every write failure with the file path
// attached (see [JSONLSink]). The sink never fails the run.
func JSONL(dir string, onErr func(error)) *JSONLSink {
	return &JSONLSink{dir: dir, onErr: onErr}
}

// Observe appends ev to its run's JSONL file. It satisfies
// [llmkit.Observer]. The Start event opens the run record (it derives the
// filename); the Finalize event closes the file.
func (s *JSONLSink) Observe(_ context.Context, ev llmkit.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.runs == nil {
		s.runs = map[llmkit.RunID]*jsonlRun{}
	}
	run := s.runs[ev.RunID]
	if ev.Kind == llmkit.KindStart && ev.Start != nil {
		if run == nil {
			run = &jsonlRun{}
			s.runs[ev.RunID] = run
		}
		run.path = s.pathFor(ev)
	}
	if run == nil || run.disabled || run.path == "" {
		// An event before its run's Start, or a run whose file already
		// failed: nothing to append to.
		return
	}
	if run.file == nil {
		f, path, err := s.openFile(run)
		if err != nil {
			// Open failures retire the run's writes for good, matching the
			// never-fail-the-run autosave contract.
			run.disabled = true
			s.fail(run.path, err)
			return
		}
		run.file = f
		run.path = path
		run.enc = json.NewEncoder(f)
		run.opened = true
	}
	if err := run.enc.Encode(&ev); err != nil {
		// Leave the file open (a later event might still succeed); drop this
		// line, matching autosave's discard-on-error contract.
		s.fail(run.path, fmt.Errorf("encode: %w", err))
		return
	}
	if ev.Kind == llmkit.KindFinalize {
		// The run is over. The path entry survives so a late event for the
		// same run reopens O_APPEND instead of erroring.
		_ = run.file.Close()
		run.file = nil
		run.enc = nil
	}
}

// Events returns the recorded events of run by reading its JSONL file back.
// It satisfies [Source], so [NewReplayClient] and [ReplayTools] replay from
// the sink exactly as from a [Transcript] — replay never depends on which
// sink recorded a run.
//
// The file is located by its "<RunID>-*.jsonl" name. A directory with no
// file for run yields an error naming the run and the directory: replaying a
// run that was never recorded is a caller bug, not an empty success.
func (s *JSONLSink) Events(_ context.Context, run llmkit.RunID) ([]llmkit.Event, error) {
	if run == "" {
		return nil, errors.New("agent: JSONL source needs a run id")
	}
	matches, err := filepath.Glob(filepath.Join(s.dir, string(run)+"-*.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("agent: scan %s for run %s: %w", s.dir, run, err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("agent: no JSONL transcript for run %s under %s", run, s.dir)
	}
	var evs []llmkit.Event
	for _, m := range matches {
		f, err := os.Open(m)
		if err != nil {
			return nil, fmt.Errorf("agent: open %s: %w", m, err)
		}
		t, err := LoadJSONL(f)
		_ = f.Close()
		if err != nil {
			return nil, fmt.Errorf("agent: %s: %w", m, err)
		}
		evs = append(evs, t.Record...)
	}
	return evs, nil
}

// pathFor derives a run's file name from its Start event:
// "<RunID>-<task-slug>.jsonl". Events that arrive without a RunID (a sink
// driven outside a Runner) fall back to the literal name "run".
func (s *JSONLSink) pathFor(ev llmkit.Event) string {
	id := string(ev.RunID)
	if id == "" {
		id = "run"
	}
	return filepath.Join(s.dir, id+"-"+slug(ev.Start.Task)+".jsonl")
}

// openFile creates the sink file for a run's first append and returns the
// (possibly disambiguated) path. The FIRST open is exclusive (O_EXCL): two
// concurrent runs that derive the same path must never interleave two JSONL
// streams into one file, so the loser appends a numeric suffix before the
// extension ("-1", "-2", …) and gets its own file. A reopen after the run's
// file was closed is NOT exclusive: the file already exists and must be
// appended to.
func (s *JSONLSink) openFile(run *jsonlRun) (*os.File, string, error) {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return nil, run.path, fmt.Errorf("mkdir %s: %w", s.dir, err)
	}
	if run.opened {
		f, err := os.OpenFile(run.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return nil, run.path, fmt.Errorf("open %s: %w", run.path, err)
		}
		return f, run.path, nil
	}
	path := run.path
	for i := 0; ; i++ {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_APPEND, 0o644)
		if err == nil {
			return f, path, nil
		}
		if !errors.Is(err, os.ErrExist) || i >= 99 {
			return nil, path, fmt.Errorf("open %s: %w", path, err)
		}
		// Collision with a concurrent run or a leftover file: disambiguate.
		path = strings.TrimSuffix(run.path, ".jsonl") + "-" + strconv.Itoa(i+1) + ".jsonl"
	}
}

// fail reports a sink failure through onErr, path-qualified so an operator
// can tell WHICH autosave file stopped being written.
func (s *JSONLSink) fail(path string, err error) {
	if s.onErr != nil {
		s.onErr(fmt.Errorf("agent: jsonl transcript %s: %w", path, err))
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

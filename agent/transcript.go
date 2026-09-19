package agent

import (
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/dpoage/llmkit"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// EventKind tags each [Event] in a [Transcript].
type EventKind string

const (
	// EventRequest records the messages sent to the model for one completion
	// (the full conversation state at that step).
	EventRequest EventKind = "request"
	// EventAssistant records the model's response: its text and any tool calls,
	// plus the usage and stop reason for that completion.
	EventAssistant EventKind = "assistant"
	// EventToolResult records the result of executing one tool call.
	EventToolResult EventKind = "tool_result"
)

// Event is a single ordered entry in a [Transcript]. Only the fields relevant
// to its Kind are populated; the rest stay at their zero value and are omitted
// from JSON. Events serialize one-per-line as JSONL.
type Event struct {
	// Kind discriminates the event.
	Kind EventKind `json:"kind"`
	// Step is the 1-based model-turn this event belongs to.
	Step int `json:"step"`
	// Time is when the event was recorded.
	Time time.Time `json:"time"`

	// Messages is set on EventRequest: the full conversation sent to the model.
	Messages []llmkit.Message `json:"messages,omitempty"`

	// Text is set on EventAssistant: the model's text output.
	Text string `json:"text,omitempty"`
	// Blocks is set on EventAssistant: every content block (text,
	// thinking, ...) the model returned, so JSONL transcripts round-trip
	// reasoning payloads verbatim. Block.Data is base64 in JSONL per
	// encoding/json's []byte handling.
	Blocks []llmkit.Block `json:"blocks,omitempty"`
	// ToolCalls is set on EventAssistant: the model's tool-use requests.
	ToolCalls []llmkit.ToolCall `json:"tool_calls,omitempty"`
	// StopReason is set on EventAssistant.
	StopReason llmkit.StopReason `json:"stop_reason,omitempty"`
	// Usage is set on EventAssistant: token usage for that completion.
	Usage *llmkit.Usage `json:"usage,omitempty"`

	// ToolCallID is set on EventToolResult: the call this result answers.
	ToolCallID string `json:"tool_call_id,omitempty"`
	// ToolName is set on EventToolResult: the tool that produced the result.
	ToolName string `json:"tool_name,omitempty"`
	// Result is set on EventToolResult: the textual tool output (possibly an
	// "ERROR:"-prefixed message).
	Result string `json:"result,omitempty"`
	// IsError is set on EventToolResult when the tool returned an error.
	IsError bool `json:"is_error,omitempty"`
}

// Transcript is the ordered record of a single [Runner.Run]. It stores full
// content (not hashes) so offline evaluation can replay it deterministically.
// A Transcript is not safe for concurrent mutation, but a single Runner
// appends to it sequentially.
//
// Streaming: the Runner arms streaming when [WithTranscriptDir] is set.
// While streaming is armed, every recorded event is also appended as one
// JSON line to the autosave file, so an operator can `tail -f` a stuck
// run's transcript instead of waiting for the run to finish. The file opens
// lazily on the first recorded event and closes at run end. Streaming is
// best-effort: an open or encode failure disables it for the rest of the
// transcript, a broken disk never affects the run's result, and every
// failure is reported through [Hooks.TranscriptError].
type Transcript struct {
	Events []Event `json:"-"`
	clock  func() time.Time

	streamPath string
	streamFile *os.File
	streamEnc  *json.Encoder
	// streamOpened records that a stream file was created for this transcript
	// at least once; the first open is exclusive (collision-avoiding), a
	// reopen after closeStream must append to the existing file.
	streamOpened bool
	// onStreamErr, when non-nil, receives every streaming failure.
	onStreamErr func(error)
}

// NewTranscript returns an empty transcript using the real wall clock.
func NewTranscript() *Transcript {
	return &Transcript{clock: time.Now}
}

// enableStreaming arms incremental JSONL writes to path: the file is created
// (directories included) on the first subsequent record* call, not here, so a
// run that records nothing never touches disk. A no-op path (empty string)
// leaves streaming disabled. onErr, when non-nil, is called once per streaming
// failure (directory creation, file open, line encode) with a wrapped,
// path-qualified error; the Runner passes [Hooks.TranscriptError].
func (t *Transcript) enableStreaming(path string, onErr func(error)) {
	t.streamPath = path
	t.onStreamErr = onErr
}

// streamAppend writes ev as one JSON line to the streaming file, opening it
// (and its parent directory) on first use. The file is opened in APPEND mode
// so a reopen after closeStream (e.g. RunJSON's repair() completion, which
// reuses the run's Transcript after the main loop already closed the stream)
// continues the same JSONL file instead of truncating it. Best-effort: any
// failure disables further attempts for this transcript by clearing
// streamPath (open failures) or drops the single line (encode failures),
// matching the never-fail-the-run autosave contract; every failure is
// reported through onErr before it is swallowed.
func (t *Transcript) streamAppend(ev *Event) {
	if t.streamPath == "" {
		return
	}
	if t.streamFile == nil {
		f, path, err := t.openStreamFile()
		if err != nil {
			t.streamFail(err)
			t.streamPath = ""
			return
		}
		t.streamFile = f
		t.streamPath = path
		t.streamEnc = json.NewEncoder(f)
		t.streamOpened = true
	}
	if err := t.streamEnc.Encode(ev); err != nil {
		// Leave the file open (a later event might still succeed); just drop
		// this line, matching autosave's discard-on-error contract.
		t.streamFail(fmt.Errorf("encode: %w", err))
		return
	}
}

// openStreamFile creates the stream file for the first append and returns the
// (possibly disambiguated) path. The FIRST open is exclusive (O_EXCL): two
// concurrent runs that derive the same path — same millisecond timestamp,
// same task slug — must never interleave two JSONL streams into one file, so
// the loser appends a numeric suffix before the extension ("-1", "-2", …)
// and gets its own file. A reopen after closeStream (the RunJSON repair pass
// appending to the same transcript) is NOT exclusive: the file already
// exists and must be appended to.
func (t *Transcript) openStreamFile() (*os.File, string, error) {
	if err := os.MkdirAll(filepath.Dir(t.streamPath), 0o755); err != nil {
		return nil, "", fmt.Errorf("mkdir: %w", err)
	}
	path := t.streamPath
	if t.streamOpened {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return nil, "", fmt.Errorf("open %s: %w", path, err)
		}
		return f, path, nil
	}
	for i := 0; ; i++ {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_APPEND, 0o644)
		if err == nil {
			return f, path, nil
		}
		if !os.IsExist(err) || i >= 99 {
			return nil, "", fmt.Errorf("open %s: %w", path, err)
		}
		// Collision with a concurrent run or a leftover file: disambiguate.
		path = strings.TrimSuffix(t.streamPath, ".jsonl") + "-" + strconv.Itoa(i+1) + ".jsonl"
	}
}

// streamFail reports a streaming failure through onErr, path-qualified so an
// operator can tell WHICH autosave file stopped being written.
func (t *Transcript) streamFail(err error) {
	if t.onStreamErr != nil {
		t.onStreamErr(fmt.Errorf("agent: transcript stream %s: %w", t.streamPath, err))
	}
}

// closeStream closes the streaming file, if one was opened. Idempotent and
// best-effort: called unconditionally at run end (and every early-return
// path) regardless of whether streaming was ever armed or actually opened a
// file.
//
// streamPath deliberately survives the close: a Transcript outlives one
// run() call — RunJSON's repair() reuses the same *Transcript for its one
// repair completion AFTER run() has already called closeStream. Because
// streamAppend opens with O_APPEND (never truncating), a post-close record
// simply reopens the same path and appends, so the repair request/assistant
// turns land in the SAME on-disk transcript as the main run — the missing
// tail that made unparseable-repair outputs undiagnosable from the JSONL
// alone.
func (t *Transcript) closeStream() {
	if t.streamFile != nil {
		_ = t.streamFile.Close()
		t.streamFile = nil
		t.streamEnc = nil
	}
}

// now returns the current time, allowing tests to inject a fixed clock.
func (t *Transcript) now() time.Time {
	if t.clock != nil {
		return t.clock()
	}
	return time.Now()
}

// recordRequest appends an EventRequest capturing the messages sent for step.
func (t *Transcript) recordRequest(step int, msgs []llmkit.Message) {
	// Copy the slice so later mutation of the conversation doesn't alias the
	// recorded snapshot.
	snap := make([]llmkit.Message, len(msgs))
	copy(snap, msgs)
	t.Events = append(t.Events, Event{
		Kind:     EventRequest,
		Step:     step,
		Time:     t.now(),
		Messages: snap,
	})
	t.streamAppend(&t.Events[len(t.Events)-1])
}

// recordAssistant appends an EventAssistant for the model's response at step.
func (t *Transcript) recordAssistant(step int, resp llmkit.Response) {
	u := resp.Usage
	t.Events = append(t.Events, Event{
		Kind:       EventAssistant,
		Step:       step,
		Time:       t.now(),
		Text:       resp.Text,
		Blocks:     resp.Blocks,
		ToolCalls:  resp.ToolCalls,
		StopReason: resp.StopReason,
		Usage:      &u,
	})
	t.streamAppend(&t.Events[len(t.Events)-1])
}

// recordToolResult appends an EventToolResult for one executed tool call.
func (t *Transcript) recordToolResult(step int, call llmkit.ToolCall, result string, isErr bool) {
	t.Events = append(t.Events, Event{
		Kind:       EventToolResult,
		Step:       step,
		Time:       t.now(),
		ToolCallID: call.ID,
		ToolName:   call.Name,
		Result:     result,
		IsError:    isErr,
	})
	t.streamAppend(&t.Events[len(t.Events)-1])
}

// SaveJSONL writes the transcript as JSON Lines (one Event per line) to w.
func (t *Transcript) SaveJSONL(w io.Writer) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for i := range t.Events {
		if err := enc.Encode(&t.Events[i]); err != nil {
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

// LoadJSONL reads a JSONL transcript (as written by SaveJSONL) from r.
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
		var ev Event
		if err := json.Unmarshal(raw, &ev); err != nil {
			return nil, fmt.Errorf("agent: decode transcript line %d: %w", line, err)
		}
		t.Events = append(t.Events, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("agent: read transcript: %w", err)
	}
	return t, nil
}

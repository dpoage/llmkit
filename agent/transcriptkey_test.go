package agent

import (
	"context"
	"os"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestWithRunID_PinnedFilename verifies WithRunID pins the run's identity AND
// the durable sink's filename: the JSONL file is named after the run id
// ("<RunID>-<task-slug>.jsonl"), so a caller that generates stable
// identifiers up front (the motivation WithTranscriptKey served before the
// cutover) recovers the exact transcript file later by exact name instead of
// guessing from a timestamp window. Events carry the pinned id too.
func TestWithRunID_PinnedFilename(t *testing.T) {
	dir := t.TempDir()
	fc := newFakeClient(textResp("done", 1, 1))
	r := NewRunner(fc, nil, "sys", WithObserver(JSONL(dir, nil)))

	out, err := r.Run(context.Background(), "My Task!", WithRunID("run-abc123"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1: %v", len(entries), entries)
	}
	name := entries[0].Name()
	if name != "run-abc123.jsonl" {
		t.Errorf("filename = %q, want exactly the pinned id: run-abc123.jsonl (identity is the RunID alone; the task rides the Start event)", name)
	}
	if out.RunID != "run-abc123" {
		t.Errorf("Outcome.RunID = %q, want the pinned id", out.RunID)
	}
	for _, ev := range out.Transcript.Record {
		if ev.RunID != "run-abc123" {
			t.Errorf("event kind %s carries RunID %q, want the pinned id", ev.Kind, ev.RunID)
		}
	}

	// The pinned id makes the sink's read side addressable by name.
	src := JSONL(dir, nil)
	if _, err := src.Events(context.Background(), "run-abc123"); err != nil {
		t.Errorf("sink Events(%q): %v", "run-abc123", err)
	}
	if _, err := src.Events(context.Background(), "other-run"); err == nil {
		t.Error("sink Events for an unrecorded run succeeded; want an error")
	}
}

// TestJSONL_RefusesPostFinalizeEvents pins the one-file-one-run rule: after
// a run's Finalize closed its file, a late event for the same RunID re-runs
// the exclusive open, fails against the existing file, and is reported
// through onErr — never appended, never swallowed.
func TestJSONL_RefusesPostFinalizeEvents(t *testing.T) {
	dir := t.TempDir()
	var mu sync.Mutex
	var errs []error
	sink := JSONL(dir, func(err error) {
		mu.Lock()
		defer mu.Unlock()
		errs = append(errs, err)
	})
	ctx := context.Background()

	mk := func(kind llmkit.EventKind, task string) llmkit.Event {
		ev := llmkit.NewEvent(llmkit.WithRun(ctx, "rid-1"), kind)
		switch kind {
		case llmkit.KindStart:
			ev.Start = &llmkit.StartEvent{Task: task}
		case llmkit.KindFinalize:
			ev.Finalize = &llmkit.FinalizeEvent{}
		}
		return ev
	}
	sink.Observe(ctx, mk(llmkit.KindStart, "closed"))
	sink.Observe(ctx, mk(llmkit.KindFinalize, "closed"))
	sink.Observe(ctx, mk(llmkit.KindFinalize, "closed")) // post-close: refused

	evs, err := sink.Events(ctx, "rid-1")
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if len(evs) != 2 {
		t.Fatalf("sink file holds %d events, want 2 (post-close event dropped, file untouched)", len(evs))
	}
	mu.Lock()
	defer mu.Unlock()
	if len(errs) == 0 {
		t.Fatal("post-close event was dropped silently; want an onErr report")
	}
}

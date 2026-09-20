package agent

import (
	"context"
	"os"
	"strings"
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
	if !strings.HasPrefix(name, "run-abc123-") {
		t.Errorf("filename %q does not start with the pinned run id", name)
	}
	if !strings.HasSuffix(name, "-my-task.jsonl") {
		t.Errorf("filename %q does not end with the expected task slug", name)
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

// TestJSONL_ReopenAfterFinalize pins the O_APPEND contract: after a run's
// Finalize closed its file, a late event for the same run REOPENS and
// appends to the same file instead of failing or truncating.
func TestJSONL_ReopenAfterFinalize(t *testing.T) {
	dir := t.TempDir()
	sink := JSONL(dir, nil)
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
	sink.Observe(ctx, mk(llmkit.KindStart, "reopened"))
	sink.Observe(ctx, mk(llmkit.KindFinalize, "reopened"))
	sink.Observe(ctx, mk(llmkit.KindFinalize, "reopened")) // post-close append

	evs, err := sink.Events(ctx, "rid-1")
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if len(evs) != 3 {
		t.Fatalf("sink file holds %d events, want 3 (append, never truncate)", len(evs))
	}
}

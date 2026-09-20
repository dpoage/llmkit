package llmkit

import (
	"context"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestRunFromContextAbsent(t *testing.T) {
	if got := RunFromContext(context.Background()); got != "" {
		t.Fatalf("RunFromContext(absent) = %q, want empty", got)
	}
	// A context carrying a non-RunID value under a colliding key type must
	// not be readable: the key type is unexported and unique to this package.
	type sneaky struct{}
	ctx := context.WithValue(context.Background(), sneaky{}, RunID("other"))
	if got := RunFromContext(ctx); got != "" {
		t.Fatalf("RunFromContext(foreign key) = %q, want empty", got)
	}
}

func TestWithRunRoundTrip(t *testing.T) {
	ctx := WithRun(context.Background(), RunID("1758366600000-deadbeef00112233"))
	if got := RunFromContext(ctx); got != "1758366600000-deadbeef00112233" {
		t.Fatalf("RunFromContext = %q, want the stored id", got)
	}
	// Overwriting wins.
	inner := WithRun(ctx, RunID("second"))
	if got := RunFromContext(inner); got != "second" {
		t.Fatalf("RunFromContext(overwritten) = %q, want %q", got, "second")
	}
	// The outer context keeps its own value.
	if got := RunFromContext(ctx); got != "1758366600000-deadbeef00112233" {
		t.Fatalf("outer context mutated: %q", got)
	}
}

func TestWithRunEmptyIsAsAbsent(t *testing.T) {
	if got := RunFromContext(WithRun(context.Background(), "")); got != "" {
		t.Fatalf("RunFromContext(WithRun(\"\")) = %q, want empty", got)
	}
}

func TestNewRunIDFormat(t *testing.T) {
	id := NewRunID()
	if !regexp.MustCompile(`^[0-9]{13}-[0-9a-f]{16}$`).MatchString(string(id)) {
		t.Fatalf("NewRunID() = %q, want <13-digit millis>-<16 hex chars>", id)
	}
}

func TestNewRunIDUniqueAcross10k(t *testing.T) {
	seen := make(map[RunID]struct{}, 10000)
	for i := 0; i < 10000; i++ {
		id := NewRunID()
		if _, dup := seen[id]; dup {
			t.Fatalf("duplicate RunID %q at iteration %d", id, i)
		}
		seen[id] = struct{}{}
	}
}

func TestNewRunIDLexicallyMonotonicAcrossTick(t *testing.T) {
	first := NewRunID()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().UnixMilli() == parseMillisPrefix(first) {
		if time.Now().After(deadline) {
			t.Fatal("clock never advanced past the first id's millisecond")
		}
		time.Sleep(time.Millisecond)
	}
	second := NewRunID()
	if !(first < second) {
		t.Fatalf("ids not lexically monotonic across a tick: %q !< %q", first, second)
	}
	if strings.Compare(string(first), string(second)) >= 0 {
		t.Fatalf("strings.Compare disagrees with <: %q vs %q", first, second)
	}
}

// parseMillisPrefix extracts the 13-digit millisecond prefix so the tick
// loop compares the same field NewRunID sorts by.
func parseMillisPrefix(id RunID) int64 {
	var millis int64
	for _, c := range string(id[:13]) {
		millis = millis*10 + int64(c-'0')
	}
	return millis
}

func TestNewEventStampsHeader(t *testing.T) {
	before := time.Now()
	ctx := WithRun(context.Background(), RunID("1758366600000-deadbeef00112233"))
	ev := NewEvent(ctx, KindCompletion)
	after := time.Now()

	if ev.Kind != KindCompletion {
		t.Fatalf("Kind = %q, want %q", ev.Kind, KindCompletion)
	}
	if ev.RunID != "1758366600000-deadbeef00112233" {
		t.Fatalf("RunID = %q, want the context's run id", ev.RunID)
	}
	if ev.SchemaVersion != EventSchemaVersion {
		t.Fatalf("SchemaVersion = %d, want %d", ev.SchemaVersion, EventSchemaVersion)
	}
	if ev.Duration != 0 {
		t.Fatalf("Duration = %s, want zero (emitter sets it)", ev.Duration)
	}
	if ev.Time.Before(before) || ev.Time.After(after) {
		t.Fatalf("Time %v outside [%v, %v]", ev.Time, before, after)
	}
	// Outside a run, RunID stays empty.
	if got := NewEvent(context.Background(), KindExec).RunID; got != "" {
		t.Fatalf("NewEvent(outside run).RunID = %q, want empty", got)
	}
}

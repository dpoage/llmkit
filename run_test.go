package llmkit

import (
	"context"
	"encoding/json"
	"reflect"
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

// TestWithRunEmptyPreservesOuter pins the documented contract: an empty id
// returns ctx unchanged, so a decorator whose options left the id unset can
// never erase an enclosing run. The outer context is non-empty — that is
// the only case where "absent" and "erased" differ.
func TestWithRunEmptyPreservesOuter(t *testing.T) {
	outer := WithRun(context.Background(), RunID("1758366600000-deadbeef00112233"))
	if got := RunFromContext(WithRun(outer, "")); got != "1758366600000-deadbeef00112233" {
		t.Fatalf("WithRun(outer, \"\") erased the enclosing run id: got %q", got)
	}
	if got := RunFromContext(context.Background()); got != "" {
		t.Fatalf("RunFromContext(absent) = %q, want empty", got)
	}
}

func TestSpanFromContextAbsent(t *testing.T) {
	if got := SpanFromContext(context.Background()); got != "" {
		t.Fatalf("SpanFromContext(absent) = %q, want empty", got)
	}
	type sneaky struct{}
	ctx := context.WithValue(context.Background(), sneaky{}, SpanID("other"))
	if got := SpanFromContext(ctx); got != "" {
		t.Fatalf("SpanFromContext(foreign key) = %q, want empty", got)
	}
}

func TestWithSpanRoundTrip(t *testing.T) {
	ctx := WithSpan(context.Background(), SpanID("1758366600001-0123456789abcdef"))
	if got := SpanFromContext(ctx); got != "1758366600001-0123456789abcdef" {
		t.Fatalf("SpanFromContext = %q, want the stored id", got)
	}
	inner := WithSpan(ctx, SpanID("second"))
	if got := SpanFromContext(inner); got != "second" {
		t.Fatalf("SpanFromContext(overwritten) = %q, want %q", got, "second")
	}
	if got := SpanFromContext(ctx); got != "1758366600001-0123456789abcdef" {
		t.Fatalf("outer context mutated: %q", got)
	}
}

func TestWithSpanEmptyPreservesOuter(t *testing.T) {
	outer := WithSpan(context.Background(), SpanID("1758366600001-0123456789abcdef"))
	if got := SpanFromContext(WithSpan(outer, "")); got != "1758366600001-0123456789abcdef" {
		t.Fatalf("WithSpan(outer, \"\") erased the enclosing span id: got %q", got)
	}
}

// Run and span ids live under independent keys: one context carries both,
// which is how a Completion's Attempts join on span inside a run.
func TestRunAndSpanKeysAreIndependent(t *testing.T) {
	ctx := WithSpan(WithRun(context.Background(), RunID("run-1")), SpanID("span-1"))
	if got := RunFromContext(ctx); got != "run-1" {
		t.Fatalf("RunFromContext = %q, want run-1", got)
	}
	if got := SpanFromContext(ctx); got != "span-1" {
		t.Fatalf("SpanFromContext = %q, want span-1", got)
	}
}

func TestNewRunIDFormat(t *testing.T) {
	id := NewRunID()
	if !regexp.MustCompile(`^[0-9]{13}-[0-9a-f]{16}$`).MatchString(string(id)) {
		t.Fatalf("NewRunID() = %q, want <13-digit millis>-<16 hex chars>", id)
	}
}

func TestNewSpanIDFormat(t *testing.T) {
	id := NewSpanID()
	if !regexp.MustCompile(`^[0-9]{13}-[0-9a-f]{16}$`).MatchString(string(id)) {
		t.Fatalf("NewSpanID() = %q, want <13-digit millis>-<16 hex chars>", id)
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

func TestNewSpanIDUniqueAcross10k(t *testing.T) {
	seen := make(map[SpanID]struct{}, 10000)
	for i := 0; i < 10000; i++ {
		id := NewSpanID()
		if _, dup := seen[id]; dup {
			t.Fatalf("duplicate SpanID %q at iteration %d", id, i)
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
	if first >= second {
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
	ctx := WithSpan(WithRun(context.Background(), RunID("1758366600000-deadbeef00112233")), SpanID("1758366600001-0123456789abcdef"))
	ev := NewEvent(ctx, KindCompletion)
	after := time.Now()

	if ev.Kind != KindCompletion {
		t.Fatalf("Kind = %q, want %q", ev.Kind, KindCompletion)
	}
	if ev.RunID != "1758366600000-deadbeef00112233" {
		t.Fatalf("RunID = %q, want the context's run id", ev.RunID)
	}
	if ev.SpanID != "1758366600001-0123456789abcdef" {
		t.Fatalf("SpanID = %q, want the context's span id", ev.SpanID)
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
	// Outside a run and span, both stay empty.
	bare := NewEvent(context.Background(), KindExec)
	if bare.RunID != "" || bare.SpanID != "" {
		t.Fatalf("NewEvent(outside run).RunID/SpanID = %q/%q, want empty/empty", bare.RunID, bare.SpanID)
	}
}

// TestNewEventRoundTripsDeepEqual pins that a stamped event survives the
// wire exactly: the header Time is stamped without a monotonic clock
// reading, in UTC, so the decoded copy is DeepEqual to the emitted one.
func TestNewEventRoundTripsDeepEqual(t *testing.T) {
	ctx := WithSpan(WithRun(context.Background(), RunID("1758366600000-deadbeef00112233")), SpanID("1758366600001-0123456789abcdef"))
	ev := NewEvent(ctx, KindExec)
	if ev.Time.Location() != time.UTC {
		t.Fatalf("NewEvent stamped %s, want UTC so the decode is DeepEqual", ev.Time.Location())
	}
	ev.Exec = &ExecEvent{Backend: "docker", Command: []string{"sh"}, ExitCode: 0}
	data, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Event
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(ev, back) {
		t.Fatalf("stamped event did not survive the wire\nwant: %#v\ngot:  %#v", ev, back)
	}
}

func TestStepFromContextAbsent(t *testing.T) {
	if got := StepFromContext(context.Background()); got != 0 {
		t.Fatalf("StepFromContext(absent) = %d, want 0", got)
	}
}

func TestWithStepRoundTrip(t *testing.T) {
	ctx := WithStep(context.Background(), 3)
	if got := StepFromContext(ctx); got != 3 {
		t.Fatalf("StepFromContext = %d, want the stored step", got)
	}
}

// TestWithStepNonPositivePreservesOuter pins the documented contract: step
// <= 0 returns ctx unchanged, mirroring WithRun's empty-id rule, so a caller
// with no turn in scope can pass 0 unconditionally without erasing an
// enclosing step.
func TestWithStepNonPositivePreservesOuter(t *testing.T) {
	outer := WithStep(context.Background(), 2)
	for _, step := range []int{0, -1} {
		if got := StepFromContext(WithStep(outer, step)); got != 2 {
			t.Fatalf("WithStep(outer, %d) read back %d, want the enclosing 2", step, got)
		}
	}
	if got := StepFromContext(WithStep(context.Background(), 0)); got != 0 {
		t.Fatalf("WithStep(absent, 0) read back %d, want 0", got)
	}
}

func TestWithStepOverwrite(t *testing.T) {
	ctx := WithStep(WithStep(context.Background(), 2), 5)
	if got := StepFromContext(ctx); got != 5 {
		t.Fatalf("StepFromContext = %d, want the innermost 5", got)
	}
}

// TestRunSpanStepKeysAreIndependent: one context carries all three header
// facts — run, span, and step — which is how a decorator-emitted event
// inside a Runner turn joins on every axis at once.
func TestRunSpanStepKeysAreIndependent(t *testing.T) {
	ctx := WithStep(WithSpan(WithRun(context.Background(), RunID("run-1")), SpanID("span-1")), 4)
	if got := RunFromContext(ctx); got != "run-1" {
		t.Errorf("RunFromContext = %q, want run-1", got)
	}
	if got := SpanFromContext(ctx); got != "span-1" {
		t.Errorf("SpanFromContext = %q, want span-1", got)
	}
	if got := StepFromContext(ctx); got != 4 {
		t.Errorf("StepFromContext = %d, want 4", got)
	}
}

// TestNewEventStampsStepFromContext pins the NewEvent half: the step rides
// the header like RunID/SpanID, stays 0 outside a Runner turn, and an
// emitter's explicit assignment still wins over the context's value.
func TestNewEventStampsStepFromContext(t *testing.T) {
	if ev := NewEvent(WithStep(context.Background(), 3), KindToolRun); ev.Step != 3 {
		t.Errorf("NewEvent Step = %d, want 3 from the context", ev.Step)
	}
	if ev := NewEvent(context.Background(), KindToolRun); ev.Step != 0 {
		t.Errorf("NewEvent Step outside a turn = %d, want 0", ev.Step)
	}
	// "Explicit assignment wins" has no meaningful pin at this layer: a
	// local assignment to a struct field is true by construction and cannot
	// detect a NewEvent mutant. The real pin is Runner behavior — agent's
	// nested-Runner test asserts a child run's start event reports Step 0
	// even though the parent's tool-phase context carried the parent's
	// turn, which only holds if the Runner's explicit zero beats the
	// context's value.
}

package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// seamTool returns a 1200-byte text blob, or a caption plus a 300-byte image for args {"k":"img"}.
type seamTool struct{}

func (seamTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "mix", Description: "mixed results", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (seamTool) Run(_ context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	if strings.Contains(string(args), "img") {
		return []llmkit.Block{llmkit.Text("snap"), llmkit.Image("image/png", make([]byte, 300))}, nil
	}
	return textBlocks(strings.Repeat("x", 1200)), nil
}

func seamTurns(n int) []scriptStep { return seamTurnsFrom(1, n) }

// seamTurnsFrom scripts n tool turns numbered from first, odd ones text, even ones image.
func seamTurnsFrom(first, n int) []scriptStep {
	var steps []scriptStep
	for i := first; i < first+n; i++ {
		args := `{"k":"txt"}`
		if i%2 == 0 {
			args = `{"k":"img"}`
		}
		steps = append(steps, toolResp(fmt.Sprintf("c%d", i), "mix", args, 100, 10))
	}
	return steps
}

func unprunedResults(req llmkit.Request) int {
	n := 0
	for _, m := range req.Messages {
		if m.Role == llmkit.RoleToolResult && !isCompactStub(m.Text()) {
			n++
		}
	}
	return n
}

func dumpRun(b *bytes.Buffer, title string, out *Outcome) {
	b.WriteString("== " + title + " messages\n")
	for _, m := range out.Messages {
		j, _ := json.Marshal(m)
		b.Write(j)
		b.WriteByte('\n')
	}
	b.WriteString("== " + title + " transcript\n")
	tr := NewTranscript()
	for _, ev := range out.Transcript.Record {
		ev.Time, ev.Duration, ev.SpanID = time.Time{}, 0, ""
		tr.Record = append(tr.Record, ev)
	}
	_ = tr.SaveJSONL(b)
}

// compactorGoldenDump continues a history of media-client results (image blocks
// in history) on a degrading client, once with threshold compaction (two
// firings, so the re-armed threshold shows) and once with overflow recovery.
func compactorGoldenDump(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer

	seed := func() *Outcome {
		fc := newFakeClient(append(seamTurns(4), textResp("seeded", 100, 10))...)
		fc.caps.ToolResultMedia = true
		out, err := NewRunner(fc, []Tool{seamTool{}}, "sys", WithLimits(Limits{MaxIterations: 30})).
			Run(context.Background(), "seed", WithRunID("seam-seed"))
		if err != nil {
			t.Fatalf("seed Run: %v", err)
		}
		return out
	}

	fc := newFakeClient(append(seamTurnsFrom(5, 8), textResp("done", 100, 10))...)
	out, err := NewRunner(fc, []Tool{seamTool{}}, "sys", WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 150})).
		Run(context.Background(), "task", Continue(seed()), WithRunID("seam-threshold"))
	if err != nil {
		t.Fatalf("threshold Run: %v", err)
	}
	if n := len(compactionEvents(out.Transcript.Record)); n < 2 {
		t.Fatalf("threshold scenario compacted %d times, want at least 2", n)
	}
	dumpRun(&b, "threshold", out)

	gc := &gateClient{
		steps:    append(seamTurnsFrom(5, 3), textResp("done", 100, 10)),
		overflow: func(req llmkit.Request) bool { return unprunedResults(req) >= 7 },
	}
	out, err = NewRunner(gc, []Tool{seamTool{}}, "sys", WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 1_000_000})).
		Run(context.Background(), "task", Continue(seed()), WithRunID("seam-overflow"))
	if err != nil {
		t.Fatalf("overflow Run: %v", err)
	}
	if gc.refused != 1 || len(compactionEvents(out.Transcript.Record)) != 1 {
		t.Fatalf("overflow scenario refused %d, compactions %d, want 1 and 1", gc.refused, len(compactionEvents(out.Transcript.Record)))
	}
	dumpRun(&b, "overflow", out)
	return b.Bytes()
}

// SM2: with no WithCompactor the default path equals the ae3b7fa Runner byte for byte.
func TestCompactor_DefaultMatchesBase(t *testing.T) {
	got := compactorGoldenDump(t)
	want, err := os.ReadFile(filepath.Join("testdata", "compactor_default.golden"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("default compaction drifted from the ae3b7fa golden\n--- got\n%s\n--- want\n%s", got, want)
	}
}

type seamCall struct {
	reason  string
	aligned bool
}

// seamRecorder is a Compactor that stubs like the default and records each call.
type seamRecorder struct {
	calls []seamCall
	err   error
}

func (s *seamRecorder) Compact(ctx context.Context, in CompactInput) (CompactOutput, error) {
	aligned := len(in.Messages) == len(in.Sent)
	for i := range in.Messages {
		aligned = aligned && i < len(in.Sent) && in.Sent[i].Role == in.Messages[i].Role && in.Sent[i].ToolCallID == in.Messages[i].ToolCallID
	}
	s.calls = append(s.calls, seamCall{in.Reason, aligned})
	if s.err != nil {
		return CompactOutput{}, s.err
	}
	return stubCompactor{}.Compact(ctx, in)
}

// SM1: a custom compactor serves both triggers and its output becomes history.
func TestCompactor_CustomServesThresholdAndOverflow(t *testing.T) {
	const marker = "[custom compacted]"
	var reasons []string
	custom := CompactorFunc(func(_ context.Context, in CompactInput) (CompactOutput, error) {
		reasons = append(reasons, in.Reason)
		out := make([]llmkit.Message, len(in.Messages))
		copy(out, in.Messages)
		pruned := 0
		for i, m := range out {
			if m.Role == llmkit.RoleToolResult && !strings.HasPrefix(m.Text(), marker) && i < len(out)-8 {
				m.Content = textBlocks(marker)
				out[i] = m
				pruned++
			}
		}
		return CompactOutput{Messages: out, Pruned: pruned}, nil
	})
	markers := func(msgs []llmkit.Message) int {
		n := 0
		for _, m := range msgs {
			if m.Role == llmkit.RoleToolResult && m.Text() == marker {
				n++
			}
		}
		return n
	}

	t.Run("threshold", func(t *testing.T) {
		reasons = nil
		fc := newFakeClient(append(overflowTurns(8), textResp("done", 100, 10))...)
		out, err := overflowRunner(fc, 1500, WithCompactor(custom)).Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if len(reasons) == 0 || reasons[0] != "" {
			t.Fatalf("compactor reasons = %q, want threshold (empty) first", reasons)
		}
		if markers(out.Messages) == 0 {
			t.Errorf("compactor output did not become history")
		}
		if comps := compactionEvents(out.Transcript.Record); len(comps) == 0 || comps[0].Reason != "" {
			t.Errorf("compaction events = %+v, want threshold events", comps)
		}
	})
	t.Run("overflow", func(t *testing.T) {
		reasons = nil
		gc := &gateClient{
			steps:    append(overflowTurns(6), textResp("done", 100, 10)),
			overflow: func(req llmkit.Request) bool { return toolResultBytes(req) > 5*overflowBlobBytes },
		}
		out, err := overflowRunner(gc, 1_000_000, WithCompactor(custom)).Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if len(reasons) != 1 || reasons[0] != llmkit.CompactionReasonOverflow {
			t.Fatalf("compactor reasons = %q, want exactly [overflow]", reasons)
		}
		if gc.refused != 1 || markers(out.Messages) == 0 {
			t.Errorf("refused %d, markers in history %d: compactor output was not the retried history", gc.refused, markers(out.Messages))
		}
		comps := compactionEvents(out.Transcript.Record)
		if len(comps) != 1 || comps[0].Reason != llmkit.CompactionReasonOverflow || comps[0].Pruned != markers(out.Messages) {
			t.Errorf("compaction events = %+v, want one overflow event with the compactor's Pruned", comps)
		}
	})
}

// SM1 (input contract): Sent is index-aligned with Messages.
func TestCompactor_InputIsIndexAlignedRequestCopy(t *testing.T) {
	rec := &seamRecorder{}
	fc := newFakeClient(append(seamTurns(7), textResp("done", 100, 10))...)
	_, err := NewRunner(fc, []Tool{seamTool{}}, "sys", WithLimits(Limits{HistoryTokenBudget: 150}), WithCompactor(rec)).
		Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.calls) == 0 {
		t.Fatal("compactor never called")
	}
	for _, c := range rec.calls {
		if !c.aligned {
			t.Errorf("call %+v: Sent is not index-aligned with Messages", c)
		}
	}
}

// SM3: a Compactor without a HistoryTokenBudget compacts on overflow only.
func TestCompactor_WithoutBudgetRecoversButNeverThresholds(t *testing.T) {
	rec := &seamRecorder{}
	gc := &gateClient{
		steps:    append(overflowTurns(6), textResp("done", 100, 10)),
		overflow: func(req llmkit.Request) bool { return toolResultBytes(req) > 5*overflowBlobBytes },
	}
	out, err := overflowRunner(gc, 0, WithCompactor(rec)).Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rec.calls) != 1 || rec.calls[0].reason != llmkit.CompactionReasonOverflow {
		t.Fatalf("compactor calls = %+v, want exactly one overflow call", rec.calls)
	}
	if gc.refused != 1 || gc.calls() != 8 {
		t.Errorf("refused %d calls %d, want 1 and 8", gc.refused, gc.calls())
	}
	if comps := compactionEvents(out.Transcript.Record); len(comps) != 1 {
		t.Errorf("compaction events = %+v, want only the overflow one", comps)
	}

	rec = &seamRecorder{}
	fc := newFakeClient(append(overflowTurns(8), textResp("done", 100, 10))...)
	if _, err := overflowRunner(fc, 0, WithCompactor(rec)).Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if len(rec.calls) != 0 {
		t.Errorf("compactor called %d times with no overflow and no budget, want 0", len(rec.calls))
	}
}

// SM4: a compactor error ends the run wrapped, with no retry.
func TestCompactor_ErrorEndsRunWithoutRetry(t *testing.T) {
	boom := errors.New("summarizer down")
	t.Run("overflow", func(t *testing.T) {
		gc := &gateClient{
			steps:    append(overflowTurns(6), textResp("done", 100, 10)),
			overflow: func(req llmkit.Request) bool { return toolResultBytes(req) > 5*overflowBlobBytes },
		}
		_, err := overflowRunner(gc, 0, WithCompactor(&seamRecorder{err: boom})).Run(context.Background(), "task")
		if !errors.Is(err, boom) {
			t.Fatalf("err = %v, want it to wrap the compactor error", err)
		}
		if gc.calls() != 7 || gc.refused != 1 {
			t.Errorf("calls %d refused %d, want 7 and 1: the refused invocation was retried", gc.calls(), gc.refused)
		}
	})
	t.Run("threshold", func(t *testing.T) {
		fc := newFakeClient(append(overflowTurns(8), textResp("done", 100, 10))...)
		rec := &seamRecorder{err: boom}
		_, err := overflowRunner(fc, 1500, WithCompactor(rec)).Run(context.Background(), "task")
		if !errors.Is(err, boom) {
			t.Fatalf("err = %v, want it to wrap the compactor error", err)
		}
		if len(rec.calls) != 1 {
			t.Errorf("compactor called %d times, want 1", len(rec.calls))
		}
		if n := fc.callCount(); n >= 8 {
			t.Errorf("completion calls = %d: the run went on after the compactor failed", n)
		}
	})
}

func seamCompactingRecord(t *testing.T) *Outcome {
	t.Helper()
	fc := newFakeClient(append(overflowTurns(8), textResp("done", 100, 10))...)
	out, err := overflowRunner(fc, 1500).Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if n := len(compactionEvents(out.Transcript.Record)); n < 1 {
		t.Fatalf("record holds %d compactions, want at least 1", n)
	}
	return out
}

func seamReplay(t *testing.T, out *Outcome, budget int64, opts ...Option) *ReplayClient {
	t.Helper()
	rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := overflowRunner(rc, budget, opts...).Run(context.Background(), "task"); err != nil {
		t.Fatalf("replayed Run: %v", err)
	}
	return rc
}

// SM5: replay reports a recorded compaction its Runner does not reproduce.
func TestReplay_ReportsUnreproducedCompaction(t *testing.T) {
	t.Run("threshold", func(t *testing.T) {
		out := seamCompactingRecord(t)
		if err := seamReplay(t, out, 1500).Err(); err != nil {
			t.Errorf("same config: rc.Err = %v", err)
		}
		err := seamReplay(t, out, 0).Err()
		if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "compaction") {
			t.Errorf("compaction off: rc.Err = %v, want ErrReplayDiverged naming the compaction", err)
		}
	})
	t.Run("overflow", func(t *testing.T) {
		gc := o1Client()
		out, err := overflowRunner(gc, 1_000_000).Run(context.Background(), "task")
		if err != nil {
			t.Fatal(err)
		}
		if err := seamReplay(t, out, 1_000_000).Err(); err != nil {
			t.Errorf("same config: rc.Err = %v", err)
		}
		rc, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = overflowRunner(rc, 0).Run(context.Background(), "task")
		if err := rc.Err(); !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "compaction") {
			t.Errorf("compaction off: rc.Err = %v, want ErrReplayDiverged naming the compaction", err)
		}
	})
	t.Run("custom compactor", func(t *testing.T) {
		out := seamCompactingRecord(t)
		noop := CompactorFunc(func(_ context.Context, in CompactInput) (CompactOutput, error) {
			return CompactOutput{Messages: in.Messages}, nil
		})
		err := seamReplay(t, out, 1500, WithCompactor(noop)).Err()
		if !errors.Is(err, ErrReplayDiverged) || !strings.Contains(err.Error(), "compaction") {
			t.Errorf("no-op compactor: rc.Err = %v, want ErrReplayDiverged naming the compaction", err)
		}
	})
}

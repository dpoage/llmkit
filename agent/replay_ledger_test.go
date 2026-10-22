package agent

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/dpoage/llmkit"
)

func compactionCompletions(evs []llmkit.Event) []llmkit.Event {
	var out []llmkit.Event
	for _, ev := range evs {
		if ev.Kind == llmkit.KindCompletion && ev.Completion.Purpose == llmkit.CompletionPurposeCompaction {
			out = append(out, ev)
		}
	}
	return out
}

// ledgerOf sums the Response.Usage of every Completion and returns the
// Finalize usage with it.
func ledgerOf(evs []llmkit.Event) (ledger, finalize llmkit.Usage) {
	for _, ev := range evs {
		switch ev.Kind {
		case llmkit.KindCompletion:
			ledger = ledger.Add(ev.Completion.Response.Usage)
		case llmkit.KindFinalize:
			finalize = ev.Finalize.Usage
		}
	}
	return ledger, finalize
}

func assertLedger(t *testing.T, out *Outcome) {
	t.Helper()
	ledger, fin := ledgerOf(out.Transcript.Record)
	if ledger != out.Usage || fin != out.Usage {
		t.Errorf("Completion usage sum %+v, Outcome.Usage %+v, Finalize.Usage %+v: want all equal", ledger, out.Usage, fin)
	}
}

// splitting fails the first summary call with an overflow so a pass holds a
// failed call, two halves and a merge.
func splitting() *sumClient {
	return &sumClient{fn: func(n int, _ llmkit.Request) (llmkit.Response, error) {
		if n == 0 {
			return llmkit.Response{}, tooLong()
		}
		return sumResp(fmt.Sprintf("SUMMARY %d", n)), nil
	}}
}

func assertSameCompactionCalls(t *testing.T, got, want []llmkit.Event) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("replay recorded %d compaction completions, record has %d (want equal, nonzero)", len(got), len(want))
	}
	seen := map[llmkit.SpanID]bool{}
	for i := range want {
		g, w := got[i].Completion, want[i].Completion
		if g.Err != w.Err || g.ErrKind != w.ErrKind || g.Provider != w.Provider || g.Model != w.Model ||
			g.Response.Text != w.Response.Text || g.Response.Usage != w.Response.Usage ||
			len(g.Request.Messages) != len(w.Request.Messages) || g.Request.System != w.Request.System {
			t.Errorf("compaction completion %d = %+v, recorded %+v", i, g, w)
		}
		if got[i].SpanID == "" || seen[got[i].SpanID] {
			t.Errorf("compaction completion %d span %q is empty or repeated", i, got[i].SpanID)
		}
		seen[got[i].SpanID] = true
	}
}

// RI1, RI2, RI4: a replay with rc.Compactor() records the recorded summarizer
// calls, so its ledger holds the spend its Outcome and Finalize report, and
// the replay of that record does the same; no summarizer is called.
func TestReplayCompactor_RecordsRecordedSummarizerCalls(t *testing.T) {
	for _, tc := range []struct {
		name string
		sc   func() *sumClient
	}{{"plain", numbered}, {"split_after_overflow", splitting}} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			sc := tc.sc()
			rec, _, err := summarizeRun(t, sc, nil, append(seamTurns(8), textResp("done", 100, 10)), nil,
				WithObserver(JSONL(dir, func(err error) { t.Errorf("sink: %v", err) })))
			if err != nil || rec.FinalText != "done" || sc.calls() == 0 {
				t.Fatalf("Run: %v, final %q, %d summarizer calls", err, rec.FinalText, sc.calls())
			}
			assertLedger(t, rec)
			recorded := compactionCompletions(rec.Transcript.Record)
			calls := sc.calls()
			for name, src := range map[string]llmkit.Source{"memory": rec.Transcript, "jsonl": JSONL(dir, nil)} {
				t.Run(name, func(t *testing.T) {
					out, rc := replayWithSummaries(t, src, rec.RunID, nil, nil)
					t.Run("ledger", func(t *testing.T) {
						if err := rc.Err(); err != nil {
							t.Fatalf("rc.Err: %v", err)
						}
						if out.Usage != rec.Usage {
							t.Errorf("replayed Outcome.Usage %+v != recorded %+v", out.Usage, rec.Usage)
						}
						assertLedger(t, out)
						assertSameCompactionCalls(t, compactionCompletions(out.Transcript.Record), recorded)
					})
					t.Run("replay_of_replay", func(t *testing.T) {
						rc2, err := NewReplayClient(out.Transcript, out.RunID, llmkit.Capabilities{})
						if err != nil {
							t.Fatalf("NewReplayClient on the replay: %v", err)
						}
						out2, err := NewRunner(rc2, rc2.Tools(), "sys",
							WithLimits(Limits{MaxIterations: 30, HistoryTokenBudget: 150}), WithCompactor(rc2.Compactor()),
						).Run(context.Background(), "task", WithRunID("sum-replay-2"))
						if err != nil {
							t.Fatalf("replaying the replay: %v (rc.Err: %v)", err, rc2.Err())
						}
						if err := rc2.Err(); err != nil {
							t.Fatalf("rc.Err on the replay's replay: %v", err)
						}
						if out2.Usage != rec.Usage {
							t.Errorf("replay of the replay: Outcome.Usage %+v != recorded %+v", out2.Usage, rec.Usage)
						}
						assertLedger(t, out2)
						assertSameCompactionCalls(t, compactionCompletions(out2.Transcript.Record), recorded)
					})
					if sc.calls() != calls {
						t.Errorf("summarizer called %d more times by the replays", sc.calls()-calls)
					}
				})
			}
		})
	}
}

// A recorded failed summarizer call replays as an error with its text, and an
// overflow one still matches llmkit.ErrContextTooLong.
func TestReplayCompactor_RecordedCallErrors(t *testing.T) {
	s := recordedSummary{calls: []recordedCall{
		{ce: llmkit.CompletionEvent{Err: "boom"}},
		{ce: llmkit.CompletionEvent{Err: "too long", ErrKind: llmkit.ErrKindContextTooLong}},
		{ce: llmkit.CompletionEvent{Response: llmkit.Response{Text: "ok"}}},
	}}
	got := s.compactCalls(context.Background())
	if got[0].err == nil || got[0].err.Error() != "boom" || errors.Is(got[0].err, llmkit.ErrContextTooLong) {
		t.Errorf("call 0 err = %v, want plain boom", got[0].err)
	}
	if got[1].err == nil || got[1].err.Error() != "too long" || !errors.Is(got[1].err, llmkit.ErrContextTooLong) {
		t.Errorf("call 1 err = %v, want an ErrContextTooLong with the recorded text", got[1].err)
	}
	if got[2].err != nil || got[2].resp.Text != "ok" {
		t.Errorf("call 2 = %+v, want the recorded success", got[2])
	}
}

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// incompleteErr checks that err is the limit-stop *IncompleteError for out:
// Reason is want and equals out.TruncationReason, and the error's Outcome is
// the very pointer Run returned. It returns a description of the first
// violation, or nil, so goroutine bodies can report it with t.Errorf.
func incompleteErr(out *Outcome, err error, want TruncationReason) error {
	var ie *IncompleteError
	if !errors.As(err, &ie) {
		return fmt.Errorf("err = %v, want *IncompleteError", err)
	}
	if ie.Reason != want {
		return fmt.Errorf("IncompleteError.Reason = %q, want %q", ie.Reason, want)
	}
	if out == nil || out.TruncationReason != ie.Reason {
		return fmt.Errorf("Outcome.TruncationReason = %q, want it to equal Reason %q", out.TruncationReason, ie.Reason)
	}
	if ie.Outcome != out {
		return errors.New("IncompleteError.Outcome is not the Outcome Run returned")
	}
	return nil
}

// TestRun_IncompleteErrorIffTruncated pins Q1-B at Run: over every stop the
// loop can hit and a clean finish, Run returns an *IncompleteError exactly when
// Outcome.TruncationReason is non-empty, with that reason and the returned
// Outcome pointer.
func TestRun_IncompleteErrorIffTruncated(t *testing.T) {
	tool := []Tool{noopTool{}}
	cases := []struct {
		name string
		run  func() (*Outcome, error)
		want TruncationReason
	}{
		{
			name: "max_iterations",
			run: func() (*Outcome, error) {
				r := NewRunner(&bigSpendClient{perCall: 1}, tool, "sys",
					WithLimits(Limits{MaxIterations: 2, TokenBudget: -1}))
				return r.Run(context.Background(), "task")
			},
			want: TruncMaxIterations,
		},
		{
			// An empty turn spends 100 tokens (budget 10) and is nudged
			// rather than finished, so the next loop top sees the run over
			// budget: the pre-turn check fires.
			name: "token_budget pre-turn",
			run: func() (*Outcome, error) {
				fc := newFakeClient(textResp("", 50, 50), textResp("never reached", 1, 1))
				r := NewRunner(fc, tool, "sys",
					WithLimits(Limits{MaxIterations: -1, TokenBudget: 10}))
				return r.Run(context.Background(), "task")
			},
			want: TruncTokenBudget,
		},
		{
			// The first turn's usage crosses the budget while its tool
			// calls run: the post-tool check fires before another turn.
			name: "token_budget post-tool",
			run: func() (*Outcome, error) {
				fc := newFakeClient(toolResp("c1", "echo", `{"v":"x"}`, 50, 50), textResp("never reached", 1, 1))
				r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys",
					WithLimits(Limits{MaxIterations: -1, TokenBudget: 10}))
				return r.Run(context.Background(), "task")
			},
			want: TruncTokenBudget,
		},
		{
			name: "budget_pool",
			run: func() (*Outcome, error) {
				r := NewRunner(&bigSpendClient{perCall: 60}, tool, "sys",
					WithLimits(Limits{MaxIterations: -1, TokenBudget: -1}),
					WithBudgetPool(NewBudgetPool(100)))
				return r.Run(context.Background(), "task")
			},
			want: TruncBudgetPool,
		},
		{
			name: "clean finish",
			run: func() (*Outcome, error) {
				r := NewRunner(newFakeClient(textResp("done", 1, 1)), tool, "sys")
				return r.Run(context.Background(), "task")
			},
			want: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := tc.run()
			if out == nil {
				t.Fatal("Run returned a nil Outcome")
			}
			if out.TruncationReason != tc.want {
				t.Fatalf("TruncationReason = %q, want %q (the row does not exercise its stop)", out.TruncationReason, tc.want)
			}
			if tc.want == "" {
				if err != nil {
					t.Fatalf("clean finish returned err = %v, want nil", err)
				}
				return
			}
			if ierr := incompleteErr(out, err, tc.want); ierr != nil {
				t.Fatal(ierr)
			}
		})
	}
}

// incompleteJSONClient scripts a RunJSON that a limit stops after one tool
// turn: the second completion (the forced finalization turn) returns
// finalText, and the third (a repair, when one runs) returns repairText.
func incompleteJSONClient(finalText, repairText string) *fakeClient {
	return newFakeClient(
		toolResp("c1", "echo", `{"v":"x"}`, 1, 1),
		textResp(finalText, 1, 1),
		textResp(repairText, 1, 1),
	)
}

// TestRunJSON_IncompleteErrorRows pins Q1-B at RunJSON: a parsed answer returns
// nil whatever the TruncationReason, and a truncated run whose answer does not
// parse matches both ErrUnparseableOutput and *IncompleteError at both wrap
// sites (budget skip-repair, post-repair) with the prefix printed once.
func TestRunJSON_IncompleteErrorRows(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}`)
	const parsed = `{"path":"a.go"}`

	t.Run("parsed after forced finalization at the cap returns nil", func(t *testing.T) {
		fc := incompleteJSONClient(parsed, "unused")
		r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{MaxIterations: 1}))
		var got item
		out, err := r.RunJSON(context.Background(), "audit", schema, &got)
		if err != nil {
			t.Fatalf("RunJSON err = %v, want nil for a parsed answer", err)
		}
		if got.Path != "a.go" {
			t.Fatalf("parsed = %+v, want the finalization answer", got)
		}
		if out.TruncationReason != TruncMaxIterations || !out.Finalized {
			t.Fatalf("outcome reason=%q finalized=%v, want max_iterations after a finalization turn", out.TruncationReason, out.Finalized)
		}
		if n := fc.callCount(); n != 2 {
			t.Fatalf("completions = %d, want 2 (tool turn + finalization)", n)
		}
	})

	t.Run("truncated unparseable at the budget skip-repair site", func(t *testing.T) {
		fc := incompleteJSONClient("no json here", "unused")
		r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys",
			WithLimits(Limits{MaxIterations: -1, TokenBudget: 1}))
		var got item
		out, err := r.RunJSON(context.Background(), "audit", schema, &got)
		wantBothIncomplete(t, out, err, TruncTokenBudget)
		if n := fc.callCount(); n != 2 {
			t.Fatalf("completions = %d, want 2 (tool turn + finalization, no repair)", n)
		}
	})

	t.Run("truncated unparseable at the post-repair site", func(t *testing.T) {
		fc := incompleteJSONClient("no json here", "still no json")
		r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithLimits(Limits{MaxIterations: 1}))
		var got item
		out, err := r.RunJSON(context.Background(), "audit", schema, &got)
		wantBothIncomplete(t, out, err, TruncMaxIterations)
		if !strings.Contains(err.Error(), "after one repair") {
			t.Fatalf("err = %q, want the post-repair site", err)
		}
		if n := fc.callCount(); n != 3 {
			t.Fatalf("completions = %d, want 3 (tool turn + finalization + repair)", n)
		}
	})

	t.Run("untruncated unparseable stays a plain unparseable error", func(t *testing.T) {
		fc := newFakeClient(textResp("no json here", 1, 1), textResp("still none", 1, 1))
		r := NewRunner(fc, nil, "sys")
		var got item
		_, err := r.RunJSON(context.Background(), "audit", schema, &got)
		if !errors.Is(err, ErrUnparseableOutput) {
			t.Fatalf("err = %v, want ErrUnparseableOutput", err)
		}
		var ie *IncompleteError
		if errors.As(err, &ie) {
			t.Fatalf("err = %v matches *IncompleteError on a run that finished", err)
		}
	})
}

// wantBothIncomplete asserts err matches ErrUnparseableOutput and
// *IncompleteError (same Outcome pointer, reason want) and carries the
// "agent: " prefix exactly once.
func wantBothIncomplete(t *testing.T, out *Outcome, err error, want TruncationReason) {
	t.Helper()
	if !errors.Is(err, ErrUnparseableOutput) {
		t.Fatalf("err = %v, want errors.Is ErrUnparseableOutput", err)
	}
	if ierr := incompleteErr(out, err, want); ierr != nil {
		t.Fatal(ierr)
	}
	if n := strings.Count(err.Error(), "agent: "); n != 1 {
		t.Fatalf("err %q carries the %q prefix %d times, want once", err, "agent: ", n)
	}
}

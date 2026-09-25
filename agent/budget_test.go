package agent

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
)

// bigSpendClient is a scripted llmkit.Client that reports a large, fixed Usage on
// every completion and always requests a tool so the loop never finishes on its
// own — it can only be stopped by a limit or the shared budget pool. The RUNNER
// charges the pool after every successful completion (WithBudgetPool); the
// client itself never touches it.
type bigSpendClient struct {
	perCall   int64 // input+output tokens reported per completion
	callCount atomic.Int64
}

func (c *bigSpendClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *bigSpendClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}
	c.callCount.Add(1)
	// Split across input/output to mimic the input-dominated reality from a
	// typical run; ChargeableTokens at weight 1.0 sums back to perCall.
	in := c.perCall - c.perCall/10
	out := c.perCall - in
	return llmkit.Response{
		StopReason: llmkit.StopToolUse,
		ToolCalls:  []llmkit.ToolCall{{ID: "c", Name: "noop", Arguments: []byte(`{}`)}},
		Usage:      llmkit.Usage{InputTokens: in, OutputTokens: out},
	}, nil
}

// noopTool always succeeds, keeping the loop turning until a budget stops it.
type noopTool struct{}

func (noopTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: "noop", Description: "noop", Parameters: []byte(`{"type":"object"}`)}
}
func (noopTool) Run(ctx context.Context, args json.RawMessage) (string, error) { return "ok", nil }

// TestBudgetPool_OvershootBound is the acceptance test for shared-pool budget
// enforcement: with a pool of B tokens, P concurrent runners that each report a
// large per-call spend — each Runner checking the pool pre-turn and charging it
// post-completion via WithBudgetPool — total spend must never exceed B plus at
// most one in-flight model-call per concurrent runner.
func TestBudgetPool_OvershootBound(t *testing.T) {
	const (
		budget   int64 = 1_000_000
		perCall  int64 = 150_000 // each turn is a big chunk of the budget
		parallel       = 4
	)

	pool := NewBudgetPool(budget)

	// Each runner consults the shared pool pre-turn and has effectively unlimited
	// per-run budget/iterations, so ONLY the shared pool can stop it.
	limits := Limits{
		MaxIterations: -1,
		TokenBudget:   -1,
	}

	var wg sync.WaitGroup
	for range parallel {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := &bigSpendClient{perCall: perCall}
			r := NewRunner(client, []Tool{noopTool{}}, "sys", WithLimits(limits), WithBudgetPool(pool))
			out, err := r.Run(context.Background(), "task")
			if err != nil {
				t.Errorf("Run: %v", err)
				return
			}
			if !out.Truncated() || out.TruncationReason != TruncBudgetPool {
				t.Errorf("expected TruncBudgetPool stop, got truncated=%v reason=%q",
					out.Truncated(), out.TruncationReason)
			}
		}()
	}
	wg.Wait()

	// The bound: total spend <= B + (one in-flight model-call * P). Each runner
	// can be mid-turn when the pool tips over the limit, so each may contribute at
	// most one extra perCall beyond the shared ceiling.
	total := pool.Spent()
	upperBound := budget + perCall*int64(parallel)
	if total > upperBound {
		t.Fatalf("overshoot exceeded bound: spent=%d, bound=B(%d)+P*perCall(%d)=%d",
			total, budget, perCall*int64(parallel), upperBound)
	}
	// Sanity: the pool must actually have been driven past the budget (otherwise
	// the bound is trivially satisfied by under-spending).
	if total < budget {
		t.Fatalf("pool never reached budget: spent=%d < B=%d", total, budget)
	}
	t.Logf("spent=%d within bound %d (budget=%d, overshoot=%d, %.2fx)",
		total, upperBound, budget, total-budget, float64(total)/float64(budget))
}

// TestBudgetPool_StopsInFlight verifies a single in-flight runner stops at the
// next turn boundary once the pool is exhausted, bounding its own overshoot to
// one model-call.
func TestBudgetPool_StopsInFlight(t *testing.T) {
	const budget int64 = 500_000
	const perCall int64 = 60_000
	pool := NewBudgetPool(budget)

	client := &bigSpendClient{perCall: perCall}
	r := NewRunner(client, []Tool{noopTool{}}, "sys",
		WithLimits(Limits{MaxIterations: -1, TokenBudget: -1}), WithBudgetPool(pool))
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.TruncationReason != TruncBudgetPool {
		t.Fatalf("reason = %q, want %q", out.TruncationReason, TruncBudgetPool)
	}
	// A single runner overshoots by at most one perCall: it issues calls until the
	// pre-turn check sees the pool at/over budget, which happens at most one call
	// after crossing.
	if got, want := pool.Spent(), budget+perCall; got > want {
		t.Fatalf("single-runner overshoot = %d, want <= %d", got, want)
	}
}

// TestBudgetPool_RunnerChargesPerCompletion pins the charge side of
// WithBudgetPool: the Runner adds every successful completion's
// ChargeableTokens (CacheReadWeight-discounted) to the pool, so Spent() equals
// the summed chargeable usage. The Runner folds it directly; no Observer
// is involved. Cache reads are discounted by the configured weight.
func TestBudgetPool_RunnerChargesPerCompletion(t *testing.T) {
	// Turn 1: 100 in (40 from cache), 10 out. Turn 2: 50 in (0 cache), 5 out.
	// Weight 0.5: chargeable = (100-40)*1 + 40*0.5 + 10 + 50 + 5 = 60+20+65 = 145.
	fc := newFakeClient(
		withCache(toolResp("c1", "noop", `{}`, 100, 10), 40, 0),
		withCache(textResp("done", 50, 5), 0, 0),
	)
	pool := NewBudgetPool(1_000_000)
	r := NewRunner(fc, []Tool{noopTool{}}, "sys",
		WithLimits(Limits{TokenBudget: -1, CacheReadWeight: 0.5}), WithBudgetPool(pool))
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Truncated() {
		t.Fatalf("unexpected truncation: %s", out.TruncationReason)
	}
	want := int64(145)
	if got := pool.Spent(); got != want {
		t.Fatalf("pool.Spent() = %d, want %d (sum of ChargeableTokens at weight 0.5)", got, want)
	}
	// The charge is exactly the chargeable usage the outcome reports.
	if got := out.Usage.ChargeableTokens(0.5); got != want {
		t.Fatalf("outcome chargeable = %d, want %d", got, want)
	}
}

// TestBudgetPool_ExhaustedStopsWithoutRecorder verifies a pool installed
// with WithBudgetPool stops the run with TruncBudgetPool once exhausted.
// Only the Runner charges the pool; no Observer participates.
func TestBudgetPool_ExhaustedStopsWithoutRecorder(t *testing.T) {
	pool := NewBudgetPool(100)
	// bigSpendClient never charges: only the Runner does.
	client := &bigSpendClient{perCall: 60}
	r := NewRunner(client, []Tool{noopTool{}}, "sys",
		WithLimits(Limits{MaxIterations: -1, TokenBudget: -1}), WithBudgetPool(pool))
	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.TruncationReason != TruncBudgetPool {
		t.Fatalf("reason = %q, want %q", out.TruncationReason, TruncBudgetPool)
	}
	// 60 per turn: two turns charge 120 >= 100, the pre-turn gate fires before
	// the third completion.
	if client.callCount.Load() != 2 {
		t.Fatalf("completions = %d, want 2 (third is gated by the exhausted pool)", client.callCount.Load())
	}
	if got := pool.Spent(); got != 120 {
		t.Fatalf("pool.Spent() = %d, want 120 (two charged turns)", got)
	}
}

// TestBudgetPool_Unlimited confirms a nil *BudgetPool is the unlimited
// representation: Check always passes and Remaining returns math.MaxInt64.
func TestBudgetPool_Unlimited(t *testing.T) {
	var pool *BudgetPool
	if err := pool.Check(); err != nil {
		t.Fatalf("nil pool Check = %v, want nil", err)
	}
	pool.Add(1 << 40) // no-op on nil; must not panic
	if err := pool.Check(); err != nil {
		t.Fatalf("nil pool Check after Add = %v, want nil", err)
	}
	if rem := pool.Remaining(); rem <= 0 {
		t.Fatalf("nil pool Remaining = %d, want math.MaxInt64", rem)
	}
}

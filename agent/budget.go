package agent

import (
	"errors"
	"math"
	"sync/atomic"
)

// ErrBudgetExhausted is returned by BudgetPool.Check once the pool's
// cumulative spend has reached its limit. The [Runner] treats it as a limit
// stop (TruncBudgetPool, reported by Run as an [IncompleteError]), not an
// infrastructure error.
var ErrBudgetExhausted = errors.New("agent: shared budget pool exhausted")

// BudgetPool is a concurrency-safe token budget shared across many
// concurrent [Runner] runs. A Runner installed with [WithBudgetPool] checks
// the pool once per main-loop turn, before the turn's first completion, and
// charges it after every successful completion with that completion's
// chargeable tokens. Once a check finds the pool exhausted, a run already
// in flight starts no further main-loop turn, even with its own per-run
// allowance left.
//
// The pool gates the start of main-loop turns. It does not cap spend.
// Three kinds of completion run without a fresh check and still charge the
// pool:
//
//   - the max-tokens continuation of a completion that stopped at the
//     output cap (a capped 5/5 completion and its 5/5 continuation charge
//     20 tokens to a pool with 3 tokens of headroom);
//   - the forced-finalization turn that RunJSON takes when the iteration
//     cap, the per-run token budget, or the pool stops the run, with its
//     own continuation;
//   - the RunJSON repair completion (a run that stopped with
//     TruncTokenBudget or TruncBudgetPool skips it).
//
// The pool tracks cumulative spend (input+output tokens, cache-read
// discounted) against a fixed limit. A nil *BudgetPool is the canonical
// unlimited representation: Check always returns nil and Remaining returns
// math.MaxInt64. NewBudgetPool requires a positive limit; callers that want
// an unlimited pool should hold a nil pointer.
//
// All methods are safe for concurrent use.
type BudgetPool struct {
	limit int64
	spent atomic.Int64
}

// NewBudgetPool returns a pool whose Check fails once cumulative spend
// reaches limit tokens.
// limit must be positive; use a nil *BudgetPool to represent an unlimited pool.
func NewBudgetPool(limit int64) *BudgetPool {
	if limit <= 0 {
		panic("agent.NewBudgetPool: limit must be positive; use a nil *BudgetPool for unlimited")
	}
	return &BudgetPool{limit: limit}
}

// Add records tokens spent against the pool. Non-positive deltas are ignored.
func (p *BudgetPool) Add(tokens int64) {
	if p == nil || tokens <= 0 {
		return
	}
	p.spent.Add(tokens)
}

// Check reports whether cumulative spend is still below the limit. It
// returns ErrBudgetExhausted once cumulative spend has reached the limit, and
// nil otherwise. A nil pool is unlimited: Check always returns nil.
func (p *BudgetPool) Check() error {
	if p == nil {
		return nil
	}
	if p.spent.Load() >= p.limit {
		return ErrBudgetExhausted
	}
	return nil
}

// Remaining returns the tokens left before the pool is exhausted, clamped at
// zero. A nil pool is unlimited: Remaining returns math.MaxInt64 so callers
// deriving per-run allowances treat it as "plenty". Safe for concurrent use.
func (p *BudgetPool) Remaining() int64 {
	if p == nil {
		return math.MaxInt64
	}
	rem := p.limit - p.spent.Load()
	if rem < 0 {
		return 0
	}
	return rem
}

// Spent returns cumulative tokens recorded against the pool so far.
func (p *BudgetPool) Spent() int64 {
	if p == nil {
		return 0
	}
	return p.spent.Load()
}

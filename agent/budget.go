package agent

import (
	"errors"
	"math"
	"sync/atomic"
)

// ErrBudgetExhausted is returned by BudgetPool.Check once the pool has no
// headroom left for another model call. The [Runner] treats it as a clean
// stop (TruncBudgetPool), not an infrastructure error.
var ErrBudgetExhausted = errors.New("agent: shared budget pool exhausted")

// BudgetPool is a concurrency-safe token budget shared across many
// concurrent [Runner] runs. A Runner installed with [WithBudgetPool] checks
// the pool once per main-loop turn and charges it after every successful
// completion with that completion's chargeable tokens. A run already in
// flight stops at the next turn boundary once the pool is exhausted,
// rather than running to completion under its own per-run allowance. This
// bounds total charged overshoot to at most one in-flight model call per
// concurrent runner.
// Note a RunJSON repair pass and a run
// stopped mid-turn still spend real provider tokens that are gated pre-turn
// but can land after the ceiling is crossed, so real-dollar overshoot can
// modestly exceed the charged bound.
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

// NewBudgetPool returns a pool bounding cumulative spend to limit tokens.
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

// Check reports whether the pool still has headroom for another model call. It
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

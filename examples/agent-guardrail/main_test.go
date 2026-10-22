package main

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/decide"
)

const (
	llmSpend     = 100 // tokens per completion (S)
	decideSpend  = 100 // tokens per Ask (D)
	toolTurns    = 2   // completions that request a tool; one more completes the run
	totalTurns   = toolTurns + 1
	poolLimit    = 350 // llmOnlySpend < L, and the decide spend pushes the pool past it before the last turn
	llmOnlySpend = llmSpend * totalTurns
)

// turnClient requests one read_file call for toolTurns completions, then
// answers; every completion spends llmSpend tokens.
type turnClient struct {
	mu sync.Mutex
	n  int
}

func (c *turnClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *turnClient) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.n++
	usage := llmkit.Usage{InputTokens: llmSpend - 40, OutputTokens: 40}
	if c.n > toolTurns {
		return llmkit.Response{Text: "done", StopReason: llmkit.StopEndTurn, Usage: usage}, nil
	}
	return llmkit.Response{
		StopReason: llmkit.StopToolUse,
		Usage:      usage,
		ToolCalls:  []llmkit.ToolCall{{ID: "c" + string(rune('0'+c.n)), Name: "read_file", Arguments: []byte(`{"path":"notes.txt"}`)}},
	}, nil
}

// benignDecider answers every question as safe and on-topic and reports
// spend input tokens per Ask.
type benignDecider struct{ spend int64 }

func (d benignDecider) Ask(_ context.Context, _ any, qs decide.Questions) (decide.Response, error) {
	resp := decide.Response{Usage: llmkit.Usage{InputTokens: d.spend}}
	for id, q := range qs {
		switch q.(type) {
		case decide.Noul:
			if resp.Nouls == nil {
				resp.Nouls = map[string]float64{}
			}
			resp.Nouls[id] = 0.1
		case decide.Choice:
			if resp.Choices == nil {
				resp.Choices = map[string]decide.ChoiceAnswer{}
			}
			resp.Choices[id] = decide.ChoiceAnswer{Choice: "on_topic"}
		}
	}
	return resp, nil
}

func runGuardrail(t *testing.T, d decide.Decider) (*agent.Outcome, error) {
	t.Helper()
	pool := agent.NewBudgetPool(poolLimit)
	ws := newWorkspace(io.Discard)
	runner := newRunner(&turnClient{}, d, ws.tools(), io.Discard, pool, defaultThreshold)
	return runner.Run(context.Background(), "read notes.txt")
}

func TestGuardrail_DecideSpendStopsRunAtSharedPoolLimit(t *testing.T) {
	_, err := runGuardrail(t, benignDecider{spend: decideSpend})
	var incomplete *agent.IncompleteError
	if !errors.As(err, &incomplete) {
		t.Fatalf("run err = %v, want *agent.IncompleteError", err)
	}
	if got := incomplete.Outcome.TruncationReason; got != agent.TruncBudgetPool {
		t.Fatalf("TruncationReason = %q, want %q", got, agent.TruncBudgetPool)
	}
}

func TestGuardrail_LLMSpendAloneStaysUnderPoolLimit(t *testing.T) {
	if llmOnlySpend >= poolLimit {
		t.Fatalf("test setup: LLM-only spend %d must be under the limit %d", llmOnlySpend, poolLimit)
	}
	out, err := runGuardrail(t, benignDecider{spend: 0})
	if err != nil {
		t.Fatalf("run with free decisions: %v", err)
	}
	if out.TruncationReason != "" || out.FinalText != "done" {
		t.Fatalf("outcome = reason %q text %q, want an untruncated run answering %q", out.TruncationReason, out.FinalText, "done")
	}
}

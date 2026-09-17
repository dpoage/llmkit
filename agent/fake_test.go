package agent

import (
	"context"
	"sync"

	"github.com/dpoage/llmkit"
)

// scriptStep is one programmed turn of a fakeClient: the response to return,
// or an error to surface (an infra failure).
type scriptStep struct {
	resp llmkit.Response
	err  error
}

// fakeClient is a scripted llmkit.Client for testing the loop. It returns each
// scripted step in order and records every request it received.
type fakeClient struct {
	mu       sync.Mutex
	steps    []scriptStep
	idx      int
	requests []llmkit.Request
	caps     llmkit.Capabilities
}

func newFakeClient(steps ...scriptStep) *fakeClient {
	return &fakeClient{steps: steps}
}

func (f *fakeClient) Capabilities() llmkit.Capabilities { return f.caps }

func (f *fakeClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, req)
	if f.idx >= len(f.steps) {
		// Default to a benign end-turn so over-running tests fail on assertions,
		// not panics.
		return llmkit.Response{Text: "(unscripted)", StopReason: llmkit.StopEndTurn}, nil
	}
	step := f.steps[f.idx]
	f.idx++
	return step.resp, step.err
}

func (f *fakeClient) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.idx
}

// --- response builders ----------------------------------------------------

// textResp builds an end-turn text response with the given usage.
func textResp(text string, in, out int64) scriptStep {
	return scriptStep{resp: llmkit.Response{
		Text:       text,
		StopReason: llmkit.StopEndTurn,
		Usage:      llmkit.Usage{InputTokens: in, OutputTokens: out},
	}}
}

// maxTokensResp builds a text response that stopped at the output token cap
// (StopMaxTokens), e.g. a JSON answer cut off mid-object.
func maxTokensResp(text string, in, out int64) scriptStep {
	return scriptStep{resp: llmkit.Response{
		Text:       text,
		StopReason: llmkit.StopMaxTokens,
		Usage:      llmkit.Usage{InputTokens: in, OutputTokens: out},
	}}
}

// toolResp builds a tool-use response requesting a single tool call.
func toolResp(id, name, args string, in, out int64) scriptStep {
	return scriptStep{resp: llmkit.Response{
		StopReason: llmkit.StopToolUse,
		ToolCalls: []llmkit.ToolCall{{
			ID:        id,
			Name:      name,
			Arguments: []byte(args),
		}},
		Usage: llmkit.Usage{InputTokens: in, OutputTokens: out},
	}}
}

// thinkOnlyResp builds an end-turn response whose entire content is an inline
// <think>...</think> block and no tool calls — the observed production
// failure mode (MiniMax-M3 sometimes narrates a tool call inside the think
// block it never actually issues).
func thinkOnlyResp(think string, in, out int64) scriptStep {
	return scriptStep{resp: llmkit.Response{
		Text:       "<think>" + think + "</think>",
		StopReason: llmkit.StopEndTurn,
		Usage:      llmkit.Usage{InputTokens: in, OutputTokens: out},
	}}
}

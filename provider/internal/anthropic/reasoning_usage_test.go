package anthropic

import (
	"context"
	"testing"

	llmkit "github.com/dpoage/llmkit"
)

// Anthropic reports no reasoning count: a thinking response keeps ReasoningTokens 0.
func TestReasoningTokens_AnthropicStaysZero(t *testing.T) {
	want := llmkit.Usage{InputTokens: 18, OutputTokens: 42, CacheReadInputTokens: 4, CacheCreationInputTokens: 3}

	cl := newStreamAdapter(t, newServer(t, jsonHandler(reasoningBody)))
	resp, err := cl.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage != want {
		t.Errorf("Complete Usage = %+v, want %+v", resp.Usage, want)
	}

	cl = newStreamAdapter(t, newServer(t, sseHandler(streamReasoningEvents())))
	resp, err = cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage != want {
		t.Errorf("Stream Usage = %+v, want %+v", resp.Usage, want)
	}
}

package anthropic

import (
	"context"
	"testing"

	llmkit "github.com/dpoage/llmkit"
)

func TestUsageNeverEstimated(t *testing.T) {
	want := llmkit.Usage{InputTokens: 18, OutputTokens: 42, CacheReadInputTokens: 4, CacheCreationInputTokens: 3}
	check := func(t *testing.T, resp llmkit.Response, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage.Estimated || resp.Usage != want {
			t.Errorf("Usage = %+v, want %+v with Estimated false", resp.Usage, want)
		}
	}
	t.Run("complete", func(t *testing.T) {
		cl := newStreamAdapter(t, newServer(t, jsonHandler(reasoningBody)))
		resp, err := cl.Complete(context.Background(), simpleRequest())
		check(t, resp, err)
	})
	t.Run("stream fn", func(t *testing.T) {
		cl := newStreamAdapter(t, newServer(t, sseHandler(streamReasoningEvents())))
		resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
		check(t, resp, err)
	})
	t.Run("stream nil fn", func(t *testing.T) {
		cl := newStreamAdapter(t, newServer(t, sseHandler(streamReasoningEvents())))
		resp, err := cl.Stream(context.Background(), simpleRequest(), nil)
		check(t, resp, err)
	})
}

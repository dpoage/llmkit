package google

import (
	"context"
	"testing"

	"github.com/dpoage/llmkit"
)

func thinkingBody(usage map[string]any) map[string]any {
	return map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": []any{map[string]any{"text": "hi"}}},
			"finishReason": "STOP",
		}},
		"usageMetadata": usage,
	}
}

func TestReasoningTokens_AllPaths(t *testing.T) {
	withThoughts := map[string]any{
		"promptTokenCount": 11, "candidatesTokenCount": 50, "thoughtsTokenCount": 2000, "totalTokenCount": 2061,
	}
	want := llmkit.Usage{InputTokens: 11, OutputTokens: 2050, ReasoningTokens: 2000}

	t.Run("complete", func(t *testing.T) {
		cl := newStreamClient(t, completeServer(t, thinkingBody(withThoughts)))
		resp, err := cl.Complete(context.Background(), simpleRequest())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != want {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
		}
	})
	t.Run("stream with fn", func(t *testing.T) {
		cl := newStreamClient(t, streamServer(t, textChunk("hi"), finalChunk("STOP", withThoughts)))
		resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != want {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
		}
	})
	t.Run("stream nil fn", func(t *testing.T) {
		cl := newStreamClient(t, streamServer(t, textChunk("hi"), finalChunk("STOP", withThoughts)))
		resp, err := cl.Stream(context.Background(), simpleRequest(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != want {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
		}
	})
	t.Run("no thoughtsTokenCount unchanged", func(t *testing.T) {
		plain := map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 50, "totalTokenCount": 61}
		cl := newStreamClient(t, completeServer(t, thinkingBody(plain)))
		resp, err := cl.Complete(context.Background(), simpleRequest())
		if err != nil {
			t.Fatal(err)
		}
		if w := (llmkit.Usage{InputTokens: 11, OutputTokens: 50}); resp.Usage != w {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, w)
		}
	})
}

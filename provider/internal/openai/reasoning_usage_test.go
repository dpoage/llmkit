package openai

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

const reasoningUsage = `{"prompt_tokens":10,"completion_tokens":100,"total_tokens":110,` +
	`"prompt_tokens_details":{"cached_tokens":20},` +
	`"completion_tokens_details":{"reasoning_tokens":64}}`

var wantReasoningUsage = llmkit.Usage{
	InputTokens: 10, OutputTokens: 100, CacheReadInputTokens: 20, ReasoningTokens: 64,
}

func TestReasoningTokens_AllPaths(t *testing.T) {
	complete := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","created":1,"model":%q,`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":%s}`,
			streamID, streamModel, reasoningUsage)
	}
	chunks := []string{
		chunkJSON(`{"role":"assistant","content":"hi"}`, "", ""),
		chunkJSON(`{}`, "stop", ""),
		usageChunkJSON(reasoningUsage),
	}
	noReasoning := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("hi", 10, 100)))
	}

	t.Run("complete", func(t *testing.T) {
		client, err := New(streamModel, Options{APIKey: "k", BaseURL: newServer(t, complete)})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Complete(context.Background(), simpleRequest())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != wantReasoningUsage {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, wantReasoningUsage)
		}
	})
	t.Run("stream with fn", func(t *testing.T) {
		resp, _, err := stream(t, newServer(t, sseHandler(nil, chunks...)), simpleRequest(),
			func(llmkit.Delta) error { return nil })
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != wantReasoningUsage {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, wantReasoningUsage)
		}
	})
	t.Run("stream nil fn", func(t *testing.T) {
		resp, _, err := stream(t, newServer(t, sseHandler(nil, chunks...)), simpleRequest(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != wantReasoningUsage {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, wantReasoningUsage)
		}
	})
	t.Run("absent details stay zero", func(t *testing.T) {
		client, err := New(streamModel, Options{APIKey: "k", BaseURL: newServer(t, noReasoning)})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Complete(context.Background(), simpleRequest())
		if err != nil {
			t.Fatal(err)
		}
		if want := (llmkit.Usage{InputTokens: 10, OutputTokens: 100}); resp.Usage != want {
			t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
		}
	})
}

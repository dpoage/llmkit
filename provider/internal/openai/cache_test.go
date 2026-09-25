package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestOpenAICompatible_NoCachedTokensField covers endpoints (Ollama, vLLM, many
// MiniMax deployments) that omit prompt_tokens_details entirely: usage must
// still parse, with zero cache reads.
func TestOpenAICompatible_NoCachedTokensField(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("hi", 130, 5)))
	})
	client, err := New("llama-test", Options{
		APIKey: "test-key", BaseURL: base, Compatible: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Usage.InputTokens != 130 || resp.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v, want {130 5}", resp.Usage)
	}
	if resp.Usage.CacheReadInputTokens != 0 || resp.Usage.CacheCreationInputTokens != 0 {
		t.Errorf("cache fields = %d/%d, want 0/0 when the endpoint omits them",
			resp.Usage.CacheReadInputTokens, resp.Usage.CacheCreationInputTokens)
	}
}

package embed

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dpoage/llmkit/retry"
)

// TestDimensions_ZeroUntilFirstCall_ThenDetected pins the interface doc
// sentence on Embedder.Dimensions, for both backends: with
// Config.Dimensions 0 it returns 0 before any call and the served
// vector's length after one successful Embed; with Dimensions
// configured, it returns that value before any call.
func TestDimensions_ZeroUntilFirstCall_ThenDetected(t *testing.T) {
	for _, backend := range []Backend{BackendOllama, BackendOpenAICompatible} {
		t.Run(string(backend)+"/auto_detect", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch backend {
				case BackendOllama:
					jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1, 0.2, 0.3}}})
				case BackendOpenAICompatible:
					jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1, 0.2, 0.3}, Index: 0}}})
				}
			}))
			defer srv.Close()

			emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			if got := emb.Dimensions(); got != 0 {
				t.Errorf("Dimensions() before any call = %d, want 0", got)
			}
			if _, err := emb.Embed(context.Background(), "x"); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if got := emb.Dimensions(); got != 3 {
				t.Errorf("Dimensions() after Embed = %d, want 3 (the served vector length)", got)
			}
		})

		t.Run(string(backend)+"/configured", func(t *testing.T) {
			emb, err := New(Config{Backend: backend, Model: "m", URL: "http://localhost:1", Dimensions: 7})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if got := emb.Dimensions(); got != 7 {
				t.Errorf("Dimensions() with configured Dimensions before any call = %d, want 7", got)
			}
		})
	}
}

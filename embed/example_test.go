package embed_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/dpoage/llmkit/embed"
)

// ExampleNewOpenAICompatibleEmbedder points the embedder at an
// OpenAI-compatible endpoint (here an httptest server serving the wire
// shape from the package docs) and reads one vector back.
func ExampleNewOpenAICompatibleEmbedder() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"data": [
				{"embedding": [0.1, 0.2, 0.3], "index": 0}
			],
			"model": "text-embedding-fake"
		}`))
	}))
	defer srv.Close()

	emb, err := embed.NewOpenAICompatibleEmbedder(embed.Config{
		Embedder: "openai-compatible",
		Model:    "text-embedding-fake",
		URL:      srv.URL,
		APIKey:   "test-key",
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	vec, err := emb.Embed(context.Background(), "hello")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(vec))
	fmt.Println(emb.Dimensions())
	// Output:
	// 3
	// 3
}

// ExampleNewOllamaEmbedder points the embedder at a server speaking the
// Ollama /api/embed wire shape (here an httptest server) and reads one
// vector back.
func ExampleNewOllamaEmbedder() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "nomic-embed-text",
			"embeddings": [[0.5, -0.25, 0.75]]
		}`))
	}))
	defer srv.Close()

	emb, err := embed.NewOllamaEmbedder(embed.Config{
		Embedder: "ollama",
		Model:    "nomic-embed-text",
		URL:      srv.URL,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	vec, err := emb.Embed(context.Background(), "hello")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(vec))
	fmt.Println(emb.Dimensions())
	// Output:
	// 3
	// 3
}

// countingEmbedder is a fake embed.Embedder that counts backend calls.
type countingEmbedder struct {
	calls int
}

func (c *countingEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	c.calls++
	return []float32{1, 2, 3}, nil
}

func (c *countingEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		c.calls++
		out[i] = []float32{1, 2, 3}
	}
	return out, nil
}

func (c *countingEmbedder) Dimensions() int   { return 3 }
func (c *countingEmbedder) ModelName() string { return "fake-model" }

// ExampleCachedEmbedder wraps a fake backend and embeds the same text
// twice: the second call is a cache hit, so the backend runs once.
func ExampleCachedEmbedder() {
	backend := &countingEmbedder{}
	cached := embed.NewCachedEmbedder(backend, 0)

	if _, err := cached.Embed(context.Background(), "hello"); err != nil {
		fmt.Println("error:", err)
		return
	}
	if _, err := cached.Embed(context.Background(), "hello"); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("backend calls:", backend.calls)
	// Output: backend calls: 1
}

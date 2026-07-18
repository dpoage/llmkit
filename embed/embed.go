// Package embed provides a pluggable embedding layer.
//
// It defines the Embedder interface and provides implementations for
// Ollama (local-first) and any OpenAI-compatible API. Configuration is
// driven by environment variables, and an optional content-hash cache
// avoids redundant embedding calls.
package embed

import (
	"context"
	"fmt"
)

// Embedder generates vector embeddings from text.
type Embedder interface {
	// Embed returns the embedding vector for a single text string.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch returns embedding vectors for multiple texts.
	// Implementations should batch the request when the backend supports it.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the dimensionality of the embedding vectors
	// produced by the underlying model.
	Dimensions() int

	// ModelName returns the identifier of the model used for embedding.
	ModelName() string
}

// NewEmbedder creates an Embedder from the provided Config.
// It reads cfg.Embedder to select the backend and delegates to the
// appropriate constructor.
func NewEmbedder(cfg Config) (Embedder, error) {
	var emb Embedder
	var err error

	switch cfg.Embedder {
	case "hugot":
		return nil, fmt.Errorf("hugot embedder not bundled in llmkit; implement embed.Embedder in your application")
	case "ollama":
		emb, err = NewOllamaEmbedder(cfg)
	case "openai-compatible":
		emb, err = NewOpenAICompatibleEmbedder(cfg)
	default:
		return nil, fmt.Errorf("unknown embedder type %q: expected \"ollama\" or \"openai-compatible\"", cfg.Embedder)
	}
	if err != nil {
		return nil, err
	}

	if cfg.CacheEnabled {
		emb = NewCachedEmbedder(emb)
	}

	return emb, nil
}

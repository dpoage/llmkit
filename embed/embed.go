// Package embed provides a pluggable embedding layer.
//
// It defines the Embedder interface and provides implementations for
// Ollama (local-first) and any OpenAI-compatible API. Configuration is
// driven by environment variables, and an optional content-hash cache
// avoids redundant embedding calls.
package embed

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Embedder generates vector embeddings from text.
type Embedder interface {
	// Embed returns the embedding vector for a single text string.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch returns embedding vectors for multiple texts.
	//
	// Results are index-aligned with the input: len(result) == len(texts)
	// and result[i] embeds texts[i]. Duplicate texts are not deduplicated —
	// each occurrence is embedded independently. Implementations may split
	// the batch into multiple requests; a failed request fails the whole
	// call (no partial results).
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the dimensionality of the embedding vectors
	// produced by the underlying model.
	Dimensions() int

	// ModelName returns the identifier of the model used for embedding.
	ModelName() string
}

// NewEmbedder creates an Embedder from the provided Config.
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
		emb = NewCachedEmbedder(emb, cfg.CacheSize)
	}

	return emb, nil
}

// batchChunkSize caps remaining at maxBatch when chunking is enabled
// (maxBatch > 0); otherwise the whole remainder goes in one request.
func batchChunkSize(remaining, maxBatch int) int {
	if maxBatch <= 0 || remaining < maxBatch {
		return remaining
	}
	return maxBatch
}

// ErrEmptyVector is returned, wrapped with backend context, when dimension
// auto-detection (Config.Dimensions == 0) encounters a zero-length embedding
// vector. Such a vector carries no dimensional information: accepting it
// would silently hand callers an empty result while leaving the embedder's
// dimensionality undetected. Test with errors.Is.
var ErrEmptyVector = errors.New("empty embedding vector")

// checkDimensions enforces vector-dimension consistency for a converted
// response. expected is cfg.Dimensions when > 0; otherwise it is detected
// from the first vector and recorded under *dims (if still zero). Every
// vector in every call must match, so a later mismatch — configured or
// against the first detected value — is an error.
func checkDimensions(backend string, out [][]float32, dims *int, mu *sync.RWMutex) error {
	mu.Lock()
	defer mu.Unlock()
	expected := *dims
	if expected == 0 && len(out) > 0 {
		if len(out[0]) == 0 {
			return fmt.Errorf("%s: vector at index 0: %w", backend, ErrEmptyVector)
		}
		expected = len(out[0])
		*dims = expected
	}
	for i, v := range out {
		if len(v) != expected {
			return fmt.Errorf("%s: embedding dimension mismatch at index %d: got %d, want %d", backend, i, len(v), expected)
		}
	}
	return nil
}

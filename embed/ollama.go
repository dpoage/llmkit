package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/dpoage/llmkit"
	"io"
	"net/http"
	"strings"
	"sync"
)

// OllamaEmbedder produces embeddings via the Ollama REST API.
//
// Ollama endpoint: POST <baseURL>/api/embed
//
//	Request:  {"model": "...", "input": "..." | ["...", ...]}
//	Response: {"model": "...", "embeddings": [[...]]}
type OllamaEmbedder struct {
	baseURL    string
	model      string
	client     *http.Client
	retry      llmkit.RetryConfig
	maxBatch   int
	dimensions int
	mu         sync.RWMutex // guards dimensions
}

// NewOllamaEmbedder creates an Embedder that talks to a local Ollama instance.
func NewOllamaEmbedder(cfg Config) (*OllamaEmbedder, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("ollama config: %w", err)
	}
	return &OllamaEmbedder{
		baseURL:    strings.TrimRight(cfg.URL, "/"),
		model:      cfg.Model,
		client:     cfg.httpClient(),
		retry:      cfg.retryPolicy(),
		maxBatch:   cfg.MaxBatch,
		dimensions: cfg.Dimensions,
	}, nil
}

// ollamaRequest is the JSON body sent to Ollama /api/embed.
type ollamaRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string or []string
}

// ollamaResponse is the JSON body returned by Ollama /api/embed.
type ollamaResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}

// Embed returns the embedding for a single text.
func (o *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	var out [][]float32
	err := retryDo(ctx, o.retry, func(actx context.Context) error {
		res, err := o.doEmbed(actx, text)
		if err == nil {
			out = res
		}
		return err
	})
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("ollama: empty embeddings response")
	}
	return out[0], nil
}

// EmbedBatch returns embeddings for multiple texts, splitting the input into
// requests of at most MaxBatch texts (no splitting when MaxBatch is zero).
// Results are index-aligned with the input; a failed chunk fails the whole
// call.
func (o *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); {
		n := batchChunkSize(len(texts)-start, o.maxBatch)
		chunk := texts[start : start+n]
		var res [][]float32
		err := retryDo(ctx, o.retry, func(actx context.Context) error {
			r, err := o.doEmbed(actx, chunk)
			if err == nil {
				res = r
			}
			return err
		})
		if err != nil {
			return nil, err
		}
		out = append(out, res...)
		start += n
	}
	return out, nil
}

// Dimensions returns the vector dimensionality. If not configured up front,
// this is detected from the first successful embedding call and may return 0
// until then.
func (o *OllamaEmbedder) Dimensions() int {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.dimensions
}

// ModelName returns the Ollama model identifier.
func (o *OllamaEmbedder) ModelName() string {
	return o.model
}

// doEmbed performs the HTTP call. input can be string or []string.
func (o *OllamaEmbedder) doEmbed(ctx context.Context, input any) ([][]float32, error) {
	body, err := json.Marshal(ollamaRequest{
		Model: o.model,
		Input: input,
	})
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, newStatusError("ollama", resp.StatusCode, resp.Header.Get("Retry-After"), string(respBody))
	}

	var result ollamaResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	// The server must return exactly one embedding per input; a wrong
	// count would silently misattribute vectors to texts.
	expected := 1
	if texts, ok := input.([]string); ok {
		expected = len(texts)
	}
	if len(result.Embeddings) != expected {
		return nil, fmt.Errorf("ollama: expected %d embeddings, got %d", expected, len(result.Embeddings))
	}

	// Convert float64 -> float32, enforce dimension consistency, and
	// auto-detect dimensions from the first non-empty response.
	out := make([][]float32, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		f32 := make([]float32, len(emb))
		for j, v := range emb {
			f32[j] = float32(v)
		}
		out[i] = f32
	}
	if err := checkDimensions("ollama", out, &o.dimensions, &o.mu); err != nil {
		return nil, err
	}

	return out, nil
}

// truncate shortens s to at most n characters for error messages.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

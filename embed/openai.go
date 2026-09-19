package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/retry"
	"io"
	"net/http"
	"strings"
	"sync"
)

// OpenAICompatibleEmbedder produces embeddings by posting to any
// OpenAI-compatible /v1/embeddings endpoint (OpenAI, Azure OpenAI, vLLM,
// LiteLLM, and others).
//
// Endpoint: POST <baseURL>/v1/embeddings
//
//	Request:  {"model": "...", "input": ["...", ...]}
//	Response: {"data": [{"embedding": [...], "index": 0}], "model": "..."}
//
// When Config.APIKey is set, requests carry an "Authorization: Bearer"
// header.
type OpenAICompatibleEmbedder struct {
	baseURL    string
	model      string
	apiKey     string
	client     *http.Client
	retry      llmkit.RetryConfig
	maxBatch   int
	dimensions int
	mu         sync.RWMutex // guards dimensions
}

// NewOpenAICompatibleEmbedder creates an OpenAICompatibleEmbedder that
// posts to cfg.URL. It returns an error when cfg fails Config.Validate.
func NewOpenAICompatibleEmbedder(cfg Config) (*OpenAICompatibleEmbedder, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("openai-compatible config: %w", err)
	}
	return &OpenAICompatibleEmbedder{
		baseURL:    strings.TrimRight(cfg.URL, "/"),
		model:      cfg.Model,
		apiKey:     cfg.APIKey,
		client:     cfg.httpClient(),
		retry:      cfg.retryPolicy(),
		maxBatch:   cfg.MaxBatch,
		dimensions: cfg.Dimensions,
	}, nil
}

// openaiRequest is the JSON body sent to the embeddings endpoint.
type openaiRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// openaiResponse is the JSON body returned by the embeddings endpoint.
type openaiResponse struct {
	Data  []openaiEmbedding `json:"data"`
	Model string            `json:"model"`
	Error *openaiError      `json:"error,omitempty"`
}

type openaiEmbedding struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type openaiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// Embed returns the embedding for a single text.
func (o *OpenAICompatibleEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	var out []float32
	err := retry.Do(ctx, o.retry, retryable, func(actx context.Context) error {
		res, err := o.doEmbed(actx, []string{text})
		if err == nil {
			out = res[0]
		}
		return err
	})
	if err != nil {
		return nil, err
	}
	if out == nil {
		return nil, fmt.Errorf("openai-compatible: empty embeddings response")
	}
	return out, nil
}

// EmbedBatch returns embeddings for multiple texts, splitting the input into
// requests of at most MaxBatch texts (no splitting when MaxBatch is zero).
// Results are index-aligned with the input; a failed chunk fails the whole
// call.
func (o *OpenAICompatibleEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); {
		n := batchChunkSize(len(texts)-start, o.maxBatch)
		chunk := texts[start : start+n]
		var res [][]float32
		err := retry.Do(ctx, o.retry, retryable, func(actx context.Context) error {
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

// Dimensions returns the vector dimensionality. Like OllamaEmbedder, this
// may be auto-detected from the first response.
func (o *OpenAICompatibleEmbedder) Dimensions() int {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.dimensions
}

// ModelName returns the configured model identifier.
func (o *OpenAICompatibleEmbedder) ModelName() string {
	return o.model
}

// doEmbed performs the HTTP call to the /v1/embeddings endpoint.
func (o *OpenAICompatibleEmbedder) doEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(openaiRequest{
		Model: o.model,
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("openai-compatible: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai-compatible: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.apiKey)
	}

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai-compatible: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai-compatible: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, newStatusError("openai-compatible", resp.StatusCode, resp.Header.Get("Retry-After"), string(respBody))
	}

	var result openaiResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("openai-compatible: decode response: %w", err)
	}
	if result.Error != nil {
		return nil, fmt.Errorf("openai-compatible: API error: %s", result.Error.Message)
	}
	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("openai-compatible: expected %d embeddings, got %d", len(texts), len(result.Data))
	}

	// Response entries may arrive out of order. Build by index, rejecting
	// out-of-range and duplicate entries — together with the count check
	// above this fills every slot exactly once.
	out := make([][]float32, len(texts))
	for _, d := range result.Data {
		if d.Index < 0 || d.Index >= len(texts) {
			return nil, fmt.Errorf("openai-compatible: unexpected index %d for batch size %d", d.Index, len(texts))
		}
		if out[d.Index] != nil {
			return nil, fmt.Errorf("openai-compatible: duplicate embedding at index %d", d.Index)
		}
		f32 := make([]float32, len(d.Embedding))
		for j, v := range d.Embedding {
			f32[j] = float32(v)
		}
		out[d.Index] = f32
	}

	if err := checkDimensions("openai-compatible", out, &o.dimensions, &o.mu); err != nil {
		return nil, err
	}

	return out, nil
}

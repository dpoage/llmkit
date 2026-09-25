package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
	"github.com/dpoage/llmkit/retry"
	"io"
	"net/http"
	"strings"
	"sync"
)

// OllamaEmbedder posts to an Ollama server's REST API.
//
// POST <baseURL>/api/embed
//
//	Request:  {"model": "...", "input": "..." | ["...", ...]}
//	Response: {"model": "...", "embeddings": [[...]]}
type OllamaEmbedder struct {
	baseURL    string
	model      string
	client     *http.Client
	retry      retry.Config
	maxBatch   int
	dimensions int
	mu         sync.RWMutex // guards dimensions
}

// NewOllamaEmbedder builds an embedder that posts to cfg.URL.
// It returns an error when cfg fails [Config.Validate].
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

type ollamaRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string or []string
}

type ollamaResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}

// Embed returns the embedding for a single text.
func (o *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	var out [][]float32
	err := retry.Do(ctx, o.retry, llmkit.Classify, func(actx context.Context) error {
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
// requests of at most MaxBatch texts (0 = no splitting). Results are
// index-aligned; a failed chunk fails the whole call.

func (o *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); {
		n := batchChunkSize(len(texts)-start, o.maxBatch)
		chunk := texts[start : start+n]
		var res [][]float32
		err := retry.Do(ctx, o.retry, llmkit.Classify, func(actx context.Context) error {
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

func (o *OllamaEmbedder) ModelName() string {
	return o.model
}

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
		return nil, adapter.TransportError("ollama", ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, adapter.TransportError("ollama", ctx, err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, responseError("ollama", resp, respBody)
	}

	var result ollamaResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	// Server must return one vector per input — a wrong count would silently
	// misattribute vectors to texts.
	expected := 1
	if texts, ok := input.([]string); ok {
		expected = len(texts)
	}
	if len(result.Embeddings) != expected {
		return nil, fmt.Errorf("ollama: expected %d embeddings, got %d", expected, len(result.Embeddings))
	}

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

// responseError normalizes a non-200 response into the kit's vocabulary.
// [adapter.NormalizeSDKError] classifies the status against the full
// trimmed body (a 400's context-length phrase can sit past any cap);
// only the returned APIError's Message is capped at 200 bytes.
func responseError(backend string, resp *http.Response, body []byte) error {
	err := adapter.NormalizeSDKError(backend, adapter.VendorError{
		Status:  resp.StatusCode,
		Message: strings.TrimSpace(string(body)),
		Header:  resp.Header,
	})
	if apiErr, ok := err.(*llmkit.APIError); ok {
		apiErr.Message = truncate(apiErr.Message, 200)
	}
	return err
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

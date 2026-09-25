package embed

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/dpoage/llmkit/internal/adapter"
)

// openaiCodec is the [wireCodec] for [BackendOpenAICompatible]: any
// OpenAI-compatible /v1/embeddings endpoint (OpenAI, Azure OpenAI, vLLM,
// LiteLLM).
//
// POST <baseURL>/v1/embeddings
//
//	Request:  {"model": "...", "input": ["...", ...]}
//	Response: {"data": [{"embedding": [...], "index": 0}], "model": "..."}
//
// A 200 response whose body carries an "error" object is reported as an
// in-band vendor error rather than decoded as data.
type openaiCodec struct{}

var _ wireCodec = openaiCodec{}

func (openaiCodec) path() string { return "/v1/embeddings" }

func (openaiCodec) encodeRequest(model string, texts []string) any {
	return openaiRequest{Model: model, Input: texts}
}

func (openaiCodec) decodeResponse(body []byte, want int) ([][]float32, *adapter.VendorError, error) {
	var result openaiResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, nil, fmt.Errorf("decode response: %w", err)
	}
	if result.Error != nil {
		return nil, &adapter.VendorError{
			Status:  http.StatusOK,
			Type:    result.Error.Type,
			Message: result.Error.Message,
		}, nil
	}
	if len(result.Data) != want {
		return nil, nil, fmt.Errorf("expected %d embeddings, got %d", want, len(result.Data))
	}

	// Response entries may arrive out of order; build by index,
	// rejecting out-of-range and duplicate entries. Combined with the
	// count check above, this fills every slot exactly once.
	out := make([][]float32, want)
	for _, d := range result.Data {
		if d.Index < 0 || d.Index >= want {
			return nil, nil, fmt.Errorf("unexpected index %d for batch size %d", d.Index, want)
		}
		if out[d.Index] != nil {
			return nil, nil, fmt.Errorf("duplicate embedding at index %d", d.Index)
		}
		f32 := make([]float32, len(d.Embedding))
		for j, v := range d.Embedding {
			f32[j] = float32(v)
		}
		out[d.Index] = f32
	}
	return out, nil, nil
}

type openaiRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

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

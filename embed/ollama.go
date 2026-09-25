package embed

import (
	"encoding/json"
	"fmt"

	"github.com/dpoage/llmkit/internal/adapter"
)

// ollamaCodec is the [wireCodec] for [BackendOllama].
//
// POST <baseURL>/api/embed
//
//	Request:  {"model": "...", "input": ["...", ...]}
//	Response: {"model": "...", "embeddings": [[...]]}
//
// Ollama's API accepts "input" as a single string or a list; this codec
// always sends a list, including for a single text.
type ollamaCodec struct{}

var _ wireCodec = ollamaCodec{}

func (ollamaCodec) path() string { return "/api/embed" }

func (ollamaCodec) encodeRequest(model string, texts []string) any {
	return ollamaRequest{Model: model, Input: texts}
}

func (ollamaCodec) decodeResponse(body []byte, want int) ([][]float32, *adapter.VendorError, error) {
	var result ollamaResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, nil, fmt.Errorf("decode response: %w", err)
	}
	// Server must return one vector per input — a wrong count would silently
	// misattribute vectors to texts.
	if len(result.Embeddings) != want {
		return nil, nil, fmt.Errorf("expected %d embeddings, got %d", want, len(result.Embeddings))
	}

	out := make([][]float32, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		f32 := make([]float32, len(emb))
		for j, v := range emb {
			f32[j] = float32(v)
		}
		out[i] = f32
	}
	return out, nil, nil
}

type ollamaRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}

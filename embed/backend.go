package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
	"github.com/dpoage/llmkit/retry"
)

// maxResponseBytes bounds every response body this package reads, on
// every route and every status. It is a var, not a const, so a test can
// lower it. See the package doc's "# Errors" section for what exceeding
// it returns.
var maxResponseBytes int64 = 64 << 20

// wireCodec is what a backend contributes to the shared embed loop: the
// endpoint path, the request body shape, and how a 200 response decodes
// (including an in-band error object when the wire carries one — only
// openai-compatible does). Everything else (retrying, batching,
// dimension tracking, the body-size bound, and error classification) is
// shared by [backendEmbedder].
type wireCodec interface {
	path() string
	encodeRequest(model string, texts []string) any
	// decodeResponse parses a 200 response body for want texts. A
	// non-nil vendorErr means the body carries an in-band error object;
	// the shared loop classifies it through the same trim-and-cap helper
	// as a non-200 response ([classifyVendorError]). vectors and
	// vendorErr are never both set.
	decodeResponse(body []byte, want int) (vectors [][]float32, vendorErr *adapter.VendorError, err error)
}

// backendEmbedder is the Embedder every wireCodec drives. It owns the
// one retry/batch/dimension loop; a codec supplies only the wire shape.
type backendEmbedder struct {
	name     string // also the backend tag in error text
	baseURL  string
	model    string
	apiKey   string
	client   *http.Client
	retry    retry.Config
	maxBatch int
	codec    wireCodec

	mu   sync.RWMutex // guards dims
	dims int
}

var _ Embedder = (*backendEmbedder)(nil)

func newBackendEmbedder(cfg Config, codec wireCodec) *backendEmbedder {
	return &backendEmbedder{
		name:     string(cfg.Backend),
		baseURL:  strings.TrimRight(cfg.URL, "/"),
		model:    cfg.Model,
		apiKey:   cfg.APIKey,
		client:   cfg.httpClient(),
		retry:    cfg.retryPolicy(),
		maxBatch: cfg.MaxBatch,
		codec:    codec,
		dims:     cfg.Dimensions,
	}
}

// Embed returns the embedding for a single text. Defined as
// EmbedBatch([]string{text})[0]: same vector, same error, for every
// backend.
func (b *backendEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	out, err := b.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// EmbedBatch returns embeddings for multiple texts, splitting the input
// into requests of at most maxBatch texts (0 = no splitting). Results
// are index-aligned; a failed chunk fails the whole call.
func (b *backendEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); {
		n := batchChunkSize(len(texts)-start, b.maxBatch)
		chunk := texts[start : start+n]
		var res [][]float32
		err := retry.Do(ctx, b.retry, llmkit.Classify, func(actx context.Context) error {
			r, err := b.doEmbed(actx, chunk)
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

// Dimensions returns the vector dimensionality: cfg.Dimensions when set,
// otherwise the length checkDimensions recorded from the first vector it
// accepted (0 until then), even when that call failed later.
func (b *backendEmbedder) Dimensions() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.dims
}

func (b *backendEmbedder) ModelName() string {
	return b.model
}

// doEmbed sends one request for texts and returns the decoded vectors.
// Every response body — success or failure, any status — is read through
// an io.LimitReader bounded by maxResponseBytes+1, so a hostile or
// misbehaving server can never force more than that many bytes into
// memory.
func (b *backendEmbedder) doEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(b.codec.encodeRequest(b.model, texts))
	if err != nil {
		return nil, fmt.Errorf("%s: marshal request: %w", b.name, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.baseURL+b.codec.path(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: create request: %w", b.name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	if b.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.apiKey)
	}

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, adapter.TransportError(b.name, ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		return nil, adapter.TransportError(b.name, ctx, err)
	}

	if int64(len(respBody)) > maxResponseBytes {
		if resp.StatusCode == http.StatusOK {
			return nil, fmt.Errorf("%s: response body exceeds the %d-byte limit", b.name, maxResponseBytes)
		}
		// Non-200: classify by status from the truncated body; the
		// 200-byte Message cap in classifyVendorError still applies on top.
		respBody = respBody[:maxResponseBytes]
	}

	if resp.StatusCode != http.StatusOK {
		return nil, responseError(b.name, resp, respBody)
	}

	vectors, vendorErr, err := b.codec.decodeResponse(respBody, len(texts))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", b.name, err)
	}
	if vendorErr != nil {
		vendorErr.Header = resp.Header
		return nil, classifyVendorError(b.name, *vendorErr)
	}

	if err := checkDimensions(b.name, vectors, &b.dims, &b.mu); err != nil {
		return nil, err
	}
	return vectors, nil
}

// classifyVendorError normalizes v into the kit's error vocabulary and
// caps the resulting APIError's Message at 200 bytes plus "..." after
// trimming surrounding whitespace. Both APIError routes built from
// server-supplied text — a non-200 response ([responseError]) and the
// openai-compatible 200 error-object route — call this one helper, so
// the trim-and-cap rule cannot drift between them.
// [adapter.NormalizeSDKError] classifies against the whole trimmed
// v.Message, which doEmbed has already cut to maxResponseBytes — only
// the returned Message is capped.
func classifyVendorError(backend string, v adapter.VendorError) error {
	v.Message = strings.TrimSpace(v.Message)
	err := adapter.NormalizeSDKError(backend, v)
	if apiErr, ok := err.(*llmkit.APIError); ok {
		apiErr.Message = truncate(apiErr.Message, 200)
	}
	return err
}

// responseError normalizes a non-200 response into the kit's vocabulary
// through classifyVendorError.
func responseError(backend string, resp *http.Response, body []byte) error {
	return classifyVendorError(backend, adapter.VendorError{
		Status:  resp.StatusCode,
		Message: string(body),
		Header:  resp.Header,
	})
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

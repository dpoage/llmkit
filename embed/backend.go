package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
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
	// the shared loop classifies it through the same helper as a non-200
	// response ([adapter.NormalizeCapped], which trims and delegates the
	// 200-byte cap to [adapter.NormalizeSDKError]). vectors and
	// vendorErr are never both set.
	decodeResponse(body []byte, want int) (vectors [][]float32, vendorErr *adapter.VendorError, err error)
}

// backendCodecs is the one Backend-to-codec mapping. It is the accepted
// set of [ParseBackend] and [Config.Validate] and the codec source of
// [New]; the order is the order of ParseBackend's error text. A Backend
// added here needs its codec in the same row, so no accepted Backend can
// reach the embed loop without one.
var backendCodecs = []struct {
	backend Backend
	codec   wireCodec
}{
	{BackendOllama, ollamaCodec{}},
	{BackendOpenAICompatible, openaiCodec{}},
}

// codecFor returns the codec of the backend named s. The error names s
// and lists every accepted backend, so a bad config value is actionable
// without reading this package.
func codecFor(s string) (wireCodec, error) {
	names := make([]string, len(backendCodecs))
	for i, e := range backendCodecs {
		if string(e.backend) == s {
			return e.codec, nil
		}
		names[i] = strconv.Quote(string(e.backend))
	}
	return nil, fmt.Errorf("embed: unknown backend %q: expected %s", s, joinOr(names))
}

// joinOr renders names as `"a"`, `"a" or "b"`, or `"a", "b" or "c"`.
func joinOr(names []string) string {
	if len(names) < 2 {
		return strings.Join(names, "")
	}
	return strings.Join(names[:len(names)-1], ", ") + " or " + names[len(names)-1]
}

// backendEmbedder is the Embedder every wireCodec drives. It owns the
// one retry/batch/dimension loop; a codec supplies only the wire shape.
type backendEmbedder struct {
	name     string   // also the backend tag in error text
	endpoint *url.URL // BaseURL joined with the codec's path; never mutated
	model    string
	apiKey   string
	redact   *adapter.Redactor // masks echoes of apiKey and the BaseURL password, and BaseURL userinfo, in returned errors
	client   *http.Client
	retry    retry.Config
	maxBatch int
	codec    wireCodec

	mu   sync.RWMutex // guards dims
	dims int
}

var _ Embedder = (*backendEmbedder)(nil)

// newBackendEmbedder builds the embedder for a validated Config: codec
// and root are what [Config.validate] returned for cfg.
func newBackendEmbedder(cfg Config, codec wireCodec, root *url.URL) *backendEmbedder {
	return &backendEmbedder{
		name:     string(cfg.Backend),
		endpoint: adapter.JoinEndpoint(root, codec.path()),
		model:    cfg.Model,
		apiKey:   cfg.Secret,
		redact:   adapter.NewRedactor(cfg.Secret, root),
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
			if err != nil {
				// Scrub every attempt's error before retry.Do, the
				// observer or the caller sees it.
				return b.redact.Scrub(err)
			}
			res = r
			return nil
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

	req, err := adapter.NewRequest(ctx, http.MethodPost, b.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: create request: %w", b.name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	if b.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.apiKey)
	}

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, adapter.DoError(b.name, ctx, resp, err)
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
		// 200-byte Message cap in [adapter.NormalizeSDKError] still applies on top.
		respBody = respBody[:maxResponseBytes]
	}

	if resp.StatusCode != http.StatusOK {
		return nil, b.responseError(resp, respBody)
	}

	vectors, vendorErr, err := b.codec.decodeResponse(respBody, len(texts))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", b.name, err)
	}
	if vendorErr != nil {
		vendorErr.Header = resp.Header
		vendorErr.Redact = b.redact
		return nil, adapter.NormalizeCapped(b.name, *vendorErr)
	}

	if err := checkDimensions(b.name, vectors, &b.dims, &b.mu); err != nil {
		return nil, err
	}
	return vectors, nil
}

// responseError normalizes a non-200 response into the kit's vocabulary.
// Both APIError routes built from server-supplied text — this one and the
// openai-compatible 200 error-object route — call
// [adapter.NormalizeCapped] with the embedder's Redactor, so the
// trim-redact-cap rule cannot drift between them. It classifies against
// the whole trimmed body, which doEmbed has already cut to
// maxResponseBytes; only the returned Message is redacted and capped.
func (b *backendEmbedder) responseError(resp *http.Response, body []byte) error {
	return adapter.NormalizeCapped(b.name, adapter.VendorError{
		Status:  resp.StatusCode,
		Message: string(body),
		Header:  resp.Header,
		Redact:  b.redact,
	})
}

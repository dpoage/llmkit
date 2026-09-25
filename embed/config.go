package embed

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

const (
	// defaultEmbedMaxAttempts is the embed default for Retry.MaxAttempts.
	defaultEmbedMaxAttempts = 3

	// defaultEmbedRequestTimeout bounds a single embedding attempt when
	// Retry.RequestTimeout is unset. Tightened from the kit's 5m default
	// to 60s: embeddings are far lighter than LLM completions.
	defaultEmbedRequestTimeout = 60 * time.Second
)

// Backend selects the wire protocol an [Embedder] built by [New] speaks.
// The zero value names no backend and is refused by [Config.Validate].
type Backend string

const (
	// BackendOllama posts to <URL>/api/embed on an Ollama server.
	BackendOllama Backend = "ollama"
	// BackendOpenAICompatible posts to <URL>/v1/embeddings on any
	// OpenAI-compatible API (OpenAI, Azure OpenAI, vLLM, LiteLLM).
	BackendOpenAICompatible Backend = "openai-compatible"
)

// validBackends lists the accepted backend names in ParseBackend's error,
// which Validate returns for a bad Backend.
const validBackends = `"ollama" or "openai-compatible"`

// ParseBackend maps a backend name onto a Backend. The error names the
// bad value and lists every accepted one, so a bad config value is
// actionable without reading this package.
func ParseBackend(s string) (Backend, error) {
	switch b := Backend(s); b {
	case BackendOllama, BackendOpenAICompatible:
		return b, nil
	default:
		return "", fmt.Errorf("embed: unknown backend %q: expected %s", s, validBackends)
	}
}

// Config configures an embedding backend and its retry policy. [New]
// calls Validate before building the backend named by Backend.
type Config struct {
	// Backend selects the wire protocol: [BackendOllama] or
	// [BackendOpenAICompatible]. The zero value is refused by Validate.
	Backend Backend

	// Model is the embedding model name (e.g. "nomic-embed-text").
	// Validate requires a non-empty Model.
	Model string

	// URL is the base URL of the embedding service. Validate requires a
	// non-empty URL; there is no default.
	URL string

	// APIKey, when non-empty, is sent as "Authorization: Bearer <APIKey>"
	// on every request to BOTH backends; when empty, neither backend
	// sends an Authorization header. Validate refuses leading/trailing
	// whitespace and any byte net/http refuses in a header value (bytes
	// 0x00-0x1F except tab, and 0x7F). The error never echoes APIKey,
	// raw or trimmed. An empty APIKey is always allowed.
	APIKey string

	// Dimensions overrides the expected vector dimensionality. When
	// zero, the embedder detects the dimensionality from the first
	// vector it accepts. A later vector of a different length is an
	// error either way. Validate rejects a negative value.
	Dimensions int

	// MaxBatch caps the number of texts sent in one HTTP request. Zero
	// sends the whole batch in one request; Validate rejects a negative
	// value. EmbedBatch splits larger inputs and preserves order; a
	// failed chunk fails the whole call.
	MaxBatch int

	// HTTPClient optionally injects a custom HTTP client (custom
	// transport, proxy, test double). When non-nil, the embedder uses it
	// as-is, including its Timeout: an attempt ends at the earlier of
	// Retry.RequestTimeout and the client's Timeout. When nil, the
	// embedder uses a client with no Timeout, so Retry.RequestTimeout
	// is the only per-attempt bound.
	HTTPClient *http.Client

	// Retry tunes transient-failure retries via [retry.Config]. Unset
	// fields are completed at construction by [retry.Config.Or] against
	// the embed defaults (MaxAttempts 3, RequestTimeout 60s,
	// BaseDelay/MaxDelay/Jitter from [retry.Default]). MaxAttempts,
	// BaseDelay, MaxDelay, and RequestTimeout resolve when <= 0;
	// Jitter resolves when == 0 — to run the resolved defaults with no
	// jitter, set Retry.Rand to a function returning 0.5. Validate
	// rejects a Jitter outside [0, 1].
	Retry retry.Config
}

// Validate reports whether c is internally consistent: Backend is
// [BackendOllama] or [BackendOpenAICompatible], Model and URL are
// non-empty, Dimensions and MaxBatch are non-negative, APIKey carries no
// leading/trailing whitespace and no byte net/http cannot send in a
// header value, and Retry.Jitter is in [0, 1]. Every rejection wraps
// [llmkit.ErrInvalidRequest] and never echoes APIKey.
func (c Config) Validate() error {
	if _, err := ParseBackend(string(c.Backend)); err != nil {
		return fmt.Errorf("%w: %w", err, llmkit.ErrInvalidRequest)
	}
	if c.Model == "" {
		return fmt.Errorf("embedding model name is required: %w", llmkit.ErrInvalidRequest)
	}
	if c.URL == "" {
		return fmt.Errorf("embedding service URL is required: %w", llmkit.ErrInvalidRequest)
	}
	if c.Dimensions < 0 {
		return fmt.Errorf("dimensions must be non-negative, got %d: %w", c.Dimensions, llmkit.ErrInvalidRequest)
	}
	if c.MaxBatch < 0 {
		return fmt.Errorf("max batch must be non-negative, got %d: %w", c.MaxBatch, llmkit.ErrInvalidRequest)
	}
	if strings.TrimSpace(c.APIKey) != c.APIKey {
		return fmt.Errorf("embed: API key must not have leading or trailing whitespace: %w", llmkit.ErrInvalidRequest)
	}
	if !validHeaderValue(c.APIKey) {
		return fmt.Errorf("embed: API key must not contain control characters: %w", llmkit.ErrInvalidRequest)
	}
	if c.Retry.Jitter < 0 || c.Retry.Jitter > 1 {
		return fmt.Errorf("retry jitter must be in [0, 1], got %v: %w", c.Retry.Jitter, llmkit.ErrInvalidRequest)
	}
	return nil
}

// validHeaderValue reports whether net/http sends v in a header value:
// its transport refuses any control byte other than tab (bytes below
// 0x20, and DEL), the rule of httpguts.ValidHeaderFieldValue.
func validHeaderValue(v string) bool {
	for i := range len(v) {
		if b := v[i]; (b < ' ' && b != '\t') || b == 0x7f {
			return false
		}
	}
	return true
}

// httpClient returns the injected client as-is, or a plain client with
// no Timeout. With the plain client, retry.Do bounds each round trip
// with the per-attempt Retry.RequestTimeout deadline. An injected
// client keeps its own Timeout, which can end an attempt earlier.
// Callers must treat the returned client as read-only.
func (c Config) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{}
}

// defaultRetry is [retry.Default] narrowed to the embed bounds:
// MaxAttempts 3 and a 60s per-attempt RequestTimeout. BaseDelay,
// MaxDelay, and Jitter keep the kit defaults.
func defaultRetry() retry.Config {
	d := retry.Default()
	d.MaxAttempts = defaultEmbedMaxAttempts
	d.RequestTimeout = defaultEmbedRequestTimeout
	return d
}

// retryPolicy returns the effective retry policy: c.Retry completed by
// [retry.Config.Or] against defaultRetry. MaxAttempts, BaseDelay,
// MaxDelay, and RequestTimeout resolve when <= 0; Jitter resolves when
// == 0; Sleep and Rand pass through from c.Retry unchanged.
func (c Config) retryPolicy() retry.Config {
	return c.Retry.Or(defaultRetry())
}

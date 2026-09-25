package embed

import (
	"fmt"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	// defaultEmbedMaxAttempts is the embed default for Retry.MaxAttempts:
	// embedding calls retry less than LLM completions (3 attempts total).
	defaultEmbedMaxAttempts = 3

	// defaultEmbedRequestTimeout bounds a single embedding attempt when
	// Retry.RequestTimeout is unset. Embeddings are far lighter than LLM
	// completions, so the kit's 5m default is tightened to 60s; worst case
	// before giving up is roughly MaxAttempts * (60s + backoff).
	defaultEmbedRequestTimeout = 60 * time.Second
)

// Config configures an embedding backend, its retry policy, and its
// optional cache. NewEmbedder, NewOllamaEmbedder, and
// NewOpenAICompatibleEmbedder call Validate before use.
type Config struct {
	// Embedder selects the backend: "ollama" or "openai-compatible".
	Embedder string

	// Model is the embedding model name (e.g. "nomic-embed-text").
	Model string

	// URL is the base URL of the embedding service. Validate requires a
	// non-empty URL; there is no default.
	URL string

	// APIKey is the bearer token for the openai-compatible backend. Ollama
	// ignores it, and Validate does not require it.
	APIKey string

	// Dimensions overrides the expected vector dimensionality. When zero,
	// the embedder detects the dimensionality from the first response. A
	// later vector of a different length is an error either way.
	Dimensions int

	// CacheEnabled turns on the content-hash embedding cache, bounded by
	// CacheSize.
	CacheEnabled bool

	// HTTPClient optionally injects a custom HTTP client (custom transport,
	// proxy, test double). When non-nil, the embedder uses the injected
	// client as-is, including its Timeout: an attempt then ends at the
	// earlier of Retry.RequestTimeout and the client's Timeout. When nil,
	// the embedder uses a client with no Timeout, so Retry.RequestTimeout
	// is the only per-attempt bound.
	HTTPClient *http.Client

	// chunking: the whole batch goes in one request. Validate rejects a
	// negative value. EmbedBatch splits larger inputs and preserves
	// order; a failed chunk fails the whole call.
	MaxBatch int

	// CacheSize caps the number of cache entries when CacheEnabled is true.
	// Zero means unbounded; Validate rejects a negative value. When
	// positive, the cache evicts the least recently used entry past the
	// bound.
	CacheSize int

	// Retry tunes transient-failure retries via [retry.Config]. Unset
	// fields are completed at backend construction by [retry.Config.Or]
	// against the embed defaults (MaxAttempts 3, RequestTimeout 60s,
	// BaseDelay/MaxDelay/Jitter from [retry.Default]). MaxAttempts,
	// BaseDelay, MaxDelay, and RequestTimeout resolve when <= 0; Jitter
	// resolves when == 0 (an explicit 0 is unset, like every other zero
	// field — to run the resolved defaults with no jitter, set
	// Retry.Rand to a function returning 0.5). Validate rejects a
	// Jitter outside [0, 1].
	Retry retry.Config
}

// defaults returns a Config with sensible local-first defaults. Retry is
// left zero: retryPolicy resolves it via [retry.Config.Or] against
// defaultRetry at construction.
func defaults() Config {
	return Config{
		Embedder:     "ollama",
		Model:        "nomic-embed-text",
		CacheEnabled: false,
	}
}

// LoadConfig reads embedder configuration from environment variables.
//
// Environment variables (prefix is the provided prefix, e.g. "LLMKIT"):
//
//	<PREFIX>_EMBEDDER          - "ollama" (default) or "openai-compatible"
//	<PREFIX>_EMBED_MODEL       - model name (default: nomic-embed-text)
//	<PREFIX>_EMBED_URL         - base URL (required; no default)
//	<PREFIX>_EMBED_API_KEY     - bearer token for the openai-compatible backend (optional)
//	<PREFIX>_EMBED_DIMENSIONS  - vector dimensions (integer; 0 = auto-detect)
//	<PREFIX>_EMBED_CACHE       - "true" (case-insensitive) to enable caching
//	<PREFIX>_EMBED_TIMEOUT     - per-attempt timeout (Go duration, e.g. "30s"; zero, negative, or unset: 60s embed default)
//	<PREFIX>_EMBED_MAX_BATCH   - max texts per HTTP request (integer; 0 = no chunking)
//	<PREFIX>_EMBED_CACHE_SIZE  - max cache entries (integer; 0 = unbounded)
//
// Unset variables keep their defaults. A malformed integer or duration value
// returns an error naming the variable; the returned Config is zero.
func LoadConfig(prefix string) (Config, error) {
	p := strings.ToUpper(prefix)
	cfg := defaults()

	if v := os.Getenv(p + "_EMBEDDER"); v != "" {
		cfg.Embedder = v
	}
	if v := os.Getenv(p + "_EMBED_MODEL"); v != "" {
		cfg.Model = v
	}
	if v := os.Getenv(p + "_EMBED_URL"); v != "" {
		cfg.URL = v
	}
	if v := os.Getenv(p + "_EMBED_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv(p + "_EMBED_DIMENSIONS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s_EMBED_DIMENSIONS: invalid integer %q", p, v)
		}
		cfg.Dimensions = n
	}
	if v := os.Getenv(p + "_EMBED_TIMEOUT"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s_EMBED_TIMEOUT: invalid duration %q", p, v)
		}
		cfg.Retry.RequestTimeout = d
	}
	if v := os.Getenv(p + "_EMBED_MAX_BATCH"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s_EMBED_MAX_BATCH: invalid integer %q", p, v)
		}
		cfg.MaxBatch = n
	}
	if v := os.Getenv(p + "_EMBED_CACHE_SIZE"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s_EMBED_CACHE_SIZE: invalid integer %q", p, v)
		}
		cfg.CacheSize = n
	}
	if strings.EqualFold(os.Getenv(p+"_EMBED_CACHE"), "true") {
		cfg.CacheEnabled = true
	}

	return cfg, nil
}

// Validate reports whether c is internally consistent: Embedder names a
// known backend, Model and URL are non-empty, Dimensions, MaxBatch, and
// CacheSize are non-negative, and Retry.Jitter is in [0, 1]. Every
// rejection wraps [llmkit.ErrInvalidRequest].
func (c Config) Validate() error {
	switch c.Embedder {
	case "ollama", "openai-compatible":
	default:
		return fmt.Errorf("unknown embedder type %q: %w", c.Embedder, llmkit.ErrInvalidRequest)
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
	if c.CacheSize < 0 {
		return fmt.Errorf("cache size must be non-negative, got %d: %w", c.CacheSize, llmkit.ErrInvalidRequest)
	}
	if c.Retry.Jitter < 0 || c.Retry.Jitter > 1 {
		return fmt.Errorf("retry jitter must be in [0, 1], got %v: %w", c.Retry.Jitter, llmkit.ErrInvalidRequest)
	}
	return nil
}

// httpClient returns the injected client as-is, or a plain client with no
// Timeout. With the plain client, retry.Do bounds each round trip
// with the per-attempt Retry.RequestTimeout deadline. An injected client
// keeps its own Timeout, which can end an attempt earlier. Callers must
// treat the returned client as read-only.
func (c Config) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{}
}

// defaultRetry is [retry.Default] narrowed to the embed bounds:
// MaxAttempts 3 and a 60s per-attempt RequestTimeout (embeddings retry
// less than LLM completions, and each attempt is far lighter — worst
// case before giving up is roughly 3 * (60s + backoff)). BaseDelay,
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

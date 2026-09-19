package embed

import (
	"fmt"
	"github.com/dpoage/llmkit"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds all embedder-related settings.
type Config struct {
	// Embedder selects the backend: "ollama" or "openai-compatible".
	Embedder string

	// Model is the embedding model name (e.g. "nomic-embed-text").
	Model string

	// URL is the base URL for the embedding service.
	URL string

	// APIKey is the bearer token for authenticated APIs (optional for Ollama).
	APIKey string

	// Dimensions overrides the expected vector dimensionality.
	// When zero the embedder auto-detects from the first response; a later
	// response with a different dimensionality is an error either way.
	Dimensions int

	// CacheEnabled turns on the content-hash embedding cache.
	CacheEnabled bool

	// HTTPClient optionally injects a custom HTTP client (custom transport,
	// proxy, test double). When non-nil it is used as-is, with its own
	// transport and timeout policy.
	HTTPClient *http.Client

	// MaxBatch caps how many texts are sent per HTTP request. Zero disables
	// chunking (the whole batch goes in one request); negative is rejected
	// by Validate. EmbedBatch splits larger batches and preserves input
	// order; a failed chunk fails the whole call.
	MaxBatch int

	// CacheSize caps the number of cache entries. Zero means unbounded;
	// negative is rejected by Validate. When positive, inserting past the
	// bound evicts the least recently used entry.
	CacheSize int

	// Retry tunes transient-failure retries via the shared
	// llmkit.RetryConfig. Unset knobs (<= 0) are filled from
	// llmkit.DefaultRetryConfig when a backend is constructed, so a partial
	// policy like llmkit.RetryConfig{MaxAttempts: 5} is safe to use as
	// written. Retry.Jitter is the exception and is taken literally: 0 (the
	// zero value) disables jitter for deterministic backoff, and values
	// outside [0, 1] are rejected by Validate. LoadConfig seeds the whole
	// policy from llmkit.DefaultRetryConfig, so environment-driven configs
	// get the kit default 20% jitter.
	Retry llmkit.RetryConfig
}

// defaults returns a Config with sensible local-first defaults.
func defaults() Config {
	return Config{
		Embedder:     "ollama",
		Model:        "nomic-embed-text",
		CacheEnabled: false,
		Retry:        llmkit.DefaultRetryConfig(),
	}
}

// LoadConfig reads embedder configuration from environment variables.
//
// Environment variables (prefix is the provided prefix, e.g. "LLMKIT"):
//
//	<PREFIX>_EMBEDDER          - "ollama" (default) or "openai-compatible"
//	<PREFIX>_EMBED_MODEL       - model name (default: nomic-embed-text)
//	<PREFIX>_EMBED_URL         - base URL
//	<PREFIX>_EMBED_API_KEY     - API key / bearer token
//	<PREFIX>_EMBED_DIMENSIONS  - vector dimensions (integer; 0 = auto-detect)
//	<PREFIX>_EMBED_CACHE       - "true" to enable caching
//	<PREFIX>_EMBED_TIMEOUT     - per-attempt timeout (Go duration, e.g. "30s"; zero, negative, or unset: llmkit's 5m request timeout)
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

// Validate checks that the configuration is internally consistent.
func (c Config) Validate() error {
	switch c.Embedder {
	case "ollama", "openai-compatible":
		// ok
	default:
		return fmt.Errorf("unknown embedder type %q", c.Embedder)
	}
	if c.Model == "" {
		return fmt.Errorf("embedding model name is required")
	}
	// URL is required for all HTTP-based backends.
	if c.URL == "" {
		return fmt.Errorf("embedding service URL is required")
	}
	if c.Dimensions < 0 {
		return fmt.Errorf("dimensions must be non-negative, got %d", c.Dimensions)
	}
	if c.MaxBatch < 0 {
		return fmt.Errorf("max batch must be non-negative, got %d", c.MaxBatch)
	}
	if c.CacheSize < 0 {
		return fmt.Errorf("cache size must be non-negative, got %d", c.CacheSize)
	}
	if c.Retry.Jitter < 0 || c.Retry.Jitter > 1 {
		return fmt.Errorf("retry jitter must be in [0, 1], got %v", c.Retry.Jitter)
	}
	return nil
}

// httpClient returns the injected client as-is, or a plain client. Round
// trips are bounded by the per-attempt Retry.RequestTimeout deadline (see
// retryDo), not by an http.Client timeout. Callers must treat the returned
// client as read-only.
func (c Config) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{}
}

// retryPolicy returns the effective retry policy: c.Retry with unset knobs
// (<= 0) filled from llmkit.DefaultRetryConfig — MaxAttempts, the delays, and
// the per-attempt RequestTimeout. Jitter is deliberately not normalized here:
// Validate has already bounded it to [0, 1] and it is taken literally, so 0
// means no jitter; LoadConfig seeds it from the kit default, keeping the
// default 20% jitter reachable for environment-driven configs.
func (c Config) retryPolicy() llmkit.RetryConfig {
	p := c.Retry
	def := llmkit.DefaultRetryConfig()
	if p.MaxAttempts <= 0 {
		p.MaxAttempts = def.MaxAttempts
	}
	if p.BaseDelay <= 0 {
		p.BaseDelay = def.BaseDelay
	}
	if p.MaxDelay <= 0 {
		p.MaxDelay = def.MaxDelay
	}
	if p.RequestTimeout <= 0 {
		p.RequestTimeout = def.RequestTimeout
	}
	return p
}

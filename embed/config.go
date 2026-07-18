package embed

import (
	"fmt"
	"os"
	"strconv"
	"strings"
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
	// When zero the embedder auto-detects from the first response.
	Dimensions int

	// CacheEnabled turns on the content-hash embedding cache.
	CacheEnabled bool
}

// defaults returns a Config with sensible local-first defaults.
func defaults() Config {
	return Config{
		Embedder:     "ollama",
		Model:        "nomic-embed-text",
		CacheEnabled: false,
	}
}

// LoadConfig reads embedder configuration from environment variables.
//
// Environment variables (prefix is the provided prefix, e.g. "KNOWN"):
//
//	<PREFIX>_EMBEDDER          - "ollama" (default) or "openai-compatible"
//	<PREFIX>_EMBED_MODEL       - model name (default: nomic-embed-text)
//	<PREFIX>_EMBED_URL         - base URL
//	<PREFIX>_EMBED_API_KEY     - API key / bearer token
//	<PREFIX>_EMBED_DIMENSIONS  - override vector dimensions
//	<PREFIX>_EMBED_CACHE       - "true" to enable caching
func LoadConfig(prefix string) Config {
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
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Dimensions = n
		}
	}
	if strings.EqualFold(os.Getenv(p+"_EMBED_CACHE"), "true") {
		cfg.CacheEnabled = true
	}

	return cfg
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
	return nil
}

// Package provider builds fully-wrapped llmkit.Clients for a named provider
// type. It is the single construction entry point: callers describe the
// endpoint with a Spec and tune the wrapper stack with Options; this package
// dispatches to the right first-party adapter and decorates it with
// serialize -> recorder -> retry (outer to inner).
package provider

import (
	"context"
	"fmt"
	"net/http"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/anthropic"
	"github.com/dpoage/llmkit/provider/google"
	"github.com/dpoage/llmkit/provider/openai"
)

// Type enumerates the supported LLM provider backends.
type Type string

const (
	TypeAnthropic        Type = "anthropic"
	TypeOpenAI           Type = "openai"
	TypeGoogle           Type = "google"
	TypeOpenAICompatible Type = "openai-compatible"
)

// Spec is the caller-owned description of a provider endpoint. Callers
// construct it from their own config types and pass it to New. The provider
// package never reads external config directly.
//
// Auth selects the credential mode: empty string or "api_key" → standard API
// key (x-api-key header); "oauth-token" → OAuth bearer-token (Anthropic only).
//
// StructuredOutput, when non-nil, overrides the adapter's built-in default.
// nil = use the adapter default (true for first-party providers, false for
// openai-compatible endpoints).
type Spec struct {
	Type    Type
	BaseURL string // optional; for testing or non-default endpoints
	Auth    string // "" | "api_key" | "oauth-token"

	// StructuredOutput overrides the adapter's StructuredOutput capability.
	// nil = adapter default.
	StructuredOutput *bool
}

// Options tunes client construction. The zero value is valid: it uses default
// retry policy, no recorder, and the default HTTP transport.
type Options struct {
	// Retry configures the shared retry wrapper. If MaxAttempts is 0,
	// DefaultRetryConfig is used.
	Retry llmkit.RetryConfig
	// Recorder, if non-nil, receives a UsageEvent after each successful
	// completion.
	Recorder llmkit.Recorder
	// Role tags emitted UsageEvents.
	Role string
	// HTTPClient overrides the transport used by the underlying SDKs. Primarily
	// for tests (httptest) and proxies. nil uses the SDK default.
	HTTPClient *http.Client
}

// New builds a fully-wrapped Client for the given provider spec and model.
// The returned client is decorated, outer-to-inner, with:
//
//	serialize -> recorder -> retry -> adapter
//
// so usage is recorded only for the final successful attempt, retries see
// the raw adapter errors, and non-parallel-capable providers (e.g. arbitrary
// openai-compatible endpoints) have their multi-tool-call responses
// truncated to one before the agent loop sees them. The API key is handed
// to the SDK; it is never logged.
//
// WithSerializedToolCalls is a no-op for providers whose Capabilities report
// ParallelToolCalls=true (Anthropic, Google, first-party OpenAI), so
// decorating unconditionally is safe and capability-driven.
//
// apiKey is the resolved secret (caller obtains it via their own config).
// Passing it explicitly keeps this constructor free of environment lookups and
// testable without real keys.
func New(ctx context.Context, spec Spec, providerName, model, apiKey string, opts Options) (llmkit.Client, error) {
	if model == "" {
		return nil, fmt.Errorf("llmkit: model must not be empty for provider %q", providerName)
	}

	var adapter llmkit.Client
	switch spec.Type {
	case TypeAnthropic:
		aopts := anthropic.Options{
			BaseURL:    spec.BaseURL,
			HTTPClient: opts.HTTPClient,
		}
		// The secret parameter carries either an API key or an OAuth bearer token
		// depending on the provider's auth mode. Route it to the right field so
		// anthropic.New can select the correct authentication path.
		if spec.Auth == "oauth-token" {
			aopts.AuthToken = apiKey
		} else {
			aopts.APIKey = apiKey
		}
		aopts.StructuredOutput = spec.StructuredOutput
		adapter = anthropic.New(model, aopts)
	case TypeOpenAI:
		adapter = openai.New(model, openai.Options{
			APIKey:           apiKey,
			BaseURL:          spec.BaseURL,
			HTTPClient:       opts.HTTPClient,
			StructuredOutput: spec.StructuredOutput,
		})
	case TypeOpenAICompatible:
		adapter = openai.New(model, openai.Options{
			APIKey:           apiKey,
			BaseURL:          spec.BaseURL,
			HTTPClient:       opts.HTTPClient,
			Compatible:       true,
			StructuredOutput: spec.StructuredOutput,
		})
	case TypeGoogle:
		ga, err := google.New(ctx, model, google.Options{
			APIKey:           apiKey,
			BaseURL:          spec.BaseURL,
			HTTPClient:       opts.HTTPClient,
			StructuredOutput: spec.StructuredOutput,
		})
		if err != nil {
			return nil, err
		}
		adapter = ga
	default:
		return nil, fmt.Errorf("llmkit: unsupported provider type %q", spec.Type)
	}

	retryCfg := opts.Retry
	if retryCfg.MaxAttempts == 0 {
		retryCfg = llmkit.DefaultRetryConfig()
	}
	client := llmkit.WithRetry(adapter, retryCfg)
	client = llmkit.WithRecorder(client, opts.Recorder, opts.Role, providerName, model)
	// Outermost: force at-most-one tool call per response when the backend
	// does not support parallel tool calls (see llmkit serialize.go). Capable
	// providers short-circuit inside the wrapper, so this is a free check
	// for anthropic/google/openai and the safety net for openai-compatible.
	return llmkit.WithSerializedToolCalls(client), nil
}

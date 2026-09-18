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

// Auth selects how Spec.Secret is presented to the provider. The zero
// value is API-key mode.
type Auth string

const (
	// AuthAPIKey sends Secret as a standard API key (e.g. the x-api-key
	// header). It is the zero value.
	AuthAPIKey Auth = ""
	// AuthOAuthToken sends Secret as an OAuth bearer token (Authorization
	// header). Anthropic only.
	AuthOAuthToken Auth = "oauth-token"
)

// ParseType maps a provider name onto a Type. The error names the bad
// value and lists every accepted one, so a bad config value is actionable
// without reading this package.
func ParseType(s string) (Type, error) {
	switch t := Type(s); t {
	case TypeAnthropic, TypeOpenAI, TypeOpenAICompatible, TypeGoogle:
		return t, nil
	default:
		return "", fmt.Errorf("llmkit: unknown provider type %q: expected %q, %q, %q, or %q",
			s, TypeAnthropic, TypeOpenAI, TypeOpenAICompatible, TypeGoogle)
	}
}

// Spec is the caller-owned description of a provider endpoint. Callers
// construct it from their own config types and pass it to New. The provider
// package never reads external config directly.
//
// Auth selects the credential mode. The zero value AuthAPIKey sends Secret
// as a standard API key (x-api-key header); AuthOAuthToken sends it as an
// OAuth bearer token (Authorization header — Anthropic only). Any other
// value is refused by New with an error wrapping ErrInvalidRequest; there
// is no silent fallback to API-key mode.
//
// Capabilities, when non-nil, tunes the adapter's model-table profile: the
// function receives the table-derived profile and returns the effective
// one, applied once at construction. Flip a single field in a closure, or
// return a fixed profile to pin exact values for models the table doesn't
// know (adapters report ContextWindow 0 for unknown models; llmkit never
// fabricates a window). The override is plumbed into the adapter itself, so
// wire behavior follows the effective profile, not just the reported one —
// concretely: on a profile with Thinking=false the thinking config is
// dropped silently; on a profile with ToolChoice=false every explicit
// Request.ToolChoice mode is rejected with ErrInvalidRequest before the
// wire call (auto stays allowed); StructuredOutput=false gates
// response_format (OpenAI) and the synthetic forced-output tool (Anthropic)
// off, and ParallelToolCalls=false installs the tool-call serializer.
type Spec struct {
	Type Type

	// Model is the model identifier (e.g. "claude-sonnet-4-5"). Empty is
	// refused by New.
	Model string
	// BaseURL is optional; for testing or non-default endpoints.
	BaseURL string
	// Auth selects the credential mode; the zero value is API-key mode.
	Auth Auth
	// Secret is the resolved credential: an API key in AuthAPIKey mode, an
	// OAuth bearer token in AuthOAuthToken mode. The caller obtains it via
	// its own config; New hands it to the SDK and never logs it.
	Secret string

	// Capabilities tunes the adapter's model-table profile; nil keeps the
	// table.
	Capabilities func(llmkit.Capabilities) llmkit.Capabilities
}

// Options tunes client construction. The zero value is valid: it uses
// default retry policy, no recorder, and the default HTTP transport.
type Options struct {
	// Retry configures the shared retry wrapper. If MaxAttempts is 0,
	// DefaultRetryConfig is used.
	Retry llmkit.RetryConfig
	// Recorder, if non-nil, receives a UsageEvent after each successful
	// completion.
	Recorder llmkit.Recorder
	// Provider overrides the provider tag on emitted UsageEvents. Empty
	// tags events with string(spec.Type); set it when your ledger keys on
	// a config-map name rather than the provider type.
	Provider string
	// HTTPClient overrides the transport used by the underlying SDKs.
	// Primarily for tests (httptest) and proxies. nil uses the SDK
	// default.
	HTTPClient *http.Client
}

// New builds a fully-wrapped Client for the given provider spec.
// The returned client is decorated, outer-to-inner, with:
//
//	serialize -> recorder -> retry -> adapter
//
// so usage is recorded only for the final successful attempt, retries see
// the raw adapter errors, and non-parallel-capable providers (e.g. arbitrary
// openai-compatible endpoints) have their multi-tool-call responses
// truncated to one before the agent loop sees them. WithSerializedToolCalls
// is a no-op for providers whose Capabilities report ParallelToolCalls=true
// (Anthropic, Google, first-party OpenAI), so decorating unconditionally is
// safe and capability-driven.
//
// spec.Secret is the resolved credential (callers obtain it via their own
// config); New performs no environment lookups, so it stays testable
// without real keys. spec.Auth routes the secret to the right credential
// field; unknown Auth values, an empty spec.Model, and an unknown
// spec.Type are errors wrapping ErrInvalidRequest.
func New(ctx context.Context, spec Spec, opts Options) (llmkit.Client, error) {
	if spec.Model == "" {
		return nil, fmt.Errorf("llmkit: model must not be empty for provider %q: %w", spec.Type, llmkit.ErrInvalidRequest)
	}
	switch spec.Auth {
	case AuthAPIKey, AuthOAuthToken:
	default:
		return nil, fmt.Errorf("llmkit: unknown Auth %q for provider %q (want AuthAPIKey or AuthOAuthToken): %w",
			string(spec.Auth), spec.Type, llmkit.ErrInvalidRequest)
	}

	var adapter llmkit.Client
	switch spec.Type {
	case TypeAnthropic:
		aopts := anthropic.Options{
			BaseURL:      spec.BaseURL,
			HTTPClient:   opts.HTTPClient,
			Capabilities: spec.Capabilities,
		}
		// The secret carries either an API key or an OAuth bearer token
		// depending on the spec's auth mode. Route it to the right field so
		// anthropic.New can select the correct authentication path.
		if spec.Auth == AuthOAuthToken {
			aopts.AuthToken = spec.Secret
		} else {
			aopts.APIKey = spec.Secret
		}
		adapter = anthropic.New(spec.Model, aopts)
	case TypeOpenAI:
		adapter = openai.New(spec.Model, openai.Options{
			APIKey:       spec.Secret,
			BaseURL:      spec.BaseURL,
			HTTPClient:   opts.HTTPClient,
			Capabilities: spec.Capabilities,
		})
	case TypeOpenAICompatible:
		adapter = openai.New(spec.Model, openai.Options{
			APIKey:       spec.Secret,
			BaseURL:      spec.BaseURL,
			HTTPClient:   opts.HTTPClient,
			Compatible:   true,
			Capabilities: spec.Capabilities,
		})
	case TypeGoogle:
		ga, err := google.New(ctx, spec.Model, google.Options{
			APIKey:       spec.Secret,
			BaseURL:      spec.BaseURL,
			HTTPClient:   opts.HTTPClient,
			Capabilities: spec.Capabilities,
		})
		if err != nil {
			return nil, err
		}
		adapter = ga
	default:
		return nil, fmt.Errorf("llmkit: unsupported provider type %q: expected %q, %q, %q, or %q (see ParseType): %w",
			spec.Type, TypeAnthropic, TypeOpenAI, TypeOpenAICompatible, TypeGoogle, llmkit.ErrInvalidRequest)
	}

	retryCfg := opts.Retry
	if retryCfg.MaxAttempts == 0 {
		retryCfg = llmkit.DefaultRetryConfig()
	}
	client := llmkit.WithRetry(adapter, retryCfg)
	providerTag := opts.Provider
	if providerTag == "" {
		providerTag = string(spec.Type)
	}
	client = llmkit.WithRecorder(client, opts.Recorder, providerTag, spec.Model)
	// Outermost: force at-most-one tool call per response when the backend
	// does not support parallel tool calls (see llmkit serialize.go). Capable
	// providers short-circuit inside the wrapper, so this is a free check
	// for anthropic/google/openai and the safety net for openai-compatible.
	return llmkit.WithSerializedToolCalls(client), nil
}

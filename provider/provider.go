// Package provider builds fully-wrapped llmkit.Clients for a named provider
// type. It is the single construction entry point: callers describe the
// endpoint with a Spec and tune the wrapper stack with Options; this package
// dispatches to the right first-party adapter and decorates it with
// serialize -> recorder -> retry (outer to inner).
//
// The vendor-SDK adapters live under provider/internal/{anthropic,openai,
// google} and are internal on purpose: provider.New is the only construction
// path, so Spec validation (auth mode, Secret shape, OpenAI-compatible
// BaseURL, model presence) cannot be bypassed.
package provider

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/internal/anthropic"
	"github.com/dpoage/llmkit/provider/internal/google"
	"github.com/dpoage/llmkit/provider/internal/openai"
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

// validTypes is the single source of the accepted provider-name list in
// ParseType's and New's errors, so the two cannot drift.
const validTypes = `"anthropic", "openai", "openai-compatible", or "google"`

// ParseType maps a provider name onto a Type. The error names the bad
// value and lists every accepted one, so a bad config value is actionable
// without reading this package.
func ParseType(s string) (Type, error) {
	switch t := Type(s); t {
	case TypeAnthropic, TypeOpenAI, TypeOpenAICompatible, TypeGoogle:
		return t, nil
	default:
		return "", fmt.Errorf("llmkit: unknown provider type %q: expected %s", s, validTypes)
	}
}

// Spec is the caller-owned description of a provider endpoint. Callers
// construct it from their own config types and pass it to New. The provider
// package never reads external config directly.
//
// Auth selects the credential mode. The zero value AuthAPIKey sends Secret
// as a standard API key (x-api-key header); AuthOAuthToken sends it as an
// OAuth bearer token (Authorization header) and is Anthropic-only — New
// refuses AuthOAuthToken on any other Type with an error wrapping
// ErrInvalidRequest. Any other Auth value is likewise refused; there is no
// silent fallback to API-key mode.
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
	// BaseURL overrides the vendor endpoint; mainly for tests, proxies, and
	// self-hosted gateways. Empty means the vendor default: the Anthropic
	// SDK targets api.anthropic.com, the OpenAI SDK api.openai.com/v1, and
	// the Google GenAI SDK generativelanguage.googleapis.com. The SDKs
	// themselves (not New) also honor ANTHROPIC_BASE_URL, OPENAI_BASE_URL,
	// and GOOGLE_GEMINI_BASE_URL when BaseURL is empty, so an unset
	// Spec.BaseURL can still route the caller's Secret to an
	// ambient-configured host.
	// TypeOpenAICompatible is the exception: it has no vendor default of
	// its own — empty would silently target first-party api.openai.com/v1
	// — so New refuses it with an error wrapping ErrInvalidRequest.
	BaseURL string
	// Auth selects the credential mode; the zero value is API-key mode.
	Auth Auth
	// Secret is the resolved credential: an API key in AuthAPIKey mode, an
	// OAuth bearer token in AuthOAuthToken mode. New refuses a Secret that
	// is empty or whitespace-padded, with an error wrapping
	// ErrInvalidRequest; the value is never echoed. New never reads the
	// environment for credentials and never logs the Secret. For a
	// credential-less endpoint — a local Ollama or vLLM server — pass any
	// non-empty placeholder; New only checks that Secret is present, never
	// that the backend accepts it.
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
// without real keys. spec.Auth routes the secret
// to the right credential field; unknown Auth values, AuthOAuthToken on
// a non-Anthropic Type, an empty spec.Model, a spec.Secret that is
// empty, whitespace-only, or differs from its own strings.TrimSpace, an
// empty spec.BaseURL on TypeOpenAICompatible, and an unknown spec.Type
// are errors wrapping ErrInvalidRequest.
func New(ctx context.Context, spec Spec, opts Options) (llmkit.Client, error) {
	switch spec.Auth {
	case AuthAPIKey, AuthOAuthToken:
	default:
		return nil, fmt.Errorf("llmkit: unknown Auth %q for provider %q (want AuthAPIKey or AuthOAuthToken): %w",
			string(spec.Auth), spec.Type, llmkit.ErrInvalidRequest)
	}
	if spec.Auth == AuthOAuthToken && spec.Type != TypeAnthropic {
		// OAuth bearer-token authentication is implemented only by the
		// Anthropic adapter; routing Secret to another adapter's API-key
		// field would silently downgrade the credential.
		return nil, fmt.Errorf("llmkit: Auth %q is only supported for provider %q, not %q: %w",
			spec.Auth, TypeAnthropic, spec.Type, llmkit.ErrInvalidRequest)
	}
	if spec.Model == "" {
		return nil, fmt.Errorf("llmkit: model must not be empty for provider %q: %w", spec.Type, llmkit.ErrInvalidRequest)
	}
	// New refuses a Secret that is empty or differs from its own
	// strings.TrimSpace. No vendor issues a credential with surrounding
	// whitespace, and such a value is almost always a copy/paste or
	// `cat`/`pass` artifact; refusing it here turns a confusing runtime
	// failure into an immediate, actionable one. The error never echoes
	// the secret.
	if trimmed := strings.TrimSpace(spec.Secret); trimmed == "" || trimmed != spec.Secret {
		return nil, fmt.Errorf("llmkit: secret must be a non-empty value with no leading or trailing whitespace for provider %q: %w", spec.Type, llmkit.ErrInvalidRequest)
	}
	// An openai-compatible Type with no BaseURL would silently target the
	// first-party OpenAI host, so the endpoint must be given explicitly.
	if spec.Type == TypeOpenAICompatible && spec.BaseURL == "" {
		return nil, fmt.Errorf("llmkit: base URL must not be empty for provider %q: the endpoint must be given: %w", spec.Type, llmkit.ErrInvalidRequest)
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
		return nil, fmt.Errorf("llmkit: unsupported provider type %q: expected %s (see ParseType): %w",
			spec.Type, validTypes, llmkit.ErrInvalidRequest)
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

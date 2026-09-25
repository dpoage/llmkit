// Package provider builds fully-wrapped llmkit.Clients for a named provider
// type. Callers describe the endpoint with a [Spec], tune the wrapper stack
// with [Options], and call [New]; this package dispatches to the right
// first-party adapter and decorates it. It is the single construction entry
// point, so Spec validation cannot be bypassed.
//
// The vendor-SDK adapters live under provider/internal/{anthropic,openai,
// google} and are internal on purpose: [New] is the only way in.
//
// # Provider types
//
// [Type] selects the adapter:
//
//   - [TypeAnthropic]: Anthropic Claude models through the official SDK.
//   - [TypeOpenAI]: first-party OpenAI models through Chat Completions.
//   - [TypeOpenAICompatible]: any OpenAI-compatible endpoint (Ollama, vLLM,
//     Groq, ...) through the OpenAI adapter under a conservative profile.
//   - [TypeGoogle]: Gemini models through the official GenAI SDK.
//
// [Spec.BaseURL] overrides the vendor endpoint per type. Anthropic, OpenAI,
// and Google have vendor defaults; TypeOpenAICompatible has none, so New
// refuses an empty BaseURL there. New also refuses an empty BaseURL when
// the Type's base-URL environment variable is present, or when the
// Anthropic SDK profile file the SDK would load sets a base_url (see
// [Spec.BaseURL] for the exact rules).
//
// # Credentials
//
// [Spec.Auth] selects the credential mode: [AuthAPIKey] (the zero value)
// sends Secret as the provider's standard API-key credential;
// [AuthOAuthToken] sends it as an OAuth bearer token and is Anthropic-only.
// [Spec.Secret] must be a non-empty value without surrounding whitespace.
// Secret is the only credential New sends: no vendor-SDK environment
// variable or profile file supplies one. New never logs the Secret.
// A credential-less endpoint (a local Ollama or vLLM server) takes any
// non-empty placeholder.
//
// # Capability profiles
//
// [Spec.Capabilities] tunes the adapter's model-table profile once at
// construction: flip a single field in a closure, or return a fixed profile
// to pin exact values for models the table does not know. The override is
// plumbed into the adapter, so wire behavior follows the effective profile.
// See [llmkit.Capabilities] for the enforcement classes and
// docs/capabilities.md for the per-field table.
//
// # Construction
//
// [New] validates the spec and returns an error wrapping
// llmkit.ErrInvalidRequest for a malformed Auth, Model, Secret, BaseURL, or
// Type; [New] lists the exact rejections. New also refuses an empty
// [Spec.BaseURL] when a vendor-SDK base-URL environment variable or an
// Anthropic SDK profile file would have sent the request to another host:
// OpenAI's `OPENAI_BASE_URL`, Anthropic's `ANTHROPIC_BASE_URL` and its
// profile files, Google's `GOOGLE_GEMINI_BASE_URL`. No value read from
// those sources — or from any other vendor-SDK environment variable or
// profile file — sets a request's host or any of its headers: every request
// goes to Spec.BaseURL when set, else to the vendor default host, and its
// credential header carries Spec.Secret. New performs no network I/O.
//
// # The stack
//
// [Wrap] decorates any [llmkit.Client] with this package's stack; [New]
// applies the same stack to a freshly built adapter. The stage order
// between the caller and the adapter is this package's secret: outer
// to inner, a completion emitter, then tool-call serialization, then
// attempt-observed retry. A caller sees: models without parallel tool
// calls have their multi-call responses truncated to one, transient
// failures retry with backoff, and [Options.Observer] receives one
// Attempt event per provider attempt (failures included) plus one
// Completion event per logical completion the caller's context does not
// already belong to (see [llmkit.BeginCompletion]) — so a New/Wrap
// client run under the agent Runner reports Attempts only, the
// Runner's Completion being the one that already claimed the ctx.
// Wrap ignores [Options.HTTPClient] (adapters only) and, over a foreign
// client with no [llmkit.Identity] of its own and no [Options.Provider]
// set, emits events with empty Provider/Model — documented, not an
// error.
package provider

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/config"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/internal/anthropic"
	"github.com/dpoage/llmkit/provider/internal/google"
	"github.com/dpoage/llmkit/provider/internal/openai"
	"github.com/dpoage/llmkit/retry"
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
	// AuthAPIKey sends Secret as the provider's standard API-key credential:
	// the x-api-key header on Anthropic, an Authorization bearer token on
	// OpenAI and openai-compatible, and the x-goog-api-key header on Google.
	// It is the zero value.
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
// construct it from their own config types and pass it to [New]; this
// package never reads external config directly. The field docs below state
// the rules New enforces.
type Spec struct {
	// Type selects the adapter: [TypeAnthropic], [TypeOpenAI],
	// [TypeOpenAICompatible], or [TypeGoogle]. New refuses any other value
	// with an error wrapping llmkit.ErrInvalidRequest.
	Type Type

	// Model is the model identifier (e.g. "claude-sonnet-4-5"). Empty is
	// refused by New.
	Model string
	// BaseURL overrides the vendor endpoint; mainly for tests, proxies, and
	// self-hosted gateways. Empty means the vendor default: the Anthropic
	// SDK targets api.anthropic.com, the OpenAI SDK api.openai.com/v1, and
	// the Google GenAI SDK generativelanguage.googleapis.com. The openai
	// adapter selects its SDK's production environment explicitly, the
	// google adapter passes the vendor host itself, and the anthropic
	// adapter runs its SDK with environment defaults off so the SDK's
	// built-in production host applies. No vendor-SDK
	// environment variable or profile file sets the host or a request
	// header.
	// TypeOpenAICompatible is the exception: it has no vendor default of
	// its own — empty would silently target first-party api.openai.com/v1
	// — so New refuses it with an error wrapping ErrInvalidRequest.
	//
	// For TypeOpenAI, TypeAnthropic, and TypeGoogle, an empty BaseURL is
	// refused when one of these sources would have set the host:
	//
	//   - The Type's base-URL variable (OPENAI_BASE_URL, ANTHROPIC_BASE_URL,
	//     or GOOGLE_GEMINI_BASE_URL) is present in the process environment.
	//     An empty value counts as present.
	//   - For TypeAnthropic, the Anthropic SDK profile file that
	//     anthropic-sdk-go would load has a base_url. The SDK looks for
	//     configs/<profile>.json in ANTHROPIC_CONFIG_DIR, else
	//     $XDG_CONFIG_HOME/anthropic, else $HOME/.config/anthropic outside
	//     Windows, else the CWD-relative configs/<profile>.json when HOME
	//     is unset or empty and XDG_CONFIG_HOME is unset or empty, or when
	//     ANTHROPIC_CONFIG_DIR is present but empty. The profile is
	//     ANTHROPIC_PROFILE when that variable is non-empty (present but
	//     empty: no profile is read), else the one the active_config file
	//     names, else "default". The SDK reads no profile when ANTHROPIC_API_KEY
	//     or ANTHROPIC_AUTH_TOKEN is non-empty. When environment federation
	//     is fully configured, the SDK reads only a profile named by
	//     ANTHROPIC_PROFILE and skips the active_config-named and default
	//     profiles. New follows the same rules.
	//
	// The error wraps ErrInvalidRequest, names the variable or the profile
	// file, and tells you to set BaseURL; nothing is sent. New reads these
	// sources only to decide the refusal. To use the host they name, set
	// BaseURL to it; to reach the vendor default, remove the source.
	BaseURL string
	// Auth selects the credential mode. The zero value [AuthAPIKey] sends
	// Secret as the provider's standard API-key credential (see the const
	// list for the per-provider wire form); [AuthOAuthToken]
	// sends it as an OAuth bearer token (Authorization header) and is
	// Anthropic-only. New refuses AuthOAuthToken on any other Type with an
	// error wrapping llmkit.ErrInvalidRequest, and refuses any other Auth
	// value likewise: there is no silent fallback to API-key mode.
	Auth Auth
	// Secret is the resolved credential: an API key in AuthAPIKey mode, an
	// OAuth bearer token in AuthOAuthToken mode. New refuses a Secret that
	// is empty or whitespace-padded, with an error wrapping
	// ErrInvalidRequest; the value is never echoed. Secret is the only
	// credential New sends: no vendor-SDK environment variable or profile
	// file supplies one. New never logs the Secret. For a
	// credential-less endpoint — a local Ollama or vLLM server — pass any
	// non-empty placeholder; New only checks that Secret is present, never
	// that the backend accepts it.
	Secret string

	// Capabilities tunes the adapter's model-table profile; nil keeps the
	// table. When non-nil, the function receives the table-derived profile
	// and returns the effective one, applied once at construction: flip a
	// single field in a closure, or return a fixed profile to pin exact
	// values for models the table does not know (adapters report
	// ContextWindow 0 for unknown models; llmkit never fabricates a
	// window). The override is plumbed into the adapter itself, so wire
	// behavior follows the effective profile, not just the reported one —
	// concretely: on a profile with Thinking=false the thinking config is
	// dropped silently; on a profile with ToolChoice=false every explicit
	// Request.ToolChoice mode is rejected with llmkit.ErrInvalidRequest
	// before the wire call (auto stays allowed); StructuredOutput=false
	// gates response_format (OpenAI) and the synthetic forced-output tool
	// (Anthropic) off; ParallelToolCalls=false installs the tool-call
	// serializer. StopSequences, TopP, TopK, and Seed gate the same way:
	// false drops the matching Request field from the wire, including
	// when the override replaces the whole profile.
	//
	// Each Type has a ceiling naming the wire-gated fields it can
	// actually send (StructuredOutput, Thinking, ToolChoice,
	// StopSequences, TopP, TopK, Seed): anthropic can send everything but
	// Seed (the Messages API has no seed parameter); openai and
	// openai-compatible can send everything but Thinking and TopK (the
	// Chat Completions API has neither); google can send all seven. An
	// effective profile that reports true for a field above its Type's
	// ceiling makes New refuse with an error wrapping
	// llmkit.ErrInvalidRequest naming the field and the Type — the
	// profile can never claim a feature the adapter cannot put on the
	// wire.
	Capabilities func(llmkit.Capabilities) llmkit.Capabilities
}

// Options tunes client construction. The zero value is valid: it uses
// default retry policy, no observer, and the default HTTP transport.
type Options struct {
	// Retry configures the shared retry wrapper. Unset schedule fields
	// (<= 0; == 0 for Jitter) are completed field-wise from retry.Default
	// ([retry.Config.Or]); the fields the caller sets are preserved.
	// MaxAttempts, BaseDelay, MaxDelay, and RequestTimeout resolve when
	// <= 0; Jitter resolves when == 0 (an explicit 0 is unset, like
	// every other zero field — to run the resolved defaults with no
	// jitter, set Retry.Rand to a function returning 0.5).
	Retry retry.Config
	// Observer, if non-nil, receives one Attempt event per provider
	// attempt from the retry stage (failures included), tagged with the
	// resolved [llmkit.Identity], AND one Completion event per logical
	// completion the ctx does not already belong to (see
	// [llmkit.BeginCompletion]) — so a New/Wrap client run under the
	// agent Runner reports Attempts only, the Runner's Completion having
	// already claimed the ctx. It is the only observer a stack feeds
	// Completion events: an [llmkit.Observe] wrapped around the client
	// before it was handed to [Wrap] sits below the retry stage, finds
	// every attempt already claimed, and emits nothing — pass that
	// observer here instead.
	Observer llmkit.Observer
	// Provider overrides the provider tag on the resolved Identity. Empty
	// tags events with string(spec.Type) for New, or with the wrapped
	// client's own IdentityOf(c).Provider for Wrap; set it when your
	// ledger keys on a config-map name rather than the provider type.
	Provider string
	// HTTPClient overrides the transport used by the underlying SDKs.
	// Primarily for tests (httptest) and proxies. Headers that its
	// Transport adds reach the wire. nil gives each adapter its own plain
	// http.Client, which sends through http.DefaultTransport and never
	// through http.DefaultClient; Retry's RequestTimeout bounds each
	// attempt. Wrap ignores this field: it decorates an already-built
	// client and never touches a transport.
	HTTPClient *http.Client
}

// New builds a fully-wrapped Client for the given provider spec: the same
// stack [Wrap] documents, over a freshly built adapter, with Identity
// resolved as {opts.Provider or string(spec.Type), spec.Model}
// (spec.Secret is never part of it).
//
// spec.Secret is the resolved credential (callers obtain it via their own
// config); New performs no network I/O and is testable without real keys.
// New reads the process environment and the Anthropic SDK profile files
// only to decide the empty-BaseURL refusal (see [Spec.BaseURL]). For
// TypeGoogle, the GenAI SDK also reads its own GOOGLE_* and GEMINI_*
// variables while New builds the client, and it logs a warning when both
// GOOGLE_API_KEY and GEMINI_API_KEY are set to non-empty values. No value
// read from either source sets a request's host or headers.
// spec.Auth routes the secret to the right credential field.
//
// New returns an error wrapping ErrInvalidRequest for any of:
//
//   - an unknown spec.Auth value;
//   - AuthOAuthToken on a non-Anthropic spec.Type;
//   - an empty spec.Model;
//   - a spec.Secret that is empty, whitespace-only, or differs from its own
//     strings.TrimSpace;
//   - an empty spec.BaseURL on TypeOpenAICompatible;
//   - an empty spec.BaseURL when the Type's base-URL environment variable
//     is present, or, for TypeAnthropic, when the Anthropic SDK profile
//     file would supply a base_url (see [Spec.BaseURL]);
//   - an unknown spec.Type;
//   - a spec.Capabilities override (or the table it left untouched) that
//     reports true for a wire-gated field above the Type's ceiling — see
//     [Spec.Capabilities].
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
	// A deployment that the base-URL variable, or an Anthropic SDK profile
	// file, routed to a gateway must not have its Secret silently sent to
	// the vendor's default host. The adapters read neither source (see
	// [Spec.BaseURL]), so New refuses instead.
	if spec.BaseURL == "" {
		var baseURLVar string
		switch spec.Type {
		case TypeOpenAI:
			baseURLVar = "OPENAI_BASE_URL"
		case TypeAnthropic:
			baseURLVar = "ANTHROPIC_BASE_URL"
		case TypeGoogle:
			baseURLVar = "GOOGLE_GEMINI_BASE_URL"
		}
		if _, set := os.LookupEnv(baseURLVar); set && baseURLVar != "" {
			return nil, fmt.Errorf("llmkit: %s is set but Spec.BaseURL is empty for provider %q: set Spec.BaseURL explicitly — New no longer falls back to %s: %w",
				baseURLVar, spec.Type, baseURLVar, llmkit.ErrInvalidRequest)
		}
		if spec.Type == TypeAnthropic {
			if path := anthropicProfileBaseURLFile(); path != "" {
				return nil, fmt.Errorf("llmkit: Anthropic profile file %q sets base_url but Spec.BaseURL is empty for provider %q: set Spec.BaseURL explicitly — New does not send requests to a profile's base_url: %w",
					path, spec.Type, llmkit.ErrInvalidRequest)
			}
		}
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
		a, err := anthropic.New(spec.Model, aopts)
		if err != nil {
			return nil, err
		}
		adapter = a
	case TypeOpenAI:
		a, err := openai.New(spec.Model, openai.Options{
			APIKey:       spec.Secret,
			BaseURL:      spec.BaseURL,
			HTTPClient:   opts.HTTPClient,
			Capabilities: spec.Capabilities,
		})
		if err != nil {
			return nil, err
		}
		adapter = a
	case TypeOpenAICompatible:
		a, err := openai.New(spec.Model, openai.Options{
			APIKey:       spec.Secret,
			BaseURL:      spec.BaseURL,
			HTTPClient:   opts.HTTPClient,
			Compatible:   true,
			Capabilities: spec.Capabilities,
		})
		if err != nil {
			return nil, err
		}
		adapter = a
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

	identity := llmkit.Identity{Provider: providerTag(spec, opts), Model: spec.Model}
	return wrap(adapter, opts, identity), nil
}

// Wrap decorates c with the provider stack, outer to inner:
//
//	[completion emitter] -> serialize -> retry[Attempt events] -> c
//
// Identity is [llmkit.IdentityOf](c), with Provider replaced by
// opts.Provider when set; a foreign c that does not implement
// [llmkit.IdentifiedClient] gets the zero Identity when opts.Provider
// is also empty (documented, not an error — Wrap never requires one),
// and `{opts.Provider, ""}` when only opts.Provider is set. The stack
// implements [llmkit.IdentifiedClient] and [llmkit.StreamingClient].
// Serialize is a no-op when c.Capabilities().ParallelToolCalls is true.
// Wrap ignores [Options.HTTPClient] (adapters only): the inner adapter
// and its transport are kept, so a [New] client built with HTTPClient
// o1 keeps o1 across every Wrap.
//
// Wrap is idempotent over its own stacks: when c is a client [New] or
// Wrap returned, Wrap rebuilds one stack from c's base with the new opts
// — Retry, Observer, and Provider (when set) apply; the inner adapter
// and its transport are kept, and c's Identity carries through unless
// opts.Provider is set. Wrap(New(spec, o1), o2) therefore runs one
// retry loop under o2.Retry: one call makes at most o2.Retry's
// resolved MaxAttempts (a zero Retry resolves via [retry.Config.Or] to
// retry.Default's 4), never the product of the two schedules. A
// foreign decorator between the two stacks hides the inner one, and
// both retry.
//
// The stack claims every call whose ctx carries no span (see
// [llmkit.BeginCompletion]) above its retry stage, whether or not
// opts.Observer is set, so a call yields at most one Completion event
// across the stack. An [llmkit.Observe] below the stack — c itself, or
// one c calls with the ctx it was handed — is therefore silenced:
// Wrap(llmkit.Observe(c, obs), Options{}) delivers no Completion event
// to obs, which drops out of the spend ledger. Pass obs as
// opts.Observer instead.
//
// Wrap is not like-for-like with a bare retry wrapper — it also
// serializes tool calls when ParallelToolCalls is false, a step a caller
// retrying a foreign Client by hand would otherwise have to add.
func Wrap(c llmkit.Client, opts Options) llmkit.Client {
	identity := llmkit.IdentityOf(c)
	if opts.Provider != "" {
		identity.Provider = opts.Provider
	}
	if s, ok := c.(*stackClient); ok {
		c = s.base
	}
	return wrap(c, opts, identity)
}

// wrap builds the stack for a fixed identity, outer to inner:
// stackClient (outermost — Wrap's handle on c, and the claim when no
// observer is wired) -> completion emitter (llmkit.Observe, only when
// opts.Observer is set; otherwise the stack mints the span and emits no
// Completion event) -> serialize -> retry (carrying identity, the
// innermost provider stage). The caller's partial Retry config keeps
// its set fields and takes retry.Default's schedule for everything it
// leaves unset ([retry.Config.Or]).
func wrap(c llmkit.Client, opts Options, identity llmkit.Identity) llmkit.Client {
	retryCfg := opts.Retry.Or(retry.Default())
	stack := retryStage(c, retryCfg, opts.Observer, identity)
	stack = serializeStage(stack)
	return &stackClient{
		base:     c,
		identity: identity,
		next:     llmkit.Observe(stack, opts.Observer),
		claims:   opts.Observer == nil,
	}
}

// anthropicProfileBaseURLFile returns the path of the Anthropic SDK profile
// file whose base_url anthropic.DefaultClientOptions (anthropic-sdk-go
// v1.58.0) would apply to this process's clients, or "" when it would apply
// none. It follows the SDK's order using its own config loaders:
//
//  1. A non-empty ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN ends the lookup
//     before any profile is read.
//  2. A non-empty ANTHROPIC_PROFILE loads that profile only; a load error
//     contributes no base_url.
//  3. Complete environment federation (ANTHROPIC_FEDERATION_RULE_ID and
//     ANTHROPIC_ORGANIZATION_ID present, plus a non-empty
//     ANTHROPIC_IDENTITY_TOKEN_FILE or ANTHROPIC_IDENTITY_TOKEN) skips the
//     fallback profile.
//  4. Otherwise the fallback profile loads: the active_config file names
//     it, else "default".
//
// New calls it only after refusing a present ANTHROPIC_BASE_URL, which the
// SDK would apply ahead of any profile. It reads files and sends nothing.
func anthropicProfileBaseURLFile() string {
	if os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" {
		return ""
	}
	dir := config.DefaultDir()
	if profile := os.Getenv("ANTHROPIC_PROFILE"); profile != "" {
		if cfg, err := config.LoadProfile(dir, profile); err == nil && cfg.BaseURL != "" {
			return config.ProfilePath(dir, profile)
		}
		return ""
	}
	_, fedRule := os.LookupEnv("ANTHROPIC_FEDERATION_RULE_ID")
	_, orgID := os.LookupEnv("ANTHROPIC_ORGANIZATION_ID")
	token := os.Getenv("ANTHROPIC_IDENTITY_TOKEN_FILE") != "" || os.Getenv("ANTHROPIC_IDENTITY_TOKEN") != ""
	if fedRule && orgID && token {
		return ""
	}
	cfg, err := config.LoadConfig()
	if err != nil || cfg.BaseURL == "" {
		return ""
	}
	// LoadConfig succeeded, so ANTHROPIC_PROFILE is absent: an empty value
	// fails the SDK's profile-name check. The two remaining tiers of the
	// SDK's unexported resolveProfile name the file.
	profile := "default"
	if data, err := os.ReadFile(config.ActiveConfigPath(dir)); err == nil {
		if name := strings.TrimSpace(string(data)); name != "" {
			profile = name
		}
	}
	return config.ProfilePath(dir, profile)
}

// providerTag resolves the provider identity tag: opts.Provider when set,
// else string(spec.Type). It is unexported because no caller needs its
// own copy — New calls it directly and Wrap reads IdentityOf(c).
func providerTag(spec Spec, opts Options) string {
	if opts.Provider != "" {
		return opts.Provider
	}
	return string(spec.Type)
}

package decide

import (
	"fmt"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
	"net/http"
	"strings"
	"time"
)

const (
	// defaultBaseURL is the TypeSafe production endpoint root. Config.BaseURL
	// overrides it per client (for tests and gateways); "/v1/systemone" is
	// appended in both cases.
	defaultBaseURL = "https://api.typesafe.ai"

	// systemOnePath is appended to the base URL for every Ask.
	systemOnePath = "/v1/systemone"

	// defaultMaxAttempts is the decide default for Retry.MaxAttempts. Jev
	// answers in under a second, so three attempts keep the worst case tight.
	defaultMaxAttempts = 3

	// defaultRequestTimeout bounds a single Ask attempt when
	// Retry.RequestTimeout is unset. Tightened from the kit's 5m because a
	// decision response is far lighter than an LLM completion. Worst case
	// before giving up is roughly MaxAttempts * (30s + backoff).
	defaultRequestTimeout = 30 * time.Second

	// providerName tags every error and usage event from this package.
	providerName = "typesafe"
)

// Config configures a TypeSafe System One client. New validates it.
type Config struct {
	// APIKey is the TypeSafe credential, sent as "Authorization: Bearer".
	// Required: a non-empty value without surrounding whitespace (the same
	// rule as provider.Spec.Secret). New never echoes it, logs it, or reads
	// it from the environment.
	APIKey string

	// Model is the Jev model id or alias to ask ("jev-1.13.0",
	// "jev-latest", ...). Required; there is no default alias. The response
	// reports the versioned id actually used.
	Model string

	// BaseURL overrides the vendor endpoint root. Empty selects
	// "https://api.typesafe.ai"; the "/v1/systemone" path is appended.
	BaseURL string

	// HTTPClient is optional. nil selects a plain client with no
	// http.Client.Timeout, so Config.Retry.RequestTimeout is the only
	// per-attempt bound. An injected client keeps its own Timeout.
	HTTPClient *http.Client

	// Retry tunes the shared retry policy. New completes it field-wise
	// from decide's defaults (3 attempts, 30s per-attempt RequestTimeout;
	// BaseDelay, MaxDelay, Jitter from [retry.Default]). MaxAttempts,
	// BaseDelay, MaxDelay, and RequestTimeout resolve when <= 0; Jitter
	// resolves when == 0 (an explicit 0 is unset, like every other zero
	// field — to run the resolved defaults with no jitter, set
	// Retry.Rand to a function returning 0.5).
	Retry retry.Config

	// Recorder is optional. A non-nil Recorder receives exactly one
	// UsageEvent per successful Ask — Provider "typesafe", the response's
	// reported Model, and the token usage — and nothing on failure.
	Recorder llmkit.Recorder
}

// New validates cfg and returns a Client. It returns an error wrapping
// llmkit.ErrInvalidRequest for an APIKey that is empty, whitespace-only, or
// whitespace-padded, an empty Model, and a Retry.Jitter outside [0, 1].
// Construction is hermetic: no network I/O, no environment lookups, and
// the error never echoes the API key.
func New(cfg Config) (*Client, error) {
	// Refuse an APIKey that is empty or differs from strings.TrimSpace —
	// the same refusal provider.New applies to Spec.Secret. No vendor
	// issues a credential with surrounding whitespace, and such a value
	// is almost always a copy/paste or `cat`/`pass` artifact; refusing it
	// here turns a confusing 401 into an immediate, actionable one. The
	// error never echoes the key.
	if trimmed := strings.TrimSpace(cfg.APIKey); trimmed == "" || trimmed != cfg.APIKey {
		return nil, fmt.Errorf("decide: API key must be a non-empty value with no leading or trailing whitespace: %w", llmkit.ErrInvalidRequest)
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("decide: model is required: %w", llmkit.ErrInvalidRequest)
	}
	if cfg.Retry.Jitter < 0 || cfg.Retry.Jitter > 1 {
		return nil, fmt.Errorf("decide: retry jitter must be in [0, 1], got %v: %w", cfg.Retry.Jitter, llmkit.ErrInvalidRequest)
	}
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	return &Client{
		apiKey:   cfg.APIKey,
		model:    cfg.Model,
		endpoint: strings.TrimRight(base, "/") + systemOnePath,
		client:   cfg.httpClient(),
		retry:    cfg.retryPolicy(),
		recorder: cfg.Recorder,
	}, nil
}

// Client evaluates questions against a state with TypeSafe's Jev (System
// One) API. Construct with [New]; a *Client is safe for concurrent use and
// its fields are immutable afterwards.
type Client struct {
	apiKey   string
	model    string
	endpoint string
	client   *http.Client
	retry    retry.Config
	recorder llmkit.Recorder
}

// httpClient returns the injected client as-is, or a plain client with no
// Timeout. With the plain client, the retry loop bounds each round trip
// with the per-attempt Retry.RequestTimeout deadline. An injected client
// keeps its own Timeout, which can end an attempt earlier. The returned
// client is read-only.
func (c Config) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{}
}

// defaultRetry is decide's retry policy baseline: [retry.Default] with
// MaxAttempts set to 3 and RequestTimeout to 30s (Jev answers in under
// a second, so a shorter budget keeps the worst case tight);
// BaseDelay, MaxDelay, and Jitter come from retry.Default as-is.
func defaultRetry() retry.Config {
	d := retry.Default()
	d.MaxAttempts = defaultMaxAttempts
	d.RequestTimeout = defaultRequestTimeout
	return d
}

// retryPolicy completes cfg.Retry with decide's defaults via
// [retry.Config.Or]: MaxAttempts, BaseDelay, MaxDelay, and RequestTimeout
// resolve when <= 0; Jitter resolves when == 0 (an explicit 0 is
// unset, like every other zero field).
func (c Config) retryPolicy() retry.Config {
	return c.Retry.Or(defaultRetry())
}

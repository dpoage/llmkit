package adapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// Capability matrix (deliberate, not a bug):
//   Structured output: Anthropic and Google drop the synthetic tool
//     injection when the caller also supplies tools (Google cannot mix
//     structured output with function-calling at all; Anthropic's
//     tool_choice + synthetic tool is ambiguous). OpenAI honors
//     response_format alongside tools. The caller's repair round-trip
//     drops tools to restore native schema, then re-adds them.
//   CacheCreationInputTokens: only Anthropic exposes a cache-write
//     primitive; the field is 0 on OpenAI and Google Usage.
//   MaxTokens: zero (or negative) req.MaxTokens sends
//     llmkit.DefaultMaxTokens — max_tokens on Anthropic,
//     max_completion_tokens on OpenAI, maxOutputTokens on Google.
//     Explicit values pass through verbatim.

// ParseToolParameters returns the JSON-Schema "properties" and "required"
// keys in the Anthropic SDK's wire shape (the SDK's ToolInputSchemaParam
// carries the decoded properties directly, not an envelope). OpenAI and
// Google adapters, which expect the envelope on the wire, call this for
// validation only. A zero-length input returns nil, nil, nil.
func ParseToolParameters(params json.RawMessage) (properties map[string]any, required []string, err error) {
	if len(params) == 0 {
		return nil, nil, nil
	}
	var raw map[string]any
	if err := json.Unmarshal(params, &raw); err != nil {
		return nil, nil, err
	}

	if p, ok := raw["properties"]; ok {
		if pm, ok := p.(map[string]any); ok {
			properties = pm
		}
	} else {
		// Caller passed a bare properties object rather than a full schema.
		properties = raw
	}
	if r, ok := raw["required"]; ok {
		if rs, ok := r.([]any); ok {
			for _, v := range rs {
				if s, ok := v.(string); ok {
					required = append(required, s)
				}
			}
		}
	}
	return properties, required, nil
}

// ParseResponseSchema returns the unmarshaled schema and exactly the
// caller-supplied defaultName — the helper does not inspect the schema
// for a "name" key, so each adapter decides the name itself.
func ParseResponseSchema(raw json.RawMessage, defaultName string) (schema any, name string, err error) {
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil, "", err
	}
	return schema, defaultName, nil
}

// VendorError is what an SDK reported for one failed call, before
// classification.
type VendorError struct {
	// Status is the HTTP status of the response; 0 when none was received.
	Status int
	// Type is the vendor error type from the body ("overloaded_error"); ""
	// when absent.
	Type string
	// Message is the full response body. Classification reads it (a 400's
	// context-length phrase can sit past any cap), so callers pass the
	// full body here and cap the returned APIError.Message themselves.
	Message string
	// Header carries the response headers (Retry-After); nil when the SDK
	// hides them.
	Header http.Header
	// Err is the SDK error, chained for Unwrap.
	Err error
}

// vendorKind maps an in-band vendor error type onto its llmkit sentinel
// for sub-400 status errors and post-2xx stream error events. Sources
// for the type strings:
//   - Anthropic: the shared.ErrorType constants
//     (anthropic-sdk-go@v1.58.0/shared/shared.go:218-229);
//   - OpenAI: https://platform.openai.com/docs/guides/error-codes —
//     server_error and insufficient_quota are documented there, not SDK
//     constants.
var vendorKind = map[string]error{
	"overloaded_error":      llmkit.ErrOverloaded,
	"rate_limit_error":      llmkit.ErrRateLimited,
	"api_error":             llmkit.ErrServer,
	"server_error":          llmkit.ErrServer,
	"timeout_error":         llmkit.ErrServer,
	"invalid_request_error": llmkit.ErrInvalidRequest,
	"not_found_error":       llmkit.ErrInvalidRequest,
	"authentication_error":  llmkit.ErrAuth,
	"permission_error":      llmkit.ErrAuth,
	"billing_error":         llmkit.ErrAuth,
	"insufficient_quota":    llmkit.ErrAuth,
}

// NormalizeSDKError classifies one VendorError and wraps the result as a
// *llmkit.APIError. A status >= 400 classifies from the status
// ([ClassifyStatus], body-disambiguated); a status < 400 classifies from
// the vendor type in band, falling back to ErrServer (a sub-400 failure
// the vendor did not type is a server-class failure). Retry-After is
// parsed for every status: RetryAfter/HasRetryAfter are set whenever the
// header is present and parses. v.Err is preserved for Unwrap chaining.
func NormalizeSDKError(provider string, v VendorError) error {
	var kind error
	if v.Status >= 400 {
		kind = ClassifyStatus(v.Status, v.Message)
	} else {
		kind = llmkit.ErrServer
		if k, ok := vendorKind[v.Type]; ok {
			kind = k
		}
	}
	var ra time.Duration
	var hasRA bool
	if v.Header != nil {
		if d, ok := retry.ParseRetryAfter(v.Header.Get("Retry-After"), time.Now()); ok {
			ra, hasRA = d, true
		}
	}
	return &llmkit.APIError{
		Kind:          kind,
		StatusCode:    v.Status,
		RetryAfter:    ra,
		HasRetryAfter: hasRA,
		Provider:      provider,
		Message:       v.Message,
		Err:           v.Err,
	}
}

// ResponseHeader returns the headers of an SDK error's HTTP response, or nil
// when resp is nil (the SDK attached no response).
func ResponseHeader(resp *http.Response) http.Header {
	if resp == nil {
		return nil
	}
	return resp.Header
}

// TransportError normalizes a failure with no usable HTTP response: a
// dial failure, a reset, a stalled attempt reaped by the per-attempt
// RequestTimeout, the caller's context ending mid-call, or a protocol
// or decode failure on a received 200 response (adapters route those
// through it with a live ctx). A caller's cancellation is never an
// *APIError: the function returns a plain error chaining
// context.Canceled ([llmkit.Classify] treats it as terminal — retrying
// would run against a dead context). Every other case, including a
// DeadlineExceeded from the per-attempt RequestTimeout, is a retryable
// *llmkit.APIError{Kind: ErrServer, StatusCode: 0} with the SDK error
// chained, so errors.Is still reaches the context cause.
func TransportError(provider string, ctx context.Context, err error) error {
	if cerr := ctx.Err(); errors.Is(cerr, context.Canceled) {
		// Chain the cancellation so errors.Is finds Canceled even when the
		// SDK's error does not carry it — but wrap an SDK error that
		// already chains Canceled once, so the text names the provider
		// once and the cause once instead of repeating "context canceled".
		if errors.Is(err, context.Canceled) {
			return fmt.Errorf("llmkit: %s: %w", provider, err)
		}
		return fmt.Errorf("llmkit: %s: %w: %w", provider, err, context.Canceled)
	}
	return &llmkit.APIError{
		Kind:     llmkit.ErrServer,
		Provider: provider,
		Message:  err.Error(),
		Err:      err,
	}
}

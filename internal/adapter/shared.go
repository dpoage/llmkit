package adapter

import (
	"encoding/json"
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

// NormalizeSDKError classifies the status, parses Retry-After when the
// kind warrants it, and wraps the result as a *llmkit.APIError. underlying
// is preserved for Unwrap chaining. resp is nil when the SDK error
// carries no *http.Response (e.g. genai.APIError); the retry wrapper then
// falls back to exponential backoff since Retry-After is unavailable.
func NormalizeSDKError(provider string, status int, msg string, resp *http.Response, underlying error) error {
	kind := ClassifyStatus(status, msg)
	var ra time.Duration
	if kind == llmkit.ErrRateLimited || kind == llmkit.ErrOverloaded {
		// Absent Retry-After leaves ra at 0.
		if resp != nil {
			ra, _ = retry.ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
		}
	}
	return &llmkit.APIError{Kind: kind, StatusCode: status, RetryAfter: ra, Provider: provider, Message: msg, Err: underlying}
}

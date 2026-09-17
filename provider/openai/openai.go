// Package openai adapts the normalized llmkit types onto the OpenAI Go SDK's
// Chat Completions API, for both first-party OpenAI and arbitrary
// OpenAI-compatible endpoints.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// openaiAdapter maps the normalized types onto the OpenAI Go SDK's Chat
// Completions API. It backs both the "openai" provider and the
// "openai-compatible" provider (Ollama/vLLM/Groq/etc.) — the only difference is
// the base URL and capability profile, set at construction.
type openaiAdapter struct {
	client   openai.Client
	model    string
	caps     llmkit.Capabilities
	provider string // "openai" or "openai-compatible", for error tagging
	// requireBoolAdditionalProps forces every object/array-valued
	// "additionalProperties" in outbound tool-parameter and response_format
	// schemas down to a permissive boolean. Strict openai-compatible schema
	// validators (notably MiniMax) reject the JSON-Schema subschema form with a
	// 400 before any tokens are generated. Off for first-party OpenAI, which
	// accepts the subschema form in lenient mode. See
	// adapter.CoerceBoolAdditionalProperties.
	requireBoolAdditionalProps bool
}

// Options configures an OpenAI adapter.
//
// Compatible selects the openai-compatible profile: conservative capability
// defaults (no parallel tool calls, no caching, StructuredOutput=false),
// boolean downgrades for object-valued additionalProperties, and the
// "openai-compatible" provider label on errors.
//
// StructuredOutput, when non-nil, overrides the profile's StructuredOutput
// capability. nil = profile default (true for first-party OpenAI, false for
// openai-compatible).
type Options struct {
	APIKey     string
	BaseURL    string       // optional; for testing or non-default endpoints
	HTTPClient *http.Client // optional; for testing (httptest)

	Compatible       bool
	StructuredOutput *bool
}

// New builds an OpenAI-backed Client. With a custom BaseURL it serves any
// OpenAI-compatible endpoint (set Compatible=true). The vendor SDK's retries
// are disabled so the shared retry wrapper governs retry policy.
func New(model string, opts Options) llmkit.Client {
	reqOpts := []option.RequestOption{
		option.WithAPIKey(opts.APIKey),
		option.WithMaxRetries(0),
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	if opts.HTTPClient != nil {
		reqOpts = append(reqOpts, option.WithHTTPClient(opts.HTTPClient))
	}
	provider := "openai"
	caps := openAICapabilities(model)
	requireBoolAdditionalProps := false
	if opts.Compatible {
		provider = "openai-compatible"
		caps = openAICompatibleCapabilities(model)
		// Strict openai-compatible validators (MiniMax) reject object-valued
		// additionalProperties; downgrade it to a boolean on the wire. See
		// adapter.CoerceBoolAdditionalProperties. First-party OpenAI keeps the
		// subschema.
		requireBoolAdditionalProps = true
	}
	// The conservative default for openai-compatible endpoints is
	// StructuredOutput=false; the option can flip it on (e.g. for a
	// MiniMax-style endpoint that supports it). The override is also applied
	// to first-party OpenAI for symmetry — its default is already true, so
	// flipping it off is the only meaningful change, and that flows through
	// the same code path.
	if opts.StructuredOutput != nil {
		caps.StructuredOutput = *opts.StructuredOutput
	}
	return &openaiAdapter{
		client:                     openai.NewClient(reqOpts...),
		model:                      model,
		caps:                       caps,
		provider:                   provider,
		requireBoolAdditionalProps: requireBoolAdditionalProps,
	}
}

func (o *openaiAdapter) Capabilities() llmkit.Capabilities { return o.caps }

func (o *openaiAdapter) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	params, err := o.buildParams(req)
	if err != nil {
		return llmkit.Response{}, err
	}
	cc, err := o.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return llmkit.Response{}, o.normalizeErr(err)
	}
	return o.toResponse(cc), nil
}

func (o *openaiAdapter) buildParams(req llmkit.Request) (openai.ChatCompletionNewParams, error) {
	params := openai.ChatCompletionNewParams{
		Model: shared.ChatModel(o.model),
	}
	if req.MaxTokens > 0 {
		params.MaxCompletionTokens = openai.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}

	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, openai.SystemMessage(req.System))
	}
	converted, err := toOpenAIMessages(req.Messages)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	msgs = append(msgs, converted...)
	params.Messages = msgs

	if len(req.Tools) > 0 {
		tools := make([]openai.ChatCompletionToolUnionParam, 0, len(req.Tools))
		for _, t := range req.Tools {
			fn := shared.FunctionDefinitionParam{Name: t.Name}
			if t.Description != "" {
				fn.Description = openai.String(t.Description)
			}
			if len(t.Parameters) > 0 {
				var schema map[string]any
				if err := json.Unmarshal(t.Parameters, &schema); err != nil {
					return openai.ChatCompletionNewParams{}, llmkit.NewAPIError(o.provider, 0, 0,
						llmkit.ErrInvalidRequest, "tool "+t.Name+": invalid parameters JSON schema", err)
				}
				if o.requireBoolAdditionalProps {
					adapter.CoerceBoolAdditionalProperties(schema)
				}
				fn.Parameters = shared.FunctionParameters(schema)
			}
			tools = append(tools, openai.ChatCompletionFunctionTool(fn))
		}
		params.Tools = tools
	}

	// Schema-constrained output. Honored even when tools are present
	// (OpenAI permits response_format alongside tools; compatible backends
	// that don't will surface a 4xx, which is the correct failure mode
	// rather than silent loss). Skipped when the capability is off so a
	// provider with StructuredOutput=false (e.g. conservative
	// openai-compatible endpoint without opt-in) does not get a schema it
	// can't honor.
	if len(req.ResponseSchema) > 0 && o.caps.StructuredOutput {
		// ParseResponseSchema unmarshals the JSON Schema and returns the
		// resolved name (caller's name, or "response" if unset). The OpenAI
		// SDK expects the verbatim schema object, not the properties/required
		// unwrap that Anthropic's ToolInputSchemaParam requires, so we
		// deliberately do NOT call ParseToolParameters here — the on-the-wire
		// shape would change.
		defaultName := req.ResponseSchemaName
		if defaultName == "" {
			defaultName = "response"
		}
		schema, name, err := adapter.ParseResponseSchema(req.ResponseSchema, defaultName)
		if err != nil {
			return openai.ChatCompletionNewParams{}, llmkit.NewAPIError(o.provider, 0, 0,
				llmkit.ErrInvalidRequest, "ResponseSchema: invalid JSON", err)
		}
		if o.requireBoolAdditionalProps {
			schema = adapter.CoerceBoolAdditionalProperties(schema)
		}
		// strict=false: our schemas are not strict-mode-clean (extra fields,
		// union types) and lenient mode still drives grammar-constrained
		// decoding on OpenAI and on compatible backends that implement it.
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   name,
					Schema: schema,
				},
			},
		}
	}
	return params, nil
}

func toOpenAIMessages(msgs []llmkit.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case llmkit.RoleSystem:
			out = append(out, openai.SystemMessage(m.Content))
		case llmkit.RoleUser:
			out = append(out, openai.UserMessage(m.Content))
		case llmkit.RoleAssistant:
			am := openai.ChatCompletionAssistantMessageParam{}
			if m.Content != "" {
				am.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(m.Content),
				}
			}
			for _, tc := range m.ToolCalls {
				args := string(tc.Arguments)
				if args == "" {
					args = "{}"
				}
				am.ToolCalls = append(am.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID: tc.ID,
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: args,
						},
					},
				})
			}
			out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: &am})
		case llmkit.RoleToolResult:
			out = append(out, openai.ToolMessage(m.Content, m.ToolCallID))
		default:
			return nil, llmkit.NewAPIError("openai", 0, 0, llmkit.ErrInvalidRequest,
				"unknown message role "+string(m.Role), nil)
		}
	}
	return out, nil
}

func (o *openaiAdapter) toResponse(cc *openai.ChatCompletion) llmkit.Response {
	var resp llmkit.Response
	if len(cc.Choices) > 0 {
		choice := cc.Choices[0]
		resp.Text = choice.Message.Content
		for _, tc := range choice.Message.ToolCalls {
			// Only function tool calls carry an arguments payload we surface.
			resp.ToolCalls = append(resp.ToolCalls, llmkit.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			})
		}
		resp.StopReason = mapOpenAIStop(choice.FinishReason, len(resp.ToolCalls) > 0)
	}
	// prompt_tokens already INCLUDES cached tokens (OpenAI convention), which
	// matches the normalized Usage semantics directly. Caching is automatic
	// server-side for OpenAI; prompt_tokens_details.cached_tokens reports the
	// discounted subset. OpenAI-compatible endpoints (Ollama/vLLM/MiniMax/...)
	// often omit prompt_tokens_details entirely, in which case the SDK leaves
	// CachedTokens at zero — exactly the "no cache activity" value.
	resp.Usage = llmkit.Usage{
		InputTokens:          cc.Usage.PromptTokens,
		OutputTokens:         cc.Usage.CompletionTokens,
		CacheReadInputTokens: cc.Usage.PromptTokensDetails.CachedTokens,
	}
	return resp
}

func mapOpenAIStop(reason string, hasToolCalls bool) llmkit.StopReason {
	switch reason {
	case "stop":
		return llmkit.StopEndTurn
	case "tool_calls", "function_call":
		return llmkit.StopToolUse
	case "length":
		return llmkit.StopMaxTokens
	case "content_filter":
		return llmkit.StopError
	default:
		// Some OpenAI-compatible servers return tool calls with an empty or
		// nonstandard finish_reason; trust the presence of tool calls.
		if hasToolCalls {
			return llmkit.StopToolUse
		}
		return llmkit.StopEndTurn
	}
}

func (o *openaiAdapter) normalizeErr(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return adapter.NormalizeSDKError(o.provider, apiErr.StatusCode, apiErr.Error(), apiErr.Response, err)
	}
	return llmkit.NewAPIError(o.provider, 0, 0, llmkit.ErrServer, err.Error(), err)
}

// openAICapabilities returns the capability profile for first-party OpenAI
// models. Context window is per-model where known, with a 128k default for
// unrecognized models.
func openAICapabilities(model string) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     openAIContextWindow(model),
		ParallelToolCalls: true,
		PromptCaching:     true,
		StructuredOutput:  true,
	}
}

// openAIContextWindow returns the known context-window size for a first-party
// OpenAI model. Unrecognized models get the 128k default which covers the
// current GPT-4o family.
func openAIContextWindow(model string) int {
	switch model {
	case "gpt-3.5-turbo", "gpt-3.5-turbo-0125":
		return 16_385
	case "gpt-3.5-turbo-16k":
		return 16_385
	case "gpt-4", "gpt-4-0613":
		return 8_192
	case "gpt-4-32k", "gpt-4-32k-0613":
		return 32_768
	case "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview":
		return 128_000
	case "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20":
		return 128_000
	case "gpt-4o-mini", "gpt-4o-mini-2024-07-18":
		return 128_000
	case "o1", "o1-2024-12-17":
		return 200_000
	case "o1-mini", "o1-mini-2024-09-12":
		return 128_000
	case "o1-preview", "o1-preview-2024-09-12":
		return 128_000
	case "o3", "o3-2025-04-16":
		return 200_000
	case "o3-mini", "o3-mini-2025-01-31":
		return 200_000
	case "o4-mini", "o4-mini-2025-04-16":
		return 200_000
	case "gpt-4.1", "gpt-4.1-2025-04-14":
		return 1_047_576
	case "gpt-4.1-mini", "gpt-4.1-mini-2025-04-14":
		return 1_047_576
	case "gpt-4.1-nano", "gpt-4.1-nano-2025-04-14":
		return 1_047_576
	default:
		return 128_000
	}
}

// openAICompatibleCapabilities returns a conservative profile for arbitrary
// OpenAI-compatible endpoints (Ollama/vLLM/Groq/etc.). We can't know the
// backend's true capabilities, so we assume no parallel tool calls (the
// degraded path serializes them) and no caching. The adapter still parses
// usage.prompt_tokens_details.cached_tokens opportunistically when the
// endpoint reports it (e.g. MiniMax), so cache hits are ledgered even with
// PromptCaching=false. Callers can override.
//
// The model parameter is accepted for symmetry with other capability
// constructors; no per-model lookup is available for arbitrary endpoints,
// so ContextWindow is left at 0 (unknown).
func openAICompatibleCapabilities(model string) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     0, // unknown for arbitrary endpoints
		ParallelToolCalls: false,
		PromptCaching:     false,
		StructuredOutput:  false,
	}
}

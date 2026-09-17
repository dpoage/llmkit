// Package openai adapts the normalized llmkit types onto the OpenAI Go SDK's
// Chat Completions API, for both first-party OpenAI and arbitrary
// OpenAI-compatible endpoints.
package openai

import (
	"context"
	"encoding/base64"
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
// Capabilities, when non-nil, REPLACES the selected profile wholesale (all
// fields, including ContextWindow). nil = profile default.
//
// StructuredOutput, when non-nil, overrides the profile's StructuredOutput
// capability — applied after Capabilities, so it wins over an override that
// carries the field. nil = profile default (true for first-party OpenAI,
// false for openai-compatible).
type Options struct {
	APIKey     string
	BaseURL    string       // optional; for testing or non-default endpoints
	HTTPClient *http.Client // optional; for testing (httptest)

	Compatible       bool
	Capabilities     *llmkit.Capabilities
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
	if opts.Capabilities != nil {
		// Caller-provided profile replaces whichever table was selected.
		caps = *opts.Capabilities
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
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		// Uniform rule: the same documented default on every adapter instead
		// of relying on each backend's unset behavior.
		maxTokens = llmkit.DefaultMaxTokens
	}
	params.MaxCompletionTokens = openai.Int(int64(maxTokens))
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if len(req.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: req.StopSequences}
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
	}
	if req.Seed != nil {
		params.Seed = openai.Int(*req.Seed)
	}
	// Request.TopK: the Chat Completions API has no top_k parameter, so it
	// is dropped here and documented via Capabilities.TopK = false.
	// Request.Thinking: not mapped onto reasoning_effort (a coarse
	// low/medium/high dial, not a token budget), so it is dropped and
	// documented via Capabilities.Thinking = false.

	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, openai.SystemMessage(req.System))
	}
	converted, err := toOpenAIMessages(o.provider, req.Messages)
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

	// Request.ToolChoice. Auto (and the zero value) is the provider default
	// and is never serialized. Unlike Anthropic, no synthetic forcing
	// competes here: response_format coexists with tool_choice.
	if err := o.applyToolChoice(&params, req.ToolChoice); err != nil {
		return openai.ChatCompletionNewParams{}, err
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

// applyToolChoice maps the normalized tool-choice request onto the Chat
// Completions tool_choice parameter.
func (o *openaiAdapter) applyToolChoice(params *openai.ChatCompletionNewParams, tc llmkit.ToolChoice) error {
	// Capability gate: models without function calling (o1-mini/o1-preview
	// — see openAICapabilities) report ToolChoice=false, and every explicit
	// steering mode is then REJECTED before the wire call. Silently dropping
	// "none" would escalate permissions (the model stays free to call the
	// offered tools) and dropping "required"/"tool" would silently degrade;
	// auto and the zero value are the provider default and pass. An
	// unrecognized mode falls through so the mapper below produces its
	// precise unknown-mode error.
	if err := adapter.GateToolChoice(o.provider, tc, o.caps.ToolChoice); err != nil {
		return err
	}
	switch tc.Mode {
	case "", llmkit.ToolChoiceAuto:
		return nil
	case llmkit.ToolChoiceNone:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("none"),
		}
	case llmkit.ToolChoiceRequired:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("required"),
		}
	case llmkit.ToolChoiceTool:
		if tc.Name == "" {
			return llmkit.NewAPIError(o.provider, 0, 0, llmkit.ErrInvalidRequest,
				"ToolChoice.Mode=tool requires ToolChoice.Name", nil)
		}
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: tc.Name},
			},
		}
	default:
		return llmkit.NewAPIError(o.provider, 0, 0, llmkit.ErrInvalidRequest,
			"unknown ToolChoice.Mode "+string(tc.Mode), nil)
	}
	return nil
}

// toOpenAIMessages converts normalized messages into Chat Completions
// messages. Content blocks map in order: text → text part, image →
// image_url part, document → file part; assistant and tool-result messages
// carry their concatenated text.
//
// Per-role block rule (ValidateMessageBlocks, before any mapping):
// user text/image/document; assistant text/thinking (no thinking is
// re-emitted on OpenAI — Capabilities.Thinking=false); system and
// tool-result text only. Violations are ErrInvalidRequest. Image/document source validation happens
// here, BEFORE any wire call.
func toOpenAIMessages(provider string, msgs []llmkit.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		// Per-role block-kind rule + media source rule, before any mapping.
		// Without this, an image in a system/tool-result/assistant message
		// would be silently dropped by m.Text().
		if err := adapter.ValidateMessageBlocks(provider, m); err != nil {
			return nil, err
		}
		switch m.Role {
		case llmkit.RoleSystem:
			out = append(out, openai.SystemMessage(m.Text()))
		case llmkit.RoleUser:
			parts, err := openAIUserParts(m)
			if err != nil {
				return nil, err
			}
			if len(parts) == 1 && parts[0].OfText != nil {
				// Common text-only form: keep the flat string content the
				// API (and compatible backends) have always accepted.
				out = append(out, openai.UserMessage(parts[0].OfText.Text))
			} else {
				out = append(out, openai.UserMessage(parts))
			}
		case llmkit.RoleAssistant:
			am := openai.ChatCompletionAssistantMessageParam{}
			if text := m.Text(); text != "" {
				am.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(text),
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
			out = append(out, openai.ToolMessage(m.Text(), m.ToolCallID))
		default:
			return nil, llmkit.NewAPIError("openai", 0, 0, llmkit.ErrInvalidRequest,
				"unknown message role "+string(m.Role), nil)
		}
	}
	return out, nil
}

// dataURI renders inline block bytes as an RFC-2397 data URL, the form the
// image_url and file content parts accept for embedded media.
func dataURI(mediaType string, data []byte) string {
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

// openAIUserParts maps a user message's content blocks onto Chat Completions
// content parts: image → image_url part (data: URI for inline bytes, URL
// verbatim), document → file part (file_data as a data: URI). Document URLs
// are rejected: the Chat Completions file part has no URL source. A message
// with no blocks yields one empty text part so the wire shape of a
// previously-valid empty user turn is preserved.
func openAIUserParts(m llmkit.Message) ([]openai.ChatCompletionContentPartUnionParam, error) {
	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			parts = append(parts, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{Text: b.Text},
			})
		case llmkit.BlockImage:
			if err := adapter.ValidateMediaBlock("openai", b); err != nil {
				return nil, err
			}
			url := b.URL
			if len(b.Data) > 0 {
				url = dataURI(b.MediaType, b.Data)
			}
			parts = append(parts, openai.ChatCompletionContentPartUnionParam{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: url},
				},
			})
		case llmkit.BlockDocument:
			if err := adapter.ValidateMediaBlock("openai", b); err != nil {
				return nil, err
			}
			if b.URL != "" {
				return nil, llmkit.NewAPIError("openai", 0, 0, llmkit.ErrInvalidRequest,
					"document block: the Chat Completions file part accepts inline data only, not URLs", nil)
			}
			filename := b.Title
			if filename == "" {
				filename = "document"
			}
			parts = append(parts, openai.ChatCompletionContentPartUnionParam{
				OfFile: &openai.ChatCompletionContentPartFileParam{
					File: openai.ChatCompletionContentPartFileFileParam{
						FileData: openai.String(dataURI(b.MediaType, b.Data)),
						Filename: openai.String(filename),
					},
				},
			})
		}
	}
	if len(parts) == 0 {
		parts = append(parts, openai.ChatCompletionContentPartUnionParam{
			OfText: &openai.ChatCompletionContentPartTextParam{Text: ""},
		})
	}
	return parts, nil
}

func (o *openaiAdapter) toResponse(cc *openai.ChatCompletion) llmkit.Response {
	var resp llmkit.Response
	if len(cc.Choices) > 0 {
		choice := cc.Choices[0]
		// message.refusal: when the model declines on policy grounds OpenAI
		// puts the explanation here and leaves content empty. Surface it as
		// the response text with StopRefusal, so a caller sees WHY the turn
		// ended instead of an empty StopEndTurn.
		refused := choice.Message.Refusal != ""
		if refused {
			resp.Text = choice.Message.Refusal
		} else {
			resp.Text = choice.Message.Content
		}
		if resp.Text != "" {
			resp.Blocks = append(resp.Blocks, llmkit.Block{Kind: llmkit.BlockText, Text: resp.Text})
		}
		for _, tc := range choice.Message.ToolCalls {
			// Only function tool calls carry an arguments payload we surface.
			resp.ToolCalls = append(resp.ToolCalls, llmkit.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			})
		}
		if refused {
			resp.StopReason = llmkit.StopRefusal
		} else {
			resp.StopReason = mapOpenAIStop(choice.FinishReason, len(resp.ToolCalls) > 0)
		}
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
		return llmkit.StopContentFilter
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

// Sources (vendor docs consulted for this table):
//   - https://platform.openai.com/docs/models — per-model context windows:
//     gpt-5 / gpt-5-mini / gpt-5-nano 400,000; gpt-4.1 / -mini / -nano
//     1,047,576; gpt-4.5-preview 128,000; gpt-4o / gpt-4o-mini 128,000;
//     o1 / o3 / o3-mini / o4-mini 200,000 with o1-mini and o1-preview at
//     128,000; gpt-4-turbo (incl. the gpt-4-0125-preview and gpt-4-1106-*
//     snapshots) 128,000; gpt-4 8,192; gpt-4-32k 32,768; gpt-3.5-turbo
//     16,385.
//   - https://platform.openai.com/docs/guides/prompt-caching — automatic
//     prompt caching exists only for models introduced from Aug 2024 on
//     (gpt-4o and later, o1 and later); the gpt-4 / gpt-4-turbo /
//     gpt-3.5-turbo families predate it.
//   - https://platform.openai.com/docs/guides/structured-outputs plus the
//     per-model pages — structured outputs are unsupported on gpt-4,
//     gpt-4-32k, gpt-4-turbo, and gpt-3.5-turbo; o1-mini and o1-preview
//     support neither structured outputs nor function calling at all
//     (https://platform.openai.com/docs/models/o1-mini), so their
//     ToolChoice and ParallelToolCalls are false too.
//
// The table is matched by LONGEST key with a "-" segment boundary
// (adapter.MatchesModelFamily), so "gpt-4-turbo" and "gpt-4.1" win over the
// shorter "gpt-4" key, "o1-mini" wins over "o1", and the dotted sibling
// "gpt-4.5" needs — and has — its own key. Only the fields that genuinely
// vary by model are stored per entry (window + the
// parallel/caching/structured/tool-choice bools); everything else is shared
// Chat Completions API behavior (see firstPartyCaps).
//
// An unknown first-party model reports ContextWindow 0 — unknown is never
// fabricated into a number (the old 128k fallback overstated gpt-4's 8k
// window) — while the feature bools keep the API-level defaults: an
// unrecognized name on the first-party endpoint is most likely a NEW model
// with the modern feature set, and the server is the final validator. Pin
// exact values for unknown models via Options.Capabilities.
type openAIModelCaps struct {
	prefix string
	caps   llmkit.Capabilities
}

// firstPartyCaps fills a first-party profile: the API-level defaults with
// the per-model fields supplied by the caller.
func firstPartyCaps(window int, parallel, caching, structured, toolChoice bool) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     window,
		ParallelToolCalls: parallel,
		PromptCaching:     caching,
		StructuredOutput:  structured,
		Thinking:          false, // reasoning_effort is a coarse dial, not a budget — never mapped
		ToolChoice:        toolChoice,
		Images:            true,
		Documents:         true,
		StopSequences:     true,
		TopP:              true,
		TopK:              false, // Chat Completions has no top_k
		Seed:              true,
	}
}

var openAIModelTable = []openAIModelCaps{
	{"gpt-5", firstPartyCaps(400_000, true, true, true, true)},
	{"gpt-4.1", firstPartyCaps(1_047_576, true, true, true, true)},
	// gpt-4.5: a dotted sibling of gpt-4 — under segment matching it can
	// never inherit the gpt-4 entry, so it carries its own key.
	{"gpt-4.5", firstPartyCaps(128_000, true, true, true, true)},
	{"gpt-4o", firstPartyCaps(128_000, true, true, true, true)},
	{"gpt-4-turbo", firstPartyCaps(128_000, true, false, false, true)},
	// The GPT-4 Turbo Preview snapshots (128k, parallel function calling
	// since the 1106 generation) — without these keys the 8,192 gpt-4
	// entry would swallow them via the "-" continuation.
	{"gpt-4-0125", firstPartyCaps(128_000, true, false, false, true)},
	{"gpt-4-1106", firstPartyCaps(128_000, true, false, false, true)},
	{"gpt-4-32k", firstPartyCaps(32_768, false, false, false, true)},
	{"gpt-4", firstPartyCaps(8_192, false, false, false, true)},
	{"gpt-3.5-turbo", firstPartyCaps(16_385, true, false, false, true)},
	{"o1-mini", firstPartyCaps(128_000, false, true, false, false)},
	{"o1-preview", firstPartyCaps(128_000, false, true, false, false)},
	{"o1", firstPartyCaps(200_000, true, true, true, true)},
	{"o3", firstPartyCaps(200_000, true, true, true, true)},
	{"o4-mini", firstPartyCaps(200_000, true, true, true, true)},
}

func openAICapabilities(model string) llmkit.Capabilities {
	best := adapter.BestMatchingFamily(model, openAIModelTable, func(e openAIModelCaps) string { return e.prefix })
	if best >= 0 {
		return openAIModelTable[best].caps
	}
	return firstPartyCaps(0, true, true, true, true)
}

// OpenAI-compatible endpoints (Ollama/vLLM/Groq/etc.). We can't know the
// backend's true capabilities, so we assume no parallel tool calls (the
// degraded path serializes them) and no caching. The adapter still parses
// usage.prompt_tokens_details.cached_tokens opportunistically when the
// endpoint reports it (e.g. MiniMax), so cache hits are ledgered even with
// PromptCaching=false. Callers can override.
//
// Thinking is false: this adapter does not map Request.Thinking onto
// reasoning_effort. TopK is false: the Chat Completions API has no top_k.
// Tool choice, images, documents, stop sequences, top_p, and seed are part
// of the base Chat Completions contract and are sent when requested.
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
		Thinking:          false,
		ToolChoice:        true,
		Images:            true,
		Documents:         true,
		StopSequences:     true,
		TopP:              true,
		TopK:              false,
		Seed:              true,
	}
}

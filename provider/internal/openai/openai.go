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
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"
)

// openaiAdapter maps the normalized types onto the OpenAI Go SDK's Chat
// Completions API for both the "openai" and "openai-compatible" provider
// (Ollama/vLLM/Groq/etc.); the only difference is the base URL and capability
// profile, set at construction.
type openaiAdapter struct {
	client   openai.ChatCompletionService
	model    string
	caps     llmkit.Capabilities
	provider string // "openai" or "openai-compatible", for error tagging
	// requireBoolAdditionalProps forces every object/array-valued
	// "additionalProperties" in outbound tool-parameter and response_format
	// schemas down to a permissive boolean — strict openai-compatible schema
	// validators (notably MiniMax) reject the JSON-Schema subschema form with
	// HTTP 400 before any tokens are generated. Off for first-party OpenAI,
	// which accepts the subschema form in lenient mode. See
	// coerceBoolAdditionalProperties.
	requireBoolAdditionalProps bool
}

// Options configures an OpenAI adapter.
//
// Compatible selects the openai-compatible profile: conservative capability
// defaults (no parallel tool calls, no caching, StructuredOutput=false),
// boolean downgrades for object-valued additionalProperties, and the
// "openai-compatible" provider label on errors.
//
// Capabilities, when non-nil, receives the selected profile and returns the
// effective one, applied once at construction. Flip a single field in a
// closure, or return a fixed profile to pin exact values for models the
// tables do not know. nil = profile default. An effective profile that
// reports true for a wire-gated field above the adapter's ceiling
// (Thinking, TopK) is refused — see New.
type Options struct {
	APIKey     string
	BaseURL    string       // optional; for testing or non-default endpoints
	HTTPClient *http.Client // optional; for testing (httptest)

	Compatible   bool
	Capabilities func(llmkit.Capabilities) llmkit.Capabilities
}

// openAICeiling names every wire-gated field this adapter can put on the
// wire for both the first-party and openai-compatible profiles. The Chat
// Completions API has no reasoning-budget parameter (Thinking) and no
// top_k, so an effective profile reporting either true is refused at
// construction rather than silently mismapped.
var openAICeiling = llmkit.Capabilities{
	StructuredOutput: true,
	Thinking:         false,
	ToolChoice:       true,
	StopSequences:    true,
	TopP:             true,
	TopK:             false,
	Seed:             true,
}

// New builds an OpenAI-backed Client. With a custom BaseURL it serves any
// OpenAI-compatible endpoint (set Compatible=true). The vendor SDK's retries
// are disabled so the shared retry wrapper governs retry policy. New
// returns an error wrapping llmkit.ErrInvalidRequest, with no network I/O,
// when the effective capability profile (table or Options.Capabilities
// override) reports Thinking or TopK true — see openAICeiling.
//
// The client is built with openai.NewChatCompletionService rather than
// openai.NewClient: the service constructor never calls DefaultClientOptions,
// so it reads no environment variable and no config file (openai-go has no
// WithoutEnvironmentDefaults marker; the service constructor is the only
// env-free entry point). option.WithEnvironmentProduction supplies the vendor
// default host explicitly since the service constructor would otherwise have
// none; option.WithBaseURL always overrides it regardless of option order.
// A nil opts.HTTPClient gets an explicit &http.Client{} rather than the
// process-global client the SDK falls back to without one.
func New(model string, opts Options) (llmkit.Client, error) {
	provider := "openai"
	table := openAICapabilities(model)
	requireBoolAdditionalProps := false
	if opts.Compatible {
		provider = "openai-compatible"
		table = openAICompatibleCapabilities(model)
		// Strict openai-compatible validators (MiniMax) reject object-valued
		// additionalProperties; downgrade it to a boolean on the wire. First-party
		// OpenAI keeps the subschema.
		requireBoolAdditionalProps = true
	}
	caps, err := adapter.ApplyOverride(provider, openAICeiling, table, opts.Capabilities)
	if err != nil {
		return nil, err
	}

	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	reqOpts := []option.RequestOption{
		option.WithEnvironmentProduction(),
		option.WithAPIKey(opts.APIKey),
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(0),
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	return &openaiAdapter{
		client:                     openai.NewChatCompletionService(reqOpts...),
		model:                      model,
		caps:                       caps,
		provider:                   provider,
		requireBoolAdditionalProps: requireBoolAdditionalProps,
	}, nil
}

func (o *openaiAdapter) Capabilities() llmkit.Capabilities { return o.caps }

func (o *openaiAdapter) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	params, err := o.buildParams(req)
	if err != nil {
		return llmkit.Response{}, err
	}
	cc, err := o.client.New(ctx, params)
	if err != nil {
		return llmkit.Response{}, o.normalizeErr(ctx, err)
	}
	return o.toResponse(cc), nil
}

// buildParams applies the shared rules once via adapter.Prepare and maps the
// Prepared value onto Chat Completions params without re-checking anything
// Prepare already validated or gated.
func (o *openaiAdapter) buildParams(req llmkit.Request) (openai.ChatCompletionNewParams, error) {
	p, err := adapter.Prepare(o.provider, o.caps, req)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	params := openai.ChatCompletionNewParams{
		Model:               shared.ChatModel(o.model),
		MaxCompletionTokens: openai.Int(int64(p.MaxTokens)),
	}
	if p.Temperature != nil {
		params.Temperature = openai.Float(*p.Temperature)
	}
	if len(p.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: p.StopSequences}
	}
	if p.TopP != nil {
		params.TopP = openai.Float(*p.TopP)
	}
	if p.Seed != nil {
		params.Seed = openai.Int(*p.Seed)
	}
	// Prepared.TopK and Prepared.Thinking are always nil here: the Chat
	// Completions API has no top_k, and reasoning_effort is a coarse dial
	// rather than a token budget — never mapped. Every openai profile reports
	// both false, and Prepare drops both.

	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(p.Messages)+1)
	if p.System != "" {
		msgs = append(msgs, openai.SystemMessage(p.System))
	}
	converted, err := toOpenAIMessages(o.provider, p.Messages)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	msgs = append(msgs, converted...)
	params.Messages = msgs

	if len(p.Tools) > 0 {
		tools := make([]openai.ChatCompletionToolUnionParam, 0, len(p.Tools))
		for _, t := range p.Tools {
			fn := shared.FunctionDefinitionParam{Name: t.Def.Name}
			if t.Def.Description != "" {
				fn.Description = openai.String(t.Def.Description)
			}
			if t.Schema != nil {
				schema := t.Schema
				if o.requireBoolAdditionalProps {
					coerceBoolAdditionalProperties(schema)
				}
				fn.Parameters = shared.FunctionParameters(schema)
			}
			tools = append(tools, openai.ChatCompletionFunctionTool(fn))
		}
		params.Tools = tools
	}

	// Request.ToolChoice is already validated and gated by Prepare. Auto (and
	// the zero value) is the provider default and is never serialized.
	applyOpenAIToolChoice(&params, p.ToolChoice)

	// Schema-constrained output is honored even when tools are present: OpenAI
	// permits response_format alongside tools, and a compatible backend that
	// refuses the combination will surface a 4xx, which is the correct
	// failure mode. p.ResponseSchema is already nil when the capability is off
	// or the caller supplied none.
	if p.ResponseSchema != nil {
		name := p.ResponseSchema.Name
		if name == "" {
			name = "response"
		}
		schema := p.ResponseSchema.Value
		if o.requireBoolAdditionalProps {
			coerceBoolAdditionalProperties(schema)
		}
		// strict=false: our schemas are not strict-mode-clean (extra fields,
		// union types); lenient mode still drives grammar-constrained decoding
		// on OpenAI and on compatible backends that implement it.
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

// applyOpenAIToolChoice maps an already-validated ToolChoice onto the Chat
// Completions tool_choice parameter. Prepare has already applied the
// capability gate and validated Mode and Name; auto (the only remaining
// unmapped mode) is the provider default and is never serialized.
func applyOpenAIToolChoice(params *openai.ChatCompletionNewParams, tc llmkit.ToolChoice) {
	switch tc.Mode {
	case llmkit.ToolChoiceNone:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("none"),
		}
	case llmkit.ToolChoiceRequired:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("required"),
		}
	case llmkit.ToolChoiceTool:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: tc.Name},
			},
		}
	}
}

// toOpenAIMessages converts normalized messages into Chat Completions
// messages. Content blocks map in order: text → text part, image →
// image_url part, document → file part; assistant and tool-result messages
// carry their concatenated text. Every role and block-kind rule
// (adapter.Prepare) already holds — this is a pure mapping except for the
// vendor-specific constraint the file part imposes: no URL source for
// documents (openAIUserParts).
func toOpenAIMessages(provider string, msgs []llmkit.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case llmkit.RoleSystem:
			out = append(out, openai.SystemMessage(m.Text()))
		case llmkit.RoleUser:
			parts, err := openAIUserParts(provider, m)
			if err != nil {
				return nil, err
			}
			if len(parts) == 1 && parts[0].OfText != nil {
				// Common text-only form: keep the flat string content the
				// API and compatible backends have always accepted.
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
		}
	}
	return out, nil
}

// dataURI renders inline block bytes as an RFC-2397 data URL — the form
// the image_url and file content parts accept for embedded media.
func dataURI(mediaType string, data []byte) string {
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

// openAIUserParts maps a user message's content blocks onto Chat Completions
// content parts: image → image_url part (data: URI for inline bytes, URL
// verbatim), document → file part (file_data as a data: URI). The media
// source rule (exactly one of Data/URL, MediaType required with Data) is
// already enforced by adapter.Prepare; the constraint left here is
// vendor-specific — document URLs are rejected, since the file part has no
// URL source. A message with no blocks yields one empty text part so an
// empty user turn still serializes the same wire shape.
func openAIUserParts(provider string, m llmkit.Message) ([]openai.ChatCompletionContentPartUnionParam, error) {
	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			parts = append(parts, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{Text: b.Text},
			})
		case llmkit.BlockImage:
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
			if b.URL != "" {
				return nil, adapter.Refuse(provider,
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
		// the response text with StopRefusal so a caller sees why the turn
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
	// prompt_tokens already INCLUDES cached tokens (OpenAI convention),
	// matching the normalized Usage semantics directly. Caching is automatic
	// server-side for OpenAI; prompt_tokens_details.cached_tokens reports the
	// discounted subset. OpenAI-compatible endpoints (Ollama/vLLM/MiniMax/...)
	// often omit prompt_tokens_details entirely, leaving CachedTokens at
	// zero — exactly the "no cache activity" value.
	resp.Usage = llmkit.Usage{
		InputTokens:          cc.Usage.PromptTokens,
		OutputTokens:         cc.Usage.CompletionTokens,
		CacheReadInputTokens: cc.Usage.PromptTokensDetails.CachedTokens,
	}
	return resp
}

// mapOpenAIStop is this adapter's recognized finish_reason table: "stop" is
// tool-call-aware (a compatible backend that emits "stop" alongside
// tool_calls means the model wants to call them, not that it finished
// naturally); "tool_calls" and "function_call" always mean ToolUse; "length"
// and "content_filter" map directly. Every other reason — including "abort"
// and an empty string, which some openai-compatible servers send — falls to
// adapter.StopFallback: ToolUse when tool calls are present, StopError
// otherwise (never assumed to be a natural end-turn).
func mapOpenAIStop(reason string, hasToolCalls bool) llmkit.StopReason {
	switch reason {
	case "stop":
		if hasToolCalls {
			return llmkit.StopToolUse
		}
		return llmkit.StopEndTurn
	case "tool_calls", "function_call":
		return llmkit.StopToolUse
	case "length":
		return llmkit.StopMaxTokens
	case "content_filter":
		return llmkit.StopContentFilter
	default:
		return adapter.StopFallback(hasToolCalls)
	}
}

func (o *openaiAdapter) normalizeErr(ctx context.Context, err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return adapter.NormalizeSDKError(o.provider, adapter.VendorError{
			Status:  apiErr.StatusCode,
			Type:    apiErr.Type,
			Message: apiErr.Error(),
			Header:  adapter.ResponseHeader(apiErr.Response),
			Err:     err,
		})
	}
	// In-band SSE error: the stream was opened on a 2xx response, so the SDK
	// reports the body's error object as a StreamError with no status and no
	// headers. Status 200 is the status the wire actually returned.
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		typ, msg := decodeSSEError(streamErr.Event.Data)
		if msg == "" {
			msg = streamErr.Error()
		}
		return adapter.NormalizeSDKError(o.provider, adapter.VendorError{
			Status:  http.StatusOK,
			Type:    typ,
			Message: msg,
			Err:     err,
		})
	}
	// No HTTP response: transport failure or caller's context ending mid-call.
	return adapter.TransportError(o.provider, ctx, err)
}

// sseErrorBody is the error object an OpenAI-compatible server sends inside
// an SSE data line of an already-committed 2xx stream.
type sseErrorBody struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// decodeSSEError extracts the vendor type and message from a StreamError's
// raw Event.Data (JSON with a trailing newline); an unparseable event
// leaves both empty.
func decodeSSEError(data []byte) (typ, msg string) {
	var body sseErrorBody
	if err := json.Unmarshal(data, &body); err != nil {
		return "", ""
	}
	return body.Error.Type, body.Error.Message
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
// (adapter.BestMatchingFamily), so "gpt-4-turbo" and "gpt-4.1" win over the
// shorter "gpt-4" key, "o1-mini" wins over "o1", and the dotted sibling
// "gpt-4.5" needs — and has — its own key. Only the fields that genuinely
// vary by model are stored per entry (window + the
// parallel/caching/structured/tool-choice bools); everything else is shared
// Chat Completions API behavior (see firstPartyCaps).
//
// An unknown first-party model reports ContextWindow 0 — unknown is never
// fabricated into a number — while the feature bools keep the API-level
// defaults: an unrecognized name on the first-party endpoint is most likely
// a new model with the modern feature set, and the server is the final
// validator. Pin exact values for unknown models via Options.Capabilities.
type openAIModelCaps struct {
	prefix string
	caps   llmkit.Capabilities
}

// firstPartyCaps fills a first-party profile with the API-level defaults
// and the per-model fields supplied by the caller.
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
	// gpt-4.5: a dotted sibling of gpt-4 — segment matching can never
	// inherit the gpt-4 entry, so it carries its own key.
	{"gpt-4.5", firstPartyCaps(128_000, true, true, true, true)},
	{"gpt-4o", firstPartyCaps(128_000, true, true, true, true)},
	{"gpt-4-turbo", firstPartyCaps(128_000, true, false, false, true)},
	// The GPT-4 Turbo Preview snapshots (128k, parallel function calling
	// since the 1106 generation): without these keys the 8,192 gpt-4 entry
	// would swallow them via the "-" continuation.
	{"gpt-4-0125", firstPartyCaps(128_000, true, false, false, true)},
	{"gpt-4-1106", firstPartyCaps(128_000, true, false, false, true)},
	// gpt-4-vision: the retired vision variant of the 1106 Turbo generation
	// (128k, parallel function calling) — without its own key it would
	// inherit the 8,192 gpt-4 entry.
	{"gpt-4-vision", firstPartyCaps(128_000, true, false, false, true)},
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

// openAICompatibleCapabilities returns a conservative profile for arbitrary
// OpenAI-compatible endpoints (Ollama/vLLM/Groq/etc.). The backend's true
// capabilities are unknown, so the profile assumes no parallel tool calls
// (the degraded path serializes them) and no caching. The adapter still
// parses usage.prompt_tokens_details.cached_tokens opportunistically when
// the endpoint reports it (e.g. MiniMax), so cache hits are ledgered even
// with PromptCaching=false. Callers can override.
//
// Thinking is false (this adapter does not map Request.Thinking onto
// reasoning_effort) and TopK is false (the Chat Completions API has no
// top_k). Tool choice, images, documents, stop sequences, top_p, and seed
// are part of the base Chat Completions contract and are sent when
// requested.
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

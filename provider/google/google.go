// Package google adapts the normalized llmkit types onto the
// google.golang.org/genai SDK (Gemini API backend).
package google

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
	"google.golang.org/genai"
)

// googleAdapter maps the normalized types onto the google.golang.org/genai SDK
// (Gemini API backend).
type googleAdapter struct {
	client *genai.Client
	model  string
	caps   llmkit.Capabilities
}

// Options configures a Gemini adapter.
//
// Capabilities, when non-nil, REPLACES the adapter's model-table profile
// wholesale (all fields, including ContextWindow). nil = model-table default.
//
// StructuredOutput, when non-nil, overrides the adapter's built-in default
// (true) — applied after Capabilities, so it wins over an override that
// carries the field. nil = default.
type Options struct {
	APIKey     string
	BaseURL    string       // optional; for testing or non-default endpoints
	HTTPClient *http.Client // optional; for testing (httptest)

	Capabilities     *llmkit.Capabilities
	StructuredOutput *bool
}

// New builds a Gemini-backed Client. genai's only built-in retry path is for
// file uploads, so the shared retry wrapper is the sole retry layer for
// completions.
func New(ctx context.Context, model string, opts Options) (llmkit.Client, error) {
	cc := &genai.ClientConfig{
		APIKey:  opts.APIKey,
		Backend: genai.BackendGeminiAPI,
	}
	if opts.BaseURL != "" {
		cc.HTTPOptions.BaseURL = opts.BaseURL
	}
	if opts.HTTPClient != nil {
		cc.HTTPClient = opts.HTTPClient
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
			"failed to construct genai client: "+err.Error(), err)
	}
	caps := googleCapabilities(model)
	if opts.Capabilities != nil {
		// Caller-provided profile replaces the model table wholesale.
		caps = *opts.Capabilities
	}
	if opts.StructuredOutput != nil {
		caps.StructuredOutput = *opts.StructuredOutput
	}
	return &googleAdapter{
		client: client,
		model:  model,
		caps:   caps,
	}, nil
}

func (g *googleAdapter) Capabilities() llmkit.Capabilities { return g.caps }

func (g *googleAdapter) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	contents, err := toGoogleContents(req.Messages)
	if err != nil {
		return llmkit.Response{}, err
	}
	cfg := &genai.GenerateContentConfig{}
	if req.System != "" {
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: req.System}},
		}
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		// Uniform rule: the same documented default on every adapter instead
		// of leaving the cap unset.
		maxTokens = llmkit.DefaultMaxTokens
	}
	cfg.MaxOutputTokens = int32(maxTokens)
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if len(req.StopSequences) > 0 {
		cfg.StopSequences = req.StopSequences
	}
	if req.TopP != nil {
		t := float32(*req.TopP)
		cfg.TopP = &t
	}
	if req.TopK != nil {
		k := float32(*req.TopK)
		cfg.TopK = &k
	}
	if req.Seed != nil {
		// genai carries seed as int32; reject out-of-range values instead of
		// silently truncating to a different deterministic seed.
		if *req.Seed < int64(math.MinInt32) || *req.Seed > int64(math.MaxInt32) {
			return llmkit.Response{}, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
				"Seed out of range for int32", nil)
		}
		s := int32(*req.Seed)
		cfg.Seed = &s
	}
	// Gated on the capability profile: thinkingConfig exists only on the
	// 2.5 family (see googleCapabilities). The documented contract for a
	// false feature is a silent drop rather than a server 400.
	if req.Thinking != nil && g.caps.Thinking {
		if req.Thinking.BudgetTokens <= 0 {
			return llmkit.Response{}, llmkit.NewAPIError("google", 0, 0,
				llmkit.ErrInvalidRequest, "Thinking.BudgetTokens must be positive", nil)
		}
		// IncludeThoughts makes the model return thought-summary parts so
		// reasoning is visible (and round-trippable) in Response.Blocks.
		budget := int32(req.Thinking.BudgetTokens)
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget:  &budget,
			IncludeThoughts: true,
		}
	}

	// Capability gate first: a profile with ToolChoice=false (e.g.
	// gemini-2.0-flash-lite, which launched without function calling) must
	// reject every explicit mode before the wire call — dropping "none"
	// would escalate permissions, dropping "required"/"tool" would silently
	// degrade. Request.ToolChoice then maps onto function_calling_config;
	// auto (and the zero value) is the provider default and is never
	// serialized.
	if err := adapter.GateToolChoice("google", req.ToolChoice, g.caps.ToolChoice); err != nil {
		return llmkit.Response{}, err
	}
	if err := applyGoogleToolChoice(cfg, req.ToolChoice); err != nil {
		return llmkit.Response{}, err
	}

	if len(req.Tools) > 0 {
		decls := make([]*genai.FunctionDeclaration, 0, len(req.Tools))
		for _, t := range req.Tools {
			fd := &genai.FunctionDeclaration{
				Name:        t.Name,
				Description: t.Description,
			}
			if len(t.Parameters) > 0 {
				var schema any
				if err := json.Unmarshal(t.Parameters, &schema); err != nil {
					return llmkit.Response{}, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
						"tool "+t.Name+": invalid parameters JSON schema", err)
				}
				// ParametersJsonSchema accepts a raw JSON-schema object, avoiding a
				// lossy conversion into genai's typed *Schema.
				fd.ParametersJsonSchema = schema
			}
			decls = append(decls, fd)
		}
		cfg.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
	}

	// Schema-constrained output. Gemini cannot combine structured output
	// with function-calling, so we only attach the schema when the caller
	// didn't ask for tools — when they did, we silently fall back to the
	// prompt-embedded schema, which is the same behavior the caller would
	// have gotten before this field existed.
	if len(req.ResponseSchema) > 0 && g.caps.StructuredOutput && len(req.Tools) == 0 {
		// ParseResponseSchema unmarshals the JSON Schema; the returned name
		// is unused on Google (ResponseJsonSchema has no name field) but the
		// helper takes one for symmetry with the other adapters. We pass
		// "response" as the default; callers can supply ResponseSchemaName
		// to override if they want a name surfaced in logs.
		defaultName := req.ResponseSchemaName
		if defaultName == "" {
			defaultName = "response"
		}
		schema, _, err := adapter.ParseResponseSchema(req.ResponseSchema, defaultName)
		if err != nil {
			return llmkit.Response{}, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
				"ResponseSchema: invalid JSON", err)
		}
		// ResponseJsonSchema accepts a raw JSON Schema object, mirroring
		// ParametersJsonSchema — no lossy conversion to genai's typed *Schema.
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = schema
	}

	resp, err := g.client.Models.GenerateContent(ctx, g.model, contents, cfg)
	if err != nil {
		return llmkit.Response{}, g.normalizeErr(err)
	}
	return g.toResponse(resp), nil
}

// applyGoogleToolChoice maps the normalized tool-choice request onto the
// Gemini function_calling_config: required → mode ANY, none → mode NONE,
// and a named tool → mode ANY restricted to that function's name.
func applyGoogleToolChoice(cfg *genai.GenerateContentConfig, tc llmkit.ToolChoice) error {
	fcc := &genai.FunctionCallingConfig{}
	switch tc.Mode {
	case "", llmkit.ToolChoiceAuto:
		return nil
	case llmkit.ToolChoiceNone:
		fcc.Mode = genai.FunctionCallingConfigModeNone
	case llmkit.ToolChoiceRequired:
		fcc.Mode = genai.FunctionCallingConfigModeAny
	case llmkit.ToolChoiceTool:
		if tc.Name == "" {
			return llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
				"ToolChoice.Mode=tool requires ToolChoice.Name", nil)
		}
		fcc.Mode = genai.FunctionCallingConfigModeAny
		fcc.AllowedFunctionNames = []string{tc.Name}
	default:
		return llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
			"unknown ToolChoice.Mode "+string(tc.Mode), nil)
	}
	cfg.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: fcc}
	return nil
}

// toGoogleContents converts normalized messages into genai Contents. Gemini
// uses "user"/"model" roles; tool results are sent as user-turn
// functionResponse parts.
//
// Per-role block rule (ValidateMessageBlocks, before any mapping):
// user text/image/document; assistant text/thinking (Provider-matched only);
// system and tool-result text only. Violations are ErrInvalidRequest.
//
// Content blocks map in order: text → text parts, image/document →
// inline_data (bytes) or file_data (URL) parts. Assistant thinking blocks
// re-emit Text/Thought/ThoughtSignature from Raw (Provider "google" only —
// foreign thinking blocks are dropped silently per the llmkit.Block
// contract). Image/document source validation happens here, BEFORE any wire
// call.
func toGoogleContents(msgs []llmkit.Message) ([]*genai.Content, error) {
	out := make([]*genai.Content, 0, len(msgs))
	for _, m := range msgs {
		// Per-role block-kind rule + media source rule, before any mapping.
		if err := adapter.ValidateMessageBlocks("google", m); err != nil {
			return nil, err
		}
		switch m.Role {
		case llmkit.RoleSystem:
			// Hoisted into SystemInstruction by the caller; if inline, attach as a
			// user turn to preserve content.
			out = append(out, &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: m.Text()}},
			})
		case llmkit.RoleUser:
			parts, err := googleUserParts(m)
			if err != nil {
				return nil, err
			}
			out = append(out, &genai.Content{Role: "user", Parts: parts})
		case llmkit.RoleAssistant:
			parts, err := googleAssistantParts(m)
			if err != nil {
				return nil, err
			}
			out = append(out, &genai.Content{Role: "model", Parts: parts})
		case llmkit.RoleToolResult:
			// Gemini expects the function result wrapped under an "output" key.
			resultObj := map[string]any{"output": m.Text()}
			if m.IsError {
				resultObj = map[string]any{"error": m.Text()}
			}
			out = append(out, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       m.ToolCallID,
						Name:     m.ToolCallID, // genai matches on name; ID is best-effort
						Response: resultObj,
					},
				}},
			})
		default:
			return nil, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
				"unknown message role "+string(m.Role), nil)
		}
	}
	return out, nil
}

// googleUserParts maps a user message's content blocks onto genai parts.
func googleUserParts(m llmkit.Message) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			parts = append(parts, &genai.Part{Text: b.Text})
		case llmkit.BlockImage, llmkit.BlockDocument:
			if err := adapter.ValidateMediaBlock("google", b); err != nil {
				return nil, err
			}
			if len(b.Data) > 0 {
				parts = append(parts, &genai.Part{InlineData: &genai.Blob{
					MIMEType: b.MediaType,
					Data:     b.Data,
				}})
			} else {
				parts = append(parts, &genai.Part{FileData: &genai.FileData{
					FileURI:  b.URL,
					MIMEType: b.MediaType,
				}})
			}
		}
	}
	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}
	return parts, nil
}

// googleAssistantParts maps an assistant message onto genai model parts:
// text in order, then thinking parts rebuilt from Raw (preserving the
// ThoughtSignature Gemini requires on later turns), then one functionCall
// part per ToolCall.
//
// Signature carriers: a BlockThinking whose Raw decodes to a Part with a
// FunctionCall is not emitted as a part — that would duplicate the call.
// It is the carrier toResponse produced for a signed function call (see
// toResponse); its ThoughtSignature is re-attached to the matching
// functionCall part, matched by ID and name. A carrier with no matching
// ToolCall contributes nothing.
func googleAssistantParts(m llmkit.Message) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, 1+len(m.Content)+len(m.ToolCalls))
	sigs := make(map[string][]byte)
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			if b.Text != "" {
				parts = append(parts, &genai.Part{Text: b.Text})
			}
		case llmkit.BlockThinking:
			if b.Provider != "google" {
				continue
			}
			var p genai.Part
			if err := json.Unmarshal(b.Raw, &p); err != nil {
				return nil, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
					"thinking block: malformed Raw JSON", err)
			}
			if p.FunctionCall != nil {
				// Signature carrier, not a thought part.
				if len(p.ThoughtSignature) > 0 {
					sigs[functionCallKey(p.FunctionCall.ID, p.FunctionCall.Name)] = p.ThoughtSignature
				}
				continue
			}
			parts = append(parts, &p)
		}
	}
	for _, tc := range m.ToolCalls {
		var args map[string]any
		if len(tc.Arguments) > 0 {
			if err := json.Unmarshal(tc.Arguments, &args); err != nil {
				return nil, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
					"assistant tool call "+tc.Name+": invalid arguments JSON", err)
			}
		}
		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Name,
				Args: args,
			},
		}
		// Echo the model's thought signature back on the re-sent call; the
		// API rejects unsigned replayed calls on signing models (2.5+).
		if sig, ok := sigs[functionCallKey(tc.ID, tc.Name)]; ok {
			part.ThoughtSignature = sig
		}
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}
	return parts, nil
}

// functionCallKey identifies a function call for signature matching: by ID
// when the server set one, with the declaration name as the discriminator.
func functionCallKey(id, name string) string {
	return id + "\x00" + name
}

func (g *googleAdapter) toResponse(resp *genai.GenerateContentResponse) llmkit.Response {
	var out llmkit.Response
	var text string
	var stop genai.FinishReason

	if len(resp.Candidates) > 0 {
		cand := resp.Candidates[0]
		stop = cand.FinishReason
		if cand.Content != nil {
			for _, p := range cand.Content.Parts {
				if p == nil {
					continue
				}
				switch {
				case p.FunctionCall != nil:
					args, _ := json.Marshal(p.FunctionCall.Args)
					if len(args) == 0 || string(args) == "null" {
						args = json.RawMessage("{}")
					}
					out.ToolCalls = append(out.ToolCalls, llmkit.ToolCall{
						ID:        p.FunctionCall.ID,
						Name:      p.FunctionCall.Name,
						Arguments: json.RawMessage(args),
					})
					if len(p.ThoughtSignature) > 0 {
						// Gemini 2.5+ signs function calls with an opaque
						// thoughtSignature that must be echoed back when the
						// assistant turn is re-sent. llmkit.ToolCall has no
						// field for it, so surface a companion BlockThinking
						// (assistant messages may carry thinking blocks)
						// whose Raw is the re-encodable Part;
						// googleAssistantParts re-attaches the signature to
						// the matching functionCall on the next request.
						raw, _ := json.Marshal(p)
						out.Blocks = append(out.Blocks, llmkit.Block{
							Kind:     llmkit.BlockThinking,
							Provider: "google",
							Raw:      raw,
						})
					}
				case p.Thought:
					// Thought-summary part. Raw re-encodes the parsed part
					// (the SDK exposes no raw wire bytes) preserving text
					// and ThoughtSignature for verbatim re-emission.
					raw, _ := json.Marshal(p)
					out.Blocks = append(out.Blocks, llmkit.Block{
						Kind:     llmkit.BlockThinking,
						Text:     p.Text,
						Provider: "google",
						Raw:      raw,
					})
				default:
					if p.Text != "" {
						text += p.Text
						out.Blocks = append(out.Blocks, llmkit.Block{Kind: llmkit.BlockText, Text: p.Text})
					}
				}
			}
		}
	}
	out.Text = text
	if resp.UsageMetadata != nil {
		// promptTokenCount already INCLUDES cachedContentTokenCount (Gemini
		// convention), matching the normalized Usage semantics directly.
		// Gemini 2.x caches implicitly server-side; we only surface the hit
		// count — no explicit CachedContent management in this thin layer.
		out.Usage = llmkit.Usage{
			InputTokens:          int64(resp.UsageMetadata.PromptTokenCount),
			OutputTokens:         int64(resp.UsageMetadata.CandidatesTokenCount),
			CacheReadInputTokens: int64(resp.UsageMetadata.CachedContentTokenCount),
		}
	}
	out.StopReason = mapGoogleStop(stop, len(out.ToolCalls) > 0)
	return out
}

func mapGoogleStop(reason genai.FinishReason, hasToolCalls bool) llmkit.StopReason {
	switch reason {
	case genai.FinishReasonStop, genai.FinishReasonUnspecified, "":
		if hasToolCalls {
			return llmkit.StopToolUse
		}
		return llmkit.StopEndTurn
	case genai.FinishReasonMaxTokens:
		return llmkit.StopMaxTokens
	case genai.FinishReasonSafety, genai.FinishReasonRecitation,
		genai.FinishReasonProhibitedContent, genai.FinishReasonBlocklist,
		genai.FinishReasonSPII:
		return llmkit.StopContentFilter
	default:
		if hasToolCalls {
			return llmkit.StopToolUse
		}
		return llmkit.StopError
	}
}

func (g *googleAdapter) normalizeErr(err error) error {
	// genai returns APIError by value (not a pointer). It carries no
	// *http.Response, so Retry-After is unavailable; the retry wrapper
	// falls back to exponential backoff. We pass nil for resp, which the
	// shared helper recognizes and skips Retry-After parsing for.
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return adapter.NormalizeSDKError("google", apiErr.Code, apiErr.Message, nil, err)
	}
	return llmkit.NewAPIError("google", 0, 0, llmkit.ErrServer, err.Error(), err)
}

// Sources (vendor docs consulted for this table; every number comes from
// the model's own spec page):
//   - https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite and
//     the sibling 2.5 text-model pages — input token limit 1,048,576 for 2.5
//     pro / flash / flash-lite; function calling, structured outputs,
//     caching, and thinking all supported (verified on the flash-lite page).
//   - https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image —
//     the image-generation variant: input token limit 65,536; caching,
//     function calling, structured outputs, and thinking all NOT supported.
//     Explicitly listed so the text "gemini-2.5-flash" key cannot swallow it.
//   - https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-preview-tts —
//     input token limit 8,192; caching, function calling, structured
//     outputs, and thinking all NOT supported (text in, audio out). Listed
//     so the text entry cannot swallow it.
//   - https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-native-audio-preview-12-2025 —
//     input token limit 131,072; function calling and thinking SUPPORTED;
//     structured outputs and caching NOT supported. An audio-dialog
//     generation (Live API-capable) — listed so the text entry cannot
//     swallow it.
//   - Gemini 1.5 / 2.0 model pages (now retired from the docs): 1.5-pro
//     2,097,152 (the 002 refresh; 1M before), 1.5-flash and both 2.0 flash
//     tiers 1,048,576. The 2.0/1.5 generations have no thinkingConfig —
//     Thinking is false there. gemini-2.0-flash-lite launched without
//     function calling, so its ParallelToolCalls and ToolChoice are false.
//
// The table is matched by LONGEST key with a "-" segment boundary
// (adapter.MatchesModelFamily), so "gemini-2.5-flash-lite" and the
// non-text variant keys win over "gemini-2.5-flash", while a mid-token
// extension like "gemini-2.5-flashy" matches nothing. Image/document
// parts, stop sequences, top_p, top_k, and seed hold for every entry and
// stay in the shared defaults; window, thinking, tools, structured output,
// and caching are stored per entry because the 2.5 flash non-text variants
// genuinely differ.
//
// An unknown model reports ContextWindow 0 — unknown is never fabricated
// into a number — with the API-level feature defaults; the server is the
// final validator for unrecognized names. Pin exact values for unknown
// models via Options.Capabilities.
type googleModelCaps struct {
	prefix     string
	window     int
	thinking   bool
	tools      bool // function calling supported at all
	structured bool // responseMIMEType/JSON-schema structured output
	caching    bool // context/prompt caching accepted
}

var googleModelTable = []googleModelCaps{
	// Text generation models.
	{"gemini-2.5-pro", 1_048_576, true, true, true, true},
	{"gemini-2.5-flash", 1_048_576, true, true, true, true},
	{"gemini-2.5-flash-lite", 1_048_576, true, true, true, true},
	{"gemini-2.0-flash", 1_048_576, false, true, true, true},
	{"gemini-2.0-flash-lite", 1_048_576, false, false, true, true},
	{"gemini-1.5-pro", 2_097_152, false, true, true, true},
	{"gemini-1.5-flash", 1_048_576, false, true, true, true},
	// Non-text variants of the 2.5 flash generation — pinned so the text
	// entry above cannot swallow them (its Thinking=true is load-bearing:
	// it gates thinkingConfig on the wire). Facts are from each variant's
	// own spec page, cited above.
	{"gemini-2.5-flash-image", 65_536, false, false, false, false},
	{"gemini-2.5-flash-preview-tts", 8_192, false, false, false, false},
	{"gemini-2.5-flash-native-audio", 131_072, true, true, false, false},
}

func googleCapabilities(model string) llmkit.Capabilities {
	caps := llmkit.Capabilities{
		ContextWindow:     0, // unknown until the table matches
		ParallelToolCalls: true,
		PromptCaching:     true,
		StructuredOutput:  true,
		Thinking:          true,
		ToolChoice:        true,
		Images:            true,
		Documents:         true,
		StopSequences:     true,
		TopP:              true,
		TopK:              true,
		Seed:              true,
	}
	best := adapter.BestMatchingFamily(model, googleModelTable, func(e googleModelCaps) string { return e.prefix })
	if best >= 0 {
		e := googleModelTable[best]
		caps.ContextWindow = e.window
		caps.Thinking = e.thinking
		caps.ParallelToolCalls = e.tools
		caps.ToolChoice = e.tools
		caps.StructuredOutput = e.structured
		caps.PromptCaching = e.caching
	}
	return caps
}

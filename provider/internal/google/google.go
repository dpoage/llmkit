// Package google adapts the normalized llmkit types onto the
// google.golang.org/genai SDK (Gemini API backend).
package google

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
	"google.golang.org/genai"
)

type googleAdapter struct {
	client *genai.Client
	model  string
	caps   llmkit.Capabilities
}

// Options configures a Gemini adapter.
//
// Capabilities, when non-nil, receives the adapter's model-table profile
// and returns the effective one; it is applied once at construction. Return
// a fixed profile to pin exact values for models the table doesn't know.
// nil = model-table default. Every wire-gated field is within Google's
// ceiling — see New.
type Options struct {
	APIKey     string
	BaseURL    string       // optional; for testing or non-default endpoints
	HTTPClient *http.Client // optional; for testing (httptest)

	Capabilities func(llmkit.Capabilities) llmkit.Capabilities
}

// googleCeiling names every wire-gated field this adapter can put on the
// wire: the Gemini API has a wire field for all seven, so no override is
// ever refused at construction — the one documented case where every field
// is within ceiling.
var googleCeiling = llmkit.Capabilities{
	StructuredOutput: true,
	Thinking:         true,
	ToolChoice:       true,
	StopSequences:    true,
	TopP:             true,
	TopK:             true,
	Seed:             true,
}

// New builds a Gemini-backed Client. genai's only built-in retry path is for
// file uploads, so the shared retry wrapper is the sole retry layer for
// completions. The installed HTTP client wraps the caller's transport with
// a status recorder so error classification can fall back to the transport
// status when genai drops it (see normalizeErr).
//
// HTTPOptions.BaseURL is always set explicitly — opts.BaseURL, or the
// vendor default when empty — because genai's getBaseURL checks an explicit
// HTTPOptions.BaseURL first, before a process-global SetDefaultBaseURLs
// tier and GOOGLE_GEMINI_BASE_URL; only setting it unconditionally keeps
// both of those from ever being consulted. Backend is always the explicit
// BackendGeminiAPI (never BackendUnspecified), so GOOGLE_GENAI_USE_VERTEXAI
// has no effect; APIKey is always the non-empty Spec.Secret, so genai
// never uses GOOGLE_API_KEY or GEMINI_API_KEY. genai.NewClient still reads
// both on every call, and logs a warning when both are set.
func New(ctx context.Context, model string, opts Options) (llmkit.Client, error) {
	caps, err := adapter.ApplyOverride("google", googleCeiling, googleCapabilities(model), opts.Capabilities)
	if err != nil {
		return nil, err
	}

	cc := &genai.ClientConfig{
		APIKey:  opts.APIKey,
		Backend: genai.BackendGeminiAPI,
	}
	cc.HTTPOptions.BaseURL = opts.BaseURL
	if cc.HTTPOptions.BaseURL == "" {
		cc.HTTPOptions.BaseURL = "https://generativelanguage.googleapis.com/"
	}
	// Always install the recording client (see recordStatusTransport):
	// genai substitutes its own default client when HTTPClient is nil, so
	// without this the transport status would be invisible to normalizeErr.
	base := opts.HTTPClient
	if base == nil {
		base = &http.Client{}
	}
	wrapped := *base
	if wrapped.Transport == nil {
		wrapped.Transport = http.DefaultTransport
	}
	wrapped.Transport = &recordStatusTransport{rt: wrapped.Transport}
	cc.HTTPClient = &wrapped
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, adapter.Refuse("google", "failed to construct genai client: "+err.Error(), err)
	}
	return &googleAdapter{
		client: client,
		model:  model,
		caps:   caps,
	}, nil
}

// transportStatusKey keys the per-request status recorder in the request
// context.
type transportStatusKey struct{}

// transportStatus records the HTTP status of the most recent response for
// one Complete call. genai drops the transport *http.Response when an error
// body parses as a Google error object without a "code" field (its only
// error type, genai.APIError, has no Unwrap), so the status captured at the
// transport is the only available fallback in that branch.
type transportStatus struct {
	code int
}

// recordStatusTransport wraps the SDK's transport, recording each response's
// status code into the per-request recorder carried on the request context.
// Last write wins: redirect hops overwrite earlier ones, leaving the final
// status the SDK itself saw. The recorder lives in the context, so
// concurrent Complete calls on a shared adapter never contend.
type recordStatusTransport struct {
	rt http.RoundTripper
}

func (t *recordStatusTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.rt.RoundTrip(req)
	if resp != nil {
		if s, ok := req.Context().Value(transportStatusKey{}).(*transportStatus); ok && s != nil {
			s.code = resp.StatusCode
		}
	}
	return resp, err
}

func (g *googleAdapter) Capabilities() llmkit.Capabilities { return g.caps }

func (g *googleAdapter) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	// Per-request transport-status recorder for normalizeErr's fallback.
	ctx = context.WithValue(ctx, transportStatusKey{}, &transportStatus{})
	contents, cfg, err := g.buildRequest(req)
	if err != nil {
		return llmkit.Response{}, err
	}
	resp, err := g.client.Models.GenerateContent(ctx, g.model, contents, cfg)
	if err != nil {
		return llmkit.Response{}, g.normalizeErr(ctx, err)
	}
	return g.toResponse(resp), nil
}

// buildRequest applies the shared rules exactly once via adapter.Prepare,
// then maps the Prepared value onto the genai wire types (Contents and
// GenerateContentConfig) without re-checking anything Prepare already
// validated or gated. Complete and Stream both send this builder's result
// to the wire.
func (g *googleAdapter) buildRequest(req llmkit.Request) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	p, err := adapter.Prepare("google", g.caps, req)
	if err != nil {
		return nil, nil, err
	}

	contents, err := toGoogleContents(p.Messages)
	if err != nil {
		return nil, nil, err
	}
	cfg := &genai.GenerateContentConfig{}
	if p.System != "" {
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: p.System}},
		}
	}
	cfg.MaxOutputTokens = int32(p.MaxTokens)
	if p.Temperature != nil {
		t := float32(*p.Temperature)
		cfg.Temperature = &t
	}
	if len(p.StopSequences) > 0 {
		cfg.StopSequences = p.StopSequences
	}
	if p.TopP != nil {
		t := float32(*p.TopP)
		cfg.TopP = &t
	}
	if p.TopK != nil {
		k := float32(*p.TopK)
		cfg.TopK = &k
	}
	if p.Seed != nil {
		// genai carries seed as int32; reject out-of-range values instead of
		// silently truncating to a different deterministic seed. Vendor-
		// specific: no llmkit.Capabilities field describes an int range.
		if *p.Seed < int64(math.MinInt32) || *p.Seed > int64(math.MaxInt32) {
			return nil, nil, adapter.Refuse("google", "Seed out of range for int32", nil)
		}
		s := int32(*p.Seed)
		cfg.Seed = &s
	}
	// p.Thinking is nil unless g.caps.Thinking is true AND the caller
	// supplied one; BudgetTokens is already validated >= 1. IncludeThoughts
	// makes the model return thought-summary parts so reasoning is visible
	// (and round-trippable) in Response.Blocks.
	if p.Thinking != nil {
		budget := int32(p.Thinking.BudgetTokens)
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget:  &budget,
			IncludeThoughts: true,
		}
	}

	// Request.ToolChoice: already validated and gated by Prepare. auto (and
	// the zero value) is the provider default and is never serialized.
	applyGoogleToolChoice(cfg, p.ToolChoice)

	if len(p.Tools) > 0 {
		decls := make([]*genai.FunctionDeclaration, 0, len(p.Tools))
		for _, t := range p.Tools {
			fd := &genai.FunctionDeclaration{
				Name:        t.Def.Name,
				Description: t.Def.Description,
			}
			if t.Schema != nil {
				// ParametersJsonSchema accepts a raw JSON-schema object,
				// avoiding a lossy conversion into genai's typed *Schema.
				fd.ParametersJsonSchema = t.Schema
			}
			decls = append(decls, fd)
		}
		cfg.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
	}

	// Schema-constrained output. Gemini cannot combine structured output
	// with function-calling, so we only attach the schema when the caller
	// didn't ask for tools — when they did, we silently fall back to the
	// prompt-embedded schema, which is the same behavior the caller would
	// have gotten before this field existed. p.ResponseSchema is already
	// nil when the capability is off or the caller supplied none.
	if p.ResponseSchema != nil && len(p.Tools) == 0 {
		// ResponseJsonSchema accepts a raw JSON Schema object, mirroring
		// ParametersJsonSchema — no lossy conversion to genai's typed
		// *Schema. The name is unused on Google (ResponseJsonSchema has no
		// name field).
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = p.ResponseSchema.Value
	}

	return contents, cfg, nil
}

// applyGoogleToolChoice maps an already-validated ToolChoice onto the
// Gemini function_calling_config: required → mode ANY, none → mode NONE,
// and a named tool → mode ANY restricted to that function's name. Prepare
// has already applied the capability gate and validated Mode and Name, so
// this is a pure mapping — auto is the provider default and is never
// serialized.
func applyGoogleToolChoice(cfg *genai.GenerateContentConfig, tc llmkit.ToolChoice) {
	if tc.Mode == llmkit.ToolChoiceAuto {
		return
	}
	fcc := &genai.FunctionCallingConfig{}
	switch tc.Mode {
	case llmkit.ToolChoiceNone:
		fcc.Mode = genai.FunctionCallingConfigModeNone
	case llmkit.ToolChoiceRequired:
		fcc.Mode = genai.FunctionCallingConfigModeAny
	case llmkit.ToolChoiceTool:
		fcc.Mode = genai.FunctionCallingConfigModeAny
		fcc.AllowedFunctionNames = []string{tc.Name}
	}
	cfg.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: fcc}
}

// toGoogleContents converts normalized messages into genai Contents. Gemini
// uses "user"/"model" roles; tool results are sent as user-turn
// functionResponse parts. Every role and block-kind rule (adapter.Prepare)
// already holds — this is a pure mapping.
//
// A tool result's FunctionResponse.name must be the DECLARED tool name
// (Gemini correlates responses to declarations by name; the id is
// best-effort), so the conversion records each assistant turn's tool-call
// ids and names as it goes and resolves the name from the map. An id no
// assistant turn declares (e.g. a truncated history) falls back to the id
// itself.
//
// Content blocks map in order: text → text parts, image/document →
// inline_data (bytes) or file_data (URL) parts. Assistant thinking blocks
// re-emit Text/Thought/ThoughtSignature from Raw (Provider "google" only —
// foreign thinking blocks are dropped silently per the llmkit.Block
// contract).
func toGoogleContents(msgs []llmkit.Message) ([]*genai.Content, error) {
	out := make([]*genai.Content, 0, len(msgs))
	// toolNames maps a tool-call id to its declared function name, filled in
	// from each assistant turn as it is converted.
	toolNames := make(map[string]string)
	for _, m := range msgs {
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
			for _, tc := range m.ToolCalls {
				if tc.ID != "" {
					toolNames[tc.ID] = tc.Name
				}
			}
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
			// FunctionResponse.name must be the declared tool name; fall
			// back to the id when no assistant call in this history carries
			// it (better an unknown-but-present name than an empty one).
			name, ok := toolNames[m.ToolCallID]
			if !ok {
				name = m.ToolCallID
			}
			out = append(out, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       m.ToolCallID,
						Name:     name,
						Response: resultObj,
					},
				}},
			})
		}
	}
	return out, nil
}

// googleUserParts maps a user message's content blocks onto genai parts.
// The media source rule (adapter.Prepare) already holds.
func googleUserParts(m llmkit.Message) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			parts = append(parts, &genai.Part{Text: b.Text})
		case llmkit.BlockImage, llmkit.BlockDocument:
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
			// A thinking block whose Raw was lost (nil) or decodes to
			// nothing ("null", "{}", with or without surrounding JSON
			// whitespace) contributes no wire content: skip it instead of
			// forwarding an empty part. Trim only the bytes JSON permits
			// as space — a Raw padded with anything else (e.g. U+00A0) is
			// not parseable and falls through to the malformed error
			// below. Unlike anthropic — whose signed thinking replay is
			// mandatory, making a missing Raw a hard error — a
			// decoded-to-empty genai Part carries nothing Gemini requires
			// on later turns.
			trimmed := bytes.Trim(b.Raw, " \t\n\r")
			if len(trimmed) == 0 || string(trimmed) == "null" || string(trimmed) == "{}" {
				continue
			}
			var p genai.Part
			if err := json.Unmarshal(b.Raw, &p); err != nil {
				return nil, adapter.Refuse("google", "thinking block: malformed Raw JSON", err)
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
				return nil, adapter.Refuse("google",
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
					out.ToolCalls = append(out.ToolCalls, llmkit.ToolCall{
						ID:        p.FunctionCall.ID,
						Name:      p.FunctionCall.Name,
						Arguments: functionCallArgs(p.FunctionCall.Args),
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

// functionCallArgs renders a FunctionCall's Args as the canonical JSON the
// normalized ToolCall carries: the marshaled object, or "{}" when the call
// has none (the SDK decodes an absent object to nil). Stream reuses it for
// tool-call deltas, so one call's delta Arguments concatenate to exactly
// its final ToolCall.Arguments.
func functionCallArgs(args any) json.RawMessage {
	b, _ := json.Marshal(args)
	if len(b) == 0 || string(b) == "null" {
		return json.RawMessage("{}")
	}
	return b
}

// mapGoogleStop is this adapter's recognized finishReason table: STOP,
// FINISH_REASON_UNSPECIFIED, and "" (no candidates) all map to a
// tool-call-aware end-turn — ToolUse when the response carries tool calls,
// EndTurn otherwise; MAX_TOKENS maps to MaxTokens; SAFETY, RECITATION,
// PROHIBITED_CONTENT, BLOCKLIST, and SPII all map to ContentFilter. Every
// other reason — including "OTHER" — falls to adapter.StopFallback:
// ToolUse when tool calls are present, StopError otherwise.
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
		return adapter.StopFallback(hasToolCalls)
	}
}

func (g *googleAdapter) normalizeErr(ctx context.Context, err error) error {
	// genai returns APIError by value (not a pointer) and exposes neither
	// response headers nor a vendor type: no Retry-After is parsed, and a
	// sub-400 status without a vendor type falls back to ErrServer.
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		status := apiErr.Code
		if status == 0 {
			// The error body parsed as a Google error object without a code
			// field, so the SDK discarded the transport response. Classify
			// from the status the wrapping transport recorded for this
			// request instead of mislabeling every such failure a 400.
			if s, ok := ctx.Value(transportStatusKey{}).(*transportStatus); ok && s != nil {
				status = s.code
			}
		}
		return adapter.NormalizeSDKError("google", adapter.VendorError{
			Status:  status,
			Message: apiErr.Message,
			Err:     err,
		})
	}
	// No HTTP response: transport failure or caller's context ending mid-call.
	return adapter.TransportError("google", ctx, err)
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
// (adapter.BestMatchingFamily), so "gemini-2.5-flash-lite" and the
// non-text variant keys win over "gemini-2.5-flash", while a mid-token
// extension like "gemini-2.5-flashy" matches nothing. Per-entry fields are
// only those that genuinely vary by model — window, thinking, tools,
// structured output, caching — because the 2.5 flash non-text variants
// differ; image/document parts, stop sequences, top_p, top_k, and seed hold
// for every entry and stay in the shared defaults.
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

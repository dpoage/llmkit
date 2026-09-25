// Package anthropic adapts the normalized llmkit types onto the official
// Anthropic SDK (API-key or OAuth bearer-token authentication).
package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
)

type anthropicAdapter struct {
	client anthropic.Client
	model  string
	caps   llmkit.Capabilities
}

// Options configures an Anthropic adapter.
//
// Exactly one of APIKey and AuthToken must be non-empty — they are mutually
// exclusive. APIKey drives x-api-key authentication (api_key mode). AuthToken
// drives OAuth bearer-token authentication (oauth-token mode), which requires
// the anthropic-beta: oauth-2025-04-20 header and must NOT also set x-api-key
// (the Anthropic API rejects requests that carry both credentials).
//
// Capabilities, when non-nil, receives the adapter's model-table profile
// and returns the effective one; it is applied once at construction. Return
// a fixed profile to pin exact values for models the table doesn't know.
// nil = model-table default. An effective profile that reports Seed=true
// is refused — see New.
type Options struct {
	APIKey     string
	AuthToken  string       // OAuth bearer token; mutually exclusive with APIKey
	BaseURL    string       // optional; for testing or proxies
	HTTPClient *http.Client // optional; for testing (httptest)

	Capabilities func(llmkit.Capabilities) llmkit.Capabilities
}

// anthropicCeiling names every wire-gated field this adapter can put on the
// wire: the Messages API has no seed parameter, so an effective profile
// reporting Seed=true is refused at construction rather than silently
// dropped.
var anthropicCeiling = llmkit.Capabilities{
	StructuredOutput: true,
	Thinking:         true,
	ToolChoice:       true,
	StopSequences:    true,
	TopP:             true,
	TopK:             true,
	Seed:             false,
}

// New builds an Anthropic-backed Client. The vendor SDK's built-in retries are
// disabled (WithMaxRetries(0)) so the shared retry wrapper is the single
// source of retry policy. New returns an error wrapping
// llmkit.ErrInvalidRequest, with no network I/O, when the effective
// capability profile (table or Options.Capabilities override) reports
// Seed=true — see anthropicCeiling.
//
// When opts.AuthToken is non-empty the adapter uses OAuth bearer-token
// authentication: the SDK sets Authorization: Bearer <token> via
// option.WithAuthToken, and option.WithHeaderAdd appends the required
// "oauth-2025-04-20" value to the anthropic-beta header without clobbering any
// other beta values the SDK may already have set.
//
// option.WithoutEnvironmentDefaults makes anthropic.NewClient skip
// DefaultClientOptions entirely — no ANTHROPIC_API_KEY/AUTH_TOKEN/PROFILE/
// CONFIG_DIR/CUSTOM_HEADERS, env federation, or the $HOME/.config/anthropic
// dotfile profile is read — and keep only the hardcoded production base-URL
// default, so an explicit option.WithBaseURL below (when opts.BaseURL is set)
// is the sole source of a non-default host. Because no env credential can
// reach the request anymore, an env-derived X-Api-Key or Authorization header
// cannot appear; the SDK sets neither on its own. A nil opts.HTTPClient gets
// an explicit &http.Client{} rather than the process-global client the SDK
// falls back to when no explicit HTTP client option is applied
// (WithoutEnvironmentDefaults keeps none of DefaultClientOptions's own
// client default).
func New(model string, opts Options) (llmkit.Client, error) {
	caps, err := adapter.ApplyOverride("anthropic", anthropicCeiling, anthropicCapabilities(model), opts.Capabilities)
	if err != nil {
		return nil, err
	}

	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	reqOpts := []option.RequestOption{
		option.WithoutEnvironmentDefaults(),
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(0),
	}
	if opts.AuthToken != "" {
		// OAuth mode: bearer token in Authorization header; x-api-key must be absent.
		reqOpts = append(reqOpts, option.WithAuthToken(opts.AuthToken))
		// Append rather than replace so other anthropic-beta values set by the SDK
		// are preserved alongside the oauth beta flag.
		reqOpts = append(reqOpts, option.WithHeaderAdd("anthropic-beta", "oauth-2025-04-20"))
	} else {
		reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	return &anthropicAdapter{
		client: anthropic.NewClient(reqOpts...),
		model:  model,
		caps:   caps,
	}, nil
}

func (a *anthropicAdapter) Capabilities() llmkit.Capabilities { return a.caps }

func (a *anthropicAdapter) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	params, prepared, err := a.buildParams(req)
	if err != nil {
		return llmkit.Response{}, err
	}

	msg, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return llmkit.Response{}, a.normalizeErr(ctx, err)
	}
	return a.finalize(prepared, a.toResponse(msg)), nil
}

// finalize coerces the synthetic structured-output tool's lone call into
// Response.Text and StopEndTurn: the call's Arguments become Text
// (downstream layers have no handler for the synthetic tool), and
// Anthropic's "tool_use" stop reason for the forced call would otherwise
// mis-classify the completion. The surfaced text is also appended as a
// BlockText: under forced tool_choice the wire cannot also carry a visible
// text block, so this appends the ONE BlockText the Response's invariant
// (Text equals the concatenation of BlockText blocks) needs — without it,
// verbatim-block consumers such as the agent's assistant history would
// drop the text. Complete and Stream both go through this step over the
// same Prepared value, so the same wire exchange returns identical Responses.
func (a *anthropicAdapter) finalize(p adapter.Prepared, resp llmkit.Response) llmkit.Response {
	if toolName, ok := structuredOutputToolName(p); ok &&
		len(resp.ToolCalls) == 1 && resp.ToolCalls[0].Name == toolName {
		resp.Text = string(resp.ToolCalls[0].Arguments)
		resp.ToolCalls = nil
		resp.StopReason = llmkit.StopEndTurn
		if resp.Text != "" {
			resp.Blocks = append(resp.Blocks, llmkit.Block{Kind: llmkit.BlockText, Text: resp.Text})
		}
	}
	return resp
}

// buildParams applies the shared rules exactly once via adapter.Prepare,
// then maps the Prepared value onto Anthropic Messages params without
// re-checking anything Prepare already validated or gated. It returns the
// Prepared value alongside params so Complete and Stream can reuse it for
// the synthetic-tool bookkeeping (structuredOutputToolName, finalize).
func (a *anthropicAdapter) buildParams(req llmkit.Request) (anthropic.MessageNewParams, adapter.Prepared, error) {
	p, err := adapter.Prepare("anthropic", a.caps, req)
	if err != nil {
		return anthropic.MessageNewParams{}, adapter.Prepared{}, err
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(a.model),
		MaxTokens: int64(p.MaxTokens),
	}
	if p.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: p.System}}
	}
	if p.Temperature != nil {
		params.Temperature = anthropic.Float(*p.Temperature)
	}

	// p.Thinking is nil unless a.caps.Thinking is true AND the caller
	// supplied one; BudgetTokens is already validated >= 1. thinkingSent
	// is reused by the forced-tool-use refusal below.
	thinkingSent := p.Thinking != nil
	if thinkingSent {
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: int64(p.Thinking.BudgetTokens),
			},
		}
	}
	if len(p.StopSequences) > 0 {
		params.StopSequences = p.StopSequences
	}
	if p.TopP != nil {
		params.TopP = anthropic.Float(*p.TopP)
	}
	if p.TopK != nil {
		params.TopK = anthropic.Int(int64(*p.TopK))
	}
	// Request.Seed: the Anthropic Messages API has no seed parameter, so it
	// is always nil here (Capabilities.Seed = false on every profile).
	msgs, err := toAnthropicMessages(p.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, adapter.Prepared{}, err
	}
	params.Messages = msgs

	if len(p.Tools) > 0 {
		tools := make([]anthropic.ToolUnionParam, 0, len(p.Tools))
		for _, t := range p.Tools {
			tp, err := toAnthropicTool(t)
			if err != nil {
				return anthropic.MessageNewParams{}, adapter.Prepared{}, err
			}
			tools = append(tools, anthropic.ToolUnionParam{OfTool: tp})
		}
		params.Tools = tools
	}

	// Request-level tool choice: already validated and gated by Prepare.
	// The synthetic forcing below then overwrites whatever was mapped: a
	// structured-output request is meaningless without the forced tool
	// call, so it keeps precedence over p.ToolChoice.
	applyAnthropicToolChoice(&params, p.ToolChoice)

	// Schema-constrained output. Anthropic has no native response_format, so
	// we inject a single synthetic tool and force tool_choice to it — the
	// model returns a tool_use block whose `input` is the schema-conformant
	// JSON, which Complete surfaces as Response.Text. Only valid when the
	// caller didn't supply user tools (structuredOutputToolName gates this).
	synthName, syntheticOutput := structuredOutputToolName(p)
	if syntheticOutput {
		inputSchema, err := toAnthropicInputSchema("ResponseSchema", p.ResponseSchema.Value)
		if err != nil {
			return anthropic.MessageNewParams{}, adapter.Prepared{}, err
		}
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name:        synthName,
			InputSchema: inputSchema,
			Description: anthropic.String("Emit the final answer that conforms to the response schema."),
		}})
		params.ToolChoice = anthropic.ToolChoiceParamOfTool(synthName)
	}

	// Forced tool use cannot coexist with manual extended thinking: "tool
	// use with manual extended thinking ... only supports tool_choice auto
	// or none ... Using tool_choice: {"type": "any"} or tool_choice:
	// {"type": "tool", "name": "..."} results in an error"
	// (https://platform.claude.com/docs/en/build-with-claude/thinking,
	// "Thinking with tool use"). The scope is manual mode — adaptive
	// thinking supports forced tool use — and llmkit only ever emits
	// thinking {type: "enabled"} (the gate above), so the refusal is exact
	// for llmkit's surface today. The forcing may come from the caller
	// (ToolChoice required or a named tool) or from the synthetic
	// structured-output tool above, which overwrites whatever the caller
	// mapped, so the built params here are the one place both sources are
	// visible and one predicate covers both. A guaranteed remote 400 is
	// refused locally, naming the field that did the forcing.
	if thinkingSent &&
		(params.ToolChoice.OfAny != nil || params.ToolChoice.OfTool != nil) {
		forcing := "ToolChoice.Mode=required forces tool_choice any"
		if params.ToolChoice.OfTool != nil {
			forcing = "ToolChoice.Mode=tool forces tool_choice to the " + params.ToolChoice.OfTool.Name + " tool"
		}
		if syntheticOutput {
			forcing = "ResponseSchema forces tool_choice to the " + synthName + " tool"
		}
		return anthropic.MessageNewParams{}, adapter.Prepared{}, adapter.Refuse("anthropic",
			"Thinking cannot be combined with forced tool use: manual extended thinking only supports tool_choice auto or none, and "+forcing, nil)
	}

	applyCacheBreakpoints(&params)
	return params, p, nil
}

// applyAnthropicToolChoice maps an already-validated ToolChoice onto the
// Anthropic tool_choice parameter. Prepare has already applied the
// ToolChoice gate and validated Mode and Name, so this is a pure mapping —
// auto is the provider default and is never serialized.
func applyAnthropicToolChoice(params *anthropic.MessageNewParams, tc llmkit.ToolChoice) {
	switch tc.Mode {
	case llmkit.ToolChoiceNone:
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	case llmkit.ToolChoiceRequired:
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	case llmkit.ToolChoiceTool:
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{Name: tc.Name},
		}
	}
}

// applyCacheBreakpoints marks ephemeral prompt-cache breakpoints on the
// request. Anthropic caches the prefix up to each breakpoint (max 4 per
// request), so the agent loop's stable prefix — tool definitions, then the
// system prompt, then the append-only conversation — is re-read from cache
// instead of re-billed at full price each iteration.
//
// Placement (≤ 4 total):
//
//   - the last tool definition (tools serialize before system/messages, so one
//     breakpoint there caches the whole tool block);
//   - the last system block (caches tools + system);
//   - the last content block of the final message AND of the message before it.
//     The final-message breakpoint writes this iteration's full conversation to
//     cache; the second-to-last one lands exactly where the previous iteration's
//     final breakpoint was, anchoring an exact cache hit even when a turn
//     appends more blocks than the server's automatic ~20-block lookback covers.
//
// Moving breakpoints between requests does not invalidate cache entries —
// lookup is by content prefix, not by marker position.
func applyCacheBreakpoints(params *anthropic.MessageNewParams) {
	if n := len(params.Tools); n > 0 {
		if tp := params.Tools[n-1].OfTool; tp != nil {
			tp.CacheControl = anthropic.NewCacheControlEphemeralParam()
		}
	}
	if n := len(params.System); n > 0 {
		params.System[n-1].CacheControl = anthropic.NewCacheControlEphemeralParam()
	}
	for i, marked := len(params.Messages)-1, 0; i >= 0 && marked < 2; i-- {
		if markLastBlock(&params.Messages[i]) {
			marked++
		}
	}
}

// markLastBlock sets an ephemeral cache_control on the last content block of m,
// reporting whether a markable block was found. Thinking (and redacted
// thinking) blocks have no CacheControl field, so GetCacheControl returns nil
// and they are skipped — the marker moves to the nearest markable block.
func markLastBlock(m *anthropic.MessageParam) bool {
	for i := len(m.Content) - 1; i >= 0; i-- {
		if cc := m.Content[i].GetCacheControl(); cc != nil {
			*cc = anthropic.NewCacheControlEphemeralParam()
			return true
		}
	}
	return false
}

// toAnthropicTool maps a prepared tool onto the Anthropic SDK's ToolParam.
// t.Schema is already decoded and validated by adapter.Prepare; the only
// refusal left is toAnthropicInputSchema's.
func toAnthropicTool(t adapter.Tool) (*anthropic.ToolParam, error) {
	inputSchema, err := toAnthropicInputSchema("tool "+t.Def.Name+": Parameters", t.Schema)
	if err != nil {
		return nil, err
	}
	tp := &anthropic.ToolParam{
		Name:        t.Def.Name,
		InputSchema: inputSchema,
	}
	if t.Def.Description != "" {
		tp.Description = anthropic.String(t.Def.Description)
	}
	return tp, nil
}

// toAnthropicInputSchema converts a decoded JSON-Schema root object into the
// Anthropic SDK's ToolInputSchemaParam: "properties" and "required" lift
// into the typed fields, Type is always forced to the literal "object" (a
// caller's root "type" is discarded, NEVER copied into ExtraFields — the
// SDK's ExtraFields marshal alongside and OVERRIDE a same-named typed
// field, so copying "type" would let a caller's "array" silently replace
// the constant on the wire), and every other root key — $defs,
// additionalProperties, description, and anything else except "" (below) —
// rides in ExtraFields so it reaches the wire verbatim.
//
// Type is set explicitly, not left at its Go zero value, because
// ToolParam.InputSchema carries its own `omitzero` tag: a
// ToolInputSchemaParam whose Properties, Required, Type, and ExtraFields
// are ALL at their zero value reflects as zero itself, and omitzero would
// then drop the whole `input_schema` key — even though the SDK's `default`
// tag on Type documents it as always "object" once the key IS present. An
// explicit Type is what keeps the struct non-zero so a nil schema (no
// Parameters/ResponseSchema supplied) still sends {"type":"object"}:
// input_schema is api:"required" and must always be present.
//
// A root key the SDK cannot marshal is refused before the wire, naming
// label: the empty-string key "" fails the SDK's ExtraFields encoding
// ("path cannot be empty"), which would otherwise surface from the HTTP
// call as a retried ErrServer. Absent or empty "properties"/"required"
// entries are dropped — the typed fields are omitzero.
func toAnthropicInputSchema(label string, schema map[string]any) (anthropic.ToolInputSchemaParam, error) {
	out := anthropic.ToolInputSchemaParam{Type: "object"}
	if schema == nil {
		return out, nil
	}
	if _, ok := schema[""]; ok {
		return anthropic.ToolInputSchemaParam{}, adapter.Refuse("anthropic",
			label+`: the empty-string root key "" cannot be sent: the Anthropic SDK cannot encode it`, nil)
	}
	if p, ok := schema["properties"]; ok {
		out.Properties = p
	}
	if r, ok := schema["required"]; ok {
		if rs, ok := r.([]any); ok {
			for _, v := range rs {
				if s, ok := v.(string); ok {
					out.Required = append(out.Required, s)
				}
			}
		}
	}
	var extra map[string]any
	for k, v := range schema {
		switch k {
		case "type", "properties", "required":
			continue
		default:
			if extra == nil {
				extra = make(map[string]any, len(schema))
			}
			extra[k] = v
		}
	}
	out.ExtraFields = extra
	return out, nil
}

// structuredOutputToolName returns the synthetic tool name this adapter
// injects to coerce schema-constrained output from Anthropic, and reports
// whether injection is active for the given Prepared request. The bool is
// false when p.ResponseSchema is nil (the cap was off, or the caller
// didn't ask for a schema — adapter.Prepare already applied both) or the
// caller also asked for user tools (Anthropic can combine tool_choice with
// user tools but injecting a synthetic tool on top is ambiguous — better to
// fall back to the prompt-embedded schema, matching the Google adapter's
// behavior).
func structuredOutputToolName(p adapter.Prepared) (string, bool) {
	if p.ResponseSchema == nil || len(p.Tools) > 0 {
		return "", false
	}
	if p.ResponseSchema.Name != "" {
		return p.ResponseSchema.Name, true
	}
	return "emit_answer", true
}

// toAnthropicMessages converts normalized messages into Anthropic message
// params, coalescing consecutive tool-result turns into a single user message
// (Anthropic requires tool_result blocks to ride in a user turn). Every
// role and block-kind rule (adapter.Prepare) already holds — this is a
// pure mapping.
//
// Content blocks map in order: text → text blocks, image → image source
// (base64 data or URL), document → document source (base64 PDF or URL),
// thinking → thinking/redacted_thinking blocks re-emitted verbatim from Raw.
// Thinking blocks whose Provider is not "anthropic" (including empty) are
// dropped silently per the llmkit.Block contract.
func toAnthropicMessages(msgs []llmkit.Message) ([]anthropic.MessageParam, error) {
	out := make([]anthropic.MessageParam, 0, len(msgs))
	var pendingResults []anthropic.ContentBlockParamUnion

	flush := func() {
		if len(pendingResults) > 0 {
			out = append(out, anthropic.NewUserMessage(pendingResults...))
			pendingResults = nil
		}
	}

	for _, m := range msgs {
		switch m.Role {
		case llmkit.RoleSystem:
			// System messages are hoisted into params.System by the caller; if one
			// appears inline, treat it as a user instruction to preserve content.
			flush()
			out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Text())))
		case llmkit.RoleUser:
			flush()
			blocks, err := anthropicUserBlocks(m)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropic.NewUserMessage(blocks...))
		case llmkit.RoleAssistant:
			flush()
			blocks, err := anthropicAssistantBlocks(m)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		case llmkit.RoleToolResult:
			pendingResults = append(pendingResults,
				anthropic.NewToolResultBlock(m.ToolCallID, m.Text(), m.IsError))
		}
	}
	flush()
	return out, nil
}

// anthropicUserBlocks maps a user message's content blocks onto Anthropic
// content blocks. The media source rule (adapter.Prepare) already holds;
// the one constraint left here is vendor-specific: inline document bytes
// must be application/pdf. A message with no blocks yields one empty text
// block so an empty user turn still serializes the same wire shape.
func anthropicUserBlocks(m llmkit.Message) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			blocks = append(blocks, anthropic.NewTextBlock(b.Text))
		case llmkit.BlockImage:
			src := anthropic.ImageBlockParamSourceUnion{}
			if b.URL != "" {
				src.OfURL = &anthropic.URLImageSourceParam{URL: b.URL}
			} else {
				src.OfBase64 = &anthropic.Base64ImageSourceParam{
					Data:      base64.StdEncoding.EncodeToString(b.Data),
					MediaType: anthropic.Base64ImageSourceMediaType(b.MediaType),
				}
			}
			blocks = append(blocks, anthropic.ContentBlockParamUnion{
				OfImage: &anthropic.ImageBlockParam{Source: src},
			})
		case llmkit.BlockDocument:
			if len(b.Data) > 0 && b.MediaType != "application/pdf" {
				return nil, adapter.Refuse("anthropic",
					"document block: Anthropic base64 documents support application/pdf only", nil)
			}
			src := anthropic.DocumentBlockParamSourceUnion{}
			if b.URL != "" {
				src.OfURL = &anthropic.URLPDFSourceParam{URL: b.URL}
			} else {
				src.OfBase64 = &anthropic.Base64PDFSourceParam{
					Data: base64.StdEncoding.EncodeToString(b.Data),
				}
			}
			dp := &anthropic.DocumentBlockParam{Source: src}
			if b.Title != "" {
				dp.Title = anthropic.String(b.Title)
			}
			blocks = append(blocks, anthropic.ContentBlockParamUnion{OfDocument: dp})
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, anthropic.NewTextBlock(""))
	}
	return blocks, nil
}

// anthropicAssistantBlocks maps an assistant message's content blocks onto
// Anthropic content blocks: text in order, then thinking blocks re-emitted
// verbatim from Raw (Provider "anthropic" only — foreign thinking blocks are
// dropped silently), then one tool_use block per ToolCall. A message with
// no blocks yields one empty text block, matching the SDK's requirement
// that assistant turns are never empty.
func anthropicAssistantBlocks(m llmkit.Message) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := make([]anthropic.ContentBlockParamUnion, 0, 1+len(m.Content)+len(m.ToolCalls))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			if b.Text != "" {
				blocks = append(blocks, anthropic.NewTextBlock(b.Text))
			}
		case llmkit.BlockThinking:
			if b.Provider != "anthropic" {
				continue
			}
			pb, err := anthropicThinkingBlock(b)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, pb)
		}
	}
	for _, tc := range m.ToolCalls {
		var input any
		if len(tc.Arguments) > 0 {
			if err := json.Unmarshal(tc.Arguments, &input); err != nil {
				return nil, adapter.Refuse("anthropic",
					"assistant tool call "+tc.Name+": invalid arguments JSON", err)
			}
		}
		blocks = append(blocks, anthropic.ContentBlockParamUnion{
			OfToolUse: &anthropic.ToolUseBlockParam{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: input,
			},
		})
	}
	if len(blocks) == 0 {
		blocks = append(blocks, anthropic.NewTextBlock(""))
	}
	return blocks, nil
}

// anthropicThinkingBlock re-emits a captured Anthropic thinking block
// verbatim: the thinking text and signature (or the redacted payload) are
// decoded from Raw — the provider's wire block — and sent back through the
// SDK's typed params, so the next request carries the same signed reasoning.
// Raw that decodes to a thinking block without a signature is refused
// pre-wire: the API verifies the signature when a thinking block is passed
// back (https://platform.claude.com/docs/en/build-with-claude/thinking,
// "Thinking encryption"), so an unsigned block can only fail remotely.
func anthropicThinkingBlock(b llmkit.Block) (anthropic.ContentBlockParamUnion, error) {
	var probe struct {
		Type      string `json:"type"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
		Data      string `json:"data"`
	}
	// A Block that passed through encoding/json with a nil Raw re-decodes as
	// the literal bytes "null" — treat both as missing, not as an empty
	// payload to forward. Trim only the bytes JSON permits as space, so
	// padded forms (" null", whitespace-only) are caught too; padding JSON
	// does not permit (e.g. U+00A0) stays malformed below.
	trimmed := bytes.Trim(b.Raw, " \t\n\r")
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return anthropic.ContentBlockParamUnion{}, adapter.Refuse("anthropic",
			"thinking block: Raw is empty; the verbatim provider payload is required", nil)
	}
	if err := json.Unmarshal(b.Raw, &probe); err != nil {
		return anthropic.ContentBlockParamUnion{}, adapter.Refuse("anthropic",
			"thinking block: malformed Raw JSON", err)
	}
	// A payload that DECODES to nothing carries nothing the API accepts on
	// replay either: forwarding would put an unsigned
	// {"signature":"","thinking":"","type":"thinking"} (or an empty
	// redacted block) on the wire and fail remotely. Fail locally instead,
	// in the same ErrInvalidRequest class as the missing-Raw case.
	if probe.Type == "redacted_thinking" {
		if probe.Data == "" {
			return anthropic.ContentBlockParamUnion{}, adapter.Refuse("anthropic",
				"thinking block: Raw decodes to an empty payload; the verbatim provider payload is required", nil)
		}
		return anthropic.ContentBlockParamUnion{
			OfRedactedThinking: &anthropic.RedactedThinkingBlockParam{Data: probe.Data},
		}, nil
	}
	if probe.Thinking == "" && probe.Signature == "" {
		return anthropic.ContentBlockParamUnion{}, adapter.Refuse("anthropic",
			"thinking block: Raw decodes to an empty payload; the verbatim provider payload is required", nil)
	}
	// Every thinking block the API emits carries a signature, and the API
	// verifies it on replay. After the empty-payload guard above, an empty
	// signature means thinking text without one — e.g. a hand-built
	// {"type":"thinking","thinking":"why"} — which would go out unsigned
	// and fail remotely. Fail locally instead, in the same
	// ErrInvalidRequest class. A block with an empty thinking field and a
	// live signature (the display "omitted" wire shape) still replays.
	if probe.Signature == "" {
		return anthropic.ContentBlockParamUnion{}, adapter.Refuse("anthropic",
			"thinking block: Raw decodes to an unsigned thinking payload; the signature is verified on replay, so the signed provider payload is required", nil)
	}
	return anthropic.ContentBlockParamUnion{
		OfThinking: &anthropic.ThinkingBlockParam{
			Thinking:  probe.Thinking,
			Signature: probe.Signature,
		},
	}, nil
}

func (a *anthropicAdapter) toResponse(msg *anthropic.Message) llmkit.Response {
	var resp llmkit.Response
	var text string
	for _, block := range msg.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			text += v.Text
			resp.Blocks = append(resp.Blocks, llmkit.Block{Kind: llmkit.BlockText, Text: v.Text})
		case anthropic.ThinkingBlock:
			// Raw keeps the provider's wire block (thinking text + signature)
			// byte-for-byte so the caller can re-send it verbatim on the next
			// request.
			resp.Blocks = append(resp.Blocks, llmkit.Block{
				Kind:     llmkit.BlockThinking,
				Text:     v.Thinking,
				Provider: "anthropic",
				Raw:      json.RawMessage(v.RawJSON()),
			})
		case anthropic.RedactedThinkingBlock:
			resp.Blocks = append(resp.Blocks, llmkit.Block{
				Kind:     llmkit.BlockThinking,
				Provider: "anthropic",
				Raw:      json.RawMessage(v.RawJSON()),
			})
		case anthropic.ToolUseBlock:
			resp.ToolCalls = append(resp.ToolCalls, llmkit.ToolCall{
				ID:        v.ID,
				Name:      v.Name,
				Arguments: json.RawMessage(v.JSON.Input.Raw()),
			})
		}
	}
	resp.Text = text
	// Anthropic's input_tokens EXCLUDES cache reads/writes; normalize to the
	// inclusive convention documented on Usage by summing all three.
	resp.Usage = llmkit.Usage{
		InputTokens: msg.Usage.InputTokens +
			msg.Usage.CacheReadInputTokens + msg.Usage.CacheCreationInputTokens,
		OutputTokens:             msg.Usage.OutputTokens,
		CacheReadInputTokens:     msg.Usage.CacheReadInputTokens,
		CacheCreationInputTokens: msg.Usage.CacheCreationInputTokens,
	}
	resp.StopReason = mapAnthropicStop(msg.StopReason, len(resp.ToolCalls) > 0)
	return resp
}

// mapAnthropicStop is this adapter's recognized stop_reason table:
// end_turn/stop_sequence map to EndTurn, tool_use to ToolUse, max_tokens to
// MaxTokens, refusal to Refusal. Every other reason — including
// "pause_turn", a server-paused turn that is NOT a natural completion —
// falls to adapter.StopFallback: ToolUse when the response carries tool
// calls, StopError otherwise. The raw provider reason is not exposed on
// llmkit.Response, so a caller seeing StopError should treat the step as
// failed — surface it and re-issue the request if continuing matters —
// never assume the conversation reached a natural end.
func mapAnthropicStop(sr anthropic.StopReason, hasToolCalls bool) llmkit.StopReason {
	switch sr {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return llmkit.StopEndTurn
	case anthropic.StopReasonToolUse:
		return llmkit.StopToolUse
	case anthropic.StopReasonMaxTokens:
		return llmkit.StopMaxTokens
	case anthropic.StopReasonRefusal:
		return llmkit.StopRefusal
	default:
		return adapter.StopFallback(hasToolCalls)
	}
}

func (a *anthropicAdapter) normalizeErr(ctx context.Context, err error) error {
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		return adapter.NormalizeSDKError("anthropic", adapter.VendorError{
			Status:  apiErr.StatusCode,
			Type:    string(apiErr.Type()),
			Message: apiErr.Error(),
			Header:  adapter.ResponseHeader(apiErr.Response),
			Err:     err,
		})
	}
	// No HTTP response: transport failure or caller's context ending mid-call.
	return adapter.TransportError("anthropic", ctx, err)
}

// Sources (vendor docs consulted for this table):
//   - https://platform.claude.com/docs/en/about-claude/models/overview and
//     the per-model overview pages — context windows: 1M for the Opus 4.6 /
//     4.7 / 4.8 and Sonnet 4.6 generations; 200k for Opus 4.1 / 4.5, Sonnet
//     4.5, Haiku 4.5, and the whole 3.x line (Opus 4.5 / Sonnet 4.x can
//     reach 1M only behind the context-1m beta header, which this adapter
//     does not send, so 200k is the honest number for requests this adapter
//     can make);
//   - https://platform.claude.com/docs/en/build-with-claude/prompt-caching —
//     prompt caching is supported on all active Claude models;
//   - https://platform.claude.com/docs/en/build-with-claude/extended-thinking —
//     extended thinking requires Claude 3.7 Sonnet or newer.
//
// Keys name exactly the generations the cited docs verify, and matching
// (adapter.BestMatchingFamily) requires a "-" segment boundary, so an
// unverified future ID such as "claude-opus-4-9" reports ContextWindow 0
// instead of inheriting a stale window. The two date-stamped 4.0 first
// snapshots (May 2025) are listed explicitly — the vendor overview page
// gives both a 200K window with extended thinking — while the retired bare
// aliases (claude-opus-4 / claude-sonnet-4, deprecated 2026-06) stay
// unlisted: the API rejects those IDs outright, and unknown reports 0
// rather than resurrecting a stale window. Pin exact values for unknown
// models via Options.Capabilities.
//
// Per-entry fields are only those that vary by model: window and thinking.
// Everything else (parallel tool calls, prompt caching, synthetic-tool
// structured output, tool choice, images, documents, stop sequences, top_p,
// top_k) is shared across the 3+ families; there is no seed parameter.
type anthropicModelCaps struct {
	prefix   string
	window   int
	thinking bool
}

var anthropicModelTable = []anthropicModelCaps{
	{"claude-opus-4-8", 1_000_000, true},
	{"claude-opus-4-7", 1_000_000, true},
	{"claude-opus-4-6", 1_000_000, true},
	{"claude-sonnet-4-6", 1_000_000, true},
	// First 4.0-generation snapshots (2025-05-22): 200k window, extended
	// thinking. Source: https://platform.claude.com/docs/en/about-claude/models/overview
	{"claude-opus-4-20250514", 200_000, true},
	{"claude-sonnet-4-20250514", 200_000, true},
	{"claude-opus-4-5", 200_000, true},
	{"claude-opus-4-1", 200_000, true},
	{"claude-sonnet-4-5", 200_000, true},
	{"claude-haiku-4-5", 200_000, true},
	{"claude-3-7-sonnet", 200_000, true},
	{"claude-3-5-sonnet", 200_000, false},
	{"claude-3-5-haiku", 200_000, false},
	{"claude-3-opus", 200_000, false},
	{"claude-3-sonnet", 200_000, false},
	{"claude-3-haiku", 200_000, false},
}

func anthropicCapabilities(model string) llmkit.Capabilities {
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
		Seed:              false,
	}
	best := adapter.BestMatchingFamily(model, anthropicModelTable, func(e anthropicModelCaps) string { return e.prefix })
	if best >= 0 {
		e := anthropicModelTable[best]
		caps.ContextWindow = e.window
		caps.Thinking = e.thinking
	}
	return caps
}

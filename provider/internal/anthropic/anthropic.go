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

// anthropicAdapter maps the normalized types onto the official Anthropic SDK.
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
// and returns the effective one; it is applied once at construction. Flip
// a single field in a closure, or return a fixed profile to pin exact
// values for models the table doesn't know. nil = model-table default.
type Options struct {
	APIKey     string
	AuthToken  string       // OAuth bearer token; mutually exclusive with APIKey
	BaseURL    string       // optional; for testing or proxies
	HTTPClient *http.Client // optional; for testing (httptest)

	Capabilities func(llmkit.Capabilities) llmkit.Capabilities
}

// New builds an Anthropic-backed Client. The vendor SDK's built-in retries are
// disabled (WithMaxRetries(0)) so the shared retry wrapper is the single
// source of retry policy.
//
// When opts.AuthToken is non-empty the adapter uses OAuth bearer-token
// authentication: the SDK sets Authorization: Bearer <token> via
// option.WithAuthToken, and option.WithHeaderAdd appends the required
// "oauth-2025-04-20" value to the anthropic-beta header without clobbering any
// other beta values the SDK may already have set.
func New(model string, opts Options) llmkit.Client {
	reqOpts := []option.RequestOption{
		option.WithMaxRetries(0),
	}
	if opts.AuthToken != "" {
		// OAuth mode: bearer token in Authorization header; x-api-key must be absent.
		reqOpts = append(reqOpts, option.WithAuthToken(opts.AuthToken))
		// Append rather than replace so other anthropic-beta values set by the SDK
		// are preserved alongside the oauth beta flag.
		reqOpts = append(reqOpts, option.WithHeaderAdd("anthropic-beta", "oauth-2025-04-20"))
		// anthropic.NewClient applies env defaults BEFORE explicit options, and a
		// host ANTHROPIC_API_KEY eagerly sets the X-Api-Key header (WithAPIKey
		// applies WithHeader internally). The API rejects requests carrying both
		// credentials, so strip the env-derived header; the SDK documents this
		// exact WithHeaderDel pattern for parent-client key suppression.
		reqOpts = append(reqOpts, option.WithHeaderDel("X-Api-Key"))
	} else {
		// API-key mode: standard x-api-key authentication.
		reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
		// Symmetric guard: a host ANTHROPIC_AUTH_TOKEN env default would have
		// eagerly set Authorization; strip it so only x-api-key is sent.
		reqOpts = append(reqOpts, option.WithHeaderDel("Authorization"))
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	if opts.HTTPClient != nil {
		reqOpts = append(reqOpts, option.WithHTTPClient(opts.HTTPClient))
	}
	caps := adapter.ApplyOverride(anthropicCapabilities(model), opts.Capabilities)
	return &anthropicAdapter{
		client: anthropic.NewClient(reqOpts...),
		model:  model,
		caps:   caps,
	}
}

func (a *anthropicAdapter) Capabilities() llmkit.Capabilities { return a.caps }

func (a *anthropicAdapter) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	params, err := a.buildParams(req)
	if err != nil {
		return llmkit.Response{}, err
	}

	msg, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return llmkit.Response{}, a.normalizeErr(ctx, err)
	}
	return a.finalize(req, a.toResponse(msg)), nil
}

// finalize coerces the synthetic structured-output tool's lone call into
// Response.Text and StopEndTurn: the call's Arguments become Text
// (downstream layers have no handler for the synthetic tool), and
// Anthropic's "tool_use" stop reason for the forced call would otherwise
// mis-classify the completion. The surfaced text is also appended as a
// BlockText, the same way the openai and google toResponse emit surfaced
// text: under forced tool_choice the wire cannot also carry a visible text
// block, so this appends the ONE BlockText the Response's invariant (Text
// equals the concatenation of BlockText blocks) needs — without it,
// verbatim-block consumers such as the agent's assistant history would
// drop the text. Complete and Stream both go through this step so the same
// wire exchange returns identical Responses.
func (a *anthropicAdapter) finalize(req llmkit.Request, resp llmkit.Response) llmkit.Response {
	if toolName, ok := structuredOutputToolName(req, a.caps); ok &&
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

func (a *anthropicAdapter) buildParams(req llmkit.Request) (anthropic.MessageNewParams, error) {
	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		// Anthropic requires max_tokens; the uniform default applies.
		maxTokens = llmkit.DefaultMaxTokens
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(a.model),
		MaxTokens: maxTokens,
	}
	if req.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: req.System}}
	}
	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}

	// Extended thinking. The budget rides through verbatim; Anthropic
	// enforces budget < MaxTokens and its 1024 floor server-side. Gated on
	// the capability profile: extended thinking exists only on Claude 3.7+
	// (see anthropicCapabilities), and the documented contract for a false
	// feature is a silent drop rather than a server 400. thinkingSent is
	// reused by the forced-tool-use refusal below.
	thinkingSent := req.Thinking != nil && a.caps.Thinking
	if thinkingSent {
		if req.Thinking.BudgetTokens <= 0 {
			return anthropic.MessageNewParams{}, &llmkit.APIError{
				Kind:     llmkit.ErrInvalidRequest,
				Provider: "anthropic",
				Message:  "Thinking.BudgetTokens must be positive",
			}
		}
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: int64(req.Thinking.BudgetTokens),
			},
		}
	}
	if len(req.StopSequences) > 0 {
		params.StopSequences = req.StopSequences
	}
	if req.TopP != nil {
		params.TopP = anthropic.Float(*req.TopP)
	}
	if req.TopK != nil {
		params.TopK = anthropic.Int(int64(*req.TopK))
	}
	// Request.Seed: the Anthropic Messages API has no seed parameter, so it
	// is dropped here and documented via Capabilities.Seed = false.
	msgs, err := toAnthropicMessages(req.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}
	params.Messages = msgs

	if len(req.Tools) > 0 {
		tools := make([]anthropic.ToolUnionParam, 0, len(req.Tools))
		for _, t := range req.Tools {
			tp, err := toAnthropicTool(t)
			if err != nil {
				return anthropic.MessageNewParams{}, err
			}
			tools = append(tools, anthropic.ToolUnionParam{OfTool: tp})
		}
		params.Tools = tools
	}

	// Capability gate first: a profile with ToolChoice=false must reject
	// every explicit mode before the wire call (dropping "none" would
	// escalate permissions; dropping "required"/"tool" would silently
	// degrade). Request-level tool choice is then validated UNCONDITIONALLY
	// — an unknown mode or a missing Name must error even when a schema
	// below forces the synthetic tool. The synthetic forcing then overwrites
	// whatever was mapped: a structured-output request is meaningless
	// without the forced tool call, so it keeps precedence over
	// req.ToolChoice.
	if err := adapter.GateToolChoice("anthropic", req.ToolChoice, a.caps.ToolChoice); err != nil {
		return anthropic.MessageNewParams{}, err
	}
	if err := applyAnthropicToolChoice(&params, req.ToolChoice); err != nil {
		return anthropic.MessageNewParams{}, err
	}

	// Schema-constrained output. Anthropic has no native response_format, so
	// we inject a single synthetic tool and force tool_choice to it — the
	// model returns a tool_use block whose `input` is the schema-conformant
	// JSON, which Complete surfaces as Response.Text. Only valid when the
	// caller didn't supply user tools (structuredOutputToolName gates this).
	synthTool, syntheticOutput := structuredOutputToolName(req, a.caps)
	if syntheticOutput {
		// Mirror toAnthropicTool's schema unwrapping via the shared helper
		// so the synthetic tool gets the same ToolInputSchemaParam shape
		// the SDK would receive for a user tool.
		properties, required, err := adapter.ParseToolParameters(req.ResponseSchema)
		if err != nil {
			return anthropic.MessageNewParams{}, &llmkit.APIError{
				Kind:     llmkit.ErrInvalidRequest,
				Provider: "anthropic",
				Message:  "ResponseSchema: invalid JSON",
				Err:      err,
			}
		}
		schema := anthropic.ToolInputSchemaParam{
			Properties: properties,
			Required:   required,
		}
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name:        synthTool,
			InputSchema: schema,
			Description: anthropic.String("Emit the final answer that conforms to the response schema."),
		}})
		params.ToolChoice = anthropic.ToolChoiceParamOfTool(synthTool)
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
	// mapped — so the built params here are the one place both sources are
	// visible, and one predicate covers both. A guaranteed remote 400 is
	// refused locally, naming the field that did the forcing. When
	// caps.Thinking is false the thinking config is dropped earlier and the
	// combination stays legal.
	if thinkingSent &&
		(params.ToolChoice.OfAny != nil || params.ToolChoice.OfTool != nil) {
		forcing := "ToolChoice.Mode=required forces tool_choice any"
		if params.ToolChoice.OfTool != nil {
			forcing = "ToolChoice.Mode=tool forces tool_choice to the " + params.ToolChoice.OfTool.Name + " tool"
		}
		if syntheticOutput {
			forcing = "ResponseSchema forces tool_choice to the " + synthTool + " tool"
		}
		return anthropic.MessageNewParams{}, &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "Thinking cannot be combined with forced tool use: manual extended thinking only supports tool_choice auto or none, and " + forcing,
		}
	}

	applyCacheBreakpoints(&params)
	return params, nil
}

// applyAnthropicToolChoice maps the normalized tool-choice request onto the
// Anthropic tool_choice parameter. Auto (and the zero value) is the provider
// default and is never serialized.
func applyAnthropicToolChoice(params *anthropic.MessageNewParams, tc llmkit.ToolChoice) error {
	switch tc.Mode {
	case "", llmkit.ToolChoiceAuto:
		return nil
	case llmkit.ToolChoiceNone:
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	case llmkit.ToolChoiceRequired:
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	case llmkit.ToolChoiceTool:
		if tc.Name == "" {
			return &llmkit.APIError{
				Kind:     llmkit.ErrInvalidRequest,
				Provider: "anthropic",
				Message:  "ToolChoice.Mode=tool requires ToolChoice.Name",
			}
		}
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{Name: tc.Name},
		}
	default:
		return &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "unknown ToolChoice.Mode " + string(tc.Mode),
		}
	}
	return nil
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

func toAnthropicTool(t llmkit.ToolDef) (*anthropic.ToolParam, error) {
	properties, required, err := adapter.ParseToolParameters(t.Parameters)
	if err != nil {
		return nil, &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "tool " + t.Name + ": invalid parameters JSON schema",
			Err:      err,
		}
	}
	schema := anthropic.ToolInputSchemaParam{
		Properties: properties,
		Required:   required,
	}
	tp := &anthropic.ToolParam{
		Name:        t.Name,
		InputSchema: schema,
	}
	if t.Description != "" {
		tp.Description = anthropic.String(t.Description)
	}
	return tp, nil
}

// structuredOutputToolName returns the synthetic tool name this adapter
// injects to coerce schema-constrained output from Anthropic, and reports
// whether injection is active for the given request. The bool is false when
// the cap is off, the caller didn't ask for a schema, or the caller also
// asked for user tools (Anthropic can combine tool_choice with user tools
// but injecting a synthetic tool on top is ambiguous — better to fall back
// to the prompt-embedded schema, matching the Google adapter's behavior).
func structuredOutputToolName(req llmkit.Request, caps llmkit.Capabilities) (string, bool) {
	if len(req.ResponseSchema) == 0 || !caps.StructuredOutput || len(req.Tools) > 0 {
		return "", false
	}
	if req.ResponseSchemaName != "" {
		return req.ResponseSchemaName, true
	}
	return "emit_answer", true
}

// toAnthropicMessages converts normalized messages into Anthropic message
// params, coalescing consecutive tool-result turns into a single user message
// (Anthropic requires tool_result blocks to ride in a user turn).
//
// Per-role block rule (ValidateMessageBlocks, before any mapping):
// user text/image/document; assistant text/thinking (Provider-matched only);
// system and tool-result text only. Violations are ErrInvalidRequest.
//
// Content blocks map in order: text → text blocks, image → image source
// (base64 data or URL), document → document source (base64 PDF or URL),
// thinking → thinking/redacted_thinking blocks re-emitted verbatim from Raw.
// Thinking blocks whose Provider is not "anthropic" (including empty) are
// dropped silently per the llmkit.Block contract. Validation of image and
// document sources happens here, BEFORE any wire call.
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
		// Per-role block-kind rule + media source rule, before any mapping.
		if err := adapter.ValidateMessageBlocks("anthropic", m); err != nil {
			return nil, err
		}
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
		default:
			return nil, &llmkit.APIError{
				Kind:     llmkit.ErrInvalidRequest,
				Provider: "anthropic",
				Message:  "unknown message role " + string(m.Role),
			}
		}
	}
	flush()
	return out, nil
}

// anthropicUserBlocks maps a user message's content blocks onto Anthropic
// content blocks. A message with no blocks yields one empty text block so
// an empty user turn still serializes the same wire shape.
func anthropicUserBlocks(m llmkit.Message) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Kind {
		case llmkit.BlockText:
			blocks = append(blocks, anthropic.NewTextBlock(b.Text))
		case llmkit.BlockImage:
			if err := adapter.ValidateMediaBlock("anthropic", b); err != nil {
				return nil, err
			}
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
			if err := adapter.ValidateMediaBlock("anthropic", b); err != nil {
				return nil, err
			}
			if len(b.Data) > 0 && b.MediaType != "application/pdf" {
				return nil, &llmkit.APIError{
					Kind:     llmkit.ErrInvalidRequest,
					Provider: "anthropic",
					Message:  "document block: Anthropic base64 documents support application/pdf only",
				}
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
				return nil, &llmkit.APIError{
					Kind:     llmkit.ErrInvalidRequest,
					Provider: "anthropic",
					Message:  "assistant tool call " + tc.Name + ": invalid arguments JSON",
					Err:      err,
				}
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
		return anthropic.ContentBlockParamUnion{}, &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "thinking block: Raw is empty; the verbatim provider payload is required",
		}
	}
	if err := json.Unmarshal(b.Raw, &probe); err != nil {
		return anthropic.ContentBlockParamUnion{}, &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "thinking block: malformed Raw JSON",
			Err:      err,
		}
	}
	// A payload that DECODES to nothing carries nothing the API accepts on
	// replay either: forwarding would put an unsigned
	// {"signature":"","thinking":"","type":"thinking"} (or an empty
	// redacted block) on the wire and fail remotely. Fail locally instead,
	// in the same ErrInvalidRequest class as the missing-Raw case.
	if probe.Type == "redacted_thinking" {
		if probe.Data == "" {
			return anthropic.ContentBlockParamUnion{}, &llmkit.APIError{
				Kind:     llmkit.ErrInvalidRequest,
				Provider: "anthropic",
				Message:  "thinking block: Raw decodes to an empty payload; the verbatim provider payload is required",
			}
		}
		return anthropic.ContentBlockParamUnion{
			OfRedactedThinking: &anthropic.RedactedThinkingBlockParam{Data: probe.Data},
		}, nil
	}
	if probe.Thinking == "" && probe.Signature == "" {
		return anthropic.ContentBlockParamUnion{}, &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "thinking block: Raw decodes to an empty payload; the verbatim provider payload is required",
		}
	}
	// Every thinking block the API emits carries a signature, and the API
	// verifies it on replay (URL in the anthropicThinkingBlock doc
	// comment). After the empty-payload guard above, an empty signature
	// means thinking text without one — e.g. a hand-built
	// {"type":"thinking","thinking":"why"} — which would go out unsigned
	// and fail remotely. Fail locally instead, in the same
	// ErrInvalidRequest class. A block with an empty thinking field and a
	// live signature (the display "omitted" wire shape) still replays.
	if probe.Signature == "" {
		return anthropic.ContentBlockParamUnion{}, &llmkit.APIError{
			Kind:     llmkit.ErrInvalidRequest,
			Provider: "anthropic",
			Message:  "thinking block: Raw decodes to an unsigned thinking payload; the signature is verified on replay, so the signed provider payload is required",
		}
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
	resp.StopReason = mapAnthropicStop(msg.StopReason)
	return resp
}

func mapAnthropicStop(sr anthropic.StopReason) llmkit.StopReason {
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
		// Catch-all per llmkit.StopError's contract. This includes
		// pause_turn: a server-paused turn is NOT a natural completion, and
		// reporting StopEndTurn made an agent loop treat it as one. The raw
		// provider reason is not exposed on llmkit.Response, so a caller
		// seeing StopError should treat the step as failed — surface it and
		// re-issue the request if continuing matters — never assume the
		// conversation reached a natural end.
		return llmkit.StopError
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
// (adapter.MatchesModelFamily) requires a "-" segment boundary, so an
// unverified future ID such as "claude-opus-4-9" reports ContextWindow 0
// instead of inheriting a stale window. The two date-stamped 4.0 first
// snapshots (May 2025) are listed explicitly — the vendor overview page
// (https://platform.claude.com/docs/en/about-claude/models/overview) gives
// both a 200K window with extended thinking — while the retired bare
// aliases (claude-opus-4 / claude-sonnet-4, deprecated 2026-06) stay
// unlisted: the API rejects those IDs outright, and unknown reports 0
// rather than resurrecting a stale window. Pin exact values for unknown
// models via Options.Capabilities.
//
// Only the fields that genuinely vary by model are stored per entry;
// everything else is shared (parallel tool calls, prompt caching,
// synthetic-tool structured output, tool choice, images, documents, stop
// sequences, top_p, and top_k are supported across the 3+ families; there
// is no seed parameter).
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

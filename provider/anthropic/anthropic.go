// Package anthropic adapts the normalized llmkit types onto the official
// Anthropic SDK (API-key or OAuth bearer-token authentication).
package anthropic

import (
	"context"
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
// StructuredOutput, when non-nil, overrides the adapter's built-in default
// (true). nil = adapter default.
type Options struct {
	APIKey     string
	AuthToken  string       // OAuth bearer token; mutually exclusive with APIKey
	BaseURL    string       // optional; for testing or proxies
	HTTPClient *http.Client // optional; for testing (httptest)

	StructuredOutput *bool
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
	caps := anthropicCapabilities(model)
	if opts.StructuredOutput != nil {
		caps.StructuredOutput = *opts.StructuredOutput
	}
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
		return llmkit.Response{}, a.normalizeErr(err)
	}
	resp := a.toResponse(msg)
	// When we forced a synthetic structured-output tool, the model replies
	// with a tool_use block carrying the schema-conformant JSON. Convert
	// that single tool call into Response.Text so downstream layers see JSON
	// text instead of a tool call they have no handler for.
	if toolName, ok := structuredOutputToolName(req, a.caps); ok &&
		len(resp.ToolCalls) == 1 && resp.ToolCalls[0].Name == toolName {
		resp.Text = string(resp.ToolCalls[0].Arguments)
		resp.ToolCalls = nil
		// Anthropic's stop_reason for a forced tool call is "tool_use",
		// which would mis-classify the response downstream; coerce to
		// StopEndTurn ("end_turn") so the run loop treats it as a normal
		// completion.
		resp.StopReason = llmkit.StopEndTurn
	}
	return resp, nil
}

func (a *anthropicAdapter) buildParams(req llmkit.Request) (anthropic.MessageNewParams, error) {
	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 4096
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

	// Schema-constrained output. Anthropic has no native response_format, so
	// we inject a single synthetic tool and force tool_choice to it — the
	// model returns a tool_use block whose `input` is the schema-conformant
	// JSON, which Complete surfaces as Response.Text. Only valid when the
	// caller didn't supply user tools (structuredOutputToolName gates this).
	if toolName, ok := structuredOutputToolName(req, a.caps); ok {
		// Mirror toAnthropicTool's schema unwrapping via the shared helper
		// so the synthetic tool gets the same ToolInputSchemaParam shape
		// the SDK would receive for a user tool.
		properties, required, err := adapter.ParseToolParameters(req.ResponseSchema)
		if err != nil {
			return anthropic.MessageNewParams{}, llmkit.NewAPIError("anthropic", 0, 0,
				llmkit.ErrInvalidRequest, "ResponseSchema: invalid JSON", err)
		}
		schema := anthropic.ToolInputSchemaParam{
			Properties: properties,
			Required:   required,
		}
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name:        toolName,
			InputSchema: schema,
			Description: anthropic.String("Emit the final answer that conforms to the response schema."),
		}})
		params.ToolChoice = anthropic.ToolChoiceParamOfTool(toolName)
	}

	applyCacheBreakpoints(&params)
	return params, nil
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
// reporting whether a markable block was found. Thinking blocks (and any other
// variant without a CacheControl field) are skipped by GetCacheControl
// returning nil; the adapter never produces those today.
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
		return nil, llmkit.NewAPIError("anthropic", 0, 0, llmkit.ErrInvalidRequest,
			"tool "+t.Name+": invalid parameters JSON schema", err)
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
			out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
		case llmkit.RoleUser:
			flush()
			out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
		case llmkit.RoleAssistant:
			flush()
			blocks := make([]anthropic.ContentBlockParamUnion, 0, 1+len(m.ToolCalls))
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var input any
				if len(tc.Arguments) > 0 {
					if err := json.Unmarshal(tc.Arguments, &input); err != nil {
						return nil, llmkit.NewAPIError("anthropic", 0, 0, llmkit.ErrInvalidRequest,
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
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		case llmkit.RoleToolResult:
			pendingResults = append(pendingResults,
				anthropic.NewToolResultBlock(m.ToolCallID, m.Content, m.IsError))
		default:
			return nil, llmkit.NewAPIError("anthropic", 0, 0, llmkit.ErrInvalidRequest,
				"unknown message role "+string(m.Role), nil)
		}
	}
	flush()
	return out, nil
}

func (a *anthropicAdapter) toResponse(msg *anthropic.Message) llmkit.Response {
	var resp llmkit.Response
	var text string
	for _, block := range msg.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			text += v.Text
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
		return llmkit.StopError
	default:
		return llmkit.StopEndTurn
	}
}

func (a *anthropicAdapter) normalizeErr(err error) error {
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		return adapter.NormalizeSDKError("anthropic", apiErr.StatusCode, apiErr.Error(), apiErr.Response, err)
	}
	// Transport/timeout error: leave status 0, mark as server-class so it is
	// retried.
	return llmkit.NewAPIError("anthropic", 0, 0, llmkit.ErrServer, err.Error(), err)
}

func anthropicCapabilities(model string) llmkit.Capabilities {
	// Anthropic models support parallel tool calls, prompt caching, and
	// structured output. Context window varies; 200k is a safe floor for the
	// 4.x family and beyond. The model parameter is accepted for symmetry
	// with the other capability constructors.
	return llmkit.Capabilities{
		ContextWindow:     200_000,
		ParallelToolCalls: true,
		PromptCaching:     true,
		StructuredOutput:  true,
	}
}

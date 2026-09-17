// Package google adapts the normalized llmkit types onto the
// google.golang.org/genai SDK (Gemini API backend).
package google

import (
	"context"
	"encoding/json"
	"errors"
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
// StructuredOutput, when non-nil, overrides the adapter's built-in default
// (true). nil = adapter default.
type Options struct {
	APIKey     string
	BaseURL    string       // optional; for testing or non-default endpoints
	HTTPClient *http.Client // optional; for testing (httptest)

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
	if req.MaxTokens > 0 {
		cfg.MaxOutputTokens = int32(req.MaxTokens)
	}
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
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

// toGoogleContents converts normalized messages into genai Contents. Gemini
// uses "user"/"model" roles; tool results are sent as user-turn
// functionResponse parts.
func toGoogleContents(msgs []llmkit.Message) ([]*genai.Content, error) {
	out := make([]*genai.Content, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case llmkit.RoleSystem:
			// Hoisted into SystemInstruction by the caller; if inline, attach as a
			// user turn to preserve content.
			out = append(out, &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: m.Content}},
			})
		case llmkit.RoleUser:
			out = append(out, &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: m.Content}},
			})
		case llmkit.RoleAssistant:
			parts := make([]*genai.Part, 0, 1+len(m.ToolCalls))
			if m.Content != "" {
				parts = append(parts, &genai.Part{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				var args map[string]any
				if len(tc.Arguments) > 0 {
					if err := json.Unmarshal(tc.Arguments, &args); err != nil {
						return nil, llmkit.NewAPIError("google", 0, 0, llmkit.ErrInvalidRequest,
							"assistant tool call "+tc.Name+": invalid arguments JSON", err)
					}
				}
				parts = append(parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   tc.ID,
						Name: tc.Name,
						Args: args,
					},
				})
			}
			out = append(out, &genai.Content{Role: "model", Parts: parts})
		case llmkit.RoleToolResult:
			// Gemini expects the function result wrapped under an "output" key.
			resultObj := map[string]any{"output": m.Content}
			if m.IsError {
				resultObj = map[string]any{"error": m.Content}
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
				if p.Text != "" {
					text += p.Text
				}
				if p.FunctionCall != nil {
					args, _ := json.Marshal(p.FunctionCall.Args)
					if len(args) == 0 || string(args) == "null" {
						args = json.RawMessage("{}")
					}
					out.ToolCalls = append(out.ToolCalls, llmkit.ToolCall{
						ID:        p.FunctionCall.ID,
						Name:      p.FunctionCall.Name,
						Arguments: json.RawMessage(args),
					})
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
		return llmkit.StopError
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

func googleCapabilities(model string) llmkit.Capabilities {
	// Gemini supports parallel function calls and structured output. Prompt
	// caching is implicit (server-side, automatic on 2.x models); the adapter
	// surfaces cache hits via Usage.CacheReadInputTokens but does not manage
	// explicit CachedContent. The model parameter is accepted for symmetry
	// with the other capability constructors.
	return llmkit.Capabilities{
		ContextWindow:     1_000_000,
		ParallelToolCalls: true,
		PromptCaching:     true,
		StructuredOutput:  true,
	}
}

// Package llmkit is a thin, provider-agnostic abstraction over LLM backends
// (Anthropic, OpenAI, Google, and any OpenAI-compatible endpoint). Provider
// construction lives in llmkit/provider and its adapter subpackages; this
// package holds the normalized types, errors, and client wrappers.
//
// It normalizes three things that otherwise differ wildly between providers:
//
//   - the request/response shape (messages, content blocks, tools, tool-call
//     round-trips, usage);
//   - capability profiles (context window, parallel tool calls, prompt caching,
//     structured output, and which sampling/thinking/image parameters each
//     adapter honors) so callers can adapt without provider sniffing;
//   - errors (rate limiting, auth, context-too-long, ...) into a small typed set.
//
// The layer is deliberately thin: each adapter maps these normalized types
// to/from its vendor SDK and nothing more. Higher-level concerns (agent tool
// loops, budgets, transcripts) live in the caller's code.
//
// API keys are never logged. They are resolved by the caller and handed
// straight to the vendor SDK.
package llmkit

import (
	"context"
	"encoding/json"
	"strings"
)

// Role enumerates the normalized message roles. Each adapter maps these to its
// provider's own role vocabulary.
type Role string

const (
	// RoleSystem is a system / developer instruction. Adapters hoist these into
	// the provider's dedicated system field where one exists.
	RoleSystem Role = "system"
	// RoleUser is an end-user (or harness) turn.
	RoleUser Role = "user"
	// RoleAssistant is a model turn. When it carries ToolCalls it represents the
	// model's request to invoke tools (Anthropic tool_use / OpenAI tool_calls /
	// Gemini functionCall).
	RoleAssistant Role = "assistant"
	// RoleToolResult carries the result of a previously-requested tool call back
	// to the model. ToolCallID must reference the originating ToolCall.ID.
	RoleToolResult Role = "tool-result"
)

// BlockKind enumerates the kinds of content a single Block can carry.
type BlockKind string

const (
	// BlockText is plain text (the common case).
	BlockText BlockKind = "text"
	// BlockImage is an image, carried inline (Data + MediaType) or by
	// reference (URL). Exactly one of Data and URL must be set; adapters
	// reject any other combination with an error wrapping ErrInvalidRequest
	// before any wire call.
	BlockImage BlockKind = "image"
	// BlockDocument is a document (e.g. PDF), carried like BlockImage.
	// Provider-specific format constraints (Anthropic accepts base64 PDFs
	// only; the OpenAI file part accepts inline data only) are rejected
	// with an error wrapping ErrInvalidRequest before any wire call.
	BlockDocument BlockKind = "document"
	// BlockThinking is a provider reasoning block (e.g. Anthropic extended
	// thinking). Provider names the adapter that produced/signed it (e.g.
	// "anthropic"); Raw carries the provider's wire block verbatim and is
	// never interpreted. An adapter re-emits a thinking block verbatim ONLY
	// when its Provider matches the adapter itself — blocks from a foreign
	// provider are silently dropped on the way out. For anthropic/google
	// blocks Text additionally carries the thinking payload for readability;
	// Raw remains authoritative for the round-trip.
	BlockThinking BlockKind = "thinking"
)

// Block is one unit of message content. A Message's Content is an ordered
// list of blocks; adapters translate each kind onto the provider's wire
// format (Anthropic image/document/thinking blocks, OpenAI image_url/file
// content parts, Gemini inline_data/file_data parts, ...).
type Block struct {
	Kind BlockKind
	// Text holds text for BlockText and, informationally, the thinking
	// payload for BlockThinking. Message.Text concatenates BlockText only.
	Text string
	// MediaType is the MIME type of Data (image/document only).
	MediaType string
	// Data holds inline bytes for image/document blocks (the raw bytes, NOT
	// pre-encoded). encoding/json base64s Data in JSONL transcripts and
	// decodes it back; adapters base64-encode it again per provider wire
	// format. Mutually exclusive with URL: exactly one must be set.
	Data []byte
	// URL references an image/document instead of inlining it. Mutually
	// exclusive with Data. Providers without a URL source for a kind reject
	// such blocks with ErrInvalidRequest.
	URL string
	// Title is an optional document label (BlockDocument only).
	Title string
	// Provider names the adapter that produced/signed a BlockThinking block
	// ("anthropic", "google"). See BlockThinking for the round-trip rule.
	Provider string
	// Raw carries a BlockThinking block's provider wire JSON verbatim and is
	// never interpreted. When the vendor SDK does not expose the original
	// bytes (google), Raw is a faithful re-encoding of the parsed block.
	Raw json.RawMessage
}

// Message is a single normalized turn in a conversation.
//
// The meaning of the fields depends on Role:
//
//   - system/user: Content holds the content blocks (text, images,
//     documents). TextMessage builds the common single-text-block form.
//   - assistant: Content holds any text and thinking blocks; ToolCalls holds
//     tool-use requests. Either or both may be present. An assistant message
//     that carries thinking blocks re-emits them verbatim on the next
//     request (see BlockThinking for the per-provider rule).
//   - tool-result: ToolCallID identifies the call being answered, Content
//     holds the (textual) result, and IsError marks a failed execution.
//
// Per-role block kinds, enforced by every adapter BEFORE any wire call
// (violations are errors wrapping ErrInvalidRequest): RoleUser carries
// text/image/document; RoleAssistant carries text/thinking; RoleSystem and
// RoleToolResult carry text only.
type Message struct {
	Role Role
	// Content is the message's content blocks. Build the common
	// single-text form with TextMessage; read it back with Text.
	Content []Block
	// ToolCalls is set on assistant turns that request tool invocations.
	ToolCalls []ToolCall
	// ToolCallID is set only on RoleToolResult messages.
	ToolCallID string
	// IsError marks a RoleToolResult as a failed tool execution.
	IsError bool
}

// TextMessage returns a Message whose content is the single text block s —
// the one-line form for the common text-only case.
func TextMessage(role Role, s string) Message {
	return Message{Role: role, Content: []Block{{Kind: BlockText, Text: s}}}
}

// Text concatenates the message's BlockText blocks in order. Image,
// document, and thinking blocks are not text and are skipped.
func (m Message) Text() string {
	var b strings.Builder
	for _, blk := range m.Content {
		if blk.Kind == BlockText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// ToolDef declares a tool the model may call. Parameters is a JSON Schema object
// describing the tool's arguments, carried verbatim as raw JSON so callers keep
// full control over the schema and adapters never lose fidelity.
type ToolDef struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ToolCall is a single tool invocation requested by the model. Arguments is the
// raw JSON argument object exactly as the model produced it; callers should
// json.Unmarshal it rather than string-matching, since providers differ in
// escaping.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ThinkingConfig requests provider reasoning (Anthropic extended thinking,
// Gemini thinkingConfig). Nil on Request.Thinking leaves reasoning off. On
// providers that honor thinking, BudgetTokens must be positive and their
// adapters reject anything else with an error wrapping ErrInvalidRequest;
// providers without reasoning support (Capabilities.Thinking = false, e.g.
// the OpenAI adapters this round) drop the whole field and never see the
// budget.
type ThinkingConfig struct {
	BudgetTokens int
}

// ToolChoiceMode enumerates how strongly the model is steered toward tool
// use. The zero value (ToolChoiceAuto) is the provider default and is never
// serialized onto the wire.
type ToolChoiceMode string

const (
	// ToolChoiceAuto lets the model decide (provider default).
	ToolChoiceAuto ToolChoiceMode = "auto"
	// ToolChoiceNone forbids tool use.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired forces the model to call SOME tool.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces a call to the tool named in ToolChoice.Name.
	ToolChoiceTool ToolChoiceMode = "tool"
)

// ToolChoice steers the model's tool use. The zero value means auto (never
// sent on the wire). Name must be set only for ToolChoiceTool. On Anthropic
// a synthetic structured-output request keeps precedence: when a
// ResponseSchema is honored via the forced synthetic tool, ToolChoice is
// overridden by that forcing.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

// Request is a normalized completion request.
type Request struct {
	// System is an optional system prompt. It is kept separate from Messages so
	// adapters can route it to the provider's dedicated system field.
	System string
	// Messages is the ordered conversation. It must not be empty.
	Messages []Message
	// Tools is the set of tools the model may call. May be empty.
	Tools []ToolDef
	// MaxTokens caps output tokens. If zero, the adapter applies a sane default.
	MaxTokens int
	// Temperature is the sampling temperature. Nil means "use the provider
	// default" (some models reject an explicit temperature). Use a pointer so
	// callers can distinguish "0.0" from "unset".
	Temperature *float64
	// Thinking requests provider reasoning. Nil means off. Adapters that do
	// not support reasoning drop it silently and report Capabilities.Thinking
	// = false.
	Thinking *ThinkingConfig
	// ToolChoice steers tool use. Zero value = auto (never sent on the wire).
	// Adapters that cannot express a mode reject the request with an error
	// wrapping ErrInvalidRequest rather than silently ignoring it.
	ToolChoice ToolChoice
	// StopSequences makes the model stop when it generates any of these
	// strings (a matching provider reports StopEndTurn).
	StopSequences []string
	// TopP is nucleus-sampling mass. Nil means provider default.
	TopP *float64
	// TopK truncates sampling to the K most likely tokens. Nil means
	// provider default; adapters without a TopK concept report
	// Capabilities.TopK = false and drop it.
	TopK *int
	// Seed biases providers that support deterministic sampling toward the
	// same output for identical requests. Nil means unset; adapters without
	// a seed concept report Capabilities.Seed = false and drop it.
	Seed *int64
	// ResponseSchema is an optional JSON Schema (encoded as raw JSON) requesting
	// schema-constrained output. Adapters honor it only when their
	// Capabilities().StructuredOutput is true; otherwise the schema is
	// silently dropped (callers should check the capability before relying
	// on structured output). Zero value (nil) means no schema request.
	ResponseSchema json.RawMessage
	// ResponseSchemaName names the schema on the wire. It is used as the
	// response_format name on OpenAI-style backends and as the synthetic
	// forced-output tool name on Anthropic. Adapters default it to
	// "response" / "emit_answer" when empty.
	ResponseSchemaName string
}

// StopReason is the normalized reason a completion ended.
type StopReason string

const (
	// StopEndTurn: the model finished its turn naturally.
	StopEndTurn StopReason = "end_turn"
	// StopToolUse: the model is requesting one or more tool calls.
	StopToolUse StopReason = "tool_use"
	// StopMaxTokens: output was truncated at the token limit.
	StopMaxTokens StopReason = "max_tokens"
	// StopRefusal: the model declined the request on policy grounds
	// (Anthropic stop_reason "refusal").
	StopRefusal StopReason = "refusal"
	// StopContentFilter: output was blocked by a provider safety filter
	// (OpenAI finish_reason "content_filter", Gemini SAFETY/RECITATION/
	// PROHIBITED_CONTENT/BLOCKLIST/SPII).
	StopContentFilter StopReason = "content_filter"
	// StopError: the model stopped for a provider-specific reason that maps to
	// none of the above. Check the response text and provider logs for detail.
	StopError StopReason = "error"
)

// Usage reports token consumption for a single completion. Callers ledger this
// per role/provider/model via a Recorder.
//
// Normalization convention: InputTokens is the TOTAL prompt size — it INCLUDES
// any tokens that were read from or written to the provider's prompt cache.
// CacheReadInputTokens and CacheCreationInputTokens are informational subsets
// of InputTokens used to compute cache savings; they never add to it.
//
// This matches the native OpenAI (prompt_tokens includes cached_tokens) and
// Gemini (promptTokenCount includes cachedContentTokenCount) conventions.
// Anthropic reports input_tokens EXCLUDING cache reads/writes, so its adapter
// sums input_tokens + cache_read_input_tokens + cache_creation_input_tokens
// into InputTokens. The payoff: existing ledger math (budgets, context-size
// estimates, input+output totals) keeps meaning "tokens the model processed"
// regardless of cache hits, and a caller that ignores the cache fields sees
// exactly the pre-caching numbers.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
	// CacheReadInputTokens is the subset of InputTokens served from the
	// provider's prompt cache (billed at a steep discount: ~0.1x on Anthropic,
	// 0.25–0.5x on OpenAI). Zero when the provider reports no cache activity.
	CacheReadInputTokens int64
	// CacheCreationInputTokens is the subset of InputTokens written to the
	// prompt cache this call (Anthropic bills these at 1.25x). Only Anthropic
	// reports this; it is zero elsewhere.
	CacheCreationInputTokens int64
}

// ChargeableTokens returns the budget-relevant token count for this usage,
// discounting cache reads by cacheReadWeight (0..1). Raw InputTokens counts
// cache reads at full weight, but they bill at a steep discount (~0.1x
// Anthropic, 0.25–0.5x OpenAI), so a cache-heavy run exhausts a raw-token
// budget far faster than its real cost warrants. cacheReadWeight of 1.0
// reproduces the old behavior.
//
// Cache CREATION tokens are left at full weight (Anthropic bills them at
// 1.25x; treating them as cheap would understate cost).
func (u Usage) ChargeableTokens(cacheReadWeight float64) int64 {
	uncached := u.InputTokens - u.CacheReadInputTokens
	if uncached < 0 {
		uncached = 0
	}
	weighted := uncached + int64(float64(u.CacheReadInputTokens)*cacheReadWeight)
	return weighted + u.OutputTokens
}

// Response is a normalized completion response.
type Response struct {
	// Text is the concatenated assistant text output (may be empty when the model
	// only requested tools).
	Text string
	// Blocks carries every content block of the response in provider order —
	// text, thinking, and any other kind the adapter surfaced. Text equals
	// the concatenation of the BlockText blocks here.
	Blocks []Block
	// ToolCalls holds any tool-use requests the model made.
	ToolCalls []ToolCall
	// Usage reports token consumption.
	Usage Usage
	// StopReason is the normalized stop reason.
	StopReason StopReason
}

// Capabilities describes what a given provider+model supports, so callers can
// adapt (e.g. serialize tool calls when ParallelToolCalls is false) without
// sniffing the provider type. The bools name features a Request can ask for;
// a false feature is silently dropped by the adapter (see each field).
type Capabilities struct {
	// ContextWindow is the model's maximum input+output token window. Zero means
	// unknown (e.g. an arbitrary OpenAI-compatible endpoint).
	ContextWindow int
	// ParallelToolCalls reports whether the model may return more than one tool
	// call in a single response.
	ParallelToolCalls bool
	// PromptCaching reports whether the provider supports prompt caching.
	PromptCaching bool
	// StructuredOutput reports whether the provider supports schema-constrained
	// JSON output.
	StructuredOutput bool
	// Thinking reports whether the adapter forwards Request.Thinking (reasoning
	// budgets) and returns BlockThinking blocks.
	Thinking bool
	// ToolChoice reports whether Request.ToolChoice is honored.
	ToolChoice bool
	// Images reports whether BlockImage content is sent on the wire.
	Images bool
	// Documents reports whether BlockDocument content is sent on the wire.
	Documents bool
	// StopSequences reports whether Request.StopSequences is honored.
	StopSequences bool
	// TopP reports whether Request.TopP is honored.
	TopP bool
	// TopK reports whether Request.TopK is honored.
	TopK bool
	// Seed reports whether Request.Seed is honored.
	Seed bool
}

// Client is the single interface every adapter implements. It is intentionally
// minimal: one synchronous completion call plus a capability probe.
type Client interface {
	// Complete runs a single completion. It is context-aware: cancellation and
	// deadlines propagate to the underlying request. Errors are normalized into
	// the typed errors in errors.go where possible.
	Complete(ctx context.Context, req Request) (Response, error)
	// Capabilities returns the static capability profile for this client's
	// provider+model.
	Capabilities() Capabilities
}

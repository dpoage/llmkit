// Package llmkit is a thin, provider-agnostic abstraction over LLM backends
// (Anthropic, OpenAI, Google, and any OpenAI-compatible endpoint). Provider
// construction lives in llmkit/provider and its adapter subpackages; this
// package holds the normalized types, errors, and client wrappers that every
// provider shares.
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
// # Messages and blocks
//
// A conversation is a list of [Message] values. Each Message carries an
// ordered list of [Block] values plus role-specific fields. Build blocks with
// [Text], [Image], [ImageURL], [Document], and [DocumentURL]; assemble
// messages with [TextMessage], [UserMessage], [SystemMessage], [ToolResult],
// and [ToolError].
//
// Each role accepts specific block kinds, and every adapter enforces the rule
// BEFORE any wire call. A violation returns an error wrapping
// [ErrInvalidRequest]:
//
//   - [RoleUser]: text, image, document
//   - [RoleAssistant]: text and thinking; [Message.ToolCalls] carries tool-use
//     requests
//   - [RoleSystem] and [RoleToolResult]: text only
//
// The constructors panic on inputs no adapter can accept (an image without
// bytes, a tool result without a call ID). Hand-built [Block] and [Message]
// literals stay legal; they reach the same adapter validation and produce the
// same [ErrInvalidRequest] when invalid.
//
// # Requests and responses
//
// [Request] is the normalized completion request; [Response] is the normalized
// completion. A [Request.MaxTokens] of zero or negative means
// [DefaultMaxTokens] on every adapter. [Response.Text] concatenates the text
// blocks; [Response.Blocks] keeps every block in provider order;
// [Response.ToolCalls] carries raw JSON arguments.
//
// # Usage
//
// [Usage] reports token consumption. [Usage.InputTokens] is the TOTAL prompt
// size: it includes cache-read and cache-creation tokens.
// [Usage.CacheReadInputTokens] and [Usage.CacheCreationInputTokens] are
// informational subsets of InputTokens, never additions to it. A caller that
// ignores the cache fields sees exactly the pre-caching numbers.
// [Usage.ChargeableTokens] discounts cache reads by a weight for budget math;
// cache-creation tokens stay at full weight.
//
// # Errors
//
// Complete and Stream return [*APIError] for provider-side failures. Each
// APIError wraps exactly one sentinel ([ErrRateLimited], [ErrAuth],
// [ErrContextTooLong], [ErrInvalidRequest], [ErrServer], [ErrOverloaded]);
// match it with errors.Is. Rate-limit and overload errors carry any
// Retry-After the server sent. Unclassifiable transport failures (timeouts,
// connection resets) surface as [ErrServer]. See docs/providers.md for the
// HTTP-status-to-sentinel table.
//
// # Capabilities
//
// [Client.Capabilities] reports what the provider+model supports as a
// [Capabilities] value, so callers adapt without provider sniffing. Every
// field belongs to one of four enforcement classes:
//
//   - DROPPED SILENTLY when false: the adapter omits the feature from the wire
//     (StructuredOutput, Thinking, StopSequences, TopP, TopK, Seed).
//   - REFUSED PRE-WIRE: the adapter rejects the request with an error wrapping
//     [ErrInvalidRequest] before the wire call (ToolChoice).
//   - DECORATOR: provider.New installs a wrapping Client that follows the
//     field (ParallelToolCalls).
//   - ADVISORY: information for callers; no adapter reads the field
//     (ContextWindow, PromptCaching, Images, Documents).
//
// See docs/capabilities.md for the per-field table.
//
// # Decorators
//
// Three wrappers compose around any [Client]:
//
//   - [WithRetry] retries transient failures (429, 5xx, transport timeouts)
//     with exponential backoff and honors Retry-After.
//   - [WithRecorder] reports each successful completion's usage to a
//     [Recorder].
//   - [WithSerializedToolCalls] truncates multi-tool-call responses to the
//     first call; a no-op on parallel-capable clients.
//
// The wrappers compose over streaming too: retry stops once a delta is
// delivered, the recorder reports the final streamed response, and
// serialization drops tool-call deltas for any index beyond the first.
//
// # Streaming
//
// A client that delivers a completion incrementally implements
// [StreamingClient]. [Stream] accepts ANY [Client]: it delegates to
// StreamingClient.Stream when the client implements it, and otherwise calls
// Complete and synthesizes deltas from the response (text and thinking blocks
// in block order, then one DeltaToolCall per tool call). Callers never
// special-case a non-streaming backend.
//
// # Reasoning text
//
// Reasoning models served through plain-text channels inline
// "<think>...</think>" spans in the visible answer. [StripThinkBlocks] removes
// leading think spans, including one unclosed by a truncated thought, and
// leaves spans embedded inside the body intact.
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
	// RoleToolResult carries the result of a tool call back
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
	// when its Provider matches the adapter — blocks from a foreign
	// provider are silently dropped on the way out. For anthropic/google
	// blocks Text additionally carries the thinking payload for readability;
	// Raw remains authoritative for the round-trip.
	BlockThinking BlockKind = "thinking"
)

// Block is one unit of message content. A Message's Content is an ordered
// list of blocks; adapters translate each kind onto the provider's wire
// format (Anthropic image/document/thinking blocks, OpenAI image_url/file
// content parts, Gemini inline_data/file_data parts, ...).
//
// Build blocks with [Text], [Image], [ImageURL], [Document], and
// [DocumentURL]. The struct stays flat so JSONL transcripts keep
// round-tripping; hand-built literals remain legal and are validated by the
// adapters before any wire call. The json tags exist for transcripts: zero
// fields are omitted, so a text block serializes as exactly
// {"kind":"text","text":"…"} and a nil Raw never emits "raw":null.
type Block struct {
	Kind BlockKind `json:"kind"`
	// Text holds text for BlockText and, informationally, the thinking
	// payload for BlockThinking. Message.Text concatenates BlockText only.
	Text string `json:"text,omitempty"`
	// MediaType is the MIME type of Data (image/document only).
	MediaType string `json:"media_type,omitempty"`
	// Data holds inline bytes for image/document blocks (the raw bytes, NOT
	// pre-encoded). encoding/json base64s Data in JSONL transcripts and
	// decodes it back; adapters base64-encode it again per provider wire
	// format. Mutually exclusive with URL: exactly one must be set.
	Data []byte `json:"data,omitempty"`
	// URL references an image/document instead of inlining it. Mutually
	// exclusive with Data. Providers without a URL source for a kind reject
	// such blocks with ErrInvalidRequest.
	URL string `json:"url,omitempty"`
	// Title is an optional document label (BlockDocument only).
	Title string `json:"title,omitempty"`
	// Provider names the adapter that produced/signed a BlockThinking block
	// ("anthropic", "google"). See BlockThinking for the round-trip rule.
	Provider string `json:"provider,omitempty"`
	// Raw carries a BlockThinking block's provider wire JSON verbatim and is
	// never interpreted. When the vendor SDK does not expose the original
	// bytes (google), Raw is a faithful re-encoding of the parsed block.
	Raw json.RawMessage `json:"raw,omitempty"`
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
	// Content is the message's content blocks.
	Content []Block
	// ToolCalls holds tool-use requests on assistant turns.
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

// Text returns a Block carrying the plain text s.
func Text(s string) Block { return Block{Kind: BlockText, Text: s} }

// Image returns an inline image block: mediaType is the MIME type
// ("image/png", "image/jpeg", ...), data the raw image bytes (not
// base64-encoded; encoding/json handles that in transcripts). It panics on
// an empty mediaType or data: an inline image without either cannot satisfy
// the BlockImage contract on any provider, and the violation is a
// programming error at the call site. Hand-built blocks skip this check;
// the adapters reject them with ErrInvalidRequest before any wire call.
func Image(mediaType string, data []byte) Block {
	if mediaType == "" {
		panic("llmkit.Image: empty mediaType")
	}
	if len(data) == 0 {
		panic("llmkit.Image: empty data")
	}
	return Block{Kind: BlockImage, MediaType: mediaType, Data: data}
}

// ImageURL returns an image block referencing url instead of carrying
// inline data. It panics on an empty url, like [Image].
func ImageURL(url string) Block {
	if url == "" {
		panic("llmkit.ImageURL: empty url")
	}
	return Block{Kind: BlockImage, URL: url}
}

// Document returns an inline document block (e.g. a PDF). title is an
// optional label and may be empty. It panics on an empty mediaType or
// data, like [Image].
func Document(mediaType, title string, data []byte) Block {
	if mediaType == "" {
		panic("llmkit.Document: empty mediaType")
	}
	if len(data) == 0 {
		panic("llmkit.Document: empty data")
	}
	return Block{Kind: BlockDocument, MediaType: mediaType, Title: title, Data: data}
}

// DocumentURL returns a document block referencing url instead of carrying
// inline data. title is an optional label and may be empty. It panics on an
// empty url, like [Image].
func DocumentURL(url, title string) Block {
	if url == "" {
		panic("llmkit.DocumentURL: empty url")
	}
	return Block{Kind: BlockDocument, URL: url, Title: title}
}

// UserMessage returns a user turn carrying the given blocks — the form for
// a turn that mixes text with an image or document ([TextMessage] covers
// the text-only case). It panics on zero blocks: a user turn with no
// content is rejected by every adapter before any wire call.
func UserMessage(blocks ...Block) Message {
	if len(blocks) == 0 {
		panic("llmkit.UserMessage: no blocks")
	}
	return Message{Role: RoleUser, Content: blocks}
}

// SystemMessage returns a system-instruction turn carrying the single text
// block s.
func SystemMessage(s string) Message {
	return Message{Role: RoleSystem, Content: []Block{Text(s)}}
}

// ToolResult returns a tool-result message answering the call with the
// given ID. callID must be non-empty: a result naming no call can never be
// paired with its ToolCall, and building one is a programming error. Use
// [ToolError] for a failed execution.
func ToolResult(callID, text string) Message {
	if callID == "" {
		panic("llmkit.ToolResult: empty callID")
	}
	return Message{Role: RoleToolResult, Content: []Block{Text(text)}, ToolCallID: callID}
}

// ToolError returns a failed tool-result message answering the call with
// the given ID: the IsError mark tells the model the execution failed.
// callID must be non-empty, like [ToolResult].
func ToolError(callID, text string) Message {
	if callID == "" {
		panic("llmkit.ToolError: empty callID")
	}
	return Message{Role: RoleToolResult, Content: []Block{Text(text)}, ToolCallID: callID, IsError: true}
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
// OpenAI adapters) drop the whole field and never see the budget.
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
	// Messages is the ordered conversation. It must not be empty. An inline
	// RoleSystem message is honored, but where it lands on the wire differs:
	// OpenAI keeps it as a system-role entry in place; Anthropic and Gemini
	// have no system role inside the message list, so it is sent as a user
	// turn (Request.System is the reliable system channel).
	Messages []Message
	// Tools is the set of tools the model may call. May be empty.
	Tools []ToolDef
	// MaxTokens caps output tokens. If zero (or negative), every adapter
	// applies the same default: llmkit.DefaultMaxTokens. An explicit value is
	// passed through verbatim.
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
	// Two rejection paths wrap ErrInvalidRequest rather than silently
	// ignoring the request: an adapter that cannot express a mode, and —
	// enforced by every adapter before the wire call — a model whose
	// Capabilities.ToolChoice is false receives an error for any explicit
	// mode (auto stays allowed). Dropping an explicit "none" would let the
	// model call tools the caller tried to forbid, so it is refused, not
	// ignored.
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
	// (Anthropic stop_reason "refusal"; a non-empty OpenAI message.refusal,
	// surfaced as the response text).
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
// budget far faster than its real cost warrants. cacheReadWeight = 1.0
// reproduces the uncached-discount-free count.

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
// sniffing the provider type. Every field is classified by its first
// sentence into one of four enforcement classes:
//
//   - Dropped silently when false: the adapter omits the request feature
//     from the wire (no error). Check the field before relying on the
//     feature.
//   - Refused pre-wire: the adapter rejects the request with an error
//     wrapping ErrInvalidRequest before any wire call.
//   - Decorator: provider.New installs a wrapping Client whose behavior
//     follows the field.
//   - Advisory: information for callers; no adapter reads the field.
type Capabilities struct {
	// Advisory: the model's maximum input+output token window; no adapter
	// reads it — callers use it to bound their own history (a zero value
	// means unknown). Adapters report 0 for any model outside their
	// per-model table (on every provider, never a fabricated fallback
	// number) and for arbitrary OpenAI-compatible endpoints. Pin a value for
	// such models via provider.Spec.Capabilities.
	ContextWindow int
	// Decorator: when false, provider.New installs the tool-call serializer,
	// which truncates a multi-tool-call response to its first call so
	// callers always see at most one call per turn; when true the client is
	// passed through unwrapped. True also reports that the model may return
	// more than one tool call in a single response.
	ParallelToolCalls bool
	// Advisory: whether the provider supports prompt caching; no adapter
	// reads it — callers use it to decide whether a stable request prefix
	// pays off.
	PromptCaching bool
	// Dropped silently when false: a Request.ResponseSchema is omitted from
	// the wire (the prompt-embedded schema instruction, if any, is the only
	// enforcement). When true, adapters honor the schema natively
	// (grammar-constrained decoding / synthetic forced-output tool).
	StructuredOutput bool
	// Dropped silently when false: Request.Thinking is omitted from the wire
	// and no BlockThinking blocks are returned. When true, the adapter
	// forwards the reasoning budget and thinking blocks round-trip.
	Thinking bool
	// Refused pre-wire: any explicit Request.ToolChoice mode other than auto
	// makes the adapter return an error wrapping ErrInvalidRequest before
	// the wire call (auto stays allowed — dropping an explicit "none" would
	// let the model call tools the caller tried to forbid).
	ToolChoice bool
	// Advisory: whether BlockImage content is sent on the wire; no adapter
	// reads it — image blocks are passed through to providers that accept
	// them (conservatively false profiles on some endpoints still carry
	// vision models).
	Images bool
	// Advisory: whether BlockDocument content is sent on the wire; no
	// adapter reads it — document blocks are passed through to providers
	// that accept them.
	Documents bool
	// Dropped silently when false: adapters whose profile reports false
	// never serialize Request.StopSequences. The field reports the
	// adapter's own mapping — it is not a gate a caller-pinned profile can
	// use to disable a supported feature.
	StopSequences bool
	// Dropped silently when false: adapters whose profile reports false
	// never serialize Request.TopP (same adapter-mapping semantics as
	// [Capabilities.StopSequences]).
	TopP bool
	// Dropped silently when false: adapters whose profile reports false
	// never serialize Request.TopK (same adapter-mapping semantics as
	// [Capabilities.StopSequences]).
	TopK bool
	// Dropped silently when false: adapters whose profile reports false
	// never serialize Request.Seed (same adapter-mapping semantics as
	// [Capabilities.StopSequences]).
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

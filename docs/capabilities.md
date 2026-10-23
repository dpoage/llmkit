# Capabilities

`Capabilities` is the profile a llmkit client reports about its provider+model pair. The profile lists what the model supports, how big its context window is, and which request parameters the adapter honors. Callers read the profile to adapt their requests without sniffing the provider type. See the [`llmkit.Capabilities` reference](https://pkg.go.dev/github.com/dpoage/llmkit#Capabilities).

## What a client reports

`Client.Capabilities()` returns the profile. Adapters derive it from a per-model table at construction:

- A table entry for the model gives the model's context window and feature flags.
- No table entry gives `ContextWindow: 0` with the API-level feature defaults. llmkit never fabricates a window.
- The `openai-compatible` provider uses one conservative profile for every model, because llmkit cannot know what an arbitrary endpoint supports. The profile still reports `Images` and `Documents` as `true`.
- `provider.Spec.Capabilities` overrides the derived profile once, at construction.

## The four enforcement classes

Every `Capabilities` field belongs to exactly one enforcement class. The class tells you what happens when the field is false.

1. **Dropped silently** — the adapter omits the feature from the request. No error is returned. The caller checks the field before relying on the feature.
2. **Refused pre-wire** — the adapter rejects the request with an error wrapping `llmkit.ErrInvalidRequest` before any network call.
3. **Decorator** — `provider.New` installs a wrapping client whose behavior follows the field.
4. **Advisory** — no adapter reads the field. The field is information for callers; nothing on the wire changes when it is false.

## Per-field table

| Field | Class | What happens when false (or zero) | Who checks |
|---|---|---|---|
| `ContextWindow` | Advisory | Nothing: no adapter reads it. `0` means unknown. Callers use it to bound their own history. | Caller |
| `ParallelToolCalls` | Decorator | `provider.New` installs the tool-call serializer: a response with several tool calls is truncated to the first call, and `Capabilities()` reports `ParallelToolCalls: false`. | `provider.New` |
| `PromptCaching` | Advisory | Nothing: no adapter reads it. Callers use it to decide whether a stable request prefix pays off. The OpenAI adapter still parses `cached_tokens` when an endpoint reports it. | Caller |
| `StructuredOutput` | Dropped silently | `Request.ResponseSchema` is omitted from the wire. When true, OpenAI sends `response_format` (alongside tools too). Google sends `responseMimeType` with the parsed `ResponseJsonSchema`. Anthropic installs a synthetic forced-output tool. The Google and Anthropic paths are skipped when the request also carries user tools. | Adapter, at request build |
| `Thinking` | Dropped silently | `ThinkingConfig.BudgetTokens` is omitted from the wire. Reasoning an endpoint returns anyway (OpenAI-compatible `reasoning_content` / `reasoning_details`) is still surfaced as a thinking block and round-tripped, whatever the field says. When true, an invalid `ThinkingConfig` is refused pre-wire; the `ReasoningEffort` row defines invalid. On Anthropic, `Thinking` combined with forced tool use — `ResponseSchema`, an explicit `ToolChoice` of `required`, or a named `tool` — is also refused pre-wire. Manual extended thinking admits only `tool_choice` `auto` or `none`. | Adapter, at request build |
| `ReasoningEffort` | Dropped silently | `ThinkingConfig.Effort` (`low`, `medium`, `high`) is omitted from the wire. When true, OpenAI sends `reasoning_effort` and Google sends `generationConfig.thinkingConfig.thinkingLevel`. True for OpenAI `gpt-5`, `o1`, `o3` and `o4-mini` entries and Gemini 3 models; false for every other table entry, unknown models, and the `openai-compatible` profile. An `Effort` outside the set is refused pre-wire on every profile. A `ThinkingConfig` is valid with a positive `BudgetTokens`, or with a zero `BudgetTokens` and an `Effort`. Wherever `Thinking` or `ReasoningEffort` is true, an invalid config is refused pre-wire, and a negative `BudgetTokens` is invalid even with an `Effort`. When both are false, the config is dropped. Fields the profile reports false are dropped, and when nothing survives no thinking config is sent. Google sends `Effort` alone whenever both fields survive: a Gemini 3 model takes `thinkingLevel` or `thinkingBudget` but not both, and an override can set both on any Google model. | Effective profile, checked at request build |
| `ToolChoice` | Refused pre-wire | Any explicit mode other than `auto` returns an error wrapping `ErrInvalidRequest` before the wire call. `auto` and the zero value stay allowed. Dropping an explicit `none` would let the model call tools the caller tried to forbid, so the adapter refuses it instead of ignoring it. On Anthropic, `required` and a named `tool` also refuse pre-wire when combined with `Thinking` — see the `Thinking` row. | Adapter, before the wire call |
| `ToolResultMedia` | Refused pre-wire | A `RoleToolResult` message carrying an image or document block returns an error wrapping `ErrInvalidRequest` before the wire call, naming `ToolResultMedia`; the adapter never maps the message's text and drops the media. When true, Anthropic puts the blocks, in order, in the `tool_result` content (base64 or URL image, base64 PDF or URL document; inline non-PDF documents are refused). Google puts them in `functionResponse.parts` as `inlineData`, in order, with the text staying in the response `output`; only `image/png`, `image/jpeg`, `image/webp`, `application/pdf` and `text/plain` bytes are accepted, and URL sources and other media types are refused pre-wire (`FunctionResponsePart.fileData` is unsupported in the Gemini API). True for Anthropic and for Gemini 3 models; false for every other Google model, OpenAI, and `openai-compatible`. It is wire-gated: the OpenAI and `openai-compatible` ceiling is false (Chat Completions tool messages are text), so `provider.New` refuses an override reporting `true` there. A client called directly gets these refusals; the agent Runner sends only portable blocks and degrades the rest to a text marker (see [agent loop](agent-loop.md#tools)). | Adapter, before the wire call |
| `Images` | Advisory | Nothing: no adapter reads it. Every adapter's table entry, every unknown-model fallback, and the `openai-compatible` profile report `true`. llmkit does not track which models accept images, so text-only models such as `gpt-3.5-turbo` and `o1-mini` report `true` too. A `true` value does not mean the model takes images. llmkit sends an image block whatever the field says, unless a check refuses the block before the wire call. The checks include the block rules, on every adapter, the [accepted media types](#accepted-media-types) check, and, for a tool-result image, the `ToolResultMedia` gate. The block rules are the source rule (set exactly one of `Data` and `URL`; inline `Data` needs a `MediaType`) and the role rule (only user and tool-result messages carry image or document blocks). A tool-result image is gated by `ToolResultMedia`, not by this field. If no check refuses the block, the provider decides whether to accept the image. A caller who knows a model is text-only can pin `false` with `Spec.Capabilities` and check the field before sending images; the field does not stop the adapters from sending the block. | Caller |
| `Documents` | Advisory | Nothing: no adapter reads it. The profile values are the same as for `Images`: every table entry, unknown-model fallback, and the `openai-compatible` profile report `true`, whatever the model accepts. llmkit sends a document block whatever the field says, unless a check refuses the block before the wire call. No such check reads this field. The checks include the block rules, the [accepted media types](#accepted-media-types) check, and, for a tool-result document, the `ToolResultMedia` gate. The block rules include the source rule (set exactly one of `Data` and `URL`; inline `Data` needs a `MediaType`) and the role rule (only user and tool-result messages carry image or document blocks), on every adapter. The OpenAI adapters also refuse URL-sourced document blocks pre-wire (the Chat Completions file part accepts inline data only), and Anthropic refuses inline or `data:` URL documents that are not `application/pdf`. Anthropic and Google accept URL-sourced documents too. If no check refuses the block, the provider decides whether to accept the document. A document block in a tool result is gated by `ToolResultMedia` instead: with it false, the adapter refuses the request pre-wire. | Caller; the adapters refuse a block that fails a check named in this row |
| `StopSequences` | Dropped silently | The effective profile gates `Request.StopSequences` off the wire: a caller-pinned override reporting `false` drops the field, the same as an adapter's own table entry reporting `false`. An override reporting `true` above the adapter's ceiling (the field it can actually send) is refused at construction — see [Where the classes are pinned](#where-the-classes-are-pinned). | Effective profile, checked at request build |
| `TopP` | Dropped silently | The effective profile gates `Request.TopP` off the wire (same gate semantics as `StopSequences`). | Effective profile, checked at request build |
| `TopK` | Dropped silently | The effective profile gates `Request.TopK` off the wire (same gate semantics as `StopSequences`). Example: the Chat Completions API has no `top_k`, so the OpenAI ceiling forbids `TopK: true`. | Effective profile, checked at request build |
| `Seed` | Dropped silently | The effective profile gates `Request.Seed` off the wire (same gate semantics as `StopSequences`). Example: the Messages API has no seed parameter, so the Anthropic ceiling forbids `Seed: true`. | Effective profile, checked at request build |

The field classes above match each field's doc comment in `llmkit.Capabilities`.

## Accepted media types

Whatever `Capabilities.Images` and `Capabilities.Documents` say, the Anthropic, OpenAI (`TypeOpenAI`), and Google adapters check an image or document block's media type against the wire's accepted set before any network call. Two sources are checked against the same set:

- inline bytes: the block's `MediaType`;
- a `data:` URL in `Block.URL` (scheme matched case-insensitively): the text after `data:` up to the first `;` or `,`. The type is empty when neither character appears, and an empty type is refused.

A type outside the set returns an `*llmkit.APIError` wrapping `ErrInvalidRequest`, with `StatusCode` 0, whose message names the media type; no request is sent. The comparison is exact bytes, so an inline `MediaType` of `image/jpg`, `IMAGE/PNG`, or `image/png; x=y` is refused on all three wires.

| Wire | Block and position | Accepted media types |
|---|---|---|
| Anthropic | Image, inline or `data:` URL, in a user turn or a tool result | `image/png`, `image/jpeg`, `image/gif`, `image/webp` |
| Anthropic | Document, inline or `data:` URL, in a user turn or a tool result | `application/pdf` |
| OpenAI (`TypeOpenAI`) | Image, inline or `data:` URL, user turn | `image/png`, `image/jpeg`, `image/gif`, `image/webp` |
| OpenAI (`TypeOpenAI`) | Document, inline, user turn | `application/pdf` |
| OpenAI-compatible | Inline image or document, or `data:` URL image, user turn | No check: the inline `MediaType` is written to the wire unchanged and a `data:` URL is forwarded verbatim. The host decides whether to reject a type. |
| Google | Inline bytes or a `data:` URL, user turn. Image and document blocks use the same set. On a `data:` URL block, a non-empty `MediaType` is sent as `fileData.mimeType`, so it must be in this set too. | `image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`, `application/pdf`, `text/html`, `text/css`, `text/plain`, `text/xml`, `text/csv`, `text/rtf`, `text/javascript`, `application/json`, `audio/wav`, `audio/mp3`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac`, `audio/mpeg`, `audio/m4a`, `audio/l16`, `audio/opus`, `audio/alaw`, `audio/mulaw`, `audio/webm`, `video/mp4`, `video/mpeg`, `video/mov`, `video/avi`, `video/x-flv`, `video/mpg`, `video/webm`, `video/wmv`, `video/3gpp` |
| Google | Inline bytes, tool result (any URL source, `data:` included, is refused) | `image/png`, `image/jpeg`, `image/webp`, `application/pdf`, `text/plain` |

Any other URL (`https:`, `gs:`, a Gemini Files API URI) is not checked on any wire, because its type cannot be known before the request. Google sends such a block's `MediaType` unchanged as `fileData.mimeType`; Anthropic and OpenAI write no media type for a URL source. Separate rules still apply to URL sources: both OpenAI types refuse URL-sourced documents, and Google refuses any URL source in a tool result. Anthropic accepts URL-sourced documents in both positions.

Sources: [Anthropic vision](https://platform.claude.com/docs/en/build-with-claude/vision), [OpenAI images](https://developers.openai.com/api/docs/guides/images-vision), [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs), and the Gemini [image](https://ai.google.dev/gemini-api/docs/image-understanding), [audio](https://ai.google.dev/gemini-api/docs/audio), [video](https://ai.google.dev/gemini-api/docs/video-understanding), and [document](https://ai.google.dev/gemini-api/docs/document-processing) guides. `provider/media_type_test.go` pins the media-type rows. `provider/tool_result_media_test.go` pins the Google tool-result URL refusal.

## ContextWindow zero means unknown

Adapters report `ContextWindow: 0` in two cases. The model has no entry in the adapter's per-model table. The endpoint is an arbitrary `openai-compatible` server.

Treat `0` as "size unknown". Do not divide by it or compare against it.

## Overriding the profile

Set `Spec.Capabilities` to a function that receives the table-derived profile and returns the effective one. The override runs once at construction. The adapters gate wire behavior on the effective profile, not on the reported one.

Flip one field and keep the rest of the table:

```go
spec := provider.Spec{
	Type:    provider.TypeOpenAICompatible,
	Model:   "qwen3-custom",
	BaseURL: "http://localhost:11434/v1",
	Secret:  "ollama",
	Capabilities: func(c llmkit.Capabilities) llmkit.Capabilities {
		c.ContextWindow = 131072
		return c
	},
}
client, err := provider.New(context.Background(), spec, provider.Options{})
```

Replace the profile wholesale to pin exact values for a model no table knows:

```go
Capabilities: func(llmkit.Capabilities) llmkit.Capabilities {
	return llmkit.Capabilities{ContextWindow: 131072}
}
```

A wholesale profile enforces every wire-gated field it leaves at its Go zero value: the closure above sends a profile with every wire-gated field false, so `StopSequences`, `TopP`, `TopK`, `Seed`, `ToolChoice`, `Thinking`, `ReasoningEffort`, `StructuredOutput`, and `ToolResultMedia` are all dropped or refused, not only the fields the closure names. The advisory and decorator fields (`ContextWindow`, `ParallelToolCalls`, `PromptCaching`, `Images`, `Documents`) are not gated by this rule: the profile's `Images` and `Documents` fields gate no block, so an inline image block in a user message reaches the wire on every adapter even when the wholesale profile reports `Images: false`, provided its media type passes the [accepted media types](#accepted-media-types) check (`openai-compatible` has none). An inline image block in a tool result does not: with `ToolResultMedia` false it does not reach the wire, and the adapter refuses the request. Block validation is unchanged and does not depend on the profile: the OpenAI adapters refuse URL-sourced document blocks, the Anthropic adapter refuses inline documents that are not `application/pdf`, and the media-type check applies whatever the profile reports. Flip each field you need explicitly, or start from the table-derived input the closure receives instead of ignoring it.

Each provider type has a ceiling: the wire-gated fields (`StructuredOutput`, `Thinking`, `ReasoningEffort`, `ToolChoice`, `StopSequences`, `TopP`, `TopK`, `Seed`, `ToolResultMedia`) it can actually put on the wire. `provider.New` refuses an override that reports `true` for a field above the ceiling, with an error wrapping `llmkit.ErrInvalidRequest` naming the field and the Type — before any network call. Anthropic's ceiling excludes `Seed` and `ReasoningEffort` (the adapter does not map effort); OpenAI and `openai-compatible`'s ceiling excludes `Thinking`, `TopK` and `ToolResultMedia`; Google's ceiling has no excluded field.

See [providers](providers.md) for the construction rules. See `provider/example_test.go` (`ExampleSpec_capabilities`) for the runnable version of the first snippet above.

## Where the classes are pinned

`provider/capability_gate_test.go` pins the ToolChoice and Thinking classes where adapters gate on them. `provider/prepare_gate_test.go` pins the `StopSequences`/`TopP`/`TopK`/`Seed` gate, one row per (field, adapter) whose ceiling allows the field, plus the wholesale-override case. `provider/prepare_ceiling_test.go` pins the ceiling refusal.

A profile with `ToolChoice=false` rejects each explicit mode pre-wire on every adapter. Anthropic rejects through the `Spec.Capabilities` override. Google rejects through the tool-less `gemini-2.0-flash-lite` table entry.

`provider/tool_result_media_test.go` pins `ToolResultMedia`: the Anthropic and Gemini 3 wire shapes, the capability refusal on every false profile, the Google media-type and URL refusals, and the OpenAI-family ceiling.

A profile with `Thinking=false` drops `ThinkingConfig.BudgetTokens` from the wire body on the two adapters that gate a wire parameter on it: Anthropic and Google. A config that sets only `BudgetTokens` then sends no thinking config.

A profile with `ReasoningEffort=false` drops `ThinkingConfig.Effort` from the wire on every adapter; with it true, OpenAI carries `reasoning_effort` and Google carries `thinkingLevel` (`LOW`, `MEDIUM`, `HIGH`). `provider/reasoning_effort_test.go` pins this per table entry, the invalid-config outcomes per profile class (including a negative `BudgetTokens` with an `Effort`), and the anthropic ceiling refusal.

Reasoning spend reports on `Usage.ReasoningTokens` whatever the profile says: OpenAI's `completion_tokens_details.reasoning_tokens` and Gemini's `thoughtsTokenCount`; Anthropic reports no count and stays 0. It is a subset of `OutputTokens`, with one exception. The Google adapter adds `thoughtsTokenCount` into `OutputTokens` because Gemini reports it outside `candidatesTokenCount`, so budgets see thinking runs. The OpenAI and OpenAI-compatible adapters add `reasoning_tokens` into `OutputTokens` only when `total_tokens` equals prompt plus completion plus reasoning (xAI style). If an endpoint counts reasoning outside `completion_tokens` and its `total_tokens` is absent or differs from that sum, no fold happens and `ReasoningTokens` can exceed `OutputTokens`.

A response whose endpoint reports no usage (an OpenAI-compatible server that ignores `stream_options.include_usage`, a Gemini response without `usageMetadata`) carries `Usage.Estimated = true` on the OpenAI, OpenAI-compatible and Google adapters, whatever the profile says: `InputTokens` is bytes/4 of the request body sent, `OutputTokens` is bytes/4 of the response text, tool calls and thinking blocks, and `ReasoningTokens` is bytes/4 of the thinking blocks, also counted inside `OutputTokens`. A present all-zero usage object is reported zeros, not an estimate. The Anthropic adapter never sets `Estimated`: a response without usage reads as zero counts.

A profile with a sampler field false drops that field's wire key even when the caller's request sets it; a profile with the field true carries the caller's value. The wire keys are `top_p`, `seed`, and `stop` on OpenAI and openai-compatible (the Chat Completions API has no `top_k`); `top_p`, `top_k`, `stop_sequences`, and `thinking` on Anthropic (the Messages API has no `seed`); and `generationConfig.topP`, `topK`, `seed`, `stopSequences`, and `thinkingConfig` on Google. `provider.New` refuses `TypeOpenAI`/`TypeOpenAICompatible` `Thinking` or `TopK` overrides, `TypeAnthropic` `Seed` and `ReasoningEffort` overrides (the adapter does not map effort), before any network call; `TypeGoogle` has no refused field.

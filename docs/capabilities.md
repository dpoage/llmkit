# Capabilities

`Capabilities` is the profile a llmkit client reports about its provider+model pair. The profile lists what the model supports, how big its context window is, and which request parameters the adapter honors. Callers read the profile to adapt their requests without sniffing the provider type. See the [`llmkit.Capabilities` reference](https://pkg.go.dev/github.com/dpoage/llmkit#Capabilities).

## What a client reports

`Client.Capabilities()` returns the profile. Adapters derive it from a per-model table at construction:

- A table entry for the model gives the model's context window and feature flags.
- No table entry gives `ContextWindow: 0` with the API-level feature defaults. llmkit never fabricates a window.
- The `openai-compatible` provider uses one conservative profile for every model, because llmkit cannot know what an arbitrary endpoint supports.
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
| `Thinking` | Dropped silently | `Request.Thinking` is omitted from the wire and no thinking blocks are returned. When true, a non-positive `BudgetTokens` is refused pre-wire. On Anthropic, `Thinking` combined with forced tool use — `ResponseSchema`, an explicit `ToolChoice` of `required`, or a named `tool` — is also refused pre-wire. Manual extended thinking admits only `tool_choice` `auto` or `none`. | Adapter, at request build |
| `ToolChoice` | Refused pre-wire | Any explicit mode other than `auto` returns an error wrapping `ErrInvalidRequest` before the wire call. `auto` and the zero value stay allowed. Dropping an explicit `none` would let the model call tools the caller tried to forbid, so the adapter refuses it instead of ignoring it. On Anthropic, `required` and a named `tool` also refuse pre-wire when combined with `Thinking` — see the `Thinking` row. | Adapter, before the wire call |
| `Images` | Advisory | Nothing: no adapter reads it. Image blocks still go to the provider. A provider that cannot render them rejects the request itself. The caller checks the field before sending images. | Caller |
| `Documents` | Advisory | Nothing: no adapter reads it. Document blocks pass through on every provider, with one source restriction that is block validation rather than a read of this field. The OpenAI adapter refuses URL-sourced document blocks pre-wire (the Chat Completions file part accepts inline data only). Inline PDF bytes pass everywhere; the Anthropic adapter refuses inline documents that are not `application/pdf`. Anthropic and Google accept URL-sourced documents too. | Caller; OpenAI adapter refuses URL-sourced documents pre-wire |
| `StopSequences` | Dropped silently | The effective profile gates `Request.StopSequences` off the wire: a caller-pinned override reporting `false` drops the field, the same as an adapter's own table entry reporting `false`. An override reporting `true` above the adapter's ceiling (the field it can actually send) is refused at construction — see [Where the classes are pinned](#where-the-classes-are-pinned). | Effective profile, checked at request build |
| `TopP` | Dropped silently | The effective profile gates `Request.TopP` off the wire (same gate semantics as `StopSequences`). | Effective profile, checked at request build |
| `TopK` | Dropped silently | The effective profile gates `Request.TopK` off the wire (same gate semantics as `StopSequences`). Example: the Chat Completions API has no `top_k`, so the OpenAI ceiling forbids `TopK: true`. | Effective profile, checked at request build |
| `Seed` | Dropped silently | The effective profile gates `Request.Seed` off the wire (same gate semantics as `StopSequences`). Example: the Messages API has no seed parameter, so the Anthropic ceiling forbids `Seed: true`. | Effective profile, checked at request build |

The field classes above match each field's doc comment in `llmkit.Capabilities`.

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

A wholesale profile enforces every wire-gated field it leaves at its Go zero value: the closure above sends a profile with every wire-gated field false, so `StopSequences`, `TopP`, `TopK`, `Seed`, `ToolChoice`, `Thinking`, and `StructuredOutput` are all dropped or refused, not only the fields the closure names. The advisory and decorator fields (`ContextWindow`, `ParallelToolCalls`, `PromptCaching`, `Images`, `Documents`) are not gated by this rule: the profile's `Images` and `Documents` fields gate no block, so an inline image block reaches the wire on every adapter even when the wholesale profile reports `Images: false`. Block validation is unchanged and does not depend on the profile: the OpenAI adapters refuse URL-sourced document blocks and the Anthropic adapter refuses inline documents that are not `application/pdf`. Flip each field you need explicitly, or start from the table-derived input the closure receives instead of ignoring it.

Each provider type has a ceiling: the wire-gated fields (`StructuredOutput`, `Thinking`, `ToolChoice`, `StopSequences`, `TopP`, `TopK`, `Seed`) it can actually put on the wire. `provider.New` refuses an override that reports `true` for a field above the ceiling, with an error wrapping `llmkit.ErrInvalidRequest` naming the field and the Type — before any network call. Anthropic's ceiling excludes `Seed`; OpenAI and `openai-compatible`'s ceiling excludes `Thinking` and `TopK`; Google's ceiling has no excluded field.

See [providers](providers.md) for the construction rules. See `provider/example_test.go` (`ExampleSpec_capabilities`) for the runnable version of the first snippet above.

## Where the classes are pinned

`provider/capability_gate_test.go` pins the ToolChoice and Thinking classes where adapters gate on them. `provider/prepare_gate_test.go` pins the `StopSequences`/`TopP`/`TopK`/`Seed` gate, one row per (field, adapter) whose ceiling allows the field, plus the wholesale-override case. `provider/prepare_ceiling_test.go` pins the ceiling refusal.

A profile with `ToolChoice=false` rejects each explicit mode pre-wire on every adapter. Anthropic rejects through the `Spec.Capabilities` override. Google rejects through the tool-less `gemini-2.0-flash-lite` table entry.

A profile with `Thinking=false` drops the thinking config from the wire body on the two adapters that gate a wire parameter on it: Anthropic and Google.

A profile with a sampler field false drops that field's wire key even when the caller's request sets it; a profile with the field true carries the caller's value. The wire keys are `top_p`, `seed`, and `stop` on OpenAI and openai-compatible (the Chat Completions API has no `top_k`); `top_p`, `top_k`, `stop_sequences`, and `thinking` on Anthropic (the Messages API has no `seed`); and `generationConfig.topP`, `topK`, `seed`, `stopSequences`, and `thinkingConfig` on Google. `provider.New` refuses `TypeOpenAI`/`TypeOpenAICompatible` `Thinking` or `TopK` overrides and `TypeAnthropic` `Seed` overrides before any network call; `TypeGoogle` has no refused field.

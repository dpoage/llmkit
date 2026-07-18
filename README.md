# llmkit

Shared LLM tooling extracted from `bugbot`, `known`, and `go-research`.

## Packages

- **`llm`** — provider-agnostic client abstraction: one synchronous
  `Complete(ctx, Request) (Response, error)` plus a `Capabilities()` probe.
  Adapters for Anthropic (API key or OAuth bearer), OpenAI, Google Gemini, and
  any OpenAI-compatible endpoint (Ollama, vLLM, Groq, ...). Decorator
  composition: retry (exponential backoff + jitter, Retry-After, per-attempt
  timeout) → usage recorder → tool-call serializer → adapter. Normalized
  errors (`APIError` + sentinel kinds), usage accounting with prompt-cache
  conventions, stop-reason normalization, `<think>`-block stripping.
  Origin: `bugbot/internal/llm`.

- **`agent`** — tool-calling harness over `llm.Client`: `Runner` with
  iteration/token budgets, history compaction, forced finalization,
  max-tokens continuation stitching, JSONL transcripts with offline
  `ReplayClient`, schema-constrained `RunJSON`. Tools implement
  `Tool{Def, Run}`; tool errors feed back to the model, infra failures
  surface via `ToolHealthError`. Origin: `bugbot/internal/agent` (harness
  only; bugbot's concrete tools stay in bugbot).

- **`embed`** — `Embedder` interface with Ollama and OpenAI-compatible HTTP
  backends plus an in-memory caching decorator. Local ONNX inference (hugot)
  intentionally NOT included — it drags the ONNX/GoMLX dependency tree;
  implement `Embedder` in your app if you need it. Origin: `known/embed`.

## Design decisions

- Message model: flat `Message{Role, Content, ToolCalls, ToolCallID, IsError}`
  (bugbot's), not content-block unions (go-research's). See docs in `llm`.
- No streaming: none of the three donor projects used it; the interface stays
  synchronous until a consumer needs otherwise.
- Official vendor SDKs (anthropic-sdk-go, openai-go, google genai) rather than
  hand-rolled wire types.

## License

AGPL-3.0 (matching bugbot, the primary donor). The `embed` package derives
from MIT-licensed `known`; relicensed here by the copyright holder.

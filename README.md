# llmkit

Shared LLM tooling extracted from `bugbot`, `known`, and `go-research`.

## Packages

- **`llmkit`** (root) — provider-agnostic client abstraction: one synchronous
  `Complete(ctx, Request) (Response, error)` plus a `Capabilities()` probe;
  normalized errors (`APIError` + sentinel kinds), usage accounting with
  prompt-cache conventions, stop-reason normalization, `<think>`-block
  stripping, and decorator wrappers: retry (exponential backoff + jitter,
  Retry-After, per-attempt timeout), usage recorder, tool-call serializer.
  Origin: `bugbot/internal/llm`.

- **`llmkit/provider`** — client construction: `New` dispatches on
  `Spec.Type` and decorates the chosen adapter serialize → recorder →
  retry.

- **`llmkit/provider/anthropic`**, **`llmkit/provider/openai`** (first-party
  OpenAI and, via `Compatible: true`, any OpenAI-compatible endpoint —
  Ollama, vLLM, Groq, ...), **`llmkit/provider/google`** — vendor-SDK
  adapters for Anthropic (API key or OAuth bearer), OpenAI, and Gemini.

- **`llmkit/internal/adapter`** — helpers shared by the three adapters
  (status classification, error normalization, schema parsing); internal,
  not public API.

- **`llmkit/agent`** — tool-calling harness over `llmkit.Client`: `Runner` with
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

- Message model: content-block `Message{Role, Content []Block, ...}` — text,
  image, document, and opaque provider thinking blocks (Anthropic thinking
  signatures round-trip verbatim; foreign-provider thinking is dropped).
  `llmkit.TextMessage(role, s)` / `Message.Text()` keep the common
  text-only case one line; `Block.Data` is base64 in JSONL transcripts.
  See the root package docs.
- No streaming: none of the three donor projects used it; the interface stays
  synchronous until a consumer needs otherwise.
- Official vendor SDKs (anthropic-sdk-go, openai-go, google genai) rather than
  hand-rolled wire types.

## License

AGPL-3.0 (matching bugbot, the primary donor). The `embed` package derives
from MIT-licensed `known`; relicensed here by the copyright holder.

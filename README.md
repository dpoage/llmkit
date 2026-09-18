# llmkit

Shared LLM tooling extracted from `bugbot`, `known`, and `go-research`.

## Packages

- **`llmkit`** (root) — provider-agnostic client abstraction: one synchronous
  `Complete(ctx, Request) (Response, error)` plus a `Capabilities()` probe;
  a content-block `Message` model (text, image, document, thinking),
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
  `ReplayClient`, schema derivation from Go types (`SchemaOf`/`Func`,
  feeding `RunJSON`/`RunJSONAs`), multi-turn continuation via the
  `Continue` run option, synchronous lifecycle `Hooks`, per-tool timeouts,
  and optional parallel tool dispatch. Tools implement `Tool{Def, Run}` or
  come from `Func`; tool errors feed back to the model, infra failures
  surface via `ToolHealthError`. Origin: `bugbot/internal/agent`
  (harness only; bugbot's concrete tools stay in bugbot).

- **`llmkit/sandbox`** — isolated execution of untrusted, model-generated
  commands against repo snapshots: one
  `Sandbox.Exec(ctx, Spec) (Result, error)` over the Bubblewrap backend
  (Linux, unprivileged user namespaces), a container CLI backend
  (podman/docker), and a scriptable `Mock` (plus `HostExec`, the documented
  no-isolation attended escape hatch that no kit default or example
  constructs). Workspace materialization with symlink-hardened writes,
  capped output capture, and a shared idle/growth-ceiling watchdog;
  standard library plus golang.org/x/sys (the reflink fast path). Path
  containment for agent tools lives in the
  sibling `fsroot` package. Origin: `bugbot/internal/sandbox`.

- **`llmkit/fsroot`** — tool-anchored path containment for agent tools:
  `NewFSRoot(dir)` + `Resolve(rel)` reject absolute paths, `..` escapes, and
  symlink escapes (checked via the longest existing prefix), returning
  `ErrPathEscape`; `EvalExistingPrefixPath` is the shared symlink-resolution
  helper. Deliberately separate from `sandbox`'s post-exec workspace write
  hardening (different threat model). Standard library only. Origin:
  `bugbot/internal/agenttools/{fsroot,pathutil}.go`.

- **`embed`** — `Embedder` interface with Ollama and OpenAI-compatible HTTP
  backends plus an in-memory caching decorator. Local ONNX inference (hugot)
  intentionally NOT included — it drags the ONNX/GoMLX dependency tree;
  implement `Embedder` in your app if you need it. Origin: `known/embed`.

## Examples

`examples/` contains four runnable programs, one per major surface: a
plain completion with content blocks and capability gating (`basic`), a
tool-calling agent with hooks (`agent`), schema-constrained output via
`RunJSONAs` (`structured`), and a multi-turn chat REPL on `Run(...,
Continue(prev))` (`chat`). All four read
`LLMKIT_PROVIDER`/`LLMKIT_MODEL`/`LLMKIT_API_KEY` (plus optional
`LLMKIT_BASE_URL`) and print a usage message instead of touching the
network when the environment is unset:

```bash
go run ./examples/basic --image photo.jpg
go run ./examples/agent --parallel
go run ./examples/structured
go run ./examples/chat
```

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

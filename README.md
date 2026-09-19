# llmkit

Go building blocks for LLM harnesses: a provider-agnostic client, a
tool-calling agent loop with policy seams, sandboxed command execution, and
embeddings. Extracted from `bugbot`, `known`, and `go-research`.

## Packages

- **`llmkit`** (root) — provider-agnostic client abstraction: one synchronous
  `Complete(ctx, Request) (Response, error)` plus a `Capabilities()` probe;
  a content-block `Message` model (text, image, document, thinking) with
  block constructors (`Text`, `Image`, `ImageURL`, `Document`,
  `DocumentURL`) and message constructors (`UserMessage`, `SystemMessage`,
  `ToolResult`, `ToolError`; `TextMessage` stays), normalized errors
  (`APIError` + sentinel kinds), usage accounting with prompt-cache
  conventions, stop-reason normalization, `<think>`-block stripping, and
  decorator wrappers: retry (exponential backoff + jitter, Retry-After,
  per-attempt timeout), usage recorder, tool-call serializer.
  Origin: `bugbot/internal/llm`.

- **`llmkit/provider`** — client construction: `New` dispatches on
  `Spec.Type` and decorates the chosen adapter serialize → recorder →
  retry. Vendor-SDK adapters live under `provider/internal/` (internal;
  `provider.New` is the sole construction path), and `Spec.Type`
  `openai-compatible` serves any OpenAI-compatible endpoint (Ollama, vLLM,
  Groq, ...).

- **`llmkit/internal/adapter`** — helpers shared by the three adapters
  (status classification, error normalization, schema parsing); internal,
  not public API.

- **`llmkit/agent`** — tool-calling harness over `llmkit.Client`: `Runner`
  with iteration/token budgets, history compaction, forced finalization,
  max-tokens continuation stitching, JSONL transcripts with an offline
  `ReplayClient`, schema derivation from Go types (`SchemaOf`/`Func`,
  feeding `RunJSON`/`RunJSONAs`), multi-turn continuation via the
  `Continue` run option, synchronous lifecycle `Hooks`, per-tool timeouts,
  and optional parallel tool dispatch. Two policy seams shape a run:
  `RequestPolicy` edits each wire request (thinking, sampling, tool choice,
  message preprocessing) and `ToolPolicy` allows, denies, or rewrites each
  model-requested tool call before it runs. The `Attach` run option adds
  image or document blocks to the task turn. `Outcome.FinalText` holds the
  final completion's text; `WithBudgetPool` charges a shared `BudgetPool`;
  tool panics become that call's error result (hook panics propagate).
  Tools implement `Tool{Def, Run}` or come from `Func`; tool errors feed
  back to the model, infra failures surface via `ToolHealthError`. Origin:
  `bugbot/internal/agent` (harness only).

- **`llmkit/sandbox`** — isolated execution of untrusted, model-generated
  commands against repo snapshots: one `Sandbox` interface —
  `Exec(ctx, Spec) (Result, error)` plus `MaterializeWorkspace(repoDir)` —
  over the Bubblewrap backend (Linux, unprivileged user namespaces), a
  container CLI backend (podman/docker), and a scriptable `Mock` (plus
  `HostExec`, the documented no-isolation attended escape hatch that no
  kit default or example constructs). Spec fields are honest per backend:
  `Spec.Network` is a typed `NetworkMode`, and a field a backend cannot
  honor is refused at `Exec` with an `UnsupportedSpecError` naming the
  backend, field, and value — never a silent drop or substitution.
  Backend-only knobs (runtime, default image, CPUs, memory, idle window) are
  backend options configured through ONE `Option` type shared by `NewCLI`
  and `NewBwrap` (`WithRuntime`/`WithImage` are CLI-only, `WithCapPolicy`
  is Bwrap-only); no option takes a bare bool — modes are named types
  (`NetworkMode`, `CapPolicy`) — and `Spec` carries no
  CPUs/MemoryMB/IdleTimeout fields (those are backend options). Workspace
  materialization with symlink-hardened writes, output capped at
  `DefaultMaxOutputBytes` (1 MiB) per stream, and a shared
  idle/growth-ceiling watchdog; standard library plus golang.org/x/sys
  (the reflink fast path). Path containment for agent tools lives in the
  sibling `fsroot` package. Origin: `bugbot/internal/sandbox`.

- **`llmkit/fsroot`** — tool-anchored path containment for agent tools:
  `NewFSRoot(dir)` + `Resolve(rel)` reject absolute paths, `..` escapes, and
  symlink escapes (checked via the longest existing prefix), returning
  `ErrPathEscape`; path resolution is package-internal. Deliberately
  separate from `sandbox`'s post-exec workspace write hardening
  (different threat model). Standard library only. Origin:
  `bugbot/internal/agenttools/{fsroot,pathutil}.go`.

- **`llmkit/embed`** — `Embedder` interface with Ollama and
  OpenAI-compatible HTTP backends (shared `llmkit.RetryConfig` retry whose
  `Retry.RequestTimeout` is the single per-attempt bound — embed defaults
  3 attempts / 60s per attempt — plus batching) and a content-hash LRU
  caching decorator (`CachedEmbedder`). Local ONNX inference (hugot)
  intentionally NOT included — it drags the ONNX/GoMLX dependency tree;
  implement `Embedder` in your app if you need it. Origin: `known/embed`.

## Examples

`examples/` contains four runnable programs, one per major surface: a
plain completion with content blocks and capability gating (`basic`), a
tool-calling agent with hooks (`agent`), schema-constrained output via
`RunJSONAs` (`structured`), and a multi-turn chat REPL on `Run(...,
Continue(prev))` (`chat`). All four read
`LLMKIT_PROVIDER`/`LLMKIT_MODEL`/`LLMKIT_API_KEY` (`LLMKIT_BASE_URL` required for
openai-compatible, optional otherwise) and print a usage message instead of touching the
network when the environment is unset:

```bash
go run ./examples/basic --image path/to/photo.jpg
go run ./examples/agent --parallel
go run ./examples/structured
go run ./examples/chat
```

The examples are executed for real by the live acceptance suite:
`go test -tags live ./examples/...` builds and runs each binary against a
live backend, and a hermetic test asserts each exits 1 with its usage
message when the environment is unset. Adapter, agent-loop, and capability
changes are additionally gated by a live vendor matrix with recorded,
secret-free wire fixtures (`provider/testdata/`, replayed hermetically in
every plain `go test ./...`); the acceptance rule and exact commands live in
[AGENTS.md](AGENTS.md) (Build & Test → Live acceptance suite), and the
nightly `.github/workflows/live.yml` runs the whole suite against the
compat lane.

## Design decisions

- Message model: content-block `Message{Role, Content []Block, ...}` — text,
  image, document, and opaque provider thinking blocks (Anthropic thinking
  signatures round-trip verbatim; foreign-provider thinking is dropped).
  Block/message constructors (`Text`, `Image`, `ImageURL`, `Document`,
  `DocumentURL`, `UserMessage`, `SystemMessage`, `ToolResult`, `ToolError`,
  `TextMessage`) keep the common cases one line; constructors with
  required arguments (media type, data, URL, tool-call ID, at least one
  block) panic when they are missing. `Message.Text()` concatenates a
  message's text blocks.
  `Block.Data` is base64 in JSONL transcripts, and `Block` fields marshal
  snake_case with `omitempty`. See the root package docs.
- Streaming without special cases: `Client` stays synchronous; clients that
  can also stream implement `StreamingClient`, and `llmkit.Stream` gives any
  client a delta stream — native when available, otherwise synthesized from
  one `Complete` (same normalized `Response` either way). The decorators
  compose: retry stops at the first delivered delta, the recorder books the
  final usage, and the tool-call serializer forwards only the first call's
  fragments.
- Official vendor SDKs (anthropic-sdk-go, openai-go, google genai) rather than
  hand-rolled wire types.

## License

AGPL-3.0 (matching bugbot, the primary donor). The `embed` package derives
from MIT-licensed `known`; relicensed here by the copyright holder.

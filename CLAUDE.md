# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

Go module `github.com/dpoage/llmkit` (Go version pinned by `go.mod`). The
full gate — CI (`.github/workflows/ci.yml`) runs exactly this on push/PR:

```bash
go build ./...
go vet ./...
go vet -tags live ./provider/ ./agent/ ./examples/... # the `live` acceptance suite must keep compiling
go vet -tags integration ./embed/  # the `integration` Ollama test must keep compiling
go vet -tags integration ./sandbox/ # the sandbox integration test must keep compiling
go test -race -count=1 ./...
golangci-lint run ./...            # config: .golangci.yml (v2 schema, conservative set)
gofmt -l .                         # must print nothing
```

The tag-gated suites RUN (not just compile) outside CI, when their backends
exist: `go test -tags live ./provider/ ./agent/ ./examples/...` is the live
acceptance suite (next section) and skips itself without credentials;
`go test -tags integration ./embed/` needs a local Ollama (default
localhost:11434); `go test -tags integration ./sandbox/` needs bwrap
(`bubblewrap` on Linux) and/or a container runtime (podman/docker) — each
test auto-skips when its backend is missing, and CI runs the sandbox suite
in the dedicated `sandbox-integration` job. The live suite also has its own
workflow, `.github/workflows/live.yml` (workflow_dispatch, nightly schedule,
and pushes to `master` or `round/**` touching provider/agent/examples/internal
— not pull_request, so a PR editing the workflow cannot read the key), driven
by the repo secret `LLMKIT_LIVE_COMPAT_API_KEY` and the repo variables
`LLMKIT_LIVE_COMPAT_BASE_URL`, `LLMKIT_LIVE_COMPAT_MODEL`. Without the key
the job prints `no live credentials — skipped` and exits 0.

`examples/` are runnable contract checks (also compiled by
`go build ./...`): `go run ./examples/basic`, `go run ./examples/agent`,
`go run ./examples/structured`, `go run ./examples/chat`. All four no-op with
a usage message and
exit 1 unless `LLMKIT_PROVIDER`, `LLMKIT_MODEL`, and `LLMKIT_API_KEY` are
set (`LLMKIT_BASE_URL` required for openai-compatible, optional otherwise), so they never touch the network by
accident.

### Live acceptance suite

Rule: any change to an adapter, the agent loop, or a `llmkit.Capabilities`
field must name (a) its hermetic test and (b) its live case in bead
llmkit-3kl.1's registry — `provider/live_registry_test.go` reflects over
`Capabilities` and fails the plain `go test ./...` suite when any field has
no case. Oracles in development rounds run the live suite before APPROVE.

Lanes — each auto-skips, naming its lane and exact missing variables, when
its key is absent:

```bash
LLMKIT_LIVE_COMPAT_API_KEY / _BASE_URL / _MODEL   # all three required (openai-compatible)
LLMKIT_LIVE_ANTHROPIC_API_KEY [+_MODEL]           # default claude-haiku-4-5
LLMKIT_LIVE_OPENAI_API_KEY     [+_MODEL]          # default gpt-4o-mini
LLMKIT_LIVE_GOOGLE_API_KEY     [+_MODEL]          # default gemini-2.5-flash-lite
```

The only credential that exists today is MiniMax M3 (openai-compatible).
Exact local run — costs real money; prompts stay tiny and a full run is well
under $0.10:

```bash
set -a; . ~/.config/bugbot/env; set +a
export LLMKIT_LIVE_COMPAT_API_KEY="$MINIMAX_API_KEY"
export LLMKIT_LIVE_COMPAT_BASE_URL=https://api.minimax.io/v1
export LLMKIT_LIVE_COMPAT_MODEL=MiniMax-M3
export LLMKIT_LIVE_COMPAT_CAPS=parallel_tool_calls,prompt_caching
go test -tags live -count=1 ./provider/ ./agent/ ./examples/... -v
```

Add `-update` to the same command to re-record the compat fixtures
(`provider/testdata/compat/*.json`) — a deliberate local operation, since
every recording differs (model ids, sample text, token counts). The
committed fixtures are proven secret-free (the writer refuses to store any
header carrying the key or any `sk-` substring) and are two-sided under
replay: single-exchange fixtures declare `request_check: "strict"` and
assert the adapter's outgoing request (method, path, body as parsed JSON)
AND the normalized response; multi-turn fixtures declare
`"response_only"` (later requests echo model-generated ids) and compare
response normalization only. `provider/fixture_replay_test.go` replays them
hermetically in every plain `go test ./...`. The nightly `Live` workflow
(`.github/workflows/live.yml`) runs the suite WITHOUT `-update`: the live
matrix's assertions against real responses are the vendor-drift gate.

The `LIVE_TOKENS` summary line covers each test binary's own calls; the
examples package's line prints `note=child-process spend not tallied`
because the example binaries it launches spend separately.

`LLMKIT_LIVE_COMPAT_CAPS` is a comma list of `Capabilities` field names
(snake_case) the operator asserts the compat endpoint supports; listed caps
are forced true via `Spec.Capabilities`, so the gated cases run and MUST
pass instead of skipping. The set exported above — and by the nightly `Live`
workflow's job env — is **MiniMax-M3-verified, not universal**; a cap you
assert must pass, so leave out any that fail on your endpoint.

Probed 2026-09-18 against MiniMax-M3: `parallel_tool_calls` (two tool calls
in one response) and `prompt_caching` (`CacheReadInputTokens > 0` on a
repeated prefix) pass; `top_k` passes trivially (the Chat Completions
adapter never serializes top_k, so the request is accepted);
`structured_output` FAILS (`response_format` json_schema is accepted but
ignored — the model answers in prose) and `thinking` FAILS (M3 emits inline
`<think>` text; the adapter never produces `BlockThinking`).
`stop_sequences` is claimed by the compat profile, but MiniMax-M3 ignores
the `stop` parameter at the raw wire, so that case skips with the evidence
in its message.

## Architecture Overview

- **`llmkit`** (root) — the normalized vocabulary: `Message`/`Block` content
  model with block constructors (`Text`, `Image`, `ImageURL`, `Document`,
  `DocumentURL`) and message constructors (`UserMessage`, `SystemMessage`,
  `ToolResult`, `ToolError`; `TextMessage` stays),
  `Request`/`Response`/`Usage`/`Capabilities`, sentinel errors +
  `APIError`, decorator wrappers (`WithRetry`, `WithRecorder`,
  `WithSerializedToolCalls`), streaming (`Delta`, `StreamingClient`,
  `Stream` over any `Client`), `StripThinkBlocks`, `DefaultMaxTokens`.
- **`llmkit/provider`** — the single construction entry point: `Spec` +
  `Options` → `New` dispatches to an adapter and decorates it
  serialize → recorder → retry. Vendor-SDK adapters live under
  `provider/internal/{anthropic,openai,google}` — internal on purpose,
  with `provider.New` the sole construction path; `Spec.Type`
  `openai-compatible` serves any OpenAI-compatible endpoint (Ollama, vLLM,
  Groq, ...). `Spec.Capabilities`, when set, receives the adapter's
  model-table profile and returns the effective one — flip a single field
  or replace it wholesale (e.g. to pin `ContextWindow` for a model the
  table doesn't know).
- **`llmkit/internal/adapter`** — helpers shared by the three adapters
  (status classification, error normalization, schema parsing); internal,
  not public API.
- **`llmkit/agent`** — the tool-calling harness: `Runner` loop with
  iteration/token budgets, history compaction, forced finalization,
  max-tokens continuation stitching, JSONL transcripts + offline
  `ReplayClient`, schema derivation from Go types (`SchemaOf`/`Func`,
  feeding `RunJSON`/`RunJSONAs`), multi-turn continuation via the
  `Continue` run option, and the synchronous `Hooks` observer surface.
  Two policy seams sit beside the observer hooks. Each is a single-method
  interface with a `Func` adapter, so a policy written for one harness
  works in another. `RequestPolicy` (`WithRequestPolicy`) edits the wire
  request before every completion — main turn, continuation, finalization,
  repair — and before `BeforeCompletion` fires; `req.Messages` is a
  per-turn shallow clone of the history, and the transcript records the
  post-policy request. `ToolPolicy` (`WithToolPolicy`) allows, denies, or
  rewrites the arguments of each model-requested tool call. The Runner
  authorizes every call of a turn in model order on the loop goroutine
  before any `Tool.Run` starts, in both dispatch modes. A denial sends the
  model `ERROR: tool <name> denied: …` with IsError; the hooks stay silent
  and the run continues. Only Arguments may be rewritten; the history keeps
  the model's original. To rewrite results, wrap the `Tool`. The `Attach`
  run option adds image or document blocks to the task turn after
  `Text(task)` (blocks only when the task is empty). Attachments never
  appear on nudges, finalization, or repair turns; the adapters validate
  block kinds.
  `Outcome.FinalText` holds the final completion's text (empty when that
  completion produced none). `WithBudgetPool` makes the Runner check a
  shared `BudgetPool` before every model call and charge it after every
  successful completion. Tool panics are recovered and rendered as that
  call's error result in both dispatch modes (sequential and
  `WithParallelTools`); hook panics propagate to the caller. The `RunJSON`
  repair turn continues the parent run's transcript step numbering.
  `Hooks.Delta`, when set, streams every completion through `llmkit.Stream` —
  native streaming when the client implements `StreamingClient`, deltas
  synthesized from the finished `Response` otherwise — leaving the
  transcript, history, and usage paths unchanged.
  Tool-failure typing: `ToolHealthError` for infra failures,
  `StopReasonError` for model refusal/safety stops.
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
  sibling `fsroot` package.
- **`llmkit/fsroot`** — tool-anchored path containment for agent tools:
  `NewFSRoot(dir)` + `Resolve(rel)` reject absolute paths, `..` escapes,
  and symlink escapes (longest-existing-prefix check), returning
  `ErrPathEscape`; path resolution is package-internal. Deliberately
  separate from `sandbox`'s post-exec workspace write hardening
  (different threat model).
- **`llmkit/embed`** — `Embedder` interface with Ollama and
  OpenAI-compatible HTTP backends (shared `llmkit.RetryConfig` retry whose
  `Retry.RequestTimeout` is the single per-attempt bound — embed defaults
  3 attempts / 60s per attempt — plus batching) and the content-hash LRU
  `CachedEmbedder` decorator.
- **`examples/`** — one runnable program per major surface: `basic` (single
  completion + blocks/capabilities), `agent` (agent loop + hooks),
  `structured` (`RunJSONAs` schema-constrained output), `chat` (multi-turn
  REPL continued via the `Continue` run option).

## Conventions & Patterns

- **Error normalization**: adapters construct `&llmkit.APIError{Kind,
  StatusCode, RetryAfter, Provider, Message, Err}` literals — `Kind` is
  one of the sentinel kinds `ErrRateLimited`, `ErrAuth`,
  `ErrContextTooLong`, `ErrInvalidRequest`, `ErrServer`, `ErrOverloaded`;
  match with `errors.Is`. `APIError` preserves the HTTP status and any
  `Retry-After` hint, and unwraps to both the sentinel and the underlying
  vendor-SDK error (`Err`).
- **Usage convention**: `Usage.InputTokens` is the TOTAL prompt size — it
  INCLUDES cache-read and cache-creation tokens; the cache fields are
  informational subsets of it (the Anthropic adapter sums them in). Budget
  math that wants cache reads discounted uses
  `Usage.ChargeableTokens(weight)`.
- **Streaming**: `Client` remains one synchronous `Complete(ctx, Request)
  (Response, error)` plus `Capabilities()`. A client that can also stream
  implements `StreamingClient`; `llmkit.Stream(ctx, c, req, fn)` works on
  any client — delegating to `Stream` when available, otherwise
  synthesizing deltas (text/thinking in block order, then tool calls) from
  `Complete` — so callers never special-case a non-streaming backend.
  Decorators compose over both paths: `WithRetry` stops retrying once a
  delta is delivered, `WithRecorder` records the final response's usage,
  and `WithSerializedToolCalls` forwards only Index-0 tool-call deltas.
- **Per-role block rule**: user messages carry text/image/document blocks;
  assistant messages text/thinking (plus `ToolCalls`); system and
  tool-result messages text only. Every adapter enforces this BEFORE any
  wire call — violations return an error wrapping `ErrInvalidRequest`.
- **MaxTokens**: a zero or negative `Request.MaxTokens` resolves to
  `llmkit.DefaultMaxTokens` (4096) on every adapter; explicit values pass
  through verbatim.
- **Capabilities**: adapters report `ContextWindow 0` for models outside
  their per-model table — never a fabricated number. Every field's doc
  comment in `llmkit.go` names one of four enforcement classes, and the
  classes cover all fields: DROPPED SILENTLY when the profile reports
  false (`Thinking`, `StructuredOutput`, `StopSequences`, `TopP`, `TopK`,
  `Seed` — the adapter omits the feature from the wire, by adapter
  mapping; for `StopSequences`/`TopP`/`TopK`/`Seed` this reports only the
  adapter's own mapping, not a gate a caller-pinned profile can use to
  disable a supported feature), REFUSED PRE-WIRE (`ToolChoice` — an
  explicit non-auto mode against `ToolChoice=false` returns an error
  wrapping `ErrInvalidRequest` before any wire call, because silently
  dropping `none` would let the model call forbidden tools), DECORATOR
  (`ParallelToolCalls` — `provider.New` installs the tool-call serializer
  when it is false), ADVISORY (`Images`, `Documents`, `PromptCaching`,
  `ContextWindow` — no adapter reads them; gate your own image/document
  input on the capability, see `examples/basic`).
- **Transcript format**: `Block` fields marshal snake_case with
  `omitempty` on every zero field (`Kind` is always present), so a text
  block serializes as exactly `{"kind":"text","text":"…"}` and a nil `Raw`
  never emits `"raw":null`. Transcripts recorded before these tags existed
  are not supported: their image and document blocks decode with an empty
  MediaType (the old key was the Go field name) and are refused pre-wire.
  Re-record them.
- **Hooks are synchronous**: every `agent.Hooks` callback runs inline on
  the goroutine that reaches the fire point — a slow hook stalls the run.
  `ToolEvent.Step`, `CompactionEvent.Step`, and the transcript's
  `Event.Step` carry the SAME 1-based number for a turn — the `RunJSON`
  repair turn continues the parent run's sequence instead of restarting —
  so consumers can join on Step.
- **Tool concurrency**: `Tool.Run` may be invoked concurrently — within one
  run under `WithParallelTools`, and across concurrent `Runner.Run` calls
  (a Runner is safe for concurrent use) — so tools and hook functions must
  be safe for concurrent use.

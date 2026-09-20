# Changelog

All notable, user-visible changes to llmkit are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The project is pre-1.0, so minor versions may carry breaking changes; each
entry below is marked.

## [Unreleased]

### Added

- `llmkit/retry`: a leaf package (standard library only) holding the shared
  retry loop: `retry.Config`, `retry.Default`, `retry.Do`, and
  `retry.ParseRetryAfter`. `Do` runs any operation — not just an
  `llmkit.Client` — under the same policy as `WithRetry`: exponential
  backoff with jitter, server `Retry-After` precedence capped at `MaxDelay`,
  and a per-attempt `RequestTimeout` deadline. A caller-supplied classifier
  decides which errors are transient and carries the header's delay and
  presence; presence with a zero delay means an immediate retry, absence
  means the exponential schedule. `Do` clamps `MaxAttempts`,
  `RequestTimeout`, and `Jitter` to usable values. `Config.Sleep` and
  `Config.Rand` are exported hooks — nil means a real timer and the
  package-level random source — so tests can pin the schedule
  deterministically.
- `llmkit`: the observability seam. `Observer` receives a typed `Event` per
  nondeterministic boundary — one `Observe(ctx, Event)` method, with
  `ObserverFunc` as the closure adapter and `Observers` as an
  order-preserving fan-out that skips nil Observer interface values. `Event`
  is one Kind-discriminated struct with snake_case tags and a
  `schema_version` on every encoding; the kinds are start, completion,
  attempt, tool_run (policy denials ride its `Denied` field), compaction,
  steer, finalize, decision, embed, and exec, each carried by a payload
  type: `StartEvent`, `CompletionEvent`, `AttemptEvent`, `ToolRunEvent`,
  `CompactionEvent`, `SteerEvent`, `FinalizeEvent`, `DecisionEvent`,
  `EmbedEvent`, and `ExecEvent`. Run identity travels the context: `RunID`,
  `WithRun`, `RunFromContext`, and `NewRunID`, whose ids sort lexically in
  mint order.
  Spans join attempts to their completion: the Completion emitter mints a
  `SpanID` per logical completion (`WithSpan`, `SpanFromContext`,
  `NewSpanID`) and the retry stage's Attempt events inherit it.
  A Completion event is emitted once per logical completion by the outermost
  layer; Attempt events come only from the provider retry stage, and replay
  consumes Completion only. `Recorder` is unchanged; folding it into the
  stream is deferred.
- `llmkit/embed` and `llmkit/sandbox`: `Observe` decorators at the last two
  unobserved nondeterministic boundaries. `embed.Observe(e, obs)` emits one
  Embed event per `Embed`/`EmbedBatch` call — model, requested input count,
  vector dimensions, duration, and the error; `CacheHits` and `Usage` stay
  zero because the `Embedder` interface exposes neither per-call cache
  attribution nor token usage. `sandbox.Observe(s, obs)` emits one Exec
  event per `Exec` — backend name, command, exit code (`-1` when the
  process never ran to an exit), captured byte counts per stream,
  truncation, the run's measured duration, and the infrastructure error; a
  non-zero exit code is the command's verdict, not an error, exactly like
  the `Sandbox` contract.
  Both wrappers pass results and errors through
  unchanged and take `RunID`/`SpanID` from the call's context; the sandbox
  wrapper emits nothing from `MaterializeWorkspace`. Either wrapper
  returns its input unchanged for a nil observer.

- `llmkit`: `Observe(c, obs, provider, model)` wraps any `Client` so each
  logical completion — one `Complete` or `Stream` call, success or error —
  emits exactly one `Completion` event: the request as received, the final
  response (on the stream path assembled through the same synthesis
  `llmkit.Stream` performs) or the error text, tagged with the provider and
  model arguments. It mints a fresh span per call and stamps it into the
  client's context, so provider `Attempt` events join it; a nil observer
  returns the client unchanged.
- `llmkit`: `WithRetryObserver(c, cfg, obs, provider, model)` — the retry
  stage emitting one `Attempt` event per attempt, failures included,
  numbered 1..N by the loop, span inherited from the context, duration
  covering just that attempt. `WithRetry` keeps its signature and emits
  nothing.
- `provider`: `Options.Observer` plumbs an `llmkit.Observer` into `New`'s
  retry stage; `New` emits Attempt events only and never `Completion`
  events — wrap its result with `llmkit.Observe` (or run it under the agent
  Runner, which emits its own) to capture completions, and never both for
  the same client.

### Changed

- **Breaking:** the retry vocabulary moved from the root package into
  `llmkit/retry`: `llmkit.RetryConfig` is now `retry.Config`,
  `llmkit.DefaultRetryConfig` is `retry.Default`, and
  `llmkit.DefaultRequestTimeout` is `retry.DefaultRequestTimeout`.
  `WithRetry` takes a `retry.Config`, and `provider.Options.Retry`,
  `embed.Config.Retry`, and `decide.Config.Retry` are `retry.Config`.
  There are no compatibility aliases.
- **Breaking:** the root wire types now carry snake_case `json` tags:
  `Message`, `ToolDef`, `ToolCall`, `ThinkingConfig`, `ToolChoice`,
  `Request`, `Response`, and `Usage` previously serialized with Go field
  names (`InputTokens`, `ToolCallID`, ...). Transcript JSONL and any JSON
  recorded before this change no longer round-trips — per the standing
  ruling, pre-tag recordings are unsupported and must be re-recorded.
  Unmarshaling is unaffected for single-word keys (encoding/json matches
  case-insensitively), but multi-word keys such as `InputTokens` or
  `MaxTokens` silently drop.
- `internal/retry` is deleted. Its loop and header parser now live in the
  `llmkit/retry` leaf package; `embed`, `decide`, and `internal/adapter`
  call them there. Internal package; no caller-facing change; retry
  behavior is unchanged except that every in-tree loop now applies
  `internal/retry`'s stricter negative-`Retry-After` and overflow clamps,
  which no in-tree config could reach.

## [0.5.0] - 2026-09-20

### Added

- `llmkit/decide`: a client for TypeSafe's Jev decision model on the System
  One API. `decide.New` and `Client.Ask` accept sealed `Noul`, `Choice`, and
  `Score` questions and return normalized answers with probabilities and
  confidence. Errors are `*llmkit.APIError`; an optional `llmkit.Recorder`
  receives usage. `decide` populates `llmkit.APIError.RetryAfter` on every
  status that carries the header; the root `APIError` doc comment still
  describes 429 and 529 only (bead llmkit-gac tracks the Go doc fix).
- `internal/retry`: the backoff, `Retry-After`, and per-attempt-timeout loop
  extracted from `embed`, so `embed` and `decide` share one implementation.
  Internal package; no caller-facing change; `embed` behavior is unchanged.

### Fixed

- `llmkit.Stream` now emits one `DeltaText` from `Response.Text` when
  `Response.Blocks` carries no text block; previously a Complete-only client
  returning a text-only response — including `agent.ReplayClient` replaying a
  recorded run — fired `Hooks.Delta` zero times for the reply text.
- `agent.RunJSON`: errors wrapping `ErrUnparseableOutput` no longer carry a
  doubled `agent: agent:` prefix.
- Google adapter: an assistant thinking block whose `Raw` is `null` or `{}`
  padded with JSON whitespace is skipped on replay instead of being sent as
  an empty part.
- Anthropic adapter: a thinking block whose `Raw` carries nothing replayable —
  missing, `null` or whitespace-only (with or without JSON-whitespace
  padding), or a JSON payload that decodes to an empty thinking/redacted
  block (`{}`, `{"type":"thinking"}`) — now fails locally with
  `ErrInvalidRequest` before any wire call; previously such a block slipped
  past the guard and emitted an empty unsigned thinking block the API
  rejects remotely.
- Anthropic adapter: structured-output finalize appends the surfaced
  tool-call arguments as a `BlockText`; previously such completions returned
  `Response.Text` with no text block (`Blocks` empty, or only a thinking
  block), violating the Response
  invariant (Text equals the concatenation of `BlockText` blocks) for every
  consumer that reads `Blocks` — transcripts, replay, and the agent's
  assistant history. The agent additionally appends surfaced text to
  history when a client's `Blocks` omit any text block, so history can no
  longer diverge from what `llmkit.Stream` delivered.
- Anthropic adapter: a request combining `Thinking` with forced tool use now
  fails locally with `ErrInvalidRequest` before any wire call — whether the
  force comes from the synthetic structured-output tool (`ResponseSchema`) or
  from an explicit `ToolChoice` of `required` or a named `tool`; previously
  such requests were sent with manual extended thinking plus a forced
  `tool_choice`, a combination Anthropic rejects with a 400 (forced tool use
  only supports `tool_choice` auto or none under manual extended thinking).
- Anthropic adapter: a replayed thinking block whose `Raw` decodes to
  thinking text without a `signature` now fails locally with
  `ErrInvalidRequest`; previously the unsigned block was forwarded even
  though the API verifies thinking signatures on replay.

## [0.4.0] - 2026-09-19

### Added

- Streaming: `llmkit.Delta`, `StreamingClient`, and `llmkit.Stream` over any
  client, with native streams on the Anthropic, OpenAI, and Google adapters
  and decorator support across all three.
- Streaming in the agent: `agent.Hooks.Delta` streams every completion while
  transcripts, history, and usage stay unchanged.
- Mid-run steering: `agent.NewSteering` and `WithSteering` queue user turns
  mid-run (`Steer`) or at the would-be finish (`FollowUp`); `Continue` with
  the same handle delivers leftovers on the next run.
- Policy seams: `agent.RequestPolicy` edits each outgoing wire request, and
  `agent.ToolPolicy` allows, denies, or rewrites each model-requested tool
  call; the `Attach` run option adds image or document blocks to the task
  turn.
- `agent.WithBudgetPool`: one token budget shared across concurrent Runner
  runs.
- Live acceptance regime: `live`-tagged vendor lanes, the capability-keyed
  case registry (`provider/live_registry_test.go`), recorded secret-free
  wire fixtures replayed hermetically by the plain suite, and the nightly
  `Live` workflow.
- Documentation: sectioned package docs and hermetic `Example` functions in
  every package; `docs/capabilities.md`, `docs/providers.md`,
  `docs/agent-loop.md`, `docs/sandbox.md`, `docs/testing.md`; rewritten
  README, `docs/README.md`, `docs/design.md`, and this changelog.

### Changed

- **Breaking:** the vendor adapters moved under
  `provider/internal/{anthropic,openai,google}`; `provider.New` is now the
  only construction path. `llmkit.APIError.Err` is exported.
- **Breaking:** sandbox: honest `Spec` — typed `NetworkMode`, per-backend
  refusals with `UnsupportedSpecError`, one `Option` type for both container
  backends, and `MaterializeWorkspace` on the `Sandbox` interface. Compile
  breaks: `NewCLI` takes options only (runtime and image moved to
  `WithRuntime`/`WithImage`); `NewBwrap` takes `...Option`; `BwrapOption` and
  the `WithBwrap*` family are deleted; `WithNetwork` takes a `NetworkMode`;
  `Spec` lost `CPUs`, `MemoryMB`, and `IdleTimeout` (now backend options).
- **Breaking:** embed: `Config.Retry` is now `llmkit.RetryConfig`; unset
  retry knobs resolve to 3 attempts with a 60 s per-attempt bound; the cache
  copies vectors on read and exposes `Stats`/`Len`; `ErrEmptyVector` replaces
  the silent zero vector. Compile breaks: `Stats.Size` and `Config.Timeout`
  are removed; `Config.Retry.Jitter` is literal — 0 means no jitter and
  out-of-range values are rejected — instead of clamped. The package's own
  `RetryConfig`, `DefaultRetryConfig`, and `DefaultEmbedTimeout` are deleted;
  use `llmkit.RetryConfig`.
- **Breaking:** agent: a tool panic becomes that call's `ERROR:`-prefixed
  result in both dispatch modes; hook panics still propagate;
  `Outcome.FinalTextSet` is removed (non-empty `FinalText` is the signal);
  `Limits.BudgetCheck` is removed (the `WithBudgetPool` option replaces it).
- **Breaking:** `llmkit.NewAPIError` is removed; construct `APIError`
  literals directly. `fsroot.EvalExistingPrefixPath` is unexported.

### Fixed

- Doc-only: `APIError` field docs no longer claim the API key is never
  present; llmkit never inserts the credential, and vendor error bodies pass
  through as the provider sent them (bead llmkit-ygq.11). No API change.

## [0.3.0] - 2026-09-17

### Added

- `llmkit/sandbox`: isolated execution of untrusted commands against repo
  snapshots, with Bubblewrap and container-CLI backends, `HostExec`, and a
  scriptable `Mock`; capability probes and symlink-hardened workspaces.
- `llmkit/fsroot`: path containment for agent file tools; rejects absolute
  paths, `..` escapes, and symlink escapes.
- agent: `SchemaOf`, `Func`, `RunJSON`, and `RunJSONAs` for schema-derived
  tools and structured output; the `Continue` run option for multi-turn
  continuation; typed `TruncationReason`; `StopReasonError` for model
  refusal and safety stops.

### Changed

- **Breaking:** `provider.New` takes `(ctx, Spec, Options)`; `Spec.Auth` is
  a typed mode with `AuthOAuthToken` refused on non-Anthropic types;
  `provider.ParseType` added; one `Spec.Capabilities` override replaces the
  wholesale pin; the usage `Recorder` lost its `Role` field. In the adapter
  packages, `Options.Capabilities` was retyped from `*llmkit.Capabilities`
  to `func(llmkit.Capabilities) llmkit.Capabilities`, and
  `Options.StructuredOutput` was removed.
  `agent.ErrStopReason` is deleted (`StopReasonError` replaces it);
  `Outcome.Truncated` is removed (`TruncationReason` covers it);
  `provider.Spec.StructuredOutput` is removed (use the `Spec.Capabilities`
  override).

## [0.2.0] - 2026-09-17

### Added

- Model-aware capability tables per provider, with `ContextWindow 0` for
  unknown models and a `Spec.Capabilities` override.
- agent: the synchronous `Hooks` observer surface, `WithToolTimeout`, and
  opt-in `WithParallelTools`; a `Runner` is safe for concurrent `Run`.
- `llmkit.DefaultMaxTokens` (4096) applied by every adapter when
  `MaxTokens` is zero.
- embed: per-request timeout, retry on 429/5xx/timeouts, `MaxBatch` chunking,
  bounded LRU cache with `Stats`, and response validation.
- GitHub Actions CI: build, vet including tag-gated suites, `go test -race`,
  golangci-lint, gofmt.

### Changed

- **Breaking:** `Message.Content` became an ordered `[]Block` list (text,
  image, document, thinking), with `TextMessage`/`Message.Text()` for the
  common case; the per-role block rule is enforced before any wire call.
- **Breaking:** `Request` gained `Thinking`, `ToolChoice`, `StopSequences`,
  `TopP`, `TopK`, and `Seed`; `StopRefusal` and `StopContentFilter` split out
  of `StopError`.
- **Breaking:** `ToolChoice=false` refuses explicit non-auto modes pre-wire.
- **Breaking:** agent bugbot residue removed (`ToolActivity`, health sinks,
  `Severity`); embed `LoadConfig` returns an error and reads the `LLMKIT_`
  environment prefix.
- **Breaking:** `embed.NewCachedEmbedder` takes `(inner, maxSize)`; the
  cache became bounded.

## [0.1.0] - 2026-09-17

### Added

- Initial public release: the `llmkit` root vocabulary, the
  `provider/{anthropic,openai,google}` adapters, the `agent` tool-calling
  harness with JSONL transcripts and offline replay, and the `embed` package.
- Relicensed AGPL-3.0; the `embed` package derives from MIT-licensed `known`
  and is relicensed by the copyright holder.

[Unreleased]: https://github.com/dpoage/llmkit/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/dpoage/llmkit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dpoage/llmkit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/dpoage/llmkit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dpoage/llmkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dpoage/llmkit/releases/tag/v0.1.0

# Changelog

All notable, user-visible changes to llmkit are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The project is pre-1.0, so minor versions may carry breaking changes; each
entry below is marked.

## [Unreleased]

### Added

- `llmkit/decide`: a client for TypeSafe's Jev decision model (the System One
  API): `decide.New` and `Client.Ask` with the sealed `Noul`, `Choice`, and
  `Score` question types, normalized answers with probabilities and
  confidence, `*llmkit.APIError` error normalization, and a usage recorder.
  `decide` populates `llmkit.APIError.RetryAfter` on any status carrying the
  header; the root `APIError` doc comment still describes 429/529 only (bead
  llmkit-gac tracks the Go doc fix).
- `internal/retry`: the backoff, `Retry-After`, and per-attempt-timeout loop
  extracted from `embed`, so `embed` and `decide` share one implementation.
  Internal package; no caller-facing change; `embed` behavior is unchanged.

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

[Unreleased]: https://github.com/dpoage/llmkit/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/dpoage/llmkit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/dpoage/llmkit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dpoage/llmkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dpoage/llmkit/releases/tag/v0.1.0

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
  unchanged and take `RunID`/`SpanID`/`Step` from the call's context; the
  sandbox wrapper emits nothing from `MaterializeWorkspace`. Either wrapper
  returns its input unchanged for a nil observer.
- `llmkit/sandbox`: `InvalidSpecError`. Every backend — the Mock included —
  now refuses a malformed `Spec` at `Exec` with `InvalidSpecError{Field,
  Reason}` (match with `errors.As`), instead of each backend enforcing a
  different subset: an empty `Cmd`; neither `RepoDir` nor `Workspace`; a
  relative `Workspace`; a `WriteFiles` key or `CaptureFiles` entry that
  escapes the workspace (HostExec previously dropped an escaping
  `CaptureFiles` entry silently instead of refusing it); an empty or
  relative mount path; a duplicate `ContainerPath` across `ROMounts` and
  `RWMounts`; and an `Env` entry without `=` or with an empty key (see
  Fixed for the host-environment leak the first closes). The three real
  backends also refuse, with `Field: "Workspace"`, a `Workspace` that does
  not exist, is not a directory, or cannot be entered. A well-formed field
  a single backend cannot keep honoring is still refused with the
  unchanged `UnsupportedSpecError`.

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
- `llmkit/agent`: run identity and durable observation on the tool loop.
  `WithObserver(obs)` installs the Runner's single durable event sink
  (last-wins — a second call replaces the first, never a second history)
  behind the always-present in-memory `Transcript`, which now stores
  `llmkit.Event` values in `Transcript.Record` and carries
  `RunID`/`ParentRunID`. `JSONL(dir, onErr)` streams one JSON line per
  event to one `<RunID>.jsonl` file per run — created exclusively at the
  run's start, closed at its finalize, refusals reported through `onErr`
  and never failing the run — and reads them back through the same `Source`
  interface replay builds on. The Runner emits `start`, `completion` (one
  per logical completion, span-minted per C2), `tool_run` (with
  `Denied`/`deny_reason` for policy denials), `compaction`, `steer`, and
  `finalize` (on every run end including error returns). `WithRunID(id)`
  pins a run's identity; `Outcome.RunID` is exported; `Continue` chains
  carry `ParentRunID`.
- `llmkit/agent`: the read side of recording. `Source` is a single-method
  interface (`Events(ctx, run)`) implemented by `Transcript` and the JSONL
  sink; `NewReplayClient(src, run, caps)` replays a recorded run from any
  Source, `ReplayClient.Tools` serves the recorded tool results instead of
  executing them (a fully offline replay, deterministic under parallel
  dispatch), and `ReplayClient.Err` reports a diverged replay. The
  sentinels `ErrUnknownRun` (a Source has no record of the run) and
  `ErrReplayDiverged` (a replay no longer matches its record) support
  errors.Is.
- `llmkit`: `WithStep(ctx, step)` and `StepFromContext(ctx)` — the 1-based
  turn number in the context, mirroring `WithRun`'s empty-id rule (step <= 0
  is absent). `NewEvent` stamps `Event.Step` from it, so decorator-emitted
  events inside a Runner turn — the retry stage's Attempt events, a decision
  observed inside a ToolPolicy, the sandbox Exec and embed events a tool's
  decorators emit — carry the enclosing turn; Runner-emitted events keep
  setting Step explicitly. The agent Runner places the turn in the contexts
  it passes to the client, policies, hooks, and tools.
- `llmkit`: `FinalizeEvent.FinalText` (`final_text`, omitted when empty) —
  the run's answer as the Runner stitched it across a max-tokens
  continuation, so a store persists it without re-deriving the stitch.
  Additive; no schema version bump.
- `llmkit`: `Event.Validate()` — a shape check for sinks, stores, and
  tests: the Kind must be a declared constant, exactly one payload pointer
  must be non-nil and be the one the Kind names, and `SchemaVersion` must
  be non-zero (any non-zero value passes — a newer schema is the sink's
  business). It never runs on the emission path and does not gate
  encoding.
- `provider`: `Tag(spec, opts)` — the provider tag `New` puts on usage and
  Attempt events (`Options.Provider` when set, else `string(spec.Type)`),
  exported and called by `New` itself, so `llmkit.Observe` callers pass
  the same value instead of restating the rule.
- `sandbox.WithHostToolchains(res ToolchainResolution)`: one option for
  host toolchains, accepted by both `NewCLI` and `NewBwrap` (host
  toolchains were bwrap-only before). Resolve once with
  `ResolveHostToolchains(names)` and pass the same value to either
  constructor; the backend renders the read-only mounts and composes
  `PATH`. On `NewCLI`, a resolution with a `PATH` entry replaces the
  image's own `ENV PATH` with the toolchain directories followed by
  `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`; a `PATH=`
  entry in `Spec.Env` overrides it on both backends.
- `sandbox.Observe`'s return value implements `io.Closer` when the
  observer is non-nil: `Close` forwards to the inner backend's `Close`
  when it has one (`CLI`, `Bwrap`) and returns `nil` otherwise (`Mock`,
  `HostExec`). The return type stays `Sandbox`, so a caller that wants to
  `Close` type-asserts.

- `llmkit`: `llmkit.Classify`, the kit's one retryability rule, in
  `retry.Do`'s classify shape: cancellation (anything chaining
  `context.Canceled`, including an `*APIError` wrapping it) is terminal,
  `*APIError` with kind `ErrRateLimited`/`ErrServer`/`ErrOverloaded` is
  retryable honouring the carried `Retry-After`, and everything else is
  terminal. `APIError` gains `HasRetryAfter` — a presence bit;
  `RetryAfter` is meaningful only when it is true, and a present zero
  means retry immediately. **Breaking:** literal constructors that relied
  on `RetryAfter > 0` to mark a carried delay must set `HasRetryAfter`.
- `llmkit/retry`: `retry.Config.Or(base)` completes a partial config
  field-wise — each unset schedule field (`MaxAttempts <= 0`,
  `BaseDelay <= 0`, `MaxDelay <= 0`, `Jitter == 0`, `RequestTimeout <= 0`)
  is taken from `base`; the `Sleep`/`Rand` test hooks pass through
  untouched. `provider.New`, `decide.New`, and `embed.NewEmbedder` each
  run their `Config.Retry.Or(<their defaults>)`, so a partial
  `Config.Retry` keeps its fields instead of being discarded wholesale
  (`provider.New` with `Retry{MaxAttempts: -1}` now resolves to four
  attempts). **Breaking:** under every constructor that calls `Or`,
  `Jitter` 0 counts as unset — the explicit value "no jitter" no longer
  pins; the deterministic escape is the `Config.Rand` hook (a function
  that returns 0.5 produces exactly 1.0 jitter factors).

### Changed

- **Breaking:** `FinalizeEvent` loses `Iterations` — `Event.Step` on the
  finalize event already carries the completed-turn count (the Runner set
  both from the same value on every exit path). Old recordings still
  decode: the `iterations` key is ignored. No compatibility aliases.
  This is a removal at the same schema version (still 1), not a bump —
  safe only because no store exists yet to branch on the field, which is
  why it lands before the store round (llmkit-2or.1) freezes the schema.
- `agent.LoadJSONL` validates every line with `llmkit.Event.Validate`
  instead of checking only `schema_version`: the load now errors — naming
  the line — on any line that is not a valid Event: an undeclared kind, a
  missing payload, a foreign or doubled payload, or a zero
  `schema_version`. A payload-less `start` line loaded before; it fails
  now. The rule is forward-compatible on the version (a future
  `schema_version` still loads) but not on kinds: one unknown kind fails
  the whole load.
- **Breaking:** the agent transcript is the new event stream.
  `agent.Event`/`EventKind` and the `request`/`assistant`/`tool_result`
  kinds are deleted in favor of `llmkit.Event` (`Transcript.Events` is now
  `Transcript.Record`, of `llmkit.Event`); `WithTranscriptDir` and
  `WithTranscriptKey` are removed in favor of
  `WithObserver(agent.JSONL(dir, onErr))` (the key's motivation moved to
  `WithRunID`); `Hooks.TranscriptError` is removed (sink failures go to the
  callback the sink was constructed with); `NewReplayClient` takes `(src
  Source, run llmkit.RunID, caps)`. There are no compatibility aliases. A
  tool-call-only assistant turn no longer invents an empty text block in
  history (llmkit-ly5).
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
- **Breaking:** `sandbox.Mock.ResponseFunc` is now
  `func(ctx context.Context, n int, spec Spec) (Result, error)` — it
  receives the caller's `Exec` context.
- **Breaking:** `sandbox.Mock.Exec` is no longer a law-free zone: it
  refuses a `Spec` that is malformed for every backend with
  `InvalidSpecError` (see Added), and when the caller's context is already
  done it records the call, consumes nothing, and returns the zero
  `Result` with a `sandbox: execution cancelled` error instead of a
  scripted response.
- **Breaking:** `sandbox.HostExec` now requires `Spec.Workspace` to be an
  absolute path (like every backend) and pre-validates `CaptureFiles`
  entries with the same lexical workspace-escape rule the container
  backends apply, refusing an escaping entry at `Exec` instead of omitting
  it silently after the run.
- **Breaking:** `sandbox.Mock.MaterializeWorkspace` returns a FRESH EMPTY
  temporary directory (prefix `llmkit-mock-`) instead of returning
  `repoDir` unchanged — following the documented "os.RemoveAll the return
  value" contract used to delete the caller's repository. `repoDir` is
  accepted but never read and need not exist. `Mock.Materialized()`
  lists every directory created, in call order.
- **Breaking:** `sandbox.HostExec` reports a command killed by a signal as
  `ExitCode` 128+signo (SIGSEGV: 139) with a nil error, like the bwrap and
  CLI backends, instead of `ExitCode: -1`; `-1` stays reserved for the
  sandbox's own kills (timeout, idle watchdog, growth ceiling).
- **Breaking:** `sandbox.NewCLI` and `sandbox.NewBwrap` refuse a numeric
  option value that names no usable limit, at construction:
  `WithCPUs` refuses 0, a negative value, NaN, and ±Inf; `WithMemoryMB`,
  `WithTimeout`, `WithScratchSizeMB`, and `WithMaxOutputBytes` refuse a
  value `<= 0`. Before, some of these values silently weakened the run
  and others failed it outright: `WithCPUs(NaN)` rendered no CPU quota on
  either backend and ran uncapped; `WithCPUs(+Inf)` ran uncapped on the
  CLI backend, but on the bwrap backend every run exited 1 because
  systemd-run rejects `CPUQuota=+Inf%`; `WithTimeout(0)` failed every run
  at once on both backends. The error names the option, its value, and the
  backend. `WithPidsLimit`, `WithWorkspaceGrowthCeilingMB`, and
  `WithIdleTimeout` keep `0` as the documented explicit disable and
  refuse a negative value (llmkit-bk8.1.7).
- **Breaking:** `sandbox.ResolveHostToolchains(names)` returns only a
  `ToolchainResolution`, with no error (it never returned a non-nil one):
  drop the `, err` at each call site. The resolution gains
  `Unresolved []string`, listing every trimmed, non-blank entry that did
  not resolve on this host, in request order.
- **Breaking:** `sandbox.Bwrap.Exec` reports a run whose deadline passes
  before its child starts as `TimedOut` with `ExitCode` -1 and a nil
  error, like the CLI and HostExec backends. Before, it returned the error
  `sandbox: start systemd-run: context deadline exceeded` and a zero
  `Result`. A launcher that cannot start now fails with
  `sandbox: run <launcher>: ...` instead of `sandbox: start <launcher>: ...`,
  and the returned `Result` carries the launch attempt's `Duration`. One
  reachable case: the bwrap binary removed or made non-executable
  after `NewBwrap` on a host with no resource-cap method, under
  `WithCapPolicy(CapBestEffort)` (a missing `systemd-run` selects another
  cap method, not this one).

### Removed

- **Breaking:** `sandbox.WithToolchainBinds` and
  `sandbox.WithToolchainPath`. Replacement:
  `sandbox.WithHostToolchains(sandbox.ResolveHostToolchains(names))` on
  `NewCLI` or `NewBwrap`. Resolve once and pass the same value; it also
  carries `Fingerprints` and `Unresolved`.
- **Breaking:** the `sandbox.ToolchainResolution.Mounts` and
  `ToolchainResolution.PathPrepend` fields (now unexported). Replacement:
  pass the resolution to `WithHostToolchains(res)` on either constructor;
  the backend renders the mounts and composes `PATH`.
- **Breaking:** `sandbox.DefaultContainerPath` (now unexported).
  Replacement: none. The backends compose `PATH` themselves; a `PATH=`
  entry in `Spec.Env` overrides it.
- **Breaking:** `sandbox.CLI.Limits`, `sandbox.Bwrap.Limits`,
  `sandbox.CLI.ScratchAndGrowthCeiling`, and
  `sandbox.Bwrap.ScratchAndGrowthCeiling`. Replacement: none. The
  constructors refuse zero and negative values (and non-finite values
  for `WithCPUs`) — the same values the accessors used to report a
  substitute for — and no fallback is substituted. A test that read
  these accessors to confirm that a configuration reached the backend
  loses that probe.
- **Breaking:** `sandbox.CLI.Runtime`. Replacement: `sandbox.Detect()`
  returns the runtime `NewCLI` auto-detects; a caller that passed
  `WithRuntime(name)` already has the name.
- **Breaking:** `sandbox.Mock.Reset`. Replacement: construct a new `Mock`
  with `NewMock`.

- **Breaking:** `sandbox.ProbeCapabilities` is removed in favor of
  `sandbox.Probe(ctx, sb Sandbox, base Spec, probes []ProbeEntry)
  (CapabilitySet, error)` (llmkit-bk8.1.9): the base `Spec` carries
  `RepoDir`/`Image`/mounts/`Env` instead of positional arguments; a
  refusal (`InvalidSpecError`/`UnsupportedSpecError`) is returned as the
  error instead of being read as "unavailable"; and results are no longer
  cached per process — every call re-probes, and the returned
  `CapabilitySet` shape is unchanged. `base.Image` must now be empty on
  the Bwrap and HostExec backends — it was silently stripped before, so a
  caller that (like bugbot) passes its configured image on every backend
  must clear it for those two. A nil `Sandbox` is now an error instead of
  an all-false result. An empty `base.RepoDir` with no `base.Workspace`
  now returns `InvalidSpecError{Field: "RepoDir"}` instead of an all-false
  set built without any `Exec` call. On HostExec, `base.ROMounts`,
  `base.RWMounts`, and `base.SetupCmds` must also be empty. HostExec
  refuses mounts with `UnsupportedSpecError`; the old mount arguments
  read as all-false there, and the refusal is now returned. A positive
  `base.Timeout` replaces the 30-second per-probe ceiling.
  `sandbox.InvalidateCapabilityCache` is removed with no replacement —
  there is no cache left to invalidate.

### Fixed

- `llmkit/sandbox` (llmkit-bk8.1.8): a command that cannot be launched now
  reports through the shell's exit-code convention on every real backend —
  `127` when the command is missing, `126` when it is not executable —
  instead of an infrastructure error or a bare exit 1. Bwrap always execs
  the command through `/bin/sh` (with no `SetupCmds` the wrapper script is
  exactly `exec "$@"`), so bwrap's own execvp failure (exit 1) no longer
  masks the cause. HostExec maps the launch failure to the exit code and
  returns a nil error. A caller-owned `Workspace` the process cannot enter
  (no search permission) is refused before the run with
  `InvalidSpecError{Field: "Workspace"}` on every real backend, instead of
  reading as a non-executable command (126) on HostExec. HostExec also no
  longer reports a caller cancellation as `Result{ExitCode: -1}` with a nil
  error — a caller whose context is cancelled or past its deadline gets the
  shared `sandbox: execution cancelled` error, while `Spec.Timeout` expiry
  still reports the timeout shape. A context that has already ended when
  `Exec` is called, or that ends during Bwrap's admission (while it
  resolves the resource-cap method), gets that same error on every real
  backend before anything is written or launched; Bwrap used to report it
  as `ErrBwrapNoCapMethod`, or as a cgroup write failure after writing
  `WriteFiles` into a caller-owned `Workspace`.
- `llmkit/sandbox` (security): an `Env` entry without `=` is refused with
  `InvalidSpecError{Field: "Env"}` on every backend. The CLI backend used to
  pass it to the runtime as `--env NAME`, which copies the HOST's value of
  `NAME` into the container — a host-environment leak into an untrusted
  command.
- `llmkit/sandbox` (llmkit-bk8.1.16, security): the bwrap backend's
  systemd-run resource-cap wrapper now passes `--expand-environment=no` to
  systemd-run, so a model-driven `Spec.Cmd` element such as
  `${NAME}`/`$NAME` is no longer expanded from the HOST environment before
  bwrap starts (measured leak: with `ZZHOSTONLY` set on the host,
  `sh -c 'echo ${ZZHOSTONLY}'` printed the host value inside the sandbox).
  Every argv element now reaches the sandbox byte-for-byte. On a systemd
  older than 254 (no `--expand-environment`), detection fails closed:
  systemd-run is treated as unavailable (cgroup v2 or uncapped per
  `WithCapPolicy`), never "capped but leaking".

- `fsroot`: Resolve now fails closed on dangling symlinks and on any prefix
  it cannot resolve; a dangling link whose target lies outside the root is
  rejected with ErrPathEscape (previously resolved and let a write escape),
  and `file.txt/child`-style paths (ENOTDIR) — including a non-directory
  inside a symlink body — and unsearchable prefixes now return
  ErrPathEscape instead of the lexical path.

- **Breaking:** unknown non-APIError errors and cancellation are terminal
  under `WithRetry` — the hand-rolled root classifier used to retry any
  error it did not recognize. Third-party `Clients` whose `Complete` or
  `Stream` returns plain errors, bare `context.Canceled`, or an
  `*APIError` wrapping `Canceled` go from retried to terminal; wrap
  transport-shaped failures in an `*APIError{Kind: ErrServer}` (as the
  in-tree adapters do) to keep them retryable.
- **Breaking:** the kit's adapters no longer surface a caller's cancelled
  context as an `*APIError`: they return a plain error chaining
  `context.Canceled`. Consumers (such as bugbot's `isTransportError`) that
  treated `StatusCode == 0` as "transport failure" must check
  `StatusCode == 0 && errors.Is(err, llmkit.ErrServer)` instead — a
  pre-wire refusal is also `{Kind: ErrInvalidRequest, StatusCode: 0}`,
  and the `*APIError` with `{ErrServer, 0}` also covers a response the
  adapter could not decode, not only a transport failure; `Kind` is the
  discriminator.
- `Retry-After` is parsed and honoured on every status, not only 429/529:
  a 503 carrying the header is now waited out (worst case
  `(MaxAttempts − 1) × MaxDelay` — the cap applies to each of the waits
  between the `MaxAttempts` attempts).
- In-band SSE errors are classified by the vendor type at `StatusCode`
  200 (the stream was opened on a 2xx response): an Anthropic 200 stream
  whose `error` event carries `overloaded_error` returns `ErrOverloaded`,
  and an OpenAI stream whose data line carries an `error.type` of
  `invalid_request_error`/`server_error` classifies accordingly. Before,
  the two vendors collapsed differently: an Anthropic in-band error
  returned `ErrInvalidRequest` (terminal, one wire hit), while OpenAI and
  openai-compatible reported `APIError{Kind: ErrServer, StatusCode: 0}`
  and retried — so an OpenAI in-band `invalid_request_error` (and any
  auth-class type) goes from retried to terminal, and a `server_error`
  keeps retrying but now with its real status.
- A Google status below 400 (302 included) classifies `ErrServer`
  (retryable) instead of `ErrInvalidRequest`: genai exposes no vendor
  type, and an untyped sub-400 failure is a server-class failure. Real 4xx
  statuses still classify `ErrInvalidRequest`.
- `retry.Do`: when the caller's context is done at the point `Do` returns,
  it never returns a last error the classifier marks retryable: that error
  is replaced by a plain error chaining `ctx.Err()` that carries its text
  (`%v`, never `%w`) — including a parent deadline that expires
  mid-attempt, whose adapter `ErrServer` transport error used to surface as
  a retryable `*APIError`. The replacement is terminal under a classifier
  that treats context errors as terminal, as `llmkit.Classify` does, so
  `WithRetry` never returns a retryable error once the caller's context is
  done — a stream error after a delivered delta included. A last error the
  classifier marks terminal is returned as-is, identity kept (an auth
  `*APIError`, a stream callback's sentinel, an adapter's own cancellation
  error). Before, `Do` returned fn's last error.
- `retry.ParseRetryAfter`: a negative integer clamps to 0, and an integer
  too large for `time.Duration` — including one `strconv` cannot parse —
  saturates to the maximum, so `MaxDelay` caps it. Before, an integer
  outside the int64 range read as absent (the backoff schedule applied),
  and any other integer beyond ±9223372036 wrapped — to 0 (an immediate
  retry) or to an arbitrary delay.
- `retry`: the exponential backoff is capped at `MaxDelay` after jitter is
  applied — jitter never pushes a sleep past the cap (BaseDelay 20s,
  MaxDelay 30s, Jitter 0.2 slept 36s before).
- **Breaking:** `decide.Ask`'s caller cancellation and a deadline it set
  are no longer an `*llmkit.APIError`: they now surface as an error
  chaining the context error (`errors.Is(err, ctx.Err())`), never
  retryable — the same shape `retry.Do` gives every other kit caller once
  the caller's context is done. Every other error from `Ask` stays an
  `*llmkit.APIError` with `Provider "typesafe"`.
- `decide`: `Ask` retries a 200 response that violates the answer
  contract like any other `ErrServer` (each retry re-sends the same
  request) instead of returning after one attempt; `Ask` now carries
  `APIError.HasRetryAfter`; a non-200 status below 400 (a redirect
  without a client-visible resolution, an unexpected 2xx) now
  classifies `ErrServer` (retryable) instead of `ErrInvalidRequest`,
  since the vendor exposes no `Type` to classify a sub-400 failure
  from; a `BaseURL` that `net/url` cannot parse now fails immediately
  as `ErrInvalidRequest` instead of being retried to `MaxAttempts`;
  `decide.Config.Retry` completes via `retry.Config.Or` — see the
  `retry.Config.Or` bullet above for the partial-config rules.
- **Breaking:** `embed`'s errors join the kit vocabulary through the same
  normalization the chat adapters use, and `llmkit.Classify` alone
  decides retryability. A non-200 response is a `*llmkit.APIError`: 429 →
  `ErrRateLimited`, 401/403 → `ErrAuth`, 529 → `ErrOverloaded`, any other
  status 500 or above → `ErrServer`, a 400 whose body reports a
  context-length overflow → `ErrContextTooLong`, and any non-200 status
  below 400 (a 204, a 302 without `Location`) → `ErrServer`, retried.
  The status is classified against the full response body; `Message`
  keeps the first 200 bytes. A transport failure (dial, TLS, EOF while
  reading the body, a stalled attempt reaped by the per-attempt
  timeout) is `{ErrServer, 0}` and retried — a dial failure to a closed
  server used to give up after one hit. An openai-compatible 200 whose
  body carries an error object is a `*APIError` with `StatusCode` 200
  and the `Kind` its type field maps to; a missing or unknown type is
  `ErrServer`, and the route honors `Retry-After`. Before, that
  response was a plain error, terminal after one hit; now it is
  retried unless its type maps to a terminal `Kind`. The caller's
  cancellation is never an `*APIError`: it is the plain context error,
  as it already was. `embed.Config.Validate`, `NewEmbedder`,
  `NewOllamaEmbedder`, and `NewOpenAICompatibleEmbedder` wrap
  construction and validation refusals in `llmkit.ErrInvalidRequest`. A
  decode failure, a wrong embedding count, an index error, and a
  dimension mismatch stay plain terminal errors: `Classify` never
  retries an error outside the kit vocabulary. Error text changes with
  the type: `ollama: HTTP 429: slow down` is now
  `llmkit: ollama error (status 429): slow down`, a transport failure
  reads `llmkit: ollama error: ...` instead of
  `ollama: request failed: ...`, and a cancel mid body read reads
  `llmkit: ollama: context canceled` instead of
  `ollama: read response: context canceled` (the `openai-compatible`
  prefix changes the same way). `embed`'s private `statusError` type
  and classifier are deleted; `embed/retry.go` is gone.
- **Breaking:** `embed.Config.Retry` is completed by `retry.Config.Or`
  against the embed defaults (3 attempts, 60s per-attempt timeout, the
  kit's `BaseDelay`/`MaxDelay`/`Jitter`) instead of embed's own
  Jitter-literal resolver — see the `retry.Config.Or` bullet above for
  the partial-config rules. `LoadConfig` now returns
  `Retry.BaseDelay`, `MaxDelay`, and `Jitter` at zero (before: 500ms,
  30s, 0.2) and sets only `RequestTimeout`, from `<PREFIX>_EMBED_TIMEOUT`;
  the policy an embedder resolves from a loaded `Config` is unchanged.

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

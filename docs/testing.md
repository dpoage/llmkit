# Testing

llmkit ships three test suites: a hermetic default suite, a `live`-tagged acceptance suite against real vendors, and `integration`-tagged suites against local backends. This page states what each suite needs, how each one skips, and how to run the live suite locally.

## The gate

CI (`.github/workflows/ci.yml`) runs these commands on every pull request and on every push to `master`. A dedicated `sandbox-integration` job additionally runs the integration-tagged sandbox suite:

```bash
go build ./...
go vet ./...
go vet -tags live ./provider/ ./agent/ ./examples/... ./decide/ # the `live` acceptance suite must keep compiling
go vet -tags integration ./embed/  # the `integration` Ollama test must keep compiling
go vet -tags integration ./sandbox/ # the sandbox integration test must keep compiling
go test -race -count=1 ./...
golangci-lint run ./...            # config: .golangci.yml (v2 schema, conservative set)
gofmt -l .                         # must print nothing
```

The tag-gated `go vet` steps compile the gated suites without running them. CI pins golangci-lint to v2.13.2.

## The three suites

| Suite | Command | Needs | Skip behavior |
| --- | --- | --- | --- |
| Hermetic (default) | `go test -race -count=1 ./...` | No network, no credentials, no backends. | Needs no network or credentials. A few host-capability tests skip when the host lacks the tool: sandbox bwrap and container-CLI presence checks, platform guards, and the example builds under `-short`. |
| Live (`live` tag) | `go test -tags live -count=1 ./provider/ ./agent/ ./examples/... ./decide/` | Vendor credentials in `LLMKIT_LIVE_*` variables. Calls real vendor APIs and costs money. | Each lane skips at the lane level and names the lane and its missing variables. |
| Integration (`integration` tag) | `go test -tags integration -count=1 ./embed/` and `go test -tags integration -count=1 ./sandbox/` | `./embed/`: a local Ollama server (typically `localhost:11434`). `./sandbox/`: bwrap (`bubblewrap` on Linux) and/or a container runtime (podman or docker). | Each test auto-skips when its backend is missing. CI runs the sandbox suite in the `sandbox-integration` job. |

## Running the live suite locally

The live suite calls real vendor APIs and costs real money. Prompts stay tiny and a full run is well under $0.10.

1. Export the lane variables for each vendor you want to test. The `openai-compatible` lane needs all three:

   ```bash
   export LLMKIT_LIVE_COMPAT_API_KEY="..."
   export LLMKIT_LIVE_COMPAT_BASE_URL="https://api.minimax.io/v1"
   export LLMKIT_LIVE_COMPAT_MODEL="MiniMax-M3"
   ```

2. Optionally export `LLMKIT_LIVE_COMPAT_CAPS` to force capability-gated cases to run (see [below](#llmkit_live_compat_caps)):

   ```bash
   export LLMKIT_LIVE_COMPAT_CAPS="parallel_tool_calls,prompt_caching"
   ```

3. Run the suite:

   ```bash
   go test -tags live -count=1 ./provider/ ./agent/ ./examples/... ./decide/ -v
   ```

4. To re-record the compat fixtures (`provider/testdata/compat/*.json`), add `-update` to the same command. Treat this as a deliberate local operation: every recording differs in model ids, sample text, and token counts.

A keyless lane skips with a message naming the lane and its missing variables. Each test binary prints one `LIVE_TOKENS` summary line covering its own calls. The examples package prints `LIVE_TOKENS note=child-process spend not tallied`, because the example binaries it launches spend separately.

## Live lanes and variables

| Lane | Required variables | Optional | Default model |
| --- | --- | --- | --- |
| `openai-compatible` | `LLMKIT_LIVE_COMPAT_API_KEY`, `LLMKIT_LIVE_COMPAT_BASE_URL`, `LLMKIT_LIVE_COMPAT_MODEL` (all three) | `LLMKIT_LIVE_COMPAT_CAPS` | none; the model must be named |
| `anthropic` | `LLMKIT_LIVE_ANTHROPIC_API_KEY` | `LLMKIT_LIVE_ANTHROPIC_MODEL` | `claude-haiku-4-5` |
| `openai` | `LLMKIT_LIVE_OPENAI_API_KEY` | `LLMKIT_LIVE_OPENAI_MODEL` | `gpt-4o-mini` |
| `google` | `LLMKIT_LIVE_GOOGLE_API_KEY` | `LLMKIT_LIVE_GOOGLE_MODEL` | `gemini-2.5-flash-lite` |
| `typesafe` | `LLMKIT_LIVE_TYPESAFE_API_KEY` | `LLMKIT_LIVE_TYPESAFE_MODEL`, `LLMKIT_LIVE_TYPESAFE_BASE_URL` | `jev-latest` |

### CI workflow

CI runs the live suite in its own workflow, `.github/workflows/live.yml`. The workflow runs on:

- Nightly schedule
- Manual dispatch
- Pushes to `master` or `round/**` that touch any of:
  - `provider/`, `agent/`, `decide/`, `examples/`, `internal/`
  - Any root `*.go` file
  - `go.mod`, `go.sum`
  - The workflow file itself

The workflow deliberately does not run on pull requests. **A PR that edits the workflow cannot read a key.**

The job reads the repo secret `LLMKIT_LIVE_COMPAT_API_KEY` and the repo variables `LLMKIT_LIVE_COMPAT_BASE_URL` and `LLMKIT_LIVE_COMPAT_MODEL`, and sets `LLMKIT_LIVE_COMPAT_CAPS=parallel_tool_calls,prompt_caching`. It also reads the repo secret `LLMKIT_LIVE_TYPESAFE_API_KEY` for the typesafe lane.

The credentials gate keys on the compat key. A missing typesafe secret skips the decide lane inside an otherwise green job. Without the compat key, the job prints `no live credentials — skipped` and exits 0.

## LLMKIT_LIVE_COMPAT_CAPS

`LLMKIT_LIVE_COMPAT_CAPS` is a comma-separated list of `llmkit.Capabilities` field names in snake_case (see [capabilities](capabilities.md)). The suite forces each listed capability to true through `provider.Spec.Capabilities`, so the gated cases run and must pass instead of skipping. A cap you assert must pass on your endpoint, so leave out any that fail.

The set exported above is **MiniMax-M3-verified, not universal**. Probed 2026-09-18 against MiniMax-M3:

- `parallel_tool_calls` passes: two tool calls arrive in one response.
- `prompt_caching` passes: a repeated prefix yields `CacheReadInputTokens > 0` on the second call.
- `top_k` passes trivially: the Chat Completions adapter never serializes `top_k`, so the vendor accepts the request.
- `structured_output` fails: `response_format` with `json_schema` is accepted but ignored, and the model answers in prose.
- `thinking` fails: M3 emits inline `<think>` text, and the adapter never produces `BlockThinking`.
- `stop_sequences` is claimed by the compat profile, but MiniMax-M3 ignores the `stop` parameter at the raw wire. The case skips, with the evidence in its message.

## Fixture replay

The committed live fixtures double as a hermetic suite. `provider/fixture_replay_test.go` runs in plain `go test ./...` with no tag, no network, and no credentials:

1. For every fixture under `provider/testdata/`, an `httptest` server serves the recorded exchanges in order.
2. The replay re-issues the recorded `llmkit.Requests` through `provider.New`, the production construction path, under the fixture's pinned capability profile.
3. The final normalized outcome — Text, ToolCalls, StopReason, Usage — must equal the recorded one. Error cases must still match the recorded sentinel with `errors.Is`.

Each fixture declares a `request_check` mode for the request side:

- `strict`: single-exchange fixtures. Replay asserts the adapter's outgoing request (method, path, and body compared as parsed JSON; headers excluded) equals the recorded request, and the normalized response matches.
- `response_only`: multi-turn fixtures whose later requests echo model-generated ids (tool-call ids, thinking signatures). Replay compares response normalization only.

A fixture without the field replays as `strict` when it holds exactly one exchange and `response_only` otherwise.

The `decide` package replays the same way. `decide/fixture_replay_test.go` runs in plain `go test ./...` with no tag, no network, and no credentials.

For every fixture under `decide/testdata/`, an `httptest` server serves the recorded exchange. The replay re-issues the recorded `Ask` through `decide.New`. The normalized `Response` — or the recorded error's sentinel, matched with `errors.Is` — must equal the recording.

The fixtures come from the same secret-free writer as the provider fixtures.

The fixture writer is secret-free by construction: it refuses to write any fixture containing the lane credential or any `sk-` substring. The nightly `Live` workflow runs the suite without `-update`, so the live matrix's assertions against real responses are the vendor-drift gate. Editing a fixture's recorded response text fails the replay test, and so does adapter wire drift on the request side.

## The live registry rule

Contributor rule: any change to an adapter, the agent loop, or an `llmkit.Capabilities` field must name two things. They are its hermetic test and its live case in the registry in `provider/live_registry_test.go`.

`provider/live_registry_test.go` has no build tag. It reflects over `llmkit.Capabilities` and fails the plain `go test ./...` suite when a field has no registered live case. The same happens when a case gates on a nonexistent field, or when a case has no doc. A capability cannot ship without its acceptance test.

[CONTRIBUTING](../CONTRIBUTING.md) states the rest of the rules for a pull request: the sign-off, the local gate, and what does not get merged.

## Examples

The programs under `examples/` are runnable contract checks, compiled by `go build ./...`:

```bash
go run ./examples/basic
go run ./examples/agent
go run ./examples/structured
go run ./examples/chat
go run ./examples/decide
```

The first four examples no-op with a usage message and exit code 1 unless `LLMKIT_PROVIDER`, `LLMKIT_MODEL`, and `LLMKIT_API_KEY` are set. `LLMKIT_BASE_URL` is required for `openai-compatible` and optional otherwise.

`examples/decide` no-ops with a usage message and exit code 1 unless `LLMKIT_TYPESAFE_API_KEY` and `LLMKIT_TYPESAFE_MODEL` are set. `LLMKIT_TYPESAFE_BASE_URL` is optional.

Without credentials, the examples never touch the network.

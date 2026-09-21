# llmkit

llmkit is a Go toolkit for LLM applications. It ships a provider-agnostic
completion client, a tool-calling agent loop, sandboxed command execution,
embeddings, and TypeSafe Jev decisions. One normalized vocabulary covers
Anthropic, OpenAI, Google Gemini, and any OpenAI-compatible endpoint.

## Install

Requires Go 1.25 or newer.

```bash
go get github.com/dpoage/llmkit@latest
```

## Quick start

The snippets assume an OpenAI-compatible endpoint, such as a local Ollama
server. For Anthropic, OpenAI, or Google construction, see
[providers](docs/providers.md). Export the variables before running:

```bash
export LLMKIT_PROVIDER=openai-compatible
export LLMKIT_MODEL=llama3.1
export LLMKIT_BASE_URL=http://localhost:11434/v1   # required for openai-compatible
export LLMKIT_API_KEY=ollama                       # any non-empty placeholder
```

The programs under `examples/` are the runnable counterparts of the first
two snippets; run them once the variables are set.

Decisions are a separate, non-chat surface. The `decide` package asks
TypeSafe Jev questions about a state. See [decide](docs/decide.md).

### Complete a request

Build a client with `provider.New`. Construction validates the spec and wires
the decorators; it performs no network I/O.

```go
spec := provider.Spec{
	Type:    provider.TypeOpenAICompatible,
	Model:   os.Getenv("LLMKIT_MODEL"),
	BaseURL: os.Getenv("LLMKIT_BASE_URL"),
	Secret:  os.Getenv("LLMKIT_API_KEY"),
}
client, err := provider.New(context.Background(), spec, provider.Options{})
if err != nil {
	log.Fatal(err)
}
resp, err := client.Complete(context.Background(), llmkit.Request{
	Messages: []llmkit.Message{llmkit.UserMessage(llmkit.Text("Hello!"))},
})
if err != nil {
	log.Fatal(err)
}
fmt.Println(resp.Text) // the text blocks, concatenated
fmt.Println(resp.Usage.InputTokens, resp.Usage.OutputTokens)
```

`Complete` returns the normalized `Response`. `resp.Text` concatenates the
text blocks; `resp.Usage` reports token consumption.

### Run an agent with one tool

`agent.Func` derives the tool's JSON Schema from a Go struct, so the schema
and the decoding cannot drift. The runner runs the loop: it sends the task,
executes requested tool calls, feeds results back, and stops at the final
answer. This snippet reuses `client` from the previous section.

```go
type weatherArgs struct {
	City string `json:"city" jsonschema:"description=the city to look up"`
}

weather := agent.Func("weather", "look up the current weather for a city",
	func(_ context.Context, a weatherArgs) (string, error) {
		return "18°C, clear", nil
	})

runner := agent.NewRunner(client, []agent.Tool{weather}, "Answer weather questions.")

outcome, err := runner.Run(context.Background(), "What is the weather in Tokyo?")
if err != nil {
	log.Fatal(err)
}
fmt.Println(outcome.FinalText)
```

### Stream

Stream the request instead of waiting for the full answer. This snippet
reuses `client`, `resp`, and `err` from the previous snippets. `llmkit.Stream`
works on any client: a client that cannot stream gets deltas synthesized from
one `Complete`, and `resp` is the same normalized `Response` either way.

```go
req := llmkit.Request{Messages: []llmkit.Message{llmkit.UserMessage(llmkit.Text("Count to five."))}}
resp, err = llmkit.Stream(context.Background(), client, req,
	func(d llmkit.Delta) error { fmt.Print(d.Text); return nil })
```

## Packages

| Package | Provides | Reference |
|---|---|---|
| [`llmkit`](https://pkg.go.dev/github.com/dpoage/llmkit) | The normalized vocabulary: messages, blocks, requests, responses, usage, capabilities, errors, decorators, streaming. | [capabilities](docs/capabilities.md) |
| [`retry`](https://pkg.go.dev/github.com/dpoage/llmkit/retry) | The shared retry loop: `retry.Do`, `retry.Config`, `retry.Default`, and `retry.ParseRetryAfter`. | [reference](https://pkg.go.dev/github.com/dpoage/llmkit/retry) |
| [`provider`](https://pkg.go.dev/github.com/dpoage/llmkit/provider) | Client construction: one `New` per endpoint; validation, adapters, and the decorator stack. | [providers](docs/providers.md) |
| [`agent`](https://pkg.go.dev/github.com/dpoage/llmkit/agent) | The tool-calling loop: budgets, policies, hooks, steering, structured output, transcripts. | [agent loop](docs/agent-loop.md) |
| [`sandbox`](https://pkg.go.dev/github.com/dpoage/llmkit/sandbox) | Isolated execution of untrusted commands: Bubblewrap, container CLIs, `HostExec`, `Mock`. | [sandbox](docs/sandbox.md) |
| [`fsroot`](https://pkg.go.dev/github.com/dpoage/llmkit/fsroot) | Path containment for agent file tools. | [sandbox](docs/sandbox.md) |
| [`embed`](https://pkg.go.dev/github.com/dpoage/llmkit/embed) | Embeddings: Ollama and OpenAI-compatible backends, batching, retry, least-recently-used (LRU) cache. | [reference](https://pkg.go.dev/github.com/dpoage/llmkit/embed) |
| [`decide`](https://pkg.go.dev/github.com/dpoage/llmkit/decide) | Decisions with TypeSafe's Jev model: `noul`, `choice`, and `score` questions over a state; calibrated probabilities and confidence. | [decide](docs/decide.md) |

## Examples

Five runnable programs live under `examples/`:

- `examples/basic` — one completion with content blocks and capability gating.
- `examples/agent` — the agent loop with hooks, a tool policy, and optional parallel dispatch.
- `examples/structured` — schema-constrained output with `RunJSONAs`.
- `examples/chat` — a multi-turn read–eval–print loop (REPL) with mid-run steering.
- `examples/decide` — one mixed `Ask` (noul, choice, score) against TypeSafe Jev.

The first four read `LLMKIT_PROVIDER`, `LLMKIT_MODEL`, and `LLMKIT_API_KEY`.
`LLMKIT_BASE_URL` is required for `openai-compatible` and optional otherwise.
`examples/decide` reads `LLMKIT_TYPESAFE_API_KEY` and `LLMKIT_TYPESAFE_MODEL`.
`LLMKIT_TYPESAFE_BASE_URL` is optional. If a required variable is unset, an
example prints its usage and exits 1 without touching the network.

```bash
go run ./examples/basic --image path/to/photo.jpg
go run ./examples/agent --parallel
go run ./examples/structured
go run ./examples/chat
go run ./examples/decide
```

## Documentation

- [docs/README.md](docs/README.md) — the index, with a reading order for new users.
- [docs/design.md](docs/design.md) — the architecture and the decisions behind it.
- [docs/providers.md](docs/providers.md) — building a client for each provider type.
- [docs/capabilities.md](docs/capabilities.md) — what `Capabilities` reports and enforces.
- [docs/agent-loop.md](docs/agent-loop.md) — the `agent.Runner` loop, policies, hooks, steering.
- [docs/sandbox.md](docs/sandbox.md) — sandbox backends, threat model, and `fsroot`.
- [docs/decide.md](docs/decide.md) — the `decide` package: TypeSafe Jev decisions, vendor limits, and the live lane.
- [docs/testing.md](docs/testing.md) — the test suites and how to run them.

## Testing

CI runs the full gate on every pull request and on every push to `master`:
build, vet, race-enabled tests, lint, and gofmt. The live acceptance suite
runs against real vendors and skips itself without credentials. See
[docs/testing.md](docs/testing.md).

## Stability

llmkit is pre-1.0. Semver minor versions may contain breaking changes; the
[changelog](CHANGELOG.md) marks them. Every release so far has carried them:
v0.2.0 changed the message model to content blocks, and v0.3.0 changed
`provider.New` and the recorder.

## Contributing

llmkit takes changes as pull requests from a fork; direct pushes are blocked.
Every commit needs a Developer Certificate of Origin sign-off (`git commit -s`),
and your contribution ships under AGPL-3.0. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before you start, and report vulnerabilities
through [SECURITY.md](SECURITY.md) rather than a public issue.

## License

AGPL-3.0. The `embed` package derives from MIT-licensed `known`; the copyright
holder relicensed it here.

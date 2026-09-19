# Documentation

llmkit ships a provider-agnostic completion client, a tool-calling agent
loop, sandboxed command execution, and embeddings. Start with the
[README quick start](../README.md#quick-start), then read the pages in this
order:

1. [providers](providers.md) — build a client for your endpoint.
2. [capabilities](capabilities.md) — what a client reports and what it enforces.
3. [agent loop](agent-loop.md) — the `agent.Runner` loop, policies, hooks, steering.
4. [sandbox](sandbox.md) — untrusted command execution and `fsroot` path containment.
5. [testing](testing.md) — the test suites and the live acceptance regime.
6. [design](design.md) — the architecture and the decisions behind it.

| Page | Describes |
|---|---|
| [providers](providers.md) | One construction per provider type: `Spec`, credentials, capability overrides, error normalization. |
| [capabilities](capabilities.md) | The `Capabilities` profile: the four enforcement classes and the per-field table. |
| [agent loop](agent-loop.md) | Tools, limits, policies, hooks, steering, structured output, budgets, transcripts, replay. |
| [sandbox](sandbox.md) | Backends, threat model, the spec-field honor matrix, watchdog, workspaces, probes. |
| [testing](testing.md) | The hermetic, live, and integration suites; the live lanes; fixture replay. |
| [design](design.md) | The layering and one decision record per architecture choice, with costs. |

The API reference is canonical:
[pkg.go.dev/github.com/dpoage/llmkit](https://pkg.go.dev/github.com/dpoage/llmkit).
Each package links its page from the
[README package table](../README.md#packages).

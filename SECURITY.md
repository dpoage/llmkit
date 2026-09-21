# Security

## Reporting a vulnerability

Report privately through GitHub Security Advisories:
[open a draft advisory](https://github.com/dpoage/llmkit/security/advisories/new).
Do not open a public issue, and do not post the details in a pull request.

llmkit has one maintainer. You get an acknowledgement within seven days. Give
the fix and the advisory a reasonable window before you disclose publicly.

Include what a reader needs to reproduce the finding:

- The llmkit version or commit.
- The package and the entry point.
- A runnable reproduction, or the exact call sequence.
- What an attacker gains.

## Supported versions

llmkit is pre-1.0. Fixes land on `master` and ship in the next minor release.
Older minor versions get no backports.

## Scope

In scope:

- Credential leaks: a key reaching a log, an error string, a transcript, an
  observability event, or a recorded fixture.
- Sandbox escapes, measured against the threat model in
  [docs/sandbox.md](docs/sandbox.md).
- Path containment failures in `fsroot`.
- Parsing or normalization of a provider response that corrupts memory, panics
  a caller that handled the error correctly, or executes attacker-chosen data.

Out of scope:

- Prompt injection that makes a model produce bad output. The agent loop runs
  tools that a caller registers; a caller that registers a dangerous tool owns
  that decision.
- Vulnerabilities in a provider's service rather than in llmkit's client.
- A `sandbox` backend behaving as its documented honor matrix says it behaves.
  `HostExec` applies no isolation by design, and that is documented.

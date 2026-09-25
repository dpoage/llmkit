# Decide

The `decide` package evaluates decisions with TypeSafe's Jev decision
model (the System One API). One `Client.Ask` call sends a state plus a
set of typed questions and returns calibrated beliefs and probability
distributions.

Jev is not a chat model. It has no messages, no tools, no streaming, and
no text output. The package does not implement `llmkit.Client`; it does
not go through `provider.New`. See
[decision models are not Clients](design.md#decision-models-are-not-clients)
for that decision. The API reference is canonical:
[pkg.go.dev/github.com/dpoage/llmkit/decide](https://pkg.go.dev/github.com/dpoage/llmkit/decide).

## Constructing a client

```go
package main

import (
	"fmt"
	"os"

	"github.com/dpoage/llmkit/decide"
)

func main() {
	client, err := decide.New(decide.Config{
		APIKey: os.Getenv("LLMKIT_TYPESAFE_API_KEY"),
		Model:  "jev-latest",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(client != nil) // New performs no network I/O and no environment lookups
}
```

`New` validates the config and performs no network I/O and no environment
lookups. Construction is therefore hermetic and testable with a
placeholder key. An invalid field returns an error wrapping
`llmkit.ErrInvalidRequest`; the error never echoes the key.

| Field | Required | Effect and default |
|---|---|---|
| `APIKey` | Yes | Non-empty, no surrounding whitespace (the same rule as `provider.Spec.Secret`). |
| `Model` | Yes | A versioned id (`jev-1.13.0`) or alias (`jev-latest`, `jev-preview`). There is no default alias. |
| `BaseURL` | No | Endpoint root for tests and gateways. Default: `https://api.typesafe.ai`; the path `/v1/systemone` is appended. |
| `HTTPClient` | No | Used as-is, including its `Timeout`. Default: a plain client with no `http.Client.Timeout`, so the per-attempt `RequestTimeout` is the only bound. |
| `Retry` | No | Resolved at construction via `retry.Config.Or`: 3 attempts, 30 s per-attempt timeout, `BaseDelay` (500 ms), `MaxDelay` (30 s), and `Jitter` (20%) from `retry.Default`. An explicit `Jitter` of 0 resolves like every other unset field; pin `Retry.Rand` for the resolved defaults with no jitter. |
| `Recorder` | No | Receives one `llmkit.UsageEvent` per successful `Ask` through its `Record(llmkit.UsageEvent)` method. Default: nil (no recording). |

## The three question types

`Question` is a sealed interface with three implementations. The wire `type`
discriminator comes from the Go type; callers never write it. The state,
instructions, and descriptions accept a Go string, anything that marshals to a
JSON object, or a JSON array. JSON null is allowed for descriptions and
criteria. The client rejects a JSON number or boolean before sending the
request.

### Noul

`Noul` asks a binary belief question: how strongly the state supports the
`True` description over the `False` one. Both criteria are optional; a nil
side omits its key.

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/dpoage/llmkit/decide"
)

func main() {
	client, err := decide.New(decide.Config{
		APIKey: os.Getenv("LLMKIT_TYPESAFE_API_KEY"),
		Model:  "jev-latest",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	resp, err := client.Ask(context.Background(),
		"The customer was charged twice for one order and asks for a refund.",
		decide.Questions{
			"refund": decide.Noul{
				Instructions: "Does the state support granting a refund?",
				True:         "The charge history fits the refund policy.",
				False:        "The charge history does not fit the refund policy.",
			},
		})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(resp.Nouls["refund"]) // belief in [0, 1]
}
```

### Choice

`Choice` asks which option best fits the state. `Options` maps option id to
description and is required and non-empty.

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/dpoage/llmkit/decide"
)

func main() {
	client, err := decide.New(decide.Config{
		APIKey: os.Getenv("LLMKIT_TYPESAFE_API_KEY"),
		Model:  "jev-latest",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	resp, err := client.Ask(context.Background(),
		"Password reset link expired after the user clicked it once.",
		decide.Questions{
			"route": decide.Choice{
				Instructions: "Which single action best serves this user next?",
				Options: map[string]any{
					"resend_link": "Email a fresh password-reset link.",
					"human":       "Route to a human support agent.",
				},
			},
		})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	ans := resp.Choices["route"]
	fmt.Println(ans.Choice, ans.Confidence)
	for option, p := range ans.Probabilities {
		fmt.Println(option, p)
	}
}
```

### Score

`Score` asks for a rating against an ordered legend. `Levels` is required and
holds at least two descriptions, from lowest to highest.

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/dpoage/llmkit/decide"
)

func main() {
	client, err := decide.New(decide.Config{
		APIKey: os.Getenv("LLMKIT_TYPESAFE_API_KEY"),
		Model:  "jev-latest",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	resp, err := client.Ask(context.Background(),
		"Checkout has been down for 20 minutes and the queue is growing.",
		decide.Questions{
			"urgency": decide.Score{
				Instructions: "Rate the ticket's urgency.",
				Levels: []any{
					"routine",
					"pressing",
					"blocking work",
					"active outage",
				},
			},
		})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	ans := resp.Scores["urgency"]
	fmt.Println(ans.Score)
	for i, level := range ans.Levels {
		fmt.Println(level, ans.Probabilities[i])
	}
}
```
### One Ask with all three types

One `Ask` evaluates every question against the state in one logical
request; only retries put more HTTP requests on the wire. A mixed set
needs no second call from you.

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/dpoage/llmkit/decide"
)

func main() {
	client, err := decide.New(decide.Config{
		APIKey: os.Getenv("LLMKIT_TYPESAFE_API_KEY"),
		Model:  "jev-latest",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	resp, err := client.Ask(context.Background(),
		map[string]any{
			"ticket": "Charged twice for one order. Checkout is down for others too.",
			"plan":   "pro",
		},
		decide.Questions{
			"refund": decide.Noul{
				Instructions: "Does the state support granting a refund?",
			},
			"route": decide.Choice{
				Instructions: "Which single action best serves this user next?",
				Options: map[string]any{
					"resend_link": "Email a fresh password-reset link.",
					"refund":      "Issue the refund.",
					"human":       "Route to a human support agent.",
				},
			},
			"urgency": decide.Score{
				Instructions: "Rate the ticket's urgency.",
				Levels:       []any{"routine", "pressing", "blocking work", "active outage"},
			},
		})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Println(resp.Model, resp.Usage.InputTokens) // versioned id, billed input
	fmt.Println(resp.Nouls["refund"])               // belief in [0, 1]
	fmt.Println(resp.Choices["route"].Choice)
	fmt.Println(resp.Scores["urgency"].Score)
}
```

## Reading the answers

`Response` splits the answers by question kind:

- `Response.Nouls` maps a noul question id to its belief, a `float64` in
  [0, 1].
- `Response.Choices` maps a choice question id to a `ChoiceAnswer`: the
  selected option, per-option `Probabilities`, and `Confidence`.
- `Response.Scores` maps a score question id to a `ScoreAnswer`: the numeric
  `Score`, the server's `Levels` legend, `Probabilities`, and `Confidence`.

A map stays nil when the `Ask` included no question of that kind.

`ScoreAnswer` carries `Probabilities` index-aligned with `Levels`. Both
follow the server's legend order, from `"0"` to `"n-1"` with `n` equal
to the levels asked for. `decide` passes them through verbatim: it
renormalizes nothing and recomputes no argmax.

### Confidence and probabilities

Every `Choice` and `Score` answer carries both. `Probabilities` is the
full distribution. Its shape — concentrated on one outcome or spread out
— describes the model's uncertainty. `Confidence` is a vendor-computed
statistic that collapses that shape into one number in [0, 1]. Threshold
on `Confidence` when you do not want to compute a spread yourself.

A `Noul` answer carries no confidence; the belief is the number. See the
vendor page
[Confidence](https://docs.typesafe.ai/confidence.md) (no vendor review
date on the page; verified 2026-09-19).

## Errors

Every error from `Ask` is an `*llmkit.APIError` with `Provider` `"typesafe"`,
except the caller's cancellation or a deadline it set, which is an error
chaining the context error: `errors.Is(err, ctx.Err())`, never retryable.
Match an `*llmkit.APIError`'s `Kind` with `errors.Is`. The status mapping mirrors
[providers](providers.md#error-normalization); the decide-specific rows are
marked:

| HTTP status | `llmkit` kind | Retried by `Ask` |
|---|---|---|
| 429 | `ErrRateLimited` | Yes; `Retry-After` honored |
| 401, 403 | `ErrAuth` | No |
| 413 | `ErrContextTooLong` | No |
| 400 with a context-length message ("prompt is too long", "context length", ...) | `ErrContextTooLong` | No |
| 400, other | `ErrInvalidRequest` | No |
| 422 | `ErrInvalidRequest` | No (decide row) |
| 529 | `ErrOverloaded` | Yes (same `Retry-After` rules as 429) |
| Any other status 500 or above | `ErrServer` | Yes |
| Any other 4xx (404, 409, ...) | `ErrInvalidRequest` | No |
| Any other non-200 status below 400 (a 3xx, an unexpected 2xx) | `ErrServer` | Yes (decide row: no vendor `Type` to classify from) |
| Transport failure (timeout, connection reset) | `ErrServer` (`StatusCode` 0) | Yes |
| 200 body that violates the answer contract | `ErrServer` (`StatusCode` 200) | Yes (decide row: retried like any `ErrServer`) |
| The HTTP request could not be built (a `BaseURL` that `net/url` cannot parse) | `ErrInvalidRequest` | No (decide row) |

A 200 response that violates the answer contract is a server contract
violation. The client returns `ErrServer` with `StatusCode` 200 and retries
it like any other `ErrServer` — each retry re-sends the same request, so a
persistently misbehaving server still exhausts `MaxAttempts` rather than
failing after one attempt. Violations are: a missing model or usage
block; a missing or extra answer; an answer type that does not match its
question; a sparse legend.

A refused pre-wire request also returns `ErrInvalidRequest` before any
network call. The client rejects: nil state; empty questions; empty
question id; nil instructions; a `Choice` without options; a `Score`
with fewer than two levels; a JSON number or boolean where only text
kinds are accepted.

An unknown model arrives as a 400 in the live lane (observed 2026-09-20).
The vendor's API doc reserves 422 for validation failures. Both map to
`ErrInvalidRequest`.

For a non-200 response, the client parses the `Retry-After` header and
carries its presence as `APIError.HasRetryAfter`. On a retried non-200
status, a server-supplied delay — present zero included — replaces the
exponential backoff for the sleep and is capped at `retry.Config.MaxDelay`
(30 s by default); absence falls back to the schedule. A 200 response that
violates the answer contract always takes the schedule: the client does not
read its headers. `APIError.RetryAfter` carries the raw server value; it is
meaningful only when `HasRetryAfter` is true.

The caller's `ctx` bounds the whole call. Its cancellation, or a deadline it
set, ends the loop immediately — whether that happens mid-attempt or while
waiting between retries — without another attempt. The error `Ask` returns
then chains the context error and is not an `*llmkit.APIError`. One
exception: a terminal error keeps its identity even when the `ctx` is
already done. An already-cancelled `ctx` with a `BaseURL` that `net/url`
cannot parse returns `ErrInvalidRequest`. A per-attempt `RequestTimeout`
expiring under a live parent is not a parent cancellation and is retried
like any other `ErrServer`.

Error messages carry the vendor body text, truncated to 200 characters plus
an appended `...`. llmkit never places the API key into an error.

## Usage and the Recorder

TypeSafe bills input tokens only; output tokens are free
([Models](https://docs.typesafe.ai/models.md); no vendor review date on the
page; verified 2026-09-19). `Response.Usage` reports both counts.

A non-nil `Config.Recorder` receives exactly one `llmkit.UsageEvent`
per successful `Ask`. `Provider` is `"typesafe"` and `Model` is the
versioned id the server reported. The recorder fires on success only
and never on failure.

## Vendor limits and jaggedness

- A request may total at most 64k tokens, and state plus the longest question
  at most 32k. `decide` does not count tokens; oversized requests surface as
  vendor 4xx errors. ([Models](https://docs.typesafe.ai/models.md); no vendor
  review date; verified 2026-09-19)
- Input is text only: string, JSON object, or array of text values. No image,
  audio, or video. ([Models](https://docs.typesafe.ai/models.md); verified
  2026-09-19)
- English is the primary training language and where accuracy is best; other
  languages, including CJK scripts, are handled but not equally well.
  ([Models](https://docs.typesafe.ai/models.md); verified 2026-09-19)
- Rate limits (250,000 tokens per second, 1,200 requests per minute) return
  429, and the vendor warns the limits can change without notice.
  ([Models](https://docs.typesafe.ai/models.md); verified 2026-09-19)
- Aliases move: `jev-latest` and `jev-preview` point at a versioned id
  (`jev-1.13.0` today) that can change between releases. `Response.Model`
  reports the id actually used; ledger on it.
  ([Models](https://docs.typesafe.ai/models.md); verified 2026-09-19)
- Jev reads literally: it answers the question you wrote, not the one you
  meant. State the exact condition in the instructions; put boundary cases in
  the criteria. ([Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md);
  vendor "Last reviewed 2026-09-17")
- Jev is not a calculator: counting, arithmetic, and date-and-time comparison
  are unreliable. Keep them in code, and do not reconstruct a number by
  interpolating between score levels. ([Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md);
  vendor "Last reviewed 2026-09-17")
- Context rot: accuracy falls as the state grows with content unrelated to the
  decision. Filter in code first, and send only the fields the question
  needs. ([Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md);
  vendor "Last reviewed 2026-09-17")
- Adversarial state: state is data, and the model does not treat it as hostile.
  Content written to steer the model — an injected instruction, a misleading
  framing — can move the answer. ([Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md);
  vendor "Last reviewed 2026-09-17")
- No cross-question invariants: probabilities from separate questions are not
  guaranteed to satisfy arithmetic identities, and a threshold tuned on a
  Noul does not carry to a Choice. ([Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md);
  vendor "Last reviewed 2026-09-17")

## The live lane and the example program

The `live`-tagged suite calls the real API and costs money. The lane skips
itself without credentials. Lane variables:

| Variable | Role |
|---|---|
| `LLMKIT_LIVE_TYPESAFE_API_KEY` | Required. The lane skips, naming this variable, without it. |
| `LLMKIT_LIVE_TYPESAFE_MODEL` | Optional. Default `jev-latest`. |
| `LLMKIT_LIVE_TYPESAFE_BASE_URL` | Optional. Endpoint override. |

Run the lane:

```bash
go test -tags live -count=1 ./decide/ -v
```

A keyless run skips with a message naming `LLMKIT_LIVE_TYPESAFE_API_KEY`, and
the package prints one `LIVE_TOKENS` total line at exit. Add `-update` to
re-record the fixtures under `decide/testdata/`; treat that as a deliberate
local operation. The lane cases and the skip rules follow
[testing](testing.md).

The recorded fixtures replay hermetically: `decide/fixture_replay_test.go`
runs in plain `go test ./...` with no tag, no network, and no credentials.

`examples/decide` is the runnable one-Ask program:

```bash
export LLMKIT_TYPESAFE_API_KEY="..."
export LLMKIT_TYPESAFE_MODEL=jev-latest
go run ./examples/decide
```

It reads `LLMKIT_TYPESAFE_API_KEY` and `LLMKIT_TYPESAFE_MODEL` (both
required) and `LLMKIT_TYPESAFE_BASE_URL` (optional). Without the required
variables it prints its usage and exits 1 without touching the network.

CI runs the lane in the nightly Live workflow, which reads the repo secret
`LLMKIT_LIVE_TYPESAFE_API_KEY`; see [testing](testing.md) for the workflow
details.

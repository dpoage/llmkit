# Decide

Decisions are two packages:

- `decide` is the vocabulary: the `Noul`, `Choice`, and `Score` questions,
  `Questions`, `Response` with its accessors, `Check`, `Normalize`, the
  `Decider` interface, and the two decorators `Observe` and `Replay`. It
  holds no HTTP code and imports no backend.
- `decide/systemone` is the System One wire backend. `systemone.Client`
  satisfies `decide.Decider`. One `Client.Ask` call sends a state plus a
  set of typed questions to a System One host and returns calibrated
  beliefs and probability distributions. The default host is TypeSafe's
  Jev decision model.

A decision model is not a chat model. It has no messages, no tools, no
streaming, and no text output. Neither package implements `llmkit.Client`;
neither goes through `provider.New`. See
[decision models are not Clients](design.md#decision-models-are-not-clients)
for that decision. The API references are canonical:
[decide](https://pkg.go.dev/github.com/dpoage/llmkit/decide) and
[decide/systemone](https://pkg.go.dev/github.com/dpoage/llmkit/decide/systemone).

## Hosts

Every host that speaks the System One envelope at `POST /v1/systemone`
works behind `Config.BaseURL`. The default host is TypeSafe at
`https://api.typesafe.ai`. The hosts differ as follows:

| | TypeSafe Jev | Kev ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)) |
|---|---|---|
| Reported model | The versioned id (`jev-1.13.0` for `jev-latest`). | The requested alias, echoed (`kev-latest`). |
| Billing | Input tokens only; output tokens are free. | Open weights (Apache-2.0); you run the server. |
| Instructions-less questions, 1-level scores | `decide.Normalize` refuses both. | The host accepts both, but `decide.Normalize` refuses both before the request. |
| `GET /v1/models` | Listed in the OpenAPI document. The client does not call it. | Exposed. The client does not call it. |
| Probabilities | As reported. | Reported at 4 decimals. |
| Extra response fields | None known. | `latency_ms`, ignored by the client. |
| Auth | `Authorization: Bearer` key. | A local `kev.serve` needs no auth, but `systemone.New` still requires a non-empty `Secret`. |

## Constructing a client

```go
package main

import (
	"fmt"
	"os"

	"github.com/dpoage/llmkit/decide/systemone"
)

func main() {
	client, err := systemone.New(systemone.Config{
		Secret: os.Getenv("LLMKIT_SYSTEMONE_API_KEY"),
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
| `Secret` | Yes | Non-empty, no surrounding whitespace (the same rule as `provider.Spec.Secret`). |
| `Model` | Yes | A model id or alias the host serves: `jev-1.13.0`, `jev-latest`, `jev-preview`, `kev-latest`. There is no default alias. |
| `BaseURL` | No | Endpoint root of the System One host (TypeSafe, a Kev server, a gateway, a test server), without `/v1`. Default: `https://api.typesafe.ai`. `New` refuses a `BaseURL` that is not an `http` or `https` URL with a host (the scheme is case-insensitive) or that contains `#` anywhere, an empty fragment included, with `ErrInvalidRequest`; the error never echoes the URL or its userinfo. The path `/v1/systemone` is appended to the URL's path (trailing `/` characters dropped) and a query is kept after it: `https://gw.example/x?api-version=1` requests `/x/v1/systemone?api-version=1`. The path is not cleaned: `//` and dot segments are not collapsed or resolved. A URL that parses but cannot be dialed (`http://localhost:99999`) is accepted and fails at request time as a retryable transport error. |
| `HTTPClient` | No | Used as-is, including its `Timeout`. Default: a plain client with no `http.Client.Timeout`, so the per-attempt `RequestTimeout` is the only bound. |
| `Retry` | No | Resolved at construction via `retry.Config.Or`: 3 attempts, 30 s per-attempt timeout, `BaseDelay` (500 ms), `MaxDelay` (30 s), and `Jitter` (20%) from `retry.Default`. An explicit `Jitter` of 0 resolves like every other unset field; pin `Retry.Rand` for the resolved defaults with no jitter. |

## The three question types

`decide.Question` is a sealed interface with three implementations. The wire
`type` discriminator comes from the Go type; callers never write it. The state,
instructions, and descriptions accept a Go string, anything that marshals to a
JSON object, or a JSON array. JSON null is allowed for descriptions and
criteria. `decide.Normalize` rejects a JSON number or boolean before the client
sends the request.

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
	"github.com/dpoage/llmkit/decide/systemone"
)

func main() {
	client, err := systemone.New(systemone.Config{
		Secret: os.Getenv("LLMKIT_SYSTEMONE_API_KEY"),
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
	belief, err := resp.Noul("refund") // belief in [0, 1]
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(belief)
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
	"github.com/dpoage/llmkit/decide/systemone"
)

func main() {
	client, err := systemone.New(systemone.Config{
		Secret: os.Getenv("LLMKIT_SYSTEMONE_API_KEY"),
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

	ans, err := resp.Choice("route")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
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
	"github.com/dpoage/llmkit/decide/systemone"
)

func main() {
	client, err := systemone.New(systemone.Config{
		Secret: os.Getenv("LLMKIT_SYSTEMONE_API_KEY"),
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

	ans, err := resp.Score("urgency")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
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
	"github.com/dpoage/llmkit/decide/systemone"
)

func main() {
	client, err := systemone.New(systemone.Config{
		Secret: os.Getenv("LLMKIT_SYSTEMONE_API_KEY"),
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

	fmt.Println(resp.Model, resp.Usage.InputTokens) // reported model id, input tokens
	refund, err := resp.Noul("refund")              // belief in [0, 1]
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	route, err := resp.Choice("route")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	urgency, err := resp.Score("urgency")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(refund, route.Choice, urgency.Score)
}
```

## Images

A state, an `Instructions` value, or a description (`True`, `False`, an
`Options` value, a `Levels` entry) may hold an image. `decide.Normalize` finds
it in the JSON that `encoding/json` produces, replaces it with a label, and
returns it once in `Request.Images`; the backend sends the bytes its own way.

**What counts as an image.** Any JSON object that decodes as an
`llmkit.Block` of kind `"image"` and uses only `Block`'s member names. Names
match as `encoding/json` matches them (case-insensitive, escapes decoded; a
repeated member takes its last value) and `kind` must equal `"image"` exactly.
It does not matter how the object got there: an `llmkit.Image` block, a
`Message` whose content holds one, a `json.RawMessage`, or a map. An object
with any other member (`{"kind":"image","caption":"x"}`) or another kind is
left as it was. Block-shaped data you did not mean as an image is an image;
there is no escape. An image is replaced whole; nothing inside it is scanned.

**A valid image** has inline `Data` and a `MediaType` that begins `image/`;
`URL`, `Text`, `Title`, `Provider` and `Raw` are empty (an empty `"text":""`
is fine; `"raw":null` is not). An object that is an image by the rule above
but fails this, including one whose member does not decode, is refused
naming its path, for example `state["frames"][2]["frame"]: invalid image: no
inline data`. It is never sent as text. Nothing checks the pixels or sniffs
the bytes. The same bytes under two media types are refused.

**Labels and order.** `Request.Images` holds the distinct images (deduplicated
by bytes) in order of first occurrence: the state in document order, then each
question in sorted id order, instructions before criteria (true, false,
options by sorted key, levels by index). Each `Image` has `Ref`
(`<image:` and the first 16 bytes of the SHA-256 in lowercase hex, `>`),
`SHA256`, `MediaType`, `Data`, and `Placements`, the union of the places it
occurs in: `PlaceState`, `PlaceInstructions`, `PlaceCriteria`. `Request.State`
and `Request.Questions` carry each image's label in its place: its `Ref`, or
`[image N]` (N from 1, the index in `Request.Images`) under `LabelPosition`. A
string of yours that equals a label is never rewritten.

**What a backend carries.** `decide.ImageSupport{Placements, MediaTypes,
Label}` is what a backend passes to `Normalize`. The zero value refuses every
image; empty `MediaTypes` accepts any `image/*`. An image outside the
placements or media types is refused with `ErrInvalidRequest` and, in
addition, wraps `decide.ErrImageUnsupported`; an invalid image does not.
`decide.Observe` and `decide.Replay` normalize with `PlaceAll`. A `Decider`
declares its support with `ImageSupport() decide.ImageSupport`
(`decide.ImageSupporter`). `systemone.Client` carries no image yet: its refusal
names `Config.Images`.

## Reading the answers

`Response` splits the answers by question kind:

- `Response.Nouls` maps a noul question id to its belief, a `float64` in
  [0, 1].
- `Response.Choices` maps a choice question id to a `ChoiceAnswer`: the
  selected option, per-option `Probabilities`, and `Confidence`.
- `Response.Scores` maps a score question id to a `ScoreAnswer`: the numeric
  `Score`, the server's `Levels` legend, `Probabilities`, and `Confidence`.

A map stays nil when the `Ask` included no question of that kind.

`ScoreAnswer.Probabilities[i]` is the probability of asked level `i`. The
System One client fills `Levels` and `Probabilities` from the server's
legend keys `"0"` to `"n-1"`, with `n` equal to the levels asked for, and
passes them through verbatim: it renormalizes nothing and recomputes no
argmax. A backend may leave either slice nil; `Check` accepts nil or a full
slice for each.

### Accessors

`Response.Noul(id)`, `Response.Choice(id)`, and `Response.Score(id)` return
the answer for `id`, or an error wrapping `decide.ErrNotAsked` when `id` was
not asked as that kind of question. A mistyped id or an id of another kind
fails loudly; indexing the maps instead reads a zero value, which a
threshold check treats as a belief of 0.

```go
belief, err := resp.Noul("urgent")
if errors.Is(err, decide.ErrNotAsked) {
	// "urgent" was not a noul question in this Ask.
}
```

`decide.Check(qs, resp)` reports whether `resp` answers `qs` exactly: one
answer per asked id, in the map of its question's kind; a `Choice` answer
that names one of the asked options; a `Score` answer whose `Levels` and
`Probabilities` each have one entry per asked level, or none. The error
names the offending id. `systemone.Client.Ask` applies it to every answer set it decodes.

### Backends that report no distribution

`ChoiceAnswer.Probabilities`, `ScoreAnswer.Levels`, and
`ScoreAnswer.Probabilities` may be nil, and `Confidence` 0, when a backend
reports no distribution; a backend must not fabricate them. The System One
client rejects a score answer missing its legend or probabilities. It
accepts a `Choice` answer without them: `Probabilities` is nil and
`Confidence` is 0.

### Confidence and probabilities

When a host reports them, a `Choice` or `Score` answer carries `Probabilities`
(the full distribution) and `Confidence`. A missing `confidence` reads as 0.
The shape of the distribution, concentrated on one outcome or spread out,
describes the model's uncertainty. `Confidence` is a vendor-computed
statistic that collapses that shape into one number in [0, 1]. Threshold
on `Confidence` when you do not want to compute a spread yourself.

A `Noul` answer carries no confidence; the belief is the number. See the
vendor page
[Confidence](https://docs.typesafe.ai/confidence.md) (no vendor review
date on the page; verified 2026-09-19).

## Errors

Every error from `Ask` is an `*llmkit.APIError` with `Provider` `"systemone"`
(a validation refusal from `decide.Normalize` is re-tagged from `"decide"`),
except the caller's cancellation or a deadline it set, which is an error
chaining the context error: `errors.Is(err, ctx.Err())`, never retryable.
Match an `*llmkit.APIError`'s `Kind` with `errors.Is`. The status mapping mirrors
[providers](providers.md#error-normalization); the System One rows are
marked:

| HTTP status | `llmkit` kind | Retried by `Ask` |
|---|---|---|
| 429 | `ErrRateLimited` | Yes; `Retry-After` honored |
| 401, 403 | `ErrAuth` | No |
| 413 | `ErrContextTooLong` | No |
| 400 with a context-length message ("prompt is too long", "context length", ...) | `ErrContextTooLong` | No |
| 400, other | `ErrInvalidRequest` | No |
| 422 | `ErrInvalidRequest` | No (System One row) |
| 529 | `ErrOverloaded` | Yes (same `Retry-After` rules as 429) |
| Any other status 500 or above | `ErrServer` | Yes |
| Any other 4xx (404, 409, ...) | `ErrInvalidRequest` | No |
| Any other non-200 status below 400 (a 3xx, an unexpected 2xx) | `ErrServer` | Yes (System One row: no host `Type` to classify from) |
| Transport failure (timeout, connection reset) | `ErrServer` (`StatusCode` 0) | Yes |
| Redirect-policy failure (the client stopped after 10 redirects, or the `CheckRedirect` of `Config.HTTPClient` returned an error) | `ErrInvalidRequest` (`StatusCode` 0) | No; terminal |
| 200 body that violates the answer contract | `ErrServer` (`StatusCode` 200) | Yes (System One row: retried like any `ErrServer`) |

A 200 response that violates the answer contract is a server contract
violation. The client returns `ErrServer` with `StatusCode` 200 and retries
it like any other `ErrServer` — each retry re-sends the same request, so a
persistently misbehaving server still exhausts `MaxAttempts` rather than
failing after one attempt. Violations are: a missing model or usage
block; a missing or extra answer; an answer type that does not match its
question; a `Choice` answer that names an option that was not asked; a
sparse legend. These checks stay `ErrServer` and retried; the kit has no
non-retryable kind for a 200 answer violation.

A refused pre-wire request also returns `ErrInvalidRequest` before any
network call. `decide.Normalize` rejects: nil state; empty questions; empty
question id; nil instructions; a `Choice` without options; a `Score`
with fewer than two levels; a JSON number or boolean where only text
kinds are accepted; an invalid image, and an image the backend's
`ImageSupport` does not carry (see [Images](#images)).

`decide.Normalize(state, qs, support)` returns a `decide.Request` and an error.
`Request.State` is the state's marshalled JSON; `Request.Questions` holds
the questions in the root `llmkit.DecisionQuestion` vocabulary, sorted by id;
`Request.Images` holds the images found in them.

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
already done. An already-cancelled `ctx` with a question that fails pre-wire
validation returns `ErrInvalidRequest`. A per-attempt `RequestTimeout`
expiring under a live parent is not a parent cancellation and is retried
like any other `ErrServer`.

For a non-200 status, `APIError.Message` carries the host body text; for a
transport or redirect failure, it carries the error's text. A `Message` of
either kind longer than 200 bytes is cut at the last rune boundary at or
before byte 200 and gets `...` appended. llmkit never places the API key
into an error, and `Ask` masks an echo of the `Secret` or the `BaseURL`
password in the error it returns; the `decide/systemone` package documentation gives
the forms covered.

## Usage and observability

TypeSafe bills input tokens only; output tokens are free
([Models](https://docs.typesafe.ai/models.md); no vendor review date on the
page; verified 2026-09-19). `Response.Usage` reports both counts.

`systemone.Client` emits no events itself. Wrap any `Decider` in
`decide.Observe(d, obs)` to record each `Ask` as one
[`llmkit.DecisionEvent`](https://pkg.go.dev/github.com/dpoage/llmkit#DecisionEvent)
(`llmkit.KindDecision`) on the `Ask`'s context, so inside an agent `Runner`
turn the event carries the run's `RunID` and the turn's `Step`:

```go
client, _ := systemone.New(systemone.Config{Secret: secret, Model: "jev-latest"})
d := decide.Observe(client, sink) // sink is any llmkit.Observer
```

An `Ask` that fails `decide.Normalize` is passed to the wrapped `Decider`
untouched and emits nothing. Every other `Ask` emits exactly one event,
success or failure, including a caller's already-cancelled ctx (`Err` set,
zero wire hits). The event carries `State` and `Questions` as validated,
`Answers`, `Usage`, and the `Ask`'s wall time as `Duration`; `Questions`
and `Answers` are each sorted by ID. On failure `Err` is the error's text,
`Answers` is nil, and `Usage` is zero. `Observe` returns the wrapped
`Decider`'s `Response` and error exactly, and `Observe(d, nil)` returns `d`.

`Observe` reads an optional `Identity() llmkit.Identity` method from the
`Decider`: `Backend` is its `Provider` (`"systemone"` for `*systemone.Client`),
`Model` is the response's reported model on success (the identity's `Model`
when the response names none) and the identity's `Model`, the requested
alias, on failure. The wrapper implements `Identity()` too when the
wrapped `Decider` does. A `Decider` without it leaves `Backend` empty.
`Observe` does not scrub: `Err` is the error text the backend returned, so
a backend must mask secrets in its errors, as `systemone` does.

### Replay

`decide.Replay(src, run)` builds a `Decider` from the `KindDecision` events
`src` (any `llmkit.Source`) holds for `run`, so a recorded policy replays
with no network. Paired with `agent.NewReplayClient`, a run whose
`ToolPolicy` asked a `Decider` replays offline:

```go
rc, _ := agent.NewReplayClient(src, run, caps)
rd, _ := decide.Replay(src, run)
runner := agent.NewRunner(rc, rc.Tools(), system, agent.WithToolPolicy(myPolicy(rd)))
```

Each `Ask` follows these rules:

1. `decide.Normalize` runs first; a refusal is returned unchanged and
   consumes nothing. Its `Provider` is `"decide"`, so its text reads
   `llmkit: decide error: ...`, where the live client's reads
   `llmkit: systemone error: ...`.
2. The `Ask` matches the earliest unconsumed recorded decision with the same
   `Step` and the same set of question ids, and consumes it. Two `Ask`s of
   one step with different id sets match whichever order they arrive in.
3. No match returns an error wrapping `decide.ErrReplayDiverged` that names
   the step.
4. A matched record that failed returns an error wrapping
   `decide.ErrReplayedFailure` with the recorded error text. The recorded
   error's type is not kept, so it carries no root `llmkit` sentinel.
5. Otherwise the recorded answers are rebuilt into a `Response` (by the kind
   of each asked question, probabilities included) and must pass
   `decide.Check`; a failure wraps `ErrReplayDiverged`.

Two limits to know. Replay does not compare the state: an `Ask` that
judges a different state than the recorded one still gets the recorded
answer, so compare states yourself if that matters. And it does not report
recorded decisions the replay never asked for; a replay that stops early is
not an error here (`ReplayClient.Err` still reports unreached completions
and tool calls). `decide.ErrReplayDiverged` is a different sentinel from
`agent.ErrReplayDiverged`; match each with `errors.Is`. Besides the wrapped
`Decider`'s errors and a `Normalize` refusal, a `Decider` from this package
returns only errors wrapping `ErrReplayDiverged` or `ErrReplayedFailure`.
A `Decider` from `Replay` is safe for concurrent use.

## Gating an agent

A `Decider` can drive an `agent.Runner`'s policies: a `ToolPolicy` that asks
whether a tool call is destructive, and a `RequestPolicy` that classifies the
latest user turn. The policy code depends only on `decide.Decider`, so the
hosted `systemone` client and any other backend are interchangeable.

```go
// Any decide.Decider works here; systemone.New builds the hosted one.
var d decide.Decider = exBeliefs{"rm": 0.9, "read_file": 0.1}

// Charge the decider's tokens to the pool the Runner already shares.
pool := agent.NewBudgetPool(10_000)
d = decide.Observe(d, llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
	if ev.Kind == llmkit.KindDecision && ev.Decision != nil && ev.Decision.Err == "" {
		pool.Add(ev.Decision.Usage.ChargeableTokens(1))
	}
}))

policy := agent.ToolPolicyFunc(func(ctx context.Context, call *llmkit.ToolCall) error {
	state := map[string]any{"tool": call.Name, "args": call.Arguments}
	resp, err := d.Ask(ctx, state, decide.Questions{"destructive": decide.Noul{
		Instructions: "Does this tool call delete or overwrite data outside the workspace?",
	}})
	if err != nil {
		return err // fail closed: no decision, no call
	}
	belief, err := resp.Noul("destructive")
	if err != nil {
		return err
	}
	if belief >= 0.5 {
		fmt.Println("denied", call.Name)
		return fmt.Errorf("blocked: belief %.2f", belief)
	}
	return nil
})

client := &exScript{resps: []llmkit.Response{
	{
		StopReason: llmkit.StopToolUse,
		Usage:      llmkit.Usage{InputTokens: 10, OutputTokens: 5},
		ToolCalls: []llmkit.ToolCall{
			{ID: "c1", Name: "rm", Arguments: json.RawMessage(`{"path":"/var/data"}`)},
			{ID: "c2", Name: "read_file", Arguments: json.RawMessage(`{"path":"notes.txt"}`)},
		},
	},
	{Text: "done", StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 15, OutputTokens: 5}},
}}
runner := agent.NewRunner(client, []agent.Tool{exTool("rm"), exTool("read_file")}, "You tidy files.",
	agent.WithToolPolicy(policy), agent.WithBudgetPool(pool))

out, err := runner.Run(context.Background(), "Remove the old data and read the notes.")
if err != nil {
	fmt.Println("run:", err)
	return
}
fmt.Println(out.FinalText, pool.Spent())
```

The snippet is the body of `ExampleDecider_gatingAnAgent` in
`decide/example_test.go`, compiled and run with `go test ./decide/`, and
`TestDocsGatingSnippetMatchesExample` fails when the two differ. `exBeliefs`,
`exScript`, and `exTool` are the fakes defined next to it: a `Decider`, an LLM
client, and a tool. The run prints `denied rm`, `ran read_file`, and
`done 75`: the `rm` call never ran, and the pool holds 35 LLM tokens plus 40
decision tokens.

`examples/agent-guardrail` is the runnable version, against a real LLM and
System One. It reads the `LLMKIT_*` variables of the other examples plus
`LLMKIT_SYSTEMONE_API_KEY` and `LLMKIT_SYSTEMONE_MODEL` (and optionally
`LLMKIT_SYSTEMONE_BASE_URL`), and takes `--threshold` (default 0.5) and
`--budget` (the shared token limit):

```bash
go run ./examples/agent-guardrail --threshold 0.5 --budget 200000
```

The program checks only that `LLMKIT_SYSTEMONE_API_KEY` and
`LLMKIT_SYSTEMONE_MODEL` are non-empty. A key with leading or trailing
whitespace passes that check and fails in `systemone.New`. It prints
beliefs at two decimals, so a belief just below the threshold can print
as equal to it, such as `0.50` for `0.49999999`.

Design points the program follows:

- **Two `Ask`s, not one.** The request policy fires before each wire call and
  the tool policy once per tool call, so their questions never share a hook.
  One `Ask` carries several questions only where they share a hook.
- **Read the belief through the accessor.** `resp.Noul(id)` fails with
  `decide.ErrNotAsked` on a mistyped id; indexing `resp.Nouls` reads 0, which
  the threshold treats as safe.
- **Fail closed.** A failed `Ask` denies the call or aborts the run.
- **One budget.** `agent.WithBudgetPool(pool)` charges the LLM's completions.
  The observer on `decide.Observe` charges the decider's. The run stops with
  `agent.TruncBudgetPool` at the first turn boundary after combined spend
  reaches the limit, so it can overshoot the limit by one turn. A run whose
  overshooting turn is its last ends normally, without `TruncBudgetPool`.
  `agent.Limits.TokenBudget` counts completions only, so it does not see
  decision tokens.
- **Replay.** Wrap the decider in `decide.Observe` with a recording sink and
  replay the run with `decide.Replay` ([Replay](#replay)); without that, a
  replayed run asks the policies' decider live again.

## TypeSafe limits and jaggedness

- A request may total at most 64k tokens, and state plus the longest question
  at most 32k. `systemone` does not count tokens; oversized requests surface as
  host 4xx errors. ([Models](https://docs.typesafe.ai/models.md); no vendor
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
go test -tags live -count=1 ./decide/systemone/ -v
```

A keyless run skips with a message naming `LLMKIT_LIVE_TYPESAFE_API_KEY`, and
the package prints one `LIVE_TOKENS` total line at exit. Add `-update` to
re-record the fixtures under `decide/systemone/testdata/`; treat that as a deliberate
local operation. The lane cases and the skip rules follow
[testing](testing.md).

The recorded fixtures replay hermetically: `decide/systemone/fixture_replay_test.go`
runs in plain `go test ./...` with no tag, no network, and no credentials.

`examples/decide` is the runnable one-Ask program:

```bash
export LLMKIT_SYSTEMONE_API_KEY="..."
export LLMKIT_SYSTEMONE_MODEL=jev-latest
go run ./examples/decide
```

It reads `LLMKIT_SYSTEMONE_API_KEY` and `LLMKIT_SYSTEMONE_MODEL` (both
required) and `LLMKIT_SYSTEMONE_BASE_URL` (optional; set it to target a
Kev server or a gateway). The live lane variables keep the `LLMKIT_LIVE_TYPESAFE_*`
names. Without the required
variables it prints its usage and exits 1 without touching the network.

CI runs the lane in the nightly Live workflow, which reads the repo secret
`LLMKIT_LIVE_TYPESAFE_API_KEY`; see [testing](testing.md) for the workflow
details.

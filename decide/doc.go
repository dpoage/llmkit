// Package decide evaluates decisions with TypeSafe's Jev decision model (the
// System One API): given a state and a set of typed questions, [Client.Ask]
// returns calibrated scores and probability distributions.
//
// This is not a chat client. Jev has no messages, no tools, no streaming,
// and no text output, so the package does not implement [llmkit.Client] and
// shares no code path with the provider packages — only the root package's
// error types, retry configuration, usage types, and recorder.
//
// # One request, one decision
//
// [Client.Ask] posts one request to POST <BaseURL>/v1/systemone with an
// "Authorization: Bearer" header:
//
//	{"state": <string|object|array>, "model": "...", "questions": {<id>: ...}}
//
// and parses one response:
//
//	{"model": "<versioned id>", "answers": {<id>: ...}, "usage": {"input_tokens", "output_tokens"}}
//
// There is no batching endpoint, no streaming, and no GET /v1/models in this
// round: one Ask is one HTTP request.
//
// # Questions
//
// [Question] is a sealed interface: the only implementations are [Noul]
// (binary belief in [0, 1]), [Choice] (pick one option from a map of
// descriptions), and [Score] (rate against an ordered legend of at least two
// levels). The wire "type" discriminator is derived from the Go type;
// callers never write it.
//
// State, instructions, and descriptions accept Go strings, anything that
// marshals to a JSON object or array, and — for descriptions and criteria —
// null. JSON numbers and booleans are rejected before the request is sent:
// the vendor's open set is text-shaped ("Advanced: structure"), and a
// mis-typed value should fail in the caller's process, not as a vendor 422.
//
// # Answers
//
// [Response] splits answers by question kind so a noul cannot carry a
// confidence and a score's legend is slices, not string-keyed maps:
// [Response.Nouls] maps a noul question id to its 0..1 belief,
// [Response.Choices] to a [ChoiceAnswer] (selected option, per-option
// probabilities, confidence), and [Response.Scores] to a [ScoreAnswer] whose
// Levels and Probabilities are index-aligned slices ordered by the server's
// legend index ("0".."n-1"), with n equal to the number of levels asked for.
// Probabilities are passed through exactly as reported — no renormalization,
// no argmax recomputation.
//
// # Errors
//
// Every error from [Client.Ask] is an [*llmkit.APIError] with Provider
// "typesafe" whose Kind unwraps to one of the llmkit sentinels. HTTP
// statuses map as in docs/providers.md:
//
//	429                  -> ErrRateLimited (Retry-After honored)
//	401, 403             -> ErrAuth
//	413                  -> ErrContextTooLong
//	400                  -> ErrContextTooLong when the body reads like a
//	                        context-length error, else ErrInvalidRequest
//	422                  -> ErrInvalidRequest
//	529                  -> ErrOverloaded
//	other 5xx            -> ErrServer
//	other 4xx            -> ErrInvalidRequest
//	transport failure    -> ErrServer (StatusCode 0)
//
// A response that violates the answer contract — a missing or extra answer,
// a type that does not match its question, a sparse legend — is a server
// contract violation: Kind ErrServer with StatusCode 200. Pre-wire
// validation failures (nil state, empty question id, nil instructions, a
// Choice without options, a Score with fewer than two levels, a JSON number
// or boolean where only text kinds are accepted) never touch the network
// and wrap llmkit.ErrInvalidRequest. Error messages carry the vendor body
// text truncated to 200 characters; llmkit never places the API key into an
// error.
//
// # Retries
//
// Ask retries 429, 529, other 5xx, and timeout-classified transport errors
// through the shared internal loop: each attempt runs under a per-attempt
// RequestTimeout deadline, a server Retry-After replaces the exponential
// backoff and is capped at MaxDelay, and parent cancellation is always
// terminal. Unset knobs resolve at construction to the decide defaults — 3
// attempts and a 30s per-attempt timeout (Jev answers in under a second) —
// with BaseDelay and MaxDelay from [llmkit.DefaultRetryConfig] and Jitter
// taken literally (explicit 0 means no jitter).
//
// # Vendor limits and jaggedness
//
//   - A request may total at most 64k tokens, and state plus the longest
//     question at most 32k. The package does not count tokens; oversized
//     requests surface as vendor 4xx errors (ErrInvalidRequest or
//     ErrContextTooLong).
//   - State and every question field are text-only: no images, no tools, no
//     generated prose.
//   - Model aliases move: "jev-latest" and "jev-preview" point at a versioned
//     id (jev-1.13.0 today) that can change between releases, and the
//     response's Model reports the versioned id actually used. Ledger on
//     that reported Model, not on the alias that was sent.
//   - Billing counts input tokens only; output tokens are free.
//
// # Concurrency
//
// A *Client is safe for concurrent use; each Ask is independent. The
// optional [Config.Recorder] fires exactly once per successful Ask with
// Provider "typesafe" and the response's reported Model, and never on
// failure.
//
// # Construction
//
// [New] validates the config and returns an error wrapping
// llmkit.ErrInvalidRequest for an APIKey that is empty, whitespace-only, or
// whitespace-padded (the same rule as provider.Spec.Secret), an empty Model,
// and a Retry.Jitter outside [0, 1]. New performs no network I/O and no
// environment lookups, so construction is hermetic.
package decide

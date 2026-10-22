// Package systemone is the System One wire backend for [decide]: it
// answers a state and a set of typed questions over HTTP, with calibrated
// scores and probability distributions where the host reports them.
// [Client] satisfies [decide.Decider].
//
// System One is the decision API TypeSafe publishes for its Jev model. Any
// server that speaks the same envelope at POST /v1/systemone works behind
// [Config.BaseURL]; the default host is https://api.typesafe.ai. The model
// is not a chat model: it has no messages, tools, streaming, or text
// output, so this package does not implement [llmkit.Client] and shares no
// code path with the provider packages. It uses only the root package's
// error types, retry configuration, and usage types.
//
// # Hosts
//
// The wire is identical across hosts; behavior differs as follows.
//
//   - TypeSafe Jev (https://api.typesafe.ai) reports the versioned model id
//     it used ("jev-1.13.0" for the alias "jev-latest") and bills input
//     tokens only.
//   - Kev (github.com/jaredpalmer/kev) echoes the requested alias
//     ("kev-latest") as the response model. It accepts questions without
//     instructions and scores with one level; [decide.Normalize] still
//     requires instructions and at least two levels, so such a question
//     is refused before the request. It exposes GET /v1/models, reports
//     probabilities at 4 decimals, and adds a latency_ms field that this
//     package ignores. A local kev.serve needs no auth, but [New] still
//     requires a non-empty Secret.
//
// # One request, one decision
//
// [Client.Ask] sends one HTTP POST to <BaseURL>/v1/systemone with an
// "Authorization: Bearer" header and this JSON body:
//
//	{"state": <string|object|array>, "model": "...", "questions": {<id>: ...}}
//
// The client parses one HTTP response:
//
//	{"model": "<model id>", "answers": {<id>: ...}, "usage": {"input_tokens", "output_tokens"}}
//
// The protocol has no batching endpoint and no streaming. One Ask is one
// logical request; retries may put several HTTP requests on the wire (see
// Retries). Questions, answers, and the validation rules are the
// vocabulary of package decide.
//
// # Errors
//
// Every error from [Client.Ask] is an [*llmkit.APIError] with Provider
// "systemone" (a validation refusal from [decide.Normalize] is re-tagged
// from "decide"), except the caller's cancellation or a deadline it set,
// which is an error chaining the context error (errors.Is(err,
// ctx.Err()); never retryable — see Retries). An *llmkit.APIError's Kind
// unwraps to one of the llmkit sentinels. HTTP statuses map as in
// docs/providers.md:
//
//	429                        -> ErrRateLimited (Retry-After honored)
//	401, 403                   -> ErrAuth
//	413                        -> ErrContextTooLong
//	400                        -> ErrContextTooLong when the body reads like
//	                              a context-length error, else ErrInvalidRequest
//	422                        -> ErrInvalidRequest
//	529                        -> ErrOverloaded
//	any other status >= 500    -> ErrServer
//	any other 4xx              -> ErrInvalidRequest
//	any other non-200, < 400   -> ErrServer (a 3xx or unexpected 2xx the
//	                              host never documents a Type for)
//	transport failure          -> ErrServer (StatusCode 0)
//	redirect-policy failure    -> ErrInvalidRequest (StatusCode 0), terminal,
//	                              not retried: the client stopped after 10
//	                              redirects, or Config.HTTPClient's
//	                              CheckRedirect returned an error
//
// A response that violates the answer contract — a missing model or
// usage block, a missing or extra answer, a type that does not match
// its question, a sparse legend — is a server contract violation: Kind
// ErrServer with StatusCode 200, retried like any other ErrServer (each
// retry re-sends the same request). Pre-wire validation failures (nil
// state, empty question id, nil instructions, a Choice without options,
// a Score with fewer than two levels, a JSON number or boolean where
// only text kinds are accepted) never touch the network and wrap
// llmkit.ErrInvalidRequest, terminal. For a non-200 status, Message
// carries the host body text; for a transport or redirect failure, it
// carries the error's text. A Message of either kind longer than 200
// bytes is cut at the last rune boundary at or before byte 200 and gets
// "..." appended.
//
// Ask masks an echo of [Config.Secret] or the [Config.BaseURL] password in
// the error it returns, in every error under it, and in the Err of an event
// [decide.Observe] emits. The echo may be as sent, escaped by encoding/json,
// or Go-quoted (%q), in a host body, a contract-violation message, a
// transport error, or the "(last attempt: ...)" text of a cancelled retry.
// The mask is "***" when neither credential holds '*'. Credentials are
// masked as values, so a short one also masks matching characters
// elsewhere in server or transport text; text llmkit writes itself, such as
// the "llmkit: systemone error (status 500):" prefix, is not masked. The
// URL username is replaced only as userinfo ("scheme://user[:password]@").
// Redaction runs before the 200-byte cap; classification reads the
// unredacted text. A redaction cuts the error chain: the *llmkit.APIError
// keeps Kind, StatusCode, RetryAfter, HasRetryAfter and Provider, and
// errors.Is still matches context.Canceled and context.DeadlineExceeded,
// but the transport error, net.Error included, is unreachable. An error
// with nothing to redact is returned intact.
//
// # Retries
//
// Ask retries every *llmkit.APIError [llmkit.Classify] marks retryable —
// ErrRateLimited, ErrOverloaded, and every ErrServer (any status 500 or
// above, a sub-400 non-200 status, a transport failure other than a
// redirect-policy failure, a 200 answer-contract violation) — through the
// shared retry loop, [retry.Do]. A redirect-policy failure is terminal:
// ErrInvalidRequest with StatusCode 0, returned without a retry.
// Each attempt runs under a per-attempt RequestTimeout deadline; a
// stalled attempt is retried like any other ErrServer. For a non-200
// response the client parses the Retry-After header and carries its
// presence as [llmkit.APIError.HasRetryAfter]: when a retried non-200
// status carried the header, its delay — a present zero included —
// replaces the exponential backoff and is capped at MaxDelay; absence
// falls back to the schedule. A 200 that violates the answer contract
// always takes the schedule: the client does not read its headers. One
// Ask may therefore issue several HTTP requests.
//
// The caller's ctx bounds the whole call. Its cancellation, or a
// deadline it set, ends the loop immediately — whether that happens
// mid-attempt or while waiting between retries — without another
// attempt. The error Ask returns then chains the context error, so
// errors.Is(err, context.Canceled) or errors.Is(err,
// context.DeadlineExceeded) reaches it, and it is not an
// *llmkit.APIError. One exception: a terminal error keeps its identity
// even when the ctx is already done. An already-cancelled ctx with a
// question that fails pre-wire validation returns ErrInvalidRequest. A
// per-attempt RequestTimeout expiring under a live parent is not a
// parent cancellation and is retried like any other ErrServer.
//
// Unset knobs resolve at construction, via [retry.Config.Or], to the
// package defaults: 3 attempts and a 30s per-attempt timeout (TypeSafe
// Jev answers in under a second). BaseDelay, MaxDelay, and Jitter (20%)
// come from [retry.Default].
//
// # Host limits and jaggedness
//
//   - TypeSafe accepts at most 64k tokens per request, and state plus the
//     longest question at most 32k. The package does not count tokens.
//     Oversized requests surface as host 4xx errors
//     (ErrInvalidRequest or ErrContextTooLong).
//   - State and every question field are text-only: no images, no
//     tools, and no generated prose.
//   - Model aliases move on a host that versions its models: TypeSafe's
//     "jev-latest" and "jev-preview" point at a versioned id that can
//     change between releases. Response.Model reports the id the host
//     sent. Ledger on that reported Model, not on the alias sent. A host
//     that echoes the alias (Kev) gives no version.
//
// # Observability
//
// A *Client is safe for concurrent use, and each Ask is independent. A
// Client emits no events itself. Wrap it in [decide.Observe] to record each
// Ask as one [llmkit.DecisionEvent]; Observe reads [Client.Identity]
// (Provider "systemone", Model the configured alias) for the event's
// Backend and, on failure, Model. A validation refusal emits nothing; a
// caller's already-cancelled ctx emits one event with Err set and zero wire
// hits. [decide.Replay] answers from those events with no network.
//
// # Construction
//
// [New] validates the config and returns an error wrapping
// llmkit.ErrInvalidRequest for any of:
//
//   - a Secret that is empty, whitespace-only, or whitespace-padded (the
//     same rule as provider.Spec.Secret);
//   - an empty Model;
//   - a Retry.Jitter outside [0, 1], NaN included;
//   - a BaseURL that is not an http or https URL with a host (the scheme
//     is case-insensitive), or that contains '#' anywhere, an empty
//     fragment included.
//
// No such error echoes the Secret or any part of BaseURL, its userinfo
// included. A BaseURL that parses but cannot be dialed, such as
// "http://localhost:99999", is accepted and fails at request time as a
// retryable transport error. A BaseURL query is kept after the joined
// path: "https://gw.example/x?api-version=1" requests
// "/x/v1/systemone?api-version=1". The path is not cleaned: "//" and dot
// segments in BaseURL are not collapsed or resolved.
//
// New performs no network I/O and no environment lookups, so construction
// is hermetic.
package systemone

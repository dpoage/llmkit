// Package decide evaluates decisions with TypeSafe's Jev decision model
// (the System One API): given a state and a set of typed questions,
// [Client.Ask] returns calibrated scores and probability distributions.
//
// Jev is not a chat model. It has no messages, no tools, no streaming,
// and no text output, so this package does not implement [llmkit.Client]
// and shares no code path with the provider packages. It uses only the
// root package's error types, retry configuration, usage types, and
// observer vocabulary.
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
//	{"model": "<versioned id>", "answers": {<id>: ...}, "usage": {"input_tokens", "output_tokens"}}
//
// The vendor exposes no batching endpoint, no streaming, and no
// GET /v1/models. One Ask is one logical request; retries may put
// several HTTP requests on the wire (see Retries).
//
// # Questions
//
// [Question] is a sealed interface with three implementations:
// [Noul] (binary belief in [0, 1]), [Choice] (pick one option from a
// map of descriptions), and [Score] (rate against an ordered legend of
// at least two levels). The wire "type" discriminator comes from the
// Go type; callers never write it.
//
// State, instructions, and descriptions accept Go strings, anything
// that marshals to a JSON object or array, and — for descriptions and
// criteria — null. The client rejects JSON numbers and booleans before
// sending the request. The vendor's open set is text-shaped
// ("Advanced: structure"), and a mis-typed value must fail in the
// caller's process, not as a vendor 422.
//
// # Answers
//
// [Response] splits answers by question kind so a noul cannot carry a
// confidence and a score's legend is slices, not string-keyed maps.
// [Response.Nouls] maps a noul question id to its 0..1 belief.
// [Response.Choices] maps a choice question id to a [ChoiceAnswer]
// (selected option, per-option probabilities, confidence).
// [Response.Scores] maps a score question id to a [ScoreAnswer] whose
// Levels and Probabilities are index-aligned slices ordered by the
// server's legend index ("0".."n-1"), with n equal to the number of
// levels asked for. [Response] passes probabilities through exactly as
// the vendor reports them: no renormalization, no argmax recomputation.
// Response maps stay nil for question kinds the Ask did not include.
//
// # Errors
//
// Every error from [Client.Ask] is an [*llmkit.APIError] with Provider
// "typesafe", except the caller's cancellation or a deadline it set,
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
//	                              vendor never documents a Type for)
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
// carries the vendor body text; for a transport or redirect failure, it
// carries the error's text. A Message of either kind longer than 200
// bytes is cut at the last rune boundary at or before byte 200 and gets
// "..." appended.
//
// When the server or the transport echoes [Config.Secret] or the password
// of [Config.BaseURL] into an error, as sent, escaped by Go's encoding/json
// or Go-quoted (%q), Ask masks the echo in the error it returns, in every
// error under it, and in the Observer event's Err. That covers a vendor
// body, a contract-violation message that quotes an answer id or type, a
// transport error that quotes server bytes, and the last attempt's error
// in the "(last attempt: ...)" text of a cancelled retry. The mask is
// "***" when neither credential holds '*'. Credentials are masked as
// values, so a short Secret or password also masks the same characters
// elsewhere in the server's and the transport's text. Text llmkit writes
// itself, such as the "llmkit: typesafe error (status 500):" prefix or the
// context error of a cancelled retry, may still contain text that matches
// the Secret or the password, for example the characters of a short one.
// The URL username is not a value: only where it appears as URL userinfo
// ("scheme://user[:password]@") is it replaced. Redaction happens before
// the 200-byte cap, and classification reads the unredacted text, so the
// Kind is the one the vendor's text gives. When a redaction happens the
// error chain is cut: the *llmkit.APIError keeps Kind, StatusCode,
// RetryAfter, HasRetryAfter and Provider, and errors.Is still matches
// context.Canceled and context.DeadlineExceeded when they matched; the
// underlying transport error, net.Error included, is no longer reachable.
// An error with nothing to redact is returned as it was, chain intact.
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
// decide defaults: 3 attempts and a 30s per-attempt timeout (Jev
// answers in under a second). BaseDelay, MaxDelay, and Jitter (20%)
// come from [retry.Default].
//
// # Vendor limits and jaggedness
//
//   - A request may total at most 64k tokens, and state plus the
//     longest question at most 32k. The package does not count tokens.
//     Oversized requests surface as vendor 4xx errors
//     (ErrInvalidRequest or ErrContextTooLong).
//   - State and every question field are text-only: no images, no
//     tools, and no generated prose.
//   - Model aliases move. "jev-latest" and "jev-preview" point at a
//     versioned id (jev-1.13.0 today) that can change between releases.
//     Response.Model reports the versioned id the server actually
//     used. Ledger on that reported Model, not on the alias sent.
//   - Billing counts input tokens only; output tokens are free.
//
// # Observability
//
// A *Client is safe for concurrent use, and each Ask is independent. When
// [Config.Observer] is set, Ask emits exactly one [llmkit.DecisionEvent]
// per Ask that passes pre-wire validation — success or failure, none for
// a buildRequest refusal — with Backend "typesafe", Model (the
// response's reported model on success, the requested alias on
// failure), Usage, State, Questions, and Answers each sorted by ID. A
// caller's already-cancelled ctx emits one event with Err set and zero
// wire hits. decide never mints a span; the event rides whatever
// Run/Span/Step the Ask's ctx already carries.
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
package decide

// Package embed generates vector embeddings from text.
//
// # Embedders
//
// The [Embedder] interface produces vectors: [Embedder.Embed] embeds one
// text, [Embedder.EmbedBatch] embeds many. EmbedBatch results are
// index-aligned with the input: len(result) == len(texts) and result[i]
// embeds texts[i]. Duplicate texts are embedded once per occurrence.
// Implementations may split a batch into multiple requests; a failed
// request fails the whole call, with no partial results. On both
// backends, Embed of a text returns the same vector and the same error as
// EmbedBatch of a one-element slice that holds the text.
//
// Batching is capped by [Config.MaxBatch]. The zero value sends the whole
// batch in one HTTP request. A positive value splits the input into
// requests of at most MaxBatch texts. A negative value is rejected by
// [Config.Validate].
//
// # Backends
//
// [New] builds the backend named by [Config.Backend]: [BackendOllama]
// posts to <URL>/api/embed on an Ollama server, and
// [BackendOpenAICompatible] posts to <URL>/v1/embeddings on any
// OpenAI-compatible API. There is no default backend and no default URL:
// [Config.Validate] requires a known Backend and a non-empty
// [Config.URL]. A local Ollama server typically listens on
// http://localhost:11434. New is the only way to build a backend
// embedder in this package: both backend types and their constructors
// are unexported, so a caller always goes through New and the returned
// [Embedder] interface.
//
// The Ollama wire request always sends "input" as a JSON array, even for
// a single text ([Embedder.Embed] calls EmbedBatch with a one-element
// slice) — Ollama's /api/embed accepts a single string or a list, and a
// one-element list is the shape this package sends.
//
// When [Config.Dimensions] is zero, both backends detect the vector
// dimensionality from the first vector they accept, and reject any later
// vector of a different length. A zero-length vector during
// auto-detection fails with [ErrEmptyVector].
//
// The module does not bundle an in-process embedding runtime. To embed
// without a server, implement [Embedder] in your application.
//
// # Configuration
//
// [Config] holds every setting a backend honors: New reads no
// environment variables and no file — a caller fills Config from its own
// configuration source. [Config.Backend] selects the wire protocol
// ([ParseBackend] parses a string into it); [Config.Validate] rejects the
// zero Backend and any value other than [BackendOllama] or
// [BackendOpenAICompatible]. [Config.APIKey], when non-empty, is sent as
// an "Authorization: Bearer <APIKey>" header on every request to BOTH
// backends; when empty, neither backend sends an Authorization header.
// Validate also refuses an APIKey with leading or trailing whitespace,
// and an APIKey that holds a byte net/http cannot send in a header value
// (bytes 0x00-0x1F except tab, and 0x7F); the error never echoes the key,
// raw or trimmed. Validate rejects negative Dimensions and MaxBatch, and
// rejects Retry.Jitter outside [0, 1].
//
// # Retries
//
// Both backends turn a failed backend call into the kit's error
// vocabulary through the same normalization the chat adapters use. A
// non-200 response, a transport failure (dial, TLS, EOF while reading
// the body, or a stalled attempt reaped by the per-attempt timeout), and
// an openai-compatible 200 whose body carries an error object each
// become a [llmkit.APIError]. [llmkit.Classify] alone decides
// retryability. The retryable Kinds are [llmkit.ErrRateLimited],
// [llmkit.ErrServer], and [llmkit.ErrOverloaded]. They cover:
//
//   - HTTP 429.
//   - Any status 500 or above: 529 is ErrOverloaded, every other such
//     status is ErrServer.
//   - Any non-200 status below 400, such as a 204 or a 302 without a
//     Location header.
//   - A transport failure.
//   - An openai-compatible 200 error object whose type is missing,
//     unknown, or maps to a retryable Kind.
//
// A retry honors the server's Retry-After header when the response
// reaches the adapter with its headers intact. A non-200 response that
// the adapter reads all the way through carries Retry-After; a body read
// that fails before headers can be parsed does not (the result is
// {ErrServer, 0}). Every other APIError is terminal. A decode, count,
// index, or dimension error is a plain error, and Classify never retries
// a plain error. The caller's context cancellation is never an APIError:
// it surfaces as the plain context error and is always terminal.
//
// The per-attempt bound depends on [Config.HTTPClient]. When nil, the
// embedder uses a client with no [http.Client.Timeout], and
// Config.Retry.RequestTimeout is the only per-attempt bound. An injected
// client is used as-is, including its Timeout: an attempt then ends at the
// earlier of RequestTimeout and the client's Timeout. [Config.Retry] is
// completed at construction by [retry.Config.Or] against the embed
// defaults of 3 attempts, a 60s per-attempt timeout, and the kit's
// BaseDelay/MaxDelay/Jitter. Jitter 0 counts as unset like every other
// field. For the defaults without jitter, set Retry.Rand to a function
// that returns 0.5: every jitter factor is then exactly 1.
//
// # Errors
//
// A caller matches a backend failure with errors.Is against the kit's
// sentinels ([llmkit.ErrRateLimited], [llmkit.ErrAuth], [llmkit.ErrServer],
// ...) or errors.As against [llmkit.APIError] for the status code and
// Retry-After. Three routes produce an APIError:
//
//   - A non-200 response. StatusCode is the HTTP status, and the Kind
//     comes from the status. A 400 whose body reports a context-length
//     overflow is [llmkit.ErrContextTooLong]. A status below 400 is
//     ErrServer. Classification reads the body up to the 64 MiB limit
//     below, so a context-length phrase past the 200-byte Message cap
//     still counts.
//   - An openai-compatible 200 whose body carries an error object.
//     StatusCode is 200, and the Kind comes from the object's type field.
//     A missing or unknown type is ErrServer.
//   - A transport failure. StatusCode is 0 and the Kind is ErrServer. The
//     underlying error stays reachable through errors.Is.
//
// On the first two routes, Message is the server's text with surrounding
// whitespace trimmed, capped at 200 bytes plus "..." when the trimmed
// text is longer. On the transport route, Message is the underlying
// error's text unchanged: embed neither trims nor caps it.
//
// Every response body embed keeps is at most 64 MiB. If a 200 response
// body exceeds the limit, the call returns a plain error naming the limit
// — not an [llmkit.APIError], terminal under [llmkit.Classify] — and
// decodes nothing. If a non-200 response body exceeds the limit, embed
// keeps only its first 64 MiB: classification and the Message both use
// that truncated body, so a context-length phrase past 64 MiB does not
// count. A body read that fails part-way (a connection reset, an EOF
// before Content-Length) stays a retryable transport failure either way.
//
// A cancelled or expired caller context surfaces as the context error
// (errors.Is(err, context.Canceled) or context.DeadlineExceeded), never
// an APIError. [Config.Validate] and [New] reject a bad Config — an
// unknown or zero Backend, an empty Model or URL, a negative Dimensions
// or MaxBatch, an APIKey with leading or trailing whitespace or a byte
// net/http cannot send in a header value, or an out-of-range Retry.Jitter — with an error
// wrapping [llmkit.ErrInvalidRequest].
// [ErrEmptyVector] and a decode/count/dimension mismatch are plain
// errors: match the specific error, not the kit vocabulary.
//
// # Caching
//
// [NewCachedEmbedder] wraps any [Embedder] with an in-memory content-hash
// cache. New never applies one itself: to cache a backend, pass the
// Embedder that New returns to NewCachedEmbedder. Cache keys are
// SHA-256(model + "\x00" + text), so different models never share
// entries for the same text. The cache evicts the least recently used
// entry past a positive capacity; a capacity of zero or less means
// unbounded. All methods are safe for concurrent use. Every vector
// crossing the cache boundary is copied: a lookup returns a private copy
// so a caller's mutation cannot corrupt a later read, and an insert
// stores a private copy so a later mutation by the caller who computed
// the vector cannot corrupt what the cache serves next.
//
// # Observing
//
// [Observe] wraps any Embedder so every Embed and EmbedBatch call reports
// one [llmkit.Event] (KindEmbed) to an [llmkit.Observer] and otherwise
// behaves exactly like the embedder it wraps: vectors, errors,
// Dimensions, and ModelName pass through unchanged. The event carries the
// model, the requested input count, the vector dimensionality, the call's
// duration, and the error; CacheHits and Usage stay zero — the Embedder
// interface exposes neither per-call cache attribution nor token usage.
// Run and span ids come from the call's context ([llmkit.WithRun],
// [llmkit.WithSpan]), so embeddings correlate with the rest of a run's
// events. See [Observe] for the exact field contract.
package embed

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Embedder generates vector embeddings from text.
type Embedder interface {
	// Embed returns the embedding vector for one text.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch returns embedding vectors for multiple texts. Results
	// are index-aligned with the input: len(result) == len(texts) and
	// result[i] embeds texts[i]. Duplicate texts are not deduplicated:
	// each occurrence is embedded independently. Implementations may
	// split the batch into multiple requests; a failed request fails the
	// whole call with no partial results.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the dimensionality of the vectors produced by the
	// underlying model. For an Embedder built by [New] with
	// Config.Dimensions set, it returns that value from construction. With
	// Config.Dimensions 0, it returns 0 before the first call and the
	// served vector length after the first successful call. A call that
	// fails may or may not have recorded a value: the embedder records
	// the length of the first vector it accepts, even when a later vector
	// or a later batch chunk of the same call is rejected.
	Dimensions() int

	// ModelName returns the identifier of the model used for embedding.
	ModelName() string
}

// New validates cfg and builds the Embedder for cfg.Backend. New applies
// no cache: a caller who wants one wraps the result with
// [NewCachedEmbedder]. Every refusal (a bad Config or an unknown
// Backend) wraps [llmkit.ErrInvalidRequest] and never echoes cfg.APIKey.
func New(cfg Config) (Embedder, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	var codec wireCodec
	switch cfg.Backend { // Validate admitted exactly these two.
	case BackendOllama:
		codec = ollamaCodec{}
	case BackendOpenAICompatible:
		codec = openaiCodec{}
	}

	return newBackendEmbedder(cfg, codec), nil
}

// batchChunkSize caps remaining at maxBatch when chunking is enabled
// (maxBatch > 0); otherwise the whole remainder goes in one request.
func batchChunkSize(remaining, maxBatch int) int {
	if maxBatch <= 0 || remaining < maxBatch {
		return remaining
	}
	return maxBatch
}

// ErrEmptyVector reports a zero-length embedding vector. Backends return
// this error, wrapped with backend context, when dimension auto-detection
// (Config.Dimensions == 0) encounters such a vector: it carries no
// dimensional information. Test with errors.Is.
var ErrEmptyVector = errors.New("empty embedding vector")

// checkDimensions enforces vector-dimension consistency for a converted
// response. expected is cfg.Dimensions when > 0; otherwise it is
// detected from the first vector and recorded under *dims (if still
// zero). Every vector in every call must match, so a later mismatch
// (configured or against the first detected value) is an error.
func checkDimensions(backend string, out [][]float32, dims *int, mu *sync.RWMutex) error {
	mu.Lock()
	defer mu.Unlock()
	expected := *dims
	if expected == 0 && len(out) > 0 {
		if len(out[0]) == 0 {
			return fmt.Errorf("%s: vector at index 0: %w", backend, ErrEmptyVector)
		}
		expected = len(out[0])
		*dims = expected
	}
	for i, v := range out {
		if len(v) != expected {
			return fmt.Errorf("%s: embedding dimension mismatch at index %d: got %d, want %d", backend, i, len(v), expected)
		}
	}
	return nil
}

package llmkit

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Sentinel errors callers can match with errors.Is. Adapters normalize
// provider-specific failures into these where the cause is recognizable.
var (
	// ErrRateLimited indicates a 429 / rate-limit response. The retry wrapper
	// treats it as retryable and honors any Retry-After it carries.
	ErrRateLimited = errors.New("llmkit: rate limited")
	// ErrAuth indicates an authentication/authorization failure (401/403). Not
	// retryable.
	ErrAuth = errors.New("llmkit: authentication failed")
	// ErrContextTooLong indicates the request exceeded the model's context
	// window (often surfaced as a 400 with a recognizable message, or a 413).
	// Not retryable without shrinking the request.
	ErrContextTooLong = errors.New("llmkit: context too long")
	// ErrInvalidRequest indicates a malformed request the provider rejected
	// (400, other than context-too-long). Not retryable.
	ErrInvalidRequest = errors.New("llmkit: invalid request")
	// ErrServer indicates a transient server-side failure (5xx). Retryable.
	ErrServer = errors.New("llmkit: server error")
	// ErrOverloaded indicates the provider is temporarily overloaded (e.g. 529).
	// Retryable.
	ErrOverloaded = errors.New("llmkit: provider overloaded")
)

// APIError is the normalized error type adapters return. It wraps one of the
// sentinel errors (accessible via errors.Is / the Kind field), preserves the
// HTTP status code and any Retry-After hint, and chains the underlying SDK
// error for debugging. llmkit never inserts the credential or the request
// body into these fields; vendor error text passes through as the provider
// sent it, so a provider that echoes credentials into its error output is
// outside llmkit's control.
type APIError struct {
	// Kind is the sentinel error this maps to (ErrRateLimited, ErrAuth, ...).
	Kind error
	// StatusCode is the HTTP status of the response. A 0 means no usable
	// HTTP status: a transport failure (dial, reset, per-attempt timeout)
	// carries Kind ErrServer; a response the adapter could not decode also
	// carries Kind ErrServer; a pre-wire refusal carries Kind
	// ErrInvalidRequest. Kind is the discriminator — a 0 status with
	// ErrServer is retryable, a 0 status with ErrInvalidRequest is not.
	// The kit's adapters report a caller's cancelled context as a plain
	// error chaining context.Canceled, never an *APIError. When the
	// caller's context is done as the retry stage returns ([retry.Do]),
	// a retryable *APIError never comes back: it is replaced by a plain
	// error chaining ctx.Err() that carries its text — an ErrServer
	// transport failure cut short by the caller's deadline included. A
	// terminal *APIError (a 401, say) is returned as-is.
	StatusCode int
	// RetryAfter is the server-suggested wait before retrying. The header
	// is parsed on every status, not only 429/529, but honoured only when
	// Kind is retryable ([Classify]). It is meaningful only when
	// HasRetryAfter is true; a present zero means retry immediately. The
	// Google SDK hides response headers, so a Google error never carries
	// one.
	RetryAfter time.Duration
	// HasRetryAfter reports that the response carried a Retry-After header;
	// RetryAfter is meaningful only when it is true, and a present zero
	// means retry immediately.
	HasRetryAfter bool
	// Provider names the backend that produced the error (e.g. "anthropic").
	Provider string
	// Message is a short description: either fixed llmkit text or text
	// extracted from the provider's error. llmkit never inserts the
	// credential into it.
	Message string
	// Err is the underlying vendor-SDK error, for Unwrap chaining. It may
	// carry provider response detail. llmkit never inserts the credential
	// into it; vendor error bodies pass through verbatim.
	Err error
}

func (e *APIError) Error() string {
	if e.StatusCode != 0 {
		return fmt.Sprintf("llmkit: %s error (status %d): %s", e.Provider, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("llmkit: %s error: %s", e.Provider, e.Message)
}

// Unwrap returns the Kind sentinel (for errors.Is) and, when set, the
// underlying SDK error so callers can reach provider detail. Go 1.20+.
func (e *APIError) Unwrap() []error {
	if e.Err != nil {
		return []error{e.Kind, e.Err}
	}
	return []error{e.Kind}
}

// Classify is the kit's retryability rule, in retry.Do's classify shape:
// one classifier for every error Complete, Stream, and the retry stage
// return. Row order is precedence — cancellation is decided before Kind.
//
//   - nil, or an error chaining context.Canceled (the caller's context, at
//     any depth): not retryable. A retry would run against a dead context,
//     and this holds even when the error is an *APIError wrapping
//     Canceled.
//   - *APIError with Kind ErrRateLimited, ErrServer, or ErrOverloaded:
//     retryable, honouring the server's Retry-After when HasRetryAfter is
//     true (a present zero delay retries immediately).
//   - every other *APIError Kind, and every non-APIError error (including
//     a bare context.DeadlineExceeded): not retryable.
//
// When [retry.Do] runs with Classify and the caller's context is done as
// it returns, the error is never retryable here: Do replaces a retryable
// last error with a plain error chaining ctx.Err() — terminal by the
// Canceled row for a cancelled context, by the last row for a deadline —
// and returns a terminal one as-is.
func Classify(err error) (retryAfter time.Duration, hasRetryAfter, retryable bool) {
	if err == nil || errors.Is(err, context.Canceled) {
		return 0, false, false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		switch apiErr.Kind {
		case ErrRateLimited, ErrServer, ErrOverloaded:
			return apiErr.RetryAfter, apiErr.HasRetryAfter, true
		default:
			return 0, false, false
		}
	}
	return 0, false, false
}

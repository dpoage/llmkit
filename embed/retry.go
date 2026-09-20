package embed

import (
	"errors"
	"fmt"
	"github.com/dpoage/llmkit/retry"
	"net"
	"net/http"
	"time"
)

// statusError reports a non-200 HTTP response from a backend call. It carries
// the parsed Retry-After header so the retry loop can honor it; the backend
// prefix is part of Error so messages keep their "ollama: HTTP 429: ..." shape.
type statusError struct {
	backend       string
	status        int
	body          string
	retryAfter    time.Duration // valid only when hasRetryAfter
	hasRetryAfter bool
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s: HTTP %d: %s", e.backend, e.status, e.body)
}

// retryAfterDuration returns the server-supplied Retry-After delay and
// whether one was present in the response.
func (e *statusError) retryAfterDuration() (time.Duration, bool) {
	return e.retryAfter, e.hasRetryAfter
}

func newStatusError(backend string, status int, retryAfterHeader, body string) *statusError {
	se := &statusError{backend: backend, status: status, body: truncate(body, 200)}
	if d, ok := retry.ParseRetryAfter(retryAfterHeader, time.Now()); ok {
		se.retryAfter, se.hasRetryAfter = d, true
	}
	return se
}

// retryable classifies an error from a backend call as worth retrying.
// Transient: HTTP 429 and 5xx (honoring Retry-After when supplied), and
// network errors that are timeouts or marked temporary. Everything else —
// other 4xx, decode failures, context cancellation — is terminal.
func retryable(err error) (delay time.Duration, hasDelay bool, ok bool) {
	var se *statusError
	if errors.As(err, &se) {
		if se.status == http.StatusTooManyRequests || se.status >= 500 {
			if d, present := se.retryAfterDuration(); present {
				return d, true, true
			}
			return 0, false, true
		}
		return 0, false, false
	}
	var ne net.Error
	if errors.As(err, &ne) && (ne.Timeout() || ne.Temporary()) {
		return 0, false, true
	}
	return 0, false, false
}

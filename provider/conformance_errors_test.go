package provider

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// completeExpectingError drives one Complete against a server answering with
// (status, body) and returns the normalized error.
func completeExpectingError(t *testing.T, f adapterFactory, status int, body string, header http.Header) error {
	t.Helper()
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		for k, vs := range header {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	})
	client := f.build(t, base)
	_, err := client.Complete(context.Background(), simpleRequest())
	if err == nil {
		t.Fatalf("%s: expected error for status %d, got nil", f.name, status)
	}
	return err
}

// mustBeAPIError narrows err to *llmkit.APIError and checks its Kind and
// StatusCode.
func mustBeAPIError(t *testing.T, err error, f adapterFactory, kind error, status int) *llmkit.APIError {
	t.Helper()
	if !errors.Is(err, kind) {
		t.Errorf("%s: error = %v, want %v", f.name, err, kind)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("%s: error is not *llmkit.APIError: %T", f.name, err)
	}
	if apiErr.StatusCode != status {
		t.Errorf("%s: StatusCode = %d, want %d", f.name, apiErr.StatusCode, status)
	}
	return apiErr
}

// TestConformance_ErrorNormalization_ContextTooLong413 asserts a 413 maps to
// ErrContextTooLong on every adapter, with the status preserved on the
// APIError.
func TestConformance_ErrorNormalization_ContextTooLong413(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, http.StatusRequestEntityTooLarge,
				errorBody(f.name, http.StatusRequestEntityTooLarge, "request too large"), nil)
			mustBeAPIError(t, err, f, llmkit.ErrContextTooLong, http.StatusRequestEntityTooLarge)
		})
	}
}

// TestConformance_ErrorNormalization_ContextTooLong400Heuristic asserts the
// 400 disambiguation: a message that looks like context-window overflow
// promotes the error to ErrContextTooLong, while an ordinary 400 stays
// ErrInvalidRequest. The control case is what makes the heuristic
// testable — without it, a classifier that maps every 400 to
// ErrContextTooLong would pass.
func TestConformance_ErrorNormalization_ContextTooLong400Heuristic(t *testing.T) {
	tooLongMsg := "This model's maximum context length is 128000 tokens. However, your messages resulted in 200000 tokens. Please reduce the length of the messages."
	cases := []struct {
		name    string
		message string
		kind    error
	}{
		{"heuristic hit", tooLongMsg, llmkit.ErrContextTooLong},
		{"ordinary 400 stays invalid request", "Invalid parameter: temperature must be between 0 and 2", llmkit.ErrInvalidRequest},
	}
	for _, f := range allAdapters() {
		for _, tc := range cases {
			t.Run(f.name+"/"+tc.name, func(t *testing.T) {
				err := completeExpectingError(t, f, http.StatusBadRequest,
					errorBody(f.name, http.StatusBadRequest, tc.message), nil)
				mustBeAPIError(t, err, f, tc.kind, http.StatusBadRequest)
			})
		}
	}
}

// TestConformance_ErrorNormalization_Server500 asserts a plain 500 maps to
// ErrServer (retryable) on every adapter.
func TestConformance_ErrorNormalization_Server500(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, http.StatusInternalServerError,
				errorBody(f.name, http.StatusInternalServerError, "internal error"), nil)
			mustBeAPIError(t, err, f, llmkit.ErrServer, http.StatusInternalServerError)
		})
	}
}

// TestConformance_ErrorNormalization_Overloaded529 asserts Anthropic's
// non-standard 529 (overloaded) maps to ErrOverloaded on every adapter.
func TestConformance_ErrorNormalization_Overloaded529(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, 529,
				errorBody(f.name, 529, "overloaded"), nil)
			mustBeAPIError(t, err, f, llmkit.ErrOverloaded, 529)
		})
	}
}

// TestConformance_RetryAfterHTTPDate asserts the HTTP-date form of
// Retry-After is parsed at the adapter boundary (the seconds form is covered
// by TestConformance_ErrorNormalization_RateLimit). genai's APIError carries
// no *http.Response, so Google reports RetryAfter 0 by documented contract.
func TestConformance_RetryAfterHTTPDate(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			// A date ~90s out: strictly in the future (so a correct parser
			// yields a positive duration) but far from any test timeout.
			when := time.Now().UTC().Add(90 * time.Second)
			h := http.Header{}
			h.Set("Retry-After", when.Format(http.TimeFormat))
			err := completeExpectingError(t, f, http.StatusTooManyRequests,
				errorBody(f.name, http.StatusTooManyRequests, "slow down"), h)

			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("%s: error is not *llmkit.APIError: %T", f.name, err)
			}
			if f.name == "google" {
				if apiErr.RetryAfter != 0 {
					t.Errorf("google RetryAfter = %v, want 0 (genai exposes no http.Response)", apiErr.RetryAfter)
				}
				return
			}
			if apiErr.RetryAfter <= 0 {
				t.Errorf("%s RetryAfter = %v, want >0 (HTTP-date %q should parse)", f.name, apiErr.RetryAfter, when.Format(http.TimeFormat))
			}
			if apiErr.RetryAfter > 2*time.Minute {
				t.Errorf("%s RetryAfter = %v, want ~90s", f.name, apiErr.RetryAfter)
			}
		})
	}
}

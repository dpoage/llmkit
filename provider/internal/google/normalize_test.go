package google

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNormalizeErr_ClassifiesFromTransportStatus pins 5xx classification
// when the transport status is not available inside the SDK error. genai
// returns genai.APIError (Code/Message/Status only — no *http.Response, no
// Unwrap); when the error body parses as a Google error object WITHOUT a
// "code" field, the SDK reports Code=0 and drops the transport response
// entirely, which used to classify every such failure as non-retryable
// ErrInvalidRequest. The status the wrapping transport recorded must take
// over. The HTML-body case is the other half of the same contract: a body
// that is not Google error JSON at all makes the SDK fill Code from the
// transport status itself.
func TestNormalizeErr_ClassifiesFromTransportStatus(t *testing.T) {
	cases := []struct {
		name        string
		contentType string
		body        string
	}{
		{
			// Not JSON: the SDK itself stamps Code from the transport.
			name:        "html body",
			contentType: "text/html",
			body:        "<html><body>503 Service Unavailable</body></html>",
		},
		{
			// Google error JSON without "code": the SDK drops the transport
			// response, so the recorded status is the only fallback.
			name:        "code-less error JSON",
			contentType: "application/json",
			body:        `{"error":{"message":"backend unavailable","status":"UNAVAILABLE"}}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.contentType)
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(tc.body))
			})
			client, err := New(context.Background(), "gemini-test", Options{
				APIKey:  "test-key",
				BaseURL: base,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			_, err = client.Complete(context.Background(), simpleRequest())
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Fatalf("error = %v, want ErrServer for a 503", err)
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error is not *llmkit.APIError: %T", err)
			}
			if apiErr.StatusCode != http.StatusServiceUnavailable {
				t.Errorf("StatusCode = %d, want 503", apiErr.StatusCode)
			}
		})
	}
}

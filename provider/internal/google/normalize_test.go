package google

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNormalizeErr_ClassifiesFromTransportStatus pins 5xx classification
// when the SDK error does not carry the transport response: the recorded
// transport status takes over, whether the body was Google JSON without a
// "code" field or a non-JSON body the SDK stamps Code from the transport.
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

// TestNormalizeErr_Sub400ClassifiesErrServer pins a sub-400 status that
// genai reports as its APIError: with no vendor type, the in-band lookup
// falls back to ErrServer with the status preserved.
func TestNormalizeErr_Sub400ClassifiesErrServer(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusFound) // 302, no Location: not followed
		_, _ = w.Write([]byte(`{"error":{"message":"moved"}}`))
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
		t.Fatalf("error = %v, want ErrServer for a sub-400 status without a vendor type", err)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error is not *llmkit.APIError: %T", err)
	}
	if apiErr.StatusCode != http.StatusFound {
		t.Errorf("StatusCode = %d, want 302", apiErr.StatusCode)
	}
}

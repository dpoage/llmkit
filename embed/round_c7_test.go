package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// TestAPIError_MessageTrimAndCap pins: every *llmkit.APIError embed
// returns from server-supplied text carries a whitespace-trimmed Message
// of at most 200 bytes plus "...". Covers the two body routes that build
// one from server text (a non-200 response on both backends, the
// openai-compatible 200 error-object route) and a classification check
// that the whole (untruncated-for-classification) body still drives Kind.
func TestAPIError_MessageTrimAndCap(t *testing.T) {
	padded := "  " + strings.Repeat("m", 1<<20) + "  \n"

	t.Run("non_200_both_backends", func(t *testing.T) {
		for _, backend := range []Backend{BackendOllama, BackendOpenAICompatible} {
			t.Run(string(backend), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(padded))
				}))
				defer srv.Close()

				emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
				if err != nil {
					t.Fatalf("New: %v", err)
				}

				_, err = emb.Embed(context.Background(), "x")
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error = %v, want *llmkit.APIError", err)
				}
				if !errors.Is(err, llmkit.ErrServer) {
					t.Errorf("error = %v, want errors.Is ErrServer (500)", err)
				}
				if len(apiErr.Message) != 203 {
					t.Errorf("len(Message) = %d, want 203", len(apiErr.Message))
				}
				if apiErr.Message != strings.TrimSpace(apiErr.Message) {
					t.Errorf("Message = %q, must already be trimmed", apiErr.Message)
				}
			})
		}
	})

	t.Run("openai_compatible_200_error_object", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := json.Marshal(openaiResponse{Error: &openaiError{Message: padded, Type: "invalid_request_error"}})
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		emb, err := New(Config{Backend: BackendOpenAICompatible, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		_, err = emb.Embed(context.Background(), "x")
		if !errors.Is(err, llmkit.ErrInvalidRequest) {
			t.Errorf("error = %v, want errors.Is ErrInvalidRequest", err)
		}
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %v, want *llmkit.APIError", err)
		}
		if apiErr.StatusCode != http.StatusOK {
			t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
		}
		if len(apiErr.Message) != 203 {
			t.Errorf("len(Message) = %d, want 203", len(apiErr.Message))
		}
		if apiErr.Message != strings.TrimSpace(apiErr.Message) {
			t.Errorf("Message = %q, must already be trimmed", apiErr.Message)
		}
	})

	t.Run("400_context_length_past_cap_both_backends", func(t *testing.T) {
		body := strings.Repeat("x", 300) + "maximum context length"
		for _, backend := range []Backend{BackendOllama, BackendOpenAICompatible} {
			t.Run(string(backend), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = w.Write([]byte(body))
				}))
				defer srv.Close()

				emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
				if err != nil {
					t.Fatalf("New: %v", err)
				}

				_, err = emb.Embed(context.Background(), "x")
				if !errors.Is(err, llmkit.ErrContextTooLong) {
					t.Errorf("error = %v, want errors.Is ErrContextTooLong (classification sees the whole body)", err)
				}
			})
		}
	})
}

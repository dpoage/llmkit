package adapter_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
	"github.com/dpoage/llmkit/retry"
)

// anthropicStatusError makes one Anthropic call against a server that
// answers 400 with an error body carrying msg, and returns the APIError
// the caller sees.
func anthropicStatusError(t *testing.T, msg string) *llmkit.APIError {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprintf(w, `{"type":"error","error":{"type":"invalid_request_error","message":%q}}`, msg)
	}))
	defer srv.Close()
	client, err := provider.New(context.Background(),
		provider.Spec{Type: provider.TypeAnthropic, Model: "claude-test", BaseURL: srv.URL, Secret: "test-key"},
		provider.Options{HTTPClient: srv.Client(), Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	_, err = client.Complete(context.Background(), llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
		MaxTokens: 16,
	})
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
	}
	return apiErr
}

// TestAnthropicStatusRouteMessageIsCapped pins that a chat adapter's status
// route reaches the caller through the same cap: a long error message from
// the server comes back as valid UTF-8 of at most 203 bytes with the full
// text still in the chained SDK error. The SDK prefixes its own text
// (method, URL, status) to the body, so the test measures that prefix and
// places a two-byte rune across byte 200 and then exactly at byte 200.
func TestAnthropicStatusRouteMessageIsCapped(t *testing.T) {
	// Vendor env must not steer the SDK off the httptest server.
	for _, name := range []string{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE", "ANTHROPIC_CUSTOM_HEADERS"} {
		t.Setenv(name, "")
		_ = os.Unsetenv(name)
	}
	t.Setenv("ANTHROPIC_CONFIG_DIR", t.TempDir())

	const marker = "MARKER"
	prefix := strings.Index(anthropicStatusError(t, marker).Err.Error(), marker)
	if prefix < 0 || prefix >= 190 {
		t.Fatalf("SDK text puts the body message at byte %d; want 0 <= offset < 190", prefix)
	}
	for _, runeAt := range []int{199, 200} {
		t.Run(fmt.Sprintf("rune at byte %d", runeAt), func(t *testing.T) {
			msg := strings.Repeat("a", runeAt-prefix) + strings.Repeat("é", 250)
			apiErr := anthropicStatusError(t, msg)
			if !utf8.ValidString(apiErr.Message) {
				t.Errorf("Message is not valid UTF-8: %q", apiErr.Message)
			}
			if len(apiErr.Message) > 203 || !strings.HasSuffix(apiErr.Message, "...") {
				t.Errorf("Message len = %d, want a 200-byte cap plus ...", len(apiErr.Message))
			}
			if !strings.Contains(apiErr.Err.Error(), msg) {
				t.Errorf("chained SDK error lost the full %d-byte message", len(msg))
			}
		})
	}
}

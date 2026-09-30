package provider

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// TestZeroRequestRefusalMessageIsCapped: the zero-request refusal quotes the
// SDK's error text, and a malformed BaseURL puts the whole URL in that text.
// Through New, on every adapter and entry point, the refusal's Message is
// capped like every other vendor-text Message (at most 200 bytes on a rune
// boundary plus "...") while the full SDK error stays in the Unwrap chain.
func TestZeroRequestRefusalMessageIsCapped(t *testing.T) {
	urls := map[string]string{
		"ascii":     badBaseURL + "/" + strings.Repeat("a", 400-len(badBaseURL)-1),
		"multibyte": badBaseURL + "/" + strings.Repeat("é", 200),
	}
	types := map[string]Type{
		"anthropic":         TypeAnthropic,
		"openai":            TypeOpenAI,
		"openai-compatible": TypeOpenAICompatible,
		"google":            TypeGoogle,
	}
	for urlName, baseURL := range urls {
		for _, name := range wireAdapterNames {
			for _, entry := range wireEntries() {
				t.Run(urlName+"/"+name+"/"+entry.name, func(t *testing.T) {
					rt := unreachableRT()
					client, err := New(context.Background(),
						Spec{Type: types[name], Model: "m", BaseURL: baseURL, Secret: "k"},
						Options{HTTPClient: &http.Client{Transport: rt}, Retry: retry.Config{MaxAttempts: 1}})
					if err != nil {
						t.Fatalf("New: %v", err)
					}
					err = entry.call(client, context.Background(), validRequest())
					if n := rt.calls.Load(); n != 0 {
						t.Fatalf("transport saw %d requests; the probe needs a call that sends none (err: %v)", n, err)
					}
					var apiErr *llmkit.APIError
					if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrInvalidRequest) || wantRetry(t, err) {
						t.Fatalf("err = %#v; want a non-retryable *APIError wrapping ErrInvalidRequest", err)
					}
					if apiErr.Err == nil || !strings.Contains(apiErr.Err.Error(), baseURL[len(baseURL)-20:]) {
						t.Fatalf("Unwrap chain = %v; want the SDK error with the full URL", apiErr.Err)
					}
					// A cut Message is the longest rune-aligned prefix of at
					// most 200 bytes plus "...": a rune is at most
					// utf8.UTFMax bytes, so the prefix is over 200-UTFMax.
					stem, cut := strings.CutSuffix(apiErr.Message, "...")
					if !cut || len(stem) > 200 || len(stem) <= 200-utf8.UTFMax || !utf8.ValidString(apiErr.Message) ||
						!strings.Contains(apiErr.Err.Error(), stem[len(stem)-40:]) {
						t.Fatalf("Message is %d bytes (valid UTF-8: %t); want a rune-aligned cut of the SDK text to at most 200 bytes plus \"...\": %q",
							len(apiErr.Message), utf8.ValidString(apiErr.Message), apiErr.Message)
					}
				})
			}
		}
	}
}

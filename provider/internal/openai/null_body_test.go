package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestComplete_NullBodyIsAPIError: a 200 JSON body of `null` decodes to no
// completion at all. Complete returns the no-completion *APIError
// (ErrServer, the response status), never a panic or a nil error, on
// first-party and openai-compatible alike.
func TestComplete_NullBodyIsAPIError(t *testing.T) {
	for _, compat := range []bool{false, true} {
		base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, "null")
		})
		c, err := New("gpt-test", Options{APIKey: "k", BaseURL: base, Compatible: compat})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = c.Complete(context.Background(), simpleRequest())
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != http.StatusOK {
			t.Fatalf("compat=%v: Complete err = %v (%T), want *APIError{ErrServer, 200}", compat, err, err)
		}
	}
}

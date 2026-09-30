package google

import (
	"context"
	"errors"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestStream_SSEDataLineCarryingErrorObjectIsAPIError: a 200 SSE stream
// whose only data line is {"error":{...}} — genai decodes it to a chunk with
// no candidates and drops the error — carries no completion. Stream returns
// an *APIError (ErrServer, StatusCode 200), not a success; a stream of
// "data: {}" lines or no events at all is the same. Complete on the same
// body agrees.
func TestStream_SSEDataLineCarryingErrorObjectIsAPIError(t *testing.T) {
	bodies := map[string]map[string]any{
		"error-object": {"error": map[string]any{"code": 500, "message": "boom", "status": "INTERNAL"}},
		"empty-object": {},
	}
	for name, body := range bodies {
		t.Run(name+"/stream", func(t *testing.T) {
			_, err := newClient(t, streamServer(t, body)).Stream(context.Background(), simpleRequest(), nil)
			assertNoCompletion(t, err)
		})
		t.Run(name+"/complete", func(t *testing.T) {
			_, err := newClient(t, completeServer(t, body)).Complete(context.Background(), simpleRequest())
			assertNoCompletion(t, err)
		})
	}
	t.Run("no-events/stream", func(t *testing.T) {
		_, err := newClient(t, streamServer(t)).Stream(context.Background(), simpleRequest(), nil)
		assertNoCompletion(t, err)
	})
}

func assertNoCompletion(t *testing.T, err error) {
	t.Helper()
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != 200 {
		t.Fatalf("err = %v (%T), want *llmkit.APIError{ErrServer, 200}", err, err)
	}
	if _, _, retryable := llmkit.Classify(err); !retryable {
		t.Errorf("Classify(err) = terminal, want retryable")
	}
}

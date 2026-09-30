package google

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNon2xx_StatusCodeIsBodyErrorCode: a non-2xx response whose body is a
// Gemini error object with a code reports that code as StatusCode, not the
// response status, and classifies Kind from it: a 500 or an unfollowed 302
// carrying code 400 is ErrInvalidRequest at 400.
func TestNon2xx_StatusCodeIsBodyErrorCode(t *testing.T) {
	for _, status := range []int{http.StatusInternalServerError, http.StatusFound} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}}`))
			})
			_, err := newClient(t, base).Complete(context.Background(), simpleRequest())
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
			}
			if apiErr.StatusCode != 400 {
				t.Errorf("StatusCode = %d, want the body's error.code 400 (response status %d)", apiErr.StatusCode, status)
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("Kind = %v, want ErrInvalidRequest (classified from error.code)", apiErr.Kind)
			}
		})
	}
}

// TestStream_LastPromptFeedbackCounts: on Stream the promptFeedback that
// decides a blocked prompt is the last one a data: line carries. A block
// followed by an empty promptFeedback is no completion (an ErrServer
// APIError); an empty promptFeedback followed by a block is
// StopContentFilter.
func TestStream_LastPromptFeedbackCounts(t *testing.T) {
	blocked := map[string]any{"promptFeedback": map[string]any{"blockReason": "SAFETY"}}
	cleared := map[string]any{"promptFeedback": map[string]any{}}
	t.Run("block-then-empty", func(t *testing.T) {
		_, err := newClient(t, streamServer(t, blocked, cleared)).Stream(context.Background(), simpleRequest(), nil)
		assertNoCompletion(t, err)
	})
	t.Run("empty-then-block", func(t *testing.T) {
		resp, err := newClient(t, streamServer(t, cleared, blocked)).Stream(context.Background(), simpleRequest(), nil)
		if err != nil || resp.StopReason != llmkit.StopContentFilter {
			t.Fatalf("Stream = (%+v, %v), want StopContentFilter, nil", resp, err)
		}
	})
}

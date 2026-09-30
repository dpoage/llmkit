package google

import (
	"context"
	"errors"
	"testing"

	"github.com/dpoage/llmkit"
)

func newClient(t *testing.T, base string) llmkit.StreamingClient {
	t.Helper()
	c, err := New(context.Background(), "gemini-test", Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c.(llmkit.StreamingClient)
}

// TestPromptFeedbackBlock_IsContentFilterNotEndTurn: a blocked prompt (no
// candidates, promptFeedback.blockReason set) is a successful response that
// reports StopContentFilter with empty text — not StopEndTurn, and not the
// no-completion error — on Complete and Stream alike.
func TestPromptFeedbackBlock_IsContentFilterNotEndTurn(t *testing.T) {
	usage := map[string]any{"promptTokenCount": 7, "totalTokenCount": 7}
	for _, reason := range []string{"SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT"} {
		body := map[string]any{
			"promptFeedback": map[string]any{"blockReason": reason},
			"usageMetadata":  usage,
		}
		check := func(t *testing.T, resp llmkit.Response, err error) {
			t.Helper()
			if err != nil {
				t.Fatalf("err = %v, want nil for a blocked prompt", err)
			}
			if resp.StopReason != llmkit.StopContentFilter {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, llmkit.StopContentFilter)
			}
			if resp.Text != "" || len(resp.ToolCalls) != 0 {
				t.Errorf("Response = %+v, want no text and no tool calls", resp)
			}
			if resp.Usage.InputTokens != 7 {
				t.Errorf("Usage.InputTokens = %d, want 7 (usage survives the block)", resp.Usage.InputTokens)
			}
		}
		t.Run(reason+"/complete", func(t *testing.T) {
			resp, err := newClient(t, completeServer(t, body)).Complete(context.Background(), simpleRequest())
			check(t, resp, err)
		})
		t.Run(reason+"/stream", func(t *testing.T) {
			resp, err := newClient(t, streamServer(t, body)).Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
			check(t, resp, err)
		})
	}
}

// TestNoCandidates_UnspecifiedBlockReasonIsNotABlock: the proto placeholder
// BLOCKED_REASON_UNSPECIFIED names no block, so a response carrying only it
// has no completion and is an ErrServer APIError like any empty response.
func TestNoCandidates_UnspecifiedBlockReasonIsNotABlock(t *testing.T) {
	body := map[string]any{"promptFeedback": map[string]any{"blockReason": "BLOCKED_REASON_UNSPECIFIED"}}
	check := func(t *testing.T, err error) {
		t.Helper()
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != 200 {
			t.Fatalf("err = %v (%T), want *APIError{ErrServer, 200}", err, err)
		}
	}
	t.Run("complete", func(t *testing.T) {
		_, err := newClient(t, completeServer(t, body)).Complete(context.Background(), simpleRequest())
		check(t, err)
	})
	t.Run("stream", func(t *testing.T) {
		_, err := newClient(t, streamServer(t, body)).Stream(context.Background(), simpleRequest(), nil)
		check(t, err)
	})
}

// TestStopCandidateWithEmptyContent_StillSucceeds: candidates present with
// finishReason STOP is a completion even when the text is empty — the
// no-completion check keys on candidates, not on text.
func TestStopCandidateWithEmptyContent_StillSucceeds(t *testing.T) {
	body := map[string]any{"candidates": []any{map[string]any{"finishReason": "STOP"}}}
	resp, err := newClient(t, completeServer(t, body)).Complete(context.Background(), simpleRequest())
	if err != nil || resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("Complete = (%+v, %v), want StopEndTurn, nil", resp, err)
	}
	resp, err = newClient(t, streamServer(t, body)).Stream(context.Background(), simpleRequest(), nil)
	if err != nil || resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("Stream = (%+v, %v), want StopEndTurn, nil", resp, err)
	}
}

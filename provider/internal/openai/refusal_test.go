package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestOpenAIRefusal_SurfacedAsStopRefusal covers message.refusal: when the
// model declines on policy grounds OpenAI puts the explanation in
// message.refusal and leaves content empty. The adapter must surface the
// refusal text as Response.Text with StopRefusal — not an empty-text
// StopEndTurn that hides why the turn ended.
func TestOpenAIRefusal_SurfacedAsStopRefusal(t *testing.T) {
	const refusalText = "I cannot help with that request."
	body := `{
		"id": "chatcmpl-refusal", "object": "chat.completion", "created": 1, "model": "gpt-test",
		"choices": [{"index": 0,
			"message": {"role": "assistant", "content": null, "refusal": "` + refusalText + `"},
			"finish_reason": "stop"}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
	}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)

	client := New("gpt-test", Options{APIKey: "k", BaseURL: srv.URL})
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != llmkit.StopRefusal {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llmkit.StopRefusal)
	}
	if resp.Text != refusalText {
		t.Errorf("Text = %q, want the refusal text %q", resp.Text, refusalText)
	}
	if len(resp.Blocks) != 1 || resp.Blocks[0].Kind != llmkit.BlockText || resp.Blocks[0].Text != refusalText {
		t.Errorf("Blocks = %+v, want one text block carrying the refusal", resp.Blocks)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 9 {
		t.Errorf("Usage = %+v, want {5 9} (usage mapping must survive the refusal path)", resp.Usage)
	}
}

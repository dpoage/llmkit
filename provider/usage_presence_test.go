package provider

import (
	"context"
	"io"
	"net/http"
	"testing"

	llmkit "github.com/dpoage/llmkit"
)

func TestNew_CompletionEventCarriesEstimatedUsage(t *testing.T) {
	const reply = "hello world"
	bodyLen := 0
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodyLen = len(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","created":1,"model":"llama-test",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"` + reply + `"},"finish_reason":"stop"}]}`))
	})
	obs := &eventCapture{}
	client, err := New(context.Background(), observeSpec(base), Options{Observer: obs})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	want := llmkit.Usage{InputTokens: int64((bodyLen + 3) / 4), OutputTokens: int64((len(reply) + 3) / 4), Estimated: true}
	if bodyLen == 0 || resp.Usage != want {
		t.Errorf("Response.Usage = %+v, want %+v", resp.Usage, want)
	}
	completions := obs.byKind(llmkit.KindCompletion)
	if len(completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(completions))
	}
	if got := completions[0].Completion.Response.Usage; got != want {
		t.Errorf("Completion event Usage = %+v, want %+v", got, want)
	}
}

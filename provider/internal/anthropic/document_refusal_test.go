package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestInlineDocumentMustBePDF pins the vendor-specific refusal: Anthropic
// accepts base64 documents as application/pdf only, so any other inline
// media type must be refused pre-wire (no request) on Complete and Stream,
// while application/pdf and a URL document reach the wire.
func TestInlineDocumentMustBePDF(t *testing.T) {
	var hits atomic.Int32
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, mockTextBody("ok", 1, 1))
	})
	client, err := New("claude-test", Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	build := func(b llmkit.Block) llmkit.Request {
		req := simpleRequest()
		req.Messages = []llmkit.Message{llmkit.UserMessage(llmkit.Text("read this"), b)}
		return req
	}

	for _, mt := range []string{"text/plain", "application/msword"} {
		hits.Store(0)
		req := build(llmkit.Document(mt, "doc", []byte("hello")))
		_, cerr := client.Complete(context.Background(), req)
		_, serr := client.(llmkit.StreamingClient).Stream(context.Background(), req, nil)
		for entry, err := range map[string]error{"Complete": cerr, "Stream": serr} {
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("media type %q %s: err = %v; want ErrInvalidRequest", mt, entry, err)
			}
		}
		if n := hits.Load(); n != 0 {
			t.Errorf("media type %q: server saw %d requests; a refused request must send none", mt, n)
		}
	}

	for name, b := range map[string]llmkit.Block{
		"pdf bytes": llmkit.Document("application/pdf", "doc", []byte("%PDF-1.4")),
		"url":       llmkit.DocumentURL("https://example.com/a.pdf", "doc"),
	} {
		if _, err := client.Complete(context.Background(), build(b)); err != nil {
			t.Errorf("%s: Complete: %v; want it sent", name, err)
		}
	}
}

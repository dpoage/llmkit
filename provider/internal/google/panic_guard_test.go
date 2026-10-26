package google

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
)

type fnPanic struct{ msg string }

func recoverValue(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

func TestStream_CallbackPanicPropagatesUnchanged(t *testing.T) {
	want := &fnPanic{"callback"}
	base := streamServer(t, map[string]any{
		"candidates": []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": "hi"}}}}},
	})
	got := recoverValue(func() {
		_, _ = newClient(t, base).Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { panic(want) })
	})
	if got != want {
		t.Fatalf("recovered %v (%T); want the callback's own panic value", got, got)
	}
}

// holdingServer streams one text chunk, then holds the connection open until
// the client closes it. closed receives once the server sees that.
func holdingServer(t *testing.T) (base string, closed <-chan struct{}) {
	t.Helper()
	done := make(chan struct{})
	base = newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.Copy(io.Discard, r.Body) // the server notices a closed client only after the body is read
		_, _ = w.Write([]byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}` + "\n\n"))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			close(done)
		case <-time.After(5 * time.Second):
		}
	})
	return base, done
}

func TestStream_ReleasesTheStreamOnEveryExit(t *testing.T) {
	exits := map[string]func(llmkit.Delta) error{
		"callback error": func(llmkit.Delta) error { return errors.New("stop") },
		"callback panic": func(llmkit.Delta) error { panic(&fnPanic{"stop"}) },
	}
	for name, fn := range exits {
		t.Run(name, func(t *testing.T) {
			base, closed := holdingServer(t)
			recoverValue(func() {
				_, _ = newClient(t, base).Stream(context.Background(), simpleRequest(), fn)
			})
			select {
			case <-closed:
			case <-time.After(2 * time.Second):
				t.Fatal("the server never saw the connection close after Stream returned")
			}
		})
	}
}

func TestPanicError_MasksTheCredential(t *testing.T) {
	const secret = "sk-PANICPROBE-0123456789abcdef"
	c, err := New(context.Background(), "gemini-test", Options{APIKey: secret, BaseURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := adapter.WithWire(context.Background())
	got := c.(*googleAdapter).panicError(ctx, "decoding "+secret)
	var api *llmkit.APIError
	if !errors.As(got, &api) || !errors.Is(got, llmkit.ErrServer) {
		t.Fatalf("err = %v; want an ErrServer *llmkit.APIError", got)
	}
	if strings.Contains(got.Error(), secret[:12]) || !strings.Contains(api.Message, "***") {
		t.Fatalf("Message = %q; want the secret masked", api.Message)
	}
}

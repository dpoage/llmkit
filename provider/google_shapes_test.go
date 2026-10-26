package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// shapeWant is the outcome one (body, mode) cell must keep: no error, or an
// *llmkit.APIError of Kind ErrServer with status.
type shapeWant struct {
	nilErr bool
	status int
}

// shapeServe answers every request with body: as JSON for Complete, as one
// SSE data line for Stream (an Anthropic stream opens with a named event).
func shapeServe(t *testing.T, fam string, stream bool, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		switch {
		case stream && fam == "anthropic":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("event: message_start\ndata: " + body + "\n\n"))
		case stream:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: " + body + "\n\n"))
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// shapeCall returns the call's error, or a PANIC error when it panics.
func shapeCall(t *testing.T, typ Type, stream bool, base string) (err error) {
	t.Helper()
	c, nerr := New(context.Background(), Spec{Type: typ, Model: "m-test", Secret: "placeholder", BaseURL: base},
		Options{Retry: retry.Config{MaxAttempts: 1}})
	if nerr != nil {
		t.Fatalf("New: %v", nerr)
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("PANIC: %v", r)
		}
	}()
	req := llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}, MaxTokens: 8}
	if stream {
		_, err = llmkit.Stream(context.Background(), c, req, func(llmkit.Delta) error { return nil })
	} else {
		_, err = c.Complete(context.Background(), req)
	}
	return err
}

func shapeCheck(t *testing.T, err error, want shapeWant) {
	t.Helper()
	if want.nilErr {
		if err != nil {
			t.Errorf("err = %v; want nil", err)
		}
		return
	}
	var api *llmkit.APIError
	if !errors.As(err, &api) || !errors.Is(err, llmkit.ErrServer) || api.StatusCode != want.status {
		t.Errorf("err = %v; want *llmkit.APIError{ErrServer, status %d}", err, want.status)
	} else if _, _, retryable := llmkit.Classify(err); !retryable {
		t.Errorf("err = %v; want a retryable error", err)
	}
}

// TestGoogleShapes_MalformedBodiesNeverPanic: a 2xx body that genai cannot
// decode into a response returns an error and never panics, on Complete and
// Stream. A body that panicked inside genai is an ErrServer APIError with
// the response's status (200); every other body keeps the outcome it had
// before the guard (a decode failure is status 0, and a null body on Stream
// is the no-candidates error at status 200).
func TestGoogleShapes_MalformedBodiesNeverPanic(t *testing.T) {
	const valid = `{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}`
	ok200, none := shapeWant{status: 200}, shapeWant{status: 0}
	cells := []struct {
		body             string
		complete, stream shapeWant
	}{
		{`null`, ok200, ok200},
		{`[]`, none, none},
		{`[null]`, none, none},
		{`["x"]`, none, none},
		{`[[]]`, none, none},
		{`[true]`, none, none},
		{`{"candidates":[null]}`, ok200, ok200},
		{`{"candidates":[1]}`, ok200, ok200},
		{`{"candidates":"x"}`, ok200, ok200},
		{`{"candidates":{}}`, ok200, ok200},
		{`{"candidates":[null,` + valid + `]}`, ok200, ok200},
	}
	for _, c := range cells {
		for _, stream := range []bool{false, true} {
			want := c.complete
			if stream {
				want = c.stream
			}
			t.Run(fmt.Sprintf("%s/%s", c.body, rxMode(stream)), func(t *testing.T) {
				shapeCheck(t, shapeCall(t, TypeGoogle, stream, shapeServe(t, "google", stream, c.body).URL), want)
			})
		}
	}
}

// TestGoogleShapes_OtherAdaptersKeepTheirOutcome: the bodies that panic in
// genai decode without a panic on the other adapters; their outcomes stay.
// A Stream over such a body is a stream that ends early: status 0.
func TestGoogleShapes_OtherAdaptersKeepTheirOutcome(t *testing.T) {
	ok200, none, ok := shapeWant{status: 200}, shapeWant{status: 0}, shapeWant{nilErr: true}
	bodies := []string{`null`, `[]`, `[null]`, `{"content":[null],"stop_reason":"end_turn"}`, `{"choices":[null]}`}
	complete := map[string][]shapeWant{
		"anthropic": {ok200, ok200, ok200, ok, ok200},
		"openai":    {ok200, ok200, ok200, ok200, ok},
	}
	types := []struct {
		name string
		typ  Type
		fam  string
	}{
		{"anthropic", TypeAnthropic, "anthropic"},
		{"openai", TypeOpenAI, "openai"},
		{"openai-compatible", TypeOpenAICompatible, "openai"},
	}
	for _, ty := range types {
		for i, body := range bodies {
			for _, stream := range []bool{false, true} {
				want := none
				if !stream {
					want = complete[ty.fam][i]
				}
				t.Run(fmt.Sprintf("%s/%s/%s", ty.name, body, rxMode(stream)), func(t *testing.T) {
					shapeCheck(t, shapeCall(t, ty.typ, stream, shapeServe(t, ty.fam, stream, body).URL), want)
				})
			}
		}
	}
}

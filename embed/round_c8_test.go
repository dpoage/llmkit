package embed

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// countingBody wraps a response body reader and counts bytes actually
// pulled through Read, so the test observes client-side consumption
// rather than inferring it from what the server wrote.
type countingBody struct {
	body io.ReadCloser
	n    *int64
}

func (c *countingBody) Read(p []byte) (int, error) {
	n, err := c.body.Read(p)
	atomic.AddInt64(c.n, int64(n))
	return n, err
}

func (c *countingBody) Close() error { return c.body.Close() }

// countingBodyTransport wraps every response body from rt in a
// countingBody.
type countingBodyTransport struct {
	rt http.RoundTripper
	n  *int64
}

func (t *countingBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.rt.RoundTrip(req)
	if err != nil {
		return resp, err
	}
	resp.Body = &countingBody{body: resp.Body, n: t.n}
	return resp, nil
}

// paddedResponseBody returns a valid response body for backend, padded
// with an ignored extra JSON field to exactly totalBytes.
func paddedResponseBody(t *testing.T, backend Backend, totalBytes int) []byte {
	t.Helper()
	var prefix, suffix string
	switch backend {
	case BackendOllama:
		prefix, suffix = `{"model":"m","embeddings":[[0.1]],"padding":"`, `"}`
	case BackendOpenAICompatible:
		prefix, suffix = `{"model":"m","data":[{"embedding":[0.1],"index":0}],"padding":"`, `"}`
	default:
		t.Fatalf("paddedResponseBody: unsupported backend %q", backend)
	}
	padLen := totalBytes - len(prefix) - len(suffix)
	if padLen < 0 {
		t.Fatalf("paddedResponseBody: totalBytes %d too small for %q wire overhead", totalBytes, backend)
	}
	buf := make([]byte, 0, totalBytes)
	buf = append(buf, prefix...)
	for i := 0; i < padLen; i++ {
		buf = append(buf, 'x')
	}
	buf = append(buf, suffix...)
	return buf
}

// TestResponseBody_Bound pins: every response body read is bounded by
// maxResponseBytes, on both backends and every status. Uses a client-side
// counting Transport (not a server-side write-failure) because
// server-observed behaviour is host-buffer dependent.
func TestResponseBody_Bound(t *testing.T) {
	const l = 256 << 10 // 256 KiB: the test-lowered bound

	orig := maxResponseBytes
	maxResponseBytes = l
	t.Cleanup(func() { maxResponseBytes = orig })

	for _, backend := range []Backend{BackendOllama, BackendOpenAICompatible} {
		t.Run(string(backend)+"/200_oversized_terminal_plain_error", func(t *testing.T) {
			var bytesRead int64
			var calls atomic.Int32
			body := paddedResponseBody(t, backend, l+64<<10)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write(body)
			}))
			defer srv.Close()

			client := &http.Client{Transport: &countingBodyTransport{rt: http.DefaultTransport, n: &bytesRead}}
			emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, HTTPClient: client, Retry: retry.Config{MaxAttempts: 3}})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = emb.Embed(context.Background(), "x")
			if err == nil {
				t.Fatal("expected an error for an oversized 200 body")
			}
			if !strings.Contains(err.Error(), fmt.Sprint(l)) {
				t.Errorf("error %q must name the limit %d", err, l)
			}
			var apiErr *llmkit.APIError
			if errors.As(err, &apiErr) {
				t.Errorf("got *llmkit.APIError %v, want a plain terminal error", apiErr)
			}
			if got := calls.Load(); got != 1 {
				t.Errorf("wire hits = %d, want 1 (terminal under llmkit.Classify)", got)
			}
			if got := atomic.LoadInt64(&bytesRead); got > int64(l)+1 {
				t.Errorf("bytes read = %d, want <= %d", got, l+1)
			}
		})

		t.Run(string(backend)+"/500_oversized_classified_by_status", func(t *testing.T) {
			var bytesRead int64
			body := paddedResponseBody(t, backend, l+64<<10)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write(body)
			}))
			defer srv.Close()

			client := &http.Client{Transport: &countingBodyTransport{rt: http.DefaultTransport, n: &bytesRead}}
			emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, HTTPClient: client, Retry: retry.Config{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = emb.Embed(context.Background(), "x")
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != http.StatusInternalServerError {
				t.Errorf("StatusCode = %d, want 500", apiErr.StatusCode)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("error = %v, want errors.Is ErrServer", err)
			}
			if len(apiErr.Message) > 203 {
				t.Errorf("len(Message) = %d, want <= 203", len(apiErr.Message))
			}
			if got := atomic.LoadInt64(&bytesRead); got > int64(l)+1 {
				t.Errorf("bytes read = %d, want <= %d", got, l+1)
			}
		})

		t.Run(string(backend)+"/200_exactly_at_limit_succeeds", func(t *testing.T) {
			body := paddedResponseBody(t, backend, l)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write(body)
			}))
			defer srv.Close()

			emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			vec, err := emb.Embed(context.Background(), "x")
			if err != nil {
				t.Fatalf("Embed with a body exactly at the limit: %v", err)
			}
			if len(vec) != 1 {
				t.Fatalf("vec = %v, want length 1", vec)
			}
		})
	}
}

// TestResponseBody_DocTruthShippedValue pins the doc-truth pairing: the
// shipped maxResponseBytes value is 64 MiB, matching the package doc's
// "# Errors" sentence.
func TestResponseBody_DocTruthShippedValue(t *testing.T) {
	if maxResponseBytes != 64<<20 {
		t.Errorf("maxResponseBytes = %d, want 64<<20 (64 MiB, as the package doc's \"# Errors\" section states)", maxResponseBytes)
	}
}

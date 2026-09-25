package embed

import (
	"context"
	"errors"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// newTestEmbedder builds the named backend against url with retryCfg,
// failing the test on a construction error.
func newTestEmbedder(t *testing.T, backend, url string, retryCfg retry.Config) Embedder {
	t.Helper()
	return newTestEmbedderWithClient(t, backend, url, retryCfg, nil)
}

// newTestEmbedderWithClient is newTestEmbedder with an optional injected
// HTTP client (e.g. to count dial attempts a server-side handler never sees).
func newTestEmbedderWithClient(t *testing.T, backend, url string, retryCfg retry.Config, client *http.Client) Embedder {
	t.Helper()
	emb, err := New(Config{Backend: Backend(backend), Model: "m", URL: url, Retry: retryCfg, HTTPClient: client})
	if err != nil {
		t.Fatalf("New(%s): %v", backend, err)
	}
	return emb
}

// writeSuccessBody writes a minimal successful response for backend.
func writeSuccessBody(w http.ResponseWriter, backend string) {
	switch backend {
	case "ollama":
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1}}})
	case "openai-compatible":
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	}
}

// countingTransport counts every RoundTrip attempt, including ones that
// never reach a listening server (dial failures), which a handler-side
// atomic counter cannot see.
type countingTransport struct {
	calls atomic.Int32
	rt    http.RoundTripper
}

func (c *countingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	c.calls.Add(1)
	return c.rt.RoundTrip(req)
}

var backends = []string{"ollama", "openai-compatible"}

// TestE1_ClassificationAndRetryAfter pins a 429 carrying Retry-After is errors.Is(ErrRateLimited) with the header decoded onto the APIError and honoured by the retry sleep; a 401 is errors.Is(ErrAuth) and terminal after one hit; the openai-compatible 200 error-object route honours Retry-After the same way.
func TestE1_ClassificationAndRetryAfter(t *testing.T) {
	for _, backend := range backends {
		t.Run(backend+"/429_retry_after_honored", func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Retry-After", "2")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte("slow down"))
			}))
			defer srv.Close()

			var mu sync.Mutex
			var sleeps []time.Duration
			retryCfg := retry.Config{
				MaxAttempts: 2,
				Sleep: func(ctx context.Context, d time.Duration) error {
					mu.Lock()
					sleeps = append(sleeps, d)
					mu.Unlock()
					return nil
				},
			}
			emb := newTestEmbedder(t, backend, srv.URL, retryCfg)

			_, err := emb.Embed(context.Background(), "x")
			if err == nil {
				t.Fatal("expected an error after exhausting retries on a persistent 429")
			}
			if !errors.Is(err, llmkit.ErrRateLimited) {
				t.Errorf("error = %v, want errors.Is ErrRateLimited", err)
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != http.StatusTooManyRequests {
				t.Errorf("StatusCode = %d, want 429", apiErr.StatusCode)
			}
			if !apiErr.HasRetryAfter || apiErr.RetryAfter != 2*time.Second {
				t.Errorf("RetryAfter = %v (has=%v), want 2s (has=true)", apiErr.RetryAfter, apiErr.HasRetryAfter)
			}
			mu.Lock()
			gotSleeps := append([]time.Duration(nil), sleeps...)
			mu.Unlock()
			if len(gotSleeps) != 1 || gotSleeps[0] != 2*time.Second {
				t.Errorf("sleeps = %v, want exactly [2s]", gotSleeps)
			}
			if got := calls.Load(); got != 2 {
				t.Errorf("wire hits = %d, want 2", got)
			}
		})

		t.Run(backend+"/401_terminal", func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte("invalid api key"))
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
			_, err := emb.Embed(context.Background(), "x")
			if !errors.Is(err, llmkit.ErrAuth) {
				t.Errorf("error = %v, want errors.Is ErrAuth", err)
			}
			if got := calls.Load(); got != 1 {
				t.Errorf("wire hits = %d, want 1 (401 is terminal)", got)
			}
		})
	}

	t.Run("openai-compatible/200_rate_limit_error_object_retry_after_honored", func(t *testing.T) {
		var calls atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			w.Header().Set("Retry-After", "2")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error"}}`))
		}))
		defer srv.Close()

		var mu sync.Mutex
		var sleeps []time.Duration
		retryCfg := retry.Config{
			MaxAttempts: 2,
			Sleep: func(ctx context.Context, d time.Duration) error {
				mu.Lock()
				sleeps = append(sleeps, d)
				mu.Unlock()
				return nil
			},
		}
		emb := newTestEmbedder(t, "openai-compatible", srv.URL, retryCfg)

		_, err := emb.Embed(context.Background(), "x")
		if !errors.Is(err, llmkit.ErrRateLimited) {
			t.Errorf("error = %v, want errors.Is ErrRateLimited", err)
		}
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %v, want *llmkit.APIError", err)
		}
		if apiErr.StatusCode != http.StatusOK {
			t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
		}
		if !apiErr.HasRetryAfter || apiErr.RetryAfter != 2*time.Second {
			t.Errorf("RetryAfter = %v (has=%v), want 2s (has=true)", apiErr.RetryAfter, apiErr.HasRetryAfter)
		}
		mu.Lock()
		gotSleeps := append([]time.Duration(nil), sleeps...)
		mu.Unlock()
		if len(gotSleeps) != 1 || gotSleeps[0] != 2*time.Second {
			t.Errorf("sleeps = %v, want exactly [2s]", gotSleeps)
		}
		if got := calls.Load(); got != 2 {
			t.Errorf("wire hits = %d, want 2", got)
		}
	})
}

// TestE2_TransportFailuresAndCancellation pins a mid-flight caller cancel surfaces as the plain context error after one hit (never an APIError); a dial failure, a per-attempt timeout, and an EOF while reading the body all normalize as a retryable *llmkit.APIError{ErrServer, 0}.
func TestE2_TransportFailuresAndCancellation(t *testing.T) {
	for _, backend := range backends {
		t.Run(backend+"/mid_flight_cancel", func(t *testing.T) {
			var calls atomic.Int32
			reqReceived := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				// Drain the request body first: the server notices the
				// client's disconnect, and cancels r.Context(), only once it
				// reads past the body. The 2s bound is a backstop so the
				// handler always returns and srv.Close() never blocks on it.
				_, _ = io.Copy(io.Discard, r.Body)
				close(reqReceived)
				select {
				case <-r.Context().Done():
				case <-time.After(2 * time.Second):
				}
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
			ctx, cancel := context.WithCancel(context.Background())
			go func() {
				<-reqReceived
				cancel()
			}()

			_, err := emb.Embed(ctx, "x")
			if !errors.Is(err, context.Canceled) {
				t.Errorf("error = %v, want errors.Is context.Canceled", err)
			}
			var apiErr *llmkit.APIError
			if errors.As(err, &apiErr) {
				t.Errorf("got *llmkit.APIError %v, want a plain context error", apiErr)
			}
			if got := calls.Load(); got != 1 {
				t.Errorf("wire hits = %d, want 1 (cancellation is terminal)", got)
			}
		})

		t.Run(backend+"/read_body_eof_retried", func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				conn, bufrw, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Errorf("Hijack: %v", err)
					return
				}
				// Promise 1000 body bytes, send a few, then close: the
				// client's io.ReadAll ends with an unexpected EOF.
				_, _ = bufrw.WriteString("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"emb")
				_ = bufrw.Flush()
				_ = conn.Close()
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
			_, err := emb.Embed(context.Background(), "x")
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != 0 {
				t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("error = %v, want errors.Is ErrServer", err)
			}
			if got := calls.Load(); got != 3 {
				t.Errorf("wire hits = %d, want 3 (a body-read EOF is retried)", got)
			}
		})

		t.Run(backend+"/dial_failure_retried", func(t *testing.T) {
			closedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
			closedSrv.Close()

			ct := &countingTransport{rt: http.DefaultTransport}
			emb := newTestEmbedderWithClient(t, backend, closedSrv.URL,
				retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond},
				&http.Client{Transport: ct})

			_, err := emb.Embed(context.Background(), "x")
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != 0 {
				t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("error = %v, want errors.Is ErrServer", err)
			}
			if got := ct.calls.Load(); got != 3 {
				t.Errorf("dial attempts = %d, want 3", got)
			}
		})

		t.Run(backend+"/stalled_attempt_timeout_retried", func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				time.Sleep(200 * time.Millisecond)
				writeSuccessBody(w, backend)
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{
				MaxAttempts: 2, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond, RequestTimeout: 50 * time.Millisecond,
			})
			_, err := emb.Embed(context.Background(), "x")
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != 0 {
				t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("error = %v, want errors.Is ErrServer", err)
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("error = %v, want errors.Is context.DeadlineExceeded (via Err)", err)
			}
			if got := calls.Load(); got != 2 {
				t.Errorf("wire hits = %d, want 2", got)
			}
		})
	}
}

// TestE3_ConfigRefusalsWrapErrInvalidRequest pins every construction and validation refusal wraps the kit's ErrInvalidRequest sentinel.
func TestE3_ConfigRefusalsWrapErrInvalidRequest(t *testing.T) {
	tests := []struct {
		name string
		fn   func() error
	}{
		{"Validate/unknown backend", func() error {
			return Config{Backend: "nope", Model: "m", URL: "http://x"}.Validate()
		}},
		{"Validate/zero backend", func() error {
			return Config{Model: "m", URL: "http://x"}.Validate()
		}},
		{"Validate/empty model", func() error {
			return Config{Backend: BackendOllama, Model: "", URL: "http://x"}.Validate()
		}},
		{"Validate/empty URL", func() error {
			return Config{Backend: BackendOllama, Model: "m", URL: ""}.Validate()
		}},
		{"Validate/negative dimensions", func() error {
			return Config{Backend: BackendOllama, Model: "m", URL: "http://x", Dimensions: -1}.Validate()
		}},
		{"Validate/negative max batch", func() error {
			return Config{Backend: BackendOllama, Model: "m", URL: "http://x", MaxBatch: -1}.Validate()
		}},
		{"Validate/whitespace-padded API key", func() error {
			return Config{Backend: BackendOllama, Model: "m", URL: "http://x", APIKey: "  sk-test  "}.Validate()
		}},
		{"Validate/jitter out of range", func() error {
			return Config{Backend: BackendOllama, Model: "m", URL: "http://x", Retry: retry.Config{Jitter: 2}}.Validate()
		}},
		{"New/ollama bad config", func() error {
			_, err := New(Config{Backend: BackendOllama, Model: "", URL: ""})
			return err
		}},
		{"New/openai-compatible bad config", func() error {
			_, err := New(Config{Backend: BackendOpenAICompatible, Model: "", URL: ""})
			return err
		}},
		{"New/unknown backend", func() error {
			_, err := New(Config{Backend: "nope", Model: "m", URL: "http://x"})
			return err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if err == nil {
				t.Fatal("expected an error")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("error = %v, want errors.Is ErrInvalidRequest", err)
			}
		})
	}
}

// TestE5_UndecodableBodyTerminal_SubStatusRetried pins failures outside the 4xx/5xx split: a 200 with an undecodable body is a plain terminal error (one wire hit); a sub-400 non-200 status with no recognizable vendor type and an openai-compatible 200 error object of unknown type classify ErrServer and are retried.
func TestE5_UndecodableBodyTerminal_SubStatusRetried(t *testing.T) {
	for _, backend := range backends {
		t.Run(backend+"/garbage_200_body_terminal", func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("not json"))
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
			_, err := emb.Embed(context.Background(), "x")
			if err == nil {
				t.Fatal("expected a decode error")
			}
			var apiErr *llmkit.APIError
			if errors.As(err, &apiErr) {
				t.Errorf("got *llmkit.APIError %v, want a plain decode error", apiErr)
			}
			if got := calls.Load(); got != 1 {
				t.Errorf("wire hits = %d, want 1 (a decode error is terminal)", got)
			}
		})

		t.Run(backend+"/302_no_location_retried", func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusFound)
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
			_, err := emb.Embed(context.Background(), "x")
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != http.StatusFound {
				t.Errorf("StatusCode = %d, want 302", apiErr.StatusCode)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("error = %v, want errors.Is ErrServer", err)
			}
			if got := calls.Load(); got != 3 {
				t.Errorf("wire hits = %d, want 3 (a sub-400 status is retried)", got)
			}
		})
	}

	t.Run("openai-compatible/200_error_object_unknown_type_retried", func(t *testing.T) {
		var calls atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"error":{"message":"model not loaded","type":"BadRequestError"}}`))
		}))
		defer srv.Close()

		emb := newTestEmbedder(t, "openai-compatible", srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond})
		_, err := emb.Embed(context.Background(), "x")
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %v, want *llmkit.APIError", err)
		}
		if apiErr.StatusCode != http.StatusOK {
			t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
		}
		if !errors.Is(err, llmkit.ErrServer) {
			t.Errorf("error = %v, want errors.Is ErrServer", err)
		}
		if got := calls.Load(); got != 3 {
			t.Errorf("wire hits = %d, want 3 (an untyped 200 error object is retried)", got)
		}
	})
}

// TestNon200_ClassifiedOnFullBody pins that a non-200 status is
// classified against the whole response body and only the returned
// Message is capped: a 400 whose context-length phrase starts past byte
// 200 is errors.Is(ErrContextTooLong), with Message at most 200 bytes plus
// the "..." marker. Mutant: truncate the body before classifying — the
// phrase is cut off and the 400 reads as ErrInvalidRequest.
func TestNon200_ClassifiedOnFullBody(t *testing.T) {
	body := strings.Repeat("x", 250) + " prompt is too long: context length exceeded"
	for _, backend := range backends {
		t.Run(backend+"/400_context_length_past_cap", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(body))
			}))
			defer srv.Close()

			emb := newTestEmbedder(t, backend, srv.URL, retry.Config{MaxAttempts: 1})
			_, err := emb.Embed(context.Background(), "x")
			if !errors.Is(err, llmkit.ErrContextTooLong) {
				t.Errorf("error = %v, want errors.Is ErrContextTooLong", err)
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *llmkit.APIError", err)
			}
			if max := 200 + len("..."); len(apiErr.Message) > max {
				t.Errorf("len(Message) = %d, want <= %d", len(apiErr.Message), max)
			}
		})
	}
}

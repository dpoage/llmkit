package adapter_test

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/embed"
	"github.com/dpoage/llmkit/retry"
)

// consumer is one raw-http.Client.Do consumer: decide.Ask, or embed.Embed
// over one backend. run makes one call with three attempts allowed and
// returns the number of backoff sleeps the retry stage took (attempts - 1
// when every attempt fails retryably) and the call's error.
type consumer struct {
	name string
	run  func(t *testing.T, baseURL string, hc *http.Client) (sleeps int, err error)
}

func consumers() []consumer {
	policy := func(sleeps *int) retry.Config {
		return retry.Config{
			MaxAttempts: 3,
			BaseDelay:   time.Millisecond,
			MaxDelay:    time.Millisecond,
			Rand:        func() float64 { return 0.5 },
			Sleep:       func(context.Context, time.Duration) error { *sleeps++; return nil },
		}
	}
	embedder := func(backend embed.Backend) func(*testing.T, string, *http.Client) (int, error) {
		return func(t *testing.T, baseURL string, hc *http.Client) (int, error) {
			t.Helper()
			var sleeps int
			e, err := embed.New(embed.Config{Backend: backend, Model: "m", BaseURL: baseURL, HTTPClient: hc, Retry: policy(&sleeps)})
			if err != nil {
				t.Fatalf("embed.New: %v", err)
			}
			_, err = e.Embed(context.Background(), "x")
			return sleeps, err
		}
	}
	return []consumer{
		{"decide", func(t *testing.T, baseURL string, hc *http.Client) (int, error) {
			t.Helper()
			var sleeps int
			c, err := decide.New(decide.Config{Secret: "sk-redirect-test", Model: "m", BaseURL: baseURL, HTTPClient: hc, Retry: policy(&sleeps)})
			if err != nil {
				t.Fatalf("decide.New: %v", err)
			}
			_, err = c.Ask(context.Background(), "state", decide.Questions{"q": decide.Noul{Instructions: "i"}})
			return sleeps, err
		}},
		{"embed-ollama", embedder(embed.BackendOllama)},
		{"embed-openai-compatible", embedder(embed.BackendOpenAICompatible)},
	}
}

// TestRedirectPolicyFailureIsTerminal pins A2: a server that redirects every
// request to itself, under the default policy and under a caller
// CheckRedirect that errors, costs one Do per attempt-set and is terminal.
func TestRedirectPolicyFailureIsTerminal(t *testing.T) {
	errPolicy := errors.New("caller refuses redirects")
	cases := []struct {
		name     string
		client   *http.Client
		wantHits int32
	}{
		{"default policy", nil, 10},
		{"caller CheckRedirect error", &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return errPolicy }}, 1},
	}
	for _, c := range consumers() {
		for _, tc := range cases {
			t.Run(c.name+"/"+tc.name, func(t *testing.T) {
				var hits atomic.Int32
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					hits.Add(1)
					http.Redirect(w, r, r.URL.Path, http.StatusFound)
				}))
				defer srv.Close()

				sleeps, err := c.run(t, srv.URL, tc.client)

				if got := hits.Load(); got != tc.wantHits {
					t.Errorf("server hits = %d, want %d (one Do; no retry)", got, tc.wantHits)
				}
				if sleeps != 0 {
					t.Errorf("backoff sleeps = %d, want 0 (terminal error)", sleeps)
				}
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Errorf("err = %v, want it to match ErrInvalidRequest", err)
				}
				if _, _, retryable := llmkit.Classify(err); retryable {
					t.Errorf("Classify(%v) = retryable, want terminal", err)
				}
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) || apiErr.StatusCode != 0 {
					t.Errorf("err = %v, want *APIError with StatusCode 0", err)
				}
				if tc.client != nil && !errors.Is(err, errPolicy) {
					t.Errorf("err = %v, want the caller's CheckRedirect error chained", err)
				}
			})
		}
	}
}

// TestNonRedirectTransportFailuresStayRetryable pins the A2 preserve rows:
// every failure with no response is still *APIError{ErrServer, 0}, retried
// to the attempt limit. A BaseURL with an unsupported scheme or no host is
// not such a failure: both consumers refuse it at construction (see
// ParseBaseURL).
func TestNonRedirectTransportFailuresStayRetryable(t *testing.T) {
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer plain.Close()
	tlsSrv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer tlsSrv.Close()
	eof := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = conn.Close()
		}
	}))
	defer eof.Close()

	// A port nothing listens on: bind, note the address, release.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	refused := ln.Addr().String()
	_ = ln.Close()

	cases := []struct {
		name   string
		url    string
		client *http.Client
	}{
		{"dial refused", "http://" + refused, nil},
		{"https client against a plain server", "https" + strings.TrimPrefix(plain.URL, "http"), nil},
		{"default client against a TLS server", tlsSrv.URL, nil},
		{"EOF before headers", eof.URL, nil},
	}
	for _, c := range consumers() {
		for _, tc := range cases {
			t.Run(c.name+"/"+tc.name, func(t *testing.T) {
				sleeps, err := c.run(t, tc.url, tc.client)
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) || apiErr.Kind != llmkit.ErrServer || apiErr.StatusCode != 0 {
					t.Fatalf("err = %v (%T), want *APIError{ErrServer, 0}", err, err)
				}
				if _, _, retryable := llmkit.Classify(err); !retryable {
					t.Errorf("Classify(%v) = terminal, want retryable", err)
				}
				if sleeps != 2 {
					t.Errorf("backoff sleeps = %d, want 2 (3 attempts)", sleeps)
				}
			})
		}
	}
}

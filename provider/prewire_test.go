package provider

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/internal/anthropic"
	"github.com/dpoage/llmkit/provider/internal/google"
	"github.com/dpoage/llmkit/provider/internal/openai"
)

// The zero-request rule (internal/adapter/wire.go): an adapter call that
// returns an error while zero requests reached the transport, with the
// caller's context live, is a refusal — ErrInvalidRequest, never retried.
// One failure that needs no server and nothing from llmkit's own request
// validation drives every adapter and entry point: a malformed BaseURL,
// which each vendor SDK rejects locally when it builds the request.

// wireRT is the transport under test: it counts every RoundTrip call and
// answers with fn's result. It never dials.
type wireRT struct {
	calls atomic.Int32
	fn    func(*http.Request) (*http.Response, error)
}

func (rt *wireRT) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.calls.Add(1)
	return rt.fn(req)
}

// unreachableRT fails every request the way a refused dial does.
func unreachableRT() *wireRT {
	return &wireRT{fn: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("dial tcp 127.0.0.1:1: connect: connection refused")
	}}
}

// wireEntry is one way into an adapter.
type wireEntry struct {
	name string
	call func(llmkit.Client, context.Context, llmkit.Request) error
}

func wireEntries() []wireEntry {
	return []wireEntry{
		{"Complete", func(c llmkit.Client, ctx context.Context, req llmkit.Request) error {
			_, err := c.Complete(ctx, req)
			return err
		}},
		{"Stream", func(c llmkit.Client, ctx context.Context, req llmkit.Request) error {
			_, err := c.(llmkit.StreamingClient).Stream(ctx, req, nil)
			return err
		}},
	}
}

const (
	// goodBaseURL is never dialed: the tests' transport answers first.
	goodBaseURL = "http://127.0.0.1:1"
	// badBaseURL fails url.Parse, so no SDK can build a request from it.
	badBaseURL = "http://[::1"
)

// wireClients builds every adapter on rt and baseURL; a nil rt leaves
// Options.HTTPClient unset, the shipped default. The adapters run with no
// retry wrapper so the returned error is the adapter's own.
func wireClients(t *testing.T, rt http.RoundTripper, baseURL string) map[string]llmkit.Client {
	t.Helper()
	var hc *http.Client
	if rt != nil {
		hc = &http.Client{Transport: rt}
	}
	out := map[string]llmkit.Client{}
	var err error
	if out["anthropic"], err = anthropic.New("claude-test", anthropic.Options{APIKey: "k", BaseURL: baseURL, HTTPClient: hc}); err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}
	if out["openai"], err = openai.New("gpt-test", openai.Options{APIKey: "k", BaseURL: baseURL, HTTPClient: hc}); err != nil {
		t.Fatalf("openai.New: %v", err)
	}
	if out["openai-compatible"], err = openai.New("llama-test", openai.Options{APIKey: "k", BaseURL: baseURL, HTTPClient: hc, Compatible: true}); err != nil {
		t.Fatalf("openai.New compatible: %v", err)
	}
	if out["google"], err = google.New(context.Background(), "gemini-test", google.Options{APIKey: "k", BaseURL: baseURL, HTTPClient: hc}); err != nil {
		t.Fatalf("google.New: %v", err)
	}
	return out
}

var wireAdapterNames = []string{"anthropic", "openai", "openai-compatible", "google"}

func validRequest() llmkit.Request {
	return llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}, MaxTokens: 64}
}

func wantRetry(t *testing.T, err error) bool {
	t.Helper()
	_, _, retryable := llmkit.Classify(err)
	return retryable
}

// TestZeroRequestIsRefusal: a call that fails with nothing sent is an
// ErrInvalidRequest refusal that Classify never retries, on every adapter
// and entry point.
func TestZeroRequestIsRefusal(t *testing.T) {
	for _, name := range wireAdapterNames {
		for _, entry := range wireEntries() {
			t.Run(name+"/"+entry.name, func(t *testing.T) {
				rt := unreachableRT()
				err := entry.call(wireClients(t, rt, badBaseURL)[name], context.Background(), validRequest())
				if err == nil {
					t.Fatal("call succeeded; want an error")
				}
				if n := rt.calls.Load(); n != 0 {
					t.Fatalf("transport saw %d requests; the probe needs a call that sends none (err: %v)", n, err)
				}
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Fatalf("err = %v; want ErrInvalidRequest", err)
				}
				if wantRetry(t, err) {
					t.Fatalf("err = %v; Classify says retry a request that was never sent", err)
				}
			})
		}
	}
}

// TestZeroRequestIsRefusal_DefaultHTTPClient runs the same refusal with
// Options.HTTPClient unset, so the client each adapter builds for itself is
// the one that has to count requests.
func TestZeroRequestIsRefusal_DefaultHTTPClient(t *testing.T) {
	for _, name := range wireAdapterNames {
		for _, entry := range wireEntries() {
			t.Run(name+"/"+entry.name, func(t *testing.T) {
				err := entry.call(wireClients(t, nil, badBaseURL)[name], context.Background(), validRequest())
				if !errors.Is(err, llmkit.ErrInvalidRequest) || wantRetry(t, err) {
					t.Fatalf("err = %v; want a non-retryable ErrInvalidRequest", err)
				}
			})
		}
	}
}

// TestAnthropicStreamingRequiredIsRefusal is the witness for the rule: the
// SDK's own size guard rejects a large non-streaming call locally with
// "streaming is required", so the call is a refusal, not a retryable server
// failure. Stream sends the same request, so it is not refused.
func TestAnthropicStreamingRequiredIsRefusal(t *testing.T) {
	rt := unreachableRT()
	req := llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
		MaxTokens: 1 << 20,
	}
	_, err := wireClients(t, rt, goodBaseURL)["anthropic"].Complete(context.Background(), req)
	if n := rt.calls.Load(); n != 0 {
		t.Fatalf("transport saw %d requests; want 0 (err: %v)", n, err)
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v; want ErrInvalidRequest", err)
	}
	if wantRetry(t, err) {
		t.Fatalf("err = %v; Classify says retry", err)
	}
}

// TestSentThenDialFailureStaysTransport: once a request reached the
// transport, even one that died dialing, the failure keeps its transport
// classification — a retryable ErrServer with no status.
func TestSentThenDialFailureStaysTransport(t *testing.T) {
	for _, name := range wireAdapterNames {
		for _, entry := range wireEntries() {
			t.Run(name+"/"+entry.name, func(t *testing.T) {
				rt := unreachableRT()
				err := entry.call(wireClients(t, rt, goodBaseURL)[name], context.Background(), validRequest())
				if rt.calls.Load() == 0 {
					t.Fatalf("transport saw no request (err: %v)", err)
				}
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != 0 {
					t.Fatalf("err = %#v; want *APIError{ErrServer, StatusCode 0}", err)
				}
				if errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Fatalf("err = %v; a sent request is not a refusal", err)
				}
				if !wantRetry(t, err) {
					t.Fatalf("err = %v; a dial failure must stay retryable", err)
				}
			})
		}
	}
}

// TestCanceledBeforeSendStaysCancellation: with the caller's context already
// canceled the zero-request rule does not apply — the error is a plain
// error chaining context.Canceled, never an *APIError, so Classify treats
// it as terminal for the right reason. Both a request that would have been
// sent and one the SDK cannot build must behave the same.
func TestCanceledBeforeSendStaysCancellation(t *testing.T) {
	for _, name := range wireAdapterNames {
		for _, entry := range wireEntries() {
			for baseName, baseURL := range map[string]string{"sendable": goodBaseURL, "unsendable": badBaseURL} {
				t.Run(name+"/"+entry.name+"/"+baseName, func(t *testing.T) {
					ctx, cancel := context.WithCancel(context.Background())
					cancel()
					err := entry.call(wireClients(t, unreachableRT(), baseURL)[name], ctx, validRequest())
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("err = %v; want it to chain context.Canceled", err)
					}
					var apiErr *llmkit.APIError
					if errors.As(err, &apiErr) {
						t.Fatalf("err = %#v; a cancellation must be a plain error, not an *APIError", err)
					}
				})
			}
		}
	}
}

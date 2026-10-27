package adapter

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func newReq(t *testing.T, ctx context.Context) *http.Request {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://127.0.0.1:1/", nil)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

// TestWireTransport_CountsEveryRoundTrip: a request counts on entry, so one
// that fails dialing is still a request; the status of the last response is
// kept; a request whose context has no Wire passes through untouched.
func TestWireTransport_CountsEveryRoundTrip(t *testing.T) {
	dialErr := errors.New("connection refused")
	status := 0
	rt := WireTransport(rtFunc(func(*http.Request) (*http.Response, error) {
		if status == 0 {
			return nil, dialErr
		}
		return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(""))}, nil
	}))

	ctx, w := WithWire(context.Background())
	if _, err := rt.RoundTrip(newReq(t, ctx)); !errors.Is(err, dialErr) {
		t.Fatalf("err = %v; want the dial error passed through", err)
	}
	if w.Requests() != 1 || w.Status() != 0 {
		t.Fatalf("after dial failure: requests=%d status=%d; want 1, 0", w.Requests(), w.Status())
	}
	status = 503
	if _, err := rt.RoundTrip(newReq(t, ctx)); err != nil {
		t.Fatal(err)
	}
	status = 404
	if _, err := rt.RoundTrip(newReq(t, ctx)); err != nil {
		t.Fatal(err)
	}
	if w.Requests() != 3 || w.Status() != 404 {
		t.Fatalf("requests=%d status=%d; want 3, 404 (last response wins)", w.Requests(), w.Status())
	}
	if got := WireStatus(ctx); got != 404 {
		t.Fatalf("WireStatus = %d; want 404", got)
	}

	if _, err := rt.RoundTrip(newReq(t, context.Background())); err != nil {
		t.Fatalf("request without a Wire: %v", err)
	}
	if got := WireStatus(context.Background()); got != 0 {
		t.Fatalf("WireStatus without a Wire = %d; want 0", got)
	}
}

// TestNoResponseError pins the zero-request rule's boundary: it fires only
// with a Wire that saw nothing and a live caller context.
func TestNoResponseError(t *testing.T) {
	cause := errors.New("sdk: streaming is required")
	send := func(t *testing.T, ctx context.Context) {
		t.Helper()
		rt := WireTransport(rtFunc(func(*http.Request) (*http.Response, error) { return nil, errors.New("dial") }))
		_, _ = rt.RoundTrip(newReq(t, ctx))
	}

	t.Run("nothing sent is a refusal", func(t *testing.T) {
		ctx, _ := WithWire(context.Background())
		err := NoResponseError("p", ctx, cause, nil)
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrInvalidRequest) || apiErr.Provider != "p" || apiErr.StatusCode != 0 {
			t.Fatalf("err = %#v; want *APIError{ErrInvalidRequest, provider p, status 0}", err)
		}
		if !errors.Is(err, cause) {
			t.Fatalf("err = %v; want the SDK error chained", err)
		}
		if _, _, retry := llmkit.Classify(err); retry {
			t.Fatal("a refusal must not be retryable")
		}
	})

	t.Run("a sent request keeps the transport class", func(t *testing.T) {
		ctx, _ := WithWire(context.Background())
		send(t, ctx)
		err := NoResponseError("p", ctx, cause, nil)
		if !errors.Is(err, llmkit.ErrServer) {
			t.Fatalf("err = %v; want ErrServer", err)
		}
		if _, _, retry := llmkit.Classify(err); !retry {
			t.Fatal("a transport failure must stay retryable")
		}
	})

	t.Run("a canceled caller is not a refusal", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		ctx, _ = WithWire(ctx)
		cancel()
		err := NoResponseError("p", ctx, cause, nil)
		var apiErr *llmkit.APIError
		if !errors.Is(err, context.Canceled) || errors.As(err, &apiErr) {
			t.Fatalf("err = %#v; want a plain error chaining context.Canceled", err)
		}
	})

	t.Run("no Wire cannot prove zero requests", func(t *testing.T) {
		err := NoResponseError("p", context.Background(), cause, nil)
		if !errors.Is(err, llmkit.ErrServer) {
			t.Fatalf("err = %v; want the TransportError classification", err)
		}
	})
}

func TestNoResponseError_Redacts(t *testing.T) {
	const key = "sk-WIREPROBE-0123456789abcdef"
	r := NewRedactor(key, nil)
	frag := func(s string) bool {
		for i := 0; i+12 <= len(key); i++ {
			if strings.Contains(s, key[i:i+12]) {
				return true
			}
		}
		return false
	}
	quoting := func(pad int, k string) error {
		return errors.New(`Post "http://h/` + strings.Repeat("x", pad) + "/" + k + `/%2": invalid URL escape`)
	}
	dial := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
	send := func(t *testing.T, ctx context.Context) {
		t.Helper()
		rt := WireTransport(rtFunc(func(*http.Request) (*http.Response, error) { return nil, dial }))
		_, _ = rt.RoundTrip(newReq(t, ctx))
	}

	t.Run("refusal is redacted before the cap", func(t *testing.T) {
		ctx, _ := WithWire(context.Background())
		// "request not sent: " (18 bytes) and `Post "http://h/` (15) put the key at byte 184.
		err := NoResponseError("p", ctx, quoting(150, key), r)
		var api *llmkit.APIError
		if !errors.As(err, &api) || !errors.Is(err, llmkit.ErrInvalidRequest) || frag(api.Message) || frag(err.Error()) {
			t.Fatalf("err = %v; want a refusal APIError holding no fragment of the key", err)
		}
		if !strings.HasSuffix(api.Message, "...") {
			t.Fatalf("Message = %q; want the cap to have applied (the key straddled byte 200)", api.Message)
		}
	})

	t.Run("a sent request is scrubbed", func(t *testing.T) {
		ctx, _ := WithWire(context.Background())
		send(t, ctx)
		err := NoResponseError("p", ctx, fmt.Errorf("%s: %w", quoting(0, key), dial), r)
		if !errors.Is(err, llmkit.ErrServer) || frag(err.Error()) {
			t.Fatalf("err = %v; want ErrServer with the key masked", err)
		}
	})

	t.Run("a canceled caller is scrubbed", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		ctx, _ = WithWire(ctx)
		cancel()
		err := NoResponseError("p", ctx, quoting(0, key), r)
		var api *llmkit.APIError
		if !errors.Is(err, context.Canceled) || errors.As(err, &api) || frag(err.Error()) {
			t.Fatalf("err = %v; want a plain error chaining context.Canceled with the key masked", err)
		}
	})

	t.Run("nothing to mask keeps the chain", func(t *testing.T) {
		ctx, _ := WithWire(context.Background())
		send(t, ctx)
		err := NoResponseError("p", ctx, fmt.Errorf("Post %q: %w", "http://h/clean", dial), r)
		var op *net.OpError
		if !errors.As(err, &op) {
			t.Fatalf("err = %v; want the *net.OpError still in the chain", err)
		}
	})
}

package adapter

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

type closeSpy struct {
	io.Reader
	closed bool
}

func (c *closeSpy) Close() error { c.closed = true; return nil }

func TestDoError(t *testing.T) {
	policyErr := errors.New(`Post "http://x/": stopped after 10 redirects`)

	t.Run("response with error is a terminal redirect-policy failure", func(t *testing.T) {
		body := &closeSpy{Reader: strings.NewReader("")}
		err := DoError("decide", context.Background(), &http.Response{Body: body}, policyErr)
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
		}
		if apiErr.Kind != llmkit.ErrInvalidRequest || apiErr.StatusCode != 0 || apiErr.Provider != "decide" {
			t.Errorf("APIError = %+v, want Kind ErrInvalidRequest, StatusCode 0, Provider decide", apiErr)
		}
		if !errors.Is(err, policyErr) {
			t.Errorf("err = %v, want the Do error chained", err)
		}
		if _, _, retryable := llmkit.Classify(err); retryable {
			t.Errorf("Classify = retryable, want terminal")
		}
		if !body.closed {
			t.Errorf("resp.Body not closed")
		}
	})

	t.Run("no response is a retryable transport failure", func(t *testing.T) {
		err := DoError("embed", context.Background(), nil, errors.New("dial tcp: refused"))
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) || apiErr.Kind != llmkit.ErrServer || apiErr.StatusCode != 0 {
			t.Fatalf("err = %v, want *APIError{ErrServer, 0}", err)
		}
		if _, _, retryable := llmkit.Classify(err); !retryable {
			t.Errorf("Classify = terminal, want retryable")
		}
	})

	t.Run("caller cancellation wins over a response", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		body := &closeSpy{Reader: strings.NewReader("")}
		err := DoError("decide", ctx, &http.Response{Body: body}, policyErr)
		var apiErr *llmkit.APIError
		if errors.As(err, &apiErr) || !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v (%T), want a plain error chaining context.Canceled", err, err)
		}
	})
}

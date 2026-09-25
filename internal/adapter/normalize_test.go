package adapter

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNormalizeSDKError_InBandTypes pins the in-band table: sub-400 status
// classifies by vendor type, with unknown or empty type falling back to
// ErrServer.
func TestNormalizeSDKError_InBandTypes(t *testing.T) {
	rows := []struct {
		typ  string
		want error
	}{
		{"overloaded_error", llmkit.ErrOverloaded},
		{"rate_limit_error", llmkit.ErrRateLimited},
		{"api_error", llmkit.ErrServer},
		{"server_error", llmkit.ErrServer},
		{"timeout_error", llmkit.ErrServer},
		{"invalid_request_error", llmkit.ErrInvalidRequest},
		{"not_found_error", llmkit.ErrInvalidRequest},
		{"authentication_error", llmkit.ErrAuth},
		{"permission_error", llmkit.ErrAuth},
		{"billing_error", llmkit.ErrAuth},
		{"insufficient_quota", llmkit.ErrAuth},
		{"no_such_vendor_type", llmkit.ErrServer}, // unknown type: fallback
		{"", llmkit.ErrServer},                    // absent type: fallback
	}
	for _, r := range rows {
		err := NormalizeSDKError("anthropic", VendorError{Status: 200, Type: r.typ, Message: "m"})
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) {
			t.Errorf("type %q: err = %v (%T), want *llmkit.APIError", r.typ, err, err)
			continue
		}
		if apiErr.Kind != r.want {
			t.Errorf("type %q: Kind = %v, want %v", r.typ, apiErr.Kind, r.want)
		}
		if apiErr.StatusCode != 200 {
			t.Errorf("type %q: StatusCode = %d, want 200 (the committed stream's status)", r.typ, apiErr.StatusCode)
		}
	}
}

func TestTransportError_CanceledText(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	t.Run("sdk error already chains canceled", func(t *testing.T) {
		err := TransportError("anthropic", ctx, context.Canceled)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled in the chain", err)
		}
		var apiErr *llmkit.APIError
		if errors.As(err, &apiErr) {
			t.Errorf("err = %v (%T), want no *llmkit.APIError for a cancellation", err, err)
		}
		if got, want := err.Error(), "llmkit: anthropic: context canceled"; got != want {
			t.Errorf("err.Error() = %q, want %q (provider once, cause once)", got, want)
		}
	})

	t.Run("sdk error does not chain canceled", func(t *testing.T) {
		sdkErr := errors.New("connection reset")
		err := TransportError("anthropic", ctx, sdkErr)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled in the chain", err)
		}
		if !strings.Contains(err.Error(), "llmkit: anthropic: connection reset: context canceled") {
			t.Errorf("err = %v, want the provider, the SDK error, and the cancellation chained in order", err)
		}
	})
}

func TestResponseHeader(t *testing.T) {
	if got := ResponseHeader(nil); got != nil {
		t.Errorf("ResponseHeader(nil) = %v, want nil", got)
	}
}

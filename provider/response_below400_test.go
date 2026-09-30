package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// inBandError is the JSON error object a gateway may answer a sub-400
// status with. Its error.type is an Anthropic/OpenAI vendor type; Gemini's
// errors carry no such field.
const inBandError = `{"error":{"type":"invalid_request_error","message":"boom"}}`

// TestResponseBelow400_NoCompletionIsAPIError is the below-400 grid: every
// adapter's Complete, and google's Stream, on every status in {200, 201,
// 204, 299, 302 without Location} and every body in {a JSON vendor error
// object, "{}", empty} returns a non-nil *llmkit.APIError and never a nil
// error.
//
// Two routes produce that error, and the grid names which row takes which:
//
//   - the no-completion check (the SDK decoded the response, it carried no
//     completion): StatusCode is the response status, Kind follows the
//     body's error.type through the vendor table where the adapter can see
//     the body (anthropic, openai, openai-compatible) and is ErrServer
//     otherwise;
//   - the SDK decode route (the SDK could not decode the response at all,
//     e.g. EOF on an empty body): a transport-classified ErrServer with
//     StatusCode 0 — no check runs, none is needed.
func TestResponseBelow400_NoCompletionIsAPIError(t *testing.T) {
	statuses := []int{200, 201, 204, 299, 302}
	bodies := []struct{ name, body string }{
		{"error-object", inBandError},
		{"empty-object", `{}`},
		{"empty-body", ``},
	}
	for _, f := range allAdapters() {
		for _, mode := range []string{"complete", "stream"} {
			if mode == "stream" && f.name != "google" {
				continue
			}
			for _, st := range statuses {
				for _, b := range bodies {
					body := b.body
					if st == 204 {
						if b.name != "empty-body" {
							continue // a 204 cannot carry a body
						}
					}
					t.Run(fmt.Sprintf("%s/%s/%d/%s", f.name, mode, st, b.name), func(t *testing.T) {
						base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
							w.Header().Set("Content-Type", "application/json")
							w.WriteHeader(st)
							if st != 204 {
								_, _ = w.Write([]byte(body))
							}
						})
						client := f.build(t, base)
						var err error
						if mode == "complete" {
							_, err = client.Complete(context.Background(), simpleRequest())
						} else {
							_, err = llmkit.Stream(context.Background(), client, simpleRequest(), func(llmkit.Delta) error { return nil })
						}
						var apiErr *llmkit.APIError
						if !errors.As(err, &apiErr) {
							t.Fatalf("err = %v (%T), want *llmkit.APIError, never nil", err, err)
						}
						route := decodeRoute(f.name, mode, st, b.name)
						if route {
							if !errors.Is(err, llmkit.ErrServer) {
								t.Errorf("SDK-decode row: Kind = %v, want ErrServer", apiErr.Kind)
							}
							return
						}
						if apiErr.StatusCode != st {
							t.Errorf("StatusCode = %d, want the response status %d", apiErr.StatusCode, st)
						}
						wantKind := llmkit.ErrServer
						if f.name != "google" && b.name == "error-object" {
							wantKind = llmkit.ErrInvalidRequest
						}
						if !errors.Is(err, wantKind) {
							t.Errorf("Kind = %v, want %v", apiErr.Kind, wantKind)
						}
						if f.name != "google" && b.name == "error-object" && !strings.Contains(apiErr.Message, "boom") {
							t.Errorf("Message = %q, want the body's message", apiErr.Message)
						}
					})
				}
			}
		}
	}
}

// decodeRoute names the grid rows that fail inside the vendor SDK's
// decoder instead of reaching the no-completion check: anthropic, openai
// and openai-compatible Complete on an empty body (EOF), and google Stream
// on a 2xx "{}" body, a line genai rejects as an invalid stream chunk (a
// non-2xx one is reported by genai as an APIError with the status).
func decodeRoute(provider, mode string, status int, body string) bool {
	if provider != "google" {
		return body == "empty-body"
	}
	return mode == "stream" && body == "empty-object" && status/100 == 2
}

// TestResponseBelow400_NoCompletionCarriesRetryAfter pins that the
// no-completion error on anthropic, openai and openai-compatible Complete
// reports the response's Retry-After like any other response error: a 200
// carrying an in-band rate_limit_error and `Retry-After: 7` is
// ErrRateLimited with HasRetryAfter and RetryAfter 7s, and a `{}` body with
// the same header is ErrServer with the same Retry-After.
func TestResponseBelow400_NoCompletionCarriesRetryAfter(t *testing.T) {
	bodies := []struct {
		name, body string
		kind       error
	}{
		{"rate-limit-object", `{"error":{"type":"rate_limit_error","message":"slow"}}`, llmkit.ErrRateLimited},
		{"empty-object", `{}`, llmkit.ErrServer},
	}
	for _, f := range allAdapters() {
		if f.name == "google" {
			continue // genai hides response headers
		}
		for _, b := range bodies {
			t.Run(f.name+"/"+b.name, func(t *testing.T) {
				base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("Retry-After", "7")
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(b.body))
				})
				_, err := f.build(t, base).Complete(context.Background(), simpleRequest())
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
				}
				if !errors.Is(err, b.kind) {
					t.Errorf("Kind = %v, want %v", apiErr.Kind, b.kind)
				}
				if apiErr.StatusCode != http.StatusOK {
					t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
				}
				if !apiErr.HasRetryAfter || apiErr.RetryAfter != 7*time.Second {
					t.Errorf("HasRetryAfter/RetryAfter = %v/%v, want true/7s", apiErr.HasRetryAfter, apiErr.RetryAfter)
				}
			})
		}
	}
}

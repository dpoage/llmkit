package adapter

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

// wantCapped is the Message-cap oracle, written from the rule and not from
// the implementation: t when len(t) <= 200, else the longest rune-aligned
// prefix of at most 200 bytes plus "...".
func wantCapped(t string) string {
	if len(t) <= 200 {
		return t
	}
	end := 0
	for i, r := range t {
		if i+utf8.RuneLen(r) > 200 {
			break
		}
		end = i + utf8.RuneLen(r)
	}
	return t[:end] + "..."
}

// capWitnesses are the texts every APIError-producing entry point must cap
// identically.
func capWitnesses() map[string]string {
	longURL := "http://127.0.0.1:1/" + strings.Repeat("p", 400)
	urlErr := (&url.Error{Op: "Post", URL: longURL, Err: errors.New("connection refused")}).Error()
	return map[string]string{
		"empty":                  "",
		"short":                  "boom",
		"199a+eee":               strings.Repeat("a", 199) + "ééé plain",
		"exactly 200 ascii":      strings.Repeat("a", 200),
		"exactly 200 multibyte":  strings.Repeat("a", 198) + "é",
		"201 ascii":              strings.Repeat("a", 201),
		"201 ending multibyte":   strings.Repeat("a", 199) + "é",
		"198a+4-byte rune":       strings.Repeat("a", 198) + "😀 tail",
		"200 then 4-byte rune":   strings.Repeat("a", 200) + "😀",
		"all multibyte":          strings.Repeat("é", 300),
		"400-char URL url.Error": urlErr,
	}
}

func checkCapped(t *testing.T, route string, err error, text string) {
	t.Helper()
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("%s: err = %v (%T), want *llmkit.APIError", route, err, err)
	}
	if !utf8.ValidString(apiErr.Message) {
		t.Errorf("%s: Message is not valid UTF-8: %q", route, apiErr.Message)
	}
	if want := wantCapped(text); apiErr.Message != want {
		t.Errorf("%s: Message = %q (len %d), want %q (len %d)", route, apiErr.Message, len(apiErr.Message), want, len(want))
	}
}

// TestMessageCap pins the one cap rule on every route internal/adapter
// builds an *APIError from.
func TestMessageCap(t *testing.T) {
	ctx := context.Background()
	for name, text := range capWitnesses() {
		t.Run(name, func(t *testing.T) {
			checkCapped(t, "NormalizeSDKError status", NormalizeSDKError("p", VendorError{Status: 500, Message: text}), text)
			checkCapped(t, "NormalizeSDKError in-band", NormalizeSDKError("p", VendorError{Status: 200, Type: "overloaded_error", Message: text}), text)
			// The witnesses carry no surrounding whitespace, so NormalizeCapped's
			// trim is the identity on them.
			checkCapped(t, "NormalizeCapped", NormalizeCapped("p", VendorError{Status: 500, Message: text}), text)
			checkCapped(t, "TransportError", TransportError("p", ctx, errors.New(text)), text)
			checkCapped(t, "DoError resp==nil", DoError("p", ctx, nil, errors.New(text)), text)
			resp := &http.Response{Body: io.NopCloser(strings.NewReader(""))}
			checkCapped(t, "DoError resp!=nil", DoError("p", ctx, resp, errors.New(text)), text)
		})
	}
}

// TestMessageCap_ClassifiesWholeText pins that classification reads past
// the cap: a context-length phrase after byte 200 still classifies.
func TestMessageCap_ClassifiesWholeText(t *testing.T) {
	text := strings.Repeat("x", 300) + " maximum context length exceeded"
	err := NormalizeSDKError("p", VendorError{Status: http.StatusBadRequest, Message: text})
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Errorf("NormalizeSDKError: err = %v, want ErrContextTooLong", err)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || len(apiErr.Message) != 203 {
		t.Errorf("NormalizeSDKError Message = %v, want 203 bytes (200 + ...)", apiErr)
	}
	err = NormalizeCapped("p", VendorError{Status: http.StatusBadRequest, Message: "  " + text + "\n"})
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Errorf("NormalizeCapped: err = %v, want ErrContextTooLong", err)
	}
}

// TestNormalizeCapped_Trims pins the trim NormalizeCapped adds on top of
// NormalizeSDKError, which leaves whitespace alone.
func TestNormalizeCapped_Trims(t *testing.T) {
	var apiErr *llmkit.APIError
	if !errors.As(NormalizeCapped("p", VendorError{Status: 500, Message: " \n body\t "}), &apiErr) || apiErr.Message != "body" {
		t.Errorf("NormalizeCapped Message = %q, want %q", apiErr.Message, "body")
	}
	if !errors.As(NormalizeSDKError("p", VendorError{Status: 500, Message: " body "}), &apiErr) || apiErr.Message != " body " {
		t.Errorf("NormalizeSDKError Message = %q, want it untrimmed", apiErr.Message)
	}
}

// TestMessageCap_FullTextStaysInChain pins that the cap cuts only the
// Message: the SDK error chained through Unwrap carries the whole text.
func TestMessageCap_FullTextStaysInChain(t *testing.T) {
	text := strings.Repeat("z", 450)
	err := TransportError("p", context.Background(), errors.New(text))
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || apiErr.Err == nil || apiErr.Err.Error() != text {
		t.Fatalf("chained Err = %v, want the full %d-byte text", apiErr, len(text))
	}
}

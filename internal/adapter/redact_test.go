package adapter

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

const redactSecret = "sk-live-9f8e7d6c5b4a39281706f5e4d3c2b1a0"

func mustRoot(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := ParseBaseURL("test: BaseURL", raw)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func TestRedactorText(t *testing.T) {
	cases := []struct {
		name   string
		secret string
		root   string
		in     string
		want   string
	}{
		{"secret anywhere", redactSecret, "", "bad key " + redactSecret + "!", "bad key ***!"},
		{"secret twice", "abcdefgh", "", "abcdefgh abcdefgh", "*** ***"},
		{"one-byte secret", "k", "", "kick", "***ic***"},
		{"empty secret redacts nothing", "", "", "any text", "any text"},
		{"url password", "", "http://user:hunter2@127.0.0.1:1", "pw hunter2 here", "pw *** here"},
		{"short password masks matching text", "", "http://user:pw@127.0.0.1:1", "upstream pw rejected", "upstream *** rejected"},
		{"username is not a value", "", "http://user:hunter2@127.0.0.1:1", "unauthorized user u", "unauthorized user u"},
		{"one-byte username stays", "", "http://u:hunter2@127.0.0.1:1", "unauthorized user u", "unauthorized user u"},
		{"basic credential blob", "", "http://user:pw12345@127.0.0.1:1", "Authorization: Basic dXNlcjpwdzEyMzQ1", "Authorization: Basic ***"},
		{"json-escaped secret", "ab\"cd\\ef<gh", "", `{"e":"ab\"cd\\ef\u003cgh and ab\"cd\\ef<gh"}`, `{"e":"*** and ***"}`},
		{"userinfo with password", "", "", `Post "http://alice:secret@host:1/x": dial`, `Post "http://***@host:1/x": dial`},
		{"userinfo username only", "", "", `Get "https://sk-USERKEY@127.0.0.1:9/v1": refused`, `Get "https://***@127.0.0.1:9/v1": refused`},
		{"userinfo password net/http masked form", "", "", `Post "http://user:***@127.0.0.1:1/v1": refused`, `Post "http://***@127.0.0.1:1/v1": refused`},
		{"userinfo with at sign", "", "", "http://a@b@host/x", "http://***@host/x"},
		{"two userinfos", "", "", "http://a@h1/ then ftp://b:c@h2", "http://***@h1/ then ftp://***@h2"},
		{"no userinfo after path at", "", "", "http://host/path@x", "http://host/path@x"},
		{"no scheme no userinfo scrub", "", "", "user@host", "user@host"},
		{"empty scheme", "", "", "://a@h", "://a@h"},
		{"secret star", "a*b", "", "x a*b y", "x !!! y"},
		{"secret is the mask", "***", "", "key *** here", "key !!! here"},
		{"secret join", "abab", "", "aabab b", "a*** b"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var root *url.URL
			if tc.root != "" {
				root = mustRoot(t, tc.root)
			}
			got, changed := NewRedactor(tc.secret, root).Text(tc.in)
			if got != tc.want {
				t.Errorf("Text(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if changed != (tc.want != tc.in) {
				t.Errorf("changed = %v for %q -> %q", changed, tc.in, got)
			}
		})
	}
}

func TestRedactorTextNeverHoldsAValue(t *testing.T) {
	values := []string{"*", "**", "***", "*#", "a*", "\"*\"", "#*~"}
	inputs := []string{"*", "***", "****", "a**", "a*a*", "#*~*", "x*y**z", "\"*\"*"}
	for _, v := range values {
		r := NewRedactor(v, nil)
		for _, in := range inputs {
			out, _ := r.Text(in)
			if strings.Contains(out, v) {
				t.Errorf("secret %q: Text(%q) = %q still holds it", v, in, out)
			}
		}
	}
}

func TestRedactorNilAndEmptyRedactNothing(t *testing.T) {
	var nilR *Redactor
	if got, changed := nilR.Text("a b"); got != "a b" || changed {
		t.Errorf("nil Text = %q, %v", got, changed)
	}
	err := errors.New("boom")
	if got := nilR.Scrub(err); got != err {
		t.Errorf("nil Scrub returned %v, want the same error", got)
	}
	if got := NewRedactor("", nil).Scrub(err); got != err {
		t.Errorf("empty-secret Scrub returned %v, want the same error", got)
	}
}

// timeoutErr is a net.Error the way an expired per-attempt deadline is.
type timeoutErr struct{}

func (timeoutErr) Error() string   { return "i/o timeout" }
func (timeoutErr) Timeout() bool   { return true }
func (timeoutErr) Temporary() bool { return true }

func TestScrubReturnsCleanErrorsUnchanged(t *testing.T) {
	r := NewRedactor(redactSecret, mustRoot(t, "http://user:hunter2@127.0.0.1:1"))
	clean := []error{
		timeoutErr{},
		fmt.Errorf("ollama: %w", timeoutErr{}),
		NormalizeSDKError("p", VendorError{Status: 500, Message: "boom", Err: timeoutErr{}}),
		errors.Join(errors.New("a"), errors.New("b")),
		context.Canceled,
	}
	for _, err := range clean {
		got := r.Scrub(err)
		if got != err {
			t.Errorf("Scrub(%T %q) rebuilt a clean error", err, err)
		}
		var ne net.Error
		if errors.As(err, &ne) && !errors.As(got, &ne) {
			t.Errorf("Scrub(%q) lost the net.Error", err)
		}
	}
}

func TestScrubRebuildsAnAPIError(t *testing.T) {
	r := NewRedactor(redactSecret, nil)
	orig := &llmkit.APIError{
		Kind:          llmkit.ErrRateLimited,
		StatusCode:    429,
		RetryAfter:    7 * time.Second,
		HasRetryAfter: true,
		Provider:      "typesafe",
		Message:       "key " + redactSecret + " rejected",
		Err:           fmt.Errorf("wrapped %s: %w", redactSecret, timeoutErr{}),
	}
	got := r.Scrub(fmt.Errorf("outer: %w", orig))
	var api *llmkit.APIError
	if !errors.As(got, &api) {
		t.Fatalf("Scrub returned %T, want an *APIError in the chain", got)
	}
	want := llmkit.APIError{Kind: llmkit.ErrRateLimited, StatusCode: 429, RetryAfter: 7 * time.Second, HasRetryAfter: true, Provider: "typesafe", Message: "key *** rejected"}
	if api.Kind != want.Kind || api.StatusCode != want.StatusCode || api.RetryAfter != want.RetryAfter ||
		api.HasRetryAfter != want.HasRetryAfter || api.Provider != want.Provider || api.Message != want.Message {
		t.Errorf("rebuilt = %+v, want %+v", *api, want)
	}
	if !errors.Is(got, llmkit.ErrRateLimited) {
		t.Error("Kind sentinel lost")
	}
	if api.Err != nil {
		t.Errorf("chain not cut: Err = %v", api.Err)
	}
	var ne net.Error
	if errors.As(got, &ne) {
		t.Error("net.Error survived a redaction; the chain must be cut")
	}
	if _, _, retryable := llmkit.Classify(got); !retryable {
		t.Error("Classify(rebuilt ErrRateLimited) = terminal")
	}
}

func TestScrubKeepsContextErrors(t *testing.T) {
	r := NewRedactor(redactSecret, nil)
	for _, ctxErr := range []error{context.Canceled, context.DeadlineExceeded} {
		plain := fmt.Errorf("Post \"http://%s@h\": %w", redactSecret, ctxErr)
		api := NormalizeSDKError("p", VendorError{Message: plain.Error(), Err: plain})
		for name, err := range map[string]error{"plain": plain, "apierror": api} {
			got := r.Scrub(err)
			if !errors.Is(got, ctxErr) {
				t.Errorf("%s/%v: errors.Is lost after Scrub: %v", name, ctxErr, got)
			}
			assertNoText(t, got, redactSecret[:8])
		}
	}
	// A context error the original never matched is not invented.
	got := r.Scrub(errors.New("bad " + redactSecret))
	if errors.Is(got, context.Canceled) || errors.Is(got, context.DeadlineExceeded) {
		t.Errorf("Scrub invented a context error: %v", got)
	}
}

func TestScrubKeepsLLMKitSentinelsOnPlainErrors(t *testing.T) {
	r := NewRedactor(redactSecret, nil)
	err := fmt.Errorf("x %s: %w", redactSecret, llmkit.ErrInvalidRequest)
	got := r.Scrub(err)
	if !errors.Is(got, llmkit.ErrInvalidRequest) {
		t.Errorf("sentinel lost: %v", got)
	}
	assertNoText(t, got, redactSecret[:8])
}

func TestScrubWalksJoinedChains(t *testing.T) {
	r := NewRedactor(redactSecret, nil)
	err := errors.Join(errors.New("ok"), fmt.Errorf("deep: %w", errors.Join(errors.New("fine"), errors.New("k "+redactSecret))))
	got := r.Scrub(err)
	if got == err {
		t.Fatal("a secret in a joined branch was not found")
	}
	assertNoText(t, got, redactSecret[:8])
}

// A transport APIError's Message is the cap of its Err's text. When the cap
// cuts through the credential, redacting the cap would keep its front;
// Scrub redacts the text the cap was cut from.
func TestScrubReCapsATransportMessageCutThroughACredential(t *testing.T) {
	r := NewRedactor(redactSecret, nil)
	for _, off := range []int{150, 170, 185, 190, 192, 198} {
		src := fmt.Errorf("%s%s tail", strings.Repeat("x", off), redactSecret)
		err := NormalizeSDKError("p", VendorError{Message: src.Error(), Err: src})
		if !strings.Contains(err.(*llmkit.APIError).Message, redactSecret[:8]) && off < 192 {
			t.Fatalf("offset %d: setup did not leave the credential's front in the capped Message", off)
		}
		got := r.Scrub(err)
		assertNoText(t, got, redactSecret[:8])
	}
}

func TestNormalizeSDKErrorRedactsBeforeTheCap(t *testing.T) {
	r := NewRedactor(redactSecret, nil)
	for _, off := range []int{0, 100, 185, 190, 192, 199, 200, 250} {
		body := strings.Repeat("x", off) + redactSecret + strings.Repeat("y", 300)
		err := NormalizeSDKError("p", VendorError{Status: 500, Message: body, Redact: r})
		api := err.(*llmkit.APIError)
		if strings.Contains(api.Message, redactSecret[:8]) {
			t.Errorf("offset %d: Message holds the credential's front: %q", off, api.Message)
		}
		if len(api.Message) > maxMessageBytes+len("...") {
			t.Errorf("offset %d: Message is %d bytes", off, len(api.Message))
		}
	}
}

func TestNormalizeSDKErrorClassifiesOnTheUnredactedText(t *testing.T) {
	secret := "context length"
	r := NewRedactor(secret, nil)
	err := NormalizeSDKError("p", VendorError{Status: 400, Message: "maximum context length is 8192 tokens", Redact: r})
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Errorf("Kind = %v, want ErrContextTooLong", err)
	}
	if got := err.(*llmkit.APIError).Message; got != "maximum *** is 8192 tokens" {
		t.Errorf("Message = %q", got)
	}
}

func TestNormalizeSDKErrorWithoutRedactIsUnchanged(t *testing.T) {
	cause := timeoutErr{}
	err := NormalizeSDKError("p", VendorError{Status: 500, Message: redactSecret, Err: cause, Header: http.Header{}})
	api := err.(*llmkit.APIError)
	if api.Message != redactSecret || api.Err != error(cause) {
		t.Errorf("nil Redact changed the error: %+v", api)
	}
}

func assertNoText(t *testing.T, err error, forbidden string) {
	t.Helper()
	var walk func(error)
	walk = func(e error) {
		if e == nil {
			return
		}
		if strings.Contains(e.Error(), forbidden) {
			t.Errorf("%T text %q holds %q", e, e.Error(), forbidden)
		}
		switch u := e.(type) {
		case interface{ Unwrap() error }:
			walk(u.Unwrap())
		case interface{ Unwrap() []error }:
			for _, c := range u.Unwrap() {
				walk(c)
			}
		}
	}
	walk(err)
	if s := fmt.Sprintf("%+v", err); strings.Contains(s, forbidden) {
		t.Errorf("%%+v holds %q: %s", forbidden, s)
	}
}

// TestRedactorTextTerminatesAndHoldsNoValue runs every value and every
// input over an alphabet chosen to make redaction and userinfo replacement
// feed each other: Text must return, and the result holds no value.
func TestRedactorTextTerminatesAndHoldsNoValue(t *testing.T) {
	const alphabet = "a*@/:!"
	var words func(n int) []string
	words = func(n int) []string {
		if n == 0 {
			return []string{""}
		}
		var out []string
		for _, w := range words(n - 1) {
			for _, c := range alphabet {
				out = append(out, w+string(c))
			}
		}
		return out
	}
	var values, inputs []string
	for n := 1; n <= 2; n++ {
		values = append(values, words(n)...)
	}
	for n := 0; n <= 5; n++ {
		inputs = append(inputs, words(n)...)
	}
	for _, v := range values {
		r := NewRedactor(v, nil)
		for _, in := range inputs {
			out, _ := r.Text(in)
			if strings.Contains(out, v) {
				t.Fatalf("secret %q: Text(%q) = %q holds it", v, in, out)
			}
		}
	}
}

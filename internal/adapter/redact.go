package adapter

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

// Redactor removes one client's credentials from error text. It is the
// single owner of that rule for decide and embed: NewRedactor builds it
// from the Secret and the parsed BaseURL, [NormalizeSDKError] applies it
// through [VendorError.Redact] to a response body, and [Redactor.Scrub]
// applies it to every error a request attempt returns.
//
// Two kinds of text are removed, each replaced by the mask [maskFor]
// picks:
//
//   - Values, at any length: the Secret, the BaseURL password, and the
//     base64 "user:password" pair net/http sends as a Basic credential
//     for a BaseURL with userinfo, each with its JSON-escaped and Go-quoted
//     (strconv.Quote, the %q verb) forms. A value that is not valid UTF-8
//     also has those forms of its U+FFFD-replaced text, the text
//     json.Unmarshal gives for it. A short value masks every matching text
//     in the message, not only the credential.
//   - URL userinfo, where it appears as such: in every
//     "scheme://user[:password]@" occurrence the userinfo is replaced.
//     The username is not a value: the same characters elsewhere in the
//     text stay.
//
// A nil *Redactor redacts nothing. Empty values are dropped, so an empty
// Secret redacts nothing.
type Redactor struct {
	values []string // longest first, no empty entry, no duplicate
	mask   string
}

// NewRedactor returns the Redactor for a client configured with secret
// and the root URL ParseBaseURL returned. root may be nil.
func NewRedactor(secret string, root *url.URL) *Redactor {
	raw := []string{secret}
	if root != nil && root.User != nil {
		if pw, ok := root.User.Password(); ok && pw != "" {
			raw = append(raw, pw)
			raw = append(raw, base64.StdEncoding.EncodeToString([]byte(root.User.Username()+":"+pw)))
		}
	}
	seen := make(map[string]bool)
	var values []string
	add := func(v string) {
		if v != "" && !seen[v] {
			seen[v] = true
			values = append(values, v)
		}
	}
	for _, v := range raw {
		forms := []string{v}
		if !utf8.ValidString(v) {
			// json.Unmarshal replaces each invalid byte of an echoed
			// value with U+FFFD, and so does this conversion.
			forms = append(forms, string([]rune(v)))
		}
		for _, f := range forms {
			add(f)
			for _, e := range jsonEscaped(f) {
				add(e)
			}
			add(goQuoted(f))
		}
	}
	sort.SliceStable(values, func(i, j int) bool { return len(values[i]) > len(values[j]) })
	return &Redactor{values: values, mask: maskFor(values)}
}

// jsonEscaped returns the forms v takes inside a JSON string, with and
// without HTML escaping, when they differ from v: a server that echoes
// the credential in a JSON body writes one of them.
func jsonEscaped(v string) []string {
	var out []string
	if b, err := json.Marshal(v); err == nil {
		out = append(out, string(b[1:len(b)-1]))
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err == nil {
		s := strings.TrimSuffix(buf.String(), "\n")
		out = append(out, s[1:len(s)-1])
	}
	return out
}

// goQuoted returns the form v takes inside a %q or strconv.Quote string:
// decide quotes server values that way, and so do net/http's url.Error
// and malformed-response texts. strconv.Quote escapes each rune on its
// own, so the quoted text of anything holding valid-UTF-8 v holds this.
func goQuoted(v string) string {
	q := strconv.Quote(v)
	return q[1 : len(q)-1]
}

// maskFor returns "***", unless a value holds '*': then three copies of
// the first printable ASCII byte no value holds. A mask made of bytes no
// value holds cannot be, or join with its neighbours into, a value, so
// redacting text never produces text that holds a value. When every
// printable byte occurs in some value the mask is empty; removal only
// shortens the text, so the fixpoint loop in [Redactor.Text] still ends.
func maskFor(values []string) string {
	held := func(c byte) bool {
		for _, v := range values {
			if strings.IndexByte(v, c) >= 0 {
				return true
			}
		}
		return false
	}
	if !held('*') {
		return "***"
	}
	for c := byte('!'); c <= '~'; c++ {
		if !held(c) {
			return strings.Repeat(string(c), 3)
		}
	}
	return ""
}

// Text returns s with every value and every URL userinfo replaced, and
// whether that changed s. It repeats until no value remains, so a
// redaction that joins its neighbours into a value is redacted too.
func (r *Redactor) Text(s string) (string, bool) {
	if r == nil {
		return s, false
	}
	out := s
	for {
		next := out
		for _, v := range r.values {
			if strings.Contains(next, v) {
				next = strings.ReplaceAll(next, v, r.mask)
			}
		}
		next = r.userinfo(next)
		if next == out {
			return out, out != s
		}
		out = next
	}
}

// userinfo replaces the userinfo of every "scheme://user[:password]@"
// occurrence in s. The authority ends at the first '/', '?', '#',
// whitespace, quote, backslash or angle bracket; its userinfo runs to the
// last '@' before that end, as url.Parse reads it.
func (r *Redactor) userinfo(s string) string {
	var b strings.Builder
	i := 0
	for {
		j := strings.Index(s[i:], "://")
		if j < 0 {
			break
		}
		j += i
		start := j
		for start > 0 && isSchemeByte(s[start-1]) {
			start--
		}
		for start < j && !isASCIILetter(s[start]) {
			start++
		}
		auth := j + len("://")
		end := auth
		for end < len(s) && !isAuthorityEnd(s[end]) {
			end++
		}
		at := strings.LastIndexByte(s[auth:end], '@')
		if start == j || at < 0 {
			b.WriteString(s[i:auth])
			i = auth
			continue
		}
		b.WriteString(s[i:auth])
		b.WriteString(r.mask)
		b.WriteByte('@')
		i = auth + at + 1
	}
	if i == 0 {
		return s
	}
	b.WriteString(s[i:])
	return b.String()
}

func isASCIILetter(c byte) bool { return c|0x20 >= 'a' && c|0x20 <= 'z' }

func isSchemeByte(c byte) bool {
	return isASCIILetter(c) || c >= '0' && c <= '9' || c == '+' || c == '-' || c == '.'
}

func isAuthorityEnd(c byte) bool {
	return c <= ' ' || c == 0x7f || strings.IndexByte("/?#\"\\<>", c) >= 0
}

// Scrub returns err when nothing reachable from it — its own text or the
// text of any error under Unwrap() error or Unwrap() []error — changes
// under [Redactor.Text]; the error is then untouched, and errors.As still
// finds a net.Error in it.
//
// Otherwise it returns a rebuilt error whose chain is cut. What stays:
//   - for an *llmkit.APIError: Kind, StatusCode, RetryAfter,
//     HasRetryAfter and Provider, with Message redacted and capped, and
//     context.Canceled / context.DeadlineExceeded when errors.Is matched
//     them;
//   - for any other error: the redacted text, and errors.Is for the
//     context errors and the llmkit sentinels that matched.
func (r *Redactor) Scrub(err error) error {
	if r == nil || err == nil || !r.reaches(err) {
		return err
	}
	var kept []error
	for _, s := range contextErrors {
		if errors.Is(err, s) {
			kept = append(kept, s)
		}
	}
	var api *llmkit.APIError
	if errors.As(err, &api) {
		// The Message of a transport APIError is the cap of its Err's
		// text. Redacting that cap would leave the front of a credential
		// the cap cut through; redact the text it was cut from instead.
		src := api.Message
		if api.Err != nil {
			if full := api.Err.Error(); capMessage(full) == api.Message {
				src = full
			}
		}
		msg, _ := r.Text(src)
		return &llmkit.APIError{
			Kind:          api.Kind,
			StatusCode:    api.StatusCode,
			RetryAfter:    api.RetryAfter,
			HasRetryAfter: api.HasRetryAfter,
			Provider:      api.Provider,
			Message:       capMessage(msg),
			Err:           errors.Join(kept...),
		}
	}
	for _, s := range sentinels {
		if errors.Is(err, s) {
			kept = append(kept, s)
		}
	}
	text, _ := r.Text(err.Error())
	return &redactedError{text: text, is: kept}
}

// reaches reports whether the text of err, or of any error under it,
// changes under [Redactor.Text].
func (r *Redactor) reaches(err error) bool {
	if err == nil {
		return false
	}
	if _, changed := r.Text(err.Error()); changed {
		return true
	}
	switch u := err.(type) {
	case interface{ Unwrap() error }:
		return r.reaches(u.Unwrap())
	case interface{ Unwrap() []error }:
		for _, e := range u.Unwrap() {
			if r.reaches(e) {
				return true
			}
		}
	}
	return false
}

var contextErrors = []error{context.Canceled, context.DeadlineExceeded}

var sentinels = []error{
	llmkit.ErrRateLimited,
	llmkit.ErrAuth,
	llmkit.ErrContextTooLong,
	llmkit.ErrInvalidRequest,
	llmkit.ErrServer,
	llmkit.ErrOverloaded,
}

// redactedError is a non-APIError error rebuilt by [Redactor.Scrub]: the
// redacted text, and the sentinels errors.Is still finds.
type redactedError struct {
	text string
	is   []error
}

func (e *redactedError) Error() string   { return e.text }
func (e *redactedError) Unwrap() []error { return e.is }

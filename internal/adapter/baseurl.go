package adapter

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/dpoage/llmkit"
)

// ParseBaseURL validates raw as the API root of an HTTP service and
// returns it parsed. field names the config field in every refusal (for
// example "decide: BaseURL"); it is a caller constant, never user input.
//
// The rules:
//   - raw holds no '#' anywhere, an empty fragment included. A '#' ends
//     the URL on the wire, so it would silently drop the endpoint path
//     that [JoinEndpoint] appends.
//   - raw parses as a URL, in the absolute form (an opaque form such as
//     "http:host" is refused).
//   - the scheme is http or https, in any letter case.
//   - the host name is non-empty: "http://:80" is refused.
//
// A query is kept: [JoinEndpoint] places the endpoint path before it.
// A URL that parses but cannot be dialed ("http://localhost:99999") is
// accepted; the failure surfaces at request time as a retryable transport
// error.
//
// Every refusal wraps [llmkit.ErrInvalidRequest]. The error names the
// field and the rule only: it never contains raw, the userinfo, or any
// text derived from url.Parse, whose errors quote the input.
func ParseBaseURL(field, raw string) (*url.URL, error) {
	refuse := func(rule string) error {
		return fmt.Errorf("%s: %s: %w", field, rule, llmkit.ErrInvalidRequest)
	}
	if strings.Contains(raw, "#") {
		return nil, refuse("must not contain '#' (no fragment)")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, refuse("is not a valid URL")
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, refuse("scheme must be http or https")
	}
	if u.Opaque != "" || u.Hostname() == "" {
		return nil, refuse("must be an absolute URL with a non-empty host")
	}
	return u, nil
}

// JoinEndpoint returns a copy of base whose path is base's path as the
// caller wrote it, with its trailing literal slashes dropped, followed by
// endpoint: the path url.Parse gives for strings.TrimRight(raw, "/") +
// endpoint when raw has no query. The query is kept as parsed. It does no
// path cleaning, which url.JoinPath and an append to Path alone would do.
// endpoint must start with '/' and hold no character that needs escaping.
func JoinEndpoint(base *url.URL, endpoint string) *url.URL {
	u := *base
	// url.Parse keeps the path as written in RawPath whenever it differs
	// from the default encoding of Path; otherwise the default encoding is
	// the written form. EscapedPath() is not the written form: when RawPath
	// is not a valid encoding (it holds a space, a non-ASCII byte, or one of
	// | ^ " { } \ ` < >), it re-encodes Path, where a written "%2F" is
	// already a literal '/'.
	written := base.RawPath
	if written == "" {
		written = base.EscapedPath()
	}
	trimmed := strings.TrimRight(written, "/")
	// Each trailing literal '/' as written is one trailing '/' byte of
	// Path, so dropping the same count keeps RawPath the written form of
	// Path, as url.Parse would have set it.
	u.Path = base.Path[:len(base.Path)-(len(written)-len(trimmed))] + endpoint
	if base.RawPath != "" {
		u.RawPath = trimmed + endpoint
	}
	return &u
}

// NewRequest builds a request for u from the URL value itself. It never
// formats u to a string and parses it again: a re-parse of an
// already-validated URL can fail and quote the whole URL, userinfo
// included, in its error. The request holds its own copy of u.
//
// The returned error comes from net/http on the method and context only,
// which are never derived from configuration.
func NewRequest(ctx context.Context, method string, u *url.URL, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, "/", body)
	if err != nil {
		return nil, err
	}
	cu := *u
	// http.NewRequest drops an empty port from the parsed host; keep that.
	cu.Host = strings.TrimSuffix(cu.Host, ":")
	req.URL = &cu
	req.Host = cu.Host
	return req, nil
}

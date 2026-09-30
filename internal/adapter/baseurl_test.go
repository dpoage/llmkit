package adapter

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// errorTexts returns Error() of err and of every node reachable through
// Unwrap() error and Unwrap() []error.
func errorTexts(err error) []string {
	if err == nil {
		return nil
	}
	out := []string{err.Error()}
	switch u := err.(type) {
	case interface{ Unwrap() []error }:
		for _, e := range u.Unwrap() {
			out = append(out, errorTexts(e)...)
		}
	case interface{ Unwrap() error }:
		out = append(out, errorTexts(u.Unwrap())...)
	}
	return out
}

// TestParseBaseURL_Refusals pins the refusal grid: every row is an
// ErrInvalidRequest whose error tree carries neither secret, nor the raw
// value, nor any url.Parse text.
func TestParseBaseURL_Refusals(t *testing.T) {
	rows := []string{
		"http://a b",
		"api.typesafe.invalid",
		"ftp://127.0.0.1:1",
		"ftp://user:hunter2@127.0.0.1:1",
		"http://",
		"http://user:hunter2@",
		"http://[::1",
		"http://user:hunter2@a b",
		"http:user:hunter2@127.0.0.1",
		"A:hunter2@0",
		"http://user:hunter2?x@127.0.0.1",
		"http://user:hunter2/x@127.0.0.1",
		"http://user:hun#ter2@127.0.0.1",
		"http://sk-USER@127.0.0.1:1/v1#",
		"http://127.0.0.1/v1#frag",
		"http://:",
		"http://:80",
		"http://user:hunter2@:",
		"http://user:hunter2@:80",
		"http://sk-USER@:1",
		"",
	}
	for _, raw := range rows {
		t.Run(raw, func(t *testing.T) {
			u, err := ParseBaseURL("test: BaseURL", raw)
			if err == nil {
				t.Fatalf("ParseBaseURL(%q) = %v, want an error", raw, u)
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("err = %v, want ErrInvalidRequest", err)
			}
			for _, text := range errorTexts(err) {
				for _, leak := range []string{"hunter2", "sk-USER", "invalid character", "parse \""} {
					if strings.Contains(text, leak) {
						t.Errorf("error text %q contains %q", text, leak)
					}
				}
				if raw != "" && strings.Contains(text, raw) {
					t.Errorf("error text %q quotes the raw BaseURL", text)
				}
			}
			if !strings.HasPrefix(err.Error(), "test: BaseURL: ") {
				t.Errorf("err = %q, want it to name the field", err)
			}
		})
	}
}

func TestParseBaseURL_Accepts(t *testing.T) {
	rows := []struct {
		raw, scheme, host string
	}{
		{"HTTP://127.0.0.1:1234/v1", "http", "127.0.0.1:1234"},
		{"http://LOCALHOST:8080/v1", "http", "LOCALHOST:8080"},
		{"https://api.example.test/v1", "https", "api.example.test"},
		// Parseable but undialable: refused at request time, not here.
		{"http://localhost:99999", "http", "localhost:99999"},
		{"https://x.test/v1?api-version=2024", "https", "x.test"},
		{"http://user:pw@127.0.0.1:1", "http", "127.0.0.1:1"},
		{"http://[::1]:80", "http", "[::1]:80"},
	}
	for _, r := range rows {
		t.Run(r.raw, func(t *testing.T) {
			u, err := ParseBaseURL("test: BaseURL", r.raw)
			if err != nil {
				t.Fatalf("ParseBaseURL(%q): %v", r.raw, err)
			}
			if u.Scheme != r.scheme || u.Host != r.host {
				t.Errorf("got scheme %q host %q, want %q %q", u.Scheme, u.Host, r.scheme, r.host)
			}
		})
	}
}

// TestJoinEndpoint pins the join: the endpoint goes on the escaped path,
// before the query, with no path cleaning.
func TestJoinEndpoint(t *testing.T) {
	rows := []struct {
		name, base      string
		wantPath        string // decoded
		wantEscaped     string
		wantQuery       string
		wantForcedQuery bool
	}{
		{"root", "http://h.test", "/x", "/x", "", false},
		{"trailing slash", "http://h.test/v1/", "/v1/x", "/v1/x", "", false},
		{"trailing slashes", "http://h.test/v1//", "/v1/x", "/v1/x", "", false},
		{"only slashes", "http://h.test//", "/x", "/x", "", false},
		{"encoded slash", "http://h.test/a%2Fb/v1", "/a/b/v1/x", "/a%2Fb/v1/x", "", false},
		{"double slash", "http://h.test//v1", "//v1/x", "//v1/x", "", false},
		{"dot segments", "http://h.test/v1/./x/..", "/v1/./x/../x", "/v1/./x/../x", "", false},
		{"query", "http://h.test/v1?api-version=2024", "/v1/x", "/v1/x", "api-version=2024", false},
		{"query at root", "http://h.test?a=b&c=d%2Fe", "/x", "/x", "a=b&c=d%2Fe", false},
		{"slash before query", "http://h.test/v1/?a=b", "/v1/x", "/v1/x", "a=b", false},
		{"empty query", "http://h.test/v1?", "/v1/x", "/v1/x", "", true},
		{"encoded tilde", "http://h.test/%7Efoo", "/~foo/x", "/%7Efoo/x", "", false},
		{"trailing encoded slash", "http://h.test/v1%2F", "/v1//x", "/v1%2F/x", "", false},
		{"encoded slash then slash", "http://h.test/v1%2F/", "/v1//x", "/v1%2F/x", "", false},
		{"encoded slashes", "http://h.test/a%2Fb%2F", "/a/b//x", "/a%2Fb%2F/x", "", false},
		{"encoded slash before query", "http://h.test/openai%2Fdeploy%2F?api-version=1", "/openai/deploy//x", "/openai%2Fdeploy%2F/x", "api-version=1", false},
		{"only encoded slash", "http://h.test/%2F", "///x", "/%2F/x", "", false},
		{"lowercase encoded slash", "http://h.test/v1%2f", "/v1//x", "/v1%2f/x", "", false},
		// A raw space, non-ASCII byte, or one of | ^ " { } \ ` < > makes
		// RawPath an invalid encoding, so EscapedPath() re-encodes Path,
		// where the %2F is already a '/'. Only the literal slashes as
		// written are trimmed.
		{"space then encoded slash", "http://h.test/ %2F", "/ //x", "/%20//x", "", false},
		{"non-ASCII then encoded slash", "http://h.test/café%2F", "/café//x", "/caf%C3%A9//x", "", false},
		{"pipe then encoded slash", "http://h.test/a|b%2F", "/a|b//x", "/a%7Cb//x", "", false},
		{"space, encoded slash, slash", "http://h.test/v1 %2F/", "/v1 //x", "/v1%20//x", "", false},
		{"non-ASCII, encoded slash, query", "http://h.test/é%2F?api-version=1", "/é//x", "/%C3%A9//x", "api-version=1", false},
		{"space then inner encoded slash", "http://h.test/a b%2Fc", "/a b/c/x", "/a%20b/c/x", "", false},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			base, err := url.Parse(r.base)
			if err != nil {
				t.Fatal(err)
			}
			before := *base
			got := JoinEndpoint(base, "/x")
			if got.Path != r.wantPath {
				t.Errorf("Path = %q, want %q", got.Path, r.wantPath)
			}
			if got.EscapedPath() != r.wantEscaped {
				t.Errorf("EscapedPath = %q, want %q", got.EscapedPath(), r.wantEscaped)
			}
			if got.RawQuery != r.wantQuery || got.ForceQuery != r.wantForcedQuery {
				t.Errorf("RawQuery = %q ForceQuery = %v, want %q %v", got.RawQuery, got.ForceQuery, r.wantQuery, r.wantForcedQuery)
			}
			if *base != before {
				t.Errorf("JoinEndpoint mutated its base: %+v -> %+v", before, *base)
			}
			if got == base {
				t.Error("JoinEndpoint returned its base pointer")
			}
		})
	}
}

// TestJoinEndpoint_MatchesStringJoin pins the join to base 1fe7518's
// string join over every path of 1 to 4 tokens: for BaseURL raw with no
// query, the joined URL sends the request target url.Parse gives for
// strings.TrimRight(raw, "/") + endpoint. The tokens mix literal and
// encoded slashes with bytes that make RawPath an invalid encoding.
func TestJoinEndpoint_MatchesStringJoin(t *testing.T) {
	toks := []string{"/", "v1", "%2F", "%2f", "a%2Fb", " ", "é", "|", "%20", ".", "%25"}
	var mismatches int
	var walk func(path string, depth int)
	walk = func(path string, depth int) {
		if depth > 0 {
			raw := "http://h.test/" + path
			base, err := ParseBaseURL("test: BaseURL", raw)
			if err != nil {
				t.Fatalf("ParseBaseURL(%q): %v", raw, err)
			}
			want, err := url.Parse(strings.TrimRight(raw, "/") + "/x")
			if err != nil {
				t.Fatalf("url.Parse(%q): %v", raw, err)
			}
			if got := JoinEndpoint(base, "/x"); got.RequestURI() != want.RequestURI() && mismatches < 20 {
				mismatches++
				t.Errorf("%q: request target %q, want %q", raw, got.RequestURI(), want.RequestURI())
			}
		}
		if depth == 4 {
			return
		}
		for _, tok := range toks {
			walk(path+tok, depth+1)
		}
	}
	walk("", 0)
}

// TestNewRequest_FromURLValue pins that a URL that passes ParseBaseURL
// always yields a request, with its userinfo intact and without any error
// text that quotes it. `http://user:pw@[::%25\x80]` parses, but
// http.NewRequest on the parsed URL's string form fails and echoes the
// password.
func TestNewRequest_FromURLValue(t *testing.T) {
	const raw = "http://user:pw@[::%25\x80]/v1"
	base, err := ParseBaseURL("test: BaseURL", raw)
	if err != nil {
		t.Fatalf("ParseBaseURL: %v", err)
	}
	joined := JoinEndpoint(base, "/embeddings")
	_, reparseErr := http.NewRequest(http.MethodPost, joined.String(), nil)
	if reparseErr == nil || !strings.Contains(reparseErr.Error(), "pw") {
		t.Fatalf("premise: re-parsing %q gave %v; the row no longer distinguishes a re-parse", joined.String(), reparseErr)
	}
	req, err := NewRequest(context.Background(), http.MethodPost, joined, strings.NewReader("{}"))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	if pw, _ := req.URL.User.Password(); pw != "pw" || req.URL.User.Username() != "user" {
		t.Errorf("userinfo = %v, want user:pw", req.URL.User)
	}
	if req.URL.Path != "/v1/embeddings" || req.Method != http.MethodPost || req.ContentLength != 2 {
		t.Errorf("request = %s %s (len %d), want POST /v1/embeddings (len 2)", req.Method, req.URL.Path, req.ContentLength)
	}
}

// TestNewRequest_HostAndCopy pins the two things http.NewRequest did for
// a parsed URL that NewRequest keeps: Host mirrors the URL's host with an
// empty port dropped, and the request owns its URL.
func TestNewRequest_HostAndCopy(t *testing.T) {
	base, err := ParseBaseURL("test: BaseURL", "http://127.0.0.1:/v1")
	if err != nil {
		t.Fatal(err)
	}
	req, err := NewRequest(context.Background(), http.MethodPost, base, nil)
	if err != nil {
		t.Fatal(err)
	}
	if req.Host != "127.0.0.1" || req.URL.Host != "127.0.0.1" {
		t.Errorf("Host = %q, URL.Host = %q, want both 127.0.0.1", req.Host, req.URL.Host)
	}
	req.URL.Path = "/changed"
	if base.Path != "/v1" || base.Host != "127.0.0.1:" {
		t.Errorf("request mutation reached the shared URL: %+v", *base)
	}
}

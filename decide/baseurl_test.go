package decide

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
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

// refusedBaseURLs is the grid of bad BaseURLs: every row must be refused at
// New with ErrInvalidRequest and no secret in the error tree.
var refusedBaseURLs = []string{
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
}

// TestNew_RefusesBadBaseURL: each grid row is refused at construction,
// with ErrInvalidRequest, with no wire attempt (rows that name the live
// server's host are checked against its hit counter), and with no secret
// text anywhere in the error tree.
func TestNew_RefusesBadBaseURL(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	host := strings.TrimPrefix(srv.URL, "http://")

	var rows []string
	for _, raw := range refusedBaseURLs {
		rows = append(rows, raw)
		if strings.Contains(raw, "127.0.0.1") {
			rows = append(rows, strings.NewReplacer("127.0.0.1:1", host, "127.0.0.1", host).Replace(raw))
		}
	}
	for _, raw := range rows {
		t.Run(raw, func(t *testing.T) {
			hits.Store(0)
			c, err := New(Config{Secret: testSecret, Model: "jev-latest", BaseURL: raw, Retry: fastRetry})
			if err == nil {
				_, askErr := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
				t.Fatalf("New(BaseURL %q) succeeded; Ask then made %d wire attempts (err %v)", raw, hits.Load(), askErr)
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("err = %v, want ErrInvalidRequest", err)
			}
			for _, text := range errorTexts(err) {
				for _, leak := range []string{"hunter2", "sk-USER", testSecret} {
					if strings.Contains(text, leak) {
						t.Errorf("error text %q contains %q", text, leak)
					}
				}
			}
			if n := hits.Load(); n != 0 {
				t.Errorf("wire attempts = %d, want 0", n)
			}
		})
	}
}

// TestNew_AcceptsBaseURLForms pins the accept rows: a scheme in any letter
// case, a host in any letter case, and a parseable but undialable port.
func TestNew_AcceptsBaseURLForms(t *testing.T) {
	var got sync.Map
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.Store("uri", r.RequestURI)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	hostport := strings.TrimPrefix(srv.URL, "http://")
	once := retry.Config{MaxAttempts: 1}

	for _, raw := range []string{"HTTP://" + hostport + "/v1", "http://" + strings.Replace(hostport, "127.0.0.1", "LOCALHOST", 1) + "/v1"} {
		got.Delete("uri")
		c, err := New(Config{Secret: testSecret, Model: "jev-latest", BaseURL: raw, Retry: once})
		if err != nil {
			t.Fatalf("New(%q): %v", raw, err)
		}
		_, _ = c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
		if uri, _ := got.Load("uri"); uri != "/v1/v1/systemone" {
			t.Errorf("BaseURL %q: server saw %v, want /v1/v1/systemone", raw, uri)
		}
	}
	if _, err := New(Config{Secret: testSecret, Model: "jev-latest", BaseURL: "http://localhost:99999", Retry: once}); err != nil {
		t.Errorf("New(http://localhost:99999): %v; a parseable but undialable URL stays accepted", err)
	}
}

// wireURI runs one Ask against a recording server and
// returns the raw request target and Authorization header the server saw.
func wireURI(t *testing.T, base func(host string) string) (uri, auth string) {
	t.Helper()
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		uri, auth = r.RequestURI, r.Header.Get("Authorization")
		mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	c, err := New(Config{Secret: "k", Model: "jev-latest", BaseURL: base(strings.TrimPrefix(srv.URL, "http://")), Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	mu.Lock()
	defer mu.Unlock()
	return uri, auth
}

// TestAsk_BaseURLJoin: the endpoint path is joined before a query, and the
// rows base 1fe7518 sent unchanged still reach the wire with the same
// request target.
func TestAsk_BaseURLJoin(t *testing.T) {
	rows := []struct{ name, suffix, want string }{
		{"query", "/v1?api-version=2024", "/v1/v1/systemone?api-version=2024"},
		{"query at root", "?api-version=2024", "/v1/systemone?api-version=2024"},
		{"encoded slash", "/a%2Fb/v1", "/a%2Fb/v1/v1/systemone"},
		{"double slash", "//v1", "//v1/v1/systemone"},
		{"dot segments", "/v1/./x/..", "/v1/./x/../v1/systemone"},
		{"trailing slash", "/v1/", "/v1/v1/systemone"},
		{"space in path", "/a b/v1", "/a%20b/v1/v1/systemone"},
		{"trailing encoded slash", "/v1%2F", "/v1%2F/v1/systemone"},
		{"encoded slash then slash", "/v1%2F/", "/v1%2F/v1/systemone"},
		{"encoded slashes", "/a%2Fb%2F", "/a%2Fb%2F/v1/systemone"},
		{"encoded slash before query", "/openai%2Fdeploy%2F?api-version=1", "/openai%2Fdeploy%2F/v1/systemone?api-version=1"},
		{"only encoded slash", "/%2F", "/%2F/v1/systemone"},
		{"lowercase encoded slash", "/v1%2f", "/v1%2f/v1/systemone"},
		// A raw space, non-ASCII byte or '|' makes net/url re-encode the
		// path, which decodes the %2F; base still kept its slash count.
		{"space then encoded slash", "/ %2F", "/%20//v1/systemone"},
		{"non-ASCII then encoded slash", "/café%2F", "/caf%C3%A9//v1/systemone"},
		{"non-ASCII segment then encoded slash", "/deploy/é%2F", "/deploy/%C3%A9//v1/systemone"},
		{"pipe then encoded slash", "/a|b%2F", "/a%7Cb//v1/systemone"},
		{"non-ASCII, encoded slash, query", "/é%2F?api-version=1", "/%C3%A9//v1/systemone?api-version=1"},
		{"space, encoded slash, slash", "/v1 %2F/", "/v1%20//v1/systemone"},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			uri, _ := wireURI(t, func(host string) string { return "http://" + host + r.suffix })
			if uri != r.want {
				t.Errorf("request target = %q, want %q", uri, r.want)
			}
		})
	}
	t.Run("userinfo", func(t *testing.T) {
		uri, auth := wireURI(t, func(host string) string { return "http://user:pw@" + host + "/v1" })
		if uri != "/v1/v1/systemone" || auth != "Bearer k" {
			t.Errorf("request target %q, Authorization %q, want /v1/v1/systemone and the Bearer key", uri, auth)
		}
	})
}

// TestAsk_UnbuildableJoinedURLDoesNotEchoPassword pins the R2 row: a
// BaseURL that passes validation but whose string form no longer
// parses must still yield a request, and no error may carry the password.
func TestAsk_UnbuildableJoinedURLDoesNotEchoPassword(t *testing.T) {
	c, err := New(Config{Secret: "k", Model: "jev-latest", BaseURL: "http://user:pw@[::%25\x80]", Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, askErr := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	if askErr == nil {
		t.Fatal("Ask succeeded against an undialable host")
	}
	if strings.Contains(askErr.Error(), "build request") {
		t.Errorf("err = %v; the request must build from the parsed URL, not fail", askErr)
	}
	for _, text := range errorTexts(askErr) {
		if strings.Contains(text, ":pw@") {
			t.Errorf("error text %q echoes the URL password", text)
		}
	}
}

// TestNew_JitterRange: NaN joins the refused values.
func TestNew_JitterRange(t *testing.T) {
	for _, j := range []float64{math.NaN(), -0.1, 1.01, math.Inf(1), math.Inf(-1)} {
		_, err := New(Config{Secret: "k", Model: "m", Retry: retry.Config{Jitter: j}})
		if !errors.Is(err, llmkit.ErrInvalidRequest) {
			t.Errorf("Jitter %v: err = %v, want ErrInvalidRequest", j, err)
		}
	}
	for _, j := range []float64{0, 0.5, 1} {
		if _, err := New(Config{Secret: "k", Model: "m", Retry: retry.Config{Jitter: j}}); err != nil {
			t.Errorf("Jitter %v: New: %v", j, err)
		}
	}
}

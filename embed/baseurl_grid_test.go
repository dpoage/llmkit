package embed

import (
	"context"
	"errors"
	"fmt"
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

// refusedBaseURLs is the grid of bad BaseURLs: every row must be refused by
// Validate and New with ErrInvalidRequest and no secret in the error tree.
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

var bothBackends = []Backend{BackendOllama, BackendOpenAICompatible}

// okEmbeddingBody is a 200 body both wire shapes decode.
const okEmbeddingBody = `{"data":[{"embedding":[1],"index":0}],"embeddings":[[1]]}`

// TestConfig_RefusesBadBaseURL: Validate and New both refuse every grid
// row, wrapping ErrInvalidRequest, without a wire attempt (rows naming the
// live server's host are checked against its hit counter) and without
// secret text anywhere in the error tree.
func TestConfig_RefusesBadBaseURL(t *testing.T) {
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
	for _, be := range bothBackends {
		for _, raw := range rows {
			t.Run(fmt.Sprintf("%s/%s", be, raw), func(t *testing.T) {
				hits.Store(0)
				cfg := Config{Backend: be, Model: "m", BaseURL: raw, Secret: "sk-embed-secret", Retry: retry.Config{MaxAttempts: 3, BaseDelay: 1}}

				vErr := cfg.Validate()
				emb, nErr := New(cfg)
				if vErr == nil || nErr == nil {
					var embedErr error
					if emb != nil {
						_, embedErr = emb.Embed(context.Background(), "x")
					}
					t.Fatalf("BaseURL %q: Validate = %v, New = %v; Embed then made %d wire attempts (err %v)", raw, vErr, nErr, hits.Load(), embedErr)
				}
				for what, err := range map[string]error{"Validate": vErr, "New": nErr} {
					if !errors.Is(err, llmkit.ErrInvalidRequest) {
						t.Errorf("%s: err = %v, want ErrInvalidRequest", what, err)
					}
					for _, text := range errorTexts(err) {
						for _, leak := range []string{"hunter2", "sk-USER", "sk-embed-secret"} {
							if strings.Contains(text, leak) {
								t.Errorf("%s: error text %q contains %q", what, text, leak)
							}
						}
					}
				}
				if n := hits.Load(); n != 0 {
					t.Errorf("wire attempts = %d, want 0", n)
				}
			})
		}
	}
}

// TestConfig_AcceptsBaseURLForms pins the accept rows: a scheme and a host
// in any letter case, and a parseable but undialable port.
func TestConfig_AcceptsBaseURLForms(t *testing.T) {
	for _, be := range bothBackends {
		var mu sync.Mutex
		var uri string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			uri = r.RequestURI
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, okEmbeddingBody)
		}))
		hostport := strings.TrimPrefix(srv.URL, "http://")
		want := "/v1/embeddings"
		if be == BackendOllama {
			want = "/v1/api/embed"
		}
		for _, raw := range []string{"HTTP://" + hostport + "/v1", "http://" + strings.Replace(hostport, "127.0.0.1", "LOCALHOST", 1) + "/v1"} {
			cfg := Config{Backend: be, Model: "m", BaseURL: raw}
			if err := cfg.Validate(); err != nil {
				t.Fatalf("%s: Validate(%q): %v", be, raw, err)
			}
			emb, err := New(cfg)
			if err != nil {
				t.Fatalf("%s: New(%q): %v", be, raw, err)
			}
			if _, err := emb.Embed(context.Background(), "x"); err != nil {
				t.Fatalf("%s: Embed via %q: %v", be, raw, err)
			}
			mu.Lock()
			if uri != want {
				t.Errorf("%s: BaseURL %q: server saw %q, want %q", be, raw, uri, want)
			}
			mu.Unlock()
		}
		srv.Close()

		cfg := Config{Backend: be, Model: "m", BaseURL: "http://localhost:99999"}
		if err := cfg.Validate(); err != nil {
			t.Errorf("%s: Validate(http://localhost:99999): %v; a parseable but undialable URL stays accepted", be, err)
		}
		if _, err := New(cfg); err != nil {
			t.Errorf("%s: New(http://localhost:99999): %v", be, err)
		}
	}
}

// wireRequest runs one Embed against a recording server whose BaseURL is
// base(host) and returns the request the server saw.
func wireRequest(t *testing.T, be Backend, secret string, base func(host string) string) (uri, basicUser, basicPass, auth string) {
	t.Helper()
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		uri, auth = r.RequestURI, r.Header.Get("Authorization")
		basicUser, basicPass, _ = r.BasicAuth()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okEmbeddingBody)
	}))
	defer srv.Close()
	emb, err := New(Config{Backend: be, Model: "m", BaseURL: base(strings.TrimPrefix(srv.URL, "http://")), Secret: secret})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := emb.Embed(context.Background(), "x"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	return uri, basicUser, basicPass, auth
}

// TestEmbed_BaseURLJoin: the backend path is joined before a query, and the
// rows base 1fe7518 sent unchanged still reach the wire with the same
// request target.
func TestEmbed_BaseURLJoin(t *testing.T) {
	rows := []struct{ name, suffix, wantOpenAI, wantOllama string }{
		{"query", "/v1?api-version=2024", "/v1/embeddings?api-version=2024", "/v1/api/embed?api-version=2024"},
		{"query at root", "?api-version=2024", "/embeddings?api-version=2024", "/api/embed?api-version=2024"},
		{"encoded slash", "/a%2Fb/v1", "/a%2Fb/v1/embeddings", "/a%2Fb/v1/api/embed"},
		{"double slash", "//v1", "//v1/embeddings", "//v1/api/embed"},
		{"dot segments", "/v1/./x/..", "/v1/./x/../embeddings", "/v1/./x/../api/embed"},
		{"trailing slash", "/v1/", "/v1/embeddings", "/v1/api/embed"},
		{"space in path", "/a b/v1", "/a%20b/v1/embeddings", "/a%20b/v1/api/embed"},
		{"trailing encoded slash", "/v1%2F", "/v1%2F/embeddings", "/v1%2F/api/embed"},
		{"encoded slash then slash", "/v1%2F/", "/v1%2F/embeddings", "/v1%2F/api/embed"},
		{"encoded slashes", "/a%2Fb%2F", "/a%2Fb%2F/embeddings", "/a%2Fb%2F/api/embed"},
		{"encoded slash before query", "/openai%2Fdeploy%2F?api-version=1", "/openai%2Fdeploy%2F/embeddings?api-version=1", "/openai%2Fdeploy%2F/api/embed?api-version=1"},
		{"only encoded slash", "/%2F", "/%2F/embeddings", "/%2F/api/embed"},
		{"lowercase encoded slash", "/v1%2f", "/v1%2f/embeddings", "/v1%2f/api/embed"},
		// A raw space, non-ASCII byte or '|' makes net/url re-encode the
		// path, which decodes the %2F; base still kept its slash count.
		{"space then encoded slash", "/ %2F", "/%20//embeddings", "/%20//api/embed"},
		{"non-ASCII then encoded slash", "/café%2F", "/caf%C3%A9//embeddings", "/caf%C3%A9//api/embed"},
		{"non-ASCII segment then encoded slash", "/deploy/é%2F", "/deploy/%C3%A9//embeddings", "/deploy/%C3%A9//api/embed"},
		{"pipe then encoded slash", "/a|b%2F", "/a%7Cb//embeddings", "/a%7Cb//api/embed"},
		{"non-ASCII, encoded slash, query", "/é%2F?api-version=1", "/%C3%A9//embeddings?api-version=1", "/%C3%A9//api/embed?api-version=1"},
		{"space, encoded slash, slash", "/v1 %2F/", "/v1%20//embeddings", "/v1%20//api/embed"},
	}
	for _, be := range bothBackends {
		for _, r := range rows {
			t.Run(fmt.Sprintf("%s/%s", be, r.name), func(t *testing.T) {
				want := r.wantOpenAI
				if be == BackendOllama {
					want = r.wantOllama
				}
				uri, _, _, _ := wireRequest(t, be, "", func(host string) string { return "http://" + host + r.suffix })
				if uri != want {
					t.Errorf("request target = %q, want %q", uri, want)
				}
			})
		}
		t.Run(string(be)+"/userinfo empty Secret", func(t *testing.T) {
			_, user, pass, auth := wireRequest(t, be, "", func(host string) string { return "http://user:pw@" + host + "/v1" })
			if user != "user" || pass != "pw" {
				t.Errorf("Basic auth = %q:%q (Authorization %q), want user:pw", user, pass, auth)
			}
		})
	}
}

// TestEmbed_UnbuildableJoinedURLDoesNotEchoPassword pins the R2 row: a
// BaseURL that passes validation but whose string form no longer parses
// must still yield a request, and no error may carry the password.
func TestEmbed_UnbuildableJoinedURLDoesNotEchoPassword(t *testing.T) {
	for _, be := range bothBackends {
		emb, err := New(Config{Backend: be, Model: "m", BaseURL: "http://user:pw@[::%25\x80]", Retry: retry.Config{MaxAttempts: 1}})
		if err != nil {
			t.Fatalf("%s: New: %v", be, err)
		}
		_, embErr := emb.Embed(context.Background(), "x")
		if embErr == nil {
			t.Fatalf("%s: Embed succeeded against an undialable host", be)
		}
		if strings.Contains(embErr.Error(), "create request") {
			t.Errorf("%s: err = %v; the request must build from the parsed URL, not fail", be, embErr)
		}
		for _, text := range errorTexts(embErr) {
			if strings.Contains(text, ":pw@") {
				t.Errorf("%s: error text %q echoes the URL password", be, text)
			}
		}
	}
}

// TestConfig_JitterRange: NaN joins the refused values, in Validate and in
// New.
func TestConfig_JitterRange(t *testing.T) {
	cfg := func(j float64) Config {
		return Config{Backend: BackendOllama, Model: "m", BaseURL: "http://localhost:11434", Retry: retry.Config{Jitter: j}}
	}
	for _, j := range []float64{math.NaN(), -0.1, 1.01, math.Inf(1), math.Inf(-1)} {
		if err := cfg(j).Validate(); !errors.Is(err, llmkit.ErrInvalidRequest) {
			t.Errorf("Validate, Jitter %v: err = %v, want ErrInvalidRequest", j, err)
		}
		if _, err := New(cfg(j)); !errors.Is(err, llmkit.ErrInvalidRequest) {
			t.Errorf("New, Jitter %v: err = %v, want ErrInvalidRequest", j, err)
		}
	}
	for _, j := range []float64{0, 0.5, 1} {
		if err := cfg(j).Validate(); err != nil {
			t.Errorf("Validate, Jitter %v: %v", j, err)
		}
		if _, err := New(cfg(j)); err != nil {
			t.Errorf("New, Jitter %v: %v", j, err)
		}
	}
}

// TestBackends_OneMapping: every name ParseBackend accepts builds an
// embedder that can embed, and every name it refuses is refused by
// Validate and New alike. A name accepted anywhere but missing a codec
// would panic on Embed or split the three doors.
func TestBackends_OneMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okEmbeddingBody)
	}))
	defer srv.Close()

	names := []string{"", "ollama", "openai-compatible", " ollama", "ollama ", "OLLAMA", "openai", "x"}
	for _, e := range backendCodecs {
		names = append(names, string(e.backend))
	}
	for _, name := range names {
		t.Run(fmt.Sprintf("%q", name), func(t *testing.T) {
			_, parseErr := ParseBackend(name)
			cfg := Config{Backend: Backend(name), Model: "m", BaseURL: srv.URL}
			vErr := cfg.Validate()
			emb, nErr := New(cfg)
			if (parseErr == nil) != (vErr == nil) || (parseErr == nil) != (nErr == nil) {
				t.Fatalf("ParseBackend err = %v, Validate err = %v, New err = %v; the three must agree", parseErr, vErr, nErr)
			}
			if nErr != nil {
				if !errors.Is(nErr, llmkit.ErrInvalidRequest) {
					t.Errorf("New err = %v, want ErrInvalidRequest", nErr)
				}
				return
			}
			if _, err := emb.Embed(context.Background(), "x"); err != nil {
				t.Errorf("Embed via an accepted backend %q: %v", name, err)
			}
		})
	}
	for _, e := range backendCodecs {
		if e.codec == nil {
			t.Errorf("backendCodecs row %q has no codec", e.backend)
		}
	}
}

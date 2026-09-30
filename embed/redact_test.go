package embed

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// leakSecret is a 40-byte fake credential, longer than the 8-byte prefix
// the property forbids, so a cap that cuts through it leaves a checkable
// front.
const leakSecret = "sk-live-9f8e7d6c5b4a39281706f5e4d3c2b1a0"

var leakBackends = []Backend{BackendOllama, BackendOpenAICompatible}

func leakPolicy() retry.Config {
	return retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond, Rand: func() float64 { return 0.5 }}
}

// forbiddenFor returns what no error may contain: each secret whole and
// its 8-byte prefix (every longer prefix holds the 8-byte one).
func forbiddenFor(secrets ...string) []string {
	var out []string
	for _, s := range secrets {
		out = append(out, s)
		if len(s) > 8 {
			out = append(out, s[:8])
		}
	}
	return out
}

func errorTreeTexts(err error) []string {
	var out []string
	var walk func(error)
	walk = func(e error) {
		if e == nil {
			return
		}
		out = append(out, e.Error())
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
	return append(out, fmt.Sprintf("%+v", err))
}

type embedEvents struct {
	mu   sync.Mutex
	errs []string
}

func (o *embedEvents) Observe(_ context.Context, ev llmkit.Event) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if ev.Embed != nil {
		o.errs = append(o.errs, ev.Embed.Err)
	}
}

// embedObserved runs one Embed against cfg and returns its error and the
// Err of the one EmbedEvent it emitted.
func embedObserved(t *testing.T, cfg Config) (eventErr string, err error) {
	t.Helper()
	if cfg.Model == "" {
		cfg.Model = "m"
	}
	if cfg.Retry.MaxAttempts == 0 {
		cfg.Retry = leakPolicy()
	}
	e, cerr := New(cfg)
	if cerr != nil {
		t.Fatalf("New: %v", cerr)
	}
	obs := &embedEvents{}
	_, err = Observe(e, obs).Embed(context.Background(), "x")
	if err == nil {
		t.Fatal("Embed succeeded, want an error")
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.errs) != 1 {
		t.Fatalf("EmbedEvents = %d, want 1", len(obs.errs))
	}
	return obs.errs[0], err
}

func assertNoLeak(t *testing.T, err error, eventErr string, forbidden []string) {
	t.Helper()
	for _, s := range append(errorTreeTexts(err), "EmbedEvent.Err: "+eventErr) {
		for _, f := range forbidden {
			if strings.Contains(s, f) {
				t.Errorf("text holds %q:\n%s", f, s)
			}
		}
	}
}

func closedAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}

func status(w http.ResponseWriter, code int, body string) {
	w.WriteHeader(code)
	_, _ = io.WriteString(w, body)
}

// TestEmbed_D4_ServerRoutesNeverLeakTheSecret: the server echoes the
// configured Secret through every route that reaches an error, on both
// backends, and no returned error, error under it, %+v, or EmbedEvent.Err
// holds the Secret or an 8-byte prefix of it.
func TestEmbed_D4_ServerRoutesNeverLeakTheSecret(t *testing.T) {
	type row struct {
		name     string
		backends []Backend
		secret   string
		handler  http.HandlerFunc
	}
	respond := func(code int, b string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { status(w, code, b) }
	}
	rows := []row{
		{"400 body echo", leakBackends, leakSecret, respond(400, `{"error":"invalid key `+leakSecret+`"}`)},
		{"401 body echo", leakBackends, leakSecret, respond(401, `{"error":"invalid key `+leakSecret+`"}`)},
		{"429 body echo", leakBackends, leakSecret, respond(429, leakSecret)},
		{"500 body echo", leakBackends, leakSecret, respond(500, `{"error":"invalid key `+leakSecret+`"}`)},
		{"200 openai error object", []Backend{BackendOpenAICompatible}, leakSecret,
			respond(200, `{"error":{"message":"invalid key `+leakSecret+`","type":"invalid_request_error"}}`)},
		{"200 openai error object, unknown type", []Backend{BackendOpenAICompatible}, leakSecret,
			respond(200, `{"error":{"message":"`+leakSecret+`","type":"`+leakSecret+`"}}`)},
		{"200 openai digits-only secret in a number type error", []Backend{BackendOpenAICompatible}, "12345678901234567890123",
			respond(200, `{"data":[{"embedding":[0.1],"index":12345678901234567890123}]}`)},
		{"200 ollama undecodable body", []Backend{BackendOllama}, leakSecret,
			respond(200, `{"model":"m","embeddings":[["`+leakSecret+`"]]}`)},
		{"200 openai undecodable body", []Backend{BackendOpenAICompatible}, leakSecret,
			respond(200, `{"data":[{"embedding":["`+leakSecret+`"],"index":0}]}`)},
		{"200 openai count mismatch echo in model", []Backend{BackendOpenAICompatible}, leakSecret,
			respond(200, `{"data":[],"model":"`+leakSecret+`"}`)},
		{"non-200 json-escaped secret", leakBackends, `sk-"live"-9f8e7d6c5b4a\3928<1706>`,
			respond(500, `{"error":"bad key sk-\"live\"-9f8e7d6c5b4a\\3928\u003c1706\u003e"}`)},
		{"200 openai json-escaped secret in the error message", []Backend{BackendOpenAICompatible}, `sk-"live"-9f8e7d6c5b4a\3928<1706>`,
			respond(200, `{"error":{"message":"bad key sk-\"live\"-9f8e7d6c5b4a\\3928\u003c1706\u003e","type":"server_error"}}`)},
	}
	for _, off := range []int{185, 190, 192} {
		rows = append(rows,
			row{fmt.Sprintf("non-200 body secret at offset %d", off), leakBackends, leakSecret,
				respond(500, strings.Repeat("x", off)+leakSecret+strings.Repeat("y", 100))},
			row{fmt.Sprintf("non-200 trimmed body secret at offset %d", off), leakBackends, leakSecret,
				respond(400, "  \n"+strings.Repeat("x", off)+leakSecret+strings.Repeat("y", 100))},
			row{fmt.Sprintf("200 openai error object secret at offset %d", off), []Backend{BackendOpenAICompatible}, leakSecret,
				respond(200, `{"error":{"message":"`+strings.Repeat("x", off)+leakSecret+`","type":"server_error"}}`)},
		)
	}
	for _, r := range rows {
		for _, be := range r.backends {
			t.Run(fmt.Sprintf("%s/%s", r.name, be), func(t *testing.T) {
				srv := httptest.NewServer(r.handler)
				defer srv.Close()
				evErr, err := embedObserved(t, Config{Backend: be, BaseURL: srv.URL, Secret: r.secret})
				assertNoLeak(t, err, evErr, forbiddenFor(r.secret))
			})
		}
	}
}

// TestEmbed_D4_KeepsKindStatusProviderAndRetryAfter: redaction leaves the
// fields a caller acts on.
func TestEmbed_D4_KeepsKindStatusProviderAndRetryAfter(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Retry-After", "7")
				status(w, 429, "slow down "+leakSecret)
			}))
			defer srv.Close()
			evErr, err := embedObserved(t, Config{Backend: be, BaseURL: srv.URL, Secret: leakSecret, Retry: retry.Config{MaxAttempts: 1}})
			assertNoLeak(t, err, evErr, forbiddenFor(leakSecret))
			var api *llmkit.APIError
			if !errors.As(err, &api) {
				t.Fatalf("err = %v, want an *APIError", err)
			}
			if api.Kind != llmkit.ErrRateLimited || api.StatusCode != 429 || !api.HasRetryAfter || api.RetryAfter != 7*time.Second || api.Provider != string(be) {
				t.Errorf("APIError = %+v, want Kind ErrRateLimited, status 429, Retry-After 7s, Provider %s", *api, be)
			}
		})
	}
}

// TestEmbed_D4_KindComesFromTheUnredactedBody: a Secret that overlaps the
// context-length phrase still yields ErrContextTooLong.
func TestEmbed_D4_KindComesFromTheUnredactedBody(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				status(w, 400, `{"error":"this model's maximum context length is 8192 tokens"}`)
			}))
			defer srv.Close()
			evErr, err := embedObserved(t, Config{Backend: be, BaseURL: srv.URL, Secret: "context length"})
			if !errors.Is(err, llmkit.ErrContextTooLong) {
				t.Fatalf("err = %v, want ErrContextTooLong", err)
			}
			assertNoLeak(t, err, evErr, []string{"context length"})
		})
	}
}

// TestEmbed_D4_MalformedHeaderLineQuotingTheSecret: net/http quotes the
// server's malformed header line in its transport error. The pads move the
// secret across the 200-byte cap of the transport Message.
func TestEmbed_D4_MalformedHeaderLineQuotingTheSecret(t *testing.T) {
	for _, be := range leakBackends {
		for pad := 0; pad <= 110; pad += 10 {
			t.Run(fmt.Sprintf("%s/pad %d", be, pad), func(t *testing.T) {
				ln, err := net.Listen("tcp", "127.0.0.1:0")
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = ln.Close() }()
				go func() {
					for {
						conn, err := ln.Accept()
						if err != nil {
							return
						}
						go func() {
							defer func() { _ = conn.Close() }()
							req, err := http.ReadRequest(bufio.NewReader(conn))
							if err != nil {
								return
							}
							_, _ = io.Copy(io.Discard, req.Body)
							_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nX"+strings.Repeat("p", pad)+" "+leakSecret+"\r\n\r\n")
						}()
					}
				}()
				evErr, e := embedObserved(t, Config{Backend: be, BaseURL: "http://" + ln.Addr().String(), Secret: leakSecret})
				assertNoLeak(t, e, evErr, forbiddenFor(leakSecret))
				if !strings.Contains(e.Error(), "malformed") {
					t.Errorf("err = %v, want the malformed-header transport error (row does not exercise its route)", e)
				}
			})
		}
	}
}

// TestEmbed_D4_UserinfoOfTheBaseURL: net/http's url.Error carries the URL's
// username; the userinfo-form username never survives, the password never
// does either.
func TestEmbed_D4_UserinfoOfTheBaseURL(t *testing.T) {
	closed := closedAddr(t)
	rows := []struct {
		name      string
		baseURL   string
		forbidden []string
	}{
		{"percent-encoded username, decoded in the error", "http://%68unter2@" + closed, []string{"hunter2"}},
		{"username only", "http://sk-USERKEY@" + closed + "/v1", []string{"sk-USERKEY"}},
		{"username and password", "http://sk-USERKEY:pw-hunter22@" + closed + "/v1", []string{"sk-USERKEY", "pw-hunter22"}},
	}
	for _, be := range leakBackends {
		for _, r := range rows {
			t.Run(fmt.Sprintf("%s/%s", be, r.name), func(t *testing.T) {
				evErr, err := embedObserved(t, Config{Backend: be, BaseURL: r.baseURL, Secret: leakSecret})
				assertNoLeak(t, err, evErr, r.forbidden)
				if _, _, retryable := llmkit.Classify(err); !retryable {
					t.Errorf("Classify(%v) = terminal, want a retryable transport failure", err)
				}
				if !strings.Contains(err.Error(), "***@") {
					t.Errorf("err = %v, want the userinfo replaced by ***", err)
				}
			})
		}
	}
}

// TestEmbed_D4_EmptySecretUserinfoEchoedInA500: with no Secret, net/http
// sends the URL's userinfo as a Basic credential. A server that echoes what
// it received, decoded or as the header, does not put the password into
// the error.
func TestEmbed_D4_EmptySecretUserinfoEchoedInA500(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			var got sync.Map
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				u, p, ok := r.BasicAuth()
				got.Store("basic", ok && u == "user" && p == "pw-hunter22")
				status(w, 500, fmt.Sprintf(`{"error":"who is %s:%s? header %s"}`, u, p, r.Header.Get("Authorization")))
			}))
			defer srv.Close()
			base := strings.Replace(srv.URL, "http://", "http://user:pw-hunter22@", 1)
			evErr, err := embedObserved(t, Config{Backend: be, BaseURL: base})
			assertNoLeak(t, err, evErr, []string{"pw-hunter22", "dXNlcjpwdy1odW50ZXIyMg=="})
			if v, _ := got.Load("basic"); v != true {
				t.Error("the server did not receive the userinfo as Basic auth: userinfo must stay accepted")
			}
		})
	}
}

// TestEmbed_D4_RedirectLocationWithUserinfo: a redirect whose Location puts
// a name in the URL userinfo, then fails to dial it.
func TestEmbed_D4_RedirectLocationWithUserinfo(t *testing.T) {
	closed := closedAddr(t)
	for _, be := range leakBackends {
		for _, tc := range []struct {
			name      string
			user      string
			forbidden []string
		}{
			{"secret as username", leakSecret, forbiddenFor(leakSecret)},
			{"foreign username", "otheruser", []string{"otheruser"}},
		} {
			t.Run(fmt.Sprintf("%s/%s", be, tc.name), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					http.Redirect(w, r, "http://"+tc.user+"@"+closed+"/x", http.StatusFound)
				}))
				defer srv.Close()
				evErr, err := embedObserved(t, Config{Backend: be, BaseURL: srv.URL, Secret: leakSecret})
				assertNoLeak(t, err, evErr, tc.forbidden)
			})
		}
	}
}

// TestEmbed_D4_CancelDuringBackoffAfterALeakingAttempt: the ctx ends while
// the loop waits after an attempt whose body echoed the Secret. retry.Do
// then formats "(last attempt: ...)" from the attempt's error.
func TestEmbed_D4_CancelDuringBackoffAfterALeakingAttempt(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				status(w, 500, "bad key "+leakSecret)
			}))
			defer srv.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			pol := leakPolicy()
			pol.Sleep = func(ctx context.Context, d time.Duration) error {
				cancel()
				return ctx.Err()
			}
			e, err := New(Config{Backend: be, Model: "m", BaseURL: srv.URL, Secret: leakSecret, Retry: pol})
			if err != nil {
				t.Fatal(err)
			}
			obs := &embedEvents{}
			_, err = Observe(e, obs).Embed(ctx, "x")
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want context.Canceled", err)
			}
			if !strings.Contains(err.Error(), "last attempt") {
				t.Fatalf("err = %v, want the (last attempt: ...) text (row does not exercise its route)", err)
			}
			if len(obs.errs) != 1 {
				t.Fatalf("events = %d", len(obs.errs))
			}
			assertNoLeak(t, err, obs.errs[0], forbiddenFor(leakSecret))
		})
	}
}

// TestEmbed_D4_ShortPasswordIsMaskedShortUsernameIsNot: the documented
// asymmetry. A password is a value and masks matching text at any length; a
// username is redacted only as URL userinfo.
func TestEmbed_D4_ShortPasswordIsMaskedShortUsernameIsNot(t *testing.T) {
	run := func(t *testing.T, be Backend, userinfo, message string) string {
		t.Helper()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			status(w, 500, message)
		}))
		defer srv.Close()
		base := strings.Replace(srv.URL, "http://", "http://"+userinfo+"@", 1)
		_, err := embedObserved(t, Config{Backend: be, BaseURL: base})
		var api *llmkit.APIError
		if !errors.As(err, &api) {
			t.Fatalf("err = %v, want an *APIError", err)
		}
		return api.Message
	}
	for _, be := range leakBackends {
		t.Run(string(be)+"/password pw", func(t *testing.T) {
			got := run(t, be, "user:pw", `{"error":"upstream pw rejected"}`)
			if want := `{"error":"upstream *** rejected"}`; got != want {
				t.Errorf("Message = %q, want %q", got, want)
			}
		})
		t.Run(string(be)+"/username u", func(t *testing.T) {
			got := run(t, be, "u:longpassword", `{"error":"unauthorized user u"}`)
			if want := `{"error":"unauthorized user u"}`; got != want {
				t.Errorf("Message = %q, want %q (a username is not a value)", got, want)
			}
		})
	}
}

// TestEmbed_D4_NoRedactionLeavesTheErrorAlone: with nothing to redact, the
// chain is intact. The per-attempt timeout reaches the *url.Error net/http
// built and its timeout net.Error even when a Secret is configured;
// context.DeadlineExceeded is a net.Error too, so the *url.Error is what
// proves the chain was not rebuilt.
func TestEmbed_D4_NoRedactionLeavesTheErrorAlone(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				select {
				case <-r.Context().Done():
				case <-time.After(3 * time.Second):
				}
			}))
			defer srv.Close()
			e, err := New(Config{Backend: be, Model: "m", BaseURL: srv.URL, Secret: leakSecret, Retry: retry.Config{MaxAttempts: 1, RequestTimeout: 50 * time.Millisecond}})
			if err != nil {
				t.Fatal(err)
			}
			_, err = e.Embed(context.Background(), "x")
			var ue *url.Error
			if !errors.As(err, &ue) {
				t.Errorf("err = %v, want a chain that reaches the *url.Error", err)
			}
			var ne net.Error
			if !errors.As(err, &ne) || !ne.Timeout() {
				t.Errorf("err = %v, want a chain that reaches a timeout net.Error", err)
			}
			if !errors.Is(err, llmkit.ErrServer) || !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("err = %v, want ErrServer and DeadlineExceeded", err)
			}
		})
	}
}

// TestEmbed_D4_NoRedactionKeepsTheDialError: a refused dial with a Secret
// configured and nothing of it in the text keeps the *net.OpError.
func TestEmbed_D4_NoRedactionKeepsTheDialError(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			_, err := embedObserved(t, Config{Backend: be, BaseURL: "http://" + closedAddr(t), Secret: leakSecret})
			var op *net.OpError
			if !errors.As(err, &op) {
				t.Errorf("err = %v, want a chain that reaches the *net.OpError", err)
			}
		})
	}
}

// TestEmbed_D4_PreserveLegs: a URL password never leaks from a retryable
// transport failure, and a malformed 200 body keeps its decode-response error.
func TestEmbed_D4_PreserveLegs(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be)+"/transport failure with a password in the URL", func(t *testing.T) {
			evErr, err := embedObserved(t, Config{Backend: be, BaseURL: "http://user:hunter2@" + closedAddr(t), Secret: leakSecret})
			assertNoLeak(t, err, evErr, []string{"hunter2"})
			if _, _, retryable := llmkit.Classify(err); !retryable {
				t.Errorf("Classify(%v) = terminal", err)
			}
		})
		t.Run(string(be)+"/decode failure body", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				status(w, 200, "this is not json")
			}))
			defer srv.Close()
			evErr, err := embedObserved(t, Config{Backend: be, BaseURL: srv.URL, Secret: leakSecret})
			assertNoLeak(t, err, evErr, forbiddenFor(leakSecret))
			if !strings.Contains(err.Error(), "decode response") {
				t.Errorf("err = %v, want the decode-response error unchanged", err)
			}
		})
	}
}

// TestEmbed_D4_CancelMidAttemptWithUserinfo: the caller's cancellation while
// an attempt is in flight. The transport error names the URL, its username
// included, and the cancellation is still matched by errors.Is.
func TestEmbed_D4_CancelMidAttemptWithUserinfo(t *testing.T) {
	for _, be := range leakBackends {
		t.Run(string(be), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				select {
				case <-r.Context().Done():
				case <-time.After(3 * time.Second):
				}
			}))
			defer srv.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			base := strings.Replace(srv.URL, "http://", "http://sk-USERKEY@", 1)
			e, err := New(Config{Backend: be, Model: "m", BaseURL: base, Secret: leakSecret, Retry: leakPolicy()})
			if err != nil {
				t.Fatal(err)
			}
			obs := &embedEvents{}
			_, err = Observe(e, obs).Embed(ctx, "x")
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("err = %v, want the caller's DeadlineExceeded", err)
			}
			if len(obs.errs) != 1 {
				t.Fatalf("events = %d", len(obs.errs))
			}
			assertNoLeak(t, err, obs.errs[0], []string{"sk-USERKEY"})
			if !strings.Contains(err.Error(), "***@") {
				t.Errorf("err = %v, want the userinfo replaced (row does not exercise its route)", err)
			}
		})
	}
}

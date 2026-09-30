package decide

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// leakSecret is a 40-byte fake credential, longer than the 8-byte prefix
// the property forbids, so a cap that cuts through it leaves a checkable
// front.
const leakSecret = "sk-live-9f8e7d6c5b4a39281706f5e4d3c2b1a0"

var oneQuestion = Questions{"q": Noul{Instructions: "i"}}

// leakPolicy is a fast three-attempt policy: jitter-free, no real waiting.
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

// askObserved runs one Ask against cfg and returns its error and the Err
// of the one DecisionEvent it emitted.
func askObserved(t *testing.T, cfg Config, q Questions) (eventErr string, err error) {
	t.Helper()
	obs := &decisionCapture{}
	cfg.Observer = obs
	if cfg.Model == "" {
		cfg.Model = "jev-latest"
	}
	if cfg.Retry.MaxAttempts == 0 {
		cfg.Retry = leakPolicy()
	}
	c, cerr := New(cfg)
	if cerr != nil {
		t.Fatalf("New: %v", cerr)
	}
	_, err = c.Ask(context.Background(), "state", q)
	if err == nil {
		t.Fatal("Ask succeeded, want an error")
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.events) != 1 || obs.events[0].Decision == nil {
		t.Fatalf("events = %d, want one DecisionEvent", len(obs.events))
	}
	return obs.events[0].Decision.Err, err
}

func assertNoLeak(t *testing.T, err error, eventErr string, forbidden []string) {
	t.Helper()
	texts := append(errorTexts(err), fmt.Sprintf("%+v", err), "DecisionEvent.Err: "+eventErr)
	for _, s := range texts {
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

func answerJSON(t *testing.T, answers map[string]json.RawMessage) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"model":   "jev-1.13.0",
		"answers": answers,
		"usage":   map[string]int{"input_tokens": 1, "output_tokens": 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func quoted(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// TestAsk_D4_ServerRoutesNeverLeakTheSecret: the server echoes the
// configured Secret through every route that reaches an error, and no
// returned error, error under it, %+v, or DecisionEvent.Err holds the
// Secret or an 8-byte prefix of it.
func TestAsk_D4_ServerRoutesNeverLeakTheSecret(t *testing.T) {
	type row struct {
		name      string
		secret    string
		questions Questions
		handler   http.HandlerFunc
	}
	body := func(status int, b string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { writeStatus(w, status, b) }
	}
	rows := []row{
		{"400 body echo", leakSecret, oneQuestion, body(400, `{"error":"invalid key `+leakSecret+`"}`)},
		{"401 body echo", leakSecret, oneQuestion, body(401, `{"error":"invalid key `+leakSecret+`"}`)},
		{"429 body echo", leakSecret, oneQuestion, body(429, leakSecret)},
		{"500 body echo", leakSecret, oneQuestion, body(500, `{"error":"invalid key `+leakSecret+`"}`)},
		{"599 body echo", leakSecret, oneQuestion, body(599, "key="+leakSecret)},
		{"200 unknown answer id", leakSecret, oneQuestion, body(200, answerJSON(t, map[string]json.RawMessage{leakSecret: json.RawMessage(`{"type":"noul","noul":0.5}`)}))},
		{"200 answer type", leakSecret, oneQuestion, body(200, answerJSON(t, map[string]json.RawMessage{"q": json.RawMessage(`{"type":` + string(quoted(leakSecret)) + `}`)}))},
		{"200 probabilities key with wrong-typed value", leakSecret, Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": "x"}}},
			body(200, answerJSON(t, map[string]json.RawMessage{"q": json.RawMessage(`{"type":"choice","choice":"a","probabilities":{` + string(quoted(leakSecret)) + `:"wrong"}}`)}))},
		{"200 score legend key", leakSecret, Questions{"q": Score{Instructions: "i", Levels: []any{"lo", "hi"}}},
			body(200, answerJSON(t, map[string]json.RawMessage{"q": json.RawMessage(`{"type":"score","score":0.5,"legend":{` + string(quoted(leakSecret)) + `:"lo","1":"hi"},"probabilities":{"0":0.5,"1":0.5}}`)}))},
		{"200 digits-only secret in a number type error", "12345678901234567890123", oneQuestion,
			body(200, `{"model":"m","answers":{},"usage":{"input_tokens":12345678901234567890123,"output_tokens":1}}`)},
		{"200 json-escaped secret in an unknown answer id", `sk-"live"-9f8e7d6c5b4a\3928<1706>`, oneQuestion,
			body(200, `{"model":"m","answers":{"sk-\"live\"-9f8e7d6c5b4a\\3928\u003c1706\u003e":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}`)},
		{"non-200 json-escaped secret", `sk-"live"-9f8e7d6c5b4a\3928<1706>`, oneQuestion,
			body(500, `{"error":"bad key sk-\"live\"-9f8e7d6c5b4a\\3928\u003c1706\u003e"}`)},
	}
	for _, off := range []int{185, 190, 192} {
		rows = append(rows, row{fmt.Sprintf("non-200 body secret at offset %d", off), leakSecret, oneQuestion,
			body(500, strings.Repeat("x", off)+leakSecret+strings.Repeat("y", 100))})
		rows = append(rows, row{fmt.Sprintf("non-200 trimmed body secret at offset %d", off), leakSecret, oneQuestion,
			body(400, "  \n"+strings.Repeat("x", off)+leakSecret+strings.Repeat("y", 100))})
	}
	// Secrets that %q escapes where JSON does not: decide quotes the answer
	// id and type with %q, so the echo reaches the error text as \u200b or
	// as \U000e0001. A byte that is not UTF-8 reaches it as the U+FFFD
	// json.Unmarshal put in its place, and %q then escapes the \u200b
	// after it.
	const okBody = `{"model":"m","usage":{"input_tokens":1,"output_tokens":1},"answers":`
	for _, c := range []struct{ class, secret string }{
		{`\u`, "sk-proj-ABCDEFGH\u200btail"},
		{`\U`, "sk-proj-ABCDEFGH\U000e0001tail"},
		{`invalid byte and \u`, "sk-proj-ABCDEFGH\xff\u200btail"},
	} {
		rows = append(rows,
			row{"200 unknown answer id, " + c.class + " secret", c.secret, oneQuestion,
				body(200, okBody+`{"`+c.secret+`":{"type":"noul","noul":0.5}}}`)},
			row{"200 answer type, " + c.class + " secret", c.secret, oneQuestion,
				body(200, okBody+`{"q":{"type":"`+c.secret+`","noul":0.5}}}`)})
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			srv := httptest.NewServer(r.handler)
			defer srv.Close()
			evErr, err := askObserved(t, Config{Secret: r.secret, BaseURL: srv.URL}, r.questions)
			assertNoLeak(t, err, evErr, forbiddenFor(r.secret))
			var api *llmkit.APIError
			if !errors.As(err, &api) || api.Provider != providerName {
				t.Errorf("err = %v, want an *APIError with Provider %q", err, providerName)
			}
		})
	}
}

// TestAsk_D4_KeepsKindStatusAndRetryAfter: redaction leaves the fields a
// caller acts on.
func TestAsk_D4_KeepsKindStatusAndRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		writeStatus(w, 429, "slow down "+leakSecret)
	}))
	defer srv.Close()
	evErr, err := askObserved(t, Config{Secret: leakSecret, BaseURL: srv.URL, Retry: retry.Config{MaxAttempts: 1}}, oneQuestion)
	assertNoLeak(t, err, evErr, forbiddenFor(leakSecret))
	var api *llmkit.APIError
	if !errors.As(err, &api) {
		t.Fatalf("err = %v, want an *APIError", err)
	}
	if api.Kind != llmkit.ErrRateLimited || api.StatusCode != 429 || !api.HasRetryAfter || api.RetryAfter != 7*time.Second || api.Provider != "typesafe" {
		t.Errorf("APIError = %+v, want Kind ErrRateLimited, status 429, Retry-After 7s, Provider typesafe", *api)
	}
	if !errors.Is(err, llmkit.ErrRateLimited) {
		t.Errorf("errors.Is(err, ErrRateLimited) = false")
	}
}

// TestAsk_D4_KindComesFromTheUnredactedBody: a Secret that overlaps the
// context-length phrase still yields ErrContextTooLong.
func TestAsk_D4_KindComesFromTheUnredactedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, 400, `{"error":"this model's maximum context length is 8192 tokens"}`)
	}))
	defer srv.Close()
	evErr, err := askObserved(t, Config{Secret: "context length", BaseURL: srv.URL}, oneQuestion)
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("err = %v, want ErrContextTooLong", err)
	}
	assertNoLeak(t, err, evErr, []string{"context length"})
}

// TestAsk_D4_MalformedHeaderLineQuotingTheSecret: net/http quotes the
// server's malformed header line in its transport error. The pads move the
// secret across the 200-byte cap of the transport Message.
func TestAsk_D4_MalformedHeaderLineQuotingTheSecret(t *testing.T) {
	for pad := 0; pad <= 110; pad += 10 {
		t.Run(fmt.Sprintf("pad %d", pad), func(t *testing.T) {
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
			evErr, e := askObserved(t, Config{Secret: leakSecret, BaseURL: "http://" + ln.Addr().String()}, oneQuestion)
			assertNoLeak(t, e, evErr, forbiddenFor(leakSecret))
			if !strings.Contains(e.Error(), "malformed") {
				t.Errorf("err = %v, want the malformed-header transport error (row does not exercise its route)", e)
			}
		})
	}
}

// TestAsk_D4_UserinfoOfTheBaseURL: net/http's url.Error carries the URL's
// username; the userinfo-form username never survives, the password never
// does either, and Basic-auth-via-URL still reaches the wire.
func TestAsk_D4_UserinfoOfTheBaseURL(t *testing.T) {
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
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			evErr, err := askObserved(t, Config{Secret: leakSecret, BaseURL: r.baseURL}, oneQuestion)
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

// TestAsk_D4_RedirectLocationWithUserinfo: a redirect whose Location puts a
// name in the URL userinfo, then fails to dial it.
func TestAsk_D4_RedirectLocationWithUserinfo(t *testing.T) {
	closed := closedAddr(t)
	for _, tc := range []struct {
		name      string
		user      string
		forbidden []string
	}{
		{"secret as username", leakSecret, forbiddenFor(leakSecret)},
		{"foreign username", "otheruser", []string{"otheruser"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, "http://"+tc.user+"@"+closed+"/x", http.StatusFound)
			}))
			defer srv.Close()
			evErr, err := askObserved(t, Config{Secret: leakSecret, BaseURL: srv.URL}, oneQuestion)
			assertNoLeak(t, err, evErr, tc.forbidden)
		})
	}
}

// TestAsk_D4_CancelDuringBackoffAfterALeakingAttempt: the ctx ends while the
// loop waits after an attempt whose body echoed the Secret. retry.Do then
// formats "(last attempt: ...)" from the attempt's error.
func TestAsk_D4_CancelDuringBackoffAfterALeakingAttempt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, 500, "bad key "+leakSecret)
	}))
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pol := leakPolicy()
	pol.Sleep = func(ctx context.Context, d time.Duration) error {
		cancel()
		return ctx.Err()
	}
	obs := &decisionCapture{}
	c, err := New(Config{Secret: leakSecret, Model: "m", BaseURL: srv.URL, Retry: pol, Observer: obs})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Ask(ctx, "state", oneQuestion)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if !strings.Contains(err.Error(), "last attempt") {
		t.Fatalf("err = %v, want the (last attempt: ...) text (row does not exercise its route)", err)
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.events) != 1 {
		t.Fatalf("events = %d", len(obs.events))
	}
	assertNoLeak(t, err, obs.events[0].Decision.Err, forbiddenFor(leakSecret))
}

// TestAsk_D4_ShortPasswordIsMaskedShortUsernameIsNot: the documented
// asymmetry. A password is a value and masks matching text at any length; a
// username is redacted only as URL userinfo.
func TestAsk_D4_ShortPasswordIsMaskedShortUsernameIsNot(t *testing.T) {
	run := func(t *testing.T, userinfo, message string) string {
		t.Helper()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeStatus(w, 500, message)
		}))
		defer srv.Close()
		base := strings.Replace(srv.URL, "http://", "http://"+userinfo+"@", 1)
		_, err := askObserved(t, Config{Secret: leakSecret, BaseURL: base}, oneQuestion)
		var api *llmkit.APIError
		if !errors.As(err, &api) {
			t.Fatalf("err = %v, want an *APIError", err)
		}
		return api.Message
	}
	t.Run("password pw", func(t *testing.T) {
		got := run(t, "user:pw", `{"error":"upstream pw rejected"}`)
		if want := `{"error":"upstream *** rejected"}`; got != want {
			t.Errorf("Message = %q, want %q", got, want)
		}
	})
	t.Run("username u", func(t *testing.T) {
		got := run(t, "u:longpassword", `{"error":"unauthorized user u"}`)
		if want := `{"error":"unauthorized user u"}`; got != want {
			t.Errorf("Message = %q, want %q (a username is not a value)", got, want)
		}
	})
}

// TestAsk_D4_NoRedactionLeavesTheErrorAlone: with nothing to redact, the
// chain is intact. The per-attempt timeout reaches the *url.Error net/http
// built and its timeout net.Error; context.DeadlineExceeded is a net.Error
// too, so the *url.Error is what proves the chain was not rebuilt.
func TestAsk_D4_NoRedactionLeavesTheErrorAlone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
		}
	}))
	defer srv.Close()
	c, err := New(Config{Secret: leakSecret, Model: "m", BaseURL: srv.URL, Retry: retry.Config{MaxAttempts: 1, RequestTimeout: 50 * time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Ask(context.Background(), "state", oneQuestion)
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
}

// TestAsk_D4_NoRedactionKeepsTheDialError: a refused dial with a Secret
// configured and nothing of it in the text keeps the *net.OpError.
func TestAsk_D4_NoRedactionKeepsTheDialError(t *testing.T) {
	_, err := askObserved(t, Config{Secret: leakSecret, BaseURL: "http://" + closedAddr(t)}, oneQuestion)
	var op *net.OpError
	if !errors.As(err, &op) {
		t.Errorf("err = %v, want a chain that reaches the *net.OpError", err)
	}
}

// TestAsk_D4_PreserveLegs: a URL password never leaks from a retryable
// transport failure, and a malformed-JSON body keeps its ErrServer error.
func TestAsk_D4_PreserveLegs(t *testing.T) {
	t.Run("transport failure with a password in the URL", func(t *testing.T) {
		closed := closedAddr(t)
		evErr, err := askObserved(t, Config{Secret: leakSecret, BaseURL: "http://user:hunter2@" + closed}, oneQuestion)
		assertNoLeak(t, err, evErr, []string{"hunter2"})
		if _, _, retryable := llmkit.Classify(err); !retryable {
			t.Errorf("Classify(%v) = terminal", err)
		}
	})
	t.Run("decode failure body", func(t *testing.T) {
		srv := httptest.NewServer(okHandler(`this is not json`))
		defer srv.Close()
		evErr, err := askObserved(t, Config{Secret: leakSecret, BaseURL: srv.URL}, oneQuestion)
		assertNoLeak(t, err, evErr, forbiddenFor(leakSecret))
		var api *llmkit.APIError
		if !errors.As(err, &api) || api.Kind != llmkit.ErrServer || api.StatusCode != 200 || !strings.Contains(api.Message, "malformed JSON") {
			t.Errorf("err = %v, want the ErrServer 200 malformed-JSON error unchanged", err)
		}
	})
}

// TestAsk_D4_CancelMidAttemptWithUserinfo: the caller's cancellation while
// an attempt is in flight. The transport error names the URL, its username
// included, and the cancellation is still matched by errors.Is.
func TestAsk_D4_CancelMidAttemptWithUserinfo(t *testing.T) {
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
	obs := &decisionCapture{}
	c, err := New(Config{Secret: leakSecret, Model: "m", BaseURL: base, Retry: leakPolicy(), Observer: obs})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Ask(ctx, "state", oneQuestion)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the caller's DeadlineExceeded", err)
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.events) != 1 {
		t.Fatalf("events = %d", len(obs.events))
	}
	assertNoLeak(t, err, obs.events[0].Decision.Err, []string{"sk-USERKEY"})
	if !strings.Contains(err.Error(), "***@") {
		t.Errorf("err = %v, want the userinfo replaced (row does not exercise its route)", err)
	}
}

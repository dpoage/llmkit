package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	sdkanthropic "github.com/anthropics/anthropic-sdk-go"
	sdkopenai "github.com/openai/openai-go/v3"
	"google.golang.org/genai"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// Fixture credentials. rxTwin has the length of rxKey and shares no 12-byte
// fragment with rxKey or rxPass.
const (
	rxKey  = "sk-WIREPROBE-0123456789abcdef"
	rxPass = "PWPROBEpassword-0123456789XYZ"
	rxTwin = "qk-CLEANTWIN-ZYXWVUTSRQmnopqr"
)

// rxOffsets are the places the credential sits in a body; 172..199 straddle
// the 200-byte Message cap for the SDK error texts.
var rxOffsets = []int{0, 150, 172, 180, 185, 188, 190, 199, 250}

type rxClient struct {
	name string
	typ  Type
	auth Auth
	fam  string // body family: anthropic, openai or google
}

var rxClients = []rxClient{
	{"anthropic-key", TypeAnthropic, AuthAPIKey, "anthropic"},
	{"anthropic-oauth", TypeAnthropic, AuthOAuthToken, "anthropic"},
	{"openai", TypeOpenAI, AuthAPIKey, "openai"},
	{"openai-compatible", TypeOpenAICompatible, AuthAPIKey, "openai"},
	{"google", TypeGoogle, AuthAPIKey, "google"},
}

var rxFamilies = []string{"anthropic", "openai", "google"}

func rxPad(n int) string { return strings.Repeat("x", n) }

func rxJSONErr(fam, msg string, status int) string {
	var v any
	switch fam {
	case "anthropic":
		v = map[string]any{"type": "error", "error": map[string]any{"type": "authentication_error", "message": msg}}
	case "openai":
		v = map[string]any{"error": map[string]any{"message": msg, "type": "invalid_request_error", "code": "invalid_api_key", "param": nil}}
	default:
		v = map[string]any{"error": map[string]any{"code": status, "message": msg, "status": "UNAUTHENTICATED"}}
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// rxSSEErr is a 200 stream whose data line quotes msg: the family's error
// event, or for google a non-JSON line (a Gemini error object decodes to
// "no candidates" and cannot carry msg).
func rxSSEErr(fam, msg string) string {
	switch fam {
	case "anthropic":
		b, _ := json.Marshal(map[string]any{"type": "error", "error": map[string]any{"type": "overloaded_error", "message": msg}})
		return "event: error\ndata: " + string(b) + "\n\n"
	case "openai":
		b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": msg, "type": "server_error"}})
		return "data: " + string(b) + "\n\n"
	}
	return "data: " + msg + "\n\n"
}

type rxBody struct {
	name   string
	status int
	ctype  string
	body   func(fam, cred string, off int) string
}

func rxText(_, cred string, off int) string { return rxPad(off) + cred + rxPad(60) }

var rxBodies = []rxBody{
	{"401json", 401, "application/json", func(f, c string, o int) string { return rxJSONErr(f, rxPad(o)+c, 401) }},
	{"400text", 400, "text/plain", rxText},
	{"500text", 500, "text/plain", rxText},
	{"200errobj", 200, "application/json", func(f, c string, o int) string { return rxJSONErr(f, rxPad(o)+c, 500) }},
	{"200nonjson", 200, "text/plain", rxText},
	{"200badjson", 200, "application/json", rxText},
	{"200sse", 200, "text/event-stream", func(f, c string, o int) string { return rxSSEErr(f, rxPad(o)+c) }},
}

type rxObserver struct {
	mu     sync.Mutex
	events []string
}

func (o *rxObserver) Observe(_ context.Context, ev llmkit.Event) {
	b, _ := json.Marshal(ev)
	o.mu.Lock()
	o.events = append(o.events, string(b))
	o.mu.Unlock()
}

type rxOutcome struct {
	err    error
	events []string
}

// rxCall runs one Complete or Stream through provider.New with two
// attempts allowed, so a retryable failure puts two Attempt events on the
// observer.
func rxCall(ctx context.Context, t *testing.T, c rxClient, stream bool, baseURL string) (out rxOutcome) {
	t.Helper()
	obs := &rxObserver{}
	cl, err := New(context.Background(),
		Spec{Type: c.typ, Auth: c.auth, Model: "m-test", Secret: rxKey, BaseURL: baseURL},
		Options{Observer: obs, Retry: retry.Config{MaxAttempts: 2, Sleep: func(context.Context, time.Duration) error { return nil }}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.err = fmt.Errorf("PANIC: %v", r)
		}
		obs.mu.Lock()
		out.events = append([]string(nil), obs.events...)
		obs.mu.Unlock()
	}()
	req := llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}, MaxTokens: 16}
	if stream {
		_, out.err = llmkit.Stream(ctx, cl, req, func(llmkit.Delta) error { return nil })
	} else {
		_, out.err = cl.Complete(ctx, req)
	}
	return out
}

// rxReachable lists the text reachable from err: Error() of every error in
// the Unwrap tree, and Message of every *llmkit.APIError met.
func rxReachable(err error) []string {
	if err == nil {
		return nil
	}
	out := []string{err.Error()}
	if api, ok := err.(*llmkit.APIError); ok {
		out = append(out, api.Message)
	}
	switch u := err.(type) {
	case interface{ Unwrap() error }:
		out = append(out, rxReachable(u.Unwrap())...)
	case interface{ Unwrap() []error }:
		for _, e := range u.Unwrap() {
			out = append(out, rxReachable(e)...)
		}
	}
	return out
}

// rxFragments are the 12-byte substrings of each credential.
func rxFragments(creds ...string) []string {
	var out []string
	for _, c := range creds {
		for i := 0; i+12 <= len(c); i++ {
			out = append(out, c[i:i+12])
		}
	}
	return out
}

func rxHolds(s string, frags []string) bool {
	for _, f := range frags {
		if strings.Contains(s, f) {
			return true
		}
	}
	return false
}

// rxLeak names where a credential fragment is reachable, or "".
func rxLeak(o rxOutcome, frags []string) string {
	for _, s := range rxReachable(o.err) {
		if rxHolds(s, frags) {
			return "error: " + rxClip(s)
		}
	}
	for _, e := range o.events {
		if rxHolds(e, frags) {
			return "observer event: " + rxClip(e)
		}
	}
	return ""
}

func rxClip(s string) string {
	if len(s) > 400 {
		return s[:400] + "..."
	}
	return s
}

// rxClosedPort returns a loopback address nothing listens on.
func rxClosedPort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	_ = l.Close()
	return addr
}

func rxServe(t *testing.T, status int, ctype, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", ctype)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// rxVisit receives one finished call; cell names it within the subtest t.
type rxVisit func(t *testing.T, cell string, c rxClient, stream bool, o rxOutcome)

func rxMode(stream bool) string {
	if stream {
		return "Stream"
	}
	return "Complete"
}

// rxPadKinds are the cells that put cred in the BaseURL path: a dial
// failure, and two local refusals (a malformed escape after cred).
var rxPadKinds = []string{"dial", "refuse-2", "refuse-zz"}

// rxEachBody runs the body cells, with cred at each offset of the body.
func rxEachBody(t *testing.T, cred string, offsets []int, visit rxVisit) {
	t.Helper()
	for _, b := range rxBodies {
		for _, off := range offsets {
			cell := fmt.Sprintf("%s@%d", b.name, off)
			t.Run(cell, func(t *testing.T) {
				for _, fam := range rxFamilies {
					srv := rxServe(t, b.status, b.ctype, b.body(fam, cred, off))
					for _, c := range rxClients {
						if c.fam != fam {
							continue
						}
						for _, stream := range []bool{false, true} {
							t.Run(c.name+"/"+rxMode(stream), func(t *testing.T) {
								visit(t, cell, c, stream, rxCall(context.Background(), t, c, stream, srv.URL))
							})
						}
					}
				}
			})
		}
	}
}

// rxEachPadded runs the BaseURL-path cells once per pad; the pad moves cred
// across byte 200 of the Message.
func rxEachPadded(t *testing.T, cred string, pads []int, visit rxVisit) {
	t.Helper()
	url := map[string]func(addr, pad string) string{
		"dial":      func(addr, pad string) string { return "http://" + addr + "/" + pad + "/" + cred },
		"refuse-2":  func(_, pad string) string { return "http://127.0.0.1:1/" + pad + "/" + cred + "/%2" },
		"refuse-zz": func(_, pad string) string { return "http://127.0.0.1:1/" + pad + "/" + cred + "/%zz" },
	}
	for _, kind := range rxPadKinds {
		for _, c := range rxClients {
			for _, stream := range []bool{false, true} {
				t.Run(kind+"/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
					addr := rxClosedPort(t)
					for _, pad := range pads {
						visit(t, fmt.Sprintf("%s/pad=%d", kind, pad), c, stream, rxCall(context.Background(), t, c, stream, url[kind](addr, rxPad(pad))))
					}
				})
			}
		}
	}
}

// rxEachCancel cancels the caller's context 100 ms into a call to a server
// that never answers, with cred in the BaseURL path.
func rxEachCancel(t *testing.T, cred string, visit rxVisit) {
	t.Helper()
	for _, c := range rxClients {
		for _, stream := range []bool{false, true} {
			t.Run("cancel/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
					_, _ = io.Copy(io.Discard, r.Body) // the server notices a closed client only after the body is read
					select {
					case <-r.Context().Done():
					case <-time.After(10 * time.Second):
					}
				}))
				defer srv.Close()
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				time.AfterFunc(100*time.Millisecond, cancel)
				visit(t, "cancel", c, stream, rxCall(ctx, t, c, stream, srv.URL+"/"+cred))
			})
		}
	}
}

func rxEachCell(t *testing.T, cred string, pads []int, visit rxVisit) {
	t.Helper()
	rxEachBody(t, cred, rxOffsets, visit)
	rxEachPadded(t, cred, pads, visit)
	rxEachCancel(t, cred, visit)
}

func rxSweep() []int {
	var pads []int
	for p := 60; p <= 180; p++ {
		pads = append(pads, p)
	}
	return pads
}

// TestRedactMatrix_NoCredentialReachable: for every adapter, entry point
// and cell, no text reachable from the returned error or delivered to the
// observer holds a 12-byte fragment of the Secret or the BaseURL password.
func TestRedactMatrix_NoCredentialReachable(t *testing.T) {
	frags := rxFragments(rxKey, rxPass)
	rxEachCell(t, rxKey, rxSweep(), func(t *testing.T, cell string, _ rxClient, _ bool, o rxOutcome) {
		if where := rxLeak(o, frags); where != "" {
			t.Errorf("%s leaks a credential fragment in %s", cell, where)
		}
	})
}

// TestRedactMatrix_UserinfoPassword: a BaseURL password reaches no error
// text, whether the vendor echoes it or the SDK quotes the URL.
func TestRedactMatrix_UserinfoPassword(t *testing.T) {
	frags := rxFragments(rxKey, rxPass)
	type cell struct {
		name   string
		status int
		echo   bool
	}
	cells := []cell{{"401-echoes-password", 401, true}, {"401-no-echo", 401, false}, {"503-no-echo", 503, false}}
	for _, ce := range cells {
		for _, fam := range rxFamilies {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				msg := "denied"
				if ce.echo {
					msg = "bad password " + rxPass + " auth=" + r.Header.Get("Authorization")
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(ce.status)
				_, _ = w.Write([]byte(rxJSONErr(fam, msg, ce.status)))
			}))
			base := strings.Replace(srv.URL, "http://", "http://user:"+rxPass+"@", 1)
			for _, c := range rxClients {
				if c.fam != fam {
					continue
				}
				for _, stream := range []bool{false, true} {
					t.Run(ce.name+"/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
						if where := rxLeak(rxCall(context.Background(), t, c, stream, base), frags); where != "" {
							t.Errorf("password reachable in %s", where)
						}
					})
				}
			}
			srv.Close()
		}
	}
	for _, c := range rxClients {
		for _, stream := range []bool{false, true} {
			t.Run("dial/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
				base := "http://user:" + rxPass + "@" + rxClosedPort(t) + "/"
				if where := rxLeak(rxCall(context.Background(), t, c, stream, base), frags); where != "" {
					t.Errorf("password reachable in %s", where)
				}
			})
		}
	}
}

// TestRedactMatrix_PadsStraddleTheCap: the dial and refusal pads put the
// credential across byte 200 of the Message on every client. A clean twin
// of the same length shows the cut text holding its first 12 bytes only.
func TestRedactMatrix_PadsStraddleTheCap(t *testing.T) {
	if len(rxTwin) != len(rxKey) || rxHolds(rxTwin, rxFragments(rxKey, rxPass)) {
		t.Fatalf("rxTwin must match rxKey in length and share no 12-byte fragment")
	}
	seen := map[string]bool{}
	rxEachPadded(t, rxTwin, rxSweep(), func(t *testing.T, cell string, c rxClient, stream bool, o rxOutcome) {
		var api *llmkit.APIError
		if !errors.As(o.err, &api) {
			return
		}
		if strings.Contains(api.Message, rxTwin[:12]) && !strings.Contains(api.Message, rxTwin) {
			seen[strings.SplitN(cell, "/", 2)[0]+"/"+c.name+"/"+rxMode(stream)] = true
		}
	})
	if t.Failed() {
		return
	}
	for _, kind := range rxPadKinds {
		for _, c := range rxClients {
			for _, stream := range []bool{false, true} {
				if k := kind + "/" + c.name + "/" + rxMode(stream); !seen[k] {
					t.Errorf("%s: no pad leaves the credential straddling byte 200 of the Message", k)
				}
			}
		}
	}
}

// rxSDKType reports whether err's chain still holds the vendor SDK error
// type of the client's family.
func rxSDKType(fam string, err error) bool {
	switch fam {
	case "anthropic":
		var e *sdkanthropic.Error
		return errors.As(err, &e)
	case "openai":
		var e *sdkopenai.Error
		return errors.As(err, &e)
	}
	var e genai.APIError
	return errors.As(err, &e)
}

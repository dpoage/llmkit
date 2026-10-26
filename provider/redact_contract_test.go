package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// rxEcho serves a 401 whose message echoes text, in the family's shape.
func rxEcho(t *testing.T, fam, text string) *httptest.Server {
	t.Helper()
	return rxServe(t, 401, "application/json", rxJSONErr(fam, text, 401))
}

// rxCallWith is rxCall with an explicit Secret, one attempt, and the
// observer's events kept.
func rxCallWith(t *testing.T, c rxClient, stream bool, secret, baseURL string) rxOutcome {
	t.Helper()
	obs := &rxObserver{}
	cl, err := New(context.Background(),
		Spec{Type: c.typ, Auth: c.auth, Model: "m-test", Secret: secret, BaseURL: baseURL},
		Options{Observer: obs, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}, MaxTokens: 16}
	var out rxOutcome
	if stream {
		_, out.err = llmkit.Stream(context.Background(), cl, req, func(llmkit.Delta) error { return nil })
	} else {
		_, out.err = cl.Complete(context.Background(), req)
	}
	out.events = obs.events
	return out
}

func rxVisible(o rxOutcome, s string) bool {
	return rxHolds(strings.Join(rxReachable(o.err), "\n")+strings.Join(o.events, "\n"), []string{s})
}

// TestRedactContract_SecretFloor: a Secret under 8 bytes is a placeholder
// and stays visible when a vendor echoes it, with the error chain intact; a
// Secret of 8 bytes is masked with "***". A BaseURL password is masked at
// any length.
func TestRedactContract_SecretFloor(t *testing.T) {
	const short, long, pass = "ollama7", "ollama78", "zq9"
	for _, c := range rxClients {
		for _, stream := range []bool{false, true} {
			t.Run(c.name+"/"+rxMode(stream), func(t *testing.T) {
				srv := rxEcho(t, c.fam, "bad key "+short+" or "+long)
				o := rxCallWith(t, c, stream, short, srv.URL)
				var api *llmkit.APIError
				if !errors.As(o.err, &api) || !strings.Contains(api.Message, short) || !rxSDKType(c.fam, o.err) {
					t.Errorf("7-byte Secret: err = %v; want the echo visible and the SDK error still in the chain", o.err)
				}
				o = rxCallWith(t, c, stream, long, srv.URL)
				if !errors.As(o.err, &api) || rxVisible(o, long) || !strings.Contains(api.Message, "***") {
					t.Errorf("8-byte Secret: err = %v; want it masked with ***", o.err)
				}
				withPass := strings.Replace(rxEcho(t, c.fam, "bad "+pass+" secret").URL, "http://", "http://user:"+pass+"@", 1)
				o = rxCallWith(t, c, stream, short, withPass)
				if !errors.As(o.err, &api) || rxVisible(o, pass) || !strings.Contains(api.Message, "***") {
					t.Errorf("3-byte BaseURL password: err = %v; want it masked with ***", o.err)
				}
			})
		}
	}
}

// TestRedactContract_MaskingCutsTheChain: when the Redactor changes an
// error, Kind, StatusCode, RetryAfter and Provider stay, errors.Is still
// matches the sentinel (and context.Canceled), and errors.As no longer
// finds the vendor SDK error.
func TestRedactContract_MaskingCutsTheChain(t *testing.T) {
	for _, c := range rxClients {
		for _, stream := range []bool{false, true} {
			t.Run("echo/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
				status, header := 429, http.Header{"Retry-After": {"7"}}
				if c.fam == "google" {
					status, header = 401, nil // genai reports no response headers
				}
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					for k, v := range header {
						w.Header()[k] = v
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(status)
					_, _ = w.Write([]byte(rxJSONErr(c.fam, "bad key "+rxKey, status)))
				}))
				defer srv.Close()
				o := rxCallWith(t, c, stream, rxKey, srv.URL)
				var api *llmkit.APIError
				if !errors.As(o.err, &api) || api.StatusCode != status || api.Provider == "" {
					t.Fatalf("err = %v; want an *llmkit.APIError with status %d", o.err, status)
				}
				want := llmkit.ErrAuth
				if status == 429 {
					want = llmkit.ErrRateLimited
					if !api.HasRetryAfter || api.RetryAfter != 7*time.Second {
						t.Errorf("RetryAfter = %v (%t); want 7s", api.RetryAfter, api.HasRetryAfter)
					}
				}
				if !errors.Is(o.err, want) || api.Kind != want {
					t.Errorf("Kind = %v; want %v", api.Kind, want)
				}
				if rxSDKType(c.fam, o.err) || rxVisible(o, rxKey[:12]) {
					t.Errorf("err = %v; want the SDK error cut from the chain and the key masked", o.err)
				}
			})
		}
	}
	for _, stream := range []bool{false, true} {
		t.Run("cancel/google/"+rxMode(stream), func(t *testing.T) {
			var g rxClient
			for _, c := range rxClients {
				if c.name == "google" {
					g = c
				}
			}
			srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body) // the server notices a closed client only after the body is read
				<-r.Context().Done()
			}))
			defer srv.Close()
			ctx, cancel := context.WithCancel(context.Background())
			time.AfterFunc(100*time.Millisecond, cancel)
			o := rxCall(ctx, t, g, stream, srv.URL+"/"+rxKey)
			var api *llmkit.APIError
			if !errors.Is(o.err, context.Canceled) || errors.As(o.err, &api) || rxVisible(o, rxKey[:12]) {
				t.Errorf("err = %v; want a plain error chaining context.Canceled, key masked", o.err)
			}
		})
	}
}

// TestRedactContract_CallerErrorIsNotMasked: the error a Stream callback
// returns passes through as the caller built it, key text included.
func TestRedactContract_CallerErrorIsNotMasked(t *testing.T) {
	bodies := map[string]string{
		"anthropic": anthropicTextSSE,
		"openai":    `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n",
		"google":    `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}` + "\n\n",
	}
	for _, c := range rxClients {
		t.Run(c.name, func(t *testing.T) {
			srv := rxServe(t, 200, "text/event-stream", bodies[c.fam])
			cl, err := New(context.Background(), Spec{Type: c.typ, Auth: c.auth, Model: "m-test", Secret: rxKey, BaseURL: srv.URL},
				Options{Retry: retry.Config{MaxAttempts: 1}})
			if err != nil {
				t.Fatal(err)
			}
			stop := errors.New("caller stop " + rxKey)
			_, err = llmkit.Stream(context.Background(), cl,
				llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}, MaxTokens: 8},
				func(llmkit.Delta) error { return stop })
			if !errors.Is(err, stop) || !strings.Contains(err.Error(), rxKey) {
				t.Errorf("err = %v; want the callback's own error, text unchanged", err)
			}
		})
	}
}

// TestRedactContract_FormsOfTheSecret: the Secret is masked as sent,
// JSON-escaped (a vendor echo inside a JSON body), and Go-quoted (net/http
// quoting a BaseURL that holds it).
func TestRedactContract_FormsOfTheSecret(t *testing.T) {
	const secret = `ab"c<d>e-0123456789`
	inner, _ := json.Marshal(secret)
	forms := []string{secret, string(inner[1 : len(inner)-1]), strings.Trim(strconv.Quote(secret), `"`)}
	hold := func(o rxOutcome) string {
		for _, f := range forms {
			if rxVisible(o, f) {
				return f
			}
		}
		return ""
	}
	for _, c := range rxClients {
		for _, stream := range []bool{false, true} {
			t.Run("body/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
				o := rxCallWith(t, c, stream, secret, rxEcho(t, c.fam, "bad key "+secret).URL)
				if f := hold(o); f != "" || o.err == nil {
					t.Errorf("err = %v; still holds the form %q", o.err, f)
				}
			})
			t.Run("url/"+c.name+"/"+rxMode(stream), func(t *testing.T) {
				o := rxCallWith(t, c, stream, secret, "http://"+rxClosedPort(t)+"/"+secret)
				if f := hold(o); f != "" || o.err == nil {
					t.Errorf("err = %v; still holds the form %q", o.err, f)
				}
			})
		}
	}
}

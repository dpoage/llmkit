package systemone

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

const (
	wfAccount = "0123456789abcdef0123456789abcdef"
	wfModel   = "clef-flash"
	wfPath    = "/client/v4/accounts/" + wfAccount + "/ai/run/@cf/cloudflare/" + wfModel
)

const wfInner = `{"model":"clef-flash","answers":{"q":{"type":"noul","noul":0.75}},"usage":{"input_tokens":12,"output_tokens":3}}`

// cfServer answers every request with status and body and keeps the last
// request.
type cfServer struct {
	*httptest.Server
	mu     sync.Mutex
	uri    string
	path   string
	method string
	header http.Header
	body   []byte
}

func newCFServer(t *testing.T, status int, body string) *cfServer {
	t.Helper()
	s := &cfServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.uri, s.path, s.method, s.header, s.body = r.RequestURI, r.URL.Path, r.Method, r.Header.Clone(), b
		s.mu.Unlock()
		writeStatus(w, status, body)
	}))
	t.Cleanup(s.Close)
	return s
}

func workersClient(t *testing.T, baseURL string) *Client {
	t.Helper()
	pol := fastRetry
	pol.Rand = func() float64 { return 0.5 }
	c, err := NewWorkersAI(WorkersAIConfig{Secret: testSecret, AccountID: wfAccount, Model: wfModel, BaseURL: baseURL, Retry: pol})
	if err != nil {
		t.Fatalf("NewWorkersAI: %v", err)
	}
	return c
}

func cfSuccess(result string) string {
	return `{"result":` + result + `,"success":true,"errors":[],"messages":[]}`
}

var wfQuestions = decide.Questions{"q": decide.Noul{Instructions: "Is [image 1] a solid red image?"}}

// T1: endpoint, body and auth.
func TestWorkersAI_RequestShape(t *testing.T) {
	left, right := img("image/png", "left-bytes"), img("image/webp", "right-bytes")
	states := map[string]any{
		"two images": map[string]any{"left": left, "right": right},
		"no image":   map[string]any{"a": "b"},
	}
	for name, state := range states {
		t.Run(name, func(t *testing.T) {
			cf := newCFServer(t, http.StatusOK, cfSuccess(wfInner))
			if _, err := workersClient(t, cf.URL).Ask(context.Background(), state, wfQuestions); err != nil {
				t.Fatalf("Ask: %v", err)
			}
			cf.mu.Lock()
			defer cf.mu.Unlock()
			if cf.method != http.MethodPost || cf.path != wfPath || cf.uri != wfPath {
				t.Errorf("request = %s path %q uri %q, want POST %q", cf.method, cf.path, cf.uri, wfPath)
			}
			if got := cf.header.Get("Authorization"); got != "Bearer "+testSecret {
				t.Errorf("Authorization = %q, want Bearer secret", got)
			}
			if got := cf.header.Values(imagesHeader); len(got) != 0 {
				t.Errorf("%s = %v, want absent", imagesHeader, got)
			}

			ref := newWireServer(t)
			arr, err := New(Config{Secret: testSecret, Model: wfModel, BaseURL: ref.URL, Images: ImagesArray})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := arr.Ask(context.Background(), state, wfQuestions); err != nil {
				t.Fatalf("array Ask: %v", err)
			}
			want, _ := ref.last()
			if string(cf.body) != string(want) {
				t.Errorf("body =\n%s\nwant the array-dialect body\n%s", cf.body, want)
			}
		})
	}
}

func TestWorkersAI_BaseURLPathPrefix(t *testing.T) {
	cf := newCFServer(t, http.StatusOK, cfSuccess(wfInner))
	if _, err := workersClient(t, cf.URL+"/gw/").Ask(context.Background(), "s", wfQuestions); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if want := "/gw" + wfPath; cf.path != want {
		t.Errorf("path = %q, want %q", cf.path, want)
	}
}

func TestWorkersAI_DefaultBaseURL(t *testing.T) {
	c, err := NewWorkersAI(WorkersAIConfig{Secret: testSecret, AccountID: wfAccount, Model: wfModel})
	if err != nil {
		t.Fatalf("NewWorkersAI: %v", err)
	}
	if got, want := c.endpoint.String(), "https://api.cloudflare.com"+wfPath; got != want {
		t.Errorf("endpoint = %q, want %q", got, want)
	}
}

func TestWorkersAI_Identity(t *testing.T) {
	c := workersClient(t, "http://127.0.0.1:1")
	if got, want := c.Identity(), (llmkit.Identity{Provider: "systemone", Model: wfModel}); got != want {
		t.Errorf("Identity = %+v, want %+v", got, want)
	}
	if got, want := c.ImageSupport(), ImagesArray.support(); !reflect.DeepEqual(got, want) {
		t.Errorf("ImageSupport = %+v, want the array dialect's %+v", got, want)
	}
}

// T2: response unwrap and error extraction.
func TestWorkersAI_SuccessUnwraps(t *testing.T) {
	cf := newCFServer(t, http.StatusOK, cfSuccess(wfInner))
	got, err := workersClient(t, cf.URL).Ask(context.Background(), "s", wfQuestions)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	want, err := parseResponse(http.StatusOK, []byte(wfInner), wfQuestions)
	if err != nil {
		t.Fatalf("parseResponse: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Response = %+v, want what parseResponse gives the unwrapped result %+v", got, want)
	}
	if got.Model != wfModel || got.Nouls["q"] != 0.75 || got.Usage.InputTokens != 12 {
		t.Errorf("Response = %+v, want the values of result", got)
	}
}

func TestWorkersAI_SuccessEnvelopeFailures(t *testing.T) {
	tests := []struct {
		name, body, want string
	}{
		{"success false", `{"result":` + wfInner + `,"success":false,"errors":[{"code":7003,"message":"no route"}]}`, "code 7003: no route"},
		{"result null", `{"result":null,"success":true,"errors":[]}`, "no result"},
		{"result missing", `{"success":true,"errors":[]}`, "no result"},
		{"result null with errors", `{"result":null,"success":false,"errors":[{"code":1,"message":"a"},{"code":2,"message":"b"}]}`, "code 1: a; code 2: b"},
		{"not JSON", `<html>`, "malformed JSON"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cf := newCFServer(t, http.StatusOK, tt.body)
			_, err := workersClient(t, cf.URL).Ask(context.Background(), "s", wfQuestions)
			var ae *llmkit.APIError
			if !errors.As(err, &ae) || ae.Kind != llmkit.ErrServer || ae.StatusCode != http.StatusOK || ae.Provider != "systemone" {
				t.Fatalf("err = %v, want ErrServer APIError at 200", err)
			}
			if !strings.Contains(ae.Message, tt.want) {
				t.Errorf("Message = %q, want it to contain %q", ae.Message, tt.want)
			}
		})
	}
}

func TestWorkersAI_ErrorEnvelope(t *testing.T) {
	const leak = "token " + testSecret + " is bad"
	tests := []struct {
		name     string
		status   int
		body     string
		kind     error
		contains []string
	}{
		{"auth", 401, `{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}]}`, llmkit.ErrAuth, []string{"code 10000: Authentication error"}},
		{"rate limit", 429, `{"result":null,"success":false,"errors":[{"code":3036,"message":"Rate limit exceeded"}]}`, llmkit.ErrRateLimited, []string{"code 3036: Rate limit exceeded"}},
		{"invalid request", 400, `{"result":null,"success":false,"errors":[{"code":5007,"message":"No such model"}]}`, llmkit.ErrInvalidRequest, []string{"code 5007: No such model"}},
		{"two errors", 400, `{"result":null,"success":false,"errors":[{"code":1,"message":"a"},{"code":2,"message":"b"}]}`, llmkit.ErrInvalidRequest, []string{"code 1: a; code 2: b"}},
		{"secret in message", 401, `{"result":null,"success":false,"errors":[{"code":10000,"message":"` + leak + `"}]}`, llmkit.ErrAuth, []string{"code 10000: token "}},
		{"not an envelope", 400, `upstream said no`, llmkit.ErrInvalidRequest, []string{"upstream said no"}},
		{"envelope without errors", 400, `{"result":null,"success":false,"errors":[]}`, llmkit.ErrInvalidRequest, []string{`"success":false`}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cf := newCFServer(t, tt.status, tt.body)
			_, err := workersClient(t, cf.URL).Ask(context.Background(), "s", wfQuestions)
			var ae *llmkit.APIError
			if !errors.As(err, &ae) {
				t.Fatalf("err = %v, want *llmkit.APIError", err)
			}
			if ae.Kind != tt.kind || ae.StatusCode != tt.status || ae.Provider != "systemone" {
				t.Errorf("err = %+v, want Kind %v status %d provider systemone", ae, tt.kind, tt.status)
			}
			for _, want := range tt.contains {
				if !strings.Contains(ae.Message, want) {
					t.Errorf("Message = %q, want it to contain %q", ae.Message, want)
				}
			}
			if strings.Contains(err.Error(), testSecret) || strings.Contains(ae.Message, testSecret) {
				t.Errorf("error leaks the secret: %v", err)
			}
		})
	}
}

// T3: construction refuses what could escape the path or leak.
func TestNewWorkersAI_Validation(t *testing.T) {
	good := WorkersAIConfig{Secret: testSecret, AccountID: wfAccount, Model: wfModel}
	with := func(mutate func(*WorkersAIConfig)) WorkersAIConfig {
		c := good
		mutate(&c)
		return c
	}
	invalid := map[string]WorkersAIConfig{
		"empty secret":        with(func(c *WorkersAIConfig) { c.Secret = "" }),
		"blank secret":        with(func(c *WorkersAIConfig) { c.Secret = "  " }),
		"leading space":       with(func(c *WorkersAIConfig) { c.Secret = " " + testSecret }),
		"trailing newline":    with(func(c *WorkersAIConfig) { c.Secret = testSecret + "\n" }),
		"empty account":       with(func(c *WorkersAIConfig) { c.AccountID = "" }),
		"traversal account":   with(func(c *WorkersAIConfig) { c.AccountID = "../x" }),
		"uppercase account":   with(func(c *WorkersAIConfig) { c.AccountID = strings.ToUpper(wfAccount) }),
		"31 char account":     with(func(c *WorkersAIConfig) { c.AccountID = wfAccount[:31] }),
		"33 char account":     with(func(c *WorkersAIConfig) { c.AccountID = wfAccount + "0" }),
		"non-hex account":     with(func(c *WorkersAIConfig) { c.AccountID = strings.Repeat("g", 32) }),
		"slash in account":    with(func(c *WorkersAIConfig) { c.AccountID = wfAccount[:16] + "/" + wfAccount[17:] }),
		"newline in account":  with(func(c *WorkersAIConfig) { c.AccountID = wfAccount[:31] + "\n" }),
		"empty model":         with(func(c *WorkersAIConfig) { c.Model = "" }),
		"slash in model":      with(func(c *WorkersAIConfig) { c.Model = "clef/flash" }),
		"traversal model":     with(func(c *WorkersAIConfig) { c.Model = "../clef" }),
		"uppercase model":     with(func(c *WorkersAIConfig) { c.Model = "Clef" }),
		"leading dot model":   with(func(c *WorkersAIConfig) { c.Model = ".clef" }),
		"leading dash model":  with(func(c *WorkersAIConfig) { c.Model = "-clef" }),
		"space in model":      with(func(c *WorkersAIConfig) { c.Model = "clef flash" }),
		"newline in model":    with(func(c *WorkersAIConfig) { c.Model = "clef\n" }),
		"query in model":      with(func(c *WorkersAIConfig) { c.Model = "clef?x=1" }),
		"bad scheme base url": with(func(c *WorkersAIConfig) { c.BaseURL = "ftp://example.com" }),
		"fragment base url":   with(func(c *WorkersAIConfig) { c.BaseURL = "https://example.com#x" }),
		"bad jitter":          with(func(c *WorkersAIConfig) { c.Retry.Jitter = 2 }),
	}
	for name, cfg := range invalid {
		t.Run(name, func(t *testing.T) {
			c, err := NewWorkersAI(cfg)
			if c != nil || !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("NewWorkersAI = (%v, %v), want (nil, ErrInvalidRequest)", c, err)
			}
			for _, secret := range []string{testSecret, strings.TrimSpace(cfg.Secret), cfg.AccountID, wfAccount} {
				if secret != "" && strings.Contains(err.Error(), secret) {
					t.Errorf("error %q echoes %q", err, secret)
				}
			}
		})
	}
	for _, model := range []string{"clef", "clef-flash", "clef-omni", "a.b-c", "0"} {
		t.Run("valid "+model, func(t *testing.T) {
			if _, err := NewWorkersAI(with(func(c *WorkersAIConfig) { c.Model = model })); err != nil {
				t.Fatalf("NewWorkersAI(%q): %v", model, err)
			}
		})
	}
}

package livetest

import (
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

func TestSanitizeHeaders(t *testing.T) {
	h := http.Header{}
	h.Set("Authorization", "Bearer sk-live-key")
	h.Set("X-Api-Key", "sk-live-key")
	h.Set("X-Goog-Api-Key", "goog-key")
	h.Set("Cookie", "session=1")
	h.Set("Set-Cookie", "a=b")
	h.Set("Content-Type", "application/json")
	h.Set("X-Custom", "contains sk-live-key inside")

	got := SanitizeHeaders(h, "sk-live-key")
	for _, name := range []string{"authorization", "x-api-key", "x-goog-api-key", "cookie", "set-cookie", "x-custom"} {
		if _, ok := got[name]; ok {
			t.Errorf("header %q survived sanitization: %v", name, got)
		}
	}
	if got["content-type"] != "application/json" {
		t.Errorf("content-type = %q, want application/json", got["content-type"])
	}

	// The value-contains rule must be case-insensitive on the header NAME
	// only; plain headers survive.
	h2 := http.Header{"x-request-id": []string{"abc"}}
	if got := SanitizeHeaders(h2, ""); len(got) != 1 || got["x-request-id"] != "abc" {
		t.Errorf("SanitizeHeaders(x-request-id) = %v", got)
	}
}

// stubRoundTripper answers every request with the given status and body and
// records what it received.
type stubRoundTripper struct {
	gotPath string
	gotBody string
	status  int
	body    string
}

func (s *stubRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	b, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	s.gotPath = req.URL.Path
	s.gotBody = string(b)
	rec := io.NopCloser(strings.NewReader(s.body))
	return &http.Response{
		StatusCode: s.status,
		Body:       rec,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Request:    req,
	}, nil
}

func TestTransportRecordsExchanges(t *testing.T) {
	stub := &stubRoundTripper{status: 200, body: `{"ok":true}`}
	tr := &Transport{secret: "sk-secret-value-1", base: stub}

	req, _ := http.NewRequest(http.MethodPost, "https://example.invalid/v1/chat/completions?x=1", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Authorization", "Bearer sk-secret-value-1")
	req.Header.Set("Content-Type", "application/json")

	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	// The real trip must still see the body.
	if stub.gotBody != `{"model":"m"}` {
		t.Fatalf("stub saw body %q", stub.gotBody)
	}
	if b, err := io.ReadAll(resp.Body); err != nil || string(b) != `{"ok":true}` {
		t.Fatalf("caller read body %q, %v", b, err)
	}

	ex := tr.Exchanges()
	if len(ex) != 1 {
		t.Fatalf("exchanges = %d, want 1", len(ex))
	}
	if ex[0].Request.Method != http.MethodPost || ex[0].Request.Path != "/v1/chat/completions?x=1" {
		t.Errorf("recorded request = %+v", ex[0].Request)
	}
	if _, ok := ex[0].Request.Headers["authorization"]; ok {
		t.Error("Authorization header recorded")
	}
	if ex[0].Request.Headers["content-type"] != "application/json" {
		t.Errorf("content-type not recorded: %v", ex[0].Request.Headers)
	}
	if ex[0].Response.Status != 200 || ex[0].Response.Body != `{"ok":true}` {
		t.Errorf("recorded response = %+v", ex[0].Response)
	}
}

func TestFixtureRoundTrip(t *testing.T) {
	f := &Fixture{
		Case:  "text_usage",
		Lane:  "compat",
		Model: "m",
		Inputs: []llmkit.Request{{
			Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
			MaxTokens: 16,
		}},
		Exchanges: []Exchange{{
			Request:  WireMessage{Method: "POST", Path: "/v1/chat/completions", Headers: map[string]string{"content-type": "application/json"}, Body: "{}"},
			Response: WireMessage{Headers: map[string]string{"content-type": "application/json"}, Body: "{}", Status: 200},
		}},
		Normalized: Normalized{Text: "OK", StopReason: llmkit.StopEndTurn, Usage: llmkit.Usage{InputTokens: 1, OutputTokens: 1}},
	}
	path := filepath.Join(t.TempDir(), "compat", "text_usage.json")
	if err := writeFixture(path, f, "sk-secret-value-1"); err != nil {
		t.Fatalf("writeFixture: %v", err)
	}
	got, err := ReadFixture(path)
	if err != nil {
		t.Fatalf("ReadFixture: %v", err)
	}
	if got.Case != f.Case || got.Model != f.Model || len(got.Inputs) != 1 || len(got.Exchanges) != 1 {
		t.Fatalf("round-trip mismatch: %+v", got)
	}
	if got.Inputs[0].MaxTokens != 16 || got.Normalized.Text != "OK" {
		t.Fatalf("round-trip mismatch: %+v", got)
	}

	// ErrKind round-trips through the sentinel mapping.
	f.Normalized = Normalized{ErrKind: ErrKindOf(llmkit.ErrAuth)}
	if f.Normalized.ErrKind != "ErrAuth" {
		t.Fatalf("ErrKindOf = %q", f.Normalized.ErrKind)
	}
	if !errors.Is(SentinelByKind(f.Normalized.ErrKind), llmkit.ErrAuth) {
		t.Fatal("SentinelByKind(ErrAuth) does not match llmkit.ErrAuth")
	}
	if SentinelByKind("") != nil || SentinelByKind("bogus") != nil {
		t.Fatal("SentinelByKind: unknown kinds must map to nil")
	}
}

func TestWriteFixtureRefusesSecrets(t *testing.T) {
	f := &Fixture{
		Case: "leaky", Lane: "compat", Model: "m",
		Exchanges:  []Exchange{{Response: WireMessage{Body: "echo sk-secret-value-1 back", Status: 200}}},
		Normalized: Normalized{},
	}
	dir := t.TempDir()

	if err := writeFixture(filepath.Join(dir, "leaky.json"), f, "sk-secret-value-1"); err == nil {
		t.Fatal("writeFixture(secret in body): want refusal, got nil")
	} else if !strings.Contains(err.Error(), "credential") {
		t.Errorf("refusal error = %v, want a credential message", err)
	}

	f.Exchanges[0].Response.Body = "model said: sk-abc12345xyz"
	if err := writeFixture(filepath.Join(dir, "leaky.json"), f, "sk-secret-value-1"); err == nil {
		t.Fatal("writeFixture(sk- substring): want refusal, got nil")
	} else if !strings.Contains(err.Error(), "sk-") {
		t.Errorf("refusal error = %v, want an sk- message", err)
	}

	// Nothing may be written on refusal.
	if _, err := os.Stat(filepath.Join(dir, "leaky.json")); !os.IsNotExist(err) {
		t.Fatalf("refused fixture exists on disk: %v", err)
	}

	// A clean recording writes successfully.
	f.Exchanges[0].Response.Body = "clean"
	if err := writeFixture(filepath.Join(dir, "clean.json"), f, "sk-secret-value-1"); err != nil {
		t.Fatalf("writeFixture(clean): %v", err)
	}
}

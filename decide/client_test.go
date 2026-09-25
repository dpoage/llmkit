package decide

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// testAPIKey is a distinctive fake credential used to prove keys never leak into error strings.
const testAPIKey = "sk-super-secret-do-not-echo"

var fastRetry = retry.Config{
	MaxAttempts: 3,
	BaseDelay:   time.Millisecond,
	MaxDelay:    5 * time.Millisecond,
}

// newTestClient points a client at srv with a jitter-free retry policy.
// A policy without its own Rand gets one pinned to 0.5, a jitter factor of exactly 1.
func newTestClient(t *testing.T, srvURL string, policy retry.Config) *Client {
	t.Helper()
	if policy.Rand == nil {
		policy.Rand = func() float64 { return 0.5 }
	}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srvURL, Retry: policy})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

// capture records one request's bytes and headers under a lock so tests can read them after Ask returns without racing the server goroutine.
type capture struct {
	mu     sync.Mutex
	body   []byte
	auth   string
	ctype  string
	method string
	path   string
}

func (c *capture) record(r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.body, c.auth, c.ctype, c.method, c.path = b, r.Header.Get("Authorization"), r.Header.Get("Content-Type"), r.Method, r.URL.Path
}

func (c *capture) snapshot() (body []byte, auth, ctype, method, path string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.body, c.auth, c.ctype, c.method, c.path
}

func mixedQuestions() Questions {
	return Questions{
		"belief":  Noul{Instructions: "Is the sky blue today?"},
		"pick":    Choice{Instructions: "Pick a color.", Options: map[string]any{"blue": "the sky", "red": "a stop sign"}},
		"quality": Score{Instructions: "Rate the answer.", Levels: []any{"bad", "ok", "good"}},
	}
}

func mixedAnswers(model string) string {
	return fmt.Sprintf(`{"model":%q,"answers":{`+
		`"belief":{"type":"noul","noul":0.9},`+
		`"pick":{"type":"choice","choice":"blue","probabilities":{"blue":0.8,"red":0.15},"confidence":0.7},`+
		`"quality":{"type":"score","score":0.75,"legend":{"0":"bad","1":"ok","2":"good"},"probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.6}`+
		`},"usage":{"input_tokens":120,"output_tokens":15}}`, model)
}

func writeStatus(w http.ResponseWriter, status int, body string) {
	w.WriteHeader(status)
	_, _ = io.WriteString(w, body)
}

// decisionCapture is a concurrency-safe llmkit.Observer collecting
// DecisionEvents.
type decisionCapture struct {
	mu     sync.Mutex
	events []llmkit.Event
}

func (c *decisionCapture) Observe(_ context.Context, ev llmkit.Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
}

func (c *decisionCapture) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.events)
}

func TestAsk_GoldenRequestEnvelope(t *testing.T) {
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.record(r)
		_, _ = io.WriteString(w, `{"model":"jev-1.13.0","answers":{"truth":{"type":"noul","noul":0.9}},"usage":{"input_tokens":10,"output_tokens":2}}`)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	state := map[string]any{"transcript": []string{"user: hi", "agent: hello"}}
	questions := Questions{
		"truth": Noul{Instructions: "Is 2+2 equal to 4?", True: "yes", False: "no"},
	}
	if _, err := c.Ask(context.Background(), state, questions); err != nil {
		t.Fatalf("Ask: %v", err)
	}

	body, auth, ctype, method, path := cap.snapshot()
	want := `{"state":{"transcript":["user: hi","agent: hello"]},"model":"jev-latest","questions":{"truth":{"type":"noul","instructions":"Is 2+2 equal to 4?","criteria":{"true":"yes","false":"no"}}}}`
	if string(body) != want {
		t.Errorf("body =\n%s\nwant\n%s", body, want)
	}
	if auth != "Bearer "+testAPIKey {
		t.Errorf("Authorization = %q, want bearer credential", auth)
	}
	if ctype != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ctype)
	}
	if method != http.MethodPost || path != "/v1/systemone" {
		t.Errorf("request = %s %s, want POST /v1/systemone", method, path)
	}
}

func TestAsk_ResponseNormalized(t *testing.T) {
	srv := httptest.NewServer(okHandler(mixedAnswers("jev-1.13.0")))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	resp, err := c.Ask(context.Background(), "s", mixedQuestions())
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	if resp.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the response's reported model", resp.Model)
	}
	if resp.Usage.InputTokens != 120 || resp.Usage.OutputTokens != 15 {
		t.Errorf("Usage = %+v, want 120 in / 15 out", resp.Usage)
	}
	if got := resp.Nouls["belief"]; got != 0.9 {
		t.Errorf("Nouls[belief] = %v, want 0.9", got)
	}
	choice := resp.Choices["pick"]
	if choice.Choice != "blue" || choice.Confidence != 0.7 {
		t.Errorf("Choices[pick] = %+v, want choice blue with confidence 0.7", choice)
	}
	if choice.Probabilities["blue"] != 0.8 || choice.Probabilities["red"] != 0.15 {
		t.Errorf("Choices[pick].Probabilities = %v, want blue 0.8 / red 0.2", choice.Probabilities)
	}
	score := resp.Scores["quality"]
	if score.Score != 0.75 || score.Confidence != 0.6 {
		t.Errorf("Scores[quality] = %+v, want score 0.75 with confidence 0.6", score)
	}
	if want := []string{"bad", "ok", "good"}; !reflect.DeepEqual(score.Levels, want) {
		t.Errorf("Levels = %v, want %v", score.Levels, want)
	}
	if want := []float64{0.1, 0.5, 0.4}; !reflect.DeepEqual(score.Probabilities, want) {
		t.Errorf("Probabilities = %v, want %v", score.Probabilities, want)
	}
}

// TestAsk_UnaskedKindsAreNil pins response maps stay nil for unasked question kinds.
func TestAsk_UnaskedKindsAreNil(t *testing.T) {
	srv := httptest.NewServer(okHandler(`{"model":"jev-1.13.0","answers":{"b":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":0}}`))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	resp, err := c.Ask(context.Background(), "s", Questions{"b": Noul{Instructions: "i"}})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if resp.Nouls == nil || resp.Nouls["b"] != 0.5 {
		t.Errorf("Nouls = %v, want one entry", resp.Nouls)
	}
	if resp.Choices != nil || resp.Scores != nil {
		t.Errorf("Choices/Scores = %v/%v, want nil (no questions of those kinds)", resp.Choices, resp.Scores)
	}
}

func okHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, body)
	}
}

func TestAsk_StatusMapping(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		wantKind error
	}{
		{"429 rate limited", http.StatusTooManyRequests, `{"error":"slow down"}`, llmkit.ErrRateLimited},
		{"401 unauthorized", http.StatusUnauthorized, `{"error":"bad key"}`, llmkit.ErrAuth},
		{"403 forbidden", http.StatusForbidden, `{"error":"forbidden"}`, llmkit.ErrAuth},
		{"413 too large", http.StatusRequestEntityTooLarge, `{"error":"too large"}`, llmkit.ErrContextTooLong},
		{"400 plain", http.StatusBadRequest, `{"error":"bad field"}`, llmkit.ErrInvalidRequest},
		{"400 context length", http.StatusBadRequest, `{"error":"prompt is too long"}`, llmkit.ErrContextTooLong},
		{"400 context length past byte 200", http.StatusBadRequest, `{"error":"` + strings.Repeat("x", 250) + ` prompt is too long: context length exceeded"}`, llmkit.ErrContextTooLong},
		{"422 validation", 422, `{"error":"state must be a string"}`, llmkit.ErrInvalidRequest},
		{"529 overloaded", 529, `{"error":"overloaded"}`, llmkit.ErrOverloaded},
		{"500 server", http.StatusInternalServerError, `{"error":"boom"}`, llmkit.ErrServer},
		{"503 unavailable", http.StatusServiceUnavailable, `{"error":"unavailable"}`, llmkit.ErrServer},
		{"404 other 4xx", http.StatusNotFound, `{"error":"no route"}`, llmkit.ErrInvalidRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				writeStatus(w, tt.status, tt.body)
			}))
			defer srv.Close()

			c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 1})
			_, err := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
			if err == nil {
				t.Fatal("expected an error")
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %T, want *llmkit.APIError", err)
			}
			if !errors.Is(err, tt.wantKind) {
				t.Errorf("Kind = %v, want %v", apiErr.Kind, tt.wantKind)
			}
			if apiErr.StatusCode != tt.status {
				t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, tt.status)
			}
			if len(apiErr.Message) > 200+len("...") {
				t.Errorf("len(Message) = %d, want at most 200 plus the ellipsis", len(apiErr.Message))
			}
			if apiErr.Provider != "typesafe" {
				t.Errorf("Provider = %q, want typesafe", apiErr.Provider)
			}
			if hits.Load() != 1 {
				t.Errorf("hits = %d, want 1 (MaxAttempts 1)", hits.Load())
			}
		})
	}
}

// TestAsk_RetryAfterCappedAtMaxDelay pins the cap: a one-hour Retry-After must not stall past MaxDelay.
func TestAsk_RetryAfterCappedAtMaxDelay(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.Header().Set("Retry-After", "3600")
			writeStatus(w, http.StatusTooManyRequests, `{"error":"come back in an hour"}`)
			return
		}
		_, _ = io.WriteString(w, mixedAnswers("jev-1.13.0"))
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 2, BaseDelay: time.Hour, MaxDelay: 50 * time.Millisecond, Jitter: 0})
	start := time.Now()
	_, err := c.Ask(context.Background(), "s", mixedQuestions())
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("hits = %d, want 2", got)
	}
	if elapsed > 5*time.Second {
		t.Errorf("elapsed = %v; Retry-After was not capped at MaxDelay", elapsed)
	}
}

func TestAsk_Overloaded529RetriedThenSucceeds(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) <= 2 {
			writeStatus(w, 529, `{"error":"overloaded"}`)
			return
		}
		_, _ = io.WriteString(w, mixedAnswers("jev-1.13.0"))
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	resp, err := c.Ask(context.Background(), "s", mixedQuestions())
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := hits.Load(); got != 3 {
		t.Errorf("hits = %d, want 3 (two 529s retried, then success)", got)
	}
	if resp.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the success attempt's model", resp.Model)
	}
}

func TestAsk_TransportFailureIsErrServer(t *testing.T) {
	srv := httptest.NewServer(okHandler(`{}`))
	url := srv.URL
	srv.Close() // dead endpoint: every dial fails

	c := newTestClient(t, url, retry.Config{MaxAttempts: 1})
	_, err := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	if err == nil {
		t.Fatal("expected a transport error")
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T, want *llmkit.APIError", err)
	}
	if !errors.Is(err, llmkit.ErrServer) {
		t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
	}
	if apiErr.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (transport failure)", apiErr.StatusCode)
	}
	if apiErr.Provider != "typesafe" {
		t.Errorf("Provider = %q, want typesafe", apiErr.Provider)
	}
}

func TestAsk_ParentCancellationTerminal(t *testing.T) {
	var hits atomic.Int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		<-release // hold the attempt open until the test releases the handler
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 5, BaseDelay: time.Millisecond, RequestTimeout: 10 * time.Second})
	_, err := c.Ask(ctx, "s", Questions{"q": Noul{Instructions: "i"}})
	close(release) // unblock the held handler so srv.Close can finish
	if err == nil {
		t.Fatal("expected a cancellation error")
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("hits = %d, want 1 (parent cancellation is terminal, never retried)", got)
	}
	// adapter.TransportError chains the caller's already-Canceled ctx as a
	// plain error (never as an *llmkit.APIError), so Classify's cancellation
	// row makes it terminal before the retry loop asks about a transport
	// ErrServer.
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want it to unwrap to context.Canceled", err)
	}
	var apiErr *llmkit.APIError
	if errors.As(err, &apiErr) {
		t.Errorf("err = %v (%T), want no *llmkit.APIError once the caller cancelled", err, err)
	}
	if !strings.Contains(err.Error(), providerName) {
		t.Errorf("err = %v, want the provider named in the text", err)
	}
}

// TestAsk_PerAttemptTimeoutRetried pins a stalled round-trip is aborted by the per-attempt deadline and retried.
func TestAsk_PerAttemptTimeoutRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 2, BaseDelay: time.Millisecond, Jitter: 0, RequestTimeout: 30 * time.Millisecond})
	_, err := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want it to unwrap to context.DeadlineExceeded", err)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("hits = %d, want 2 (timeouts are retried)", got)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T, want *llmkit.APIError (RequestTimeout under a live parent is not a parent cancellation)", err)
	}
	if !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != 0 {
		t.Errorf("Kind = %v, StatusCode = %d, want ErrServer/0 (adapter.TransportError's shape)", apiErr.Kind, apiErr.StatusCode)
	}
}

// TestAsk_ParentDeadlineDuringBackoffTerminal pins a parent deadline that
// expires while the loop sleeps between attempts (not mid-attempt) is
// terminal: the retryable 500 that triggered the wait never returns as
// an APIError.
func TestAsk_ParentDeadlineDuringBackoffTerminal(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusInternalServerError, `{"error":"boom"}`)
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Millisecond)
	defer cancel()

	c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: 300 * time.Millisecond, MaxDelay: 300 * time.Millisecond, Jitter: 0})
	_, err := c.Ask(ctx, "s", Questions{"q": Noul{Instructions: "i"}})
	if err == nil {
		t.Fatal("expected a deadline error")
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("hits = %d, want 1 (the deadline fires during the backoff sleep, before a second attempt)", got)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want it to unwrap to context.DeadlineExceeded", err)
	}
	var apiErr *llmkit.APIError
	if errors.As(err, &apiErr) {
		t.Errorf("err = %v (%T), want no *llmkit.APIError once the parent deadline ends the loop", err, err)
	}
}

// TestAsk_LegendOrderingFollowsIndex pins conversion sorts by numeric index, not server key order.
func TestAsk_LegendOrderingFollowsIndex(t *testing.T) {
	srv := httptest.NewServer(okHandler(`{"model":"jev-1.13.0","answers":{"quality":{"type":"score","score":0.75,` +
		`"legend":{"2":"good","0":"bad","1":"ok"},"probabilities":{"2":0.1,"0":0.6,"1":0.3},"confidence":0.5}},"usage":{}}`))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	resp, err := c.Ask(context.Background(), "s", Questions{
		"quality": Score{Instructions: "rate", Levels: []any{"bad", "ok", "good"}},
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	score := resp.Scores["quality"]
	if want := []string{"bad", "ok", "good"}; !reflect.DeepEqual(score.Levels, want) {
		t.Errorf("Levels = %v, want %v (index order)", score.Levels, want)
	}
	if want := []float64{0.6, 0.3, 0.1}; !reflect.DeepEqual(score.Probabilities, want) {
		t.Errorf("Probabilities = %v, want %v (index-aligned)", score.Probabilities, want)
	}
}

func TestAsk_ResponseValidation(t *testing.T) {
	base := func(mutate func(answers string) string) string {
		answers := `"belief":{"type":"noul","noul":0.9},` +
			`"pick":{"type":"choice","choice":"blue","probabilities":{"blue":0.8,"red":0.2},"confidence":0.7},` +
			`"quality":{"type":"score","score":0.75,"legend":{"0":"bad","1":"ok","2":"good"},"probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.6}`
		return `{"model":"jev-1.13.0","answers":{` + mutate(answers) + `},"usage":{"input_tokens":1,"output_tokens":1}}`
	}
	tests := []struct {
		name         string
		body         string
		wantInErrMsg string
	}{
		{
			name: "missing answer",
			body: base(func(a string) string {
				return strings.Replace(a, `,"quality":{"type":"score","score":0.75,"legend":{"0":"bad","1":"ok","2":"good"},"probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.6}`, "", 1)
			}),
			wantInErrMsg: `answers["quality"]`,
		},
		{
			name:         "extra answer for unknown id",
			body:         base(func(a string) string { return a + `,"ghost":{"type":"noul","noul":0.5}` }),
			wantInErrMsg: `answers["ghost"]`,
		},
		{
			name: "type mismatch",
			body: base(func(a string) string {
				return strings.Replace(a, `"belief":{"type":"noul","noul":0.9}`, `"belief":{"type":"choice","choice":"x"}`, 1)
			}),
			wantInErrMsg: `answers["belief"].type`,
		},
		{
			name:         "sparse legend",
			body:         `{"model":"m","answers":{"quality":{"type":"score","score":0.5,"legend":{"0":"bad","2":"good"},"probabilities":{"0":0.5,"1":0.5,"2":0.0}}},"usage":{}}`,
			wantInErrMsg: `answers["quality"].legend`,
		},
		{
			name:         "sparse probabilities",
			body:         `{"model":"m","answers":{"quality":{"type":"score","score":0.5,"legend":{"0":"bad","1":"ok","2":"good"},"probabilities":{"0":0.5,"2":0.5}}},"usage":{}}`,
			wantInErrMsg: `answers["quality"].probabilities`,
		},
		{
			name:         "legend size mismatch",
			body:         `{"model":"m","answers":{"quality":{"type":"score","score":0.5,"legend":{"0":"bad","1":"ok"},"probabilities":{"0":0.5,"1":0.5}}},"usage":{}}`,
			wantInErrMsg: `answers["quality"].legend`,
		},
		{
			name:         "missing noul value",
			body:         `{"model":"m","answers":{"b":{"type":"noul"}},"usage":{}}`,
			wantInErrMsg: `answers["b"].noul`,
		},
		{
			name:         "missing choice value",
			body:         `{"model":"m","answers":{"p":{"type":"choice"}},"usage":{}}`,
			wantInErrMsg: `answers["p"].choice`,
		},
		{
			name:         "malformed JSON",
			body:         `{"model":`,
			wantInErrMsg: "malformed JSON",
		},
		{
			name:         "missing model",
			body:         `{"answers":{},"usage":{}}`,
			wantInErrMsg: "model",
		},
		{
			name:         "missing usage",
			body:         `{"model":"m","answers":{}}`,
			wantInErrMsg: "usage",
		},
	}
	questionsFor := func(name string) Questions {
		switch name {
		case "sparse legend", "sparse probabilities", "legend size mismatch":
			return Questions{"quality": Score{Instructions: "rate", Levels: []any{"bad", "ok", "good"}}}
		case "missing noul value":
			return Questions{"b": Noul{Instructions: "i"}}
		case "missing choice value":
			return Questions{"p": Choice{Instructions: "i", Options: map[string]any{"a": "x"}}}
		default:
			return mixedQuestions()
		}
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(okHandler(tt.body))
			defer srv.Close()

			c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 1})
			_, err := c.Ask(context.Background(), "s", questionsFor(tt.name))
			if err == nil {
				t.Fatal("expected a response validation error")
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %T, want *llmkit.APIError", err)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("Kind = %v, want ErrServer (the server violated its own contract)", apiErr.Kind)
			}
			if apiErr.StatusCode != http.StatusOK {
				t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
			}
			if apiErr.Provider != "typesafe" {
				t.Errorf("Provider = %q, want typesafe", apiErr.Provider)
			}
			if !strings.Contains(apiErr.Message, tt.wantInErrMsg) {
				t.Errorf("Message = %q, want it to name %s", apiErr.Message, tt.wantInErrMsg)
			}
		})
	}
}

// TestAsk_ObserverEmitsOneDecisionEventOnSuccess pins the success shape:
// exactly one DecisionEvent per successful Ask, Backend "typesafe", Model
// the response's reported model (not the requested alias), Usage, and
// Answers and Questions sorted by ID.
func TestAsk_ObserverEmitsOneDecisionEventOnSuccess(t *testing.T) {
	srv := httptest.NewServer(okHandler(mixedAnswers("jev-1.13.0")))
	defer srv.Close()

	obs := &decisionCapture{}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srv.URL, Retry: fastRetry, Observer: obs})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := c.Ask(context.Background(), "s", mixedQuestions()); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := obs.count(); got != 1 {
		t.Fatalf("observer events = %d, want exactly 1", got)
	}
	ev := obs.events[0]
	if ev.Kind != llmkit.KindDecision || ev.Decision == nil {
		t.Fatalf("Kind=%q Decision=%+v, want KindDecision with a payload", ev.Kind, ev.Decision)
	}
	de := ev.Decision
	if de.Backend != "typesafe" {
		t.Errorf("Backend = %q, want typesafe", de.Backend)
	}
	// The ledger records the versioned id the server reported, not the alias sent.
	if de.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the response model jev-1.13.0, not the requested alias", de.Model)
	}
	if de.Usage.InputTokens != 120 || de.Usage.OutputTokens != 15 {
		t.Errorf("Usage = %+v, want 120 in / 15 out", de.Usage)
	}
	if de.Err != "" {
		t.Errorf("Err = %q, want empty on success", de.Err)
	}
	if len(de.Answers) != 3 {
		t.Fatalf("Answers = %d, want 3 (belief, pick, quality)", len(de.Answers))
	}
	wantIDs := []string{"belief", "pick", "quality"}
	for i, want := range wantIDs {
		if de.Answers[i].ID != want {
			t.Errorf("Answers[%d].ID = %q, want %q (sorted by ID)", i, de.Answers[i].ID, want)
		}
	}
	if len(de.Questions) != 3 {
		t.Fatalf("Questions = %d, want 3", len(de.Questions))
	}
	for i, want := range wantIDs {
		if de.Questions[i].ID != want {
			t.Errorf("Questions[%d].ID = %q, want %q (sorted by ID)", i, de.Questions[i].ID, want)
		}
	}
}

// TestAsk_ObserverEmitsOnFailure pins the failure shape: a DecisionEvent
// still fires for an Ask that reached the wire and failed. Err is set,
// Answers nil, Model the requested alias (the response never named a
// versioned id). DecisionEvent is not a success-only usage ledger.
func TestAsk_ObserverEmitsOnFailure(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusInternalServerError, `{"error":"boom"}`)
	}))
	defer srv.Close()

	obs := &decisionCapture{}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srv.URL, Retry: retry.Config{MaxAttempts: 2, BaseDelay: time.Millisecond}, Observer: obs})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	askErr := func() error {
		_, err := c.Ask(context.Background(), "s", mixedQuestions())
		return err
	}()
	if askErr == nil {
		t.Fatal("expected an error")
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("hits = %d, want 2", got)
	}
	if got := obs.count(); got != 1 {
		t.Fatalf("observer events = %d, want exactly 1 (a failed Ask that reached the wire still emits)", got)
	}
	de := obs.events[0].Decision
	if de == nil {
		t.Fatal("event carries no Decision payload")
	}
	if de.Err == "" || de.Err != askErr.Error() {
		t.Errorf("Err = %q, want the Ask error text %q", de.Err, askErr.Error())
	}
	if de.Answers != nil {
		t.Errorf("Answers = %+v, want nil on failure", de.Answers)
	}
	if de.Model != "jev-latest" {
		t.Errorf("Model = %q, want the requested alias jev-latest", de.Model)
	}
}

// TestAsk_NoObserverEmitsNothing pins that a nil Config.Observer costs
// nothing: Ask behaves identically with no event ever built.
func TestAsk_NoObserverEmitsNothing(t *testing.T) {
	srv := httptest.NewServer(okHandler(mixedAnswers("jev-1.13.0")))
	defer srv.Close()

	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := c.Ask(context.Background(), "s", mixedQuestions()); err != nil {
		t.Fatalf("Ask: %v", err)
	}
}

// TestAsk_DecisionEventRoundTrip pins that an Ask on a ctx carrying a
// run, a step, and a claimed span emits a DecisionEvent that survives
// encoding/json with every documented field — Backend, Model, State,
// Questions (Instructions, True/False, Options, Levels), Answers
// (Belief, Choice, Probabilities, Confidence, Score, Levels,
// LevelProbabilities), Usage — plus the RunID, Step, and SpanID
// inherited from the Ask's ctx. Twelve questions and eight Asks leave
// an unsorted conversion no realistic chance of matching by
// map-iteration luck.
func TestAsk_DecisionEventRoundTrip(t *testing.T) {
	questions := Questions{
		"zeta":  Noul{Instructions: "Is it raining?", True: "wet streets", False: map[string]any{"streets": "dry"}},
		"alpha": Choice{Instructions: "Pick a color.", Options: map[string]any{"blue": "the sky", "red": map[string]any{"hex": "#f00"}}},
		"mid":   Score{Instructions: map[string]any{"rubric": "quality"}, Levels: []any{"bad", "ok", "good"}},
	}
	extra := []string{"k", "c", "x", "e", "q", "b", "w", "g", "t"}
	for _, id := range extra {
		questions[id] = Noul{Instructions: "extra " + id}
	}
	answers := []string{
		`"zeta":{"type":"noul","noul":0.25}`,
		`"alpha":{"type":"choice","choice":"blue","probabilities":{"blue":0.8,"red":0.2},"confidence":0.7}`,
		`"mid":{"type":"score","score":0.75,"legend":{"0":"bad","1":"ok","2":"good"},"probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.6}`,
	}
	for _, id := range extra {
		answers = append(answers, fmt.Sprintf(`%q:{"type":"noul","noul":0.5}`, id))
	}
	body := `{"model":"jev-1.13.0","answers":{` + strings.Join(answers, ",") + `},"usage":{"input_tokens":120,"output_tokens":15}}`
	srv := httptest.NewServer(okHandler(body))
	defer srv.Close()

	belief := func(v float64) *float64 { return &v }
	raw := func(s string) json.RawMessage { return json.RawMessage(s) }
	want := &llmkit.DecisionEvent{
		Backend: "typesafe",
		Model:   "jev-1.13.0",
		State:   raw(`{"doc":"sky","n":1}`),
		Usage:   llmkit.Usage{InputTokens: 120, OutputTokens: 15},
	}
	wantIDs := append([]string{"alpha", "mid", "zeta"}, extra...)
	sort.Strings(wantIDs)
	for _, id := range wantIDs {
		switch id {
		case "alpha":
			want.Questions = append(want.Questions, llmkit.DecisionQuestion{
				ID: id, Kind: "choice", Instructions: raw(`"Pick a color."`),
				Options: map[string]json.RawMessage{"blue": raw(`"the sky"`), "red": raw(`{"hex":"#f00"}`)},
			})
			want.Answers = append(want.Answers, llmkit.DecisionAnswer{
				ID: id, Choice: "blue", Probabilities: map[string]float64{"blue": 0.8, "red": 0.2}, Confidence: 0.7,
			})
		case "mid":
			want.Questions = append(want.Questions, llmkit.DecisionQuestion{
				ID: id, Kind: "score", Instructions: raw(`{"rubric":"quality"}`),
				Levels: []json.RawMessage{raw(`"bad"`), raw(`"ok"`), raw(`"good"`)},
			})
			want.Answers = append(want.Answers, llmkit.DecisionAnswer{
				ID: id, Score: belief(0.75), Levels: []string{"bad", "ok", "good"},
				LevelProbabilities: []float64{0.1, 0.5, 0.4}, Confidence: 0.6,
			})
		case "zeta":
			want.Questions = append(want.Questions, llmkit.DecisionQuestion{
				ID: id, Kind: "noul", Instructions: raw(`"Is it raining?"`),
				True: raw(`"wet streets"`), False: raw(`{"streets":"dry"}`),
			})
			want.Answers = append(want.Answers, llmkit.DecisionAnswer{ID: id, Belief: belief(0.25)})
		default:
			want.Questions = append(want.Questions, llmkit.DecisionQuestion{
				ID: id, Kind: "noul", Instructions: raw(fmt.Sprintf(`"extra %s"`, id)),
			})
			want.Answers = append(want.Answers, llmkit.DecisionAnswer{ID: id, Belief: belief(0.5)})
		}
	}

	obs := &decisionCapture{}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srv.URL, Retry: fastRetry, Observer: obs})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	runID := llmkit.NewRunID()
	ctx := llmkit.BeginCompletion(llmkit.WithStep(llmkit.WithRun(context.Background(), runID), 7))
	span := llmkit.SpanFromContext(ctx)
	const asks = 8
	for range asks {
		if _, err := c.Ask(ctx, map[string]any{"doc": "sky", "n": 1}, questions); err != nil {
			t.Fatalf("Ask: %v", err)
		}
	}
	if got := obs.count(); got != asks {
		t.Fatalf("observer events = %d, want %d (one per Ask)", got, asks)
	}
	for i, ev := range obs.events {
		wire, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("Ask %d: Marshal: %v", i, err)
		}
		var got llmkit.Event
		if err := json.Unmarshal(wire, &got); err != nil {
			t.Fatalf("Ask %d: Unmarshal: %v", i, err)
		}
		if got.Kind != llmkit.KindDecision || got.RunID != runID || got.Step != 7 || got.SpanID != span {
			t.Errorf("Ask %d: Kind/RunID/Step/SpanID = %q/%q/%d/%q, want decision/%q/7/%q inherited from the Ask ctx",
				i, got.Kind, got.RunID, got.Step, got.SpanID, runID, span)
		}
		if !reflect.DeepEqual(got.Decision, want) {
			gotJSON, _ := json.Marshal(got.Decision)
			wantJSON, _ := json.Marshal(want)
			t.Fatalf("Ask %d: decoded DecisionEvent\n got %s\nwant %s", i, gotJSON, wantJSON)
		}
	}
}

// TestAsk_APIKeyNeverInErrorStrings pins the credential never appears in Error() nor %+v dumps across every error path.
func TestAsk_APIKeyNeverInErrorStrings(t *testing.T) {
	run := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("expected an error")
		}
		for _, s := range []string{err.Error(), fmt.Sprintf("%+v", err)} {
			if strings.Contains(s, testAPIKey) {
				t.Errorf("error string leaks the API key: %s", s)
			}
		}
	}

	// Pre-wire validation (no network).
	c, err := New(Config{APIKey: testAPIKey, Model: "m"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, askErr := c.Ask(context.Background(), nil, Questions{"q": Noul{Instructions: "i"}})
	run(t, askErr)

	// Vendor 401 (a body, but never the credential).
	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, http.StatusUnauthorized, `{"error":"invalid api key"}`)
	}))
	defer srv1.Close()
	c1 := newTestClient(t, srv1.URL, retry.Config{MaxAttempts: 1})
	_, err = c1.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	run(t, err)

	// Transport failure.
	dead := httptest.NewServer(okHandler(`{}`))
	deadURL := dead.URL
	dead.Close()
	c2 := newTestClient(t, deadURL, retry.Config{MaxAttempts: 1})
	_, err = c2.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	run(t, err)

	// Response contract violation.
	srv3 := httptest.NewServer(okHandler(`{"model":"m","answers":{},"usage":{}}`))
	defer srv3.Close()
	c3 := newTestClient(t, srv3.URL, retry.Config{MaxAttempts: 1})
	_, err = c3.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	run(t, err)
}

func TestAsk_ConcurrentAsksRaceClean(t *testing.T) {
	srv := httptest.NewServer(okHandler(mixedAnswers("jev-1.13.0")))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	const n = 12
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = c.Ask(context.Background(), "s", mixedQuestions())
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("ask %d: %v", i, err)
		}
	}
}

// TestAsk_RetryAfterSecondsHonored pins the server delay wins over exponential backoff for every retried status.
func TestAsk_RetryAfterSecondsHonored(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusServiceUnavailable} {
		t.Run(fmt.Sprintf("%d", status), func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if hits.Add(1) == 1 {
					w.Header().Set("Retry-After", "1")
					writeStatus(w, status, `{"error":"slow down"}`)
					return
				}
				_, _ = io.WriteString(w, mixedAnswers("jev-1.13.0"))
			}))
			defer srv.Close()

			c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 2, BaseDelay: time.Hour, MaxDelay: time.Hour, Jitter: 0})
			start := time.Now()
			_, err := c.Ask(context.Background(), "s", mixedQuestions())
			elapsed := time.Since(start)
			if err != nil {
				t.Fatalf("Ask: %v", err)
			}
			if got := hits.Load(); got != 2 {
				t.Errorf("hits = %d, want 2", got)
			}
			if elapsed < 900*time.Millisecond {
				t.Errorf("elapsed = %v; the Retry-After second was not honored", elapsed)
			}
			if elapsed > 10*time.Second {
				t.Errorf("elapsed = %v; exponential backoff ran instead of the Retry-After", elapsed)
			}
		})
	}
}

// TestAsk_RetryAfterImmediateHint pins a Retry-After whose delay clamps to 0 (literal zero or past HTTP-date) means an immediate retry.
func TestAsk_RetryAfterImmediateHint(t *testing.T) {
	pastDate := time.Now().Add(-time.Minute).UTC().Format(http.TimeFormat)
	tests := []struct {
		name       string
		retryAfter string
	}{
		{"zero delay", "0"},
		{"past HTTP-date", pastDate},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if hits.Add(1) == 1 {
					w.Header().Set("Retry-After", tt.retryAfter)
					writeStatus(w, http.StatusTooManyRequests, `{"error":"slow down"}`)
					return
				}
				_, _ = io.WriteString(w, mixedAnswers("jev-1.13.0"))
			}))
			defer srv.Close()

			// BaseDelay 500ms: any honored-as-written or exponential sleep would blow the 400ms budget.
			c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 2, BaseDelay: 500 * time.Millisecond, MaxDelay: 30 * time.Second, Jitter: 0})
			start := time.Now()
			_, err := c.Ask(context.Background(), "s", mixedQuestions())
			elapsed := time.Since(start)
			if err != nil {
				t.Fatalf("Ask: %v", err)
			}
			if got := hits.Load(); got != 2 {
				t.Errorf("hits = %d, want 2", got)
			}
			if elapsed >= 400*time.Millisecond {
				t.Errorf("elapsed = %v; a zero Retry-After must retry immediately", elapsed)
			}
		})
	}
}

// TestAsk_RetryBoundary pins auth and validation statuses end the loop after one attempt — a deterministic client-error status must not burn the retry budget.
func TestAsk_RetryBoundary(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
	}{
		{"401 auth", http.StatusUnauthorized, `{"error":"bad key"}`},
		{"422 validation", 422, `{"error":"state must be a string"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				writeStatus(w, tt.status, tt.body)
			}))
			defer srv.Close()

			c := newTestClient(t, srv.URL, retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond})
			_, err := c.Ask(context.Background(), "s", mixedQuestions())
			if err == nil {
				t.Fatal("expected an error")
			}
			if got := hits.Load(); got != 1 {
				t.Errorf("hits = %d, want 1 (terminal errors never retry)", got)
			}
		})
	}
}

// TestAsk_AnswerContractViolationRetried pins a 200 response that violates the answer contract is retried like any other ErrServer.
func TestAsk_AnswerContractViolationRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusOK, `{"model":"m","answers":{},"usage":{}}`)
	}))
	defer srv.Close()

	var sleeps []time.Duration
	policy := retry.Config{
		MaxAttempts: 3,
		BaseDelay:   time.Millisecond,
		MaxDelay:    5 * time.Millisecond,
		Sleep: func(_ context.Context, d time.Duration) error {
			sleeps = append(sleeps, d)
			return nil
		},
	}
	c := newTestClient(t, srv.URL, policy)
	_, err := c.Ask(context.Background(), "s", mixedQuestions())
	if err == nil {
		t.Fatal("expected an error")
	}
	if got := hits.Load(); got != 3 {
		t.Errorf("hits = %d, want 3 (R4: a 200 contract violation retries like any ErrServer)", got)
	}
	if len(sleeps) != 2 {
		t.Errorf("sleeps = %v, want 2 waits between 3 attempts", sleeps)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T, want *llmkit.APIError", err)
	}
	if !errors.Is(err, llmkit.ErrServer) {
		t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
	}
	if apiErr.StatusCode != http.StatusOK {
		t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
	}
}

// TestAsk_SubBadRequestStatusIsErrServerRetried pins a sub-400 status with no vendor Type falls back to ErrServer and is retried.
func TestAsk_SubBadRequestStatusIsErrServerRetried(t *testing.T) {
	for _, status := range []int{http.StatusFound, http.StatusNoContent} {
		t.Run(fmt.Sprintf("%d", status), func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.WriteHeader(status)
			}))
			defer srv.Close()

			c := newTestClient(t, srv.URL, fastRetry)
			_, err := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
			if err == nil {
				t.Fatal("expected an error")
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %T, want *llmkit.APIError", err)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
			}
			if apiErr.StatusCode != status {
				t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, status)
			}
			if got := hits.Load(); got != int32(fastRetry.MaxAttempts) {
				t.Errorf("hits = %d, want %d (retried like any ErrServer)", got, fastRetry.MaxAttempts)
			}
		})
	}
}

// TestAsk_MalformedBaseURLIsErrInvalidRequest pins a BaseURL that cannot become a valid request is a deterministic client-side defect: ErrInvalidRequest, terminal, no retry.
func TestAsk_MalformedBaseURLIsErrInvalidRequest(t *testing.T) {
	var sleeps []time.Duration
	policy := retry.Config{
		MaxAttempts: 3,
		BaseDelay:   time.Millisecond,
		Sleep: func(_ context.Context, d time.Duration) error {
			sleeps = append(sleeps, d)
			return nil
		},
	}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: "http://a b", Retry: policy})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, askErr := c.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	if askErr == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(askErr, llmkit.ErrInvalidRequest) {
		t.Errorf("err = %v, want ErrInvalidRequest", askErr)
	}
	var apiErr *llmkit.APIError
	if !errors.As(askErr, &apiErr) {
		t.Fatalf("err = %T, want *llmkit.APIError", askErr)
	}
	if apiErr.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
	}
	if len(sleeps) != 0 {
		t.Errorf("sleeps = %v, want none: a malformed BaseURL must not retry", sleeps)
	}
}

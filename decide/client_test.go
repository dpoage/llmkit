package decide

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// testAPIKey is a distinctive fake credential used to prove keys never leak
// into error strings.
const testAPIKey = "sk-super-secret-do-not-echo"

var fastRetry = llmkit.RetryConfig{
	MaxAttempts: 3,
	BaseDelay:   time.Millisecond,
	MaxDelay:    5 * time.Millisecond,
}

// newTestClient points a client at srv with a jitter-free retry policy.
func newTestClient(t *testing.T, srvURL string, retry llmkit.RetryConfig) *Client {
	t.Helper()
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srvURL, Retry: retry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

// capture records one request's bytes and headers under a lock, so tests can
// read them after Ask returns without racing the server goroutine.
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

// mixedAnswers returns a valid 200 body for mixedQuestions.
func mixedAnswers(model string) string {
	return fmt.Sprintf(`{"model":%q,"answers":{`+
		`"belief":{"type":"noul","noul":0.9},`+
		`"pick":{"type":"choice","choice":"blue","probabilities":{"blue":0.8,"red":0.2},"confidence":0.7},`+
		`"quality":{"type":"score","score":0.75,"legend":{"0":"bad","1":"ok","2":"good"},"probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.6}`+
		`},"usage":{"input_tokens":120,"output_tokens":15}}`, model)
}

func writeStatus(w http.ResponseWriter, status int, body string) {
	w.WriteHeader(status)
	_, _ = io.WriteString(w, body)
}

// recorder is a concurrency-safe llmkit.Recorder collecting usage events.
type recorder struct {
	mu     sync.Mutex
	events []llmkit.UsageEvent
}

func (r *recorder) Record(ev llmkit.UsageEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, ev)
}

func (r *recorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.events)
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
	if choice.Probabilities["blue"] != 0.8 || choice.Probabilities["red"] != 0.2 {
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

// TestAsk_UnaskedKindsAreNil pins that response maps stay nil for question
// kinds the caller did not ask.
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

			c := newTestClient(t, srv.URL, llmkit.RetryConfig{MaxAttempts: 1})
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
			if apiErr.Provider != "typesafe" {
				t.Errorf("Provider = %q, want typesafe", apiErr.Provider)
			}
			if hits.Load() != 1 {
				t.Errorf("hits = %d, want 1 (MaxAttempts 1)", hits.Load())
			}
		})
	}
}

// TestAsk_RetryAfterSecondsHonored proves the server delay wins over
// exponential backoff: BaseDelay is an hour, so only the one-second
// Retry-After can let the test finish this fast.
func TestAsk_RetryAfterSecondsHonored(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			writeStatus(w, http.StatusTooManyRequests, `{"error":"slow down"}`)
			return
		}
		_, _ = io.WriteString(w, mixedAnswers("jev-1.13.0"))
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, llmkit.RetryConfig{MaxAttempts: 2, BaseDelay: time.Hour, MaxDelay: time.Hour, Jitter: 0})
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
}

// TestAsk_RetryAfterCappedAtMaxDelay proves the cap: Retry-After is an hour,
// MaxDelay 50ms, so an uncapped sleep would stall the test for an hour.
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

	c := newTestClient(t, srv.URL, llmkit.RetryConfig{MaxAttempts: 2, BaseDelay: time.Hour, MaxDelay: 50 * time.Millisecond, Jitter: 0})
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

	c := newTestClient(t, url, llmkit.RetryConfig{MaxAttempts: 1})
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

	c := newTestClient(t, srv.URL, llmkit.RetryConfig{MaxAttempts: 5, BaseDelay: time.Millisecond, RequestTimeout: 10 * time.Second})
	_, err := c.Ask(ctx, "s", Questions{"q": Noul{Instructions: "i"}})
	close(release) // unblock the held handler so srv.Close can finish
	if err == nil {
		t.Fatal("expected a cancellation error")
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("hits = %d, want 1 (parent cancellation is terminal, never retried)", got)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want it to unwrap to context.Canceled", err)
	}
	if !errors.Is(err, llmkit.ErrServer) {
		t.Errorf("err = %v, want it to unwrap to ErrServer (transport failure)", err)
	}
}

// TestAsk_PerAttemptTimeoutRetried mirrors the embed hardening guarantee: a
// stalled round-trip is aborted by the per-attempt deadline and retried.
func TestAsk_PerAttemptTimeoutRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, llmkit.RetryConfig{MaxAttempts: 2, BaseDelay: time.Millisecond, Jitter: 0, RequestTimeout: 30 * time.Millisecond})
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
}

// TestAsk_LegendOrderingFollowsIndex serves legend and probability keys out
// of order to prove conversion sorts by numeric index, not server key order.
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

			c := newTestClient(t, srv.URL, llmkit.RetryConfig{MaxAttempts: 1})
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

func TestAsk_RecorderFiresOnceWithResponseModel(t *testing.T) {
	srv := httptest.NewServer(okHandler(mixedAnswers("jev-1.13.0")))
	defer srv.Close()

	rec := &recorder{}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srv.URL, Retry: fastRetry, Recorder: rec})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := c.Ask(context.Background(), "s", mixedQuestions()); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := rec.count(); got != 1 {
		t.Fatalf("recorder events = %d, want exactly 1", got)
	}
	ev := rec.events[0]
	if ev.Provider != "typesafe" {
		t.Errorf("Provider = %q, want typesafe", ev.Provider)
	}
	// The alias sent was jev-latest; the ledger must show the versioned id
	// the server reported.
	if ev.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the response model jev-1.13.0, not the requested alias", ev.Model)
	}
	if ev.Usage.InputTokens != 120 || ev.Usage.OutputTokens != 15 {
		t.Errorf("Usage = %+v, want 120 in / 15 out", ev.Usage)
	}
}

func TestAsk_RecorderSilentOnFailure(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusInternalServerError, `{"error":"boom"}`)
	}))
	defer srv.Close()

	rec := &recorder{}
	c, err := New(Config{APIKey: testAPIKey, Model: "jev-latest", BaseURL: srv.URL, Retry: llmkit.RetryConfig{MaxAttempts: 2, BaseDelay: time.Millisecond}, Recorder: rec})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := c.Ask(context.Background(), "s", mixedQuestions()); err == nil {
		t.Fatal("expected an error")
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("hits = %d, want 2", got)
	}
	if got := rec.count(); got != 0 {
		t.Errorf("recorder events = %d, want 0 (failures record nothing)", got)
	}
}

// TestAsk_APIKeyNeverInErrorStrings runs every error path with a distinctive
// key and asserts the key appears in neither Error() nor the %+v dump.
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
	c1 := newTestClient(t, srv1.URL, llmkit.RetryConfig{MaxAttempts: 1})
	_, err = c1.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	run(t, err)

	// Transport failure.
	dead := httptest.NewServer(okHandler(`{}`))
	deadURL := dead.URL
	dead.Close()
	c2 := newTestClient(t, deadURL, llmkit.RetryConfig{MaxAttempts: 1})
	_, err = c2.Ask(context.Background(), "s", Questions{"q": Noul{Instructions: "i"}})
	run(t, err)

	// Response contract violation.
	srv3 := httptest.NewServer(okHandler(`{"model":"m","answers":{},"usage":{}}`))
	defer srv3.Close()
	c3 := newTestClient(t, srv3.URL, llmkit.RetryConfig{MaxAttempts: 1})
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

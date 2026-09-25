package provider

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// completeExpectingError drives one Complete against a server answering with
// (status, body) and returns the normalized error.
func completeExpectingError(t *testing.T, f adapterFactory, status int, body string, header http.Header) error {
	t.Helper()
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		for k, vs := range header {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	})
	client := f.build(t, base)
	_, err := client.Complete(context.Background(), simpleRequest())
	if err == nil {
		t.Fatalf("%s: expected error for status %d, got nil", f.name, status)
	}
	return err
}

// mustBeAPIError narrows err to *llmkit.APIError and checks its Kind and
// StatusCode.
func mustBeAPIError(t *testing.T, err error, f adapterFactory, kind error, status int) *llmkit.APIError {
	t.Helper()
	if !errors.Is(err, kind) {
		t.Errorf("%s: error = %v, want %v", f.name, err, kind)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("%s: error is not *llmkit.APIError: %T", f.name, err)
	}
	if apiErr.StatusCode != status {
		t.Errorf("%s: StatusCode = %d, want %d", f.name, apiErr.StatusCode, status)
	}
	return apiErr
}

// TestConformance_ErrorNormalization_ContextTooLong413 asserts a 413 maps to
// ErrContextTooLong on every adapter, with the status preserved on the
// APIError.
func TestConformance_ErrorNormalization_ContextTooLong413(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, http.StatusRequestEntityTooLarge,
				errorBody(f.name, http.StatusRequestEntityTooLarge, "request too large"), nil)
			_ = mustBeAPIError(t, err, f, llmkit.ErrContextTooLong, http.StatusRequestEntityTooLarge)
		})
	}
}

// TestConformance_ErrorNormalization_ContextTooLong400Heuristic asserts the
// 400 disambiguation: a message that looks like context-window overflow
// promotes the error to ErrContextTooLong, while an ordinary 400 stays
// ErrInvalidRequest. The control case is what makes the heuristic
// testable — without it, a classifier that maps every 400 to
// ErrContextTooLong would pass.
func TestConformance_ErrorNormalization_ContextTooLong400Heuristic(t *testing.T) {
	tooLongMsg := "This model's maximum context length is 128000 tokens. However, your messages resulted in 200000 tokens. Please reduce the length of the messages."
	cases := []struct {
		name    string
		message string
		kind    error
	}{
		{"heuristic hit", tooLongMsg, llmkit.ErrContextTooLong},
		{"ordinary 400 stays invalid request", "Invalid parameter: temperature must be between 0 and 2", llmkit.ErrInvalidRequest},
	}
	for _, f := range allAdapters() {
		for _, tc := range cases {
			t.Run(f.name+"/"+tc.name, func(t *testing.T) {
				err := completeExpectingError(t, f, http.StatusBadRequest,
					errorBody(f.name, http.StatusBadRequest, tc.message), nil)
				_ = mustBeAPIError(t, err, f, tc.kind, http.StatusBadRequest)
			})
		}
	}
}

// TestConformance_ErrorNormalization_Server500 asserts a plain 500 maps to
// ErrServer (retryable) on every adapter.
func TestConformance_ErrorNormalization_Server500(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, http.StatusInternalServerError,
				errorBody(f.name, http.StatusInternalServerError, "internal error"), nil)
			_ = mustBeAPIError(t, err, f, llmkit.ErrServer, http.StatusInternalServerError)
		})
	}
}

// TestConformance_ErrorNormalization_Overloaded529 asserts Anthropic's
// non-standard 529 (overloaded) maps to ErrOverloaded on every adapter.
func TestConformance_ErrorNormalization_Overloaded529(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, 529,
				errorBody(f.name, 529, "overloaded"), nil)
			_ = mustBeAPIError(t, err, f, llmkit.ErrOverloaded, 529)
		})
	}
}

// TestConformance_RetryAfterHTTPDate asserts the HTTP-date form of
// Retry-After is parsed at the adapter boundary (the seconds form is covered
// by TestConformance_ErrorNormalization_RateLimit). genai's APIError carries
// no *http.Response, so Google reports RetryAfter 0 by documented contract.
func TestConformance_RetryAfterHTTPDate(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			// A date ~90s out: strictly in the future (so a correct parser
			// yields a positive duration) but far from any test timeout.
			when := time.Now().UTC().Add(90 * time.Second)
			h := http.Header{}
			h.Set("Retry-After", when.Format(http.TimeFormat))
			err := completeExpectingError(t, f, http.StatusTooManyRequests,
				errorBody(f.name, http.StatusTooManyRequests, "slow down"), h)

			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("%s: error is not *llmkit.APIError: %T", f.name, err)
			}
			if f.name == "google" {
				if apiErr.RetryAfter != 0 {
					t.Errorf("google RetryAfter = %v, want 0 (genai exposes no http.Response)", apiErr.RetryAfter)
				}
				return
			}
			if apiErr.RetryAfter <= 0 {
				t.Errorf("%s RetryAfter = %v, want >0 (HTTP-date %q should parse)", f.name, apiErr.RetryAfter, when.Format(http.TimeFormat))
			}
			if apiErr.RetryAfter > 2*time.Minute {
				t.Errorf("%s RetryAfter = %v, want ~90s", f.name, apiErr.RetryAfter)
			}
		})
	}
}

// TestConformance_CallerCancellationIsNotAPIError pins a caller's already-cancelled context surfaces on every adapter, Complete and Stream, as a plain error chaining context.Canceled — never an *llmkit.APIError, terminal under llmkit.Classify. A closed server yields *APIError{ErrServer, StatusCode 0}, retryable, no Canceled.
func TestConformance_CallerCancellationIsNotAPIError(t *testing.T) {
	closed := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedURL := closed.URL
	closed.Close() // nothing listens: every dial fails

	for _, f := range allAdapters() {
		for _, mode := range []string{"complete", "stream"} {
			t.Run(f.name+"/"+mode, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel() // already cancelled before the call
				client := f.build(t, newServer(t, func(w http.ResponseWriter, r *http.Request) {
					<-r.Context().Done() // never answers a live call
				}))
				var err error
				if mode == "complete" {
					_, err = client.Complete(ctx, simpleRequest())
				} else {
					_, err = llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error { return nil })
				}
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !errors.Is(err, context.Canceled) {
					t.Errorf("err = %v, want context.Canceled in the chain", err)
				}
				var apiErr *llmkit.APIError
				if errors.As(err, &apiErr) {
					t.Errorf("err = %v (%T), want no *llmkit.APIError for a caller cancellation", err, err)
				}
				if _, _, retryable := llmkit.Classify(err); retryable {
					t.Errorf("Classify(err) = retryable, want terminal for a caller cancellation")
				}
			})

			t.Run(f.name+"/"+mode+"/dial-failure", func(t *testing.T) {
				client := f.build(t, closedURL)
				var err error
				if mode == "complete" {
					_, err = client.Complete(context.Background(), simpleRequest())
				} else {
					_, err = llmkit.Stream(context.Background(), client, simpleRequest(), func(llmkit.Delta) error { return nil })
				}
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("err = %v (%T), want *llmkit.APIError for a dial failure", err, err)
				}
				if !errors.Is(err, llmkit.ErrServer) {
					t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
				}
				if apiErr.StatusCode != 0 {
					t.Errorf("StatusCode = %d, want 0 (no response received)", apiErr.StatusCode)
				}
				if errors.Is(err, context.Canceled) {
					t.Errorf("err = %v, want no Canceled in a dial failure's chain", err)
				}
				if _, _, retryable := llmkit.Classify(err); !retryable {
					t.Errorf("Classify(err) = terminal, want retryable for a transport failure")
				}
			})
		}
	}
}

// TestConformance_PerAttemptTimeoutRetriesTwice pins a stalled server under WithRetry{MaxAttempts: 2, RequestTimeout: 50ms} is hit exactly twice and the final error is *APIError{ErrServer, StatusCode 0} with context.DeadlineExceeded reachable through Err.
func TestConformance_PerAttemptTimeoutRetriesTwice(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			var hits atomic.Int32
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				_, _ = io.Copy(io.Discard, r.Body) // drain: the client's close must reach the handler
				<-r.Context().Done()               // stall until the attempt deadline reaps us
			})
			var slept []time.Duration
			cfg := llmkitRetryCfg(func(_ context.Context, d time.Duration) error {
				slept = append(slept, d)
				return nil
			})
			cfg.RequestTimeout = 50 * time.Millisecond
			client := llmkit.WithRetry(f.build(t, base), cfg)
			_, err := client.Complete(context.Background(), simpleRequest())
			if err == nil {
				t.Fatal("expected the final timeout error, got nil")
			}
			if hits.Load() != 2 {
				t.Errorf("wire hits = %d, want 2 (timed-out attempt retried once)", hits.Load())
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
			}
			if !errors.Is(err, llmkit.ErrServer) {
				t.Errorf("Kind = %v, want ErrServer", apiErr.Kind)
			}
			if apiErr.StatusCode != 0 {
				t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("err = %v, want context.DeadlineExceeded through Err", err)
			}
			_ = slept
		})
	}
}

// TestConformance_CtxCancelledDuringBackoffSleep pins a caller's ctx that ends while the retry loop sleeps after a retryable failure surfaces on every adapter, Complete and Stream, as a plain error chaining context.Canceled carrying the last attempt's error as text: the retryable *llmkit.APIError is replaced, terminal under llmkit.Classify.
func TestConformance_CtxCancelledDuringBackoffSleep(t *testing.T) {
	for _, f := range allAdapters() {
		for _, mode := range []string{"complete", "stream"} {
			t.Run(f.name+"/"+mode, func(t *testing.T) {
				var hits atomic.Int32
				base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
					hits.Add(1)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(errorBody(f.name, http.StatusServiceUnavailable, "overloaded")))
				})
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				cfg := llmkitRetryCfg(func(ctx context.Context, d time.Duration) error {
					cancel() // the caller's ctx ends during the backoff sleep
					return ctx.Err()
				})
				client := llmkit.WithRetry(f.build(t, base), cfg)
				var err error
				if mode == "complete" {
					_, err = client.Complete(ctx, simpleRequest())
				} else {
					_, err = llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error { return nil })
				}
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !errors.Is(err, context.Canceled) {
					t.Errorf("err = %v, want context.Canceled in the chain", err)
				}
				var apiErr *llmkit.APIError
				if errors.As(err, &apiErr) {
					t.Errorf("err = %v (%T), want no *llmkit.APIError on a cancelled path", err, err)
				}
				if _, _, retryable := llmkit.Classify(err); retryable {
					t.Errorf("Classify(err) = retryable, want terminal for a cancelled path")
				}
				if !strings.Contains(err.Error(), "503") {
					t.Errorf("err = %v, want the last attempt's message in the text", err)
				}
				if got := hits.Load(); got != 1 {
					t.Errorf("wire hits = %d, want 1 (a cancelled sleep ends the loop)", got)
				}
			})
		}
	}
}

// newTarget is one adapter as provider.New builds it; name matches the
// allAdapters and errorBody names.
type newTarget struct {
	name  string
	typ   Type
	model string
}

func newTargets() []newTarget {
	return []newTarget{
		{"anthropic", TypeAnthropic, "claude-test"},
		{"openai", TypeOpenAI, "gpt-test"},
		{"openai-compatible", TypeOpenAICompatible, "llama-test"},
		{"google", TypeGoogle, "gemini-test"},
	}
}

// build runs provider.New for the target against base.
func (n newTarget) build(t *testing.T, base string, opts Options) llmkit.Client {
	t.Helper()
	client, err := New(context.Background(), Spec{Type: n.typ, Model: n.model, BaseURL: base, Secret: "test-key"}, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return client
}

// callMode runs one Complete or one Stream (no-op callback) on client.
func callMode(ctx context.Context, client llmkit.Client, mode string) error {
	var err error
	if mode == "complete" {
		_, err = client.Complete(ctx, simpleRequest())
	} else {
		_, err = llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error { return nil })
	}
	return err
}

// TestNew_CtxCancelledDuringBackoffSleep pins a completed Options.Retry{BaseDelay: 400ms} sleeps a jittered [320ms, 480ms] after a retryable 503; a ctx cancelled at 200ms cuts that sleep, and the retryable *APIError is replaced by a plain cancellation error carrying its text — Complete and Stream.
func TestNew_CtxCancelledDuringBackoffSleep(t *testing.T) {
	for _, n := range newTargets() {
		for _, mode := range []string{"complete", "stream"} {
			t.Run(n.name+"/"+mode, func(t *testing.T) {
				var hits atomic.Int32
				base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
					hits.Add(1)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(errorBody(n.name, http.StatusServiceUnavailable, "overloaded")))
				})
				client := n.build(t, base, Options{Retry: retry.Config{BaseDelay: 400 * time.Millisecond}})
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				time.AfterFunc(200*time.Millisecond, cancel)
				err := callMode(ctx, client, mode)
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !errors.Is(err, context.Canceled) {
					t.Errorf("err = %v, want context.Canceled in the chain", err)
				}
				var apiErr *llmkit.APIError
				if errors.As(err, &apiErr) {
					t.Errorf("err = %v (%T), want no *llmkit.APIError on a cancelled path", err, err)
				}
				if _, _, retryable := llmkit.Classify(err); retryable {
					t.Errorf("Classify(err) = retryable, want terminal for a cancelled path")
				}
				if !strings.Contains(err.Error(), "503") {
					t.Errorf("err = %v, want the last attempt's message in the text", err)
				}
				if got := hits.Load(); got != 1 {
					t.Errorf("wire hits = %d, want 1 (the cancelled sleep ends the loop)", got)
				}
			})
		}
	}
}

// TestNew_ParentEndsMidAttempt pins a caller's ctx that ends while an attempt stalls: a deadline makes the adapter report a retryable *APIError{ErrServer, 0} chaining DeadlineExceeded which the retry stage replaces with a plain error chaining DeadlineExceeded carrying the adapter's text; a cancellation makes the adapter report its own plain terminal error chaining context.Canceled, returned as-is. Either way: no *APIError, terminal, one wire hit.
func TestNew_ParentEndsMidAttempt(t *testing.T) {
	for _, n := range newTargets() {
		for _, mode := range []string{"complete", "stream"} {
			for _, parent := range []string{"deadline", "cancel"} {
				t.Run(n.name+"/"+mode+"/"+parent, func(t *testing.T) {
					var hits atomic.Int32
					base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
						hits.Add(1)
						_, _ = io.Copy(io.Discard, r.Body) // drain: the client's close must reach the handler
						<-r.Context().Done()               // stall until the caller's ctx reaps us
					})
					client := n.build(t, base, Options{})
					var ctx context.Context
					var cancel context.CancelFunc
					want := context.DeadlineExceeded
					if parent == "deadline" {
						ctx, cancel = context.WithTimeout(context.Background(), 150*time.Millisecond)
					} else {
						ctx, cancel = context.WithCancel(context.Background())
						time.AfterFunc(150*time.Millisecond, cancel)
						want = context.Canceled
					}
					defer cancel()
					err := callMode(ctx, client, mode)
					if err == nil {
						t.Fatal("expected error, got nil")
					}
					if !errors.Is(err, want) {
						t.Errorf("err = %v, want %v in the chain", err, want)
					}
					var apiErr *llmkit.APIError
					if errors.As(err, &apiErr) {
						t.Errorf("err = %v (%T), want no *llmkit.APIError once the caller's ctx ended", err, err)
					}
					if _, _, retryable := llmkit.Classify(err); retryable {
						t.Errorf("Classify(err) = retryable, want terminal once the caller's ctx ended")
					}
					if parent == "deadline" && !strings.Contains(err.Error(), "(last attempt: llmkit: "+n.name+" error: ") {
						t.Errorf("err = %v, want the adapter's error in the text", err)
					}
					if parent == "cancel" && strings.Count(err.Error(), "context canceled") != 1 {
						t.Errorf("err = %v, want the adapter's own cancellation error, naming it exactly once", err)
					}
					if got := hits.Load(); got != 1 {
						t.Errorf("wire hits = %d, want 1 (an ended caller never retries)", got)
					}
				})
			}
		}
	}
}

// firstDeltaChunk is one stream event that delivers the text delta "ok" in
// the target's wire format, with no terminal event after it.
func firstDeltaChunk(name string) string {
	switch name {
	case "anthropic":
		return "event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
			"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
			"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}` + "\n\n"
	case "google":
		return "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}` + "\n\n"
	default:
		return "data: " + `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}` + "\n\n"
	}
}

// TestNew_ParentDeadlineAfterDelta pins a stream that delivered one delta then stalls until the caller's deadline: the adapter reports a retryable *APIError{ErrServer, 0} chaining DeadlineExceeded; the delivered delta makes the stage's verdict terminal only while the caller's ctx is live, so once it is done the error is replaced like any other retryable failure — plain error chaining DeadlineExceeded, no *APIError, terminal, one hit.
func TestNew_ParentDeadlineAfterDelta(t *testing.T) {
	for _, n := range newTargets() {
		t.Run(n.name, func(t *testing.T) {
			var hits atomic.Int32
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				_, _ = io.WriteString(w, firstDeltaChunk(n.name))
				w.(http.Flusher).Flush()
				<-r.Context().Done() // stall after the delta until the caller's deadline
			})
			client := n.build(t, base, Options{})
			ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
			defer cancel()
			deltas := 0
			_, err := llmkit.Stream(ctx, client, simpleRequest(), func(d llmkit.Delta) error {
				if d.Kind == llmkit.DeltaText {
					deltas++
				}
				return nil
			})
			if deltas != 1 {
				t.Fatalf("text deltas = %d, want 1 delivered before the stall", deltas)
			}
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("err = %v, want context.DeadlineExceeded in the chain", err)
			}
			var apiErr *llmkit.APIError
			if errors.As(err, &apiErr) {
				t.Errorf("err = %v (%T), want the retryable *llmkit.APIError replaced", err, err)
			}
			if _, _, retryable := llmkit.Classify(err); retryable {
				t.Errorf("Classify(err) = retryable, want terminal for an expired caller deadline")
			}
			if !strings.Contains(err.Error(), "(last attempt: llmkit: "+n.name+" error: ") {
				t.Errorf("err = %v, want the adapter's error in the text", err)
			}
			if got := hits.Load(); got != 1 {
				t.Errorf("wire hits = %d, want 1 (a delivered delta is never replayed)", got)
			}
		})
	}
}

// TestNew_StreamCallbackSentinelSurvivesCancel pins an Anthropic SSE stream whose callback cancels the caller's ctx then returns its own sentinel surfaces that sentinel — the failure is terminal, returned as-is.
func TestNew_StreamCallbackSentinelSurvivesCancel(t *testing.T) {
	events := []struct{ name, data string }{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	var hits atomic.Int32
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, ev := range events {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.name, ev.data)
		}
	})
	client := newTargets()[0].build(t, base, Options{})
	sentinel := errors.New("caller budget exceeded")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error {
		cancel()
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want the callback's sentinel returned as-is", err)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("wire hits = %d, want 1", got)
	}
}

// llmkitRetryCfg builds a WithRetry config with a deterministic no-op
// schedule for the conformance retry tests.
func llmkitRetryCfg(sleep func(context.Context, time.Duration) error) retry.Config {
	return retry.Config{
		MaxAttempts: 2,
		BaseDelay:   time.Millisecond,
		Jitter:      0,
		Sleep:       sleep,
	}
}

// TestConformance_RetryAfterHonoredOnEveryStatus pins a 503 carrying Retry-After: 7 is parsed on every adapter (Retry-After is not 429/529-only). Anthropic, OpenAI, and openai-compatible report HasRetryAfter with RetryAfter 7s; Google hides response headers and reports HasRetryAfter false.
func TestConformance_RetryAfterHonoredOnEveryStatus(t *testing.T) {
	h := http.Header{}
	h.Set("Retry-After", "7")
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, http.StatusServiceUnavailable,
				errorBody(f.name, http.StatusServiceUnavailable, "overloaded"), h)
			apiErr := mustBeAPIError(t, err, f, llmkit.ErrServer, http.StatusServiceUnavailable)
			if f.name == "google" {
				if apiErr.HasRetryAfter {
					t.Errorf("google HasRetryAfter = true, want false (genai hides response headers)")
				}
			} else {
				if !apiErr.HasRetryAfter {
					t.Fatalf("%s HasRetryAfter = false, want true (Retry-After parses on any status)", f.name)
				}
				if apiErr.RetryAfter != 7*time.Second {
					t.Errorf("%s RetryAfter = %v, want 7s", f.name, apiErr.RetryAfter)
				}
			}

			// Through the retry stage: the carried delay replaces the
			// schedule; an absent one leaves the schedule.
			var hits atomic.Int32
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				for k, vs := range h {
					for _, v := range vs {
						w.Header().Add(k, v)
					}
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(errorBody(f.name, http.StatusServiceUnavailable, "overloaded")))
			})
			var sleeps []time.Duration
			cfg := llmkitRetryCfg(func(_ context.Context, d time.Duration) error {
				sleeps = append(sleeps, d)
				return nil
			})
			client := llmkit.WithRetry(f.build(t, base), cfg)
			if _, err := client.Complete(context.Background(), simpleRequest()); !errors.Is(err, llmkit.ErrServer) {
				t.Fatalf("Complete err = %v, want ErrServer", err)
			}
			if hits.Load() != 2 {
				t.Fatalf("wire hits = %d, want 2", hits.Load())
			}
			if f.name == "google" {
				if len(sleeps) != 1 || sleeps[0] != time.Millisecond {
					t.Errorf("google sleeps = %v, want the 1ms exponential schedule", sleeps)
				}
			} else {
				if len(sleeps) != 1 || sleeps[0] != 7*time.Second {
					t.Errorf("%s sleeps = %v, want [7s] (carried Retry-After honoured)", f.name, sleeps)
				}
			}
		})
	}
}

// TestConformance_ErrorNormalization_NotFound404 pins 404 — like every other 4xx outside the special cases — classifies ErrInvalidRequest with status preserved; the sub-400 in-band lookup must not leak into real 4xx.
func TestConformance_ErrorNormalization_NotFound404(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			err := completeExpectingError(t, f, http.StatusNotFound,
				errorBody(f.name, http.StatusNotFound, "no such model"), nil)
			_ = mustBeAPIError(t, err, f, llmkit.ErrInvalidRequest, http.StatusNotFound)
		})
	}
}

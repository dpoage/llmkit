package provider

// Tests for Options.StreamIdleTimeout: the retry stage's Stream idle
// watchdog. HTTP cases run New over an httptest SSE server; the races
// (caller cancel versus the watchdog, a delta after the trip) use a stub
// inner client because an HTTP server cannot order them.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

const idleD = 500 * time.Millisecond

// idleSlack bounds how late the watchdog may fire on a loaded host.
const idleSlack = 2 * time.Second

func idleRetry() retry.Config {
	return retry.Config{
		MaxAttempts:    2,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: 30 * time.Second,
		Sleep:          func(context.Context, time.Duration) error { return nil },
	}
}

func sseTextChunk(text string) string {
	return `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"llama-test",` +
		`"choices":[{"index":0,"delta":{"content":"` + text + `"},"finish_reason":null}],"usage":null}` + "\n\n"
}

const sseFinish = `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"llama-test",` +
	`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}` + "\n\n" +
	`data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"llama-test",` +
	`"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}` + "\n\ndata: [DONE]\n\n"

// sseHandler runs script(i, w, r) for the i-th request (0-based) after
// sending 200 and the SSE headers; the returned counter is the hit count.
func sseHandler(t *testing.T, script func(i int, w http.ResponseWriter, r *http.Request)) (string, *atomic.Int32) {
	t.Helper()
	hits := &atomic.Int32{}
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		i := int(hits.Add(1)) - 1
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		script(i, w, r)
	})
	return base, hits
}

func writeFlush(w http.ResponseWriter, s string) {
	_, _ = fmt.Fprint(w, s)
	w.(http.Flusher).Flush()
}

func serveHello(w http.ResponseWriter) {
	writeFlush(w, sseTextChunk("hello")+sseFinish)
}

// stallUntilClosed holds the response open until the client goes away.
func stallUntilClosed(r *http.Request) { <-r.Context().Done() }

func idleClient(t *testing.T, base string, idle time.Duration, obs llmkit.Observer) llmkit.Client {
	t.Helper()
	client, err := New(context.Background(), observeSpec(base), Options{
		Retry:             idleRetry(),
		Observer:          obs,
		StreamIdleTimeout: idle,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return client
}

func idleCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func attemptEvents(obs *eventCapture) []*llmkit.AttemptEvent {
	var out []*llmkit.AttemptEvent
	for _, ev := range obs.byKind(llmkit.KindAttempt) {
		out = append(out, ev.Attempt)
	}
	return out
}

func isIdleErr(err error) bool {
	var ae *llmkit.APIError
	return errors.As(err, &ae) && ae.Kind == llmkit.ErrServer && strings.Contains(ae.Message, "idle timeout")
}

// D1: a stream that sends headers and then nothing fails the attempt as
// idle, and the retry stage runs a second attempt.
func TestStreamIdle_SilentStreamFailsAttemptAndRetries(t *testing.T) {
	base, hits := sseHandler(t, func(i int, w http.ResponseWriter, r *http.Request) {
		if i == 0 {
			stallUntilClosed(r)
			return
		}
		serveHello(w)
	})
	obs := &eventCapture{}
	client := idleClient(t, base, idleD, obs)

	resp, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Text != "hello" {
		t.Errorf("Text = %q, want hello from attempt 2", resp.Text)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("server hits = %d, want 2", got)
	}
	attempts := attemptEvents(obs)
	if len(attempts) != 2 {
		t.Fatalf("attempt events = %d, want 2", len(attempts))
	}
	if !strings.Contains(attempts[0].Err, "idle timeout") || attempts[1].Err != "" {
		t.Errorf("attempt errs = %q, %q; want the idle error then success", attempts[0].Err, attempts[1].Err)
	}
	d := obs.byKind(llmkit.KindAttempt)[0].Duration
	if d < idleD || d > idleD+idleSlack {
		t.Errorf("attempt 1 lasted %v, want within [%v, %v]", d, idleD, idleD+idleSlack)
	}
}

// D2: a stall after a delivered delta is a terminal idle error.
func TestStreamIdle_StallAfterDeltaIsTerminal(t *testing.T) {
	base, hits := sseHandler(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		writeFlush(w, sseTextChunk("partial"))
		stallUntilClosed(r)
	})
	obs := &eventCapture{}
	client := idleClient(t, base, idleD, obs)

	var texts []string
	start := time.Now()
	_, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(d llmkit.Delta) error {
		texts = append(texts, d.Text)
		return nil
	})
	elapsed := time.Since(start)
	var ae *llmkit.APIError
	if !errors.As(err, &ae) || ae.Kind != llmkit.ErrServer || !strings.Contains(ae.Message, "idle") {
		t.Fatalf("err = %v, want *APIError ErrServer naming the idle timeout", err)
	}
	if !strings.Contains(ae.Message, idleD.String()) {
		t.Errorf("message %q does not name the duration %v", ae.Message, idleD)
	}
	if elapsed < idleD || elapsed > idleD+idleSlack {
		t.Errorf("stream failed after %v, want within [%v, %v]", elapsed, idleD, idleD+idleSlack)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("server hits = %d, want 1 (terminal after a delivered delta)", got)
	}
	if n := len(attemptEvents(obs)); n != 1 {
		t.Errorf("attempt events = %d, want 1", n)
	}
	if strings.Join(texts, "") != "partial" {
		t.Errorf("deltas = %q, want partial once", texts)
	}
}

// D3: SSE comments are not deltas and do not reset the clock.
func TestStreamIdle_KeepaliveCommentsDoNotReset(t *testing.T) {
	base, hits := sseHandler(t, func(i int, w http.ResponseWriter, r *http.Request) {
		if i > 0 {
			serveHello(w)
			return
		}
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-tick.C:
				writeFlush(w, ": keepalive\n\n")
			}
		}
	})
	obs := &eventCapture{}
	client := idleClient(t, base, idleD, obs)

	if _, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(llmkit.Delta) error { return nil }); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("server hits = %d, want 2", got)
	}
	evs := obs.byKind(llmkit.KindAttempt)
	if len(evs) != 2 || !strings.Contains(evs[0].Attempt.Err, "idle timeout") {
		t.Fatalf("attempt events = %d, first err %q; want the keepalive attempt to fail as idle", len(evs), evs[0].Attempt.Err)
	}
	if d := evs[0].Duration; d < idleD || d > idleD+idleSlack {
		t.Errorf("attempt 1 lasted %v, want within [%v, %v]", d, idleD, idleD+idleSlack)
	}
}

// D4a: deltas arriving every d/2 for 5d never trip the clock.
func TestStreamIdle_SteadyDeltasReset(t *testing.T) {
	base, hits := sseHandler(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		for range 10 {
			writeFlush(w, sseTextChunk("x"))
			select {
			case <-r.Context().Done():
				return
			case <-time.After(idleD / 2):
			}
		}
		writeFlush(w, sseFinish)
	})
	obs := &eventCapture{}
	client := idleClient(t, base, idleD, obs)

	n := 0
	resp, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(llmkit.Delta) error { n++; return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Text != strings.Repeat("x", 10) || n != 10 {
		t.Errorf("Text = %q, deltas = %d; want ten x", resp.Text, n)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("server hits = %d, want 1", got)
	}
}

// D4b: time inside the caller's fn is not idle time.
func TestStreamIdle_SlowCallbackDoesNotCount(t *testing.T) {
	base, hits := sseHandler(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		writeFlush(w, sseTextChunk("a"))
		writeFlush(w, sseTextChunk("b"))
		writeFlush(w, sseFinish)
	})
	obs := &eventCapture{}
	client := idleClient(t, base, idleD, obs)

	resp, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(llmkit.Delta) error {
		time.Sleep(2 * idleD)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Text != "ab" {
		t.Errorf("Text = %q, want ab", resp.Text)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("server hits = %d, want 1", got)
	}
}

// D5: a zero StreamIdleTimeout leaves a silent gap longer than any idle
// value alone to RequestTimeout.
func TestStreamIdle_ZeroDisables(t *testing.T) {
	base, hits := sseHandler(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(3 * idleD):
		}
		serveHello(w)
	})
	client := idleClient(t, base, 0, nil)

	resp, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(llmkit.Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Text != "hello" || hits.Load() != 1 {
		t.Errorf("Text = %q, hits = %d; want hello after one attempt", resp.Text, hits.Load())
	}
}

// D5: Complete has no deltas to watch and ignores the field.
func TestStreamIdle_CompleteIgnoresField(t *testing.T) {
	var hits atomic.Int32
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		select {
		case <-r.Context().Done():
			return
		case <-time.After(300 * time.Millisecond):
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("openai-compatible", "hello world", 12, 5)))
	})
	client := idleClient(t, base, 50*time.Millisecond, nil)

	resp, err := client.Complete(idleCtx(t), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "hello world" || hits.Load() != 1 {
		t.Errorf("Text = %q, hits = %d; want hello world after one attempt", resp.Text, hits.Load())
	}
}

// D6: a caller cancel during a stall with the watchdog armed surfaces as
// the caller's cancellation.
func TestStreamIdle_CallerCancelDuringStall(t *testing.T) {
	base, _ := sseHandler(t, func(_ int, _ http.ResponseWriter, r *http.Request) { stallUntilClosed(r) })
	client := idleClient(t, base, idleD, nil)

	ctx, cancel := context.WithCancel(idleCtx(t))
	time.AfterFunc(idleD/5, cancel)
	_, err := llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want it to chain context.Canceled", err)
	}
	if isIdleErr(err) || strings.Contains(err.Error(), "idle") {
		t.Errorf("err = %v, want the caller's cancellation, not the idle error", err)
	}
}

// idleStub is an inner streaming client driven by a script per attempt.
type idleStub struct {
	calls  atomic.Int32
	script func(attempt int, ctx context.Context, fn func(llmkit.Delta) error) (llmkit.Response, error)
}

func (s *idleStub) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }
func (s *idleStub) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	return llmkit.Response{}, errors.New("idleStub: Complete not scripted")
}
func (s *idleStub) Stream(ctx context.Context, _ llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	return s.script(int(s.calls.Add(1)), ctx, fn)
}

// D6 (race): the watchdog has fired and the caller's ctx is done by the
// time the inner stream returns. The adapters report a cancel as a plain
// error chaining context.Canceled.
func TestStreamIdle_CallerCancelRacingWatchdog(t *testing.T) {
	ctx, cancel := context.WithCancel(idleCtx(t))
	inner := &idleStub{script: func(_ int, actx context.Context, _ func(llmkit.Delta) error) (llmkit.Response, error) {
		<-actx.Done() // the watchdog's cancel
		cancel()      // the caller cancels in the same instant
		return llmkit.Response{}, fmt.Errorf("transport: %w", actx.Err())
	}}
	client := Wrap(inner, Options{Retry: idleRetry(), StreamIdleTimeout: 50 * time.Millisecond})

	_, err := llmkit.Stream(ctx, client, simpleRequest(), func(llmkit.Delta) error { return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want it to chain context.Canceled", err)
	}
	if strings.Contains(err.Error(), "idle") {
		t.Errorf("err = %v, want the caller's cancellation, not the idle error", err)
	}
	if got := inner.calls.Load(); got != 1 {
		t.Errorf("attempts = %d, want 1", got)
	}
}

// A delta that reaches the stage after the watchdog fired is dropped, so
// the attempt stays a retryable idle failure instead of becoming a
// terminal delivered-delta error.
func TestStreamIdle_DeltaAfterTripIsDropped(t *testing.T) {
	var delivered atomic.Int32
	inner := &idleStub{script: func(attempt int, actx context.Context, fn func(llmkit.Delta) error) (llmkit.Response, error) {
		if attempt == 2 {
			return llmkit.Response{Text: "ok"}, nil
		}
		<-actx.Done()
		if err := fn(llmkit.Delta{Kind: llmkit.DeltaText, Text: "late"}); err != nil {
			return llmkit.Response{}, err
		}
		return llmkit.Response{}, actx.Err()
	}}
	client := Wrap(inner, Options{Retry: idleRetry(), StreamIdleTimeout: 50 * time.Millisecond})

	resp, err := llmkit.Stream(idleCtx(t), client, simpleRequest(), func(llmkit.Delta) error {
		delivered.Add(1)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Text != "ok" || inner.calls.Load() != 2 {
		t.Errorf("Text = %q, attempts = %d; want ok on attempt 2", resp.Text, inner.calls.Load())
	}
	if n := delivered.Load(); n != 0 {
		t.Errorf("caller saw %d deltas, want 0 (the late one is dropped)", n)
	}
}

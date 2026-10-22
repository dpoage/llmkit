package agent

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

	"github.com/dpoage/llmkit"
)

const usagelessReasoningBytes = 7998

// usagelessReasoningServer replies to every request with reasoning_content,
// the text "42" and an echo tool call, and no usage object.
func usagelessReasoningServer(t *testing.T, requests *atomic.Int64) string {
	t.Helper()
	thought := strings.Repeat("r", usagelessReasoningBytes)
	call := fmt.Sprintf(`{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":%q}}`, usagelessToolArgs)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		requests.Add(1)
		if strings.Contains(string(raw), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, delta := range []string{
				fmt.Sprintf(`{"role":"assistant","reasoning_content":%q}`, thought),
				`{"content":"42"}`,
				fmt.Sprintf(`{"tool_calls":[%s]}`, call),
			} {
				_, _ = fmt.Fprintf(w, `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":%s,"finish_reason":null}]}`+"\n\n", delta)
			}
			_, _ = fmt.Fprint(w, `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"42","reasoning_content":%q,"tool_calls":[%s]},"finish_reason":"tool_calls"}]}`, thought, call)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func TestRunner_UsagelessReasoningTripsTokenBudget(t *testing.T) {
	wantReasoning := int64((usagelessReasoningBytes + 3) / 4)
	for _, streamed := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", streamed), func(t *testing.T) {
			var requests atomic.Int64
			client := usagelessClient(t, usagelessReasoningServer(t, &requests))
			// One turn: the next request resends the reasoning, whose input estimate alone exceeds the budget.
			opts := []Option{WithLimits(Limits{TokenBudget: 500, MaxIterations: 1})}
			if streamed {
				opts = append(opts, WithHooks(Hooks{Delta: func(context.Context, int, llmkit.Delta) {}}))
			}
			r := NewRunner(client, []Tool{echoTool{name: "echo"}}, "sys", opts...)
			outcome, err := r.Run(context.Background(), "task")

			var inc *IncompleteError
			if !errors.As(err, &inc) || inc.Reason != TruncTokenBudget {
				t.Fatalf("Run err = %v, want IncompleteError %s (requests %d, usage %+v)", err, TruncTokenBudget, requests.Load(), outcome.Usage)
			}
			u := outcome.Usage
			if want := wantReasoning * requests.Load(); u.ReasoningTokens != want {
				t.Errorf("ReasoningTokens = %d, want %d over %d turn(s)", u.ReasoningTokens, want, requests.Load())
			}
			if u.ReasoningTokens > u.OutputTokens || !u.Estimated {
				t.Errorf("outcome Usage = %+v, want estimated with ReasoningTokens <= OutputTokens", u)
			}
		})
	}
}

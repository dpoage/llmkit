package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

const usagelessToolArgs = `{"v":"x"}`

// usagelessServer answers every Chat Completions request with an echo tool
// call and no usage object, recording each request body's length.
func usagelessServer(t *testing.T, bodyLens *[]int) string {
	t.Helper()
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		mu.Lock()
		*bodyLens = append(*bodyLens, len(raw))
		mu.Unlock()
		call := fmt.Sprintf(`{"id":"call_1","type":"function","function":{"name":"echo","arguments":%q}}`, usagelessToolArgs)
		if strings.Contains(string(raw), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":%q}}]},"finish_reason":null}]}`+"\n\n", usagelessToolArgs)
			_, _ = fmt.Fprint(w, `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","tool_calls":[%s]},"finish_reason":"tool_calls"}]}`, call)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func usagelessClient(t *testing.T, baseURL string) llmkit.Client {
	t.Helper()
	client, err := provider.New(context.Background(), provider.Spec{
		Type: provider.TypeOpenAICompatible, Model: "m", BaseURL: baseURL, Secret: "k",
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	return client
}

// turnEstimate is the estimate for one usage-less turn: the request body
// length measured by the server, plus the echo call's name and arguments.
func turnEstimate(bodyLen int) int64 {
	return int64((bodyLen+3)/4) + int64((len("echo")+len(usagelessToolArgs)+3)/4)
}

func TestRunner_UsagelessEndpointTripsTokenBudget(t *testing.T) {
	for _, streamed := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", streamed), func(t *testing.T) {
			var lens []int
			client := usagelessClient(t, usagelessServer(t, &lens))
			opts := []Option{WithLimits(Limits{TokenBudget: 10, MaxIterations: 5})}
			if streamed {
				opts = append(opts, WithHooks(Hooks{Delta: func(context.Context, int, llmkit.Delta) {}}))
			}
			r := NewRunner(client, []Tool{echoTool{name: "echo"}}, "sys", opts...)
			outcome, err := r.Run(context.Background(), "task")

			var inc *IncompleteError
			if !errors.As(err, &inc) || inc.Reason != TruncTokenBudget {
				t.Fatalf("Run err = %v, want IncompleteError %s (requests %d)", err, TruncTokenBudget, len(lens))
			}
			if len(lens) != 1 {
				t.Fatalf("requests = %d, want 1", len(lens))
			}
			if est := turnEstimate(lens[0]); est <= 10 {
				t.Fatalf("one turn's estimate %d does not exceed the budget 10: test is vacuous", est)
			}
			if want := turnEstimate(lens[0]); outcome.Usage.InputTokens+outcome.Usage.OutputTokens != want || !outcome.Usage.Estimated {
				t.Errorf("outcome Usage = %+v, want estimated total %d", outcome.Usage, want)
			}
		})
	}
}

func TestRunner_UsagelessEndpointChargesBudgetPool(t *testing.T) {
	var lens []int
	client := usagelessClient(t, usagelessServer(t, &lens))
	pool := NewBudgetPool(10)
	r := NewRunner(client, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{MaxIterations: 5}), WithBudgetPool(pool))
	_, err := r.Run(context.Background(), "task")

	var inc *IncompleteError
	if !errors.As(err, &inc) || inc.Reason != TruncBudgetPool {
		t.Fatalf("Run err = %v, want IncompleteError %s (requests %d)", err, TruncBudgetPool, len(lens))
	}
	if len(lens) != 1 {
		t.Fatalf("requests = %d, want 1", len(lens))
	}
	if want := turnEstimate(lens[0]); pool.Spent() != want {
		t.Errorf("pool.Spent() = %d, want %d", pool.Spent(), want)
	}
}

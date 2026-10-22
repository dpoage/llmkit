package openai

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

func TestReasoningUsageConvention(t *testing.T) {
	cases := []struct {
		name       string
		compatible bool
		usage      string
		want       llmkit.Usage
	}{
		{"openai", false,
			`{"prompt_tokens":10,"completion_tokens":100,"total_tokens":110,"completion_tokens_details":{"reasoning_tokens":64}}`,
			llmkit.Usage{InputTokens: 10, OutputTokens: 100, ReasoningTokens: 64}},
		// Shape of the docs.x.ai Chat Completions example: total = prompt + completion + reasoning.
		{"xai", true,
			`{"prompt_tokens":32,"completion_tokens":9,"total_tokens":135,"completion_tokens_details":{"reasoning_tokens":94}}`,
			llmkit.Usage{InputTokens: 32, OutputTokens: 103, ReasoningTokens: 94}},
		{"reasoning zero", true,
			`{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":0}}`,
			llmkit.Usage{InputTokens: 10, OutputTokens: 5}},
		{"details absent", true,
			`{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`,
			llmkit.Usage{InputTokens: 10, OutputTokens: 5}},
		{"total absent", true,
			`{"prompt_tokens":10,"completion_tokens":100,"completion_tokens_details":{"reasoning_tokens":64}}`,
			llmkit.Usage{InputTokens: 10, OutputTokens: 100, ReasoningTokens: 64}},
	}
	for _, tc := range cases {
		check := func(t *testing.T, got llmkit.Usage) {
			t.Helper()
			if got != tc.want {
				t.Errorf("Usage = %+v, want %+v", got, tc.want)
			}
			if got.ReasoningTokens > got.OutputTokens {
				t.Errorf("ReasoningTokens %d > OutputTokens %d", got.ReasoningTokens, got.OutputTokens)
			}
		}
		streamClient := func(t *testing.T, baseURL string) llmkit.StreamingClient {
			t.Helper()
			client, err := New(streamModel, Options{APIKey: "k", BaseURL: baseURL, Compatible: tc.compatible})
			if err != nil {
				t.Fatal(err)
			}
			return client.(llmkit.StreamingClient)
		}
		chunks := []string{
			chunkJSON(`{"role":"assistant","content":"hi"}`, "", ""),
			chunkJSON(`{}`, "stop", ""),
			usageChunkJSON(tc.usage),
		}
		t.Run(tc.name+"/complete", func(t *testing.T) {
			url := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","created":1,"model":%q,`+
					`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":%s}`,
					streamID, streamModel, tc.usage)
			})
			resp, err := streamClient(t, url).Complete(context.Background(), simpleRequest())
			if err != nil {
				t.Fatal(err)
			}
			check(t, resp.Usage)
		})
		t.Run(tc.name+"/stream with fn", func(t *testing.T) {
			sc := streamClient(t, newServer(t, sseHandler(nil, chunks...)))
			resp, err := sc.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			check(t, resp.Usage)
		})
		t.Run(tc.name+"/stream nil fn", func(t *testing.T) {
			sc := streamClient(t, newServer(t, sseHandler(nil, chunks...)))
			resp, err := sc.Stream(context.Background(), simpleRequest(), nil)
			if err != nil {
				t.Fatal(err)
			}
			check(t, resp.Usage)
		})
	}
}

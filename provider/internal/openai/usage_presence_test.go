package openai

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// usageCase is one wire usage value: raw is the JSON for the `usage` key
// ("" omits the key).
type usageCase struct {
	name      string
	raw       string
	estimated bool
}

var usageCases = []usageCase{
	{"absent", "", true},
	{"null", "null", true},
	{"all-zero object", `{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`, false},
	{"empty object", `{}`, false},
	{"reported", `{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16}`, false},
}

const (
	textReply = "hello there"
	toolName  = "get_weather"
	toolArgs  = `{"city":"Oslo"}`
)

// ceil4 is the estimate's heuristic: ceil(n/4).
func ceil4(n int) int64 { return int64((n + 3) / 4) }

func completeBody(usage string) string {
	u := ""
	if usage != "" {
		u = `,"usage":` + usage
	}
	return fmt.Sprintf(`{"id":"c1","object":"chat.completion","created":1,"model":%q,`+
		`"choices":[{"index":0,"message":{"role":"assistant","content":%q,"tool_calls":[{"id":"call_1",`+
		`"type":"function","function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}]%s}`,
		streamModel, textReply, toolName, toolArgs, u)
}

func streamChunks(usage string) []string {
	chunks := []string{
		chunkJSON(`{"role":"assistant","content":"hello"}`, "", ""),
		chunkJSON(`{"content":" there"}`, "", ""),
		chunkJSON(`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"`+toolName+`","arguments":""}}]}`, "", ""),
		argFragChunk(0, toolArgs),
		chunkJSON(`{}`, "tool_calls", ""),
	}
	if usage != "" {
		chunks = append(chunks, usageChunkJSON(usage))
	}
	return chunks
}

// recordBody wraps h and stores the byte length of each request body.
func recordBody(h http.HandlerFunc, got *int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*got = len(b)
		r.Body = io.NopCloser(bytes.NewReader(b))
		h(w, r)
	}
}

func completeHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}
}

func TestUsageEstimated_AllPaths(t *testing.T) {
	type path struct {
		name string
		run  func(t *testing.T, usage string) (llmkit.Response, int)
	}
	paths := []path{
		{"complete", func(t *testing.T, usage string) (llmkit.Response, int) {
			var n int
			client, err := New(streamModel, Options{APIKey: "k", BaseURL: newServer(t, recordBody(completeHandler(completeBody(usage)), &n))})
			if err != nil {
				t.Fatal(err)
			}
			resp, err := client.Complete(context.Background(), simpleRequest())
			if err != nil {
				t.Fatal(err)
			}
			return resp, n
		}},
		{"stream fn", func(t *testing.T, usage string) (llmkit.Response, int) {
			var n int
			resp, _, err := stream(t, newServer(t, recordBody(sseHandler(nil, streamChunks(usage)...), &n)), simpleRequest(),
				func(llmkit.Delta) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			return resp, n
		}},
		{"stream nil fn", func(t *testing.T, usage string) (llmkit.Response, int) {
			var n int
			resp, _, err := stream(t, newServer(t, recordBody(sseHandler(nil, streamChunks(usage)...), &n)), simpleRequest(), nil)
			if err != nil {
				t.Fatal(err)
			}
			return resp, n
		}},
	}
	for _, p := range paths {
		for _, c := range usageCases {
			t.Run(p.name+"/"+c.name, func(t *testing.T) {
				resp, bodyLen := p.run(t, c.raw)
				if resp.Usage.Estimated != c.estimated {
					t.Fatalf("Estimated = %v, want %v (Usage %+v)", resp.Usage.Estimated, c.estimated, resp.Usage)
				}
				if !c.estimated {
					return
				}
				want := llmkit.Usage{
					InputTokens:  ceil4(bodyLen),
					OutputTokens: ceil4(len(textReply) + len(toolName) + len(toolArgs)),
					Estimated:    true,
				}
				if bodyLen == 0 || resp.Usage != want {
					t.Errorf("Usage = %+v, want %+v (request body %d bytes)", resp.Usage, want, bodyLen)
				}
			})
		}
	}
}

func TestUsageEstimated_ReportedValuesUnchanged(t *testing.T) {
	var n int
	client, err := New(streamModel, Options{APIKey: "k", BaseURL: newServer(t, recordBody(completeHandler(completeBody(`{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16}`)), &n))})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	if want := (llmkit.Usage{InputTokens: 7, OutputTokens: 9}); resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
}

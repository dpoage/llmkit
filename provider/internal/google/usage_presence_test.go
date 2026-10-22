package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

const (
	textReply = "hello there"
	toolName  = "get_weather"
	toolArgs  = `{"city":"Oslo"}`
)

func ceil4(n int) int64 { return int64((n + 3) / 4) }

// usageCase is one wire usageMetadata value: meta nil omits the key (null
// sends JSON null); a non-nil meta, even empty, is a present object.
type usageCase struct {
	name      string
	meta      map[string]any
	null      bool
	estimated bool
}

var usageCases = []usageCase{
	{name: "absent", estimated: true},
	{name: "null", null: true, estimated: true},
	{name: "all-zero object", meta: map[string]any{"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}},
	{name: "empty object", meta: map[string]any{}},
	{name: "reported", meta: map[string]any{"promptTokenCount": 7, "candidatesTokenCount": 9, "totalTokenCount": 16}},
}

func contentChunk(c usageCase) map[string]any {
	chunk := map[string]any{"candidates": []any{map[string]any{
		"content": map[string]any{"role": "model", "parts": []any{
			map[string]any{"text": textReply},
			map[string]any{"functionCall": map[string]any{"name": toolName, "args": map[string]any{"city": "Oslo"}}},
		}},
		"finishReason": "STOP",
	}}}
	switch {
	case c.null:
		chunk["usageMetadata"] = nil
	case c.meta != nil:
		chunk["usageMetadata"] = c.meta
	}
	return chunk
}

func recordingServer(t *testing.T, got *int, contentType string, body func() []byte) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*got = len(b)
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write(body())
	})
}

func TestUsageEstimated_AllPaths(t *testing.T) {
	type path struct {
		name string
		run  func(t *testing.T, c usageCase) (llmkit.Response, int)
	}
	paths := []path{
		{"complete", func(t *testing.T, c usageCase) (llmkit.Response, int) {
			var n int
			b, _ := json.Marshal(contentChunk(c))
			cl := newStreamClient(t, recordingServer(t, &n, "application/json", func() []byte { return b }))
			resp, err := cl.Complete(context.Background(), simpleRequest())
			if err != nil {
				t.Fatal(err)
			}
			return resp, n
		}},
		{"stream fn", func(t *testing.T, c usageCase) (llmkit.Response, int) {
			var n int
			cl := newStreamClient(t, recordingServer(t, &n, "text/event-stream", sse(c)))
			resp, err := cl.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			return resp, n
		}},
		{"stream nil fn", func(t *testing.T, c usageCase) (llmkit.Response, int) {
			var n int
			cl := newStreamClient(t, recordingServer(t, &n, "text/event-stream", sse(c)))
			resp, err := cl.Stream(context.Background(), simpleRequest(), nil)
			if err != nil {
				t.Fatal(err)
			}
			return resp, n
		}},
	}
	for _, p := range paths {
		for _, c := range usageCases {
			t.Run(p.name+"/"+c.name, func(t *testing.T) {
				resp, bodyLen := p.run(t, c)
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

// sse serves the content chunk in two events; the usage, when present,
// rides only the second so a stream folds it from a later chunk.
func sse(c usageCase) func() []byte {
	return func() []byte {
		var buf bytes.Buffer
		first := contentChunk(usageCase{})
		delete(first["candidates"].([]any)[0].(map[string]any), "finishReason")
		b1, _ := json.Marshal(first)
		fmt.Fprintf(&buf, "data: %s\n\n", b1)
		last := finalChunk("STOP", c.meta)
		if c.null {
			last["usageMetadata"] = nil
		}
		b2, _ := json.Marshal(last)
		fmt.Fprintf(&buf, "data: %s\n\n", b2)
		return buf.Bytes()
	}
}

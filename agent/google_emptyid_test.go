package agent_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/provider"
)

type w0bEchoArgs struct {
	V string `json:"v"`
}

// TestW0bGoogleEmptyCallIDPairsFunctionResponseByName drives a Runner over the
// in-tree Google provider against an httptest server whose first response
// carries a functionCall with no id: the second wire request must pair the
// functionResponse with the call's declared name, and the run must complete.
func TestW0bGoogleEmptyCallIDPairsFunctionResponseByName(t *testing.T) {
	const surrogate = "llmkit-w0b-google-s1-c0"
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("bad request body: %v", err)
		}
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		if len(bodies) == 1 {
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"echo","args":{"v":"x"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":4,"totalTokenCount":8}}`))
			return
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":4,"totalTokenCount":8}}`))
	}))
	defer srv.Close()

	cl, err := provider.New(context.Background(), provider.Spec{
		Type:    provider.TypeGoogle,
		Model:   "gemini-2.0-flash",
		BaseURL: srv.URL,
		Secret:  "k",
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	echo := agent.Func("echo", "echoes its argument",
		func(_ context.Context, a w0bEchoArgs) (string, error) { return a.V, nil })
	out, err := agent.NewRunner(cl, []agent.Tool{echo}, "sys").
		Run(context.Background(), "call echo with v=x, then answer", agent.WithRunID("w0b-google"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("FinalText = %q, want done", out.FinalText)
	}
	if len(bodies) != 2 {
		t.Fatalf("wire requests = %d, want 2", len(bodies))
	}

	contents, _ := bodies[1]["contents"].([]any)
	if len(contents) < 2 {
		t.Fatalf("second request holds %d contents, want the assistant turn and the tool result", len(contents))
	}
	wire, _ := json.Marshal(contents)
	t.Logf("second request contents: %s", wire)
	last, _ := contents[len(contents)-1].(map[string]any)
	parts, _ := last["parts"].([]any)
	var respName, respID string
	for _, p := range parts {
		m, _ := p.(map[string]any)
		if fr, ok := m["functionResponse"].(map[string]any); ok {
			respName, _ = fr["name"].(string)
			respID, _ = fr["id"].(string)
		}
	}
	if respName != "echo" {
		t.Errorf("functionResponse.name = %q, want the declared tool name %q", respName, "echo")
	}
	if respID != surrogate {
		t.Errorf("functionResponse.id = %q, want the surrogate %q", respID, surrogate)
	}

	// The recorded history carries the same surrogate on the assistant call and
	// its tool result, so the pairing is reproducible from the record.
	var callID, resultID string
	for _, m := range out.Messages {
		if m.Role == llmkit.RoleAssistant && len(m.ToolCalls) > 0 {
			callID = m.ToolCalls[0].ID
		}
		if m.Role == llmkit.RoleToolResult {
			resultID = m.ToolCallID
		}
	}
	if callID != surrogate || resultID != surrogate {
		t.Errorf("history IDs = (%q, %q), want the surrogate %q on both", callID, resultID, surrogate)
	}
}

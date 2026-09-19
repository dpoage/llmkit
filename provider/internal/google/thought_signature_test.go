package google

import (
	"context"
	"encoding/json"
	"github.com/dpoage/llmkit"
	"google.golang.org/genai"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// TestGoogleThoughtSignature_CarrierRoundTrip pins the signature carrier
// mechanism end to end:
//
//  1. a functionCall part carrying a thoughtSignature (Gemini 2.5+ signs
//     calls that were preceded by hidden thought) surfaces as a companion
//     BlockThinking with Provider "google" whose Raw is the re-encodable
//     Part — llmkit.ToolCall has nowhere to keep the signature;
//  2. re-sending that assistant turn attaches the signature to the matching
//     functionCall part on the wire (the API rejects unsigned replayed
//     calls) WITHOUT emitting the carrier as a duplicate functionCall.
func TestGoogleThoughtSignature_CarrierRoundTrip(t *testing.T) {
	// base64("signature") — what the wire carries, decoded into
	// genai.Part.ThoughtSignature ([]byte) by the SDK.
	const wireSig = "c2lnbmF0dXJl"
	firstBody := `{
		"candidates": [{"content": {"role": "model", "parts": [
			{"text": "Checking."},
			{"functionCall": {"id": "call_1", "name": "read_file", "args": {"path": "config.yaml"}},
			 "thoughtSignature": "` + wireSig + `"}
		]}, "finishReason": "STOP"}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
	}`
	secondBody := `{
		"candidates": [{"content": {"role": "model", "parts": [
			{"text": "It says bugbot."}
		]}, "finishReason": "STOP"}],
		"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 4, "totalTokenCount": 24}
	}`

	var mu sync.Mutex
	requestCount := 0
	var secondRequest map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		defer mu.Unlock()
		requestCount++
		if requestCount == 2 {
			_ = json.Unmarshal(body, &secondRequest)
		}
		w.Header().Set("Content-Type", "application/json")
		if requestCount == 1 {
			_, _ = w.Write([]byte(firstBody))
			return
		}
		_, _ = w.Write([]byte(secondBody))
	}))
	t.Cleanup(srv.Close)

	client, err := New(context.Background(), "gemini-test", Options{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	readFile := llmkit.ToolDef{
		Name:        "read_file",
		Description: "Read a file.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
	}

	// Turn 1: the model signs its function call.
	first, err := client.Complete(context.Background(), llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Read config.yaml")},
		Tools:     []llmkit.ToolDef{readFile},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Complete (turn 1): %v", err)
	}
	if len(first.ToolCalls) != 1 || first.ToolCalls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v, want the signed call_1", first.ToolCalls)
	}

	// The signature must ride a companion thinking block whose Raw is the
	// re-encodable Part — losing it would make the replay unsigned.
	var carrier *llmkit.Block
	for i := range first.Blocks {
		b := &first.Blocks[i]
		if b.Kind == llmkit.BlockThinking && b.Provider == "google" {
			var p genai.Part
			if json.Unmarshal(b.Raw, &p) == nil && p.FunctionCall != nil && p.FunctionCall.ID == "call_1" {
				carrier = b
			}
		}
	}
	if carrier == nil {
		t.Fatalf("no signature-carrier thinking block in Blocks = %+v", first.Blocks)
	}
	var sigPart genai.Part
	if err := json.Unmarshal(carrier.Raw, &sigPart); err != nil {
		t.Fatalf("carrier Raw does not decode to a genai.Part: %v", err)
	}
	if string(sigPart.ThoughtSignature) != "signature" {
		t.Errorf("carrier ThoughtSignature = %q, want %q", sigPart.ThoughtSignature, "signature")
	}

	// Turn 2: replay the assistant turn with the tool result.
	assistant := llmkit.Message{Role: llmkit.RoleAssistant, Content: first.Blocks, ToolCalls: first.ToolCalls}
	toolResult := llmkit.TextMessage(llmkit.RoleToolResult, "service: bugbot\n")
	toolResult.ToolCallID = "call_1"
	second, err := client.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "Read config.yaml"),
			assistant,
			toolResult,
		},
		Tools:     []llmkit.ToolDef{readFile},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Complete (turn 2): %v", err)
	}
	if second.Text != "It says bugbot." {
		t.Errorf("turn 2 Text = %q, want %q", second.Text, "It says bugbot.")
	}

	// On the wire: exactly ONE functionCall part in the model turn (the
	// carrier must not duplicate the call) carrying the signature back.
	if secondRequest == nil {
		t.Fatal("second request was not captured")
	}
	contents, _ := secondRequest["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %d entries, want 3", len(contents))
	}
	modelTurn, _ := contents[1].(map[string]any)
	if modelTurn["role"] != "model" {
		t.Fatalf("contents[1].role = %v, want model", modelTurn["role"])
	}
	parts, _ := modelTurn["parts"].([]any)
	functionCalls := 0
	signed := false
	for _, pv := range parts {
		p, _ := pv.(map[string]any)
		if _, ok := p["functionCall"]; ok {
			functionCalls++
			if sig, ok := p["thoughtSignature"].(string); ok && sig == wireSig {
				signed = true
			}
		}
	}
	if functionCalls != 1 {
		t.Errorf("model turn has %d functionCall parts, want exactly 1 (carrier must not duplicate)", functionCalls)
	}
	if !signed {
		t.Errorf("replayed functionCall part lost thoughtSignature %q; parts = %v", wireSig, parts)
	}
	if !strings.Contains(second.Text, "bugbot") {
		t.Errorf("sanity: unexpected final text %q", second.Text)
	}
}

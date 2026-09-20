package google

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestThinking_EmptyRawSkipped pins the guard for assistant thinking blocks
// whose Raw carries nothing re-encodable: nil (a Block that passed through
// encoding/json re-decodes a nil []byte as literal "null"), "null", "{}",
// and whitespace-padded variants of each are all SKIPPED — no empty part
// reaches the wire and no error is raised. This is deliberately softer than
// anthropic's guard, which rejects a missing Raw outright: anthropic
// thinking is signed and mandatory on replay, while a decoded-to-empty
// genai Part contributes nothing Gemini requires. A malformed Raw is still
// an error.
func TestThinking_EmptyRawSkipped(t *testing.T) {
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
	client, err := New(context.Background(), "gemini-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// A real payload still forwards: a Raw decoding to an actual
	// thought part must reach the wire alongside the text part.
	real := json.RawMessage(`{"text":"pondering"}`)
	req := simpleRequest()
	req.Messages = []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "hi"),
		{
			Role: llmkit.RoleAssistant,
			Content: []llmkit.Block{
				{Kind: llmkit.BlockThinking, Provider: "google", Raw: real},
				{Kind: llmkit.BlockText, Text: "hello"},
			},
		},
	}
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Fatalf("Raw=%q: Complete: %v", real, err)
	}
	contents, _ := captured["contents"].([]any)
	model, _ := contents[1].(map[string]any)
	parts, _ := model["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("Raw=%q: model parts = %v, want the thought part and the text part", real, parts)
	}
	if p, _ := parts[0].(map[string]any); p["text"] != "pondering" {
		t.Errorf("Raw=%q: part[0] = %v, want text \"pondering\"", real, p)
	}

	for _, raw := range []json.RawMessage{
		nil,
		json.RawMessage("null"),
		json.RawMessage("{}"),
		json.RawMessage(" null"),
		json.RawMessage(" {} "),
		json.RawMessage("\t\n"),
	} {
		req := simpleRequest()
		req.Messages = []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "hi"),
			{
				Role: llmkit.RoleAssistant,
				Content: []llmkit.Block{
					{Kind: llmkit.BlockThinking, Provider: "google", Raw: raw},
					{Kind: llmkit.BlockText, Text: "hello"},
				},
			},
		}
		if _, err := client.Complete(context.Background(), req); err != nil {
			t.Fatalf("Raw=%q: Complete: %v", raw, err)
		}
		contents, _ := captured["contents"].([]any)
		model, _ := contents[1].(map[string]any)
		parts, _ := model["parts"].([]any)
		if len(parts) != 1 {
			t.Fatalf("Raw=%q: model parts = %v, want only the text part (no empty part from the skipped block)", raw, parts)
		}
		p, _ := parts[0].(map[string]any)
		if p["text"] != "hello" {
			t.Errorf("Raw=%q: part = %v, want text \"hello\"", raw, p)
		}
	}
}

// TestThinking_MalformedRawStillErrors pins the guard's boundary: an empty
// Raw is skipped, but a Raw that is neither empty nor parseable as a
// genai.Part is a malformed provider payload and must fail with
// ErrInvalidRequest rather than be silently dropped.
func TestThinking_MalformedRawStillErrors(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
	client, err := New(context.Background(), "gemini-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.Messages = []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "hi"),
		{
			Role: llmkit.RoleAssistant,
			Content: []llmkit.Block{
				{Kind: llmkit.BlockThinking, Provider: "google", Raw: json.RawMessage("{not json")},
			},
		},
	}
	_, err = client.Complete(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for malformed thinking Raw, got nil")
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("error = %v, want ErrInvalidRequest", err)
	}
}

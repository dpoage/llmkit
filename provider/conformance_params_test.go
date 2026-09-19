package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// captureComplete runs one Complete against a server that records the JSON
// request body and replies with a successful text response for the provider.
func captureComplete(t *testing.T, f adapterFactory, req llmkit.Request) map[string]any {
	t.Helper()
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody(f.name, "ok", 1, 1)))
	})
	client := f.build(t, base)
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Fatalf("%s Complete: %v", f.name, err)
	}
	return captured
}

// outputTokenCapOf extracts the provider's output-token cap from a captured
// wire body. It requires the field to be PRESENT — "rely on the backend's
// unset behavior" is exactly what the uniform rule forbids.
func outputTokenCapOf(t *testing.T, provider string, body map[string]any) float64 {
	t.Helper()
	switch provider {
	case "anthropic":
		return numberField(t, body, "max_tokens")
	case "openai", "openai-compatible":
		return numberField(t, body, "max_completion_tokens")
	case "google":
		cfg, ok := body["generationConfig"].(map[string]any)
		if !ok {
			t.Fatalf("google: no generationConfig object in %v", body)
		}
		return numberField(t, cfg, "maxOutputTokens")
	}
	t.Fatalf("unknown provider %q", provider)
	return 0
}

func numberField(t *testing.T, m map[string]any, key string) float64 {
	t.Helper()
	v, ok := m[key].(float64)
	if !ok {
		t.Fatalf("field %q missing or not a number in %v", key, m)
	}
	return v
}

// TestConformance_DefaultMaxTokens asserts the uniform rule on the wire:
// with Request.MaxTokens unset, EVERY adapter sends exactly
// llmkit.DefaultMaxTokens as its output cap (max_tokens on Anthropic, which
// requires the field, max_completion_tokens on OpenAI, maxOutputTokens on
// Gemini) — none of them omit the field and leave the cap to the backend.
func TestConformance_DefaultMaxTokens(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			for _, maxTokens := range []int{0, -5} {
				req := simpleRequest()
				req.MaxTokens = maxTokens
				body := captureComplete(t, f, req)
				want := float64(llmkit.DefaultMaxTokens)
				if got := outputTokenCapOf(t, f.name, body); got != want {
					t.Errorf("%s wire cap with MaxTokens=%d = %v, want %v", f.name, maxTokens, got, want)
				}
			}
		})
	}
}

// TestConformance_ExplicitMaxTokensPassesThrough asserts an explicit
// Request.MaxTokens reaches the wire verbatim on every adapter — no floor,
// no clamping, no provider-specific forcing.
func TestConformance_ExplicitMaxTokensPassesThrough(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			req := simpleRequest()
			req.MaxTokens = 64
			body := captureComplete(t, f, req)
			if got := outputTokenCapOf(t, f.name, body); got != 64 {
				t.Errorf("%s wire cap with MaxTokens=64 = %v, want 64", f.name, got)
			}
		})
	}
}

// --- inline RoleSystem placement --------------------------------------------
//
// Request.System is the reliable system channel, but the block model also
// allows a RoleSystem message inline in Messages. Where it lands differs by
// provider (documented on Request.Messages); these tests pin the exact wire
// placement.

// TestConformance_InlineSystemMessage asserts, per adapter, exactly where an
// inline RoleSystem message between two user turns ends up: OpenAI keeps it
// as a system-role entry in place; Anthropic and Gemini have no system role
// inside the message list, so it is sent as a user turn in place, while the
// Request.System text still rides the provider's system channel.
func TestConformance_InlineSystemMessage(t *testing.T) {
	req := llmkit.Request{
		System: "global system",
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "before"),
			llmkit.TextMessage(llmkit.RoleSystem, "inline directive"),
			llmkit.TextMessage(llmkit.RoleUser, "after"),
		},
		MaxTokens: 64,
	}
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			body := captureComplete(t, f, req)
			switch f.name {
			case "openai", "openai-compatible":
				// [0] is Request.System (the adapter prepends it); the
				// inline system message stays at its position among the
				// conversation entries.
				msgs := arrayField(t, body, "messages")
				if len(msgs) != 4 {
					t.Fatalf("messages = %d, want 4 (Request.System + 3 turns)", len(msgs))
				}
				if role, text := roleAndText(t, msgs[0]); role != "system" || text != "global system" {
					t.Errorf("messages[0] = (%q, %q), want (system, global system)", role, text)
				}
				if role, text := roleAndText(t, msgs[2]); role != "system" || text != "inline directive" {
					t.Errorf("messages[2] = (%q, %q), want (system, inline directive)", role, text)
				}
			case "anthropic":
				// Request.System owns the system channel...
				sysBlocks := arrayField(t, body, "system")
				if len(sysBlocks) != 1 {
					t.Fatalf("system blocks = %d, want 1 (from Request.System only)", len(sysBlocks))
				}
				if text := mapField(t, sysBlocks[0])["text"]; text != "global system" {
					t.Errorf("system[0].text = %v, want %q", text, "global system")
				}
				// ...and the inline system message rides as a USER turn in place.
				msgs := arrayField(t, body, "messages")
				if len(msgs) != 3 {
					t.Fatalf("messages = %d, want 3", len(msgs))
				}
				if role, text := roleAndText(t, msgs[1]); role != "user" || text != "inline directive" {
					t.Errorf("messages[1] = (%q, %q), want (user, inline directive)", role, text)
				}
			case "google":
				si := mapField(t, body["systemInstruction"])
				parts := arrayField(t, si, "parts")
				if len(parts) != 1 || mapField(t, parts[0])["text"] != "global system" {
					t.Errorf("systemInstruction = %v, want the Request.System text", body["systemInstruction"])
				}
				contents := arrayField(t, body, "contents")
				if len(contents) != 3 {
					t.Fatalf("contents = %d, want 3", len(contents))
				}
				cm := mapField(t, contents[1])
				if cm["role"] != "user" {
					t.Errorf("contents[1].role = %v, want user", cm["role"])
				}
				cparts := arrayField(t, cm, "parts")
				if len(cparts) != 1 || mapField(t, cparts[0])["text"] != "inline directive" {
					t.Errorf("contents[1].parts = %v, want the inline directive text", cparts)
				}
			}
		})
	}
}

func arrayField(t *testing.T, m map[string]any, key string) []any {
	t.Helper()
	v, ok := m[key].([]any)
	if !ok {
		t.Fatalf("field %q missing or not an array in %v", key, m)
	}
	return v
}

func mapField(t *testing.T, v any) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("value is not an object: %T (%v)", v, v)
	}
	return m
}

// roleAndText reads (role, concatenated text content) from a Chat
// Completions message entry.
func roleAndText(t *testing.T, entry any) (string, string) {
	t.Helper()
	m := mapField(t, entry)
	role, _ := m["role"].(string)
	switch c := m["content"].(type) {
	case string:
		return role, c
	case []any:
		text := ""
		for _, p := range c {
			if s, ok := mapField(t, p)["text"].(string); ok {
				text += s
			}
		}
		return role, text
	case nil:
		return role, ""
	default:
		t.Fatalf("unexpected content shape: %T", m["content"])
		return "", ""
	}
}

// TestConformance_AnthropicCacheTokenSummation asserts the normalized Usage
// convention at the adapter boundary: Anthropic's input_tokens EXCLUDES
// cache reads/writes, so InputTokens is their sum while the subsets stay
// reported in their own fields.
func TestConformance_AnthropicCacheTokenSummation(t *testing.T) {
	var anthropic adapterFactory
	for _, f := range allAdapters() {
		if f.name == "anthropic" {
			anthropic = f
		}
	}
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_cache", "type": "message", "role": "assistant", "model": "claude-test",
			"content": [{"type": "text", "text": "hi"}],
			"stop_reason": "end_turn", "stop_sequence": null,
			"usage": {"input_tokens": 100, "output_tokens": 10,
				"cache_read_input_tokens": 40, "cache_creation_input_tokens": 7}
		}`))
	})
	client := anthropic.build(t, base)
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	want := llmkit.Usage{
		InputTokens:              147, // 100 + 40 + 7
		OutputTokens:             10,
		CacheReadInputTokens:     40,
		CacheCreationInputTokens: 7,
	}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v (InputTokens must include both cache subsets)", resp.Usage, want)
	}
}

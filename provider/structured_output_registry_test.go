package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/internal/anthropic"
	"github.com/dpoage/llmkit/provider/internal/google"
	"github.com/dpoage/llmkit/provider/internal/openai"
)

// structuredOutputOverride returns a Capabilities override that flips
// StructuredOutput to b while keeping every other value of the adapter's
// table profile.
func structuredOutputOverride(b bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities {
		c.StructuredOutput = b
		return c
	}
}

// hasNestedField returns a closure suitable for use with runOne that
// reports whether `body` contains the nested key path (e.g. ["a","b"] checks
// body["a"]["b"]). The leaf must be present and non-nil.

// schemaForTest is a small, valid JSON Schema used across the
// structured-output tests. It declares a top-level object with a single
// required string property.
var schemaForTest = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

func hasNestedField(keys ...string) func(map[string]any) bool {
	return func(body map[string]any) bool {
		cur := body
		for i, k := range keys {
			v, ok := cur[k]
			if !ok || v == nil {
				return false
			}
			if i == len(keys)-1 {
				return true
			}
			m, ok := v.(map[string]any)
			if !ok {
				return false
			}
			cur = m
		}
		return false
	}
}

// TestStructuredOutput_AllGatedByCapability walks each provider with
// StructuredOutput=false and asserts the schema is dropped on the wire.
// This is the cross-adapter safety net: if any adapter ever forgets to gate
// the schema on the cap, this test surfaces it.
func TestStructuredOutput_AllGatedByCapability(t *testing.T) {
	// Pick a response body shape that works for the provider under test.
	respBody := func(name string) string {
		switch name {
		case "anthropic":
			return `{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
		case "google":
			return `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`
		default:
			return `{"id":"x","object":"chat.completion","created":1,"model":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
		}
	}
	runOne := func(t *testing.T, name string, build func(base string) llmkit.Client, leak func(map[string]any) bool) {
		var captured map[string]any
		base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(respBody(name)))
		})
		client := build(base)
		req := simpleRequest()
		req.ResponseSchema = schemaForTest
		if _, err := client.Complete(context.Background(), req); err != nil {
			t.Fatalf("%s Complete: %v", name, err)
		}
		if leak(captured) {
			t.Errorf("%s: schema leaked onto wire despite cap off: %v", name, captured)
		}
	}

	t.Run("openai", func(t *testing.T) {
		runOne(t, "openai", func(base string) llmkit.Client {
			c, err := openai.New("gpt-test", openai.Options{
				APIKey: "k", BaseURL: base,
				Capabilities: structuredOutputOverride(false),
			})
			if err != nil {
				t.Fatalf("openai.New: %v", err)
			}
			return c
		}, hasNestedField("response_format"))
	})
	t.Run("openai-compatible", func(t *testing.T) {
		runOne(t, "openai-compatible", func(base string) llmkit.Client {
			c, err := openai.New("llama-test", openai.Options{
				APIKey: "k", BaseURL: base, Compatible: true,
				Capabilities: structuredOutputOverride(false),
			})
			if err != nil {
				t.Fatalf("openai.New: %v", err)
			}
			return c
		}, hasNestedField("response_format"))
	})
	t.Run("google", func(t *testing.T) {
		runOne(t, "google", func(base string) llmkit.Client {
			a, err := google.New(context.Background(), "gemini-test", google.Options{
				APIKey: "k", BaseURL: base,
				Capabilities: structuredOutputOverride(false),
			})
			if err != nil {
				t.Fatalf("google.New: %v", err)
			}
			return a
		}, hasNestedField("generationConfig", "responseJsonSchema"))
	})
	t.Run("anthropic", func(t *testing.T) {
		runOne(t, "anthropic", func(base string) llmkit.Client {
			c, err := anthropic.New("claude-test", anthropic.Options{
				APIKey: "k", BaseURL: base,
				Capabilities: structuredOutputOverride(false),
			})
			if err != nil {
				t.Fatalf("anthropic.New: %v", err)
			}
			return c
		}, func(body map[string]any) bool {
			tools, ok := body["tools"].([]any)
			return ok && len(tools) > 0
		})
	})
}

// TestStructuredOutput_ConfigOverride_FlipsOpenAICompatibleCapabilities
// asserts the registry-level integration: a Spec whose Capabilities
// override flips StructuredOutput on causes New to return an
// openai-compatible client whose capabilities report StructuredOutput=true.
func TestStructuredOutput_ConfigOverride_FlipsOpenAICompatibleCapabilities(t *testing.T) {
	provider := Spec{
		Type:         TypeOpenAICompatible,
		BaseURL:      "http://example.invalid",
		Model:        "llama3",
		Secret:       "k",
		Capabilities: structuredOutputOverride(true),
	}
	client, err := New(context.Background(), provider, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !client.Capabilities().StructuredOutput {
		t.Error("Capabilities().StructuredOutput = false, want true (config override should flip it on)")
	}
}

// TestStructuredOutput_ConfigOverride_OffUnaffected asserts that an
// override forcing StructuredOutput off suppresses it even on a first-party
// provider that would default to true.
func TestStructuredOutput_ConfigOverride_OffUnaffected(t *testing.T) {
	isolateBaseURLSources(t)
	provider := Spec{
		Type:         TypeOpenAI,
		Model:        "gpt-test",
		Secret:       "k",
		Capabilities: structuredOutputOverride(false),
	}
	client, err := New(context.Background(), provider, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if client.Capabilities().StructuredOutput {
		t.Error("Capabilities().StructuredOutput = true, want false (config override should force it off)")
	}
}

// TestStructuredOutput_ConfigOverride_UnaffectedWhenUnset is a regression
// guard: a provider without the override must keep its default capability
// (true for first-party, false for openai-compatible).
func TestStructuredOutput_ConfigOverride_UnaffectedWhenUnset(t *testing.T) {
	isolateBaseURLSources(t)
	t.Run("first-party default true", func(t *testing.T) {
		provider := Spec{Type: TypeOpenAI, Model: "gpt-x", Secret: "k"}
		client, err := New(context.Background(), provider, Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if !client.Capabilities().StructuredOutput {
			t.Error("first-party default should be true")
		}
	})
	t.Run("openai-compatible default false", func(t *testing.T) {
		provider := Spec{Type: TypeOpenAICompatible, BaseURL: "http://x", Model: "llama", Secret: "k"}
		client, err := New(context.Background(), provider, Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if client.Capabilities().StructuredOutput {
			t.Error("openai-compatible default should be false")
		}
	})
}

package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider/anthropic"
	"github.com/dpoage/llmkit/provider/google"
	"github.com/dpoage/llmkit/provider/openai"
)

func ptr(b bool) *bool { return &b }

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
			return openai.New("gpt-test", openai.Options{
				APIKey: "k", BaseURL: base,
				StructuredOutput: ptr(false),
			})
		}, hasNestedField("response_format"))
	})
	t.Run("openai-compatible", func(t *testing.T) {
		runOne(t, "openai-compatible", func(base string) llmkit.Client {
			return openai.New("llama-test", openai.Options{
				APIKey: "k", BaseURL: base, Compatible: true,
				StructuredOutput: ptr(false),
			})
		}, hasNestedField("response_format"))
	})
	t.Run("google", func(t *testing.T) {
		runOne(t, "google", func(base string) llmkit.Client {
			a, err := google.New(context.Background(), "gemini-test", google.Options{
				APIKey: "k", BaseURL: base,
				StructuredOutput: ptr(false),
			})
			if err != nil {
				t.Fatalf("google.New: %v", err)
			}
			return a
		}, hasNestedField("generationConfig", "responseJsonSchema"))
	})
	t.Run("anthropic", func(t *testing.T) {
		runOne(t, "anthropic", func(base string) llmkit.Client {
			return anthropic.New("claude-test", anthropic.Options{
				APIKey: "k", BaseURL: base,
				StructuredOutput: ptr(false),
			})
		}, func(body map[string]any) bool {
			tools, ok := body["tools"].([]any)
			return ok && len(tools) > 0
		})
	})
}

// TestStructuredOutput_ConfigOverride_FlipsOpenAICompatibleCapabilities
// asserts the registry-level integration: a Spec with
// StructuredOutput=&true causes New to return an openai-compatible
// client whose capabilities report StructuredOutput=true.
func TestStructuredOutput_ConfigOverride_FlipsOpenAICompatibleCapabilities(t *testing.T) {
	tr := true
	provider := Spec{
		Type:             TypeOpenAICompatible,
		BaseURL:          "http://example.invalid",
		StructuredOutput: &tr,
	}
	client, err := New(context.Background(), provider, "test", "llama3", "k", Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !client.Capabilities().StructuredOutput {
		t.Error("Capabilities().StructuredOutput = false, want true (config override should flip it on)")
	}
}

// TestStructuredOutput_ConfigOverride_OffUnaffected asserts that an explicit
// false in the config suppresses StructuredOutput even on a first-party
// provider that would default to true.
func TestStructuredOutput_ConfigOverride_OffUnaffected(t *testing.T) {
	f := false
	provider := Spec{
		Type:             TypeOpenAI,
		StructuredOutput: &f,
	}
	client, err := New(context.Background(), provider, "test", "gpt-test", "k", Options{})
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
	t.Run("first-party default true", func(t *testing.T) {
		provider := Spec{Type: TypeOpenAI}
		client, err := New(context.Background(), provider, "t", "gpt-x", "k", Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if !client.Capabilities().StructuredOutput {
			t.Error("first-party default should be true")
		}
	})
	t.Run("openai-compatible default false", func(t *testing.T) {
		provider := Spec{Type: TypeOpenAICompatible, BaseURL: "http://x"}
		client, err := New(context.Background(), provider, "t", "llama", "k", Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if client.Capabilities().StructuredOutput {
			t.Error("openai-compatible default should be false")
		}
	})
}

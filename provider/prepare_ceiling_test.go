package provider

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNew_CeilingRefusesAboveWireCapability pins that provider.New refuses,
// with no network I/O, an effective profile that reports true for a
// wire-gated field the adapter's ceiling forbids — Thinking or TopK on
// TypeOpenAI/TypeOpenAICompatible (the Chat Completions API has neither),
// Seed on TypeAnthropic (the Messages API has no seed parameter). The
// error wraps llmkit.ErrInvalidRequest and its text names both the field
// and the provider.
func TestNew_CeilingRefusesAboveWireCapability(t *testing.T) {
	isolateBaseURLSources(t)
	cases := []struct {
		name     string
		typ      Type
		model    string
		override func(llmkit.Capabilities) llmkit.Capabilities
		field    string
	}{
		{"openai/Thinking", TypeOpenAI, "gpt-test", forceThinking(true), "Thinking"},
		{"openai/TopK", TypeOpenAI, "gpt-test", forceTopK(true), "TopK"},
		{"openai-compatible/Thinking", TypeOpenAICompatible, "llama-test", forceThinking(true), "Thinking"},
		{"openai-compatible/TopK", TypeOpenAICompatible, "llama-test", forceTopK(true), "TopK"},
		{"anthropic/Seed", TypeAnthropic, "claude-test", forceSeed(true), "Seed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hits := 0
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				hits++
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(okBody(string(tc.typ))))
			})
			spec := Spec{Type: tc.typ, Model: tc.model, BaseURL: base, Secret: "k", Capabilities: tc.override}
			client, err := New(context.Background(), spec, Options{})
			if client != nil {
				t.Errorf("New returned a non-nil client alongside an error")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("New err = %v, want ErrInvalidRequest", err)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("err = %q, want it to name the field %q", err.Error(), tc.field)
			}
			if !strings.Contains(err.Error(), string(tc.typ)) {
				t.Errorf("err = %q, want it to name the provider %q", err.Error(), tc.typ)
			}
			if hits != 0 {
				t.Errorf("wire hits = %d, want 0 (no network I/O)", hits)
			}
		})
	}
}

// TestNew_CeilingAllowsEverythingElse is the control for the above: New
// succeeds for overrides that set advisory/decorator fields (ContextWindow,
// ParallelToolCalls, PromptCaching), StructuredOutput=true on compat
// (within its ceiling), or any wire-gated field to false on any Type.
// Google has no refused field: setting every wire-gated field true still
// succeeds there.
func TestNew_CeilingAllowsEverythingElse(t *testing.T) {
	isolateBaseURLSources(t)
	cases := []struct {
		name     string
		typ      Type
		model    string
		override func(llmkit.Capabilities) llmkit.Capabilities
	}{
		{"contextWindow", TypeOpenAI, "gpt-test", func(c llmkit.Capabilities) llmkit.Capabilities {
			c.ContextWindow = 12345
			return c
		}},
		{"parallelToolCalls", TypeOpenAI, "gpt-test", func(c llmkit.Capabilities) llmkit.Capabilities {
			c.ParallelToolCalls = false
			return c
		}},
		{"promptCaching", TypeAnthropic, "claude-test", func(c llmkit.Capabilities) llmkit.Capabilities {
			c.PromptCaching = false
			return c
		}},
		{"compat/StructuredOutput=true", TypeOpenAICompatible, "llama-test", forceStructuredOutput(true)},
		{"openai/TopK=false", TypeOpenAI, "gpt-test", forceTopK(false)},
		{"openai/Thinking=false", TypeOpenAI, "gpt-test", forceThinking(false)},
		{"anthropic/Seed=false", TypeAnthropic, "claude-test", forceSeed(false)},
		{"google/every wire field true", TypeGoogle, "gemini-test", func(llmkit.Capabilities) llmkit.Capabilities {
			return llmkit.Capabilities{
				StructuredOutput: true, Thinking: true, ToolChoice: true,
				StopSequences: true, TopP: true, TopK: true, Seed: true,
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(okBody(string(tc.typ))))
			})
			spec := Spec{Type: tc.typ, Model: tc.model, BaseURL: base, Secret: "k", Capabilities: tc.override}
			client, err := New(context.Background(), spec, Options{})
			if err != nil {
				t.Fatalf("New: %v, want success (within ceiling)", err)
			}
			if client == nil {
				t.Fatal("New returned a nil client with no error")
			}
		})
	}
}

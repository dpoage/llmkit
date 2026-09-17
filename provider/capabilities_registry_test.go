package provider

import (
	"context"
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// overrideEveryField is a capability profile that differs from every
// adapter's table defaults on every field, so a DeepEqual against
// client.Capabilities() proves the Spec.Capabilities override REPLACED the
// table wholesale — any field that falls through from the table fails.
func overrideEveryField() *llmkit.Capabilities {
	return &llmkit.Capabilities{
		ContextWindow:     123_456,
		ParallelToolCalls: false,
		PromptCaching:     false,
		StructuredOutput:  false,
		Thinking:          false,
		ToolChoice:        false,
		Images:            false,
		Documents:         false,
		StopSequences:     false,
		TopP:              false,
		TopK:              true,
		Seed:              true,
	}
}

// TestCapabilities_SpecOverride_ReplacesTableWholesale walks every provider
// type with a KNOWN model (so a table entry exists to be replaced) and
// asserts the effective profile equals the override exactly.
func TestCapabilities_SpecOverride_ReplacesTableWholesale(t *testing.T) {
	cases := []struct {
		name  string
		spec  Spec
		model string
	}{
		{"anthropic", Spec{Type: TypeAnthropic}, "claude-opus-4-5"},
		{"openai", Spec{Type: TypeOpenAI}, "gpt-5"},
		{"openai-compatible", Spec{Type: TypeOpenAICompatible}, "llama3"},
		{"google", Spec{Type: TypeGoogle}, "gemini-2.5-pro"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.spec.Capabilities = overrideEveryField()
			client, err := New(context.Background(), tc.spec, "test", tc.model, "k", Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			want := *overrideEveryField()
			if got := client.Capabilities(); !reflect.DeepEqual(got, want) {
				t.Errorf("Capabilities() =\n  %+v\nwant the Spec.Capabilities override wholesale:\n  %+v", got, want)
			}
		})
	}
}

// TestCapabilities_SpecOverride_ComposesWithStructuredOutput pins the
// documented composition order: the adapter table first, Spec.Capabilities
// replaces it wholesale, and Spec.StructuredOutput is applied LAST so the
// field-level override wins even over an override that carries the field.
func TestCapabilities_SpecOverride_ComposesWithStructuredOutput(t *testing.T) {
	// The same four-provider table as the wholesale test: the composition
	// order (table -> Capabilities wholesale -> StructuredOutput last) is
	// contract surface on EVERY adapter, so skipping one in New must fail
	// here.
	providers := []struct {
		name  string
		spec  Spec
		model string
	}{
		{"anthropic", Spec{Type: TypeAnthropic}, "claude-opus-4-5"},
		{"openai", Spec{Type: TypeOpenAI}, "gpt-5"},
		{"openai-compatible", Spec{Type: TypeOpenAICompatible}, "llama3"},
		{"google", Spec{Type: TypeGoogle}, "gemini-2.5-pro"},
	}
	for _, tc := range providers {
		t.Run(tc.name+"/option wins over override", func(t *testing.T) {
			tc.spec.Capabilities = overrideEveryField() // StructuredOutput: false
			tc.spec.StructuredOutput = ptr(true)
			client, err := New(context.Background(), tc.spec, "test", tc.model, "k", Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			got := client.Capabilities()
			if !got.StructuredOutput {
				t.Error("Capabilities().StructuredOutput = false, want true (Spec.StructuredOutput applied last)")
			}
			// ...and the rest of the override still holds.
			if got.ContextWindow != 123_456 {
				t.Errorf("ContextWindow = %d, want 123456 (override must still apply)", got.ContextWindow)
			}
		})
		t.Run(tc.name+"/option can force it off", func(t *testing.T) {
			ov := overrideEveryField()
			ov.StructuredOutput = true
			tc.spec.Capabilities = ov
			tc.spec.StructuredOutput = ptr(false)
			client, err := New(context.Background(), tc.spec, "test", tc.model, "k", Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if client.Capabilities().StructuredOutput {
				t.Error("Capabilities().StructuredOutput = true, want false (explicit off wins)")
			}
		})
	}
}

// TestCapabilities_NilSpec_KeepsTableProfile is the regression guard: with
// Spec.Capabilities unset every provider keeps its model-table profile,
// including the unknown-model rule (ContextWindow 0) and the conservative
// openai-compatible defaults. The whole struct is compared.
func TestCapabilities_NilSpec_KeepsTableProfile(t *testing.T) {
	// anthropicProfile fills the shared Claude Messages API surface.
	anthropicProfile := func(window int, thinking bool) llmkit.Capabilities {
		return llmkit.Capabilities{
			ContextWindow: window, ParallelToolCalls: true, PromptCaching: true,
			StructuredOutput: true, Thinking: thinking, ToolChoice: true,
			Images: true, Documents: true, StopSequences: true, TopP: true,
			TopK: true, Seed: false,
		}
	}
	// openAIFirstParty fills the shared Chat Completions surface.
	openAIFirstParty := func(window int, parallel, caching, structured, toolChoice bool) llmkit.Capabilities {
		return llmkit.Capabilities{
			ContextWindow: window, ParallelToolCalls: parallel, PromptCaching: caching,
			StructuredOutput: structured, Thinking: false, ToolChoice: toolChoice,
			Images: true, Documents: true, StopSequences: true, TopP: true,
			TopK: false, Seed: true,
		}
	}
	googleProfile := func(window int, thinking, tools bool) llmkit.Capabilities {
		return llmkit.Capabilities{
			ContextWindow: window, ParallelToolCalls: tools, PromptCaching: true,
			StructuredOutput: true, Thinking: thinking, ToolChoice: tools,
			Images: true, Documents: true, StopSequences: true, TopP: true,
			TopK: true, Seed: true,
		}
	}
	compatible := llmkit.Capabilities{
		ParallelToolCalls: false, PromptCaching: false, StructuredOutput: false,
		Thinking: false, ToolChoice: true, Images: true, Documents: true,
		StopSequences: true, TopP: true, TopK: false, Seed: true,
	}

	cases := []struct {
		name  string
		spec  Spec
		model string
		want  llmkit.Capabilities
	}{
		{"anthropic table entry", Spec{Type: TypeAnthropic}, "claude-3-5-haiku-20241022", anthropicProfile(200_000, false)},
		{"anthropic unknown model", Spec{Type: TypeAnthropic}, "mystery-claude", anthropicProfile(0, true)},
		{"openai table entry", Spec{Type: TypeOpenAI}, "gpt-4", openAIFirstParty(8_192, false, false, false, true)},
		{"openai unknown model", Spec{Type: TypeOpenAI}, "mystery-gpt", openAIFirstParty(0, true, true, true, true)},
		{"openai-compatible", Spec{Type: TypeOpenAICompatible}, "llama3", compatible},
		{"google table entry", Spec{Type: TypeGoogle}, "gemini-2.0-flash-lite", googleProfile(1_048_576, false, false)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, err := New(context.Background(), tc.spec, "test", tc.model, "k", Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if got := client.Capabilities(); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Capabilities() =\n  %+v\nwant table profile\n  %+v", got, tc.want)
			}
		})
	}
}

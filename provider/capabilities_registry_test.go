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
func overrideEveryField() llmkit.Capabilities {
	return llmkit.Capabilities{
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
		{"openai-compatible", Spec{Type: TypeOpenAICompatible, BaseURL: "http://example.invalid"}, "llama3"},
		{"google", Spec{Type: TypeGoogle}, "gemini-2.5-pro"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.spec.Model = tc.model
			tc.spec.Secret = "k"
			tc.spec.Capabilities = func(llmkit.Capabilities) llmkit.Capabilities { return overrideEveryField() }
			client, err := New(context.Background(), tc.spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			want := overrideEveryField()
			if got := client.Capabilities(); !reflect.DeepEqual(got, want) {
				t.Errorf("Capabilities() =\n  %+v\nwant the Spec.Capabilities override wholesale:\n  %+v", got, want)
			}
		})
	}
}

// TestCapabilities_SpecOverride_ClosureSeesTableProfile pins the override
// contract: the closure receives the adapter's table-derived profile as
// input — not a zero Capabilities — and its return value is the effective
// profile, so a one-field flip keeps the rest of the table while a closure
// that ignores its input replaces the profile wholesale.
func TestCapabilities_SpecOverride_ClosureSeesTableProfile(t *testing.T) {
	cases := []struct {
		name  string
		spec  Spec
		model string
	}{
		{"anthropic", Spec{Type: TypeAnthropic}, "claude-opus-4-5"},
		{"openai", Spec{Type: TypeOpenAI}, "gpt-5"},
		{"openai-compatible", Spec{Type: TypeOpenAICompatible, BaseURL: "http://example.invalid"}, "llama3"},
		{"google", Spec{Type: TypeGoogle}, "gemini-2.5-pro"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.spec.Model = tc.model
			tc.spec.Secret = "k"
			base, err := New(context.Background(), tc.spec, Options{})
			if err != nil {
				t.Fatalf("New (no override): %v", err)
			}
			table := base.Capabilities()

			var saw llmkit.Capabilities
			tc.spec.Capabilities = func(c llmkit.Capabilities) llmkit.Capabilities {
				saw = c
				c.StructuredOutput = !c.StructuredOutput
				return c
			}
			client, err := New(context.Background(), tc.spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if !reflect.DeepEqual(saw, table) {
				t.Errorf("override input =\n  %+v\nwant the adapter table profile\n  %+v", saw, table)
			}
			got := client.Capabilities()
			if got.StructuredOutput == table.StructuredOutput {
				t.Errorf("Capabilities().StructuredOutput = %t, want the closure's flipped value (its return is the effective profile)", got.StructuredOutput)
			}
			if got.ContextWindow != table.ContextWindow {
				t.Errorf("ContextWindow = %d, want %d (a one-field flip must keep the rest of the table)", got.ContextWindow, table.ContextWindow)
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
		{"openai-compatible", Spec{Type: TypeOpenAICompatible, BaseURL: "http://example.invalid"}, "llama3", compatible},
		{"google table entry", Spec{Type: TypeGoogle}, "gemini-2.0-flash-lite", googleProfile(1_048_576, false, false)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.spec.Model = tc.model
			tc.spec.Secret = "k"
			client, err := New(context.Background(), tc.spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if got := client.Capabilities(); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Capabilities() =\n  %+v\nwant table profile\n  %+v", got, tc.want)
			}
		})
	}
}

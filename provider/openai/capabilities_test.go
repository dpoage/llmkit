package openai

import (
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// firstPartyWant builds the full expected profile for a first-party OpenAI
// model. Chat Completions behavior is shared across models (thinking is a
// coarse reasoning_effort dial, never a budget; no top_k; seed, images,
// documents, stop sequences and top_p always available); the variable fields
// are the arguments.
func firstPartyWant(window int, parallel, caching, structured, toolChoice bool) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     window,
		ParallelToolCalls: parallel,
		PromptCaching:     caching,
		StructuredOutput:  structured,
		Thinking:          false,
		ToolChoice:        toolChoice,
		Images:            true,
		Documents:         true,
		StopSequences:     true,
		TopP:              true,
		TopK:              false,
		Seed:              true,
	}
}

// TestOpenAICapabilities_PerModel walks the first-party model table: one
// representative per family, date-suffixed snapshots to pin LONGEST-prefix
// matching (o1-mini-2024-… must resolve to the o1-mini entry, not "o1"), and
// the unknown-model rule (ContextWindow 0, never the removed 128k fallback).
// The whole struct is compared so an entry that silently regresses any bool
// fails here.
func TestOpenAICapabilities_PerModel(t *testing.T) {
	cases := []struct {
		model string
		want  llmkit.Capabilities
	}{
		// gpt-5 family: 400k window, full modern feature set.
		{"gpt-5", firstPartyWant(400_000, true, true, true, true)},
		{"gpt-5-mini-2025-08-07", firstPartyWant(400_000, true, true, true, true)},
		// gpt-4.1 family: 1,047,576 window.
		{"gpt-4.1", firstPartyWant(1_047_576, true, true, true, true)},
		{"gpt-4.1-nano", firstPartyWant(1_047_576, true, true, true, true)},
		// gpt-4o family: 128k, prompt caching (post-Aug-2024 model).
		{"gpt-4o", firstPartyWant(128_000, true, true, true, true)},
		{"gpt-4o-mini-2024-07-18", firstPartyWant(128_000, true, true, true, true)},
		// gpt-4-turbo / gpt-4 / gpt-4-32k: pre-caching, no structured
		// outputs; only the 1106+ generations parallelize tool calls.
		{"gpt-4-turbo-2024-04-09", firstPartyWant(128_000, true, false, false, true)},
		{"gpt-4-32k", firstPartyWant(32_768, false, false, false, true)},
		{"gpt-4", firstPartyWant(8_192, false, false, false, true)},
		{"gpt-3.5-turbo", firstPartyWant(16_385, true, false, false, true)},
		// o1-mini / o1-preview: no function calling at all, so ToolChoice
		// (and parallel tool calls) are false despite the 128k window.
		{"o1-mini", firstPartyWant(128_000, false, true, false, false)},
		{"o1-mini-2024-09-12", firstPartyWant(128_000, false, true, false, false)},
		{"o1-preview", firstPartyWant(128_000, false, true, false, false)},
		// Reasoning models with the full tool surface.
		{"o1", firstPartyWant(200_000, true, true, true, true)},
		{"o3-mini", firstPartyWant(200_000, true, true, true, true)},
		{"o4-mini", firstPartyWant(200_000, true, true, true, true)},
		// Unknown model: the window is UNKNOWN (0) — the old 128k fallback
		// overstated gpt-4's 8k window — while the bools keep the API-level
		// defaults (the server validates model-specific features).
		{"unknown-model-xyz", firstPartyWant(0, true, true, true, true)},
	}
	for _, tc := range cases {
		got := openAICapabilities(tc.model)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("openAICapabilities(%q) =\n  %+v\nwant\n  %+v", tc.model, got, tc.want)
		}
	}
}

// TestOpenAICompatibleCapabilities_Profile pins the conservative profile for
// arbitrary OpenAI-compatible endpoints: window unknown (0), serialized tool
// calls, no caching, structured output off unless overridden.
func TestOpenAICompatibleCapabilities_Profile(t *testing.T) {
	for _, model := range []string{"llama3", "MiniMax-M3", "some-future-model"} {
		got := openAICompatibleCapabilities(model)
		want := llmkit.Capabilities{
			ParallelToolCalls: false,
			PromptCaching:     false,
			StructuredOutput:  false,
			Thinking:          false,
			ToolChoice:        true,
			Images:            true,
			Documents:         true,
			StopSequences:     true,
			TopP:              true,
			TopK:              false,
			Seed:              true,
			// ContextWindow 0 = unknown; never fabricated for endpoints we
			// cannot introspect.
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("openAICompatibleCapabilities(%q) =\n  %+v\nwant\n  %+v", model, got, want)
		}
	}
}

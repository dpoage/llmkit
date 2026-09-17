package anthropic

import (
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// anthropicWant builds the full expected profile for a Claude model. The
// Messages API surface is shared across the 3+ families (parallel tool
// calls, prompt caching, synthetic-tool structured output, tool choice,
// images, documents, stop sequences, top_p, top_k; no seed); per-model the
// table pins only the window and extended thinking (Claude 3.7+).
func anthropicWant(thinking bool) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     200_000,
		ParallelToolCalls: true,
		PromptCaching:     true,
		StructuredOutput:  true,
		Thinking:          thinking,
		ToolChoice:        true,
		Images:            true,
		Documents:         true,
		StopSequences:     true,
		TopP:              true,
		TopK:              true,
		Seed:              false,
	}
}

// TestAnthropicCapabilities_PerModel walks the model table: one
// representative per family (claude-3 through 4.5), date-suffixed snapshots
// to pin prefix matching against real model IDs, and the unknown-model rule
// (ContextWindow 0, never a fabricated number). The whole struct is
// compared so an entry that silently regresses any bool fails here.
func TestAnthropicCapabilities_PerModel(t *testing.T) {
	cases := []struct {
		model    string
		thinking bool
	}{
		// 4.x / 4.5 families: extended thinking everywhere.
		{"claude-opus-4-5", true},
		{"claude-sonnet-4-5-20250929", true},
		{"claude-haiku-4-5", true},
		{"claude-opus-4-1-20250805", true},
		{"claude-opus-4-20250514", true},
		{"claude-sonnet-4-20250514", true},
		// 3.7: extended thinking's first family.
		{"claude-3-7-sonnet-20250219", true},
		// 3.5 / 3: no thinkingConfig on the wire.
		{"claude-3-5-sonnet-20241022", false},
		{"claude-3-5-haiku-20241022", false},
		{"claude-3-opus-20240229", false},
		{"claude-3-sonnet-20240229", false},
		{"claude-3-haiku-20240307", false},
	}
	for _, tc := range cases {
		got := anthropicCapabilities(tc.model)
		want := anthropicWant(tc.thinking)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("anthropicCapabilities(%q) =\n  %+v\nwant\n  %+v", tc.model, got, want)
		}
	}

	// Unknown model: window UNKNOWN (0) with the API-level feature defaults;
	// the server is the final validator for unrecognized names.
	got := anthropicCapabilities("unknown-model-xyz")
	want := anthropicWant(true)
	want.ContextWindow = 0
	if !reflect.DeepEqual(got, want) {
		t.Errorf("anthropicCapabilities(unknown) =\n  %+v\nwant\n  %+v", got, want)
	}
}

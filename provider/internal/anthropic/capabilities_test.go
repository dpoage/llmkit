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
func anthropicWant(window int, thinking bool) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     window,
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
// representative per verified generation, date-suffixed snapshots to pin
// segment-boundary matching against real model IDs, and the unknown-model
// rule (ContextWindow 0, never a fabricated number). The whole struct is
// compared so an entry that silently regresses any bool fails here.
func TestAnthropicCapabilities_PerModel(t *testing.T) {
	cases := []struct {
		model string
		want  llmkit.Capabilities
	}{
		// 1M-window generations (Opus 4.6+, Sonnet 4.6).
		{"claude-opus-4-8", anthropicWant(1_000_000, true)},
		{"claude-opus-4-7", anthropicWant(1_000_000, true)},
		{"claude-opus-4-6", anthropicWant(1_000_000, true)},
		{"claude-sonnet-4-6", anthropicWant(1_000_000, true)},
		// First 4.0-generation snapshots (May 2025): 200k windows,
		// extended thinking (vendor models overview page).
		{"claude-opus-4-20250514", anthropicWant(200_000, true)},
		{"claude-sonnet-4-20250514", anthropicWant(200_000, true)},
		// 200k generations (1M only behind the context-1m beta header this
		// adapter never sends).
		{"claude-opus-4-5", anthropicWant(200_000, true)},
		{"claude-opus-4-5-20251101", anthropicWant(200_000, true)},
		{"claude-opus-4-1-20250805", anthropicWant(200_000, true)},
		{"claude-sonnet-4-5-20250929", anthropicWant(200_000, true)},
		{"claude-haiku-4-5-20251001", anthropicWant(200_000, true)},
		// 3.x: extended thinking starts with 3.7.
		{"claude-3-7-sonnet-20250219", anthropicWant(200_000, true)},
		{"claude-3-5-sonnet-20241022", anthropicWant(200_000, false)},
		{"claude-3-5-haiku-20241022", anthropicWant(200_000, false)},
		{"claude-3-opus-20240229", anthropicWant(200_000, false)},
		{"claude-3-sonnet-20240229", anthropicWant(200_000, false)},
		{"claude-3-haiku-20240307", anthropicWant(200_000, false)},

		// Unknown models: window UNKNOWN (0) with the API-level feature
		// defaults. claude-opus-4-9 is the oracle-mandated probe — an
		// unverified future generation must NOT inherit 200k or 1M from any
		// listed key. claude-3-5-sonnetish extends a key mid-token and is a
		// different (nonexistent) name, not a snapshot. The retired bare
		// 4.0 aliases (claude-opus-4 / claude-sonnet-4) are deliberately
		// unlisted — the API rejects those IDs outright — while their
		// date-stamped snapshots are listed above.
		{"claude-opus-4-9", anthropicWant(0, true)},
		{"claude-sonnet-4-7", anthropicWant(0, true)},
		{"claude-3-5-sonnetish", anthropicWant(0, true)},
		{"claude-opus-4", anthropicWant(0, true)},
		{"unknown-model-xyz", anthropicWant(0, true)},
	}
	for _, tc := range cases {
		got := anthropicCapabilities(tc.model)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("anthropicCapabilities(%q) =\n  %+v\nwant\n  %+v", tc.model, got, tc.want)
		}
	}
}

package google

import (
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// googleWant builds the full expected profile for a Gemini model. Image and
// document parts, stop sequences, top_p, top_k and seed hold for every
// entry; window, thinking, tools, structured output and caching are stored
// per entry because the 2.5 flash non-text variants genuinely differ.
func googleWant(window int, thinking, tools, structured, caching bool) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     window,
		ParallelToolCalls: tools,
		PromptCaching:     caching,
		StructuredOutput:  structured,
		Thinking:          thinking,
		ToolChoice:        tools,
		Images:            true,
		Documents:         true,
		StopSequences:     true,
		TopP:              true,
		TopK:              true,
		Seed:              true,
	}
}

// TestGoogleCapabilities_PerModel walks the model table: one representative
// per generation, preview/date-suffixed snapshots to pin segment-boundary
// matching, the non-text 2.5 flash variants (facts from each variant's own
// spec page), and the unknown-model rule (ContextWindow 0, never a
// fabricated number). The whole struct is compared so an entry that
// silently regresses any bool fails here.
func TestGoogleCapabilities_PerModel(t *testing.T) {
	cases := []struct {
		model string
		want  llmkit.Capabilities
	}{
		// 2.5 text family: 1M window, thinkingConfig, full tool surface.
		{"gemini-2.5-pro", googleWant(1_048_576, true, true, true, true)},
		{"gemini-2.5-pro-preview-06-05", googleWant(1_048_576, true, true, true, true)},
		{"gemini-2.5-flash", googleWant(1_048_576, true, true, true, true)},
		{"gemini-2.5-flash-lite", googleWant(1_048_576, true, true, true, true)},
		// 2.0: no thinkingConfig. flash-lite launched WITHOUT function
		// calling — and because "gemini-2.0-flash-lite" is the LONGER key,
		// the -lite suffix must win over the bare "gemini-2.0-flash" entry:
		// a first/shortest-match lookup would wrongly report ToolChoice=true
		// here.
		{"gemini-2.0-flash", googleWant(1_048_576, false, true, true, true)},
		{"gemini-2.0-flash-lite-001", googleWant(1_048_576, false, false, true, true)},
		// 1.5: pro shipped the 2M window (002 refresh); flash stayed at 1M.
		{"gemini-1.5-pro-002", googleWant(2_097_152, false, true, true, true)},
		{"gemini-1.5-flash", googleWant(1_048_576, false, true, true, true)},

		// Non-text variants of the 2.5 flash generation, pinned from each
		// variant's own spec page (the text entry's Thinking=true is
		// load-bearing: it gates thinkingConfig on the wire, so swallowing
		// these would send an unsupported parameter and misstate the
		// window).
		{"gemini-2.5-flash-image", googleWant(65_536, false, false, false, false)},
		{"gemini-2.5-flash-preview-tts", googleWant(8_192, false, false, false, false)},
		// Audio-dialog generation: function calling and thinking supported,
		// structured output and caching not.
		{"gemini-2.5-flash-native-audio-preview-12-2025", googleWant(131_072, true, true, false, false)},

		// Unknown models: window UNKNOWN (0) with the API-level feature
		// defaults. gemini-2.5-flashy extends a listed key mid-token and is
		// a different (nonexistent) name, not a snapshot — the
		// segment-boundary matcher must reject it.
		{"gemini-2.5-flashy", googleWant(0, true, true, true, true)},
		{"unknown-model-xyz", googleWant(0, true, true, true, true)},
	}
	for _, tc := range cases {
		got := googleCapabilities(tc.model)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("googleCapabilities(%q) =\n  %+v\nwant\n  %+v", tc.model, got, tc.want)
		}
	}
}

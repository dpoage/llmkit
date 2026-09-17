package google

import (
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// googleWant builds the full expected profile for a Gemini model. Structured
// output, prompt caching (implicit server-side on 2.x), image/document
// parts, stop sequences, top_p, top_k and seed hold for every entry; the
// variable fields are the window, thinkingConfig (2.5 family only) and
// function calling (absent on gemini-2.0-flash-lite).
func googleWant(window int, thinking, tools bool) llmkit.Capabilities {
	return llmkit.Capabilities{
		ContextWindow:     window,
		ParallelToolCalls: tools,
		PromptCaching:     true,
		StructuredOutput:  true,
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
// per generation, preview/date-suffixed snapshots to pin LONGEST-prefix
// matching, and the unknown-model rule (ContextWindow 0, never a fabricated
// number). The whole struct is compared so an entry that silently regresses
// any bool fails here.
func TestGoogleCapabilities_PerModel(t *testing.T) {
	cases := []struct {
		model string
		want  llmkit.Capabilities
	}{
		// 2.5 family: 1M window, thinkingConfig, full tool surface.
		{"gemini-2.5-pro", googleWant(1_048_576, true, true)},
		{"gemini-2.5-pro-preview-06-05", googleWant(1_048_576, true, true)},
		{"gemini-2.5-flash", googleWant(1_048_576, true, true)},
		{"gemini-2.5-flash-lite", googleWant(1_048_576, true, true)},
		// 2.0: no thinkingConfig. flash-lite launched WITHOUT function
		// calling — and because "gemini-2.0-flash-lite" is the LONGER
		// prefix, the -lite suffix must win over the bare "gemini-2.0-flash"
		// entry: a first/shortest-match lookup would wrongly report
		// ToolChoice=true here.
		{"gemini-2.0-flash", googleWant(1_048_576, false, true)},
		{"gemini-2.0-flash-lite-001", googleWant(1_048_576, false, false)},
		// 1.5: pro shipped the 2M window (002 refresh); flash stayed at 1M.
		{"gemini-1.5-pro-002", googleWant(2_097_152, false, true)},
		{"gemini-1.5-flash", googleWant(1_048_576, false, true)},
	}
	for _, tc := range cases {
		got := googleCapabilities(tc.model)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("googleCapabilities(%q) =\n  %+v\nwant\n  %+v", tc.model, got, tc.want)
		}
	}

	// Unknown model: window UNKNOWN (0) with the API-level feature defaults;
	// the server is the final validator for unrecognized names.
	got := googleCapabilities("unknown-model-xyz")
	want := googleWant(0, true, true)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("googleCapabilities(unknown) =\n  %+v\nwant\n  %+v", got, want)
	}
}

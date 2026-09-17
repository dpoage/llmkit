package openai

import "testing"

// TestOpenAIContextWindow_PerModel asserts that openAIContextWindow returns
// distinct values for models with different context sizes, and that unknown
// models get the 128k default. This proves the capability constructors consult
// the model parameter (Part B — scaleFinderForContext fires correctly for small
// models like gpt-4 which is 8k, not 128k).
func TestOpenAIContextWindow_PerModel(t *testing.T) {
	cases := []struct {
		model string
		want  int
	}{
		{"gpt-4", 8_192},
		{"gpt-4-32k", 32_768},
		{"gpt-4o", 128_000},
		{"o1", 200_000},
		{"gpt-4.1", 1_047_576},
		{"unknown-model-xyz", 128_000}, // default fallback
	}
	for _, tc := range cases {
		caps := openAICapabilities(tc.model)
		if caps.ContextWindow != tc.want {
			t.Errorf("openAICapabilities(%q).ContextWindow = %d, want %d", tc.model, caps.ContextWindow, tc.want)
		}
	}
}

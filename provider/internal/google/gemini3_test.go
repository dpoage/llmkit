package google

import "testing"

func TestGoogleCapabilities_ReasoningEffortIsGemini3Only(t *testing.T) {
	cases := map[string]bool{
		"gemini-3":               true,
		"gemini-3-flash-preview": true,
		"gemini-3-pro-preview":   true,
		"gemini-3.1-pro-preview": true,
		"gemini-3.5-flash":       true,
		"gemini-30-x":            false,
		"gemini-3x":              false,
		"gemini-2.5-pro":         false,
		"gemini-2.5-flash":       false,
		"gemini-2.0-flash":       false,
		"gemini-1.5-pro":         false,
		"gemini-unknown":         false,
		"models/gemini-3-flash":  false,
		"":                       false,
	}
	for model, want := range cases {
		if got := googleCapabilities(model).ReasoningEffort; got != want {
			t.Errorf("googleCapabilities(%q).ReasoningEffort = %v, want %v", model, got, want)
		}
	}
}

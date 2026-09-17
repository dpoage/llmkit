package llmkit

import "testing"

// TestDefaultMaxTokens_Value pins the documented default to a literal so a
// change to DefaultMaxTokens is a deliberate, reviewable decision — every
// adapter sends this exact number on the wire when Request.MaxTokens is
// unset (the uniformity itself is asserted against the constant by
// TestConformance_DefaultMaxTokens in the provider package).
func TestDefaultMaxTokens_Value(t *testing.T) {
	if DefaultMaxTokens != 4096 {
		t.Errorf("DefaultMaxTokens = %d, want 4096 (change deliberately, and with it the documented contract on Request.MaxTokens)", DefaultMaxTokens)
	}
}

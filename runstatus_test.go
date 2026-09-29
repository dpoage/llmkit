package llmkit_test

import (
	"testing"

	"github.com/dpoage/llmkit"
)

// TestRunStatus_WireLiterals pins the six RunStatus values as the exact
// strings a store keys on. The other status tests round-trip the constants
// themselves, so renaming a literal would pass them and silently fork every
// recorded run.
func TestRunStatus_WireLiterals(t *testing.T) {
	for _, tc := range []struct {
		status llmkit.RunStatus
		want   string
	}{
		{llmkit.RunCompleted, "completed"},
		{llmkit.RunIncomplete, "incomplete"},
		{llmkit.RunRefused, "refused"},
		{llmkit.RunFailed, "failed"},
		{llmkit.RunCanceled, "canceled"},
		{llmkit.RunPanicked, "panicked"},
	} {
		if string(tc.status) != tc.want {
			t.Errorf("RunStatus %q, want the wire literal %q", tc.status, tc.want)
		}
	}
}

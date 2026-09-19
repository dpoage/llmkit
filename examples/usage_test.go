package examples_test

// Hermetic execution test for the example programs: with the LLMKIT_*
// environment unset, every example must print its usage message and exit 1
// WITHOUT touching the network. This runs in plain `go test ./...` — no
// build tag, no credentials.
import (
	"strings"
	"testing"
)

func TestExamplesUsageExitWithoutEnv(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	root := moduleRoot(t)
	tmp := t.TempDir()

	for _, name := range []string{"basic", "agent", "structured", "chat"} {
		name := name
		t.Run(name, func(t *testing.T) {
			bin := buildExample(t, root, tmp, name)
			// Exactly the documented contract: NO LLMKIT_* variables at all.
			code, stdout, stderr := runExample(t, bin, nil, map[string]string{})
			if code != 1 {
				t.Fatalf("exit %d, want 1\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
			}
			// The usage message may land on stderr (the error path) or
			// stdout; it must name the LLMKIT_ variables it needs.
			combined := stdout + stderr
			if !strings.Contains(combined, "LLMKIT_PROVIDER") {
				t.Errorf("usage message does not name LLMKIT_PROVIDER:\n%s", combined)
			}
		})
	}
}

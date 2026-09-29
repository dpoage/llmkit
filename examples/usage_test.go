package examples_test

// Hermetic execution test for the example programs: with the LLMKIT_*
// environment unset (and, for replay, no arguments), the examples below
// print their usage message and exit 1 without touching the network. This
// runs in plain `go test ./...` — no build tag, no credentials.
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

	// want is the marker each usage message must carry. The four
	// provider-driven examples read the LLMKIT_* variables and name them;
	// replay reads none and names its arguments instead.
	for _, ex := range []struct{ name, want string }{
		{"basic", "LLMKIT_PROVIDER"},
		{"agent", "LLMKIT_PROVIDER"},
		{"structured", "LLMKIT_PROVIDER"},
		{"chat", "LLMKIT_PROVIDER"},
		{"replay", "<record-dir> <run-id>"},
	} {
		ex := ex
		t.Run(ex.name, func(t *testing.T) {
			bin := buildExample(t, root, tmp, ex.name)
			// Exactly the documented contract: NO LLMKIT_* variables at all,
			// no arguments.
			code, stdout, stderr := runExample(t, bin, nil, map[string]string{})
			if code != 1 {
				t.Fatalf("exit %d, want 1\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
			}
			// The usage message may land on stderr (the error path) or
			// stdout.
			combined := stdout + stderr
			if !strings.Contains(combined, ex.want) {
				t.Errorf("usage message does not name %q:\n%s", ex.want, combined)
			}
		})
	}
}

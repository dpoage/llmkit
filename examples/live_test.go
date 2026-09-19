//go:build live

// Live execution tests for the example programs: build each example once,
// run it against the compat lane exactly the way a user would (LLMKIT_* env
// vars, no special flags), and assert exit 0 plus the output marker the
// program actually prints. The examples' main.go files are contract
// checks for the public API — this suite proves they keep RUNNING against a
// real backend.
//
// Skips unless LLMKIT_LIVE_COMPAT_API_KEY / _BASE_URL / _MODEL are all set.
// Run with
//
//	go test -tags live -count=1 ./examples/... -v
package examples_test

import (
	"os"
	"strings"
	"testing"

	"github.com/dpoage/llmkit/internal/livetest"
)

func TestMain(m *testing.M) {
	code := m.Run()
	livetest.DefaultTally().PrintSummary()
	os.Exit(code)
}

// liveEnv maps the compat lane onto the LLMKIT_* variables every example
// reads.
func liveEnv(t *testing.T) map[string]string {
	t.Helper()
	sess := livetest.Resolve(t, "compat")
	return map[string]string{
		"LLMKIT_PROVIDER": "openai-compatible",
		"LLMKIT_MODEL":    sess.Model,
		"LLMKIT_BASE_URL": sess.BaseURL,
		"LLMKIT_API_KEY":  sess.Key,
	}
}

func TestLiveExamplesRun(t *testing.T) {
	env := liveEnv(t)
	root := moduleRoot(t)
	tmp := t.TempDir()

	t.Run("basic", func(t *testing.T) {
		bin := buildExample(t, root, tmp, "basic")
		code, stdout, stderr := runExample(t, bin, nil, env)
		if code != 0 {
			t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		if !strings.Contains(stdout, "usage:") || !strings.Contains(stdout, "stop reason:") {
			t.Errorf("basic did not print its normalized response summary:\n%s", stdout)
		}
	})

	t.Run("agent", func(t *testing.T) {
		bin := buildExample(t, root, tmp, "agent")
		code, stdout, stderr := runExample(t, bin, nil, env)
		if code != 0 {
			t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		if !strings.Contains(stdout, "final:") {
			t.Errorf("agent did not print its final answer:\n%s", stdout)
		}
	})

	t.Run("structured", func(t *testing.T) {
		bin := buildExample(t, root, tmp, "structured")
		code, stdout, stderr := runExample(t, bin, nil, env)
		if code != 0 {
			t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		if !strings.Contains(stdout, "parsed ok:") || !strings.Contains(stdout, "{") {
			t.Errorf("structured did not print a schema-valid answer:\n%s", stdout)
		}
	})

	t.Run("chat", func(t *testing.T) {
		bin := buildExample(t, root, tmp, "chat")
		// Two conversational turns, then EOF: the REPL must answer both and
		// exit cleanly.
		code, stdout, stderr := runExample(t, bin, []string{
			"What is 2+2? Answer with just the number.",
			"And what is 3+3? Answer with just the number.",
		}, env)
		if code != 0 {
			t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		if got := strings.Count(stdout, "assistant>"); got != 2 {
			t.Errorf("chat printed %d assistant replies, want 2:\n%s", got, stdout)
		}
	})
}

package examples_test

// Shared build/run helpers for the example execution tests. No build tag:
// the hermetic usage-exit test (usage_test.go) and the live-tagged
// execution test (live_test.go) both use them.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// moduleRoot walks up from the test's working directory to the directory
// containing go.mod.
func moduleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found above the test directory")
		}
		dir = parent
	}
}

// buildExample compiles one example into dir and returns the binary path.
func buildExample(t *testing.T, root, dir, name string) string {
	t.Helper()
	bin := filepath.Join(dir, name)
	cmd := exec.Command("go", "build", "-o", bin, "./examples/"+name)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build ./examples/%s: %v\n%s", name, err, out)
	}
	return bin
}

// runExample executes bin with exactly the given stdin and a MINIMAL
// environment (the caller's mapping plus PATH), returning its exit code,
// stdout, and stderr. A minimal env keeps the child from inheriting ambient
// LLMKIT_* values the test did not intend to set.
func runExample(t *testing.T, bin string, stdin []string, env map[string]string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(bin)
	if len(stdin) > 0 {
		cmd.Stdin = strings.NewReader(strings.Join(stdin, "\n") + "\n")
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	fullEnv := []string{"PATH=" + os.Getenv("PATH")}
	for k, v := range env {
		fullEnv = append(fullEnv, k+"="+v)
	}
	cmd.Env = fullEnv
	err := cmd.Run()
	code := 0
	if err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("run %s: %v\nstdout:\n%s\nstderr:\n%s", bin, err, stdout.String(), stderr.String())
		}
		code = exitErr.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

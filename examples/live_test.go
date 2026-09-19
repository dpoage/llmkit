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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit/internal/livetest"
)

func TestMain(m *testing.M) {
	code := m.Run()
	livetest.DefaultTally().PrintSummary()
	// The example binaries are separate processes: their vendor spend is
	// visible in their own output, not in this binary's tally.
	fmt.Println("LIVE_TOKENS note=child-process spend not tallied")
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
		// Two conversational turns, paced: line 2 is written only after
		// line 1's assistant reply is on stdout, so each line runs as its
		// own task and the REPL answers both and exits cleanly on EOF.
		code, stdout, stderr := runChatPaced(t, bin, env,
			"What is 2+2? Answer with just the number.",
			"And what is 3+3? Answer with just the number.",
		)
		if code != 0 {
			t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		if got := strings.Count(stdout, "assistant>"); got != 2 {
			t.Errorf("chat printed %d assistant replies, want 2:\n%s", got, stdout)
		}
	})

	t.Run("chat-steer", func(t *testing.T) {
		bin := buildExample(t, root, tmp, "chat")
		// Both lines fed at once: the second arrives while the first run is
		// in flight and becomes mid-run steering, so one run answers both
		// questions in a single reply.
		code, stdout, stderr := runExample(t, bin, []string{
			"What is 2+2? Answer with just the number.",
			"And what is 3+3? Answer with just the number.",
		}, env)
		if code != 0 {
			t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		if got := strings.Count(stdout, "assistant>"); got != 1 {
			t.Errorf("chat printed %d assistant replies, want 1 (the second line steers the first run):\n%s", got, stdout)
		}
		for _, want := range []string{"4", "6"} {
			if !strings.Contains(stdout, want) {
				t.Errorf("merged reply lost the answer %q:\n%s", want, stdout)
			}
		}
	})
}

// runChatPaced runs the chat REPL with stdin on a pipe and paces line2
// behind line1's reply: line2 is written only after the first "assistant>"
// reply is observed on stdout, so the two lines run as two separate tasks
// instead of the second steering the first run. stdin closes after line2,
// so the REPL exits on EOF. A child that produces no reply fails the test
// instead of hanging it.
func runChatPaced(t *testing.T, bin string, env map[string]string, line1, line2 string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(bin)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("stdin pipe: %v", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		t.Fatalf("stderr pipe: %v", err)
	}
	cmd.Env = append([]string{"PATH=" + os.Getenv("PATH")}, func() (out []string) {
		for k, v := range env {
			out = append(out, k+"="+v)
		}
		return out
	}()...)

	var stdoutBuf, stderrBuf bytes.Buffer
	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		_, _ = io.Copy(&stderrBuf, stderrPipe)
	}()
	// A stuck child (no reply ever printed) fails the test instead of
	// hanging it.
	timer := time.AfterFunc(2*time.Minute, func() { _ = cmd.Process.Kill() })
	defer timer.Stop()
	if err := cmd.Start(); err != nil {
		t.Fatalf("start %s: %v", bin, err)
	}

	writeLine := func(s string) {
		if _, werr := io.WriteString(stdin, s+"\n"); werr != nil {
			t.Fatalf("write %q to stdin: %v", s, werr)
		}
	}
	writeLine(line1)
	// The REPL prints "you> " only when idle, so the second line goes in
	// once a reply has appeared AND the idle prompt follows it; pacing on
	// the reply alone would send line 2 while the first run is still
	// streaming and turn it into mid-run steering.
	reader := bufio.NewReader(stdoutPipe)
	sawReply := false
	for {
		chunk, rerr := reader.ReadString('>')
		stdoutBuf.WriteString(chunk)
		if rerr != nil {
			break
		}
		out := stdoutBuf.String()
		if i := strings.Index(out, "assistant>"); i >= 0 && strings.Contains(out[i+len("assistant>"):], "you>") {
			sawReply = true
			break
		}
	}
	if !sawReply {
		<-stderrDone
		_ = cmd.Wait()
		t.Fatalf("chat produced no assistant reply for the first line\nstdout:\n%s\nstderr:\n%s",
			stdoutBuf.String(), stderrBuf.String())
	}

	writeLine(line2)
	if err := stdin.Close(); err != nil {
		t.Fatalf("close stdin: %v", err)
	}
	_, _ = io.Copy(&stdoutBuf, reader)
	<-stderrDone
	err = cmd.Wait()
	code := 0
	if err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("run %s: %v\nstdout:\n%s\nstderr:\n%s", bin, err, stdoutBuf.String(), stderrBuf.String())
		}
		code = exitErr.ExitCode()
	}
	return code, stdoutBuf.String(), stderrBuf.String()
}

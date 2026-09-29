package examples_test

// Hermetic record → replay test for examples/agent and examples/replay: the
// agent binary records a run through agent.JSONL against a scripted local
// openai-compatible server, and the replay binary replays that record with
// no server and no LLMKIT_* environment. No build tag, no credentials, no
// network beyond the httptest server: it runs in plain `go test ./...`.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// scriptedAnswer is the final text the fake server returns after the tool
// turn; the replay binary must print exactly it on its "final:" line.
const scriptedAnswer = "41 plus 58 is 99."

// fakeCompat starts a fake openai-compatible chat-completions server. Each
// request is answered with an assistant tool call to "add"; when
// toolEveryTurn is false, a request that already carries a tool result is
// answered with scriptedAnswer instead. The returned counter is the number
// of completions served.
func fakeCompat(t *testing.T, toolEveryTurn bool) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var served atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Messages []struct {
				Role string `json:"role"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		hasToolResult := false
		for _, m := range req.Messages {
			if m.Role == "tool" {
				hasToolResult = true
			}
		}
		n := served.Add(1)
		message := fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":"call_%d","type":"function","function":{"name":"add","arguments":"{\"a\":41,\"b\":58}"}}]}`, n)
		finish := "tool_calls"
		if hasToolResult && !toolEveryTurn {
			b, _ := json.Marshal(scriptedAnswer)
			message = fmt.Sprintf(`{"role":"assistant","content":%s}`, b)
			finish = "stop"
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-%d","object":"chat.completion","model":"m","choices":[{"index":0,"message":%s,"finish_reason":%q}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			n, message, finish)
	}))
	t.Cleanup(srv.Close)
	return srv, &served
}

// runExampleArgs is runExample with arguments and a deadline, so a hung
// child fails the test instead of the run.
func runExampleArgs(t *testing.T, bin string, args []string, env map[string]string) (int, string, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	err := cmd.Run()
	if err == nil {
		return 0, stdout.String(), stderr.String()
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok || ctx.Err() != nil {
		t.Fatalf("run %s %v: %v\nstdout:\n%s\nstderr:\n%s", bin, args, err, stdout.String(), stderr.String())
	}
	return exitErr.ExitCode(), stdout.String(), stderr.String()
}

var (
	runIDLine = regexp.MustCompile(`(?m)^run id: +(\S+)$`)
	finalLine = regexp.MustCompile(`(?m)^final: +(.*)$`)
)

// recordRun runs the agent binary against srv with --record and returns the
// record directory and the run id it printed. The record file must exist
// under that id.
func recordRun(t *testing.T, agentBin string, srv *httptest.Server) (dir, runID string) {
	t.Helper()
	dir = filepath.Join(t.TempDir(), "records")
	code, stdout, stderr := runExampleArgs(t, agentBin, []string{"--record", dir}, map[string]string{
		"LLMKIT_PROVIDER": "openai-compatible",
		"LLMKIT_MODEL":    "m",
		"LLMKIT_API_KEY":  "k",
		"LLMKIT_BASE_URL": srv.URL + "/v1",
	})
	if code != 0 {
		t.Fatalf("agent exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	m := runIDLine.FindStringSubmatch(stdout)
	if m == nil {
		t.Fatalf("agent printed no run id line:\n%s", stdout)
	}
	runID = m[1]
	if _, err := os.Stat(filepath.Join(dir, runID+".jsonl")); err != nil {
		t.Fatalf("record for the printed run id: %v\nstdout:\n%s\nstderr:\n%s", err, stdout, stderr)
	}
	return dir, runID
}

// editLastOfKind copies the record dir/<id>.jsonl into a new directory with
// its lines passed through edit, which receives the index of the record's
// last event of the given kind, and returns the new directory.
func editLastOfKind(t *testing.T, dir, id, kind string, edit func(lines []string, last int) []string) string {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, id+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	last := -1
	for i, l := range lines {
		var ev struct {
			Kind string `json:"kind"`
		}
		if err := json.Unmarshal([]byte(l), &ev); err != nil {
			t.Fatalf("record line %d: %v", i+1, err)
		}
		if ev.Kind == kind {
			last = i
		}
	}
	if last < 0 {
		t.Fatalf("record has no %s line", kind)
	}
	lines = edit(lines, last)
	out := filepath.Join(t.TempDir(), "tampered")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, id+".jsonl"), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return out
}

// dropLastCompletion copies the record without its last completion line.
func dropLastCompletion(t *testing.T, dir, id string) string {
	t.Helper()
	return editLastOfKind(t, dir, id, "completion", func(lines []string, last int) []string {
		return append(lines[:last], lines[last+1:]...)
	})
}

// tamperLastToolArgs copies the record with the arguments of its last
// tool_run event replaced by args.
func tamperLastToolArgs(t *testing.T, dir, id, args string) string {
	t.Helper()
	return editLastOfKind(t, dir, id, "tool_run", func(lines []string, last int) []string {
		dec := json.NewDecoder(strings.NewReader(lines[last]))
		dec.UseNumber()
		var ev map[string]any
		if err := dec.Decode(&ev); err != nil {
			t.Fatal(err)
		}
		run, _ := ev["tool_run"].(map[string]any)
		call, _ := run["call"].(map[string]any)
		if call == nil {
			t.Fatalf("tool_run line has no call: %s", lines[last])
		}
		call["arguments"] = json.RawMessage(args)
		b, err := json.Marshal(ev)
		if err != nil {
			t.Fatal(err)
		}
		lines[last] = string(b)
		return lines
	})
}

func TestExamplesRecordReplay(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	root := moduleRoot(t)
	tmp := t.TempDir()
	agentBin := buildExample(t, root, tmp, "agent")
	replayBin := buildExample(t, root, tmp, "replay")

	t.Run("replay reproduces the recorded answer", func(t *testing.T) {
		srv, served := fakeCompat(t, false)
		dir, id := recordRun(t, agentBin, srv)
		if got := served.Load(); got != 2 {
			t.Fatalf("recording run made %d completions, want 2 (a tool call, then the answer)", got)
		}
		// Replay with the server gone and no LLMKIT_* variables: it must
		// need neither.
		srv.Close()
		code, stdout, stderr := runExampleArgs(t, replayBin, []string{dir, id}, nil)
		if code != 0 {
			t.Fatalf("replay exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
		}
		m := finalLine.FindStringSubmatch(stdout)
		if m == nil || m[1] != scriptedAnswer {
			t.Fatalf("replay \"final:\" line = %q, want %q\nstdout:\n%s", m, scriptedAnswer, stdout)
		}
	})

	t.Run("replay of a record missing its last completion fails", func(t *testing.T) {
		srv, _ := fakeCompat(t, false)
		dir, id := recordRun(t, agentBin, srv)
		tampered := dropLastCompletion(t, dir, id)
		code, stdout, stderr := runExampleArgs(t, replayBin, []string{tampered, id}, nil)
		if code == 0 {
			t.Fatalf("replay of a truncated record exited 0\nstdout:\n%s\nstderr:\n%s", stdout, stderr)
		}
	})

	t.Run("replay of a step-capped record fails", func(t *testing.T) {
		srv, _ := fakeCompat(t, true)
		dir, id := recordRun(t, agentBin, srv)
		code, stdout, stderr := runExampleArgs(t, replayBin, []string{dir, id}, nil)
		if code == 0 {
			t.Fatalf("replay of a step-capped record exited 0\nstdout:\n%s\nstderr:\n%s", stdout, stderr)
		}
	})

	t.Run("replay of a step-capped record with tampered final tool arguments names the divergence", func(t *testing.T) {
		srv, _ := fakeCompat(t, true)
		dir, id := recordRun(t, agentBin, srv)
		tampered := tamperLastToolArgs(t, dir, id, `{"a":1,"b":2}`)
		code, stdout, stderr := runExampleArgs(t, replayBin, []string{tampered, id}, nil)
		if code == 0 {
			t.Fatalf("replay of a tampered step-capped record exited 0\nstdout:\n%s\nstderr:\n%s", stdout, stderr)
		}
		if !strings.Contains(stderr, "diverged") {
			t.Fatalf("replay output does not name the divergence\nstdout:\n%s\nstderr:\n%s", stdout, stderr)
		}
	})
}

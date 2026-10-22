package examples_test

// Hermetic execution test for examples/agent-guardrail: the built binary runs
// against a scripted openai-compatible chat server and a scripted System One
// server. No build tag, no credentials, no network beyond the two httptest
// servers.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

type guardrailBackends struct {
	env map[string]string

	mu       sync.Mutex
	llmCalls []guardrailChatRequest
	chatHits int
	sysHits  int
}

type guardrailChatRequest struct {
	Messages []struct {
		Role       string `json:"role"`
		ToolCallID string `json:"tool_call_id"`
		Content    string `json:"content"`
	} `json:"messages"`
}

// toolMessages returns the content of the last chat request's tool messages
// by tool_call_id.
func (b *guardrailBackends) toolMessages(t *testing.T) map[string]string {
	t.Helper()
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.llmCalls) == 0 {
		t.Fatal("the chat server saw no request")
	}
	out := map[string]string{}
	for _, m := range b.llmCalls[len(b.llmCalls)-1].Messages {
		if m.Role == "tool" {
			out[m.ToolCallID] = m.Content
		}
	}
	return out
}

// guardrailTurns scripts the chat server by number of tool results seen: rm,
// then read_file, then the answer. One call per turn because the provider
// stack serializes an openai-compatible model's parallel tool calls.
var guardrailTurns = []string{
	`{"role":"assistant","content":null,"tool_calls":[{"id":"call_rm","type":"function","function":{"name":"rm","arguments":"{\"path\":\"/var/data/customers.db\",\"recursive\":true}"}}]}`,
	`{"role":"assistant","content":null,"tool_calls":[{"id":"call_read","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"notes.txt\"}"}}]}`,
	`{"role":"assistant","content":"The rm call was blocked; notes.txt says to keep the customer database."}`,
}

// startGuardrailBackends starts the chat server and a System One server that
// rates rm destructive (0.9), read_file safe (0.1), and classifies the user
// turn as topic.
func startGuardrailBackends(t *testing.T, topic string) *guardrailBackends {
	t.Helper()
	b := &guardrailBackends{}

	chat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.chatHits++
		b.mu.Unlock()
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var req guardrailChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		b.mu.Lock()
		b.llmCalls = append(b.llmCalls, req)
		b.mu.Unlock()
		toolResults := 0
		for _, m := range req.Messages {
			if m.Role == "tool" {
				toolResults++
			}
		}
		message, finish := guardrailTurns[min(toolResults, len(guardrailTurns)-1)], "tool_calls"
		if toolResults >= len(guardrailTurns)-1 {
			finish = "stop"
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"m","choices":[{"index":0,"message":%s,"finish_reason":%q}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`, message, finish)
	}))
	t.Cleanup(chat.Close)

	sys := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.sysHits++
		b.mu.Unlock()
		if r.URL.Path != "/v1/systemone" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			State     struct{ Tool string } `json:"state"`
			Questions map[string]any        `json:"questions"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch {
		case req.Questions["destructive"] != nil:
			belief := 0.1
			if strings.EqualFold(req.State.Tool, "rm") {
				belief = 0.9
			}
			_, _ = fmt.Fprintf(w, `{"model":"jev-test","answers":{"destructive":{"type":"noul","noul":%v}},"usage":{"input_tokens":20,"output_tokens":1}}`, belief)
		case req.Questions["topic"] != nil:
			probs := fmt.Sprintf(`{"on_topic":0.05,"off_topic":0.05,"injection":0.05,%q:0.85}`, topic)
			_, _ = fmt.Fprintf(w, `{"model":"jev-test","answers":{"topic":{"type":"choice","choice":%q,"probabilities":%s,"confidence":0.85}},"usage":{"input_tokens":20,"output_tokens":1}}`, topic, probs)
		default:
			http.Error(w, "unexpected questions", http.StatusBadRequest)
		}
	}))
	t.Cleanup(sys.Close)

	b.env = map[string]string{
		"LLMKIT_PROVIDER":           "openai-compatible",
		"LLMKIT_MODEL":              "m",
		"LLMKIT_API_KEY":            "k",
		"LLMKIT_BASE_URL":           chat.URL + "/v1",
		"LLMKIT_SYSTEMONE_API_KEY":  "k",
		"LLMKIT_SYSTEMONE_MODEL":    "jev-test",
		"LLMKIT_SYSTEMONE_BASE_URL": sys.URL,
	}
	return b
}

func TestGuardrailDeniesDestructiveCallAndAllowsRead(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	bin := buildExample(t, moduleRoot(t), t.TempDir(), "agent-guardrail")
	b := startGuardrailBackends(t, "on_topic")

	code, stdout, stderr := runExampleArgs(t, bin, nil, b.env)
	if code != 0 {
		t.Fatalf("exit %d, want 0\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	for _, want := range []string{
		"policy: DENIED rm",
		"policy: allowed read_file",
		"tool: read_file ran on notes.txt",
		"files: /var/data/customers.db, notes.txt",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout lacks %q:\n%s", want, stdout)
		}
	}
	if strings.Contains(stdout, "tool: rm ran") {
		t.Errorf("the denied rm tool ran:\n%s", stdout)
	}

	msgs := b.toolMessages(t)
	if got := msgs["call_rm"]; !strings.Contains(got, "denied") {
		t.Errorf("rm tool result sent to the model = %q, want a denial", got)
	}
	if got := msgs["call_read"]; !strings.Contains(got, "Quarterly cleanup notes") {
		t.Errorf("read_file tool result sent to the model = %q, want the file content", got)
	}
}

func TestGuardrailAbortsOnInjection(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	bin := buildExample(t, moduleRoot(t), t.TempDir(), "agent-guardrail")
	b := startGuardrailBackends(t, "injection")

	code, stdout, stderr := runExampleArgs(t, bin, nil, b.env)
	if code == 0 {
		t.Fatalf("exit 0, want non-zero\nstdout:\n%s\nstderr:\n%s", stdout, stderr)
	}
	if !strings.Contains(stderr, "injection") {
		t.Errorf("stderr does not name the injection verdict:\n%s", stderr)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.llmCalls) != 0 {
		t.Errorf("the chat server saw %d request(s); the request policy must abort before the wire call", len(b.llmCalls))
	}
}

// TestGuardrailThresholdRange runs the binary against the scripted backends
// (rm belief 0.9, read_file 0.1) with each --threshold. A run starts iff the
// value is in (0, 1]; every other value the flag parser accepts exits 1, names
// the flag, and sends nothing to either server.
func TestGuardrailThresholdRange(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	bin := buildExample(t, moduleRoot(t), t.TempDir(), "agent-guardrail")
	for _, tc := range []struct {
		threshold string
		accepted  bool
		rm        string // accepted rows: the policy line for rm
	}{
		{threshold: "NaN"},
		{threshold: "nan"},
		{threshold: "+Inf"},
		{threshold: "-Inf"},
		{threshold: "0"},
		{threshold: "-0.5"},
		{threshold: "1.1"},
		{threshold: "1", accepted: true, rm: "policy: allowed rm"},
		{threshold: "0.5", accepted: true, rm: "policy: DENIED rm"},
	} {
		t.Run(tc.threshold, func(t *testing.T) {
			b := startGuardrailBackends(t, "on_topic")
			code, stdout, stderr := runExampleArgs(t, bin, []string{"--threshold", tc.threshold}, b.env)
			b.mu.Lock()
			llm, sys := b.chatHits, b.sysHits
			b.mu.Unlock()

			if tc.accepted {
				if code != 0 {
					t.Fatalf("exit %d, want 0\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
				}
				if !strings.Contains(stdout, tc.rm) {
					t.Errorf("stdout lacks %q:\n%s", tc.rm, stdout)
				}
				if llm == 0 || sys == 0 {
					t.Errorf("chat server saw %d request(s), System One server %d; an accepted run reaches both", llm, sys)
				}
				return
			}
			if code != 1 {
				t.Errorf("exit %d, want 1\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
			}
			if !strings.Contains(stderr, "--threshold") {
				t.Errorf("stderr does not name --threshold:\n%s", stderr)
			}
			if llm != 0 || sys != 0 {
				t.Errorf("chat server saw %d request(s), System One server %d; want 0 for a rejected threshold\nstdout:\n%s", llm, sys, stdout)
			}
		})
	}
}

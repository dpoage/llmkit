package decide_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/decide/systemone"
)

// scriptedClient is a hermetic llmkit.Client that serves fixed responses in
// order.
type scriptedClient struct {
	mu    sync.Mutex
	resps []llmkit.Response
	i     int
}

func (c *scriptedClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *scriptedClient) Complete(context.Context, llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.i >= len(c.resps) {
		return llmkit.Response{}, fmt.Errorf("scriptedClient: no response %d scripted", c.i+1)
	}
	r := c.resps[c.i]
	c.i++
	return r, nil
}

// namedTool is a no-argument-checking tool that returns a fixed result.
type namedTool struct{ name, result string }

func (t namedTool) Def() llmkit.ToolDef { return llmkit.ToolDef{Name: t.name} }
func (t namedTool) Run(context.Context, json.RawMessage) ([]llmkit.Block, error) {
	return []llmkit.Block{llmkit.Text(t.result)}, nil
}

const denyAt = 0.5

// beliefPolicy asks one Noul question about the call through d and denies
// the call when the belief that it is dangerous reaches denyAt. It fails
// closed: a decision it cannot get denies the call.
func beliefPolicy(d decide.Decider) agent.ToolPolicy {
	return agent.ToolPolicyFunc(func(ctx context.Context, call *llmkit.ToolCall) error {
		state := map[string]any{"tool": call.Name, "arguments": json.RawMessage(call.Arguments)}
		qs := decide.Questions{"dangerous": decide.Noul{
			Instructions: "Would running this tool call destroy data or cause irreversible harm?",
			True:         "The call is destructive.",
			False:        "The call is safe and read-only.",
		}}
		resp, err := d.Ask(ctx, state, qs)
		if err != nil {
			return fmt.Errorf("decision unavailable: %w", err)
		}
		belief, err := resp.Noul("dangerous")
		if err != nil {
			return err
		}
		if belief >= denyAt {
			return fmt.Errorf("blocked: belief %.2f that the call is dangerous", belief)
		}
		return nil
	})
}

// deniedCalls returns the tool-call ids whose result is a policy denial.
func deniedCalls(msgs []llmkit.Message) map[string]string {
	out := map[string]string{}
	for _, m := range msgs {
		if m.Role == llmkit.RoleToolResult && strings.Contains(m.Text(), " denied: ") {
			out[m.ToolCallID] = m.Text()
		}
	}
	return out
}

func TestObserveReplay_RunnerToolPolicyRecordsAndReplaysWithoutNetwork(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		body, _ := io.ReadAll(r.Body)
		belief := 0.1
		if strings.Contains(string(body), "rm -rf") {
			belief = 0.9
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"model":"jev-1.13.0","answers":{"dangerous":{"type":"noul","noul":%v}},"usage":{"input_tokens":21,"output_tokens":0}}`, belief)
	}))
	defer srv.Close()

	client, err := systemone.New(systemone.Config{Secret: "o3-test-secret", Model: "jev-latest", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("systemone.New: %v", err)
	}
	sink := agent.JSONL(t.TempDir(), func(err error) { t.Errorf("JSONL sink: %v", err) })

	script := func() *scriptedClient {
		return &scriptedClient{resps: []llmkit.Response{
			{
				StopReason: llmkit.StopToolUse,
				ToolCalls: []llmkit.ToolCall{
					{ID: "call-rm", Name: "shell", Arguments: json.RawMessage(`{"cmd":"rm -rf /var/data"}`)},
					{ID: "call-read", Name: "read_file", Arguments: json.RawMessage(`{"path":"/etc/hostname"}`)},
				},
			},
			{StopReason: llmkit.StopEndTurn, Text: "read done, shell call was refused"},
		}}
	}
	tools := []agent.Tool{namedTool{"shell", "SHELL RAN"}, namedTool{"read_file", "box-1"}}

	// Record: live decisions through Observe into the same sink the Runner
	// records into.
	recorder := agent.NewRunner(script(), tools, "sys",
		agent.WithObserver(sink),
		agent.WithToolPolicy(beliefPolicy(decide.Observe(client, sink))))
	recorded, err := recorder.Run(context.Background(), "clean up")
	if err != nil {
		t.Fatalf("record run: %v", err)
	}
	if got := hits.Load(); got != 2 {
		t.Fatalf("record run wire hits = %d, want 2 (one decision per tool call)", got)
	}
	recordedDenials := deniedCalls(recorded.Messages)
	if len(recordedDenials) != 1 || recordedDenials["call-rm"] == "" {
		t.Fatalf("record run denials = %v, want exactly call-rm denied", recordedDenials)
	}

	events, err := sink.Events(context.Background(), recorded.RunID)
	if err != nil {
		t.Fatalf("sink.Events: %v", err)
	}
	// The tool turn's Step: the completion that requested the tool calls.
	toolStep := -1
	var completionIdx, toolRunIdx []int
	var decisions []llmkit.Event
	for i, ev := range events {
		switch ev.Kind {
		case llmkit.KindCompletion:
			if ev.Completion != nil && len(ev.Completion.Response.ToolCalls) > 0 {
				toolStep = ev.Step
				completionIdx = append(completionIdx, i)
			}
		case llmkit.KindToolRun:
			toolRunIdx = append(toolRunIdx, i)
		case llmkit.KindDecision:
			decisions = append(decisions, ev)
		}
	}
	if toolStep < 0 || len(completionIdx) != 1 || len(toolRunIdx) != 2 {
		t.Fatalf("record shape: toolStep=%d completions=%v toolruns=%v", toolStep, completionIdx, toolRunIdx)
	}
	if len(decisions) != 2 {
		t.Fatalf("Decision events in the record = %d, want 2 (one per tool call)", len(decisions))
	}
	for i, ev := range events {
		if ev.Kind != llmkit.KindDecision {
			continue
		}
		if ev.RunID != recorded.RunID {
			t.Errorf("decision event RunID = %q, want the run's %q", ev.RunID, recorded.RunID)
		}
		if ev.Step != toolStep {
			t.Errorf("decision event Step = %d, want the tool turn's %d", ev.Step, toolStep)
		}
		if i < completionIdx[0] || i > toolRunIdx[0] {
			t.Errorf("decision event at index %d is not between the tool turn's completion (%d) and its first tool run (%d)", i, completionIdx[0], toolRunIdx[0])
		}
		de := ev.Decision
		if de.Backend != "systemone" || de.Model != "jev-1.13.0" || de.Err != "" || len(de.Answers) != 1 {
			t.Errorf("decision payload = %+v, want a successful systemone decision with one answer", de)
		}
		if de.Usage.InputTokens != 21 {
			t.Errorf("decision usage = %+v, want 21 input tokens", de.Usage)
		}
	}

	// Replay with the network gone.
	srv.Close()
	hitsBefore := hits.Load()
	rc, err := agent.NewReplayClient(sink, recorded.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	rd, err := decide.Replay(sink, recorded.RunID)
	if err != nil {
		t.Fatalf("decide.Replay: %v", err)
	}
	replayed, err := agent.NewRunner(rc, rc.Tools(), "sys", agent.WithToolPolicy(beliefPolicy(rd))).
		Run(context.Background(), "clean up")
	if err != nil {
		t.Fatalf("replay run: %v", err)
	}
	if rc.Err() != nil {
		t.Errorf("ReplayClient.Err() = %v, want nil", rc.Err())
	}
	if replayed.FinalText != recorded.FinalText {
		t.Errorf("replayed FinalText = %q, want %q", replayed.FinalText, recorded.FinalText)
	}
	replayedDenials := deniedCalls(replayed.Messages)
	if len(replayedDenials) != 1 || replayedDenials["call-rm"] != recordedDenials["call-rm"] {
		t.Errorf("replayed denials = %v, want the recorded %v", replayedDenials, recordedDenials)
	}
	if got := hits.Load(); got != hitsBefore {
		t.Errorf("replay wire hits grew from %d to %d, want zero network", hitsBefore, got)
	}
}

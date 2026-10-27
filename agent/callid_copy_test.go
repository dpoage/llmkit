package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

func assistantCallIDs(msgs []llmkit.Message) []string {
	var ids []string
	for _, m := range msgs {
		if m.Role == llmkit.RoleAssistant {
			for _, tc := range m.ToolCalls {
				ids = append(ids, tc.ID)
			}
		}
	}
	return ids
}

func TestW0b_EmptyCallIDReusedStepStaysUnique(t *testing.T) {
	step := toolResp("", "blob", `{}`, 1, 1)
	fc := newFakeClient(step, step, textResp("done", 1, 1))
	out, err := NewRunner(fc, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys").
		Run(context.Background(), "task", WithRunID("w0b-reuse"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := assistantCallIDs(out.Messages)
	want := []string{"llmkit-w0b-reuse-s1-c0", "llmkit-w0b-reuse-s2-c0"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("history call IDs = %q, want %q", got, want)
	}
	if id := step.resp.ToolCalls[0].ID; id != "" {
		t.Errorf("client's Response was written: ToolCalls[0].ID = %q", id)
	}
}

// sharedRespClient answers every turn that does not follow a tool result with
// the same Response value, so all runs share its ToolCalls backing array.
type sharedRespClient struct {
	shared llmkit.Response
}

func (c *sharedRespClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *sharedRespClient) Complete(_ context.Context, req llmkit.Request) (llmkit.Response, error) {
	if n := len(req.Messages); n > 0 && req.Messages[n-1].Role == llmkit.RoleToolResult {
		return llmkit.Response{Text: "done", StopReason: llmkit.StopEndTurn}, nil
	}
	return c.shared, nil
}

func TestW0b_EmptyCallIDSharedResponseConcurrentRuns(t *testing.T) {
	client := &sharedRespClient{shared: llmkit.Response{
		StopReason: llmkit.StopToolUse,
		ToolCalls:  []llmkit.ToolCall{{Name: "blob", Arguments: json.RawMessage(`{}`)}},
	}}
	r := NewRunner(client, []Tool{blocksTool{"blob", textBlocks("ok")}}, "sys")
	const runs = 8
	got := make([][]string, runs)
	errs := make([]error, runs)
	var wg sync.WaitGroup
	for i := range runs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := r.Run(context.Background(), "task", WithRunID(llmkit.RunID(fmt.Sprintf("w0b-par-%d", i))))
			if err != nil {
				errs[i] = err
				return
			}
			got[i] = assistantCallIDs(out.Messages)
		}()
	}
	wg.Wait()
	for i := range runs {
		if errs[i] != nil {
			t.Errorf("run %d: %v", i, errs[i])
			continue
		}
		want := fmt.Sprintf("llmkit-w0b-par-%d-s1-c0", i)
		if len(got[i]) != 1 || got[i][0] != want {
			t.Errorf("run %d call IDs = %q, want [%q]", i, got[i], want)
		}
	}
	if id := client.shared.ToolCalls[0].ID; id != "" {
		t.Errorf("shared Response was written: ToolCalls[0].ID = %q", id)
	}
}

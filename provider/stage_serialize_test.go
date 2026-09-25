package provider

// Serialize-stage tests pin the stage's truncation through Wrap and its
// no-op identity (serializeStage passthrough) directly, since Wrap adds
// the retry stage around it.

import (
	"context"
	"encoding/json"
	"testing"

	llmkit "github.com/dpoage/llmkit"
)

func TestSerializeStage_TruncatesParallelToolCalls(t *testing.T) {
	fc := &fakeClient{
		caps: llmkit.Capabilities{ParallelToolCalls: false},
		responses: []llmkit.Response{{
			ToolCalls: []llmkit.ToolCall{
				{ID: "1", Name: "a", Arguments: json.RawMessage(`{}`)},
				{ID: "2", Name: "b", Arguments: json.RawMessage(`{}`)},
				{ID: "3", Name: "c", Arguments: json.RawMessage(`{}`)},
			},
			StopReason: llmkit.StopToolUse,
		}},
	}
	client := serializeStage(fc)
	resp, err := client.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "1" {
		t.Errorf("kept tool call ID = %q, want first (1)", resp.ToolCalls[0].ID)
	}
	if resp.StopReason != llmkit.StopToolUse {
		t.Errorf("StopReason = %q, want tool_use", resp.StopReason)
	}
	if client.Capabilities().ParallelToolCalls {
		t.Error("wrapped Capabilities should report ParallelToolCalls=false")
	}
}

func TestSerializeStage_NoOpForParallelCapableClient(t *testing.T) {
	fc := &fakeClient{caps: llmkit.Capabilities{ParallelToolCalls: true}}
	client := serializeStage(fc)
	// Must return the same client unchanged (no wrapping).
	if client != llmkit.Client(fc) {
		t.Error("parallel-capable client was wrapped; want passthrough")
	}
}

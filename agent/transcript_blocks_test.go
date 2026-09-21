package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestTranscriptBlocksSurviveJSONLReplay pins the Blocks round-trip:
// Response.Blocks reach the Completion event, survive SaveJSONL/LoadJSONL
// (Block.Data/.Raw through encoding/json), and NewReplayClient serves
// them back on Complete.
func TestTranscriptBlocksSurviveJSONLReplay(t *testing.T) {
	const thinkRaw = `{"type":"thinking","thinking":"why","signature":"sig-1"}`

	src := NewReplayClientFromResponses([]llmkit.Response{{
		Text: "answer",
		Blocks: []llmkit.Block{
			{Kind: llmkit.BlockThinking, Text: "why", Provider: "anthropic", Raw: json.RawMessage(thinkRaw)},
			{Kind: llmkit.BlockText, Text: "answer"},
		},
		StopReason: llmkit.StopEndTurn,
	}}, llmkit.Capabilities{})
	rec := NewRunner(src, nil, "sys")
	out, err := rec.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	var buf bytes.Buffer
	if err := out.Transcript.SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	loaded, err := LoadJSONL(&buf)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}

	replay, err := NewReplayClient(loaded, loaded.RunID, llmkit.Capabilities{})
	if err != nil {
		t.Fatalf("NewReplayClient: %v", err)
	}
	resp, err := replay.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "task")},
	})
	if err != nil {
		t.Fatalf("replay Complete: %v", err)
	}
	if len(resp.Blocks) != 2 {
		t.Fatalf("replayed Blocks = %d, want 2", len(resp.Blocks))
	}
	th := resp.Blocks[0]
	if th.Kind != llmkit.BlockThinking || th.Provider != "anthropic" {
		t.Errorf("replayed block 0 = %+v", th)
	}
	if string(th.Raw) != thinkRaw {
		t.Errorf("replayed Raw = %s, want byte-identical %s", th.Raw, thinkRaw)
	}
	if resp.Text != "answer" {
		t.Errorf("replayed Text = %q, want answer", resp.Text)
	}
}

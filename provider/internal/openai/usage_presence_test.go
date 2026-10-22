package openai

import (
	"testing"

	"github.com/dpoage/llmkit"
)

func TestStreamUsageUnreportedIndistinguishableFromZero(t *testing.T) {
	text := []string{
		chunkJSON(`{"role":"assistant","content":"hello there"}`, "", ""),
		chunkJSON(`{}`, "stop", ""),
	}
	unreported, _, err := stream(t, newServer(t, sseHandler(nil, text...)), simpleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	reportedChunks := append(append([]string{}, text...),
		usageChunkJSON(`{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`))
	reported, _, err := stream(t, newServer(t, sseHandler(nil, reportedChunks...)), simpleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if unreported.Usage != (llmkit.Usage{}) {
		t.Errorf("unreported Usage = %+v, want zero value", unreported.Usage)
	}
	if unreported.Usage != reported.Usage {
		t.Errorf("unreported %+v differs from reported zero %+v", unreported.Usage, reported.Usage)
	}
}

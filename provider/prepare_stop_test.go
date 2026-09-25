package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// stopRow is one (adapter, reason, tool-calls-present) row.
type stopRow struct {
	provider string
	reason   string
	hasTool  bool
	want     llmkit.StopReason
}

// stopRows is the full stop-reason table: every recognized reason in each
// adapter's own table plus its unrecognized/empty StopFallback rows.
func stopRows() []stopRow {
	var rows []stopRow
	add := func(provider, reason string, hasTool bool, want llmkit.StopReason) {
		rows = append(rows, stopRow{provider, reason, hasTool, want})
	}

	for _, p := range []string{"openai", "openai-compatible"} {
		add(p, "stop", false, llmkit.StopEndTurn)
		add(p, "stop", true, llmkit.StopToolUse)
		add(p, "tool_calls", true, llmkit.StopToolUse)
		add(p, "function_call", true, llmkit.StopToolUse)
		// tool_calls/function_call map to ToolUse unconditionally (not
		// merely because tool calls happen to be present) — a row with no
		// tool-call block distinguishes that from adapter.StopFallback,
		// which would report Error for hasToolCalls=false.
		add(p, "tool_calls", false, llmkit.StopToolUse)
		add(p, "function_call", false, llmkit.StopToolUse)
		add(p, "length", false, llmkit.StopMaxTokens)
		add(p, "content_filter", false, llmkit.StopContentFilter)
		add(p, "abort", true, llmkit.StopToolUse)
		add(p, "abort", false, llmkit.StopError)
		add(p, "", true, llmkit.StopToolUse)
		add(p, "", false, llmkit.StopError)
	}

	add("anthropic", "end_turn", false, llmkit.StopEndTurn)
	add("anthropic", "stop_sequence", false, llmkit.StopEndTurn)
	add("anthropic", "tool_use", true, llmkit.StopToolUse)
	add("anthropic", "max_tokens", false, llmkit.StopMaxTokens)
	add("anthropic", "refusal", false, llmkit.StopRefusal)
	add("anthropic", "pause_turn", true, llmkit.StopToolUse)
	add("anthropic", "pause_turn", false, llmkit.StopError)

	for _, r := range []string{"STOP", "FINISH_REASON_UNSPECIFIED", ""} {
		add("google", r, false, llmkit.StopEndTurn)
		add("google", r, true, llmkit.StopToolUse)
	}
	add("google", "MAX_TOKENS", false, llmkit.StopMaxTokens)
	for _, r := range []string{"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"} {
		add("google", r, false, llmkit.StopContentFilter)
	}
	add("google", "OTHER", true, llmkit.StopToolUse)
	add("google", "OTHER", false, llmkit.StopError)

	return rows
}

// --- Complete-mode bodies ---------------------------------------------------

func anthropicCompleteStopBody(reason string, hasTool bool) string {
	content := `[{"type":"text","text":"ok"}]`
	if hasTool {
		content = `[{"type":"tool_use","id":"tu_1","name":"do_it","input":{}}]`
	}
	return fmt.Sprintf(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test",`+
		`"content":%s,"stop_reason":%s,"usage":{"input_tokens":1,"output_tokens":1}}`,
		content, jsonStrOrNull(reason))
}

func openAICompleteStopBody(reason string, hasTool bool) string {
	msg := `{"role":"assistant","content":"ok"}`
	if hasTool {
		msg = `{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"do_it","arguments":"{}"}}]}`
	}
	return fmt.Sprintf(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-test",`+
		`"choices":[{"index":0,"message":%s,"finish_reason":%s}],`+
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		msg, jsonStrOrNull(reason))
}

func googleCompleteStopBody(reason string, hasTool bool) string {
	part := `{"text":"ok"}`
	if hasTool {
		part = `{"functionCall":{"id":"call-1","name":"do_it","args":{}}}`
	}
	finish := ""
	if reason != "" {
		finish = `,"finishReason":` + jsonStrOrNull(reason)
	}
	return fmt.Sprintf(`{"candidates":[{"content":{"role":"model","parts":[%s]}%s}],`+
		`"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
		part, finish)
}

func jsonStrOrNull(s string) string {
	if s == "" {
		return "null"
	}
	b, _ := json.Marshal(s)
	return string(b)
}

func completeStopBody(provider, reason string, hasTool bool) string {
	switch provider {
	case "anthropic":
		return anthropicCompleteStopBody(reason, hasTool)
	case "google":
		return googleCompleteStopBody(reason, hasTool)
	default:
		return openAICompleteStopBody(reason, hasTool)
	}
}

// --- Stream-mode (SSE) bodies ------------------------------------------------

func anthropicStreamStopBody(reason string, hasTool bool) string {
	var blockStart, blockDelta string
	if hasTool {
		blockStart = `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"do_it","input":{}}}`
		blockDelta = `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`
	} else {
		blockStart = `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
		blockDelta = `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`
	}
	events := []struct{ name, data string }{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`},
		{"content_block_start", blockStart},
		{"content_block_delta", blockDelta},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%s,"stop_sequence":null},"usage":{"output_tokens":1}}`, jsonStrOrNull(reason))},
		{"message_stop", `{"type":"message_stop"}`},
	}
	var out string
	for _, ev := range events {
		out += "event: " + ev.name + "\ndata: " + ev.data + "\n\n"
	}
	return out
}

func openAIStreamStopBody(reason string, hasTool bool) string {
	firstDelta := `{"role":"assistant","content":"ok"}`
	if hasTool {
		firstDelta = `{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"do_it","arguments":"{}"}}]}`
	}
	chunk := func(delta, finish string) string {
		f := "null"
		if finish != "" {
			f = jsonStrOrNull(finish)
		}
		return fmt.Sprintf(`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test",`+
			`"choices":[{"index":0,"delta":%s,"finish_reason":%s}],"usage":null}`, delta, f)
	}
	usageChunk := `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test",` +
		`"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	var out string
	for _, c := range []string{chunk(firstDelta, ""), chunk("{}", reason), usageChunk} {
		out += "data: " + c + "\n\n"
	}
	out += "data: [DONE]\n\n"
	return out
}

func googleStreamStopBody(reason string, hasTool bool) string {
	part := `{"text":"ok"}`
	if hasTool {
		part = `{"functionCall":{"id":"call-1","name":"do_it","args":{}}}`
	}
	chunk1 := fmt.Sprintf(`{"candidates":[{"content":{"role":"model","parts":[%s]}}]}`, part)
	cand := "{}"
	if reason != "" {
		cand = fmt.Sprintf(`{"finishReason":%s}`, jsonStrOrNull(reason))
	}
	chunk2 := fmt.Sprintf(`{"candidates":[%s],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`, cand)
	return "data: " + chunk1 + "\n\ndata: " + chunk2 + "\n\n"
}

func streamStopBody(provider, reason string, hasTool bool) string {
	switch provider {
	case "anthropic":
		return anthropicStreamStopBody(reason, hasTool)
	case "google":
		return googleStreamStopBody(reason, hasTool)
	default:
		return openAIStreamStopBody(reason, hasTool)
	}
}

// TestPrepare_StopReasonMapping exercises one table over allAdapters(),
// Complete AND Stream, covering each adapter's whole recognized
// stop-reason table plus its unrecognized-reason and empty-reason
// StopFallback rows. Stream's Response comes from the same toResponse
// (anthropic/openai) or g.toResponse (google) normalizer Complete uses, so
// running both modes pins that the identical mapping function backs each.
func TestPrepare_StopReasonMapping(t *testing.T) {
	factories := map[string]adapterFactory{}
	for _, f := range allAdapters() {
		factories[f.name] = f
	}
	for _, row := range stopRows() {
		name := fmt.Sprintf("%s/reason=%s/tool=%v", row.provider, row.reason, row.hasTool)
		t.Run(name, func(t *testing.T) {
			f := factories[row.provider]

			t.Run("complete", func(t *testing.T) {
				base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(completeStopBody(row.provider, row.reason, row.hasTool)))
				})
				client := f.build(t, base)
				resp, err := client.Complete(context.Background(), simpleRequest())
				if err != nil {
					t.Fatalf("Complete: %v", err)
				}
				if resp.StopReason != row.want {
					t.Errorf("StopReason = %q, want %q", resp.StopReason, row.want)
				}
			})

			// An empty finish/stop reason has no terminal-signal
			// representation in streaming mode on openai/openai-compatible
			// (stream.go treats an absent finish_reason as a truncated
			// generation, not a completion) or on google (absorbChunk's
			// sawFinish only latches on a non-empty FinishReason) — only
			// Complete can carry a response with no terminal field at all.
			// Anthropic has no analogous "" row (StopFallback covers
			// pause_turn instead).
			if row.reason == "" {
				return
			}
			t.Run("stream", func(t *testing.T) {
				base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte(streamStopBody(row.provider, row.reason, row.hasTool)))
				})
				client := f.build(t, base)
				sc, ok := client.(llmkit.StreamingClient)
				if !ok {
					t.Fatalf("%s adapter does not implement llmkit.StreamingClient", row.provider)
				}
				resp, err := sc.Stream(context.Background(), simpleRequest(), func(llmkit.Delta) error { return nil })
				if err != nil {
					t.Fatalf("Stream: %v", err)
				}
				if resp.StopReason != row.want {
					t.Errorf("StopReason = %q, want %q", resp.StopReason, row.want)
				}
			})
		})
	}
}

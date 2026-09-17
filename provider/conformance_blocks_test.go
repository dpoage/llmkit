package provider

// Conformance tests for the content-block message model (bead llmkit-yr7.3):
// image/document wire shapes, the Anthropic thinking+tool signature
// round-trip, tool_choice mapping, refusal/content-filter stop reasons, the
// Data/URL source rule (rejected before any wire call), and the
// foreign-thinking-block drop rule.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// pngBytes is a tiny fake image payload; only its base64 wire form matters.
var pngBytes = []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a}

// pdfBytes is a tiny fake PDF payload.
var pdfBytes = []byte("%PDF-1.4 fake")

// decodeBody is a JSON body decoded into generic structures.
type wireBody = map[string]any

// captureNext wraps handler so every request body is decoded and counted.
func captureWireBody(t *testing.T, count *int, respBody func() string, captured *wireBody) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		*count++
		dec := wireBody{}
		if err := json.NewDecoder(r.Body).Decode(&dec); err != nil {
			t.Errorf("undecodable request body: %v", err)
		}
		*captured = dec
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(respBody()))
	}
}

// userImageRequest builds a request carrying one inline PNG image block.
func userImageRequest() llmkit.Request {
	return llmkit.Request{
		Messages: []llmkit.Message{{
			Role: llmkit.RoleUser,
			Content: []llmkit.Block{
				{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes},
				{Kind: llmkit.BlockText, Text: "what is this?"},
			},
		}},
		MaxTokens: 64,
	}
}

// TestConformance_ImageBlockWireShape asserts the exact wire JSON each
// provider receives for an inline image block (base64 + media type), and
// that the response still normalizes.
func TestConformance_ImageBlockWireShape(t *testing.T) {
	wantB64 := base64.StdEncoding.EncodeToString(pngBytes)
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			var captured wireBody
			n := 0
			base := newServer(t, captureWireBody(t, &n, func() string {
				return mockTextBody(f.name, "a cat", 3, 2)
			}, &captured))
			client := f.build(t, base)

			resp, err := client.Complete(context.Background(), userImageRequest())
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if resp.Text != "a cat" {
				t.Errorf("Text = %q, want %q", resp.Text, "a cat")
			}
			if n != 1 {
				t.Fatalf("wire calls = %d, want 1", n)
			}

			// Extract the first user content entry per provider wire shape.
			var part map[string]any
			switch f.name {
			case "anthropic":
				msgs := captured["messages"].([]any)
				first := msgs[0].(map[string]any)
				blocks := first["content"].([]any)
				part = blocks[0].(map[string]any)
				if part["type"] != "image" {
					t.Fatalf("block type = %v, want image", part["type"])
				}
				src := part["source"].(map[string]any)
				if src["type"] != "base64" || src["media_type"] != "image/png" || src["data"] != wantB64 {
					t.Errorf("anthropic image source = %v, want base64/image/png/%q", src, wantB64)
				}
				if blocks[1].(map[string]any)["text"] != "what is this?" {
					t.Errorf("text block not preserved after image: %v", blocks[1])
				}
			case "openai", "openai-compatible":
				msgs := captured["messages"].([]any)
				first := msgs[0].(map[string]any)
				parts := first["content"].([]any)
				part = parts[0].(map[string]any)
				if part["type"] != "image_url" {
					t.Fatalf("part type = %v, want image_url", part["type"])
				}
				u := part["image_url"].(map[string]any)["url"].(string)
				want := "data:image/png;base64," + wantB64
				if u != want {
					t.Errorf("image_url = %q, want %q", u, want)
				}
				if parts[1].(map[string]any)["text"] != "what is this?" {
					t.Errorf("text part not preserved after image: %v", parts[1])
				}
			case "google":
				contents := captured["contents"].([]any)
				first := contents[0].(map[string]any)
				parts := first["parts"].([]any)
				part = parts[0].(map[string]any)
				inline := part["inlineData"].(map[string]any)
				if inline["mimeType"] != "image/png" {
					t.Errorf("inlineData.mimeType = %v, want image/png", inline["mimeType"])
				}
				if inline["data"] != wantB64 {
					t.Errorf("inlineData.data = %v, want %q", inline["data"], wantB64)
				}
				if parts[1].(map[string]any)["text"] != "what is this?" {
					t.Errorf("text part not preserved after image: %v", parts[1])
				}
			}
		})
	}
}

// TestConformance_DocumentBlockWireShape asserts the wire JSON for an inline
// document block on every adapter (all three support documents).
func TestConformance_DocumentBlockWireShape(t *testing.T) {
	wantB64 := base64.StdEncoding.EncodeToString(pdfBytes)
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			var captured wireBody
			n := 0
			base := newServer(t, captureWireBody(t, &n, func() string {
				return mockTextBody(f.name, "summarized", 3, 2)
			}, &captured))
			client := f.build(t, base)

			req := llmkit.Request{
				Messages: []llmkit.Message{{
					Role: llmkit.RoleUser,
					Content: []llmkit.Block{
						{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: pdfBytes, Title: "spec"},
					},
				}},
				MaxTokens: 64,
			}
			if _, err := client.Complete(context.Background(), req); err != nil {
				t.Fatalf("Complete: %v", err)
			}

			switch f.name {
			case "anthropic":
				msgs := captured["messages"].([]any)
				blocks := msgs[0].(map[string]any)["content"].([]any)
				part := blocks[0].(map[string]any)
				if part["type"] != "document" {
					t.Fatalf("block type = %v, want document", part["type"])
				}
				src := part["source"].(map[string]any)
				if src["type"] != "base64" || src["media_type"] != "application/pdf" || src["data"] != wantB64 {
					t.Errorf("anthropic document source = %v", src)
				}
				if part["title"] != "spec" {
					t.Errorf("document title = %v, want spec", part["title"])
				}
			case "openai", "openai-compatible":
				msgs := captured["messages"].([]any)
				parts := msgs[0].(map[string]any)["content"].([]any)
				part := parts[0].(map[string]any)
				if part["type"] != "file" {
					t.Fatalf("part type = %v, want file", part["type"])
				}
				file := part["file"].(map[string]any)
				if file["file_data"] != "data:application/pdf;base64,"+wantB64 {
					t.Errorf("file_data = %v", file["file_data"])
				}
			case "google":
				contents := captured["contents"].([]any)
				parts := contents[0].(map[string]any)["parts"].([]any)
				inline := parts[0].(map[string]any)["inlineData"].(map[string]any)
				if inline["mimeType"] != "application/pdf" || inline["data"] != wantB64 {
					t.Errorf("inlineData = %v", inline)
				}
			}
		})
	}
}

// TestConformance_AnthropicThinkingToolRoundTrip is the signature round-trip
// the bead demands: a response carrying a thinking block (with signature)
// plus a tool_use must let the next request re-send the thinking block
// VERBATIM — signature bytes equal on the wire — ahead of the tool_result.
func TestConformance_AnthropicThinkingToolRoundTrip(t *testing.T) {
	const (
		callID        = "call_think"
		toolName      = "read_file"
		thinkText     = "I should read the file before answering."
		thinkSig      = "sig-abc123-PlaceOp"
		thinkBlockJsn = `{"type":"thinking","thinking":"` + thinkText + `","signature":"` + thinkSig + `"}`
	)

	calls := 0
	var second wireBody
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body wireBody
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		if calls == 1 {
			_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","model":"claude-test",` +
				`"content":[` + thinkBlockJsn + `,{"type":"tool_use","id":"` + callID + `","name":"` + toolName + `","input":{"path":"a.go"}}],` +
				`"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`))
			return
		}
		second = body
		_, _ = w.Write([]byte(`{"id":"m2","type":"message","role":"assistant","model":"claude-test",` +
			`"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn",` +
			`"usage":{"input_tokens":20,"output_tokens":3}}`))
	})
	client := fBuildAnthropic(t, base)

	// Turn 1: capture the thinking + tool_use response.
	resp1, err := client.Complete(context.Background(), llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "read a.go")},
		Tools:     []llmkit.ToolDef{{Name: toolName, Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)}},
		MaxTokens: 128,
	})
	if err != nil {
		t.Fatalf("Complete 1: %v", err)
	}
	if len(resp1.Blocks) == 0 || resp1.Blocks[0].Kind != llmkit.BlockThinking {
		t.Fatalf("response blocks = %+v, want a thinking block first", resp1.Blocks)
	}
	if resp1.Blocks[0].Provider != "anthropic" {
		t.Errorf("thinking Provider = %q, want anthropic", resp1.Blocks[0].Provider)
	}
	if len(resp1.ToolCalls) != 1 || resp1.ToolCalls[0].ID != callID {
		t.Fatalf("ToolCalls = %+v, want one %s call", resp1.ToolCalls, callID)
	}
	if resp1.StopReason != llmkit.StopToolUse {
		t.Errorf("StopReason = %q, want tool_use", resp1.StopReason)
	}

	// Turn 2: re-send the assistant turn exactly as the model produced it
	// (thinking block + tool call), followed by the tool result.
	assistant := llmkit.Message{
		Role:      llmkit.RoleAssistant,
		Content:   []llmkit.Block{resp1.Blocks[0]},
		ToolCalls: resp1.ToolCalls,
	}
	toolResult := llmkit.Message{
		Role:       llmkit.RoleToolResult,
		ToolCallID: callID,
		Content:    []llmkit.Block{{Kind: llmkit.BlockText, Text: "package a"}},
	}
	if _, err := client.Complete(context.Background(), llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "read a.go"), assistant, toolResult},
		Tools:     []llmkit.ToolDef{{Name: toolName, Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)}},
		MaxTokens: 128,
	}); err != nil {
		t.Fatalf("Complete 2: %v", err)
	}

	// The second wire request must carry: thinking (verbatim signature),
	// tool_use, then the tool_result in the trailing user turn.
	msgs := second["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("second request has %d messages, want 3", len(msgs))
	}
	assistWire := msgs[1].(map[string]any)["content"].([]any)
	firstBlock := assistWire[0].(map[string]any)
	if firstBlock["type"] != "thinking" {
		t.Fatalf("first assistant wire block type = %v, want thinking", firstBlock["type"])
	}
	if got := firstBlock["signature"]; got != thinkSig {
		t.Errorf("wire signature = %q, want EXACT bytes %q", got, thinkSig)
	}
	if got := firstBlock["thinking"]; got != thinkText {
		t.Errorf("wire thinking = %q, want %q", got, thinkText)
	}
	toolUseWire := assistWire[1].(map[string]any)
	if toolUseWire["type"] != "tool_use" || toolUseWire["id"] != callID {
		t.Errorf("second assistant wire block = %v, want tool_use %s", toolUseWire, callID)
	}
	resultWire := msgs[2].(map[string]any)["content"].([]any)
	tr := resultWire[0].(map[string]any)
	if tr["type"] != "tool_result" || tr["tool_use_id"] != callID {
		t.Errorf("trailing wire block = %v, want tool_result for %s", tr, callID)
	}
}

// fBuildAnthropic builds just the anthropic adapter against baseURL.
func fBuildAnthropic(t *testing.T, baseURL string) llmkit.Client {
	for _, f := range allAdapters() {
		if f.name == "anthropic" {
			return f.build(t, baseURL)
		}
	}
	t.Fatal("anthropic factory missing")
	return nil
}

// TestConformance_ToolChoiceWireShape asserts each ToolChoice mode maps to
// the provider's exact wire field, and that auto/zero is never serialized.
func TestConformance_ToolChoiceWireShape(t *testing.T) {
	cases := []struct {
		name string
		tc   llmkit.ToolChoice
	}{
		{"none", llmkit.ToolChoice{Mode: llmkit.ToolChoiceNone}},
		{"required", llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}},
		{"tool", llmkit.ToolChoice{Mode: llmkit.ToolChoiceTool, Name: "read_file"}},
	}
	for _, f := range allAdapters() {
		for _, c := range cases {
			t.Run(f.name+"/"+c.name, func(t *testing.T) {
				var captured wireBody
				n := 0
				base := newServer(t, captureWireBody(t, &n, func() string {
					return mockTextBody(f.name, "ok", 1, 1)
				}, &captured))
				client := f.build(t, base)

				req := simpleRequest()
				req.ToolChoice = c.tc
				if _, err := client.Complete(context.Background(), req); err != nil {
					t.Fatalf("Complete: %v", err)
				}

				switch f.name {
				case "anthropic":
					got := captured["tool_choice"].(map[string]any)
					switch c.name {
					case "none":
						if got["type"] != "none" {
							t.Errorf("tool_choice = %v, want none", got)
						}
					case "required":
						if got["type"] != "any" {
							t.Errorf("tool_choice = %v, want any", got)
						}
					case "tool":
						if got["type"] != "tool" || got["name"] != "read_file" {
							t.Errorf("tool_choice = %v, want tool/read_file", got)
						}
					}
				case "openai", "openai-compatible":
					switch v := captured["tool_choice"].(type) {
					case string:
						if (c.name == "none" && v != "none") || (c.name == "required" && v != "required") {
							t.Errorf("tool_choice = %q", v)
						}
						if c.name == "tool" {
							t.Errorf("tool_choice = %q, want named function object", v)
						}
					case map[string]any:
						if c.name != "tool" {
							t.Errorf("tool_choice = %v, want plain mode string", v)
							break
						}
						fn := v["function"].(map[string]any)
						if fn["name"] != "read_file" {
							t.Errorf("tool_choice.function = %v, want read_file", fn)
						}
					default:
						t.Fatalf("tool_choice missing on wire")
					}
				case "google":
					tc := captured["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
					switch c.name {
					case "none":
						if tc["mode"] != "NONE" {
							t.Errorf("mode = %v, want NONE", tc["mode"])
						}
					case "required":
						if tc["mode"] != "ANY" {
							t.Errorf("mode = %v, want ANY", tc["mode"])
						}
					case "tool":
						names, _ := tc["allowedFunctionNames"].([]any)
						if tc["mode"] != "ANY" || len(names) != 1 || names[0] != "read_file" {
							t.Errorf("config = %v, want ANY restricted to read_file", tc)
						}
					}
				}
			})
		}

		// Auto (zero value) must NOT appear on the wire for any provider.
		t.Run(f.name+"/auto-absent", func(t *testing.T) {
			var captured wireBody
			n := 0
			base := newServer(t, captureWireBody(t, &n, func() string {
				return mockTextBody(f.name, "ok", 1, 1)
			}, &captured))
			client := f.build(t, base)
			if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			switch f.name {
			case "anthropic":
				if _, ok := captured["tool_choice"]; ok {
					t.Errorf("tool_choice present for auto: %v", captured["tool_choice"])
				}
			case "openai", "openai-compatible":
				if _, ok := captured["tool_choice"]; ok {
					t.Errorf("tool_choice present for auto: %v", captured["tool_choice"])
				}
			case "google":
				if _, ok := captured["toolConfig"]; ok {
					t.Errorf("toolConfig present for auto: %v", captured["toolConfig"])
				}
			}
		})
	}
}

// TestConformance_StopRefusalAndContentFilter asserts the refusal and
// content-filter stop mappings per provider.
func TestConformance_StopRefusalAndContentFilter(t *testing.T) {
	cases := []struct {
		provider string
		body     string
		wantStop llmkit.StopReason
	}{
		{"anthropic", `{"id":"m","type":"message","role":"assistant","model":"claude-test",` +
			`"content":[{"type":"text","text":"no"}],"stop_reason":"refusal","usage":{"input_tokens":1,"output_tokens":1}}`,
			llmkit.StopRefusal},
		{"openai", `{"id":"c","object":"chat.completion","created":1,"model":"gpt-test",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"blocked"},"finish_reason":"content_filter"}],` +
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			llmkit.StopContentFilter},
		{"openai-compatible", `{"id":"c","object":"chat.completion","created":1,"model":"llama-test",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"blocked"},"finish_reason":"content_filter"}],` +
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			llmkit.StopContentFilter},
		{"google", `{"candidates":[{"content":{"role":"model","parts":[{"text":"no"}]},` +
			`"finishReason":"SAFETY"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			llmkit.StopContentFilter},
	}
	for _, c := range cases {
		t.Run(c.provider, func(t *testing.T) {
			var client llmkit.Client
			for _, f := range allAdapters() {
				if f.name == c.provider {
					base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(c.body))
					})
					client = f.build(t, base)
					break
				}
			}
			if client == nil {
				t.Fatalf("no adapter named %s", c.provider)
			}
			resp, err := client.Complete(context.Background(), simpleRequest())
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if resp.StopReason != c.wantStop {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, c.wantStop)
			}
		})
	}
}

// TestConformance_MediaSourceRejectedBeforeWire asserts that image/document
// blocks violating the exactly-one-of-Data/URL rule are rejected with an
// error wrapping ErrInvalidRequest and that ZERO HTTP requests are made.
func TestConformance_MediaSourceRejectedBeforeWire(t *testing.T) {
	bad := []llmkit.Block{
		{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes, URL: "https://x/y.png"},
		{Kind: llmkit.BlockImage, MediaType: "image/png"},
		{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: pdfBytes, URL: "https://x/y.pdf"},
		{Kind: llmkit.BlockDocument, MediaType: "application/pdf"},
	}
	for _, f := range allAdapters() {
		for i, blk := range bad {
			t.Run(f.name+"/"+string(blk.Kind)+"/case"+string(rune('a'+i)), func(t *testing.T) {
				n := 0
				base := newServer(t, captureWireBody(t, &n, func() string {
					return mockTextBody(f.name, "should not be reached", 0, 0)
				}, &wireBody{}))
				client := f.build(t, base)

				req := llmkit.Request{
					Messages: []llmkit.Message{{Role: llmkit.RoleUser, Content: []llmkit.Block{blk}}},
				}
				_, err := client.Complete(context.Background(), req)
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Errorf("error = %v, want ErrInvalidRequest", err)
				}
				if n != 0 {
					t.Errorf("wire calls = %d, want 0 (validation must precede the request)", n)
				}
			})
		}
	}
}

// TestConformance_ForeignThinkingBlockDropped asserts that a thinking block
// whose Provider does not match the adapter is silently dropped, never sent.
func TestConformance_ForeignThinkingBlockDropped(t *testing.T) {
	t.Run("anthropic-drops-google", func(t *testing.T) {
		var captured wireBody
		n := 0
		base := newServer(t, captureWireBody(t, &n, func() string {
			return mockTextBody("anthropic", "ok", 1, 1)
		}, &captured))
		client := fBuildAnthropic(t, base)

		foreign := llmkit.Block{
			Kind:     llmkit.BlockThinking,
			Provider: "google",
			Text:     "gemini thought",
			Raw:      json.RawMessage(`{"thought":true,"text":"gemini thought","thoughtSignature":"c2ln"}`),
		}
		req := llmkit.Request{
			Messages: []llmkit.Message{
				llmkit.TextMessage(llmkit.RoleUser, "hi"),
				{Role: llmkit.RoleAssistant, Content: []llmkit.Block{foreign}},
			},
		}
		if _, err := client.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		msgs := captured["messages"].([]any)
		content := msgs[1].(map[string]any)["content"].([]any)
		for _, b := range content {
			if bm, _ := b.(map[string]any); bm["type"] == "thinking" || bm["type"] == "redacted_thinking" {
				t.Errorf("foreign thinking block reached the anthropic wire: %v", bm)
			}
		}
	})

	t.Run("google-drops-anthropic", func(t *testing.T) {
		var captured wireBody
		n := 0
		base := newServer(t, captureWireBody(t, &n, func() string {
			return mockTextBody("google", "ok", 1, 1)
		}, &captured))
		var client llmkit.Client
		for _, f := range allAdapters() {
			if f.name == "google" {
				client = f.build(t, base)
				break
			}
		}
		foreign := llmkit.Block{
			Kind:     llmkit.BlockThinking,
			Provider: "anthropic",
			Text:     "claude thought",
			Raw:      json.RawMessage(`{"type":"thinking","thinking":"claude thought","signature":"sig"}`),
		}
		req := llmkit.Request{
			Messages: []llmkit.Message{
				llmkit.TextMessage(llmkit.RoleUser, "hi"),
				{Role: llmkit.RoleAssistant, Content: []llmkit.Block{foreign}},
			},
		}
		if _, err := client.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		contents := captured["contents"].([]any)
		parts := contents[1].(map[string]any)["parts"].([]any)
		for _, p := range parts {
			pm, _ := p.(map[string]any)
			if thought, ok := pm["thought"]; ok && thought == true {
				t.Errorf("foreign thinking block reached the google wire: %v", pm)
			}
			if txt, _ := pm["text"].(string); strings.Contains(txt, "claude thought") {
				t.Errorf("foreign thinking text reached the google wire: %v", pm)
			}
		}
	})
}

// TestConformance_ResponseBlocksCarryText asserts Response.Blocks carries
// the response text as ordered text blocks alongside everything else.
func TestConformance_ResponseBlocksCarryText(t *testing.T) {
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(mockTextBody(f.name, "hello world", 12, 5)))
			})
			client := f.build(t, base)
			resp, err := client.Complete(context.Background(), simpleRequest())
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if len(resp.Blocks) == 0 || resp.Blocks[0].Kind != llmkit.BlockText || resp.Blocks[0].Text != "hello world" {
				t.Errorf("Blocks = %+v, want one text block %q", resp.Blocks, "hello world")
			}
		})
	}
}

package provider

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

type trmProfile struct {
	name  string
	typ   Type
	model string
	media bool
}

var trmProfiles = []trmProfile{
	{"anthropic", TypeAnthropic, "claude-opus-4-5", true},
	{"google/gemini-3", TypeGoogle, "gemini-3-flash-preview", true},
	{"google/gemini-2.5", TypeGoogle, "gemini-2.5-pro", false},
	{"openai", TypeOpenAI, "gpt-4o", false},
	{"compat", TypeOpenAICompatible, "llama3", false},
}

func trmRequest(blocks ...llmkit.Block) llmkit.Request {
	return llmkit.Request{
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "snap"),
			{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "call_1", Name: "snap", Arguments: json.RawMessage(`{}`)}}},
			{Role: llmkit.RoleToolResult, ToolCallID: "call_1", Content: blocks},
		},
		MaxTokens: 64,
	}
}

func trmCall(t *testing.T, p trmProfile, override func(llmkit.Capabilities) llmkit.Capabilities, req llmkit.Request) (map[string]any, error, int) {
	t.Helper()
	isolateEnv(t, baseURLEnvSources...)
	var body map[string]any
	hits := 0
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits++
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okBody(string(p.typ))))
	})
	client, err := New(context.Background(), Spec{Type: p.typ, Model: p.model, BaseURL: base, Secret: "k", Capabilities: override}, Options{})
	if err != nil {
		return nil, err, hits
	}
	_, err = client.Complete(context.Background(), req)
	return body, err, hits
}

func trmLast(t *testing.T, body map[string]any, key string) map[string]any {
	t.Helper()
	list, _ := body[key].([]any)
	if len(list) == 0 {
		t.Fatalf("wire body has no %s: %v", key, body)
	}
	m, _ := list[len(list)-1].(map[string]any)
	return m
}

func trmAnthropicContent(t *testing.T, body map[string]any) []any {
	t.Helper()
	blocks, _ := trmLast(t, body, "messages")["content"].([]any)
	if len(blocks) != 1 {
		t.Fatalf("last message content = %v, want one tool_result block", blocks)
	}
	tr, _ := blocks[0].(map[string]any)
	if tr["type"] != "tool_result" || tr["tool_use_id"] != "call_1" {
		t.Fatalf("block = %v, want tool_result for call_1", tr)
	}
	content, _ := tr["content"].([]any)
	return content
}

func trmGoogleResponse(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	parts, _ := trmLast(t, body, "contents")["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("last content parts = %v, want one functionResponse part", parts)
	}
	fr, _ := parts[0].(map[string]any)["functionResponse"].(map[string]any)
	if fr == nil || fr["name"] != "snap" {
		t.Fatalf("part = %v, want functionResponse named snap", parts[0])
	}
	return fr
}

func TestToolResultMedia_CapabilityTable(t *testing.T) {
	for _, p := range trmProfiles {
		client, err := New(context.Background(), Spec{Type: p.typ, Model: p.model, BaseURL: "http://example.invalid", Secret: "k"}, Options{})
		if err != nil {
			t.Fatalf("New(%s): %v", p.name, err)
		}
		if got := client.Capabilities().ToolResultMedia; got != p.media {
			t.Errorf("%s: ToolResultMedia = %v, want %v", p.name, got, p.media)
		}
	}
}

func TestToolResultMedia_AnthropicWire(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString(pngBytes)
	pdfB64 := base64.StdEncoding.EncodeToString(pdfBytes)
	p := trmProfiles[0]

	t.Run("text-image", func(t *testing.T) {
		body, err, hits := trmCall(t, p, nil, trmRequest(
			llmkit.Text("cap"), llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes}))
		if err != nil || hits != 1 {
			t.Fatalf("err = %v, hits = %d", err, hits)
		}
		c := trmAnthropicContent(t, body)
		if len(c) != 2 {
			t.Fatalf("content = %v, want text then image", c)
		}
		if b := c[0].(map[string]any); b["type"] != "text" || b["text"] != "cap" {
			t.Errorf("content[0] = %v, want text cap", b)
		}
		img := c[1].(map[string]any)
		src, _ := img["source"].(map[string]any)
		if img["type"] != "image" || src["type"] != "base64" || src["media_type"] != "image/png" || src["data"] != pngB64 {
			t.Errorf("content[1] = %v, want base64 png image", img)
		}
	})

	t.Run("order-preserved", func(t *testing.T) {
		body, err, _ := trmCall(t, p, nil, trmRequest(
			llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes}, llmkit.Text("tail")))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		c := trmAnthropicContent(t, body)
		if len(c) != 2 || c[0].(map[string]any)["type"] != "image" || c[1].(map[string]any)["type"] != "text" {
			t.Errorf("content = %v, want image then text", c)
		}
	})

	t.Run("document", func(t *testing.T) {
		body, err, _ := trmCall(t, p, nil, trmRequest(
			llmkit.Text("cap"), llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: pdfBytes}))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		c := trmAnthropicContent(t, body)
		if len(c) != 2 {
			t.Fatalf("content = %v, want text then document", c)
		}
		doc := c[1].(map[string]any)
		src, _ := doc["source"].(map[string]any)
		if doc["type"] != "document" || src["type"] != "base64" || src["media_type"] != "application/pdf" || src["data"] != pdfB64 {
			t.Errorf("content[1] = %v, want base64 pdf document", doc)
		}
	})

	t.Run("url-sources", func(t *testing.T) {
		body, err, _ := trmCall(t, p, nil, trmRequest(
			llmkit.Block{Kind: llmkit.BlockImage, URL: "https://x/y.png"},
			llmkit.Block{Kind: llmkit.BlockDocument, URL: "https://x/y.pdf"}))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		c := trmAnthropicContent(t, body)
		if len(c) != 2 {
			t.Fatalf("content = %v, want image then document", c)
		}
		for i, want := range []string{"https://x/y.png", "https://x/y.pdf"} {
			src, _ := c[i].(map[string]any)["source"].(map[string]any)
			if src["type"] != "url" || src["url"] != want {
				t.Errorf("content[%d] source = %v, want url %s", i, src, want)
			}
		}
	})

	t.Run("text-only-shape-unchanged", func(t *testing.T) {
		body, err, _ := trmCall(t, p, nil, trmRequest(llmkit.Text("a"), llmkit.Text("b")))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		c := trmAnthropicContent(t, body)
		if len(c) != 1 || c[0].(map[string]any)["text"] != "ab" {
			t.Errorf("content = %v, want one joined text block", c)
		}
	})

	t.Run("non-pdf-document-refused", func(t *testing.T) {
		_, err, hits := trmCall(t, p, nil, trmRequest(
			llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "text/plain", Data: []byte("x")}))
		if !errors.Is(err, llmkit.ErrInvalidRequest) || hits != 0 {
			t.Errorf("err = %v, hits = %d, want ErrInvalidRequest with zero hits", err, hits)
		}
	})
}

func TestToolResultMedia_GoogleWire(t *testing.T) {
	p := trmProfiles[1]

	t.Run("text-image", func(t *testing.T) {
		body, err, hits := trmCall(t, p, nil, trmRequest(
			llmkit.Text("cap"), llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes}))
		if err != nil || hits != 1 {
			t.Fatalf("err = %v, hits = %d", err, hits)
		}
		fr := trmGoogleResponse(t, body)
		if out, _ := fr["response"].(map[string]any); out["output"] != "cap" {
			t.Errorf("response = %v, want output cap", fr["response"])
		}
		parts, _ := fr["parts"].([]any)
		if len(parts) != 1 {
			t.Fatalf("parts = %v, want one inlineData part", parts)
		}
		in, _ := parts[0].(map[string]any)["inlineData"].(map[string]any)
		if in["mimeType"] != "image/png" || in["data"] != base64.StdEncoding.EncodeToString(pngBytes) {
			t.Errorf("inlineData = %v, want png bytes", in)
		}
	})

	t.Run("document", func(t *testing.T) {
		body, err, _ := trmCall(t, p, nil, trmRequest(
			llmkit.Text("cap"), llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: pdfBytes}))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		parts, _ := trmGoogleResponse(t, body)["parts"].([]any)
		if len(parts) != 1 {
			t.Fatalf("parts = %v, want one inlineData part", parts)
		}
		in, _ := parts[0].(map[string]any)["inlineData"].(map[string]any)
		if in["mimeType"] != "application/pdf" || in["data"] != base64.StdEncoding.EncodeToString(pdfBytes) {
			t.Errorf("inlineData = %v, want pdf bytes", in)
		}
	})

	t.Run("accepted-types-in-order", func(t *testing.T) {
		types := []string{"image/png", "image/jpeg", "image/webp", "application/pdf", "text/plain"}
		blocks := make([]llmkit.Block, 0, len(types))
		for _, mt := range types {
			kind := llmkit.BlockImage
			if !strings.HasPrefix(mt, "image/") {
				kind = llmkit.BlockDocument
			}
			blocks = append(blocks, llmkit.Block{Kind: kind, MediaType: mt, Data: []byte("x")})
		}
		body, err, _ := trmCall(t, p, nil, trmRequest(blocks...))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		parts, _ := trmGoogleResponse(t, body)["parts"].([]any)
		if len(parts) != len(types) {
			t.Fatalf("parts = %v, want %d", parts, len(types))
		}
		for i, mt := range types {
			in, _ := parts[i].(map[string]any)["inlineData"].(map[string]any)
			if in["mimeType"] != mt {
				t.Errorf("parts[%d] mimeType = %v, want %s", i, in["mimeType"], mt)
			}
		}
	})

	t.Run("text-only-has-no-parts", func(t *testing.T) {
		body, err, _ := trmCall(t, p, nil, trmRequest(llmkit.Text("cap")))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if parts, has := trmGoogleResponse(t, body)["parts"]; has {
			t.Errorf("parts = %v on a text-only result, want absent", parts)
		}
	})
}

func TestToolResultMedia_GoogleUnsupportedRefused(t *testing.T) {
	p := trmProfiles[1]
	cases := []struct {
		name string
		blk  llmkit.Block
	}{
		{"gif", llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/gif", Data: []byte("GIF89a")}},
		{"markdown", llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "text/markdown", Data: []byte("#")}},
		{"url-image", llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", URL: "https://x/y.png"}},
		{"url-pdf", llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "application/pdf", URL: "https://x/y.pdf"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err, hits := trmCall(t, p, nil, trmRequest(llmkit.Text("cap"), c.blk))
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("err = %v, want ErrInvalidRequest", err)
			}
			if hits != 0 {
				t.Errorf("wire hits = %d, want 0", hits)
			}
		})
	}
}

func TestToolResultMedia_RefusedByCapability(t *testing.T) {
	blocks := map[string]llmkit.Block{
		"image":    {Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes},
		"document": {Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: pdfBytes},
	}
	for _, p := range trmProfiles {
		if p.media {
			continue
		}
		for kind, blk := range blocks {
			t.Run(p.name+"/"+kind, func(t *testing.T) {
				_, err, hits := trmCall(t, p, nil, trmRequest(llmkit.Text("cap"), blk))
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Fatalf("err = %v, want ErrInvalidRequest", err)
				}
				if hits != 0 {
					t.Errorf("wire hits = %d, want 0", hits)
				}
				if !strings.Contains(err.Error(), "ToolResultMedia") || strings.Contains(err.Error(), "not allowed in a") {
					t.Errorf("err = %q, want it to name ToolResultMedia, not the per-role rule", err)
				}
			})
		}
	}
}

func TestToolResultMedia_OverrideOffRefusesAnthropic(t *testing.T) {
	off := func(c llmkit.Capabilities) llmkit.Capabilities { c.ToolResultMedia = false; return c }
	_, err, hits := trmCall(t, trmProfiles[0], off, trmRequest(
		llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes}))
	if !errors.Is(err, llmkit.ErrInvalidRequest) || hits != 0 || !strings.Contains(err.Error(), "ToolResultMedia") {
		t.Errorf("err = %v, hits = %d, want ErrInvalidRequest naming ToolResultMedia with zero hits", err, hits)
	}
}

func TestToolResultMedia_CeilingRefusesOpenAIFamily(t *testing.T) {
	on := func(c llmkit.Capabilities) llmkit.Capabilities { c.ToolResultMedia = true; return c }
	for _, p := range trmProfiles {
		if p.typ != TypeOpenAI && p.typ != TypeOpenAICompatible {
			continue
		}
		t.Run(p.name, func(t *testing.T) {
			_, err, hits := trmCall(t, p, on, trmRequest(
				llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes}))
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("err = %v, want ErrInvalidRequest from New", err)
			}
			if hits != 0 {
				t.Errorf("wire hits = %d, want 0", hits)
			}
			for _, want := range []string{"ToolResultMedia", "ceiling"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("err = %q, want it to contain %q", err, want)
				}
			}
		})
	}
}

func TestToolResultMedia_SystemMediaStillRefused(t *testing.T) {
	for _, p := range trmProfiles {
		t.Run(p.name, func(t *testing.T) {
			req := llmkit.Request{Messages: []llmkit.Message{
				{Role: llmkit.RoleSystem, Content: []llmkit.Block{{Kind: llmkit.BlockImage, MediaType: "image/png", Data: pngBytes}}},
			}}
			_, err, hits := trmCall(t, p, nil, req)
			if !errors.Is(err, llmkit.ErrInvalidRequest) || hits != 0 {
				t.Errorf("err = %v, hits = %d, want ErrInvalidRequest with zero hits", err, hits)
			}
		})
	}
}

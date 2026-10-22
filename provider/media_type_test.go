package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

var (
	mtImageSet = []string{"image/png", "image/jpeg", "image/gif", "image/webp"}
	mtPDFSet   = []string{"application/pdf"}

	mtGoogleInline = []string{
		"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif",
		"application/pdf", "text/html", "text/css", "text/plain", "text/xml", "text/csv", "text/rtf",
		"text/javascript", "application/json",
		"audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac", "audio/mpeg",
		"audio/m4a", "audio/l16", "audio/opus",
		"video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm",
		"video/wmv", "video/3gpp",
	}
	mtGoogleURL = []string{
		"text/html", "text/css", "text/plain", "text/xml", "text/csv", "text/rtf",
		"text/javascript", "application/json", "application/pdf",
		"image/bmp", "image/jpeg", "image/png", "image/webp",
		"video/mp4", "video/mpeg", "video/quicktime", "video/avi", "video/x-flv", "video/mpg",
		"video/webm", "video/wmv", "video/3gpp",
	}
	mtGoogleToolSet = []string{"image/png", "image/jpeg", "image/webp", "application/pdf", "text/plain"}

	// mtProbeExtras joins every wire's accepted set in the probe: near
	// misses, other families, and spellings that differ from a member only
	// in case or parameters.
	mtProbeExtras = []string{
		"image/bmp", "image/tiff", "image/svg+xml", "image/heic", "image/gif", "application/pdf",
		"text/plain", "image/jpg", "IMAGE/PNG", "image/png; x=y", "audio/wav",
	}
)

type mtRecorder struct {
	mu   sync.Mutex
	hits int
	body map[string]any
}

func (r *mtRecorder) reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.hits, r.body = 0, nil
}

func (r *mtRecorder) snapshot() (int, map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.hits, r.body
}

// mtList returns body[key] as a slice of objects.
func mtList(t *testing.T, v any, what string) []map[string]any {
	t.Helper()
	raw, _ := v.([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, e := range raw {
		m, _ := e.(map[string]any)
		out = append(out, m)
	}
	if len(out) == 0 {
		t.Fatalf("wire body has no %s", what)
	}
	return out
}

func mtObj(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func mtUserRequest(b llmkit.Block) llmkit.Request {
	return llmkit.Request{
		Messages:  []llmkit.Message{{Role: llmkit.RoleUser, Content: []llmkit.Block{llmkit.Text("look"), b}}},
		MaxTokens: 64,
	}
}

func mtToolResultRequest(b llmkit.Block) llmkit.Request {
	return llmkit.Request{
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "snap"),
			{Role: llmkit.RoleAssistant, ToolCalls: []llmkit.ToolCall{{ID: "call_1", Name: "snap", Arguments: json.RawMessage(`{}`)}}},
			{Role: llmkit.RoleToolResult, ToolCallID: "call_1", Content: []llmkit.Block{llmkit.Text("cap"), b}},
		},
		MaxTokens: 64,
	}
}

// mtBlock builds the block under test: inline bytes, or a URL source.
func mtBlock(kind llmkit.BlockKind, url bool, m string) llmkit.Block {
	if url {
		return llmkit.Block{Kind: kind, MediaType: m, URL: "https://example.test/media"}
	}
	return llmkit.Block{Kind: kind, MediaType: m, Data: []byte("bytes")}
}

// mtForwardedData returns the media type of a data: URL.
func mtForwardedData(url string) string {
	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		return "<not a data URL: " + url + ">"
	}
	m, _, _ := strings.Cut(rest, ";base64,")
	return m
}

// mtForwarded extractors return the string a wire wrote into its media-type
// slot for the one media block of the request.
func mtAnthropicUser(t *testing.T, body map[string]any) string {
	t.Helper()
	msgs := mtList(t, body["messages"], "messages")
	for _, b := range mtList(t, msgs[len(msgs)-1]["content"], "content") {
		if b["type"] == "image" || b["type"] == "document" {
			s, _ := mtObj(b["source"])["media_type"].(string)
			return s
		}
	}
	t.Fatalf("no media block in %v", body)
	return ""
}

func mtAnthropicToolResult(t *testing.T, body map[string]any) string {
	t.Helper()
	msgs := mtList(t, body["messages"], "messages")
	tr := mtList(t, msgs[len(msgs)-1]["content"], "content")[0]
	for _, b := range mtList(t, tr["content"], "tool_result content") {
		if b["type"] == "image" || b["type"] == "document" {
			s, _ := mtObj(b["source"])["media_type"].(string)
			return s
		}
	}
	t.Fatalf("no media block in %v", body)
	return ""
}

func mtOpenAI(field string) func(*testing.T, map[string]any) string {
	return func(t *testing.T, body map[string]any) string {
		t.Helper()
		msgs := mtList(t, body["messages"], "messages")
		for _, p := range mtList(t, msgs[len(msgs)-1]["content"], "content") {
			switch {
			case field == "image_url" && p["type"] == "image_url":
				s, _ := mtObj(p["image_url"])["url"].(string)
				return mtForwardedData(s)
			case field == "file" && p["type"] == "file":
				s, _ := mtObj(p["file"])["file_data"].(string)
				return mtForwardedData(s)
			}
		}
		t.Fatalf("no %s part in %v", field, body)
		return ""
	}
}

func mtGoogleUser(field string) func(*testing.T, map[string]any) string {
	return func(t *testing.T, body map[string]any) string {
		t.Helper()
		contents := mtList(t, body["contents"], "contents")
		for _, p := range mtList(t, contents[len(contents)-1]["parts"], "parts") {
			if d := mtObj(p[field]); d != nil {
				s, _ := d["mimeType"].(string)
				return s
			}
		}
		t.Fatalf("no %s part in %v", field, body)
		return ""
	}
}

func mtGoogleToolResult(t *testing.T, body map[string]any) string {
	t.Helper()
	contents := mtList(t, body["contents"], "contents")
	part := mtList(t, contents[len(contents)-1]["parts"], "parts")[0]
	parts := mtList(t, mtObj(part["functionResponse"])["parts"], "functionResponse parts")
	s, _ := mtObj(parts[0]["inlineData"])["mimeType"].(string)
	return s
}

type mtCase struct {
	name string
	spec Spec
	// accepted is the wire's set; nil means the wire does no check and every
	// probed type must reach the server.
	accepted  []string
	kind      llmkit.BlockKind
	url       bool
	toolRes   bool
	forwarded func(*testing.T, map[string]any) string
}

func mtCases() []mtCase {
	anth := Spec{Type: TypeAnthropic, Model: "claude-opus-4-5"}
	oai := Spec{Type: TypeOpenAI, Model: "gpt-4o"}
	compat := Spec{Type: TypeOpenAICompatible, Model: "llama3"}
	goog := Spec{Type: TypeGoogle, Model: "gemini-3-flash-preview"}
	img, doc := llmkit.BlockImage, llmkit.BlockDocument
	return []mtCase{
		{name: "anthropic/image/user", spec: anth, accepted: mtImageSet, kind: img, forwarded: mtAnthropicUser},
		{name: "anthropic/document/user", spec: anth, accepted: mtPDFSet, kind: doc, forwarded: mtAnthropicUser},
		{name: "anthropic/image/tool-result", spec: anth, accepted: mtImageSet, kind: img, toolRes: true, forwarded: mtAnthropicToolResult},
		{name: "anthropic/document/tool-result", spec: anth, accepted: mtPDFSet, kind: doc, toolRes: true, forwarded: mtAnthropicToolResult},
		{name: "openai/image/user", spec: oai, accepted: mtImageSet, kind: img, forwarded: mtOpenAI("image_url")},
		{name: "openai/document/user", spec: oai, accepted: mtPDFSet, kind: doc, forwarded: mtOpenAI("file")},
		{name: "openai-compatible/image/user", spec: compat, kind: img, forwarded: mtOpenAI("image_url")},
		{name: "openai-compatible/document/user", spec: compat, kind: doc, forwarded: mtOpenAI("file")},
		{name: "google/image/user-inline", spec: goog, accepted: mtGoogleInline, kind: img, forwarded: mtGoogleUser("inlineData")},
		{name: "google/document/user-inline", spec: goog, accepted: mtGoogleInline, kind: doc, forwarded: mtGoogleUser("inlineData")},
		{name: "google/image/user-url", spec: goog, accepted: mtGoogleURL, kind: img, url: true, forwarded: mtGoogleUser("fileData")},
		{name: "google/document/user-url", spec: goog, accepted: mtGoogleURL, kind: doc, url: true, forwarded: mtGoogleUser("fileData")},
		{name: "google/image/tool-result", spec: goog, accepted: mtGoogleToolSet, kind: img, toolRes: true, forwarded: mtGoogleToolResult},
		{name: "google/document/tool-result", spec: goog, accepted: mtGoogleToolSet, kind: doc, toolRes: true, forwarded: mtGoogleToolResult},
	}
}

// mtProbe is accepted plus the shared extras, deduplicated, in order. A wire
// without a check probes the first-party sets so the test still names the
// types a first-party check would refuse.
func mtProbe(c mtCase) []string {
	base := c.accepted
	if base == nil {
		base = append(append([]string{}, mtImageSet...), mtPDFSet...)
	}
	seen := make(map[string]bool)
	var out []string
	for _, m := range append(append([]string{}, base...), mtProbeExtras...) {
		if !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	return out
}

func TestMediaTypeAcceptedSetsPerWire(t *testing.T) {
	for _, c := range mtCases() {
		t.Run(c.name, func(t *testing.T) {
			isolateEnv(t, baseURLEnvSources...)
			rec := &mtRecorder{}
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				var body map[string]any
				_ = json.Unmarshal(raw, &body)
				rec.mu.Lock()
				rec.hits++
				rec.body = body
				rec.mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(okBody(string(c.spec.Type))))
			})
			spec := c.spec
			spec.BaseURL, spec.Secret = base, "k"
			client, err := New(context.Background(), spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			accepted := make(map[string]bool, len(c.accepted))
			for _, m := range c.accepted {
				accepted[m] = true
			}
			for _, m := range mtProbe(c) {
				rec.reset()
				req := mtUserRequest(mtBlock(c.kind, c.url, m))
				if c.toolRes {
					req = mtToolResultRequest(mtBlock(c.kind, c.url, m))
				}
				_, err := client.Complete(context.Background(), req)
				hits, body := rec.snapshot()
				if c.accepted == nil || accepted[m] {
					if err != nil || hits != 1 {
						t.Errorf("%q: err = %v, hits = %d, want a request on the wire", m, err, hits)
						continue
					}
					want := m
					if c.name == "anthropic/document/user" || c.name == "anthropic/document/tool-result" {
						want = "application/pdf" // the SDK's base64 PDF source fixes its own media_type
					}
					if got := c.forwarded(t, body); got != want {
						t.Errorf("%q: forwarded media type = %q, want %q", m, got, want)
					}
					continue
				}
				var apiErr *llmkit.APIError
				switch {
				case hits != 0:
					t.Errorf("%q: %d requests reached the server, want 0", m, hits)
				case !errors.Is(err, llmkit.ErrInvalidRequest) || !errors.As(err, &apiErr):
					t.Errorf("%q: err = %v, want an APIError wrapping ErrInvalidRequest", m, err)
				case apiErr.StatusCode != 0:
					t.Errorf("%q: StatusCode = %d, want 0 (refused before the wire)", m, apiErr.StatusCode)
				case !strings.Contains(err.Error(), m):
					t.Errorf("%q: error %q does not name the media type", m, err)
				}
			}
		})
	}
}

// TestMediaTypeGoogleURLWithoutMediaType pins that a URL block with no
// MediaType still reaches the wire with an empty mimeType.
func TestMediaTypeGoogleURLWithoutMediaType(t *testing.T) {
	isolateEnv(t, baseURLEnvSources...)
	rec := &mtRecorder{}
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		rec.mu.Lock()
		rec.hits++
		rec.body = body
		rec.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okBody("google")))
	})
	client, err := New(context.Background(), Spec{Type: TypeGoogle, Model: "gemini-3-flash-preview", BaseURL: base, Secret: "k"}, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Complete(context.Background(), mtUserRequest(llmkit.ImageURL("https://example.test/a.png")))
	hits, body := rec.snapshot()
	if err != nil || hits != 1 {
		t.Fatalf("err = %v, hits = %d, want one request", err, hits)
	}
	contents := mtList(t, body["contents"], "contents")
	var found bool
	for _, p := range mtList(t, contents[len(contents)-1]["parts"], "parts") {
		if fd := mtObj(p["fileData"]); fd != nil {
			found = true
			if fd["fileUri"] != "https://example.test/a.png" {
				t.Errorf("fileData = %v, want the URL forwarded", fd)
			}
		}
	}
	if !found {
		t.Errorf("no fileData part in %v", body)
	}
}

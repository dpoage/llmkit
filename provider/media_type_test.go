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
		"audio/m4a", "audio/l16", "audio/opus", "audio/alaw", "audio/mulaw", "audio/webm",
		"video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm",
		"video/wmv", "video/3gpp",
	}
	mtGoogleToolSet = []string{"image/png", "image/jpeg", "image/webp", "application/pdf", "text/plain"}

	// mtProbeExtras are near misses, other families, and spellings differing
	// from a member only in case or parameters.
	mtProbeExtras = []string{
		"image/bmp", "image/tiff", "image/svg+xml", "image/jpg", "IMAGE/PNG", "image/png; x=y",
		"video/quicktime", "audio/x-wav",
	}

	mtHTTPSURL   = "https://example.test/media"
	mtFilesURI   = "https://generativelanguage.googleapis.com/v1beta/files/abc123"
	mtGCSURI     = "gs://bucket/clip.wav"
	mtBytesB64   = "Ynl0ZXM=" // base64("bytes")
	mtAnthSpec   = Spec{Type: TypeAnthropic, Model: "claude-opus-4-5"}
	mtOpenAISpec = Spec{Type: TypeOpenAI, Model: "gpt-4o"}
	mtCompatSpec = Spec{Type: TypeOpenAICompatible, Model: "llama3"}
	mtGoogleSpec = Spec{Type: TypeGoogle, Model: "gemini-3-flash-preview"}
)

// mtProbeSet is every wire's accepted set plus the extras, deduplicated.
func mtProbeSet() []string {
	seen := make(map[string]bool)
	var out []string
	for _, set := range [][]string{mtImageSet, mtPDFSet, mtGoogleInline, mtGoogleToolSet, mtProbeExtras} {
		for _, m := range set {
			if !seen[m] {
				seen[m] = true
				out = append(out, m)
			}
		}
	}
	return out
}

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

// mtClient builds a client for spec against a recording fake server.
func mtClient(t *testing.T, spec Spec) (llmkit.Client, *mtRecorder) {
	t.Helper()
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
		_, _ = w.Write([]byte(okBody(string(spec.Type))))
	})
	spec.BaseURL, spec.Secret = base, "k"
	client, err := New(context.Background(), spec, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return client, rec
}

// mtList returns v as a slice of objects.
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

func mtDataURL(m string) string { return "data:" + m + ";base64," + mtBytesB64 }

// mtDataType is the media type of mtDataURL(m): m up to its first ';' or ','.
func mtDataType(m string) string {
	if i := strings.IndexAny(m, ";,"); i >= 0 {
		return m[:i]
	}
	return m
}

// Extractors return what the wire wrote for the request's one media block.
func mtAnthropic(toolRes bool, field string) func(*testing.T, map[string]any) string {
	return func(t *testing.T, body map[string]any) string {
		t.Helper()
		msgs := mtList(t, body["messages"], "messages")
		blocks := mtList(t, msgs[len(msgs)-1]["content"], "content")
		if toolRes {
			blocks = mtList(t, blocks[0]["content"], "tool_result content")
		}
		for _, b := range blocks {
			if b["type"] == "image" || b["type"] == "document" {
				s, _ := mtObj(b["source"])[field].(string)
				return s
			}
		}
		t.Fatalf("no media block in %v", body)
		return ""
	}
}

func mtOpenAI(part, field string) func(*testing.T, map[string]any) string {
	return func(t *testing.T, body map[string]any) string {
		t.Helper()
		msgs := mtList(t, body["messages"], "messages")
		for _, p := range mtList(t, msgs[len(msgs)-1]["content"], "content") {
			if p["type"] == part {
				s, _ := mtObj(p[part])[field].(string)
				return s
			}
		}
		t.Fatalf("no %s part in %v", part, body)
		return ""
	}
}

func mtGoogleUserParts(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	contents := mtList(t, body["contents"], "contents")
	return mtList(t, contents[len(contents)-1]["parts"], "parts")
}

func mtGoogleInlineData(t *testing.T, body map[string]any) string {
	t.Helper()
	for _, p := range mtGoogleUserParts(t, body) {
		if d := mtObj(p["inlineData"]); d != nil {
			s, _ := d["mimeType"].(string)
			return s
		}
	}
	t.Fatalf("no inlineData part in %v", body)
	return ""
}

// mtGoogleFileData returns "fileUri|mimeType".
func mtGoogleFileData(t *testing.T, body map[string]any) string {
	t.Helper()
	for _, p := range mtGoogleUserParts(t, body) {
		if d := mtObj(p["fileData"]); d != nil {
			uri, _ := d["fileUri"].(string)
			mt, _ := d["mimeType"].(string)
			return uri + "|" + mt
		}
	}
	t.Fatalf("no fileData part in %v", body)
	return ""
}

func mtGoogleToolResult(t *testing.T, body map[string]any) string {
	t.Helper()
	contents := mtList(t, body["contents"], "contents")
	part := mtList(t, contents[len(contents)-1]["parts"], "parts")[0]
	parts := mtList(t, mtObj(part["functionResponse"])["parts"], "functionResponse parts")
	s, _ := mtObj(parts[0]["inlineData"])["mimeType"].(string)
	return s
}

func mtWantMediaType(b llmkit.Block) string { return b.MediaType }
func mtWantPDF(llmkit.Block) string         { return "application/pdf" } // the SDK fixes a base64 PDF's media_type
func mtWantURL(b llmkit.Block) string       { return b.URL }
func mtWantDataURI(b llmkit.Block) string   { return mtDataURL(b.MediaType) }
func mtWantFileData(b llmkit.Block) string  { return b.URL + "|" + b.MediaType }

type mtSource int

const (
	mtInline    mtSource = iota // Data + MediaType m
	mtData                      // URL mtDataURL(m), no MediaType
	mtDataTyped                 // URL mtDataURL(m) and MediaType m
	mtRemote                    // URL mtCase.url, MediaType m
)

type mtCase struct {
	name string
	spec Spec
	kind llmkit.BlockKind
	src  mtSource
	url  string
	req  func(llmkit.Block) llmkit.Request
	// accepted is the set the wire checks against; nil means every probe passes.
	accepted []string
	extract  func(*testing.T, map[string]any) string
	want     func(llmkit.Block) string
}

func (c mtCase) block(m string) llmkit.Block {
	switch c.src {
	case mtInline:
		return llmkit.Block{Kind: c.kind, MediaType: m, Data: []byte("bytes")}
	case mtData:
		return llmkit.Block{Kind: c.kind, URL: mtDataURL(m)}
	case mtDataTyped:
		return llmkit.Block{Kind: c.kind, MediaType: m, URL: mtDataURL(m)}
	default:
		return llmkit.Block{Kind: c.kind, MediaType: m, URL: c.url}
	}
}

// refused returns the type the wire must refuse and name for probe m, or
// ok=false when m must reach the server.
func (c mtCase) refused(accepted map[string]bool, m string) (string, bool) {
	if c.accepted == nil {
		return "", false
	}
	switch c.src {
	case mtInline:
		return m, !accepted[m]
	case mtData, mtDataTyped:
		if dt := mtDataType(m); !accepted[dt] {
			return dt, true
		}
		return m, c.src == mtDataTyped && !accepted[m]
	}
	return "", false
}

func mtCases() []mtCase {
	img, doc := llmkit.BlockImage, llmkit.BlockDocument
	var cs []mtCase
	for _, pos := range []struct {
		name    string
		req     func(llmkit.Block) llmkit.Request
		toolRes bool
	}{{"user", mtUserRequest, false}, {"tool-result", mtToolResultRequest, true}} {
		for _, k := range []struct {
			kind   llmkit.BlockKind
			set    []string
			inline func(llmkit.Block) string
		}{{img, mtImageSet, mtWantMediaType}, {doc, mtPDFSet, mtWantPDF}} {
			n := "anthropic/" + string(k.kind) + "/" + pos.name
			mt, url := mtAnthropic(pos.toolRes, "media_type"), mtAnthropic(pos.toolRes, "url")
			cs = append(cs,
				mtCase{name: n + "/inline", spec: mtAnthSpec, kind: k.kind, src: mtInline, req: pos.req, accepted: k.set, extract: mt, want: k.inline},
				mtCase{name: n + "/data-url", spec: mtAnthSpec, kind: k.kind, src: mtData, req: pos.req, accepted: k.set, extract: url, want: mtWantURL},
				mtCase{name: n + "/https-url", spec: mtAnthSpec, kind: k.kind, src: mtRemote, url: mtHTTPSURL, req: pos.req, extract: url, want: mtWantURL},
			)
		}
	}
	for _, w := range []struct {
		name string
		spec Spec
		img  []string
		pdf  []string
	}{{"openai", mtOpenAISpec, mtImageSet, mtPDFSet}, {"openai-compatible", mtCompatSpec, nil, nil}} {
		imgURL := mtOpenAI("image_url", "url")
		cs = append(cs,
			mtCase{name: w.name + "/image/user/inline", spec: w.spec, kind: img, src: mtInline, req: mtUserRequest, accepted: w.img, extract: imgURL, want: mtWantDataURI},
			mtCase{name: w.name + "/image/user/data-url", spec: w.spec, kind: img, src: mtData, req: mtUserRequest, accepted: w.img, extract: imgURL, want: mtWantURL},
			mtCase{name: w.name + "/image/user/https-url", spec: w.spec, kind: img, src: mtRemote, url: mtHTTPSURL, req: mtUserRequest, extract: imgURL, want: mtWantURL},
			mtCase{name: w.name + "/document/user/inline", spec: w.spec, kind: doc, src: mtInline, req: mtUserRequest, accepted: w.pdf, extract: mtOpenAI("file", "file_data"), want: mtWantDataURI},
		)
	}
	for _, kind := range []llmkit.BlockKind{img, doc} {
		n := "google/" + string(kind)
		cs = append(cs,
			mtCase{name: n + "/user/inline", spec: mtGoogleSpec, kind: kind, src: mtInline, req: mtUserRequest, accepted: mtGoogleInline, extract: mtGoogleInlineData, want: mtWantMediaType},
			mtCase{name: n + "/user/data-url", spec: mtGoogleSpec, kind: kind, src: mtData, req: mtUserRequest, accepted: mtGoogleInline, extract: mtGoogleFileData, want: mtWantFileData},
			mtCase{name: n + "/user/data-url-typed", spec: mtGoogleSpec, kind: kind, src: mtDataTyped, req: mtUserRequest, accepted: mtGoogleInline, extract: mtGoogleFileData, want: mtWantFileData},
			mtCase{name: n + "/user/https-url", spec: mtGoogleSpec, kind: kind, src: mtRemote, url: mtHTTPSURL, req: mtUserRequest, extract: mtGoogleFileData, want: mtWantFileData},
			mtCase{name: n + "/user/files-api-uri", spec: mtGoogleSpec, kind: kind, src: mtRemote, url: mtFilesURI, req: mtUserRequest, extract: mtGoogleFileData, want: mtWantFileData},
			mtCase{name: n + "/user/gs-uri", spec: mtGoogleSpec, kind: kind, src: mtRemote, url: mtGCSURI, req: mtUserRequest, extract: mtGoogleFileData, want: mtWantFileData},
			mtCase{name: n + "/tool-result/inline", spec: mtGoogleSpec, kind: kind, src: mtInline, req: mtToolResultRequest, accepted: mtGoogleToolSet, extract: mtGoogleToolResult, want: mtWantMediaType},
		)
	}
	return cs
}

// mtCheckRefusal reports why err/hits is not a pre-wire refusal naming named.
func mtCheckRefusal(err error, hits int, named string) string {
	var apiErr *llmkit.APIError
	switch {
	case hits != 0:
		return "requests reached the server, want 0"
	case !errors.Is(err, llmkit.ErrInvalidRequest) || !errors.As(err, &apiErr):
		return "err is not an APIError wrapping ErrInvalidRequest"
	case apiErr.StatusCode != 0:
		return "StatusCode != 0 (not refused before the wire)"
	case !strings.Contains(err.Error(), named):
		return "error does not name the media type " + named
	}
	return ""
}

func TestMediaTypeAcceptedSetsPerWire(t *testing.T) {
	probe := mtProbeSet()
	for _, c := range mtCases() {
		t.Run(c.name, func(t *testing.T) {
			client, rec := mtClient(t, c.spec)
			accepted := make(map[string]bool, len(c.accepted))
			for _, m := range c.accepted {
				accepted[m] = true
			}
			for _, m := range probe {
				rec.reset()
				b := c.block(m)
				_, err := client.Complete(context.Background(), c.req(b))
				hits, body := rec.snapshot()
				if named, refused := c.refused(accepted, m); refused {
					if why := mtCheckRefusal(err, hits, named); why != "" {
						t.Errorf("%q: %s (err = %v, hits = %d)", m, why, err, hits)
					}
					continue
				}
				if err != nil || hits != 1 {
					t.Errorf("%q: err = %v, hits = %d, want a request on the wire", m, err, hits)
					continue
				}
				if got, want := c.extract(t, body), c.want(b); got != want {
					t.Errorf("%q: wire wrote %q, want %q", m, got, want)
				}
			}
		})
	}
}

func TestMediaTypeDataURLRows(t *testing.T) {
	img := func(url string) llmkit.Block { return llmkit.ImageURL(url) }
	doc := func(url, m string) llmkit.Block {
		return llmkit.Block{Kind: llmkit.BlockDocument, URL: url, MediaType: m}
	}
	rows := []struct {
		name  string
		spec  Spec
		req   llmkit.Request
		named string // refused naming this type; "" means one request reaches the wire
	}{
		{"anthropic data:image/bmp refused", mtAnthSpec, mtUserRequest(img("data:image/bmp;base64,Qk0=")), "image/bmp"},
		{"anthropic tool-result data:image/bmp refused", mtAnthSpec, mtToolResultRequest(img("data:image/bmp;base64,Qk0=")), "image/bmp"},
		{"anthropic data:image/png passes", mtAnthSpec, mtUserRequest(img("data:image/png;base64,AAAA")), ""},
		{"anthropic https passes", mtAnthSpec, mtUserRequest(img("https://example.test/a.bmp")), ""},
		{"anthropic document data:image/png refused", mtAnthSpec, mtUserRequest(doc("data:image/png;base64,AAAA", "")), "image/png"},
		{"anthropic document data:application/pdf passes", mtAnthSpec, mtUserRequest(doc("data:application/pdf;base64,AAAA", "")), ""},
		{"anthropic data:,x refused (empty type)", mtAnthSpec, mtUserRequest(img("data:,x")), `""`},
		{"anthropic data:;base64 refused (empty type)", mtAnthSpec, mtUserRequest(img("data:;base64,AAAA")), `""`},
		{"openai data:image/bmp refused", mtOpenAISpec, mtUserRequest(img("data:image/bmp;base64,Qk0=")), "image/bmp"},
		{"openai DATA:image/bmp refused", mtOpenAISpec, mtUserRequest(img("DATA:image/bmp;base64,Qk0=")), "image/bmp"},
		{"openai data:image/png passes", mtOpenAISpec, mtUserRequest(img("data:image/png;base64,AAAA")), ""},
		{"openai https passes", mtOpenAISpec, mtUserRequest(img("https://example.test/a.bmp")), ""},
		{"google data:image/bmp refused", mtGoogleSpec, mtUserRequest(img("data:image/bmp;base64,Qk0=")), "image/bmp"},
		{"google data:image/heic passes", mtGoogleSpec, mtUserRequest(img("data:image/heic;base64,AAAA")), ""},
		{"google data:image/png without separator refused", mtGoogleSpec, mtUserRequest(img("data:image/png")), `""`},
		{"google data:application/pdf with MediaType image/bmp refused", mtGoogleSpec, mtUserRequest(doc("data:application/pdf;base64,AAAA", "image/bmp")), "image/bmp"},
		{"google Files API URI audio/mp3 passes", mtGoogleSpec, mtUserRequest(doc(mtFilesURI, "audio/mp3")), ""},
		{"google gs:// audio/wav passes", mtGoogleSpec, mtUserRequest(doc(mtGCSURI, "audio/wav")), ""},
		{"openai-compatible data:image/bmp passes", mtCompatSpec, mtUserRequest(img("data:image/bmp;base64,Qk0=")), ""},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			client, rec := mtClient(t, r.spec)
			_, err := client.Complete(context.Background(), r.req)
			hits, body := rec.snapshot()
			if r.named != "" {
				if why := mtCheckRefusal(err, hits, r.named); why != "" {
					t.Errorf("%s (err = %v, hits = %d)", why, err, hits)
				}
				return
			}
			if err != nil || hits != 1 {
				t.Fatalf("err = %v, hits = %d, want one request", err, hits)
			}
			raw, _ := json.Marshal(body)
			b := r.req.Messages[len(r.req.Messages)-1].Content[1]
			if !strings.Contains(string(raw), b.URL) {
				t.Errorf("URL %q not on the wire: %s", b.URL, raw)
			}
		})
	}
}

// TestMediaTypeGoogleURLWithoutMediaType pins that a URL block with no
// MediaType still reaches the wire with an empty mimeType.
func TestMediaTypeGoogleURLWithoutMediaType(t *testing.T) {
	client, rec := mtClient(t, mtGoogleSpec)
	_, err := client.Complete(context.Background(), mtUserRequest(llmkit.ImageURL("https://example.test/a.png")))
	hits, body := rec.snapshot()
	if err != nil || hits != 1 {
		t.Fatalf("err = %v, hits = %d, want one request", err, hits)
	}
	if got := mtGoogleFileData(t, body); got != "https://example.test/a.png|" {
		t.Errorf("fileData = %q, want the URL forwarded with an empty mimeType", got)
	}
}

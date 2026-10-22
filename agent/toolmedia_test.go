package agent

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

var tmPNG = []byte("\x89PNG\r\n\x1a\nfake")

const (
	tmImageMarker = "[image omitted: this model does not accept images in tool results]"
	tmDocMarker   = "[document omitted: this model does not accept documents in tool results]"
	tmPNGMarker   = "[image: image/png, 12 bytes]"
)

// tmWire is one hermetic provider endpoint: it answers its first request with
// the tool-call turn, every later one with the final text turn, and keeps the
// request bodies.
type tmWire struct {
	typ         provider.Type
	model       string
	tool, final string

	mu     sync.Mutex
	bodies []map[string]any
}

func (w *tmWire) client(t *testing.T) llmkit.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		w.mu.Lock()
		w.bodies = append(w.bodies, body)
		n := len(w.bodies)
		w.mu.Unlock()
		rw.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_, _ = rw.Write([]byte(w.tool))
			return
		}
		_, _ = rw.Write([]byte(w.final))
	}))
	t.Cleanup(srv.Close)
	c, err := provider.New(context.Background(), provider.Spec{Type: w.typ, Model: w.model, BaseURL: srv.URL, Secret: "k"}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	return c
}

func tmOpenAIWire(typ provider.Type, model string) *tmWire {
	turn := func(msg string, finish string) string {
		return `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":` + msg +
			`,"finish_reason":"` + finish + `"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	}
	return &tmWire{typ: typ, model: model,
		tool:  turn(`{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"snap","arguments":"{}"}}]}`, "tool_calls"),
		final: turn(`{"role":"assistant","content":"done"}`, "stop"),
	}
}

func tmAnthropicWire() *tmWire {
	return &tmWire{typ: provider.TypeAnthropic, model: "claude-opus-4-5",
		tool:  `{"id":"m1","type":"message","role":"assistant","model":"x","content":[{"type":"tool_use","id":"toolu_1","name":"snap","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`,
		final: `{"id":"m2","type":"message","role":"assistant","model":"x","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
	}
}

func tmGoogleWire() *tmWire {
	return &tmWire{typ: provider.TypeGoogle, model: "gemini-3-flash-preview",
		tool:  `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc_1","name":"snap","args":{}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
		final: `{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
	}
}

// tmRun drives one snap tool call through w and returns the outcome, the ToolRun
// event, and the ToolEnd hook event.
func tmRun(t *testing.T, w *tmWire, blocks []llmkit.Block) (*Outcome, llmkit.ToolRunEvent, ToolEvent) {
	t.Helper()
	var hook ToolEvent
	r := NewRunner(w.client(t), []Tool{blocksTool{"snap", blocks}}, "sys",
		WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) { hook = ev }}))
	out, err := r.Run(context.Background(), "task", WithRunID("tm"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(w.bodies) != 2 {
		t.Fatalf("wire requests = %d, want 2", len(w.bodies))
	}
	var runs []llmkit.ToolRunEvent
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindToolRun {
			runs = append(runs, *ev.ToolRun)
		}
	}
	if len(runs) != 1 {
		t.Fatalf("ToolRun events = %d, want 1", len(runs))
	}
	return out, runs[0], hook
}

func tmLastWireMessage(t *testing.T, body map[string]any, key string) string {
	t.Helper()
	list, _ := body[key].([]any)
	if len(list) == 0 {
		t.Fatalf("wire body has no %s: %v", key, body)
	}
	j, err := json.Marshal(list[len(list)-1])
	if err != nil {
		t.Fatal(err)
	}
	return string(j)
}

// TestToolMedia_FalseProfilesDegradeToMarker: on a profile without
// ToolResultMedia the run completes, the wire carries the omission marker in
// place of the image, and the event, hook and history agree.
func TestToolMedia_FalseProfilesDegradeToMarker(t *testing.T) {
	for name, w := range map[string]*tmWire{
		"openai": tmOpenAIWire(provider.TypeOpenAI, "gpt-4o"),
		"compat": tmOpenAIWire(provider.TypeOpenAICompatible, "llama3"),
	} {
		t.Run(name, func(t *testing.T) {
			out, run, hook := tmRun(t, w, []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", tmPNG)})
			wantContent := []llmkit.Block{llmkit.Text("cap"), llmkit.Text(tmImageMarker)}
			if run.Degraded != 1 || !reflect.DeepEqual(run.Content, wantContent) || run.Result != "cap"+tmImageMarker {
				t.Errorf("ToolRun = {Degraded:%d Content:%+v Result:%q}, want 1, %+v, %q", run.Degraded, run.Content, run.Result, wantContent, "cap"+tmImageMarker)
			}
			if hook.Degraded != 1 || !reflect.DeepEqual(hook.Content, wantContent) || hook.Result != run.Result {
				t.Errorf("ToolEnd = {Degraded:%d Content:%+v Result:%q}, want the ToolRun values", hook.Degraded, hook.Content, hook.Result)
			}
			var hist llmkit.Message
			for _, m := range out.Messages {
				if m.Role == llmkit.RoleToolResult {
					hist = m
				}
			}
			if !reflect.DeepEqual(hist.Content, wantContent) {
				t.Errorf("history tool result = %+v, want %+v", hist.Content, wantContent)
			}
			wire := tmLastWireMessage(t, w.bodies[1], "messages")
			if !strings.Contains(wire, "cap"+tmImageMarker) || strings.Contains(wire, "image_url") || strings.Contains(wire, base64.StdEncoding.EncodeToString(tmPNG)) {
				t.Errorf("wire tool message = %s, want %q and no image", wire, "cap"+tmImageMarker)
			}
		})
	}
}

// TestToolMedia_GeminiPortableSet: on Gemini 3 (flag true) a gif is outside the
// portable set and degrades to a marker, while a png in the same result passes
// as inlineData.
func TestToolMedia_GeminiPortableSet(t *testing.T) {
	w := tmGoogleWire()
	gif := llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/gif", Data: []byte("GIF89a")}
	_, run, _ := tmRun(t, w, []llmkit.Block{llmkit.Text("cap"), gif, llmkit.Image("image/png", tmPNG)})
	wantContent := []llmkit.Block{llmkit.Text("cap"), llmkit.Text(tmImageMarker), llmkit.Image("image/png", tmPNG)}
	if run.Degraded != 1 || !reflect.DeepEqual(run.Content, wantContent) {
		t.Errorf("ToolRun = {Degraded:%d Content:%+v}, want 1, %+v", run.Degraded, run.Content, wantContent)
	}
	if want := "cap" + tmImageMarker + tmPNGMarker; run.Result != want {
		t.Errorf("Result = %q, want %q", run.Result, want)
	}
	wire := tmLastWireMessage(t, w.bodies[1], "contents")
	if !strings.Contains(wire, tmImageMarker) {
		t.Errorf("wire = %s, want the gif's omission marker", wire)
	}
	if !strings.Contains(wire, `"mimeType":"image/png"`) || !strings.Contains(wire, base64.StdEncoding.EncodeToString(tmPNG)) || strings.Contains(wire, "image/gif") {
		t.Errorf("wire = %s, want the png as inlineData and no gif", wire)
	}
}

// TestToolMedia_AnthropicPassesVerbatim: png reaches the wire untouched; the
// event keeps the image block and its Result carries the size marker.
func TestToolMedia_AnthropicPassesVerbatim(t *testing.T) {
	w := tmAnthropicWire()
	blocks := []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", tmPNG)}
	_, run, hook := tmRun(t, w, blocks)
	if run.Degraded != 0 || !reflect.DeepEqual(run.Content, blocks) {
		t.Errorf("ToolRun = {Degraded:%d Content:%+v}, want 0, %+v", run.Degraded, run.Content, blocks)
	}
	if want := "cap" + tmPNGMarker; run.Result != want || hook.Result != want {
		t.Errorf("Result = %q, hook = %q, want %q", run.Result, hook.Result, want)
	}
	wire := tmLastWireMessage(t, w.bodies[1], "messages")
	if !strings.Contains(wire, `"media_type":"image/png"`) || !strings.Contains(wire, base64.StdEncoding.EncodeToString(tmPNG)) || strings.Contains(wire, "omitted") {
		t.Errorf("wire = %s, want the png verbatim", wire)
	}
}

// TestToolMedia_PortableNeedsInlineMatchingKind: on media-capable wires only
// inline png/jpeg/webp images and pdf documents pass; a URL source, or a kind
// that does not match its MIME type, degrades to its kind's marker.
func TestToolMedia_PortableNeedsInlineMatchingKind(t *testing.T) {
	urlPNG := llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", URL: "https://x/y.png"}
	pngDoc := llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "image/png", Data: tmPNG}
	pdfImage := llmkit.Block{Kind: llmkit.BlockImage, MediaType: "application/pdf", Data: []byte("%PDF-1")}
	png := llmkit.Image("image/png", tmPNG)
	cases := []struct {
		name   string
		wire   func() *tmWire
		key    string
		block  llmkit.Block
		marker string // "" when the block passes verbatim
	}{
		{"url png image on gemini-3", tmGoogleWire, "contents", urlPNG, tmImageMarker},
		{"png document on anthropic", tmAnthropicWire, "messages", pngDoc, tmDocMarker},
		{"pdf image on anthropic", tmAnthropicWire, "messages", pdfImage, tmImageMarker},
		{"pdf image on gemini-3", tmGoogleWire, "contents", pdfImage, tmImageMarker},
		{"inline png on anthropic", tmAnthropicWire, "messages", png, ""},
		{"inline png on gemini-3", tmGoogleWire, "contents", png, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			w := c.wire()
			blocks := []llmkit.Block{llmkit.Text("cap"), c.block}
			_, run, _ := tmRun(t, w, blocks)
			wire := tmLastWireMessage(t, w.bodies[1], c.key)
			data := base64.StdEncoding.EncodeToString(c.block.Data)
			if c.marker == "" {
				if run.Degraded != 0 || !reflect.DeepEqual(run.Content, blocks) {
					t.Errorf("ToolRun = {Degraded:%d Content:%+v}, want 0, %+v", run.Degraded, run.Content, blocks)
				}
				if !strings.Contains(wire, data) || strings.Contains(wire, "omitted") {
					t.Errorf("wire = %s, want the block verbatim", wire)
				}
				return
			}
			want := []llmkit.Block{llmkit.Text("cap"), llmkit.Text(c.marker)}
			if run.Degraded != 1 || !reflect.DeepEqual(run.Content, want) {
				t.Errorf("ToolRun = {Degraded:%d Content:%+v}, want 1, %+v", run.Degraded, run.Content, want)
			}
			if !strings.Contains(wire, c.marker) || (data != "" && strings.Contains(wire, data)) || (c.block.URL != "" && strings.Contains(wire, c.block.URL)) {
				t.Errorf("wire = %s, want %q in place of the block", wire, c.marker)
			}
		})
	}
}

// tmFake runs one snap call through a fake client reporting ToolResultMedia=media.
func tmFake(t *testing.T, media bool, blocks []llmkit.Block, opts ...Option) (*Outcome, []ToolEvent) {
	t.Helper()
	fc := newFakeClient(toolResp("c1", "snap", `{}`, 1, 1), textResp("done", 1, 1))
	fc.caps.ToolResultMedia = media
	var mu sync.Mutex
	var hooks []ToolEvent
	opts = append(opts, WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) {
		mu.Lock()
		defer mu.Unlock()
		hooks = append(hooks, ev)
	}}))
	out, err := NewRunner(fc, []Tool{blocksTool{"snap", blocks}}, "sys", opts...).Run(context.Background(), "task", WithRunID("tmf"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return out, hooks
}

func toolMessage(t *testing.T, msgs []llmkit.Message) llmkit.Message {
	t.Helper()
	for _, m := range msgs {
		if m.Role == llmkit.RoleToolResult {
			return m
		}
	}
	t.Fatal("no tool-result message in history")
	return llmkit.Message{}
}

// TestToolMedia_ReplayServesContent: sent-media, degraded and text-only runs
// replay from memory and from JSONL to the recorded history and ToolRun events.
func TestToolMedia_ReplayServesContent(t *testing.T) {
	cases := []struct {
		name   string
		media  bool
		blocks []llmkit.Block
	}{
		{"sent", true, []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", tmPNG), llmkit.Document("application/pdf", "t", []byte("%PDF"))}},
		{"degraded", false, []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", tmPNG)}},
		{"text-only", true, []llmkit.Block{llmkit.Text("plain")}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec, _ := tmFake(t, c.media, c.blocks)
			var jsonl bytes.Buffer
			if err := rec.Transcript.SaveJSONL(&jsonl); err != nil {
				t.Fatal(err)
			}
			loaded, err := LoadJSONL(&jsonl)
			if err != nil {
				t.Fatal(err)
			}
			for src, tr := range map[string]*Transcript{"memory": rec.Transcript, "jsonl": loaded} {
				t.Run(src, func(t *testing.T) {
					rc, err := NewReplayClient(tr, "tmf", llmkit.Capabilities{ToolResultMedia: c.media})
					if err != nil {
						t.Fatal(err)
					}
					out, err := NewRunner(rc, rc.Tools(), "sys").Run(context.Background(), "task", WithRunID("tmf"))
					if err != nil || rc.Err() != nil {
						t.Fatalf("Run = %v, rc.Err = %v, want both nil", err, rc.Err())
					}
					if !reflect.DeepEqual(out.Messages, rec.Messages) {
						t.Errorf("replayed history = %+v\nrecorded history = %+v", out.Messages, rec.Messages)
					}
					rdEqualRuns(t, rdToolRuns(out.Transcript.Record), rdToolRuns(rec.Transcript.Record))
				})
			}
		})
	}
}

// TestToolMedia_BaseRecordLoadsByteForByte: a record written before Content and
// Degraded existed loads and re-saves to the same bytes.
func TestToolMedia_BaseRecordLoadsByteForByte(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "base_record_denial_parallel.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	rec, err := LoadJSONL(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	if err := rec.SaveJSONL(&got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Bytes(), raw) {
		t.Errorf("re-saved record differs from the file\n--- got\n%s\n--- want\n%s", got.Bytes(), raw)
	}
}

// TestToolMedia_TextOnlyEventsCarryNoContent: text-only results, single or
// multi-block, error results and denials serialize with no content or
// degraded key and a nil Content/Degraded 0 on the hook.
func TestToolMedia_TextOnlyEventsCarryNoContent(t *testing.T) {
	for name, blocks := range map[string][]llmkit.Block{
		"single": textBlocks("hello"),
		"multi":  {llmkit.Text("a"), llmkit.Text("b")},
		"empty":  nil,
	} {
		t.Run(name, func(t *testing.T) {
			out, hooks := tmFake(t, true, blocks)
			if len(hooks) != 1 || hooks[0].Content != nil || hooks[0].Degraded != 0 {
				t.Fatalf("hooks = %+v, want one with nil Content and Degraded 0", hooks)
			}
			for _, ev := range rdToolRuns(out.Transcript.Record) {
				j, _ := json.Marshal(ev.ToolRun)
				if strings.Contains(string(j), `"content"`) || strings.Contains(string(j), `"degraded"`) {
					t.Errorf("text-only ToolRun JSON = %s, want no content or degraded key", j)
				}
			}
		})
	}
}

// TestToolMedia_HookEventRule: ToolEvent carries Content and Degraded per the
// rule: sent media keep their blocks and a size marker, degraded media become
// omission markers, a URL block (no MediaType) degrades, and the portable set
// is png/jpeg/webp/pdf.
func TestToolMedia_HookEventRule(t *testing.T) {
	pdf := llmkit.Document("application/pdf", "t", []byte("%PDF-1"))
	txtDoc := llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "text/plain", Data: []byte("hi")}
	cases := []struct {
		name         string
		media        bool
		blocks       []llmkit.Block
		wantContent  []llmkit.Block
		wantDegraded int
		wantResult   string
	}{
		{"png sent", true, []llmkit.Block{llmkit.Text("a"), llmkit.Image("image/png", tmPNG)},
			[]llmkit.Block{llmkit.Text("a"), llmkit.Image("image/png", tmPNG)}, 0, "a" + tmPNGMarker},
		{"pdf sent", true, []llmkit.Block{pdf},
			[]llmkit.Block{pdf}, 0, "[document: application/pdf, 6 bytes]"},
		{"gif degraded", true, []llmkit.Block{llmkit.Image("image/gif", []byte("GIF"))},
			[]llmkit.Block{llmkit.Text(tmImageMarker)}, 1, tmImageMarker},
		{"text/plain document degraded", true, []llmkit.Block{txtDoc},
			[]llmkit.Block{llmkit.Text(tmDocMarker)}, 1, tmDocMarker},
		{"url image degraded", true, []llmkit.Block{llmkit.ImageURL("https://x/y.png")},
			[]llmkit.Block{llmkit.Text(tmImageMarker)}, 1, tmImageMarker},
		{"false profile degrades pdf and png", false, []llmkit.Block{pdf, llmkit.Image("image/png", tmPNG)},
			[]llmkit.Block{llmkit.Text(tmDocMarker), llmkit.Text(tmImageMarker)}, 2, tmDocMarker + tmImageMarker},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, hooks := tmFake(t, c.media, c.blocks)
			if len(hooks) != 1 {
				t.Fatalf("ToolEnd fired %d times, want 1", len(hooks))
			}
			h := hooks[0]
			if !reflect.DeepEqual(h.Content, c.wantContent) || h.Degraded != c.wantDegraded || h.Result != c.wantResult {
				t.Errorf("ToolEvent = {Content:%+v Degraded:%d Result:%q}, want {%+v %d %q}", h.Content, h.Degraded, h.Result, c.wantContent, c.wantDegraded, c.wantResult)
			}
			runs := rdToolRuns(out.Transcript.Record)
			if len(runs) != 1 {
				t.Fatalf("ToolRun events = %d, want 1", len(runs))
			}
			if tr := runs[0].ToolRun; !reflect.DeepEqual(tr.Content, c.wantContent) || tr.Degraded != c.wantDegraded || tr.Result != c.wantResult {
				t.Errorf("ToolRun = {Content:%+v Degraded:%d Result:%q}, want the ToolEvent values", tr.Content, tr.Degraded, tr.Result)
			}
			if hist := toolMessage(t, out.Messages); !reflect.DeepEqual(hist.Content, c.wantContent) {
				t.Errorf("history = %+v, want %+v", hist.Content, c.wantContent)
			}
		})
	}
}

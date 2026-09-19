package llmkit

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"testing"
)

// TestTextMessageAndText covers the one-line text-message helper and the
// Text accessor's block-kind filtering.
func TestTextMessageAndText(t *testing.T) {
	m := TextMessage(RoleUser, "hello")
	if m.Role != RoleUser || len(m.Content) != 1 || m.Content[0].Kind != BlockText || m.Content[0].Text != "hello" {
		t.Fatalf("TextMessage = %+v", m)
	}
	if got := m.Text(); got != "hello" {
		t.Errorf("Text() = %q, want hello", got)
	}

	// Non-text blocks are skipped; multiple text blocks concatenate in order.
	m.Content = append(m.Content,
		Block{Kind: BlockImage, MediaType: "image/png", Data: []byte{1}},
		Block{Kind: BlockText, Text: " world"},
		Block{Kind: BlockThinking, Provider: "anthropic", Raw: json.RawMessage(`{}`)},
	)
	if got := m.Text(); got != "hello world" {
		t.Errorf("Text() = %q, want %q", got, "hello world")
	}
}

// TestBlockDataJSONRoundTrip documents and verifies the transcript contract:
// encoding/json base64s Block.Data in JSONL and restores the exact bytes on
// decode, with no custom marshalers.
func TestBlockDataJSONRoundTrip(t *testing.T) {
	media := []byte{0x00, 0x01, 0x02, 0xFF, 0xFE}
	in := Message{
		Role: RoleUser,
		Content: []Block{
			{Kind: BlockText, Text: "see attachment"},
			{Kind: BlockImage, MediaType: "image/png", Data: media},
		},
	}
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(in); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	// The wire form must carry base64 (JSONL transcripts rely on this).
	if !bytes.Contains(buf.Bytes(), []byte(base64.StdEncoding.EncodeToString(media))) {
		t.Errorf("JSONL does not carry base64 Data: %s", buf.String())
	}
	var out Message
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(out.Content) != 2 {
		t.Fatalf("blocks = %d, want 2", len(out.Content))
	}
	if out.Content[0].Text != "see attachment" {
		t.Errorf("text block = %+v", out.Content[0])
	}
	if !bytes.Equal(out.Content[1].Data, media) || out.Content[1].MediaType != "image/png" {
		t.Errorf("image block round-trip = %+v (data %v)", out.Content[1], out.Content[1].Data)
	}
}

// TestBlockConstructors pins the per-kind invariants the constructors
// encode at build time: kinds, field placement, the tool-result role pair,
// and the panic on an argument that could never satisfy the wire contract.
func TestBlockConstructors(t *testing.T) {
	if b := Text("hi"); b.Kind != BlockText || b.Text != "hi" {
		t.Errorf("Text = %+v", b)
	}
	if b := Image("image/png", []byte{1}); b.Kind != BlockImage || b.MediaType != "image/png" || !bytes.Equal(b.Data, []byte{1}) || b.URL != "" {
		t.Errorf("Image = %+v", b)
	}
	if b := ImageURL("https://x/y.png"); b.Kind != BlockImage || b.URL == "" || b.Data != nil {
		t.Errorf("ImageURL = %+v", b)
	}
	if b := Document("application/pdf", "spec", []byte{2}); b.Kind != BlockDocument || b.MediaType != "application/pdf" || b.Title != "spec" || !bytes.Equal(b.Data, []byte{2}) {
		t.Errorf("Document = %+v", b)
	}
	if b := DocumentURL("https://x/z.pdf", ""); b.Kind != BlockDocument || b.URL == "" || b.Title != "" {
		t.Errorf("DocumentURL with empty title must be legal, got %+v", b)
	}
	if m := UserMessage(Text("a"), Image("image/png", []byte{1})); m.Role != RoleUser || len(m.Content) != 2 {
		t.Errorf("UserMessage = %+v", m)
	}
	if m := SystemMessage("be terse"); m.Role != RoleSystem || len(m.Content) != 1 || m.Content[0].Kind != BlockText || m.Content[0].Text != "be terse" {
		t.Errorf("SystemMessage = %+v", m)
	}
	if m := ToolResult("c1", "42"); m.Role != RoleToolResult || m.ToolCallID != "c1" || m.IsError || m.Text() != "42" {
		t.Errorf("ToolResult = %+v", m)
	}
	if m := ToolError("c1", "disk on fire"); m.Role != RoleToolResult || m.ToolCallID != "c1" || !m.IsError || m.Text() != "disk on fire" {
		t.Errorf("ToolError = %+v", m)
	}
}

// TestBlockConstructorPanics pins the error classification: a constructor
// argument that could never satisfy the wire contract is a programming
// error and panics (hand-built literals stay legal; the adapters reject
// them pre-wire).
func TestBlockConstructorPanics(t *testing.T) {
	cases := []struct {
		name string
		f    func()
	}{
		{"Image empty mediaType", func() { Image("", []byte{1}) }},
		{"Image empty data", func() { Image("image/png", nil) }},
		{"ImageURL empty url", func() { ImageURL("") }},
		{"Document empty mediaType", func() { Document("", "t", []byte{1}) }},
		{"Document empty data", func() { Document("application/pdf", "t", nil) }},
		{"DocumentURL empty url", func() { DocumentURL("", "t") }},
		{"UserMessage no blocks", func() { UserMessage() }},
		{"ToolResult empty callID", func() { ToolResult("", "x") }},
		{"ToolError empty callID", func() { ToolError("", "x") }},
	}
	for _, tc := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", tc.name)
				}
			}()
			tc.f()
		}()
	}
}

// TestBlockJSONShape pins the transcript wire shape: a text block marshals
// to exactly {"kind":"text","text":"…"} — zero fields omitted, and a nil
// Raw never emits "raw":null.
func TestBlockJSONShape(t *testing.T) {
	b, err := json.Marshal(Text("hello"))
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got, want := string(b), `{"kind":"text","text":"hello"}`; got != want {
		t.Errorf("text block JSON = %s, want %s", got, want)
	}
	think := Block{Kind: BlockThinking, Provider: "anthropic", Raw: json.RawMessage(`{"sig":"x"}`)}
	b, err = json.Marshal(think)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got, want := string(b), `{"kind":"thinking","provider":"anthropic","raw":{"sig":"x"}}`; got != want {
		t.Errorf("thinking block JSON = %s, want %s", got, want)
	}
	// Nil Raw is omitted entirely.
	b, err = json.Marshal(Text(""))
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got, want := string(b), `{"kind":"text"}`; got != want {
		t.Errorf("empty text block JSON = %s, want %s", got, want)
	}
	if bytes.Contains(b, []byte(`"raw"`)) {
		t.Errorf("nil Raw emitted %s", b)
	}
}

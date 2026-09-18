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

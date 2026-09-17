package adapter

import (
	"errors"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestMatchesModelFamily pins the segment-boundary rule shared by all three
// capability tables: a key matches its own ID and later "-"-separated
// snapshots, never a mid-token extension or a dotted sibling generation.
func TestMatchesModelFamily(t *testing.T) {
	cases := []struct {
		model, key string
		want       bool
	}{
		{"gpt-4", "gpt-4", true},
		{"gpt-4-0613", "gpt-4", true},
		{"gpt-4-32k-0613", "gpt-4", true},
		{"gpt-4.5-preview", "gpt-4", false},  // dotted sibling generation
		{"gpt-4.5-preview", "gpt-4.5", true}, // own generation
		{"gpt-4omni", "gpt-4o", false},       // mid-token extension
		{"gpt-4o", "gpt-4", false},           // "4o" is a different segment
		{"claude-3-5-sonnetish", "claude-3-5-sonnet", false},
		{"claude-3-5-sonnet-20241022", "claude-3-5-sonnet", true},
		{"gemini-2.5-flashy", "gemini-2.5-flash", false},
		{"gemini-2.5-flash-image", "gemini-2.5-flash", true}, // family match; a longer key must win
		{"o1-mini", "o1", true},
		{"", "gpt-4", false},
		{"gpt", "gpt-4", false},
	}
	for _, tc := range cases {
		if got := MatchesModelFamily(tc.model, tc.key); got != tc.want {
			t.Errorf("MatchesModelFamily(%q, %q) = %v, want %v", tc.model, tc.key, got, tc.want)
		}
	}
}

// TestBestMatchingFamily pins longest-key-wins over the generic table walk.
func TestBestMatchingFamily(t *testing.T) {
	table := []struct{ key string }{
		{"gemini-2.5-flash"},
		{"gemini-2.5-flash-lite"},
		{"gemini-2.5-flash-image"},
	}
	key := func(e struct{ key string }) string { return e.key }
	if got := BestMatchingFamily("gemini-2.5-flash-image-preview", table, key); got != 2 {
		t.Errorf("longest match index = %d, want 2 (flash-image entry)", got)
	}
	if got := BestMatchingFamily("gemini-2.5-flash-lite-001", table, key); got != 1 {
		t.Errorf("longest match index = %d, want 1 (flash-lite entry)", got)
	}
	if got := BestMatchingFamily("gemini-2.5-flashy", table, key); got != -1 {
		t.Errorf("mid-token extension index = %d, want -1", got)
	}
	if got := BestMatchingFamily("gemini-9", table, key); got != -1 {
		t.Errorf("unmatched index = %d, want -1", got)
	}
}

// TestGateToolChoice pins the rejection contract: explicit modes are refused
// when the profile reports ToolChoice=false, auto/zero pass, unknown modes
// fall through to the adapter's mapper, and no error surfaces when supported.
func TestGateToolChoice(t *testing.T) {
	for _, mode := range []llmkit.ToolChoiceMode{llmkit.ToolChoiceNone, llmkit.ToolChoiceRequired, llmkit.ToolChoiceTool} {
		err := GateToolChoice("test", llmkit.ToolChoice{Mode: mode, Name: "f"}, false)
		if !errors.Is(err, llmkit.ErrInvalidRequest) {
			t.Errorf("GateToolChoice(mode=%s, unsupported) = %v, want ErrInvalidRequest", mode, err)
		}
	}
	for _, mode := range []llmkit.ToolChoiceMode{"", llmkit.ToolChoiceAuto} {
		if err := GateToolChoice("test", llmkit.ToolChoice{Mode: mode}, false); err != nil {
			t.Errorf("GateToolChoice(mode=%q, unsupported) = %v, want nil (provider default)", mode, err)
		}
	}
	if err := GateToolChoice("test", llmkit.ToolChoice{Mode: "bogus-mode"}, false); err != nil {
		t.Errorf("GateToolChoice(unknown mode, unsupported) = %v, want nil (mapper owns that error)", err)
	}
	if err := GateToolChoice("test", llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}, true); err != nil {
		t.Errorf("GateToolChoice(required, supported) = %v, want nil", err)
	}
}

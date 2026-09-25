package adapter

import (
	"testing"
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
		if got := matchesModelFamily(tc.model, tc.key); got != tc.want {
			t.Errorf("matchesModelFamily(%q, %q) = %v, want %v", tc.model, tc.key, got, tc.want)
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

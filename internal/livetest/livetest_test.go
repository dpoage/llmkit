package livetest

import (
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

func lane(t *testing.T, name string) Lane {
	t.Helper()
	l, ok := LaneByName(name)
	if !ok {
		t.Fatalf("LaneByName(%q): not found", name)
	}
	return l
}

func TestParseCaps(t *testing.T) {
	caps, err := ParseCaps("structured_output, parallel_tool_calls")
	if err != nil {
		t.Fatalf("ParseCaps: %v", err)
	}
	if !caps["StructuredOutput"] || !caps["ParallelToolCalls"] {
		t.Fatalf("parsed caps = %v, want StructuredOutput and ParallelToolCalls true", caps)
	}
	if len(caps) != 2 {
		t.Fatalf("parsed caps = %v, want exactly 2 entries", caps)
	}

	if _, err := ParseCaps("structured_output,no_such_cap"); err == nil {
		t.Fatal("ParseCaps(unknown name): want error, got nil")
	} else if !strings.Contains(err.Error(), "prompt_caching") {
		t.Errorf("unknown-cap error %q does not list the valid names", err)
	}

	if _, err := ParseCaps("context_window"); err == nil {
		t.Fatal("ParseCaps(context_window): want error (not a boolean capability), got nil")
	}

	caps, err = ParseCaps("")
	if err != nil || len(caps) != 0 {
		t.Fatalf("ParseCaps(\"\") = %v, %v; want empty set, nil", caps, err)
	}
}

func TestToSnake(t *testing.T) {
	cases := map[string]string{
		"ParallelToolCalls": "parallel_tool_calls",
		"TopP":              "top_p",
		"Seed":              "seed",
		"ContextWindow":     "context_window",
	}
	for in, want := range cases {
		if got := toSnake(in); got != want {
			t.Errorf("toSnake(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestLaneResolve(t *testing.T) {
	t.Setenv("LLMKIT_LIVE_COMPAT_API_KEY", "k")
	t.Setenv("LLMKIT_LIVE_COMPAT_BASE_URL", "https://example.invalid/v1")
	t.Setenv("LLMKIT_LIVE_COMPAT_MODEL", "m")
	s, missing := lane(t, "compat").resolve()
	if len(missing) != 0 {
		t.Fatalf("compat missing = %v, want none", missing)
	}
	if s.BaseURL == "" || s.Model != "m" || s.Key != "k" {
		t.Fatalf("compat session = %+v", s)
	}

	// compat: all three variables required.
	t.Setenv("LLMKIT_LIVE_COMPAT_MODEL", "")
	_, missing = lane(t, "compat").resolve()
	if len(missing) != 1 || missing[0] != "LLMKIT_LIVE_COMPAT_MODEL" {
		t.Fatalf("compat missing = %v, want [LLMKIT_LIVE_COMPAT_MODEL]", missing)
	}
	t.Setenv("LLMKIT_LIVE_COMPAT_BASE_URL", "")
	_, missing = lane(t, "compat").resolve()
	if len(missing) != 2 {
		t.Fatalf("compat missing = %v, want model + base URL", missing)
	}

	// key-only lanes fall back to their default model.
	t.Setenv("LLMKIT_LIVE_ANTHROPIC_API_KEY", "k")
	if s, missing = lane(t, "anthropic").resolve(); len(missing) != 0 || s.Model != "claude-haiku-4-5" {
		t.Fatalf("anthropic = %+v missing %v, want default model, none missing", s, missing)
	}
	t.Setenv("LLMKIT_LIVE_OPENAI_API_KEY", "k")
	t.Setenv("LLMKIT_LIVE_OPENAI_MODEL", "gpt-5-mini")
	if s, _ = lane(t, "openai").resolve(); s.Model != "gpt-5-mini" {
		t.Fatalf("openai model = %q, want the operator-set value", s.Model)
	}
	t.Setenv("LLMKIT_LIVE_GOOGLE_API_KEY", "k")
	if s, missing = lane(t, "google").resolve(); len(missing) != 0 || s.Model != "gemini-2.5-flash-lite" {
		t.Fatalf("google = %+v missing %v, want default model, none missing", s, missing)
	}
}

func TestRedact(t *testing.T) {
	got := Redact("secret-key-123", "Authorization: Bearer secret-key-123 and sk-abcdefgh1234")
	if strings.Contains(got, "secret-key-123") || strings.Contains(got, "sk-abcdefgh") {
		t.Fatalf("Redact left a secret: %q", got)
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Fatalf("Redact output missing placeholder: %q", got)
	}
	if got := Redact("", "bearer ABCDEFGH"); strings.Contains(got, "ABCDEFGH") {
		t.Fatalf("Redact left the bearer token: %q", got)
	}
}

func TestTally(t *testing.T) {
	tl := new(Tally)
	tl.Record(llmkit.UsageEvent{Usage: llmkit.Usage{InputTokens: 100, OutputTokens: 20, CacheReadInputTokens: 8}})
	tl.Record(llmkit.UsageEvent{Usage: llmkit.Usage{InputTokens: 30, OutputTokens: 5}})
	if tl.total != 155 || tl.input != 130 || tl.output != 25 || tl.cacheRead != 8 || tl.calls != 2 {
		t.Fatalf("tally = total=%d input=%d output=%d cacheRead=%d calls=%d", tl.total, tl.input, tl.output, tl.cacheRead, tl.calls)
	}
}

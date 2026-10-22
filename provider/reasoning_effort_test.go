package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

type effortProfile struct {
	name     string
	typ      Type
	model    string
	thinking bool // table Thinking
	effort   bool // table ReasoningEffort
}

// effortProfiles covers every profile class: (Thinking, ReasoningEffort) =
// (true,false) anthropic and gemini-2.x, (false,true) openai reasoning
// models, (true,true) gemini-3, (false,false) every other table entry and the
// openai-compatible profile.
var effortProfiles = []effortProfile{
	{"anthropic/opus", TypeAnthropic, "claude-opus-4-5", true, false},
	{"anthropic/haiku-3", TypeAnthropic, "claude-3-haiku", false, false},
	{"anthropic/unknown", TypeAnthropic, "claude-unknown-x", true, false},
	{"openai/gpt-5", TypeOpenAI, "gpt-5", false, true},
	{"openai/gpt-5-mini", TypeOpenAI, "gpt-5-mini-2025-08-07", false, true},
	{"openai/o1", TypeOpenAI, "o1", false, true},
	{"openai/o3-mini", TypeOpenAI, "o3-mini", false, true},
	{"openai/o4-mini", TypeOpenAI, "o4-mini", false, true},
	{"openai/o1-mini", TypeOpenAI, "o1-mini", false, false},
	{"openai/o1-preview", TypeOpenAI, "o1-preview", false, false},
	{"openai/gpt-4.1", TypeOpenAI, "gpt-4.1", false, false},
	{"openai/gpt-4.5", TypeOpenAI, "gpt-4.5-preview", false, false},
	{"openai/gpt-4o", TypeOpenAI, "gpt-4o", false, false},
	{"openai/gpt-4-turbo", TypeOpenAI, "gpt-4-turbo", false, false},
	{"openai/gpt-4", TypeOpenAI, "gpt-4-0613", false, false},
	{"openai/gpt-3.5", TypeOpenAI, "gpt-3.5-turbo", false, false},
	{"openai/unknown", TypeOpenAI, "unknown-model-xyz", false, false},
	{"compat/llama3", TypeOpenAICompatible, "llama3", false, false},
	{"compat/gpt-5-name", TypeOpenAICompatible, "gpt-5", false, false},
	{"google/gemini-3-flash", TypeGoogle, "gemini-3-flash-preview", true, true},
	{"google/gemini-3-pro", TypeGoogle, "gemini-3-pro-preview", true, true},
	{"google/gemini-3.1-pro", TypeGoogle, "gemini-3.1-pro-preview", true, true},
	{"google/gemini-2.5-pro", TypeGoogle, "gemini-2.5-pro", true, false},
	{"google/gemini-2.5-flash", TypeGoogle, "gemini-2.5-flash", true, false},
	{"google/gemini-2.0-flash", TypeGoogle, "gemini-2.0-flash", false, false},
	{"google/gemini-1.5-pro", TypeGoogle, "gemini-1.5-pro", false, false},
	{"google/gemini-1.5-flash", TypeGoogle, "gemini-1.5-flash", false, false},
	{"google/gemini-30-lookalike", TypeGoogle, "gemini-30-x", true, false},
	{"google/unknown", TypeGoogle, "gemini-unknown-x", true, false},
}

// effortWire runs one Complete on p's table profile (override applied when
// non-nil) and returns the client, the captured wire body, the call error,
// and the wire hit count.
func effortWire(t *testing.T, p effortProfile, override func(llmkit.Capabilities) llmkit.Capabilities, th *llmkit.ThinkingConfig) (llmkit.Client, map[string]any, error, int) {
	t.Helper()
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
		t.Fatalf("New(%s): %v", p.name, err)
	}
	req := simpleRequest()
	req.Thinking = th
	_, err = client.Complete(context.Background(), req)
	return client, body, err, hits
}

// wireThinking returns the vendor's thinking object and the effort key
// presence for a captured body.
func wireThinking(typ Type, body map[string]any) (thinking map[string]any, effort any, hasEffort bool) {
	switch typ {
	case TypeGoogle:
		cfg, _ := body["generationConfig"].(map[string]any)
		thinking, _ = cfg["thinkingConfig"].(map[string]any)
		effort, hasEffort = thinking["thinkingLevel"]
	case TypeAnthropic:
		thinking, _ = body["thinking"].(map[string]any)
	default:
		effort, hasEffort = body["reasoning_effort"]
	}
	return thinking, effort, hasEffort
}

func wantWireEffort(typ Type, e llmkit.ReasoningEffort) string {
	if typ == TypeGoogle {
		return strings.ToUpper(string(e))
	}
	return string(e)
}

func TestReasoningEffort_WireMapping(t *testing.T) {
	isolateEnv(t, baseURLEnvSources...)
	for _, p := range effortProfiles {
		for _, e := range []llmkit.ReasoningEffort{llmkit.EffortLow, llmkit.EffortMedium, llmkit.EffortHigh} {
			t.Run(p.name+"/"+string(e), func(t *testing.T) {
				client, body, err, hits := effortWire(t, p, nil, &llmkit.ThinkingConfig{Effort: e})
				if err != nil {
					t.Fatalf("Complete: %v", err)
				}
				caps := client.Capabilities()
				if caps.ReasoningEffort != p.effort || caps.Thinking != p.thinking {
					t.Fatalf("caps Thinking=%v ReasoningEffort=%v, want %v/%v", caps.Thinking, caps.ReasoningEffort, p.thinking, p.effort)
				}
				if hits != 1 {
					t.Fatalf("wire hits = %d, want 1", hits)
				}
				_, got, has := wireThinking(p.typ, body)
				if !p.effort {
					if has {
						t.Errorf("effort key %v on the wire for a ReasoningEffort=false profile: %v", got, body)
					}
					return
				}
				if want := wantWireEffort(p.typ, e); got != want {
					t.Errorf("effort wire value = %v (present %v), want %q: %v", got, has, want, body)
				}
			})
		}
	}
}

// TestReasoningEffort_OverrideEnablesCompat pins that the openai-compatible
// ceiling allows ReasoningEffort: an override turning it on reaches the wire.
func TestReasoningEffort_OverrideEnablesCompat(t *testing.T) {
	isolateEnv(t, baseURLEnvSources...)
	p := effortProfile{"compat", TypeOpenAICompatible, "llama3", false, false}
	on := func(c llmkit.Capabilities) llmkit.Capabilities { c.ReasoningEffort = true; return c }
	_, body, err, _ := effortWire(t, p, on, &llmkit.ThinkingConfig{Effort: llmkit.EffortMedium})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if _, got, _ := wireThinking(p.typ, body); got != "medium" {
		t.Errorf("reasoning_effort = %v, want medium: %v", got, body)
	}
}

// TestReasoningEffort_CeilingRefusesAnthropic: an override reporting
// ReasoningEffort true above the anthropic ceiling is refused at
// construction, and the text does not claim the vendor lacks a wire field.
func TestReasoningEffort_CeilingRefusesAnthropic(t *testing.T) {
	isolateEnv(t, baseURLEnvSources...)
	hits := 0
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) { hits++ })
	on := func(c llmkit.Capabilities) llmkit.Capabilities { c.ReasoningEffort = true; return c }
	client, err := New(context.Background(), Spec{Type: TypeAnthropic, Model: "claude-opus-4-5", BaseURL: base, Secret: "k", Capabilities: on}, Options{})
	if client != nil {
		t.Errorf("New returned a client alongside an error")
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("New err = %v, want ErrInvalidRequest", err)
	}
	for _, want := range []string{"ReasoningEffort", "anthropic", "does not map"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("err = %q, want it to contain %q", err.Error(), want)
		}
	}
	if strings.Contains(err.Error(), "no wire field") {
		t.Errorf("err = %q claims the vendor has no wire field", err.Error())
	}
	if hits != 0 {
		t.Errorf("wire hits = %d, want 0", hits)
	}
}

// TestReasoningEffort_InvalidConfigOutcomes states the outcome of each
// invalid ThinkingConfig on each profile class: an Effort outside the set is
// refused everywhere; a config with neither field (or a negative budget) is
// refused where Thinking or ReasoningEffort is true and dropped silently
// where both are false.
func TestReasoningEffort_InvalidConfigOutcomes(t *testing.T) {
	isolateEnv(t, baseURLEnvSources...)
	classes := []effortProfile{
		{"thinking-only/anthropic", TypeAnthropic, "claude-opus-4-5", true, false},
		{"thinking-only/gemini-2.5", TypeGoogle, "gemini-2.5-pro", true, false},
		{"effort-only/gpt-5", TypeOpenAI, "gpt-5", false, true},
		{"both/gemini-3", TypeGoogle, "gemini-3-flash-preview", true, true},
		{"neither/gpt-4o", TypeOpenAI, "gpt-4o", false, false},
		{"neither/compat", TypeOpenAICompatible, "llama3", false, false},
		{"neither/gemini-2.0", TypeGoogle, "gemini-2.0-flash", false, false},
	}
	cases := []struct {
		name         string
		cfg          llmkit.ThinkingConfig
		refuseAlways bool // refused on every profile
	}{
		{"effort-max", llmkit.ThinkingConfig{Effort: "max"}, true},
		{"effort-garbage", llmkit.ThinkingConfig{Effort: "garbage"}, true},
		{"effort-uppercase", llmkit.ThinkingConfig{Effort: "LOW"}, true},
		{"garbage-with-budget", llmkit.ThinkingConfig{Effort: "max", BudgetTokens: 1024}, true},
		{"empty", llmkit.ThinkingConfig{}, false},
		{"negative-budget", llmkit.ThinkingConfig{BudgetTokens: -5}, false},
		{"negative-budget-with-effort", llmkit.ThinkingConfig{BudgetTokens: -5, Effort: llmkit.EffortLow}, false},
	}
	for _, p := range classes {
		for _, c := range cases {
			t.Run(p.name+"/"+c.name, func(t *testing.T) {
				cfg := c.cfg
				_, body, err, hits := effortWire(t, p, nil, &cfg)
				refused := c.refuseAlways || p.thinking || p.effort
				if refused {
					if !errors.Is(err, llmkit.ErrInvalidRequest) {
						t.Fatalf("err = %v, want ErrInvalidRequest", err)
					}
					if hits != 0 {
						t.Errorf("wire hits = %d, want 0 (refusal is pre-wire)", hits)
					}
					return
				}
				if err != nil {
					t.Fatalf("err = %v, want silent drop", err)
				}
				if th, _, has := wireThinking(p.typ, body); th != nil || has {
					t.Errorf("thinking material on the wire for a dropped config: %v", body)
				}
			})
		}
	}
}

// TestReasoningEffort_ValidityTable pins what survives gating onto the wire
// for Budget-only, Effort-only, and both, on one profile per class.
func TestReasoningEffort_ValidityTable(t *testing.T) {
	isolateEnv(t, baseURLEnvSources...)
	budget := llmkit.ThinkingConfig{BudgetTokens: 1024}
	effort := llmkit.ThinkingConfig{Effort: llmkit.EffortHigh}
	both := llmkit.ThinkingConfig{BudgetTokens: 1024, Effort: llmkit.EffortHigh}

	type want struct {
		thinking bool   // a thinking config object is on the wire
		budget   bool   // it carries the budget
		effort   string // wire effort value, "" = none
		cfg      *llmkit.ThinkingConfig
	}
	cases := []struct {
		name string
		p    effortProfile
		w    []want
	}{
		{"anthropic", effortProfile{"a", TypeAnthropic, "claude-opus-4-5", true, false}, []want{
			{true, true, "", &budget}, {false, false, "", &effort}, {true, true, "", &both}}},
		{"gemini-2.5", effortProfile{"g25", TypeGoogle, "gemini-2.5-pro", true, false}, []want{
			{true, true, "", &budget}, {false, false, "", &effort}, {true, true, "", &both}}},
		{"gemini-3", effortProfile{"g3", TypeGoogle, "gemini-3-flash-preview", true, true}, []want{
			{true, true, "", &budget}, {true, false, "HIGH", &effort}, {true, false, "HIGH", &both}}},
		{"gemini-2.0", effortProfile{"g2", TypeGoogle, "gemini-2.0-flash", false, false}, []want{
			{false, false, "", &budget}, {false, false, "", &effort}, {false, false, "", &both}}},
		{"openai/gpt-5", effortProfile{"o5", TypeOpenAI, "gpt-5", false, true}, []want{
			{false, false, "", &budget}, {true, false, "high", &effort}, {true, false, "high", &both}}},
		{"openai/gpt-4o", effortProfile{"o4", TypeOpenAI, "gpt-4o", false, false}, []want{
			{false, false, "", &budget}, {false, false, "", &effort}, {false, false, "", &both}}},
	}
	names := []string{"budget-only", "effort-only", "both"}
	for _, c := range cases {
		for i, w := range c.w {
			t.Run(c.name+"/"+names[i], func(t *testing.T) {
				_, body, err, hits := effortWire(t, c.p, nil, w.cfg)
				if err != nil {
					t.Fatalf("Complete: %v", err)
				}
				if hits != 1 {
					t.Fatalf("wire hits = %d, want 1", hits)
				}
				th, gotEffort, hasEffort := wireThinking(c.p.typ, body)
				switch c.p.typ {
				case TypeOpenAI, TypeOpenAICompatible:
					// reasoning_effort is the whole thinking surface.
					if (w.effort != "") != hasEffort || (hasEffort && gotEffort != w.effort) {
						t.Errorf("reasoning_effort = %v (present %v), want %q", gotEffort, hasEffort, w.effort)
					}
					if _, ok := body["thinking"]; ok {
						t.Errorf("thinking key on an openai wire body: %v", body)
					}
					return
				}
				if (th != nil) != w.thinking {
					t.Fatalf("thinking object present = %v, want %v: %v", th != nil, w.thinking, body)
				}
				if th == nil {
					return
				}
				budgetKey := "thinkingBudget"
				if c.p.typ == TypeAnthropic {
					budgetKey = "budget_tokens"
				}
				if _, ok := th[budgetKey]; ok != w.budget {
					t.Errorf("%s present = %v, want %v: %v", budgetKey, ok, w.budget, th)
				}
				if w.effort == "" && hasEffort {
					t.Errorf("effort key %v on the wire, want none: %v", gotEffort, th)
				}
				if w.effort != "" && gotEffort != w.effort {
					t.Errorf("thinkingLevel = %v, want %q: %v", gotEffort, w.effort, th)
				}
			})
		}
	}
}

package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// okBody returns a minimal successful completion body for the named
// provider.
func okBody(name string) string {
	switch name {
	case "anthropic":
		return `{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	case "google":
		return `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`
	default:
		return `{"id":"x","object":"chat.completion","created":1,"model":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	}
}

// TestCapabilityGates_ToolChoiceRejected pins the cross-adapter contract: a
// profile reporting ToolChoice=false must reject every explicit
// Request.ToolChoice mode with an error wrapping ErrInvalidRequest BEFORE
// any wire call. Dropping "none" would escalate permissions (the model stays
// free to call the offered tools); dropping "required"/"tool" would silently
// degrade. The wire handler fails the test the moment it sees a request.
func TestCapabilityGates_ToolChoiceRejected(t *testing.T) {
	// anthropic has no table entry with ToolChoice=false (every Claude
	// generation steers tools), so its case pins the gate through the
	// Spec.Capabilities override; google uses gemini-2.0-flash-lite, which
	// really launched without function calling.
	cases := []struct {
		name     string
		spec     Spec
		model    string
		toolless bool // set ToolChoice=false via override instead of table
		mode     llmkit.ToolChoiceMode
	}{
		{"anthropic", Spec{Type: TypeAnthropic}, "claude-opus-4-5", true, llmkit.ToolChoiceRequired},
		{"anthropic-none", Spec{Type: TypeAnthropic}, "claude-opus-4-5", true, llmkit.ToolChoiceNone},
		{"openai", Spec{Type: TypeOpenAI}, "o1-mini", false, llmkit.ToolChoiceRequired},
		{"openai-none", Spec{Type: TypeOpenAI}, "o1-mini", false, llmkit.ToolChoiceNone},
		{"openai-compatible", Spec{Type: TypeOpenAICompatible}, "llama3", true, llmkit.ToolChoiceTool},
		{"google", Spec{Type: TypeGoogle}, "gemini-2.0-flash-lite", false, llmkit.ToolChoiceRequired},
		{"google-none", Spec{Type: TypeGoogle}, "gemini-2.0-flash-lite", false, llmkit.ToolChoiceNone},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/"+string(tc.mode), func(t *testing.T) {
			if tc.toolless {
				tc.spec.Capabilities = func(c llmkit.Capabilities) llmkit.Capabilities {
					c.ToolChoice = false
					return c
				}
			}
			wireCalled := false
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				wireCalled = true
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(okBody(string(tc.spec.Type))))
			})
			tc.spec.BaseURL = base
			tc.spec.Model = tc.model
			tc.spec.Secret = "k"
			client, err := New(context.Background(), tc.spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			req := simpleRequest()
			req.ToolChoice = llmkit.ToolChoice{Mode: tc.mode, Name: "read_file"}
			_, err = client.Complete(context.Background(), req)
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("error = %v, want ErrInvalidRequest (explicit ToolChoice mode on a ToolChoice=false profile)", err)
			}
			if wireCalled {
				t.Error("request reached the wire; the gate must reject before any HTTP call")
			}
		})
	}
}

// TestCapabilityGates_ToolChoiceAutoAllowed is the control: auto (and the
// zero value) are the provider default and must pass the gate even on a
// tool-less profile.
func TestCapabilityGates_ToolChoiceAutoAllowed(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okBody("google")))
	})
	spec := Spec{
		Type:    TypeGoogle,
		BaseURL: base,
		Model:   "gemini-2.0-flash-lite",
		Secret:  "k",
		Capabilities: func(c llmkit.Capabilities) llmkit.Capabilities {
			c.ToolChoice = false
			return c
		},
	}
	client, err := New(context.Background(), spec, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceAuto}
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Errorf("auto ToolChoice on a tool-less profile errored: %v (only explicit modes are rejected)", err)
	}
}

// TestCapabilityGates_ThinkingSilentlyDropped pins the Thinking gate on the
// two adapters that gate a wire parameter on it (the documented contract for
// a false feature is a silent drop, not a rejection). The captured wire body
// would carry the unsupported thinking parameter if the `&& caps.Thinking`
// gate were dropped.
func TestCapabilityGates_ThinkingSilentlyDropped(t *testing.T) {
	cases := []struct {
		name      string
		spec      Spec
		model     string
		forbidden []string // top-level (anthropic) or generationConfig (google) keys
	}{
		{"anthropic", Spec{Type: TypeAnthropic}, "claude-3-5-sonnet-20241022", []string{"thinking"}},
		{"google", Spec{Type: TypeGoogle}, "gemini-2.0-flash", []string{"thinkingConfig"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var captured map[string]any
			base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &captured)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(okBody(string(tc.spec.Type))))
			})
			tc.spec.BaseURL = base
			tc.spec.Model = tc.model
			tc.spec.Secret = "k"
			client, err := New(context.Background(), tc.spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			req := simpleRequest()
			req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
			if _, err := client.Complete(context.Background(), req); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			check := func(m map[string]any, key string) {
				if _, ok := m[key]; ok {
					t.Errorf("wire body carries %q for a Thinking=false model (gate removed?): %v", key, m)
				}
			}
			for _, key := range tc.forbidden {
				if tc.spec.Type == TypeGoogle {
					cfg, ok := captured["generationConfig"].(map[string]any)
					if !ok {
						t.Fatalf("no generationConfig object in %v", captured)
					}
					check(cfg, key)
				} else {
					check(captured, key)
				}
			}
		})
	}
}

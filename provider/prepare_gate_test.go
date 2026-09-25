package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

func forceTopP(v bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities { c.TopP = v; return c }
}
func forceTopK(v bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities { c.TopK = v; return c }
}
func forceSeed(v bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities { c.Seed = v; return c }
}
func forceStopSequences(v bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities { c.StopSequences = v; return c }
}

// captureCompleteWithCaps runs one Complete against a server that records
// the JSON request body, with the adapter's effective profile overridden by
// caps (nil = table default).
func captureCompleteWithCaps(t *testing.T, f adapterFactory, req llmkit.Request, caps func(llmkit.Capabilities) llmkit.Capabilities) map[string]any {
	t.Helper()
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody(f.name, "ok", 1, 1)))
	})
	var client llmkit.Client
	if caps != nil {
		client = f.buildWithCaps(t, base, caps)
	} else {
		client = f.build(t, base)
	}
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Fatalf("%s Complete: %v", f.name, err)
	}
	return captured
}

// wireHasField reports whether the wire body carries key at the top level
// (anthropic, openai, openai-compatible) or, for google, nested under
// generationConfig.
func wireHasField(t *testing.T, provider string, body map[string]any, key string) (any, bool) {
	t.Helper()
	if provider == "google" {
		cfg, ok := body["generationConfig"].(map[string]any)
		if !ok {
			return nil, false
		}
		v, ok := cfg[key]
		return v, ok
	}
	v, ok := body[key]
	return v, ok
}

// gateFieldRow is one (field, adapter) pair: the adapter's own wire key
// for the field (nested under generationConfig for google), a request
// mutator setting the llmkit.Request field, a caps override forcing the
// field, and the value assertion once the key is confirmed present.
type gateFieldRow struct {
	field    string
	provider string
	wireKey  string
	mutate   func(*llmkit.Request)
	force    func(bool) func(llmkit.Capabilities) llmkit.Capabilities
	checkVal func(t *testing.T, v any)
}

func gateFieldRows() []gateFieldRow {
	var rows []gateFieldRow
	topPMutate := func(r *llmkit.Request) { v := 0.5; r.TopP = &v }
	topPCheck := func(t *testing.T, v any) {
		if f, ok := v.(float64); !ok || f != 0.5 {
			t.Errorf("top_p = %#v, want 0.5", v)
		}
	}
	for _, p := range []struct{ provider, key string }{
		{"anthropic", "top_p"}, {"openai", "top_p"}, {"openai-compatible", "top_p"}, {"google", "topP"},
	} {
		rows = append(rows, gateFieldRow{"TopP", p.provider, p.key, topPMutate, forceTopP, topPCheck})
	}

	topKMutate := func(r *llmkit.Request) { v := 7; r.TopK = &v }
	topKCheck := func(t *testing.T, v any) {
		if f, ok := v.(float64); !ok || f != 7 {
			t.Errorf("top_k = %#v, want 7", v)
		}
	}
	for _, p := range []struct{ provider, key string }{
		{"anthropic", "top_k"}, {"google", "topK"},
	} {
		rows = append(rows, gateFieldRow{"TopK", p.provider, p.key, topKMutate, forceTopK, topKCheck})
	}

	seedMutate := func(r *llmkit.Request) { v := int64(42); r.Seed = &v }
	seedCheck := func(t *testing.T, v any) {
		if f, ok := v.(float64); !ok || f != 42 {
			t.Errorf("seed = %#v, want 42", v)
		}
	}
	for _, p := range []struct{ provider, key string }{
		{"openai", "seed"}, {"openai-compatible", "seed"}, {"google", "seed"},
	} {
		rows = append(rows, gateFieldRow{"Seed", p.provider, p.key, seedMutate, forceSeed, seedCheck})
	}

	stopMutate := func(r *llmkit.Request) { r.StopSequences = []string{"STOP"} }
	stopCheck := func(t *testing.T, v any) {
		arr, ok := v.([]any)
		if !ok || len(arr) != 1 || arr[0] != "STOP" {
			t.Errorf("stop sequences = %#v, want [STOP]", v)
		}
	}
	for _, p := range []struct{ provider, key string }{
		{"anthropic", "stop_sequences"}, {"openai", "stop"}, {"openai-compatible", "stop"}, {"google", "stopSequences"},
	} {
		rows = append(rows, gateFieldRow{"StopSequences", p.provider, p.key, stopMutate, forceStopSequences, stopCheck})
	}

	// Thinking keeps only its google row here: anthropic's true side and
	// the false side on both adapters are pinned by other tests; nothing
	// else checks that google carries thinkingConfig when the profile is
	// true.
	rows = append(rows, gateFieldRow{"Thinking", "google", "thinkingConfig",
		func(r *llmkit.Request) { r.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024} },
		forceThinking,
		func(t *testing.T, v any) {
			if _, ok := v.(map[string]any); !ok {
				t.Errorf("thinkingConfig = %#v, want an object", v)
			}
		}})
	return rows
}

// TestPrepare_SamplerGates_WireBody pins, for each (field, adapter) pair
// whose ceiling allows the field, that a false effective profile drops the
// vendor's wire key entirely and a true one carries the caller's value.
// Probe: wire capture, one row per (field, adapter).
func TestPrepare_SamplerGates_WireBody(t *testing.T) {
	factories := map[string]adapterFactory{}
	for _, f := range allAdapters() {
		factories[f.name] = f
	}
	for _, row := range gateFieldRows() {
		t.Run(row.field+"/"+row.provider, func(t *testing.T) {
			f := factories[row.provider]

			t.Run("false_absent", func(t *testing.T) {
				req := simpleRequest()
				row.mutate(&req)
				body := captureCompleteWithCaps(t, f, req, row.force(false))
				if v, ok := wireHasField(t, row.provider, body, row.wireKey); ok {
					t.Errorf("wire body carries %q = %#v with the profile false, want absent", row.wireKey, v)
				}
			})

			t.Run("true_present", func(t *testing.T) {
				req := simpleRequest()
				row.mutate(&req)
				body := captureCompleteWithCaps(t, f, req, row.force(true))
				v, ok := wireHasField(t, row.provider, body, row.wireKey)
				if !ok {
					t.Fatalf("wire body missing %q with the profile true: %v", row.wireKey, body)
				}
				row.checkVal(t, v)
			})
		})
	}
}

// TestPrepare_WholesaleOverride_DropsEverySamplerField pins the wholesale
// override: a Spec.Capabilities override that replaces the profile wholesale
// (`return llmkit.Capabilities{ContextWindow: 1}`, the documented "pin
// exact values" form) enforces every false field — a request setting all
// five sampler fields produces a wire body carrying none of their keys, on
// every adapter.
func TestPrepare_WholesaleOverride_DropsEverySamplerField(t *testing.T) {
	wholesale := func(llmkit.Capabilities) llmkit.Capabilities {
		return llmkit.Capabilities{ContextWindow: 1}
	}
	allFive := func(r *llmkit.Request) {
		topP, seed := 0.5, int64(42)
		topK := 7
		r.TopP = &topP
		r.TopK = &topK
		r.Seed = &seed
		r.StopSequences = []string{"STOP"}
		r.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
	}
	keysFor := map[string][]string{
		"anthropic":         {"top_p", "top_k", "stop_sequences", "thinking"}, // no seed on anthropic
		"openai":            {"top_p", "seed", "stop"},                        // no top_k/thinking on openai
		"openai-compatible": {"top_p", "seed", "stop"},
		"google":            {"topP", "topK", "seed", "stopSequences", "thinkingConfig"},
	}
	for _, f := range allAdapters() {
		t.Run(f.name, func(t *testing.T) {
			req := simpleRequest()
			allFive(&req)
			body := captureCompleteWithCaps(t, f, req, wholesale)
			for _, key := range keysFor[f.name] {
				if v, ok := wireHasField(t, f.name, body, key); ok {
					t.Errorf("wire body carries %q = %#v under a wholesale profile, want every sampler field dropped", key, v)
				}
			}
		})
	}
}

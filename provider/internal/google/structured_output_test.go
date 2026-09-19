package google

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// schemaForTest is a small, valid JSON Schema used across the
// structured-output tests. It declares a top-level object with a single
// required string property.
var schemaForTest = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

// TestStructuredOutput_Google_AttachesSchema asserts that when ResponseSchema
// is set on a Gemini request with the cap on and NO tools, the wire body
// carries responseJsonSchema + responseMimeType="application/json".
func TestStructuredOutput_Google_AttachesSchema(t *testing.T) {
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
	adapter, err := New(context.Background(), "gemini-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	gen, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing or wrong type: %T", captured["generationConfig"])
	}
	if gen["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v, want application/json", gen["responseMimeType"])
	}
	if _, ok := gen["responseJsonSchema"]; !ok {
		t.Errorf("responseJsonSchema missing from wire body")
	}
}

// TestStructuredOutput_Google_SkippedWithTools asserts that when the caller
// supplies tools alongside a schema, the Gemini adapter omits both the
// schema and the mime type (Gemini can't combine structured output with
// function-calling), but still carries the user tools.
func TestStructuredOutput_Google_SkippedWithTools(t *testing.T) {
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
	adapter, err := New(context.Background(), "gemini-test", Options{
		APIKey:  "k",
		BaseURL: base,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	req.Tools = []llmkit.ToolDef{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}}
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	gen, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing: %T", captured["generationConfig"])
	}
	if _, ok := gen["responseJsonSchema"]; ok {
		t.Errorf("responseJsonSchema present with tools — Gemini can't combine")
	}
	if _, ok := gen["responseMimeType"]; ok {
		t.Errorf("responseMimeType present with tools — Gemini can't combine")
	}
	// Tools must still be present.
	if _, ok := captured["tools"]; !ok {
		t.Error("tools missing from wire body")
	}
}

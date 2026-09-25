package openai

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

// TestStructuredOutput_OpenAI_AttachesResponseFormat asserts that when
// ResponseSchema is set on a Request and the adapter's StructuredOutput
// capability is on, the OpenAI wire body carries response_format.type ==
// "json_schema" with the schema. Also asserts the body STILL carries
// response_format when user tools are present (per spec — OpenAI permits
// the combination and the adapter honors it).
func TestStructuredOutput_OpenAI_AttachesResponseFormat(t *testing.T) {
	t.Run("schema only", func(t *testing.T) {
		var captured map[string]any
		base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
		})
		adapter, err := New("gpt-test", Options{
			APIKey:  "k",
			BaseURL: base,
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		req := simpleRequest()
		req.ResponseSchema = schemaForTest
		req.ResponseSchemaName = "answer"
		if _, err := adapter.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		rf, ok := captured["response_format"].(map[string]any)
		if !ok {
			t.Fatalf("response_format missing or wrong type: %T", captured["response_format"])
		}
		if rf["type"] != "json_schema" {
			t.Errorf("response_format.type = %v, want json_schema", rf["type"])
		}
		js, ok := rf["json_schema"].(map[string]any)
		if !ok {
			t.Fatalf("response_format.json_schema missing or wrong type: %T", rf["json_schema"])
		}
		if js["name"] != "answer" {
			t.Errorf("json_schema.name = %v, want answer", js["name"])
		}
		// Strict is intentionally NOT set on the wire. The openai-go SDK uses
		// param.Opt[bool], so the zero value is omitted; the adapter's lenient
		// policy (no Strict assignment) means the field is absent, which the
		// OpenAI API treats as non-strict. Asserting presence would couple the
		// test to a specific SDK encoding choice; absence is what we want.
		if v, ok := js["strict"]; ok && v != false {
			t.Errorf("json_schema.strict = %v, want absent or false", v)
		}
		// The schema body must round-trip through to the wire.
		sch, ok := js["schema"].(map[string]any)
		if !ok {
			t.Fatalf("json_schema.schema missing or wrong type: %T", js["schema"])
		}
		if sch["type"] != "object" {
			t.Errorf("schema.type = %v, want object", sch["type"])
		}
	})

	t.Run("honored with tools", func(t *testing.T) {
		var captured map[string]any
		base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
		})
		adapter, err := New("gpt-test", Options{
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
		if _, ok := captured["response_format"]; !ok {
			t.Error("response_format missing — schema must be honored alongside tools")
		}
		if _, ok := captured["tools"]; !ok {
			t.Error("tools missing — adapter dropped user tools")
		}
	})
}

// TestStructuredOutput_OpenAICompatible_AttachesResponseFormat mirrors the
// openai test for the openai-compatible profile. With the cap on (forced,
// because the conservative default is false), the wire body must carry
// response_format.
func TestStructuredOutput_OpenAICompatible_AttachesResponseFormat(t *testing.T) {
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
	adapter, err := New("llama-test", Options{
		APIKey:       "k",
		BaseURL:      base,
		Compatible:   true,
		Capabilities: structuredOutputOverride(true),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	rf, ok := captured["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format missing: %T", captured["response_format"])
	}
	if rf["type"] != "json_schema" {
		t.Errorf("response_format.type = %v, want json_schema", rf["type"])
	}
}

// TestStructuredOutput_OpenAI_GatedByCapability asserts that when the cap is
// OFF, the adapter silently drops ResponseSchema and no response_format field
// appears on the wire.
func TestStructuredOutput_OpenAI_GatedByCapability(t *testing.T) {
	var captured map[string]any
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
	adapter, err := New("gpt-test", Options{
		APIKey:       "k",
		BaseURL:      base,
		Capabilities: structuredOutputOverride(false),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ResponseSchema = schemaForTest
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if v, ok := captured["response_format"]; ok {
		t.Errorf("response_format present with cap off: %v", v)
	}
}

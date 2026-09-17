package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// mapSchemaForTest mirrors the real reproducer/patch "files" contract: a
// free-form string map expressed as an object-valued additionalProperties,
// wrapped in a root object that itself closes with additionalProperties:false.
// This is the exact shape MiniMax rejects with a 400 (see bugbot-d80).
var mapSchemaForTest = json.RawMessage(`{
  "type":"object",
  "properties":{
    "files":{"type":"object","additionalProperties":{"type":"string"},"minProperties":1}
  },
  "required":["files"],
  "additionalProperties":false
}`)

func mustUnmarshalSchema(t *testing.T, raw string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	return m
}

func childMap(t *testing.T, m map[string]any, key string) map[string]any {
	t.Helper()
	c, ok := m[key].(map[string]any)
	if !ok {
		t.Fatalf("key %q is %T, want object", key, m[key])
	}
	return c
}

func ptr(b bool) *bool { return &b }

// captureBody starts a server that records the request body as a decoded map and
// replies with a minimal successful completion.
func captureBody(t *testing.T, captured *map[string]any) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("ok", 1, 1)))
	})
}

// TestSchemaCompat_OpenAICompatible_DowngradesResponseFormat asserts that an
// openai-compatible adapter with the downgrade flag rewrites object-valued
// additionalProperties to boolean true in the response_format schema on the
// wire, while the root additionalProperties:false is preserved.
func TestSchemaCompat_OpenAICompatible_DowngradesResponseFormat(t *testing.T) {
	var captured map[string]any
	base := captureBody(t, &captured)
	adapter := New("llama-test", Options{
		APIKey:           "k",
		BaseURL:          base,
		Compatible:       true,
		StructuredOutput: ptr(true),
	})
	req := simpleRequest()
	req.ResponseSchema = mapSchemaForTest
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	schema := childMap(t, childMap(t, childMap(t, captured, "response_format"), "json_schema"), "schema")
	files := childMap(t, childMap(t, schema, "properties"), "files")
	if files["additionalProperties"] != true {
		t.Errorf("wire files.additionalProperties = %#v, want true", files["additionalProperties"])
	}
	if schema["additionalProperties"] != false {
		t.Errorf("wire root additionalProperties = %#v, want false", schema["additionalProperties"])
	}
}

// TestSchemaCompat_OpenAICompatible_DowngradesToolParameters asserts the same
// downgrade is applied to tool parameter schemas — the sandbox_exec tool's
// "files" argument is the latent case this protects.
func TestSchemaCompat_OpenAICompatible_DowngradesToolParameters(t *testing.T) {
	var captured map[string]any
	base := captureBody(t, &captured)
	adapter := New("llama-test", Options{
		APIKey:           "k",
		BaseURL:          base,
		Compatible:       true,
		StructuredOutput: ptr(true),
	})
	req := simpleRequest()
	req.Tools = []llmkit.ToolDef{{Name: "sandbox_exec", Parameters: mapSchemaForTest}}
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tools, ok := captured["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools missing or wrong shape: %#v", captured["tools"])
	}
	params := childMap(t, childMap(t, tools[0].(map[string]any), "function"), "parameters")
	files := childMap(t, childMap(t, params, "properties"), "files")
	if files["additionalProperties"] != true {
		t.Errorf("wire tool files.additionalProperties = %#v, want true", files["additionalProperties"])
	}
}

// TestSchemaCompat_OpenAI_PreservesSubschema locks the blast radius: a
// first-party OpenAI adapter (flag off) leaves the object-valued
// additionalProperties intact, because OpenAI accepts the subschema form.
func TestSchemaCompat_OpenAI_PreservesSubschema(t *testing.T) {
	var captured map[string]any
	base := captureBody(t, &captured)
	adapter := New("gpt-test", Options{
		APIKey:  "k",
		BaseURL: base,
		// Compatible deliberately left false: first-party OpenAI keeps the
		// subschema form.
	})
	req := simpleRequest()
	req.ResponseSchema = mapSchemaForTest
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	schema := childMap(t, childMap(t, childMap(t, captured, "response_format"), "json_schema"), "schema")
	files := childMap(t, childMap(t, schema, "properties"), "files")
	if _, ok := files["additionalProperties"].(map[string]any); !ok {
		t.Errorf("openai files.additionalProperties = %#v, want preserved object", files["additionalProperties"])
	}
}

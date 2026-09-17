package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
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

func childMap(t *testing.T, m map[string]any, key string) map[string]any {
	t.Helper()
	c, ok := m[key].(map[string]any)
	if !ok {
		t.Fatalf("key %q is %T, want object", key, m[key])
	}
	return c
}

// captureBody starts a server that records the request body as a decoded map and
// replies with a minimal successful completion.
func captureBody(t *testing.T, captured *map[string]any) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("openai-compatible", "ok", 1, 1)))
	})
}

// TestSchemaCompat_Registry_OpenAICompatibleEnablesDowngrade exercises the full
// factory path: a provider configured as openai-compatible (with structured
// output forced on) must downgrade object additionalProperties on the wire,
// proving New wires requireBoolAdditionalProps=true for Compatible=true.
func TestSchemaCompat_Registry_OpenAICompatibleEnablesDowngrade(t *testing.T) {
	var captured map[string]any
	base := captureBody(t, &captured)
	on := true
	client, err := New(context.Background(), Spec{
		Type:             TypeOpenAICompatible,
		BaseURL:          base,
		StructuredOutput: &on,
	}, "minimax", "MiniMax-M3", "k", Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ResponseSchema = mapSchemaForTest
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	schema := childMap(t, childMap(t, childMap(t, captured, "response_format"), "json_schema"), "schema")
	files := childMap(t, childMap(t, schema, "properties"), "files")
	if files["additionalProperties"] != true {
		t.Errorf("registry wire files.additionalProperties = %#v, want true", files["additionalProperties"])
	}
}

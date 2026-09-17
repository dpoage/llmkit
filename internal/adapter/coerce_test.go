package adapter

import (
	"encoding/json"
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

// TestCoerceBoolAdditionalProperties exercises the schema-aware downgrade in
// isolation: object/array additionalProperties become boolean true, booleans
// (including false) are preserved, and the walk never touches additionalProperties
// that appears as a PROPERTY NAME or as instance data inside enum/const.
func TestCoerceBoolAdditionalProperties(t *testing.T) {
	t.Run("object additionalProperties downgraded, bool false preserved", func(t *testing.T) {
		m := mustUnmarshalSchema(t, string(mapSchemaForTest))
		CoerceBoolAdditionalProperties(m)
		files := childMap(t, childMap(t, m, "properties"), "files")
		if files["additionalProperties"] != true {
			t.Errorf("files.additionalProperties = %#v, want true", files["additionalProperties"])
		}
		if m["additionalProperties"] != false {
			t.Errorf("root additionalProperties = %#v, want false (preserved)", m["additionalProperties"])
		}
		// Unrelated constraints survive the rewrite.
		if _, ok := files["minProperties"]; !ok {
			t.Error("files.minProperties dropped by rewrite")
		}
	})

	t.Run("boolean additionalProperties passed through", func(t *testing.T) {
		for _, raw := range []string{`{"additionalProperties":true}`, `{"additionalProperties":false}`} {
			m := mustUnmarshalSchema(t, raw)
			want := m["additionalProperties"]
			CoerceBoolAdditionalProperties(m)
			if m["additionalProperties"] != want {
				t.Errorf("%s: additionalProperties = %#v, want %#v", raw, m["additionalProperties"], want)
			}
		}
	})

	t.Run("array-valued additionalProperties downgraded to true", func(t *testing.T) {
		m := mustUnmarshalSchema(t, `{"additionalProperties":["weird"]}`)
		CoerceBoolAdditionalProperties(m)
		if m["additionalProperties"] != true {
			t.Errorf("additionalProperties = %#v, want true", m["additionalProperties"])
		}
	})

	t.Run("nested in items and $defs", func(t *testing.T) {
		m := mustUnmarshalSchema(t, `{
		  "type":"array",
		  "items":{"type":"object","additionalProperties":{"type":"number"}},
		  "$defs":{"M":{"type":"object","additionalProperties":{"type":"string"}}}
		}`)
		CoerceBoolAdditionalProperties(m)
		if childMap(t, m, "items")["additionalProperties"] != true {
			t.Error("items.additionalProperties not downgraded")
		}
		if childMap(t, childMap(t, m, "$defs"), "M")["additionalProperties"] != true {
			t.Error("$defs.M.additionalProperties not downgraded")
		}
	})

	t.Run("property literally named additionalProperties is untouched", func(t *testing.T) {
		// A schema describing an object that HAS a property called
		// "additionalProperties" — the inner value is a subschema, not the keyword.
		m := mustUnmarshalSchema(t, `{"type":"object","properties":{"additionalProperties":{"type":"string"}}}`)
		CoerceBoolAdditionalProperties(m)
		prop := childMap(t, m, "properties")
		ap, ok := prop["additionalProperties"].(map[string]any)
		if !ok {
			t.Fatalf("property named additionalProperties was corrupted to %#v", prop["additionalProperties"])
		}
		if ap["type"] != "string" {
			t.Errorf("property schema mangled: %#v", ap)
		}
	})

	t.Run("object literals in enum and const are untouched", func(t *testing.T) {
		m := mustUnmarshalSchema(t, `{
		  "enum":[{"additionalProperties":{"x":1}}],
		  "const":{"additionalProperties":{"y":2}}
		}`)
		CoerceBoolAdditionalProperties(m)
		enum, ok := m["enum"].([]any)
		if !ok || len(enum) != 1 {
			t.Fatalf("enum mangled: %#v", m["enum"])
		}
		if _, ok := enum[0].(map[string]any)["additionalProperties"].(map[string]any); !ok {
			t.Errorf("enum literal additionalProperties coerced: %#v", enum[0])
		}
		if _, ok := childMap(t, m, "const")["additionalProperties"].(map[string]any); !ok {
			t.Errorf("const literal additionalProperties coerced: %#v", m["const"])
		}
	})
}

package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// JSON-Schema 2020-12 meta-key fixtures. Keys never contain "/" or "~", so
// JSON-pointer tokens need no escaping; no property is named after a meta key.
const (
	schemaMetaFixtureRoot = `{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:llmkit:test","type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`
	schemaMetaFixtureRef  = `{"type":"object","properties":{"loc":{"$ref":"#/$defs/Loc"}},"required":["loc"],"$defs":{"Loc":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"}},"required":["lat","lon"]}}}`
	// Recursive: $defs/N/properties/kids/items/$ref points back at #/$defs/N.
	schemaMetaFixtureRecursive = `{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:llmkit:test","type":"object","properties":{"root":{"$ref":"#/$defs/N"}},"required":["root"],"$defs":{"N":{"type":"object","properties":{"name":{"type":"string"},"kids":{"type":"array","items":{"$ref":"#/$defs/N"}}},"required":["name"]}}}`
)

var schemaMetaKeys = map[string]bool{"$schema": true, "$id": true, "$defs": true, "$ref": true}

// TestSchemaMetaKeysPassThrough pins, per ruling D1=B (llmkit-rli.1.3), that
// every adapter sends $schema/$id/$defs/$ref — recursive $ref included —
// verbatim on both carriers: ToolDef.Parameters and Request.ResponseSchema
// (StructuredOutput forced on). Only pinned rewrites exist: anthropic forces
// root "type":"object" (toAnthropicInputSchema); openai-compatible coerces
// object-valued additionalProperties to true (coerceBoolAdditionalProperties),
// and these fixtures carry none.
func TestSchemaMetaKeysPassThrough(t *testing.T) {
	fixtures := []struct{ name, raw string }{
		{"root_meta_keys", schemaMetaFixtureRoot},
		{"defs_nonrecursive_ref", schemaMetaFixtureRef},
		{"defs_recursive_ref", schemaMetaFixtureRecursive},
	}
	for _, f := range allAdapters() {
		for _, carrier := range []string{"tool", "response"} {
			for _, fx := range fixtures {
				t.Run(f.name+"/"+carrier+"/"+fx.name, func(t *testing.T) {
					var captured map[string]any
					base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
						_ = json.NewDecoder(r.Body).Decode(&captured)
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(mockTextBody(f.name, "ok", 1, 1)))
					})
					req := simpleRequest()
					var client llmkit.Client
					if carrier == "tool" {
						req.Tools = []llmkit.ToolDef{{
							Name:        "get_weather",
							Description: "Returns the weather for a city.",
							Parameters:  json.RawMessage(fx.raw),
						}}
						client = f.build(t, base)
					} else {
						req.ResponseSchema = json.RawMessage(fx.raw)
						client = f.buildWithCaps(t, base, forceStructuredOutput(true))
					}
					if _, err := client.Complete(context.Background(), req); err != nil {
						t.Fatalf("Complete: %v", err)
					}
					if captured == nil {
						t.Fatal("server received no decodable request body")
					}
					got := wireSchemaFor(t, f.name, carrier, captured)
					var fixture map[string]any
					if err := json.Unmarshal([]byte(fx.raw), &fixture); err != nil {
						t.Fatalf("fixture decode: %v", err)
					}
					want := map[string]any{}
					collectSchemaMetaPointers(fixture, "", want)
					if len(want) == 0 {
						t.Fatal("fixture carries no meta key")
					}
					for ptr, val := range want {
						at, ok := lookupSchemaPointer(got, ptr)
						if !ok {
							t.Errorf("wire schema lacks pointer %q; schema = %#v", ptr, got)
							continue
						}
						if !reflect.DeepEqual(at, val) {
							t.Errorf("wire schema pointer %q = %#v, want %#v", ptr, at, val)
						}
					}
				})
			}
		}
	}
}

// wireSchemaFor locates the JSON Schema object the provider put on the wire:
// anthropic tools[].input_schema (the user tool, or the synthetic
// structured-output tool for the response carrier); openai-family
// tools[].function.parameters / response_format.json_schema.schema; google
// tools[].functionDeclarations[].parametersJsonSchema /
// generationConfig.responseJsonSchema.
func wireSchemaFor(t *testing.T, provider, carrier string, body map[string]any) map[string]any {
	t.Helper()
	dig := func(m map[string]any, keys ...string) map[string]any {
		t.Helper()
		cur := m
		for _, k := range keys {
			next, ok := cur[k].(map[string]any)
			if !ok {
				t.Fatalf("%s %s: key %q = %#v, want object (path %v)", provider, carrier, k, cur[k], keys)
			}
			cur = next
		}
		return cur
	}
	digList := func(m map[string]any, key string) []any {
		t.Helper()
		l, ok := m[key].([]any)
		if !ok || len(l) == 0 {
			t.Fatalf("%s %s: key %q = %#v, want a non-empty array", provider, carrier, key, m[key])
		}
		return l
	}
	toolObj := func() map[string]any {
		tools := digList(body, "tools")
		tool, ok := tools[0].(map[string]any)
		if !ok {
			t.Fatalf("%s %s: tools[0] = %#v, want object", provider, carrier, tools[0])
		}
		return tool
	}
	switch provider {
	case "anthropic":
		return dig(toolObj(), "input_schema")
	case "openai", "openai-compatible":
		if carrier == "response" {
			return dig(dig(dig(body, "response_format"), "json_schema"), "schema")
		}
		return dig(toolObj(), "function", "parameters")
	case "google":
		if carrier == "response" {
			return dig(dig(body, "generationConfig"), "responseJsonSchema")
		}
		decl, ok := digList(toolObj(), "functionDeclarations")[0].(map[string]any)
		if !ok {
			t.Fatalf("%s %s: functionDeclarations[0] is not an object", provider, carrier)
		}
		schema, ok := decl["parametersJsonSchema"].(map[string]any)
		if !ok {
			t.Fatalf("%s %s: parametersJsonSchema = %#v, want object", provider, carrier, decl["parametersJsonSchema"])
		}
		return schema
	}
	t.Fatalf("unknown provider %q", provider)
	return nil
}

// collectSchemaMetaPointers records every JSON pointer at which the fixture
// holds a meta key, recursing into values so nested pointers are covered.
func collectSchemaMetaPointers(v any, prefix string, out map[string]any) {
	switch t := v.(type) {
	case map[string]any:
		for k, child := range t {
			ptr := prefix + "/" + k
			if schemaMetaKeys[k] {
				out[ptr] = child
			}
			collectSchemaMetaPointers(child, ptr, out)
		}
	case []any:
		for i, child := range t {
			collectSchemaMetaPointers(child, prefix+"/"+strconv.Itoa(i), out)
		}
	}
}

func lookupSchemaPointer(root any, ptr string) (any, bool) {
	cur := root
	for _, tok := range strings.Split(ptr, "/")[1:] {
		switch node := cur.(type) {
		case map[string]any:
			v, ok := node[tok]
			if !ok {
				return nil, false
			}
			cur = v
		case []any:
			i, err := strconv.Atoi(tok)
			if err != nil || i < 0 || i >= len(node) {
				return nil, false
			}
			cur = node[i]
		default:
			return nil, false
		}
	}
	return cur, true
}

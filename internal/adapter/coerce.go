package adapter

// CoerceBoolAdditionalProperties rewrites an object/array-valued
// "additionalProperties" in a decoded JSON Schema to the permissive boolean
// true, recursing through nested subschemas. Some OpenAI-compatible backends
// (notably MiniMax) run a strict schema type-checker over response_format /
// tool parameters that accepts ONLY a boolean additionalProperties and rejects
// the JSON-Schema subschema form (e.g. {"type":"string"} used for free-form
// string maps) with HTTP 400 before any tokens are generated. Downgrading to
// true keeps the map "open" on the wire; the value-type constraint is still
// enforced locally by the caller's validator against the ORIGINAL schema, so no
// validation is lost. A boolean additionalProperties (including false) is passed
// through untouched.
//
// The walk is schema-aware so it never corrupts a schema that legitimately
// uses "additionalProperties" as data rather than the keyword:
//   - name-keyed subschema maps (properties, patternProperties, $defs,
//     definitions, dependentSchemas) recurse into their VALUES only, so a
//     property literally NAMED "additionalProperties" is left intact;
//   - value-bearing keywords (enum, const, default, examples) hold literal
//     instance data, not schemas, and are never descended into.
//
// Nodes are mutated in place; the return value is the same node so callers can
// reassign generically.
func CoerceBoolAdditionalProperties(v any) any {
	switch node := v.(type) {
	case map[string]any:
		for k, child := range node {
			switch k {
			case "additionalProperties":
				if _, isBool := child.(bool); !isBool {
					node[k] = true
				}
			case "properties", "patternProperties", "$defs", "definitions", "dependentSchemas":
				if named, ok := child.(map[string]any); ok {
					for name, sub := range named {
						named[name] = CoerceBoolAdditionalProperties(sub)
					}
				}
			case "enum", "const", "default", "examples":
				// literal instance data, not subschemas — leave untouched.
			default:
				node[k] = CoerceBoolAdditionalProperties(child)
			}
		}
		return node
	case []any:
		for i, child := range node {
			node[i] = CoerceBoolAdditionalProperties(child)
		}
		return node
	default:
		return v
	}
}

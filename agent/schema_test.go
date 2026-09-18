package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// schemaInner is a nested struct fixture: both fields lack omitempty, so
// SchemaOf must list both as required at the NESTED object level.
type schemaInner struct {
	Name  string `json:"name"`
	Limit int    `json:"limit"`
}

// schemaOuter exercises every shape SchemaOf must inline: a nested struct, a
// slice of structs, and a tagged numeric constraint.
type schemaOuter struct {
	Path    string        `json:"path"`
	Inner   schemaInner   `json:"inner"`
	Entries []schemaInner `json:"entries,omitempty"`
	Year    int           `json:"year" jsonschema:"minimum=1000"`
}

// assertNoRefs walks a decoded schema and fails when it carries $ref/$defs
// anywhere (validateSchema ignores $ref, so a referenced subschema's
// constraints would be silently unenforced), or when an object node lacks the
// bool "additionalProperties": false (the form provider/openai's Compatible
// downgrade requires).
func assertNoRefs(t *testing.T, path string, node any) {
	t.Helper()
	switch n := node.(type) {
	case map[string]any:
		for _, kw := range []string{"$ref", "$defs", "$schema", "$id"} {
			if _, ok := n[kw]; ok {
				t.Errorf("%s carries %q: schemas must be fully inlined", path, kw)
			}
		}
		if _, hasProps := n["properties"]; hasProps {
			ap, ok := n["additionalProperties"]
			if !ok {
				t.Errorf("%s is an object without additionalProperties", path)
			} else if b, isBool := ap.(bool); !isBool || b {
				t.Errorf("%s additionalProperties = %v, want false (bool)", path, ap)
			}
		}
		for k, v := range n {
			assertNoRefs(t, path+"."+k, v)
		}
	case []any:
		for i, v := range n {
			assertNoRefs(t, fmt.Sprintf("%s[%d]", path, i), v)
		}
	}
}

func decodeSchema(t *testing.T, s json.RawMessage) map[string]any {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(s, &doc); err != nil {
		t.Fatalf("schema does not decode: %v", err)
	}
	return doc
}

// TestSchemaOf_InlinesNestedStructs pins the reflector configuration: no
// $ref/$defs (and no $schema/$id headers), and additionalProperties:false as
// a bool at every object level — root, nested struct, and slice element.
func TestSchemaOf_InlinesNestedStructs(t *testing.T) {
	doc := decodeSchema(t, SchemaOf[schemaOuter]())
	assertNoRefs(t, "root", doc)

	// Required is derived from the missing-omitempty rule at every level.
	rootReq := mustStrings(t, doc["required"])
	if got, want := strings.Join(rootReq, ","), "path,inner,year"; got != want {
		t.Errorf("root required = %v, want %s (entries has omitempty; the rest do not)", rootReq, want)
	}
	inner := doc["properties"].(map[string]any)["inner"].(map[string]any)
	innerReq := mustStrings(t, inner["required"])
	if got, want := strings.Join(innerReq, ","), "name,limit"; got != want {
		t.Errorf("inner required = %v, want %s — nested constraints must not be dropped", innerReq, want)
	}
}

// TestSchemaOf_NestedConstraintsEnforcedByValidator proves the inlining
// matters: a generated schema must PASS a valid instance and REJECT one
// missing a nested required field — the failure mode $ref support would hide,
// since validateSchema ignores $ref.
func TestSchemaOf_NestedConstraintsEnforcedByValidator(t *testing.T) {
	schema := SchemaOf[schemaOuter]()
	valid := `{"path":"a.go","inner":{"name":"x","limit":3},"entries":[{"name":"e","limit":1}],"year":2026}`
	if err := validateSchema(schema, []byte(valid)); err != nil {
		t.Fatalf("valid instance rejected: %v", err)
	}

	missingNested := `{"path":"a.go","inner":{"limit":3},"year":2026}`
	err := validateSchema(schema, []byte(missingNested))
	if err == nil {
		t.Fatal("instance missing inner.name was accepted — nested required constraints are NOT enforced")
	}
	if !strings.Contains(err.Error(), "inner") || !strings.Contains(err.Error(), `missing required field "name"`) {
		t.Errorf("error = %v, want it to name the violated nested path (inner, required field name)", err)
	}
	if err := validateSchema(schema, []byte(`{"path":"a.go","inner":{"name":"x","limit":3},"year":999}`)); err == nil {
		t.Error("year=999 accepted — the jsonschema minimum tag was not enforced")
	}
}

// TestSchemaOf_MinimumTag pins tag-driven constraints: the generated schema
// carries minimum:1000 from the jsonschema tag.
func TestSchemaOf_MinimumTag(t *testing.T) {
	doc := decodeSchema(t, SchemaOf[schemaOuter]())
	year := doc["properties"].(map[string]any)["year"].(map[string]any)
	if year["minimum"] != float64(1000) {
		t.Errorf("year schema = %v, want minimum 1000 from the jsonschema tag", year)
	}
}

// TestSchemaOf_NonStructPanics pins the programming-error contract: deriving
// a schema from a non-struct type is a mistake that must surface loudly at
// the first call, not produce a degenerate schema.
func TestSchemaOf_NonStructPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("SchemaOf[string] did not panic")
		}
	}()
	SchemaOf[string]()
}

// TestFunc_RoundTripThroughRunner exercises the full Func path with the fake
// client: the derived Def advertises the struct's schema, the scripted tool
// call is decoded into Args, and the function's result is fed back to the
// model as the tool result of the next turn.
func TestFunc_RoundTripThroughRunner(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "add", `{"a":40,"b":2}`, 10, 4),
		textResp("the sum is 42", 8, 3),
	)
	tool := Func("add", "adds two numbers", func(_ context.Context, p addArgs) (string, error) {
		return fmt.Sprintf("%g", p.A+p.B), nil
	})

	def := tool.Def()
	if def.Name != "add" || def.Description != "adds two numbers" {
		t.Errorf("Def = %+v, want the name/description Func was given", def)
	}
	doc := decodeSchema(t, def.Parameters)
	props, _ := doc["properties"].(map[string]any)
	if _, ok := props["a"]; !ok {
		t.Errorf("parameter schema = %s, want properties derived from addArgs", def.Parameters)
	}
	if ap, ok := doc["additionalProperties"].(bool); !ok || ap {
		t.Errorf("parameter schema additionalProperties = %v, want false (bool)", doc["additionalProperties"])
	}

	r := NewRunner(fc, []Tool{tool}, "sys")
	out, err := r.Run(context.Background(), "add 40 and 2")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.FinalText != "the sum is 42" {
		t.Errorf("FinalText = %q, want the post-tool answer", out.FinalText)
	}
	// The decoded args reached the function (result 42) and that result was
	// fed back as the tool-result turn of the second completion.
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2 (tool call + final)", len(fc.requests))
	}
	var sawResult bool
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult && m.Text() == "42" {
			sawResult = true
		}
	}
	if !sawResult {
		t.Error("the function result 42 never reached the model as a tool result")
	}
}

// addArgs is the argument struct for the Func test tools; SchemaOf derives
// the advertised parameter schema from these fields.
type addArgs struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

// TestFunc_BadArgsFeedModelError pins the error contract: malformed model
// arguments surface as a model-recoverable "ERROR:" tool result, not a loop
// failure.
func TestFunc_BadArgsFeedModelError(t *testing.T) {
	fc := newFakeClient(
		toolResp("c1", "add", `{"a": not json}`, 10, 4),
		textResp("recovered", 8, 3),
	)
	tool := Func("add", "adds two numbers", func(_ context.Context, p addArgs) (string, error) {
		return fmt.Sprintf("%g", p.A+p.B), nil
	})
	r := NewRunner(fc, []Tool{tool}, "sys")
	if _, err := r.Run(context.Background(), "task"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	var sawErrResult bool
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult && strings.HasPrefix(m.Text(), "ERROR: invalid arguments:") {
			sawErrResult = true
		}
	}
	if !sawErrResult {
		t.Error("malformed args did not produce an ERROR: invalid arguments tool result")
	}
}

// TestRunJSONAs_PopulatesT pins the typed RunJSON form: schema derived from T,
// validated answer unmarshaled into a fresh T, outcome passed through.
func TestRunJSONAs_PopulatesT(t *testing.T) {
	fc := newFakeClient(textResp(`{"path":"a.go","note":"fix imports"}`, 5, 5))
	r := NewRunner(fc, nil, "sys")

	got, out, err := RunJSONAs[item](context.Background(), r, "summarize the report")
	if err != nil {
		t.Fatalf("RunJSONAs: %v", err)
	}
	if got.Path != "a.go" || got.Note != "fix imports" {
		t.Errorf("got = %+v, want the model's answer decoded into item", got)
	}
	if out == nil || out.Transcript == nil {
		t.Error("outcome not returned for inspection")
	}
	// The derived schema travels via the prompt instruction (the fake client
	// reports no StructuredOutput capability), so it must reach the request
	// as text.
	if !strings.Contains(fc.requests[0].Messages[0].Text(), `"path"`) {
		t.Error("derived schema never reached the request prompt")
	}
}

// mustStrings renders a decoded JSON array of strings for comparison.
func mustStrings(t *testing.T, v any) []string {
	t.Helper()
	arr, ok := v.([]any)
	if !ok {
		t.Fatalf("required = %v, want a JSON array", v)
	}
	out := make([]string, 0, len(arr))
	for _, e := range arr {
		s, ok := e.(string)
		if !ok {
			t.Fatalf("required element %v is not a string", e)
		}
		out = append(out, s)
	}
	return out
}

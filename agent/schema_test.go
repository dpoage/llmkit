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
// bool "additionalProperties": false (the form provider/internal/openai's
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
		t.Errorf("root required = %v, want %s (entries has omitempty; the rest do not)", got, want)
	}
	inner := doc["properties"].(map[string]any)["inner"].(map[string]any)
	innerReq := mustStrings(t, inner["required"])
	if got, want := strings.Join(innerReq, ","), "name,limit"; got != want {
		t.Errorf("inner required = %v, want %s — nested constraints must not be dropped", got, want)
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

// schemaPointers pins the pointer and any field mapping: *T maps to T's
// schema verbatim (not "null"-typed) and any maps to `true`; both stay
// required (no omitempty).
type schemaPointers struct {
	Inner *schemaInner `json:"inner"`
	Any   any          `json:"any"`
}

func TestSchemaOf_PointerAndAnyMapping(t *testing.T) {
	doc := decodeSchema(t, SchemaOf[schemaPointers]())
	props := doc["properties"].(map[string]any)
	inner := props["inner"].(map[string]any)
	if inner["type"] != "object" {
		t.Errorf("*schemaInner mapped to %v, want the plain object schema of schemaInner (non-nullable)", inner)
	}
	if props["any"] != true {
		t.Errorf("any mapped to %v, want the accept-everything schema true", props["any"])
	}
	req := strings.Join(mustStrings(t, doc["required"]), ",")
	if req != "inner,any" {
		t.Errorf("required = %q, want inner,any — pointer and any fields are required like any other", req)
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

// --- recursive types --------------------------------------------------------
//
// SchemaOf must reject cycles BEFORE the reflector sees them: inlined
// schemas cannot express recursion and invopop dies with a fatal,
// unrecoverable stack overflow (not a recoverable panic) on one.

type cyclicNode struct {
	Children []*cyclicNode `json:"children,omitempty"`
}

type cyclicA struct {
	B *cyclicB `json:"b"`
}

type cyclicB struct {
	A []*cyclicA `json:"a,omitempty"`
}

type cyclicMap struct {
	Next map[string]*cyclicMap `json:"next"`
}

// prInner is embedded (anonymous, unexported) into prOuter: invopop's
// shouldEmbed recurses into embedded structs regardless of exportedness, so
// the cycle MUST be caught even though the field is not exported.
type prInner struct {
	Self *prInner `json:"self,omitempty"`
}

type prOuter struct {
	prInner
	Name string `json:"name"`
}

// Non-struct named recursive types: recursion need not pass through a
// struct, so the walker must track slices and maps on its DFS path too.
type cyclicTree []cyclicTree

type cyclicM map[string]cyclicM

type cyclicForest struct {
	Trees   cyclicTree `json:"trees"`
	Mapping cyclicM    `json:"mapping"`
}

// skippedCycleStruct hides its cycle behind fields invopop drops entirely
// (json:"-" and jsonschema:"-"): the generated schema must be ACCEPTED and
// contain only the surviving property — the walker must not reject what the
// reflector never visits.
type skippedCycleStruct struct {
	Name string              `json:"name"`
	Next *skippedCycleStruct `json:"-"`
	Hide *skippedCycleStruct `json:"hide,omitempty" jsonschema:"-"`
}

func TestSchemaOf_RecursiveTypePanics(t *testing.T) {
	t.Run("direct", func(t *testing.T) {
		assertSchemaCyclePanic(t, "cyclicNode", func() { SchemaOf[cyclicNode]() })
	})
	t.Run("indirect", func(t *testing.T) {
		assertSchemaCyclePanic(t, "cyclicA", func() { SchemaOf[cyclicA]() })
	})
	t.Run("map value", func(t *testing.T) {
		assertSchemaCyclePanic(t, "cyclicMap", func() { SchemaOf[cyclicMap]() })
	})
	t.Run("embedded unexported", func(t *testing.T) {
		assertSchemaCyclePanic(t, "prOuter", func() { SchemaOf[prOuter]() })
	})
	t.Run("non-struct slice", func(t *testing.T) {
		assertSchemaCyclePanic(t, "cyclicForest", func() { SchemaOf[cyclicForest]() })
	})
}

// cyclicKey is comparable and self-cyclic, usable as a map key: invopop's
// reflectMap never reflects the key type, so the cycle is schema-invisible
// and must be accepted.
type cyclicKey struct {
	Next *cyclicKey `json:"next,omitempty"`
}

// inlineDoublePtr hides its cycle behind a json:",inline" field two pointer
// levels deep: invopop unwraps exactly one level, sees a non-struct, and
// drops the field — so must the walker.
type inlineDoublePtr struct {
	Name string            `json:"name"`
	Self **inlineDoublePtr `json:",inline"`
}

func TestSchemaOf_UnreflectedShapesAccepted(t *testing.T) {
	t.Run("map key cycle", func(t *testing.T) {
		type byKey struct {
			Roots map[cyclicKey]string `json:"roots"`
		}
		doc := decodeSchema(t, SchemaOf[byKey]())
		props := doc["properties"].(map[string]any)
		roots := props["roots"].(map[string]any)
		ap, ok := roots["additionalProperties"].(map[string]any)
		if !ok || ap["type"] != "string" {
			t.Errorf("map schema = %v, want additionalProperties carrying the VALUE (string) schema — the cyclic key type must never be reflected", roots)
		}
		if ap != nil {
			if _, hasCycle := ap["$ref"]; hasCycle {
				t.Errorf("map schema = %v, want no cycle reaching the schema", roots)
			}
		}
	})
	t.Run("inline double pointer", func(t *testing.T) {
		doc := decodeSchema(t, SchemaOf[inlineDoublePtr]())
		props, ok := doc["properties"].(map[string]any)
		if !ok {
			t.Fatalf("schema = %v, want a properties object", doc)
		}
		if len(props) != 1 {
			t.Errorf("schema properties = %v, want exactly {name} — the **T inline field must be dropped like invopop drops it", props)
		}
	})
}

func TestSchemaOf_SkippedCycleAccepted(t *testing.T) {
	s := SchemaOf[skippedCycleStruct]()
	doc := decodeSchema(t, s)
	props, ok := doc["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema = %s, want a properties object", s)
	}
	if _, ok := props["name"]; !ok {
		t.Errorf("schema properties = %v, want the surviving name property", props)
	}
	if len(props) != 1 {
		t.Errorf("schema properties = %v, want exactly {name} — the json:\"-\" and jsonschema:\"-\" cycles must be dropped, not rejected", props)
	}
}

// assertSchemaCyclePanic asserts fn panics with the documented cycle message
// naming both the root and the cyclic type.
func assertSchemaCyclePanic(t *testing.T, root string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("SchemaOf[%s] did not panic — a recursive type would fatal-overflow the reflector", root)
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value %v is not a string", r)
		}
		if !strings.Contains(msg, "recursive type") || !strings.Contains(msg, root) {
			t.Errorf("panic = %q, want the cycle message naming %s", msg, root)
		}
	}()
	fn()
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

// TestFunc_NilFnPanics pins the construction-time contract: a nil fn is a
// programming error surfaced at Func, not later as a tool-call panic.
func TestFunc_NilFnPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Func with nil fn did not panic")
		}
	}()
	Func[addArgs]("add", "adds two numbers", nil)
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

// TestFunc_RejectsSchemaViolations pins the strictness Func adds over a plain
// decode: Run validates the model's raw arguments against the schema derived
// from Args BEFORE invoking fn, so a contract-violating call comes back as a
// model-recoverable "ERROR: invalid arguments" result carrying the
// path-qualified validator message, and fn never sees the bad input.
func TestFunc_RejectsSchemaViolations(t *testing.T) {
	t.Run("unknown key", func(t *testing.T) {
		fc := newFakeClient(
			toolResp("c1", "add", `{"a":40,"b":2,"c":7}`, 10, 4),
			textResp("recovered", 8, 3),
		)
		invoked := false
		tool := Func("add", "adds two numbers", func(_ context.Context, p addArgs) (string, error) {
			invoked = true
			return fmt.Sprintf("%g", p.A+p.B), nil
		})
		r := NewRunner(fc, []Tool{tool}, "sys")
		if _, err := r.Run(context.Background(), "task"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if invoked {
			t.Error("fn ran despite an argument the derived schema forbids (additionalProperties:false)")
		}
		assertToolResult(t, fc, func(text string) bool {
			return strings.HasPrefix(text, "ERROR: invalid arguments:") && strings.Contains(text, `"c"`)
		}, `unexpected property "c"`)
	})

	t.Run("missing required", func(t *testing.T) {
		fc := newFakeClient(
			toolResp("c1", "add", `{"a":40}`, 10, 4),
			textResp("recovered", 8, 3),
		)
		invoked := false
		tool := Func("add", "adds two numbers", func(_ context.Context, p addArgs) (string, error) {
			invoked = true
			return fmt.Sprintf("%g", p.A+p.B), nil
		})
		r := NewRunner(fc, []Tool{tool}, "sys")
		if _, err := r.Run(context.Background(), "task"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if invoked {
			t.Error("fn ran despite a missing required field the derived schema declares")
		}
		assertToolResult(t, fc, func(text string) bool {
			return strings.HasPrefix(text, "ERROR: invalid arguments:") && strings.Contains(text, `"b"`)
		}, `missing required field "b"`)
	})
}

// assertToolResult fails when no tool-result message in the second completion
// satisfies match; label names the expected violation for the failure message.
func assertToolResult(t *testing.T, fc *fakeClient, match func(string) bool, label string) {
	t.Helper()
	if len(fc.requests) < 2 {
		t.Fatalf("completions = %d, want at least 2", len(fc.requests))
	}
	for _, m := range fc.requests[1].Messages {
		if m.Role == llmkit.RoleToolResult && match(m.Text()) {
			return
		}
	}
	t.Errorf("no tool result in completion 2 matches the expected violation (%s)", label)
}

// TestUnmarshalArgs_ToleratesUnknownKeys pins the tolerance hand-written Tools
// rely on: UnmarshalArgs stays a plain encoding/json decode, so an extra key
// in the model's arguments still decodes. The strictness for Func-declared
// tools lives in Func.Run's schema validation, not here.
func TestUnmarshalArgs_ToleratesUnknownKeys(t *testing.T) {
	var p addArgs
	if err := UnmarshalArgs(json.RawMessage(`{"a":40,"b":2,"c":7}`), &p); err != nil {
		t.Fatalf("UnmarshalArgs: %v", err)
	}
	if p.A != 40 || p.B != 2 {
		t.Errorf("decoded = %+v, want a=40 b=2 (unknown key tolerated)", p)
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

// BenchmarkFuncTool_Run measures funcTool.Run's steady-state cost (schema
// derived once by Func, then reused across every call). shp.10's fourth nit:
// funcTool's doc claimed the schema was "derived once at construction", but
// before caching a parsed [parsedSchema] node alongside the marshaled bytes,
// validateSchema re-unmarshaled the schema JSON on every single call — only
// the marshal (SchemaOf) was actually amortized. Run `go test -run ^$
// -bench BenchmarkFuncTool_Run -benchmem ./agent/` and compare allocs/op
// against a build that reverts funcTool.Run to call validateSchema(t.schema,
// args) (re-parsing every time) to see the schema-unmarshal allocations
// disappear from the steady-state path.
func BenchmarkFuncTool_Run(b *testing.B) {
	tool := Func[addArgs]("add", "adds two numbers", func(_ context.Context, a addArgs) (string, error) {
		return "", nil
	})
	args := json.RawMessage(`{"a":1,"b":2}`)
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := tool.Run(ctx, args); err != nil {
			b.Fatalf("Run: %v", err)
		}
	}
}

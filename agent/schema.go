package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/dpoage/llmkit"
	"github.com/invopop/jsonschema"
)

// schemaReflector configures how Go types become JSON Schema documents for
// [SchemaOf].
//
//   - DoNotReference inlines every nested struct instead of emitting $defs
//     with $ref pointers: the bundled validator is a deliberately closed
//     JSON-Schema subset that ignores $ref, so a referenced nested object's
//     constraints (required fields, additionalProperties) would be silently
//     unenforced.
//   - AllowAdditionalProperties=false stamps "additionalProperties": false
//     (the bool form) at every object level: provider/openai's
//     OpenAI-compatible schema downgrade requires the bool form, and the
//     closed-subset validator uses it to reject unknown keys.
//   - RequiredFromJSONSchemaTags stays false (the default), so every field
//     WITHOUT an `omitempty`/`omitzero` json tag is required — matching
//     encoding/json's own wire semantics. Descriptions and value constraints
//     ride on `jsonschema:"..."` struct tags.
var schemaReflector = jsonschema.Reflector{
	DoNotReference:            true,
	ExpandedStruct:            true,
	AllowAdditionalProperties: false,
}

// SchemaOf returns the JSON Schema for type T, derived from its exported
// fields and struct tags. It is the single source for tool parameter schemas
// ([Func]) and structured-output schemas ([Runner.RunJSONAs]): the Go type is
// the schema, so the code the model must satisfy and the code that decodes
// its answer cannot drift apart.
//
// The returned document carries no $ref/$defs (nested types are inlined),
// "additionalProperties": false at every object level, and every field
// without an `omitempty` json tag listed as required — the package's shared
// reflector settings encode the rationale. `json` tags name the properties;
// `jsonschema:"..."` tags add descriptions and constraints (e.g.
// `jsonschema:"minLength=1"`).
//
// Field-type mapping of note: a pointer field (*T) maps to T's schema
// verbatim — the pointer is NOT nullable ("null"-typed) — and the field is
// required unless tagged omitempty. An `any` field maps to the
// accept-everything schema `true` and is likewise required unless omitempty.
//
// SchemaOf panics when T is not a struct (or pointer to one), when T's type
// graph contains a struct cycle — recursive types cannot be expressed in a
// fully inlined schema, and the reflector would otherwise die with a fatal,
// unrecoverable stack overflow — or when the shape cannot be marshaled into
// a schema. All are programming errors that any test or first run surfaces
// immediately, not runtime conditions a caller could recover from.
func SchemaOf[T any]() json.RawMessage {
	t := reflect.TypeFor[T]()
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("agent: SchemaOf requires a struct type, got %s (%s)", t, t.Kind()))
	}
	assertAcyclic(t, t, map[reflect.Type]bool{})
	s := schemaReflector.Reflect(reflect.New(t).Interface())
	// The schema is consumed by validateSchema (a closed subset) and embedded
	// verbatim in prompts; the draft header and the reflector's auto-generated
	// package-path $id are noise for both.
	s.Version = ""
	s.ID = ""
	b, err := json.Marshal(s)
	if err != nil {
		panic(fmt.Sprintf("agent: SchemaOf[%s]: %v", t, err))
	}
	return b
}

// assertAcyclic panics when the type graph reachable from t contains a
// cycle. SchemaOf inlines nested types ([schemaReflector] DoNotReference),
// and an inlined schema cannot express recursion: feeding a recursive type
// to the reflector ends in a FATAL, unrecoverable stack overflow, so the
// cycle is rejected here first with a message naming the SchemaOf root type
// and the cyclic type.
//
// EVERY composite kind on the DFS path is recorded — struct, pointer,
// slice, array, and map alike, keyed by reflect.Type — because recursion
// need not pass through a struct: `type Tree []Tree` used as a field would
// otherwise loop inside this walker itself. Identity-keyed visited-ON-PATH
// (not visited-ever) keeps diamond shapes (the same type under two fields)
// false-positive-free. Interfaces and every non-composite kind terminate.
func assertAcyclic(root, t reflect.Type, onPath map[reflect.Type]bool) {
	switch t.Kind() {
	case reflect.Pointer, reflect.Struct, reflect.Slice, reflect.Array, reflect.Map:
	default:
		return
	}
	if onPath[t] {
		panic(fmt.Sprintf("agent: SchemaOf[%s]: recursive type %s is unsupported — an inlined schema cannot express cycles", root, t))
	}
	onPath[t] = true
	defer delete(onPath, t)

	switch t.Kind() {
	case reflect.Pointer:
		assertAcyclic(root, t.Elem(), onPath)
	case reflect.Struct:
		for i := range t.NumField() {
			if f := t.Field(i); schemaFieldVisited(f) {
				assertAcyclic(root, f.Type, onPath)
			}
		}
	case reflect.Slice, reflect.Array:
		assertAcyclic(root, t.Elem(), onPath)
	case reflect.Map:
		// Values only: invopop's reflectMap (reflect.go:434-443) inspects
		// t.Key().Kind() but never reflects the key type, so a cycle behind
		// a map key is invisible to the reflector and must be accepted here.
		assertAcyclic(root, t.Elem(), onPath)
	}
}

// schemaFieldVisited mirrors invopop/jsonschema v0.14.0's reflectFieldName +
// reflectStructFields: it reports whether the reflector would recurse into
// field f's type, so the cycle walk diverges in NEITHER direction — skipping
// a field invopop reflects would let a cycle reach the reflector (fatal),
// while recursing into a field invopop drops would reject schemas invopop
// accepts. The rules, in reflectFieldName's order:
//
//   - `json:"-"` and `jsonschema:"-"` drop the field entirely
//     (ignoredByJSONTags / ignoredByJSONSchemaTags) — skip;
//   - an anonymous struct or pointer-to-struct with no json name is
//     embedded and recursed into REGARDLESS of exportedness (shouldEmbed);
//   - a `json:"...,inline"` field is embedded, but reflectStructFields only
//     proceeds for struct (or pointer-to-struct) types — other shapes are
//     dropped;
//   - a non-anonymous unexported field gets no schema name — skip;
//   - everything else recurses fully, including an anonymous non-struct
//     type, which invopop reflects as a property named after the type.
//
// "Pointer-to-struct" means exactly ONE pointer level unwrapped — invopop
// inspects f.Type or f.Type.Elem() only, so **S is dropped, not embedded.
func schemaFieldVisited(f reflect.StructField) bool {
	jsonTags := strings.Split(f.Tag.Get("json"), ",")
	if jsonTags[0] == "-" {
		return false
	}
	if strings.Split(f.Tag.Get("jsonschema"), ",")[0] == "-" {
		return false
	}
	ft := f.Type
	if ft.Kind() == reflect.Pointer {
		ft = ft.Elem()
	}
	if f.Anonymous && jsonTags[0] == "" && ft.Kind() == reflect.Struct {
		return true
	}
	for _, tag := range jsonTags[1:] {
		if tag == "inline" {
			return ft.Kind() == reflect.Struct
		}
	}
	return f.Anonymous || f.PkgPath == ""
}

// Func builds a [Tool] from a plain Go function: the advertised parameter
// schema is derived from Args via [SchemaOf], and the model's raw JSON
// arguments are validated against that schema before fn runs — a violation
// (unknown key under additionalProperties:false, missing required field, type
// mismatch) surfaces as a model-recoverable "ERROR: invalid arguments"
// result naming the offending path, and fn is not invoked. Valid arguments
// are then decoded into Args with [UnmarshalArgs]. It
// replaces the two-method Tool implementation (Def + Run) wherever the schema
// is exactly the shape of a struct the function already wants.
//
// Two constraints follow from validating against the SchemaOf-derived
// schema rather than decoding directly: an `omitempty` field must be
// omitted by the model (an explicit JSON null fails validation), and a type
// whose custom UnmarshalJSON accepts a wire shape unlike its Go kind (an
// int64-backed duration decoded from "5s") is refused, because SchemaOf
// reflects the Go kind. The fix in both cases is on the type: implement
// invopop's `JSONSchema() *jsonschema.Schema` so SchemaOf reflects the true
// wire shape.
//
// An error from fn is model-recoverable: the harness feeds it back as an
// "ERROR:"-prefixed tool result and lets the model retry (see [Tool.Run]).
// Wrap genuine harness/infra failures in [ToolHealthError] so they also
// reach [Hooks.ToolHealth].
//
// Args may be struct{} for tools that take no arguments.
func Func[Args any](name, description string, fn func(context.Context, Args) (string, error)) Tool {
	if fn == nil {
		panic(fmt.Sprintf("agent: Func(%q): nil function", name))
	}
	schema := SchemaOf[Args]()
	parsed, err := parseSchema(schema)
	if err != nil {
		// SchemaOf's output is always valid JSON (it is SchemaOf's own
		// json.Marshal result) — a parse failure here is a bug in SchemaOf,
		// not a caller mistake, so it panics rather than surfacing as a
		// runtime error every Run call would have to check for.
		panic(fmt.Sprintf("agent: Func(%q): derived schema failed to parse: %v", name, err))
	}
	return funcTool[Args]{name: name, description: description, fn: fn, schema: schema, parsed: parsed}
}

// funcTool adapts a Func-declared function to the [Tool] interface. schema's
// bytes AND its parsed [parsedSchema] node are both derived once at
// construction: Def shares the marshaled bytes, Run's validation shares the
// parsed node (see [validateParsedSchema]) so it does not re-parse the same
// schema on every call — only the per-call args body is parsed.
type funcTool[Args any] struct {
	name        string
	description string
	fn          func(context.Context, Args) (string, error)
	schema      json.RawMessage
	parsed      parsedSchema
}

func (t funcTool[Args]) Def() llmkit.ToolDef {
	return llmkit.ToolDef{
		Name:        t.name,
		Description: t.description,
		Parameters:  t.schema,
	}
}

func (t funcTool[Args]) Run(ctx context.Context, args json.RawMessage) (string, error) {
	if err := validateParsedSchema(t.parsed, args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	var a Args
	if err := UnmarshalArgs(args, &a); err != nil {
		return "", err
	}
	return t.fn(ctx, a)
}

package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/dpoage/llmkit"
)

// schemaFidelityRow is one fidelity row: a caller-supplied schema (or one of the
// nil/null/{}/{"type":"object"} Parameters forms) and the object the wire
// input_schema must deep-equal once "type" is forced to the SDK's "object"
// constant.
type schemaFidelityRow struct {
	name   string
	params json.RawMessage // Parameters/ResponseSchema on the wire request
	want   map[string]any  // expected wire input_schema, end-anchored
}

func schemaFidelityRows() []schemaFidelityRow {
	return []schemaFidelityRow{
		{
			name:   "defs_additionalProperties_description_properties_required",
			params: json.RawMessage(`{"$defs":{"X":{"type":"string"}},"additionalProperties":false,"description":"d","properties":{"a":{"type":"string"}},"required":["a"]}`),
			want: map[string]any{
				"$defs":                map[string]any{"X": map[string]any{"type": "string"}},
				"additionalProperties": false,
				"description":          "d",
				"properties":           map[string]any{"a": map[string]any{"type": "string"}},
				"required":             []any{"a"},
				"type":                 "object",
			},
		},
		{
			name:   "explicit_type_object",
			params: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`),
			want: map[string]any{
				"properties": map[string]any{"a": map[string]any{"type": "string"}},
				"type":       "object",
			},
		},
		{
			name:   "type_array_wire_says_object",
			params: json.RawMessage(`{"type":"array","items":{"type":"string"}}`),
			want: map[string]any{
				"items": map[string]any{"type": "string"},
				"type":  "object",
			},
		},
		{
			name:   "bare_properties_object_no_unwrap",
			params: json.RawMessage(`{"q":{"type":"string"}}`),
			want: map[string]any{
				"q":    map[string]any{"type": "string"},
				"type": "object",
			},
		},
		{name: "nil_parameters", params: nil, want: map[string]any{"type": "object"}},
		{name: "null_parameters", params: json.RawMessage(`null`), want: map[string]any{"type": "object"}},
		{name: "empty_object_parameters", params: json.RawMessage(`{}`), want: map[string]any{"type": "object"}},
		{name: "explicit_type_object_only", params: json.RawMessage(`{"type":"object"}`), want: map[string]any{"type": "object"}},
		// The next three rows record the observed wire shape, which is not
		// the caller's object: "properties" and "required" lift into the
		// SDK's omitzero typed fields, so an empty "required" and a null
		// "properties" or "required" are omitted from the wire.
		{
			name:   "empty_properties_empty_required",
			params: json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
			want:   map[string]any{"properties": map[string]any{}, "type": "object"},
		},
		{name: "null_properties", params: json.RawMessage(`{"properties":null}`), want: map[string]any{"type": "object"}},
		{
			name:   "null_required",
			params: json.RawMessage(`{"properties":{"a":{"type":"string"}},"required":null}`),
			want: map[string]any{
				"properties": map[string]any{"a": map[string]any{"type": "string"}},
				"type":       "object",
			},
		},
	}
}

// captureToolInputSchema builds a request with one user tool carrying
// params as its Parameters, runs it against the adapter, and returns the
// captured wire input_schema for that tool.
func captureToolInputSchema(t *testing.T, params json.RawMessage) map[string]any {
	t.Helper()
	var captured map[string]any
	base := captureAnthropicBody(t, &captured, `{"id":"msg","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	adapter, err := New("claude-test", Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.Tools = []llmkit.ToolDef{{Name: "t", Parameters: params}}
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tools, ok := captured["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools missing or wrong shape: %#v", captured["tools"])
	}
	tool, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("tool is not an object: %#v", tools[0])
	}
	schema, ok := tool["input_schema"].(map[string]any)
	if !ok {
		t.Fatalf("input_schema missing or not an object: %#v", tool["input_schema"])
	}
	return schema
}

// captureSyntheticInputSchema builds a request whose ResponseSchema is
// params (StructuredOutput forced on, no user tools), runs it, and returns
// the captured wire input_schema for the injected synthetic tool.
func captureSyntheticInputSchema(t *testing.T, params json.RawMessage) map[string]any {
	t.Helper()
	var captured map[string]any
	base := captureAnthropicBody(t, &captured, `{"id":"msg","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	adapter, err := New("claude-test", Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := simpleRequest()
	req.ResponseSchema = params
	if _, err := adapter.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tools, ok := captured["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools missing or wrong shape (want exactly the synthetic tool): %#v", captured["tools"])
	}
	tool, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("tool is not an object: %#v", tools[0])
	}
	if tool["name"] != "emit_answer" {
		t.Fatalf("synthetic tool name = %v, want emit_answer", tool["name"])
	}
	schema, ok := tool["input_schema"].(map[string]any)
	if !ok {
		t.Fatalf("input_schema missing or not an object: %#v", tool["input_schema"])
	}
	return schema
}

// TestSchemaFidelity_UserTool pins fidelity for a user tool's Parameters:
// the wire input_schema, parsed as JSON, deep-equals the caller's schema
// object with "type" forced to "object" — end-anchored equality of the
// whole object, not a presence check.
func TestSchemaFidelity_UserTool(t *testing.T) {
	for _, row := range schemaFidelityRows() {
		t.Run(row.name, func(t *testing.T) {
			got := captureToolInputSchema(t, row.params)
			if !reflect.DeepEqual(got, row.want) {
				t.Errorf("wire input_schema =\n  %#v\nwant\n  %#v", got, row.want)
			}
		})
	}
}

// TestSchemaFidelity_SyntheticStructuredOutputTool pins fidelity for the
// synthetic structured-output tool: the same fidelity contract holds for
// Request.ResponseSchema as for a user tool's Parameters. nil/null are
// skipped here — for ResponseSchema those mean "no schema requested"
// (adapter.Prepare's ResponseSchema is nil, so no synthetic tool is
// injected at all), unlike a tool's Parameters where nil/null still name a
// parameter-less tool that must be declared.
func TestSchemaFidelity_SyntheticStructuredOutputTool(t *testing.T) {
	for _, row := range schemaFidelityRows() {
		if row.name == "nil_parameters" || row.name == "null_parameters" {
			continue
		}
		t.Run(row.name, func(t *testing.T) {
			got := captureSyntheticInputSchema(t, row.params)
			if !reflect.DeepEqual(got, row.want) {
				t.Errorf("wire input_schema =\n  %#v\nwant\n  %#v", got, row.want)
			}
		})
	}
}

// TestSchemaFidelity_EmptyRootKeyRefused pins the one root key the SDK
// cannot encode: a schema carrying the empty-string key "" is refused
// before the wire with ErrInvalidRequest (non-retryable), for a user
// tool's Parameters and for the synthetic structured-output tool, on
// Complete and Stream. Without the refusal the SDK marshal fails inside
// the HTTP call and the error classifies as a retryable ErrServer.
func TestSchemaFidelity_EmptyRootKeyRefused(t *testing.T) {
	const schema = `{"":1,"properties":{"a":{"type":"string"}}}`
	cases := []struct {
		name    string
		mutate  func(*llmkit.Request)
		message string
	}{
		{"user_tool", func(r *llmkit.Request) {
			r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(schema)}}
		}, `tool t: Parameters: the empty-string root key "" cannot be sent: the Anthropic SDK cannot encode it`},
		{"synthetic_tool", func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(schema)
		}, `ResponseSchema: the empty-string root key "" cannot be sent: the Anthropic SDK cannot encode it`},
	}
	modes := []struct {
		name string
		call func(*anthropicAdapter, llmkit.Request) error
	}{
		{"complete", func(a *anthropicAdapter, req llmkit.Request) error {
			_, err := a.Complete(context.Background(), req)
			return err
		}},
		{"stream", func(a *anthropicAdapter, req llmkit.Request) error {
			_, err := a.Stream(context.Background(), req, func(llmkit.Delta) error { return nil })
			return err
		}},
	}
	for _, tc := range cases {
		for _, mode := range modes {
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				hits := 0
				base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
					hits++
					w.WriteHeader(http.StatusInternalServerError)
				})
				client, err := New("claude-test", Options{APIKey: "k", BaseURL: base})
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				req := simpleRequest()
				tc.mutate(&req)
				err = mode.call(client.(*anthropicAdapter), req)
				if hits != 0 {
					t.Errorf("wire hits = %d, want 0 (refused pre-wire)", hits)
				}
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Fatalf("err = %v, want ErrInvalidRequest", err)
				}
				var apiErr *llmkit.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
				}
				if apiErr.Provider != "anthropic" || apiErr.StatusCode != 0 || apiErr.Message != tc.message {
					t.Errorf("APIError = {Provider:%q StatusCode:%d Message:%q}, want {anthropic 0 %q}",
						apiErr.Provider, apiErr.StatusCode, apiErr.Message, tc.message)
				}
			})
		}
	}
}

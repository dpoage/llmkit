package provider

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/dpoage/llmkit"
)

// forceStructuredOutput returns a Spec.Capabilities-style override that sets
// exactly the StructuredOutput field, leaving every other field at the
// adapter's table value.
func forceStructuredOutput(v bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities { c.StructuredOutput = v; return c }
}

func forceThinking(v bool) func(llmkit.Capabilities) llmkit.Capabilities {
	return func(c llmkit.Capabilities) llmkit.Capabilities { c.Thinking = v; return c }
}

// refusalRow is one refusal condition. mutate applies the condition to an
// otherwise-minimal valid request; caps optionally overrides the effective
// profile (nil = table default); only restricts which adapters the row
// applies to (nil = every adapter — the openai ceiling forbids Thinking,
// so the thinking row names anthropic and google only); wantRefused false
// is a "defined-away" row (null/zero-length Parameters or ResponseSchema)
// that must NOT refuse; message, when set, is the exact refusal text.
type refusalRow struct {
	name        string
	mutate      func(*llmkit.Request)
	caps        func(llmkit.Capabilities) llmkit.Capabilities
	only        []string
	wantRefused bool
	message     string
}

// schemaPaddingRows are the not-supplied boundary for tool Parameters and
// ResponseSchema: only zero-length input or the literal null with JSON
// whitespace (space, tab, LF, CR) around it counts as not supplied.
// Whitespace alone and padding JSON does not allow (U+00A0, U+0085, \v,
// \f) are malformed JSON and refused, whatever they wrap.
func schemaPaddingRows() []refusalRow {
	refused := []struct{ name, raw string }{
		{"whitespace_only", "  \t\n\r "},
		{"nbsp_object", "\u00a0{}"},
		{"nbsp_null", "\u00a0null"},
		{"vtab_null", "\vnull"},
		{"formfeed_null", "\fnull"},
		{"nel_object", "\u0085{}"},
	}
	accepted := []struct{ name, raw string }{
		{"json_whitespace_null", "  null  "},
		{"json_whitespace_object", "\n{}\n"},
	}
	var rows []refusalRow
	for _, in := range refused {
		raw := json.RawMessage(in.raw)
		rows = append(rows,
			refusalRow{name: "tool_parameters_" + in.name, wantRefused: true, message: "tool t: Parameters: malformed JSON",
				mutate: func(r *llmkit.Request) { r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: raw}} }},
			refusalRow{name: "response_schema_" + in.name, wantRefused: true, message: "ResponseSchema: malformed JSON",
				caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) { r.ResponseSchema = raw }},
		)
	}
	for _, in := range accepted {
		raw := json.RawMessage(in.raw)
		rows = append(rows,
			refusalRow{name: "tool_parameters_" + in.name + "_not_refused",
				mutate: func(r *llmkit.Request) { r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: raw}} }},
			refusalRow{name: "response_schema_" + in.name + "_not_refused",
				caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) { r.ResponseSchema = raw }},
		)
	}
	return rows
}

func refusalRows() []refusalRow {
	return append([]refusalRow{
		{name: "unknown_tool_choice_mode", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.ToolChoice = llmkit.ToolChoice{Mode: "bogus-mode"}
		}},
		{name: "tool_mode_empty_name", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceTool}
		}},
		{name: "tool_parameters_array", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(`[1,2]`)}}
		}},
		{name: "tool_parameters_string", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(`"nope"`)}}
		}},
		{name: "tool_parameters_malformed", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(`{`)}}
		}},
		{name: "tool_parameters_null_not_refused", wantRefused: false, mutate: func(r *llmkit.Request) {
			r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(`null`)}}
		}},
		{name: "tool_parameters_zero_length_not_refused", wantRefused: false, mutate: func(r *llmkit.Request) {
			r.Tools = []llmkit.ToolDef{{Name: "t", Parameters: json.RawMessage(``)}}
		}},
		{name: "response_schema_array_no_tools", wantRefused: true, caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(`[1,2]`)
		}},
		{name: "response_schema_malformed_no_tools", wantRefused: true, caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(`{`)
		}},
		{name: "response_schema_array_with_tools", wantRefused: true, caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(`[1,2]`)
			r.Tools = []llmkit.ToolDef{{Name: "t"}}
		}},
		{name: "response_schema_malformed_with_tools", wantRefused: true, caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(`{`)
			r.Tools = []llmkit.ToolDef{{Name: "t"}}
		}},
		{name: "response_schema_null_not_refused", wantRefused: false, caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(`null`)
		}},
		{name: "response_schema_zero_length_not_refused", wantRefused: false, caps: forceStructuredOutput(true), mutate: func(r *llmkit.Request) {
			r.ResponseSchema = json.RawMessage(``)
		}},
		{name: "thinking_budget_non_positive", wantRefused: true, caps: forceThinking(true), only: []string{"anthropic", "google"}, mutate: func(r *llmkit.Request) {
			r.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 0}
		}},
		{name: "unknown_role_no_blocks", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.Messages = []llmkit.Message{{Role: "bogus-role"}}
		}},
		{name: "block_kind_outside_role", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.Messages = []llmkit.Message{{Role: llmkit.RoleSystem, Content: []llmkit.Block{{Kind: llmkit.BlockImage}}}}
		}},
		{name: "image_data_no_mediatype", wantRefused: true, mutate: func(r *llmkit.Request) {
			r.Messages = []llmkit.Message{{Role: llmkit.RoleUser, Content: []llmkit.Block{{Kind: llmkit.BlockImage, Data: []byte("x")}}}}
		}},
		{name: "document_url_openai_label", wantRefused: true, only: []string{"openai", "openai-compatible"}, mutate: func(r *llmkit.Request) {
			r.Messages = []llmkit.Message{{Role: llmkit.RoleUser, Content: []llmkit.Block{
				{Kind: llmkit.BlockDocument, MediaType: "application/pdf", URL: "https://example.invalid/a.pdf"},
			}}}
		}},
	}, schemaPaddingRows()...)
}

func (r refusalRow) appliesTo(name string) bool {
	if r.only == nil {
		return true
	}
	for _, n := range r.only {
		if n == name {
			return true
		}
	}
	return false
}

// minimalP1Request is the base request every row mutates: one text user
// message, no tools, no schema, no explicit tool choice.
func minimalP1Request() llmkit.Request {
	return llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}}
}

// p1Modes runs a request through Complete or Stream. A not-refused row
// needs a wire reply in the mode's format; sse selects it.
var p1Modes = []struct {
	name string
	sse  bool
	call func(llmkit.Client, llmkit.Request) error
}{
	{"complete", false, func(c llmkit.Client, req llmkit.Request) error {
		_, err := c.Complete(context.Background(), req)
		return err
	}},
	{"stream", true, func(c llmkit.Client, req llmkit.Request) error {
		_, err := llmkit.Stream(context.Background(), c, req, func(llmkit.Delta) error { return nil })
		return err
	}},
}

// endTurnReason is each adapter's wire stop reason for a natural finish.
func endTurnReason(provider string) string {
	switch provider {
	case "anthropic":
		return "end_turn"
	case "google":
		return "STOP"
	default:
		return "stop"
	}
}

// TestPrepare_OneRuleRefusals exercises every condition the seam contract
// refuses (or, for the defined-away rows, explicitly does not refuse) on
// every applicable adapter over the allAdapters() harness, through
// Complete and Stream. A refused row makes zero wire hits, its error
// wraps ErrInvalidRequest, errors.As reaches a *llmkit.APIError whose
// Provider is the adapter's own label and StatusCode is 0, and the Message
// text is identical across every adapter and mode the row applies to (and
// equals the row's message when set). A not-refused row makes exactly one
// wire hit and returns no error.
func TestPrepare_OneRuleRefusals(t *testing.T) {
	for _, row := range refusalRows() {
		t.Run(row.name, func(t *testing.T) {
			messages := map[string]string{}
			for _, f := range allAdapters() {
				if !row.appliesTo(f.name) {
					continue
				}
				for _, mode := range p1Modes {
					t.Run(f.name+"/"+mode.name, func(t *testing.T) {
						hits := 0
						base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
							hits++
							if mode.sse {
								w.Header().Set("Content-Type", "text/event-stream")
								_, _ = w.Write([]byte(streamStopBody(f.name, endTurnReason(f.name), false)))
								return
							}
							w.Header().Set("Content-Type", "application/json")
							_, _ = w.Write([]byte(okBody(f.name)))
						})
						var client llmkit.Client
						if row.caps != nil {
							client = f.buildWithCaps(t, base, row.caps)
						} else {
							client = f.build(t, base)
						}
						req := minimalP1Request()
						row.mutate(&req)
						err := mode.call(client, req)

						if !row.wantRefused {
							if err != nil {
								t.Fatalf("%s: %v, want no error (defined-away row)", mode.name, err)
							}
							if hits != 1 {
								t.Errorf("wire hits = %d, want 1 (not refused)", hits)
							}
							return
						}

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
						if apiErr.Provider != f.name {
							t.Errorf("Provider = %q, want %q", apiErr.Provider, f.name)
						}
						if apiErr.StatusCode != 0 {
							t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
						}
						if row.message != "" && apiErr.Message != row.message {
							t.Errorf("Message = %q, want %q", apiErr.Message, row.message)
						}
						messages[f.name+"/"+mode.name] = apiErr.Message
					})
				}
			}
			if !row.wantRefused || len(messages) < 2 {
				return
			}
			var want string
			for _, msg := range messages {
				want = msg
				break
			}
			for name, msg := range messages {
				if msg != want {
					t.Errorf("Message on %s = %q, want %q (identical across every adapter the row applies to)", name, msg, want)
				}
			}
		})
	}
}

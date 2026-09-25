package adapter

import (
	"encoding/json"
	"fmt"

	"github.com/dpoage/llmkit"
)

// Prepared is a Request after the shared rules: validated, defaulted, and
// gated on the effective profile. Every adapter maps a Prepared value onto
// its SDK types WITHOUT re-checking any of the rules Prepare already
// applied — a rule that differs between two adapters after this type exists
// is a defect.
type Prepared struct {
	System   string
	Messages []llmkit.Message
	// MaxTokens is >= 1: llmkit.DefaultMaxTokens when the caller's
	// Request.MaxTokens was <= 0, the caller's value verbatim otherwise.
	MaxTokens   int
	Temperature *float64
	TopP        *float64 // nil when caps.TopP is false
	TopK        *int     // nil when caps.TopK is false
	Seed        *int64   // nil when caps.Seed is false
	// StopSequences is nil when caps.StopSequences is false or the caller
	// supplied none.
	StopSequences []string
	// Thinking is nil when caps.Thinking is false or the caller supplied
	// none; otherwise BudgetTokens is >= 1 (a non-positive budget while
	// caps.Thinking is true is refused, never silently dropped).
	Thinking *llmkit.ThinkingConfig
	// ToolChoice.Mode is one of the four constants ("" normalized to
	// auto); Name is non-empty when Mode is ToolChoiceTool. An explicit
	// mode on a ToolChoice=false profile never reaches here.
	ToolChoice llmkit.ToolChoice
	// Tools carries one entry per Request.Tools, in order. Schema is nil
	// when the tool declared no parameters (a zero-length or JSON null
	// Parameters); any other non-object JSON was refused before Prepare
	// returned.
	Tools []Tool
	// ResponseSchema is nil when caps.StructuredOutput is false or the
	// caller supplied no schema.
	ResponseSchema *Schema
}

// Tool is one prepared tool declaration: the caller's definition plus its
// decoded JSON-Schema root object (nil for a parameter-less tool). Schema is
// not interpreted further — a bare properties object goes out on the wire
// exactly as supplied; it is the adapter's job to lift the shape its SDK
// wants.
type Tool struct {
	Def    llmkit.ToolDef
	Schema map[string]any
}

// Schema is a prepared response schema: the caller's ResponseSchemaName
// verbatim ("" when unset — each adapter applies its own default: openai
// "response", anthropic "emit_answer", google sends no name) and the
// decoded JSON-Schema root object.
type Schema struct {
	Name  string
	Value map[string]any
}

// Prepare applies every request-side rule of llmkit.Capabilities once:
// message validation, MaxTokens defaulting, the sampler gates
// (TopP/TopK/Seed/StopSequences/Thinking dropped silently when the
// effective profile reports false), the ToolChoice gate (refused pre-wire),
// and tool/response-schema decoding. Every refusal is an error wrapping
// llmkit.ErrInvalidRequest with Provider = provider and StatusCode 0,
// returned before any wire call. A tool Parameters or ResponseSchema that
// is zero-length or the JSON literal null counts as "not supplied" (Schema
// nil / ResponseSchema nil); any other non-object JSON (array, string,
// number, malformed) is refused. A JSON object is accepted as-is — Prepare
// does not interpret its keys, so a bare properties object rides through
// unwrapped.
func Prepare(provider string, caps llmkit.Capabilities, req llmkit.Request) (Prepared, error) {
	if err := validateMessages(provider, req.Messages); err != nil {
		return Prepared{}, err
	}

	toolChoice, err := prepareToolChoice(provider, req.ToolChoice, caps)
	if err != nil {
		return Prepared{}, err
	}

	thinking, err := prepareThinking(provider, req.Thinking, caps)
	if err != nil {
		return Prepared{}, err
	}

	tools := make([]Tool, 0, len(req.Tools))
	for _, t := range req.Tools {
		schema, err := decodeSchemaObject(provider, "tool "+t.Name+": Parameters", t.Parameters)
		if err != nil {
			return Prepared{}, err
		}
		tools = append(tools, Tool{Def: t, Schema: schema})
	}

	var responseSchema *Schema
	if caps.StructuredOutput {
		obj, err := decodeSchemaObject(provider, "ResponseSchema", req.ResponseSchema)
		if err != nil {
			return Prepared{}, err
		}
		if obj != nil {
			responseSchema = &Schema{Name: req.ResponseSchemaName, Value: obj}
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = llmkit.DefaultMaxTokens
	}

	p := Prepared{
		System:         req.System,
		Messages:       req.Messages,
		MaxTokens:      maxTokens,
		Temperature:    req.Temperature,
		ToolChoice:     toolChoice,
		Tools:          tools,
		ResponseSchema: responseSchema,
		Thinking:       thinking,
	}
	if caps.TopP {
		p.TopP = req.TopP
	}
	if caps.TopK {
		p.TopK = req.TopK
	}
	if caps.Seed {
		p.Seed = req.Seed
	}
	if caps.StopSequences && len(req.StopSequences) > 0 {
		p.StopSequences = req.StopSequences
	}
	return p, nil
}

// validateMessages enforces the per-role block-kind rule, the media source
// rule, and message-role recognition for every message, in that order — a
// block kind outside the role's set (including every block on an unknown
// role) is reported before the role itself is checked, so an unknown role
// only reports as such on a message with NO blocks.
func validateMessages(provider string, msgs []llmkit.Message) error {
	for _, m := range msgs {
		if err := validateMessageBlocks(provider, m); err != nil {
			return err
		}
		switch m.Role {
		case llmkit.RoleSystem, llmkit.RoleUser, llmkit.RoleAssistant, llmkit.RoleToolResult:
		default:
			return Refuse(provider, "unknown message role "+string(m.Role), nil)
		}
	}
	return nil
}

// prepareToolChoice normalizes "" to auto, applies the ToolChoice gate
// (see gateToolChoice), and validates the mode itself: an unrecognized mode
// is refused, and ToolChoiceTool requires a non-empty Name.
func prepareToolChoice(provider string, tc llmkit.ToolChoice, caps llmkit.Capabilities) (llmkit.ToolChoice, error) {
	if tc.Mode == "" {
		tc.Mode = llmkit.ToolChoiceAuto
	}
	if err := gateToolChoice(provider, tc, caps.ToolChoice); err != nil {
		return tc, err
	}
	switch tc.Mode {
	case llmkit.ToolChoiceAuto, llmkit.ToolChoiceNone, llmkit.ToolChoiceRequired:
		return tc, nil
	case llmkit.ToolChoiceTool:
		if tc.Name == "" {
			return tc, Refuse(provider, "ToolChoice.Mode=tool requires ToolChoice.Name", nil)
		}
		return tc, nil
	default:
		return tc, Refuse(provider, "unknown ToolChoice.Mode "+string(tc.Mode), nil)
	}
}

// gateToolChoice enforces the Capabilities.ToolChoice contract at the
// request boundary: when the effective profile reports the model cannot
// steer tool calls, every explicit ToolChoice mode is rejected BEFORE any
// wire call. Dropping Mode=none would escalate permissions (the model stays
// free to call offered tools) and dropping required/tool would silently
// degrade; auto is the provider default and always fine. An unrecognized
// mode falls through so prepareToolChoice's own switch produces its precise
// unknown-mode error.
func gateToolChoice(provider string, tc llmkit.ToolChoice, supported bool) error {
	if supported {
		return nil
	}
	switch tc.Mode {
	case llmkit.ToolChoiceNone, llmkit.ToolChoiceRequired, llmkit.ToolChoiceTool:
		return Refuse(provider, "ToolChoice.Mode "+string(tc.Mode)+" requires a model with tool-choice support "+
			"(Capabilities.ToolChoice=false)", nil)
	}
	return nil
}

// prepareThinking gates Request.Thinking on caps.Thinking: false drops the
// config silently (nil, no error — the documented contract for a false
// feature), true with a nil config is a no-op, and true with a non-positive
// BudgetTokens is refused before any wire call.
func prepareThinking(provider string, tc *llmkit.ThinkingConfig, caps llmkit.Capabilities) (*llmkit.ThinkingConfig, error) {
	if tc == nil || !caps.Thinking {
		return nil, nil
	}
	if tc.BudgetTokens <= 0 {
		return nil, Refuse(provider, "Thinking.BudgetTokens must be positive", nil)
	}
	return tc, nil
}

// decodeSchemaObject decodes a tool's Parameters or a Request's
// ResponseSchema: a zero-length input, or the JSON literal null with
// optional JSON whitespace (space, tab, LF, CR) around it, counts as "not
// supplied" (nil, nil); any other input that is not a JSON object — an
// array, a string, a number, whitespace alone, or malformed JSON, including
// padding JSON does not allow such as U+00A0 or \v — is refused, naming
// label; a JSON object decodes and returns as-is.
func decodeSchemaObject(provider, label string, raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, Refuse(provider, label+": malformed JSON", err)
	}
	if v == nil {
		return nil, nil
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, Refuse(provider, label+" must be a JSON object", nil)
	}
	return obj, nil
}

// Refuse builds a pre-wire refusal error: an error wrapping
// llmkit.ErrInvalidRequest, tagged with provider, StatusCode 0. cause may
// be nil. Prepare and the adapters' vendor-specific rules return it; the
// block and media-source rules in media.go build the same error shape
// directly.
func Refuse(provider, msg string, cause error) error {
	return &llmkit.APIError{
		Kind:     llmkit.ErrInvalidRequest,
		Provider: provider,
		Message:  msg,
		Err:      cause,
	}
}

// StopFallback is the mapping for a stop reason an adapter does not
// recognize: StopToolUse when the response carries tool calls (trust their
// presence over an absent or nonstandard reason), StopError otherwise (a
// provider-specific reason mapping to none of the adapter's recognized
// table is never assumed to be a natural completion). Which reasons an
// adapter recognizes — and therefore never reaches StopFallback — is that
// adapter's own table.
func StopFallback(hasToolCalls bool) llmkit.StopReason {
	if hasToolCalls {
		return llmkit.StopToolUse
	}
	return llmkit.StopError
}

// wireGate names one llmkit.Capabilities field ApplyOverride ceiling-checks
// and reads its ceiling/effective values via reflection-free field access.
type wireGate struct {
	name      string
	ceilingOn bool
	effOn     bool
}

// ApplyOverride resolves the effective profile: override receives the
// table-derived profile and its return value becomes effective (nil
// override keeps the table unchanged). ceiling names every wire-gated
// field (StructuredOutput, Thinking, ToolChoice, StopSequences, TopP, TopK,
// Seed) the adapter can put on the wire; an effective profile that reports
// true for a wire-gated field whose ceiling is false is refused. Advisory
// and decorator fields (ContextWindow, ParallelToolCalls, PromptCaching,
// Images, Documents) are never ceiling-checked.
func ApplyOverride(provider string, ceiling, table llmkit.Capabilities, override func(llmkit.Capabilities) llmkit.Capabilities) (llmkit.Capabilities, error) {
	effective := table
	if override != nil {
		effective = override(table)
	}
	gates := []wireGate{
		{"StructuredOutput", ceiling.StructuredOutput, effective.StructuredOutput},
		{"Thinking", ceiling.Thinking, effective.Thinking},
		{"ToolChoice", ceiling.ToolChoice, effective.ToolChoice},
		{"StopSequences", ceiling.StopSequences, effective.StopSequences},
		{"TopP", ceiling.TopP, effective.TopP},
		{"TopK", ceiling.TopK, effective.TopK},
		{"Seed", ceiling.Seed, effective.Seed},
	}
	for _, g := range gates {
		if g.effOn && !g.ceilingOn {
			return llmkit.Capabilities{}, Refuse(provider,
				fmt.Sprintf("Capabilities.%s=true is above the ceiling for provider %s: the adapter has no wire field for it", g.name, provider),
				nil)
		}
	}
	return effective, nil
}

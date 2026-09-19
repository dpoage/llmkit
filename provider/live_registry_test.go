package provider_test

// The live case REGISTRY: the descriptor half of the descriptor/body split
// that keeps the hermetic coverage test compiling under plain `go test ./...`.
//
// liveCases describes every acceptance case the live matrix runs, keyed to an
// llmkit.Capabilities field (Gate) or unconditional. The case BODIES live in
// the `live`-tagged live_test.go (liveCaseBodies); the tagged runner enforces
// descriptor↔body parity in both directions. This file — and the reflection
// test below — have no build tag, so the hermetic suite fails whenever a
// capability is added or a case is deleted without a paired update.

import (
	"reflect"
	"testing"

	llmkit "github.com/dpoage/llmkit"
)

// liveCase is one acceptance case in the live vendor matrix. Gate is an
// llmkit.Capabilities field name the case is gated on, or "" for an
// unconditional case that every lane with credentials runs.
type liveCase struct {
	Name string
	Gate string
	Doc  string
}

var liveCases = []liveCase{
	// Unconditional cases: every lane with credentials runs them.
	{Name: "text_usage", Doc: "a plain completion returns text, accounted usage, and StopEndTurn"},
	{Name: "tool_round_trip", Doc: "a tool request is parsed and the tool result round-trips to a final answer"},
	{Name: "max_tokens", Doc: "a tiny MaxTokens yields StopMaxTokens"},
	{Name: "error_bad_key", Doc: "an invalid credential normalizes to ErrAuth"},
	{Name: "error_bad_model", Doc: "an unknown model normalizes to ErrInvalidRequest"},
	{Name: "usage_recorded", Doc: "WithRecorder emits one UsageEvent for a successful completion"},

	// Capability-keyed cases: one per llmkit.Capabilities field.
	{Name: "context_window", Gate: "ContextWindow", Doc: "an oversize prompt yields ErrContextTooLong when the window is known"},
	{Name: "parallel_tool_calls", Gate: "ParallelToolCalls", Doc: "two independent tools may be requested in one response"},
	{Name: "prompt_caching", Gate: "PromptCaching", Doc: "a repeated prefix yields CacheReadInputTokens > 0 on the second call"},
	{Name: "structured_output", Gate: "StructuredOutput", Doc: "ResponseSchema constrains the output to parseable JSON"},
	{Name: "thinking", Gate: "Thinking", Doc: "Thinking returns BlockThinking and the block re-sends without ErrInvalidRequest"},
	{Name: "tool_choice", Gate: "ToolChoice", Doc: "none forbids calls, required forces one, named forces that name"},
	{Name: "images", Gate: "Images", Doc: "an inline data-URI image block is accepted"},
	{Name: "documents", Gate: "Documents", Doc: "an inline PDF block is accepted"},
	{Name: "stop_sequences", Gate: "StopSequences", Doc: "generation stops before the guarded sequence with StopEndTurn"},
	{Name: "top_p", Gate: "TopP", Doc: "a request with TopP set is accepted"},
	{Name: "seed", Gate: "Seed", Doc: "a request with Seed set is accepted"},
	{Name: "top_k", Gate: "TopK", Doc: "a request with TopK set is accepted"},
}

// TestLiveRegistryCoverage is the hermetic enforcement test: every
// llmkit.Capabilities field must have a registered live case, every gate must
// name a real field, and names must be unique. Fails under plain
// `go test ./...` — no tag, no credentials.
func TestLiveRegistryCoverage(t *testing.T) {
	capsType := reflect.TypeOf(llmkit.Capabilities{})

	for i := 0; i < capsType.NumField(); i++ {
		field := capsType.Field(i).Name
		found := false
		for _, c := range liveCases {
			if c.Gate == field {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("capability %s has no live case: add a liveCase with Gate %q — a capability cannot ship without its acceptance test", field, field)
		}
	}

	for _, c := range liveCases {
		if c.Gate == "" {
			continue
		}
		if _, ok := capsType.FieldByName(c.Gate); !ok {
			t.Errorf("live case %q gates on %q, which is not an llmkit.Capabilities field", c.Name, c.Gate)
		}
	}
	for _, c := range liveCases {
		if c.Doc == "" {
			t.Errorf("live case %q has no Doc", c.Name)
		}
	}

	seen := make(map[string]bool, len(liveCases))
	for _, c := range liveCases {
		if seen[c.Name] {
			t.Errorf("duplicate live case name %q", c.Name)
		}
		seen[c.Name] = true
	}
}

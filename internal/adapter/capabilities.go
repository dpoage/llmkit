package adapter

import (
	"github.com/dpoage/llmkit"
)

// MatchesModelFamily reports whether model names the generation identified
// by key: the key equals the model ID, or the ID continues with the "-"
// segment separator immediately after the key. "gpt-4" matches "gpt-4" and
// "gpt-4-0613"; it does NOT match "gpt-4.5-preview" or "gpt-4omni" (mid-token
// extensions are different model names, not later snapshots of the same
// generation). The per-adapter capability tables match their entries through
// this one implementation so a family boundary cannot drift per provider.
func MatchesModelFamily(model, key string) bool {
	if len(model) < len(key) || model[:len(key)] != key {
		return false
	}
	return len(model) == len(key) || model[len(key)] == '-'
}

// BestMatchingFamily returns the index of the table entry whose key is the
// longest family match for model, or -1 when no entry matches. key extracts
// the family key from an entry.
func BestMatchingFamily[T any](model string, table []T, key func(T) string) int {
	best := -1
	for i, e := range table {
		k := key(e)
		if MatchesModelFamily(model, k) && (best < 0 || len(k) > len(key(table[best]))) {
			best = i
		}
	}
	return best
}

// GateToolChoice enforces the Capabilities.ToolChoice contract at the
// adapter boundary: when the effective profile reports the model cannot
// steer tool calls, every explicit ToolChoice mode is rejected with an error
// wrapping ErrInvalidRequest BEFORE any wire call. Dropping Mode=none would
// escalate permissions (the model stays free to call offered tools) and
// dropping required/tool would silently degrade; auto and the zero value are
// the provider default and always fine. An unrecognized mode falls through so
// the adapter's own mapper produces its precise unknown-mode error.
func GateToolChoice(provider string, tc llmkit.ToolChoice, supported bool) error {
	if supported {
		return nil
	}
	switch tc.Mode {
	case "", llmkit.ToolChoiceAuto:
		return nil
	case llmkit.ToolChoiceNone, llmkit.ToolChoiceRequired, llmkit.ToolChoiceTool:
		return llmkit.NewAPIError(provider, 0, 0, llmkit.ErrInvalidRequest,
			"ToolChoice.Mode "+string(tc.Mode)+" requires a model with tool-choice support "+
				"(Capabilities.ToolChoice=false)", nil)
	}
	return nil
}

package adapter

// matchesModelFamily reports whether model names the generation identified
// by key: the key equals the model ID, or the ID continues with the "-"
// segment separator immediately after the key. "gpt-4" matches "gpt-4" and
// "gpt-4-0613"; it does NOT match "gpt-4.5-preview" or "gpt-4omni" (mid-token
// extensions are different model names, not later snapshots of the same
// generation).
func matchesModelFamily(model, key string) bool {
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
		if matchesModelFamily(model, k) && (best < 0 || len(k) > len(key(table[best]))) {
			best = i
		}
	}
	return best
}

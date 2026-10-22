package decide

import (
	"sort"

	"github.com/dpoage/llmkit"
)

// DecisionAnswers converts resp's per-kind answer maps to the root
// [llmkit.DecisionAnswer] vocabulary, sorted by ID.
func DecisionAnswers(resp Response) []llmkit.DecisionAnswer {
	ids := make(map[string]struct{}, len(resp.Nouls)+len(resp.Choices)+len(resp.Scores))
	for id := range resp.Nouls {
		ids[id] = struct{}{}
	}
	for id := range resp.Choices {
		ids[id] = struct{}{}
	}
	for id := range resp.Scores {
		ids[id] = struct{}{}
	}
	sorted := make([]string, 0, len(ids))
	for id := range ids {
		sorted = append(sorted, id)
	}
	sort.Strings(sorted)

	out := make([]llmkit.DecisionAnswer, 0, len(sorted))
	for _, id := range sorted {
		da := llmkit.DecisionAnswer{ID: id}
		if belief, ok := resp.Nouls[id]; ok {
			b := belief
			da.Belief = &b
		}
		if choice, ok := resp.Choices[id]; ok {
			da.Choice = choice.Choice
			da.Probabilities = choice.Probabilities
			da.Confidence = choice.Confidence
		}
		if score, ok := resp.Scores[id]; ok {
			s := score.Score
			da.Score = &s
			da.Levels = score.Levels
			da.LevelProbabilities = score.Probabilities
			da.Confidence = score.Confidence
		}
		out = append(out, da)
	}
	return out
}

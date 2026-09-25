package decide

import (
	"encoding/json"
	"sort"

	"github.com/dpoage/llmkit"
)

// toDecisionQuestions converts qs to the root [llmkit.DecisionQuestion]
// vocabulary, sorted by ID. Field marshaling cannot fail on this path
// because buildRequest already validated Instructions/True/False/Options/
// Levels marshal cleanly for the same Questions value; marshalRaw
// degrades to nil so a telemetry conversion never fails an Ask that
// already succeeded or failed on its own.
func toDecisionQuestions(qs Questions) []llmkit.DecisionQuestion {
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]llmkit.DecisionQuestion, 0, len(qs))
	for _, id := range ids {
		dq := llmkit.DecisionQuestion{ID: id}
		switch q := qs[id].(type) {
		case Noul:
			dq.Kind = "noul"
			dq.Instructions = marshalRaw(q.Instructions)
			if q.True != nil {
				dq.True = marshalRaw(q.True)
			}
			if q.False != nil {
				dq.False = marshalRaw(q.False)
			}
		case Choice:
			dq.Kind = "choice"
			dq.Instructions = marshalRaw(q.Instructions)
			dq.Options = make(map[string]json.RawMessage, len(q.Options))
			for opt, desc := range q.Options {
				dq.Options[opt] = marshalRaw(desc)
			}
		case Score:
			dq.Kind = "score"
			dq.Instructions = marshalRaw(q.Instructions)
			dq.Levels = make([]json.RawMessage, len(q.Levels))
			for i, level := range q.Levels {
				dq.Levels[i] = marshalRaw(level)
			}
		}
		out = append(out, dq)
	}
	return out
}

// toDecisionAnswers converts resp's per-kind answer maps to the root
// [llmkit.DecisionAnswer] vocabulary, sorted by ID. Each id appears in
// exactly one of resp.Nouls/Choices/Scores, matching its question's kind.
func toDecisionAnswers(resp Response) []llmkit.DecisionAnswer {
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

// marshalRaw marshals v to json.RawMessage, degrading to nil on failure
// so a telemetry conversion never propagates an error.
func marshalRaw(v any) json.RawMessage {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return raw
}

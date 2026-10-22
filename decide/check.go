package decide

import (
	"fmt"
	"sort"
)

// Check reports whether r answers qs exactly: every asked id has one
// answer, in the map of its question's kind, and no other id is answered.
// A Choice answer must name one of the asked options. A Score answer's
// Levels and Probabilities each have one entry per asked level, or none
// when the backend reports no distribution. The error names the offending
// id.
func Check(qs Questions, r Response) error {
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if err := checkQuestion(id, qs[id], r); err != nil {
			return err
		}
	}
	return checkUnasked(qs, r)
}

func checkQuestion(id string, q Question, r Response) error {
	_, isNoul := r.Nouls[id]
	choice, isChoice := r.Choices[id]
	score, isScore := r.Scores[id]
	if !isNoul && !isChoice && !isScore {
		return fmt.Errorf("answers[%q]: missing answer for question", id)
	}
	var wantKind string
	var have bool
	switch t := q.(type) {
	case Noul:
		wantKind, have = "noul", isNoul
	case Choice:
		wantKind, have = "choice", isChoice
		if have {
			if _, ok := t.Options[choice.Choice]; !ok {
				return fmt.Errorf("answers[%q].choice: %q is not one of the asked options", id, choice.Choice)
			}
		}
	case Score:
		wantKind, have = "score", isScore
		if have {
			if err := checkScore(id, len(t.Levels), score); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("answers[%q]: question has no recognized kind", id)
	}
	if !have {
		return fmt.Errorf("answers[%q]: question is a %s but has no %s answer", id, wantKind, wantKind)
	}
	if n := b2i(isNoul) + b2i(isChoice) + b2i(isScore); n != 1 {
		return fmt.Errorf("answers[%q]: answered in %d answer kinds, want only %s", id, n, wantKind)
	}
	return nil
}

func checkScore(id string, levels int, a ScoreAnswer) error {
	if n := len(a.Probabilities); n != 0 && n != levels {
		return fmt.Errorf("answers[%q].probabilities: got %d entries, want %d or none", id, n, levels)
	}
	if n := len(a.Levels); n != 0 && n != levels {
		return fmt.Errorf("answers[%q].legend: got %d entries, want %d or none", id, n, levels)
	}
	return nil
}

// checkUnasked rejects answers whose id is not a key of qs, in sorted id
// order so the reported id is deterministic.
func checkUnasked(qs Questions, r Response) error {
	var extra []string
	for id := range r.Nouls {
		if _, ok := qs[id]; !ok {
			extra = append(extra, id)
		}
	}
	for id := range r.Choices {
		if _, ok := qs[id]; !ok {
			extra = append(extra, id)
		}
	}
	for id := range r.Scores {
		if _, ok := qs[id]; !ok {
			extra = append(extra, id)
		}
	}
	if len(extra) == 0 {
		return nil
	}
	sort.Strings(extra)
	return fmt.Errorf("answers[%q]: answer for an unknown question id", extra[0])
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

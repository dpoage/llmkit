package decide

import (
	"errors"
	"fmt"

	"github.com/dpoage/llmkit"
)

// ErrNotAsked is wrapped by the [Response] accessors when the id was not
// asked as a question of the requested kind.
var ErrNotAsked = errors.New("decide: question not asked")

// Questions indexes one Ask's question set by id. Ids are echoed back in the
// response's answer maps.
type Questions map[string]Question

// Response is the normalized result of one Ask.
type Response struct {
	// Model is the model id the backend reports using; it can differ from
	// the requested alias. Ledger on this.
	Model string
	// Usage is token consumption as the backend reports it.
	Usage llmkit.Usage
	// Nouls holds one entry per Noul question. nil when no Nouls were asked.
	Nouls map[string]float64
	// Choices holds one entry per Choice question. nil when none were asked.
	Choices map[string]ChoiceAnswer
	// Scores holds one entry per Score question. nil when none were asked.
	Scores map[string]ScoreAnswer
}

// ChoiceAnswer is the answer to a Choice question. Probabilities map option
// id to probability, passed through exactly as reported. Probabilities may
// be nil and Confidence 0 when the backend reports no distribution; a
// backend must not fabricate them.
type ChoiceAnswer struct {
	Choice        string
	Probabilities map[string]float64
	Confidence    float64
}

// ScoreAnswer is the answer to a Score question. Levels is the backend's
// legend ordered by index and Probabilities is index-aligned with Levels;
// each has one entry per level asked for. Levels and Probabilities may be
// nil and Confidence 0 when the backend reports no distribution; a backend
// must not fabricate them.
type ScoreAnswer struct {
	Score         float64
	Levels        []string
	Probabilities []float64
	Confidence    float64
}

// Noul returns the answer to the Noul question id, or an error wrapping
// [ErrNotAsked] when id is not a Noul answer in r.
func (r Response) Noul(id string) (float64, error) {
	v, ok := r.Nouls[id]
	if !ok {
		return 0, notAsked("noul", id)
	}
	return v, nil
}

// Choice returns the answer to the Choice question id, or an error
// wrapping [ErrNotAsked] when id is not a Choice answer in r.
func (r Response) Choice(id string) (ChoiceAnswer, error) {
	v, ok := r.Choices[id]
	if !ok {
		return ChoiceAnswer{}, notAsked("choice", id)
	}
	return v, nil
}

// Score returns the answer to the Score question id, or an error wrapping
// [ErrNotAsked] when id is not a Score answer in r.
func (r Response) Score(id string) (ScoreAnswer, error) {
	v, ok := r.Scores[id]
	if !ok {
		return ScoreAnswer{}, notAsked("score", id)
	}
	return v, nil
}

func notAsked(kind, id string) error {
	return fmt.Errorf("%w: no %s answer for id %q", ErrNotAsked, kind, id)
}

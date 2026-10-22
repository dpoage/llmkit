package decide

import (
	"context"
	"sort"
	"time"

	"github.com/dpoage/llmkit"
)

// Observe returns a Decider that behaves like d and reports every Ask to
// obs as one [llmkit.KindDecision] event on the Ask's context, so RunID and
// Step ride the context. A nil obs returns d unchanged. The result is safe
// for concurrent use when d and obs are.
//
// An Ask that fails [Normalize] goes to d untouched and emits nothing. Every
// other Ask emits one event: state and questions as Normalize produced them,
// answers, Usage, and Duration. A failed Ask sets Err to the error's text
// and leaves Answers nil and Usage zero. d scrubs secrets from its errors.
//
// If d implements interface{ Identity() llmkit.Identity }, the event's
// Backend is its Provider, Model is the response's model (the Identity's
// when the response names none, and always on failure), and the result
// forwards Identity. Otherwise Backend is empty.
func Observe(d Decider, obs llmkit.Observer) Decider {
	if obs == nil {
		return d
	}
	o := &observed{d: d, obs: obs}
	if id, ok := d.(identified); ok {
		return &observedIdentified{observed: o, identified: id}
	}
	return o
}

// identified is the optional interface a Decider implements to name its
// backend; see [Observe].
type identified interface{ Identity() llmkit.Identity }

type observed struct {
	d   Decider
	obs llmkit.Observer
}

// observedIdentified is the Observe result for a Decider that has an
// Identity; embedding forwards Ask, the explicit field forwards Identity.
type observedIdentified struct {
	*observed
	identified
}

func (o *observed) Ask(ctx context.Context, state any, qs Questions) (Response, error) {
	stateRaw, dqs, nerr := Normalize(state, qs)
	if nerr != nil {
		return o.d.Ask(ctx, state, qs)
	}
	start := time.Now()
	resp, err := o.d.Ask(ctx, state, qs)
	dur := time.Since(start)

	var id llmkit.Identity
	if i, ok := o.d.(identified); ok {
		id = i.Identity()
	}
	de := &llmkit.DecisionEvent{
		Backend:   id.Provider,
		State:     stateRaw,
		Questions: dqs,
	}
	if err != nil {
		de.Model = id.Model
		de.Err = err.Error()
	} else {
		de.Model = resp.Model
		if de.Model == "" {
			de.Model = id.Model
		}
		de.Answers = decisionAnswers(resp)
		de.Usage = resp.Usage
	}
	ev := llmkit.NewEvent(ctx, llmkit.KindDecision)
	ev.Duration = dur
	ev.Decision = de
	o.obs.Observe(ctx, ev)
	return resp, err
}

// decisionAnswers converts resp's per-kind answer maps to the root
// [llmkit.DecisionAnswer] vocabulary, sorted by ID.
func decisionAnswers(resp Response) []llmkit.DecisionAnswer {
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

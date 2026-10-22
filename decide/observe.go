package decide

import (
	"context"
	"sort"
	"time"

	"github.com/dpoage/llmkit"
)

// Observe returns a Decider that behaves exactly like d and reports every
// Ask to obs as one [llmkit.KindDecision] event, built with
// [llmkit.NewEvent] on the Ask's context: RunID and Step ride the context,
// so inside an agent Runner turn the event lands at that turn's Step.
//
// An Ask whose state and questions fail [Normalize] is passed to d
// untouched and emits nothing: the refusal is d's to report and no
// decision was attempted. Every other Ask emits exactly one event, success
// or failure. The event carries the state and questions as Normalize
// produced them (sorted by id), the answers sorted by id, Usage, the Ask's
// wall time as Duration, and on failure Err set to the error's text with
// Answers nil. d is responsible for scrubbing secrets from its errors.
//
// When d implements
//
//	interface{ Identity() llmkit.Identity }
//
// the event's Backend is that Identity's Provider; Model is the response's
// reported model on success (the Identity's Model when the response names
// none) and the Identity's Model on failure. The returned Decider then
// implements Identity() too, forwarding d's. When d does not, Backend is
// empty and Model is the response's.
//
// A nil obs returns d unchanged. The returned (Response, error) are d's,
// exactly. Observe is safe for concurrent use when d and obs are.
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

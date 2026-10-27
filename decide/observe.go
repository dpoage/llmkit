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
// An Ask that fails [Normalize] goes to d untouched and emits nothing.
// Normalize runs with the [ImageSupport] d declares (PlaceAll when none)
// and [LabelRef], so an image d would refuse emits nothing. Every other Ask
// emits one event: State and Questions with "<image:hex>" for each image,
// Images (hash, media type, placements; no bytes), answers, Usage and
// Duration. A failed Ask sets Err to the error's text and leaves Answers
// nil and Usage zero. d scrubs secrets from its errors.
//
// If d implements interface{ Identity() llmkit.Identity }, the event's
// Backend is its Provider, Model is the response's model (the Identity's
// when the response names none, and always on failure), and the result
// forwards Identity. Otherwise Backend is empty. The result likewise
// forwards ImageSupport when d declares one.
func Observe(d Decider, obs llmkit.Observer) Decider {
	if obs == nil {
		return d
	}
	o := &observed{d: d, obs: obs}
	id, hasID := d.(identified)
	is, hasIS := d.(ImageSupporter)
	switch {
	case hasID && hasIS:
		return &observedBoth{observed: o, identified: id, ImageSupporter: is}
	case hasID:
		return &observedIdentified{observed: o, identified: id}
	case hasIS:
		return &observedSupporting{observed: o, ImageSupporter: is}
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

type observedIdentified struct {
	*observed
	identified
}

type observedSupporting struct {
	*observed
	ImageSupporter
}

type observedBoth struct {
	*observed
	identified
	ImageSupporter
}

func (o *observed) Ask(ctx context.Context, state any, qs Questions) (Response, error) {
	support := ImageSupport{Placements: PlaceAll}
	if is, ok := o.d.(ImageSupporter); ok {
		support = is.ImageSupport()
	}
	support.Label = LabelRef
	req, nerr := Normalize(state, qs, support)
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
		State:     req.State,
		Questions: req.Questions,
		Images:    decisionImages(req.Images),
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

func decisionImages(images []Image) []llmkit.DecisionImage {
	if len(images) == 0 {
		return nil
	}
	out := make([]llmkit.DecisionImage, len(images))
	for i, img := range images {
		di := llmkit.DecisionImage{SHA256: img.SHA256, MediaType: img.MediaType, Placements: []string{}}
		for _, pn := range placementNames {
			if img.Placements&pn.place != 0 {
				di.Placements = append(di.Placements, pn.name)
			}
		}
		out[i] = di
	}
	return out
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

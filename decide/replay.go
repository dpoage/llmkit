package decide

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"sync"

	"github.com/dpoage/llmkit"
)

// ErrReplayDiverged is wrapped by a [Replay] Decider's error when an Ask has
// no matching recorded decision, or the match does not answer the asked
// questions. It is distinct from the agent package's ErrReplayDiverged.
var ErrReplayDiverged = errors.New("decide: replay diverged")

// ErrReplayedFailure is wrapped by a [Replay] Decider's error when the
// matched recorded decision is a failed Ask. Only the failure's text is
// recorded, so the error has no root llmkit sentinel.
var ErrReplayedFailure = errors.New("decide: replayed failed decision")

// Replay returns a Decider that answers from the [llmkit.KindDecision]
// events src holds for run, with no backend call. It returns src's error
// (wrapping [llmkit.ErrUnknownRun] for an unknown run). A run with no
// recorded decisions is valid; every Ask against it diverges.
//
// Each Ask follows these rules:
//
//  1. [Normalize] runs first. A refusal is returned unchanged, with Provider
//     "decide", and consumes no record.
//  2. The Ask consumes the earliest unconsumed record with the same Step
//     ([llmkit.StepFromContext]) and the same set of question ids.
//  3. With no match, Ask returns the zero Response and an error wrapping
//     [ErrReplayDiverged] that names the step.
//  4. A matched record with Err set returns the zero Response and an error
//     wrapping [ErrReplayedFailure] with the recorded text.
//  5. Otherwise Ask rebuilds the Response from the recorded answers by the
//     asked question's kind and returns it if it passes [Check]; a Check
//     failure wraps [ErrReplayDiverged].
//
// Replay compares neither the state nor the Backend: a changed state still
// receives the recorded answer. Recorded decisions the replay never asks for
// are not reported. A Decider from Replay is safe for concurrent use.
func Replay(src llmkit.Source, run llmkit.RunID) (Decider, error) {
	events, err := src.Events(context.Background(), run)
	if err != nil {
		return nil, fmt.Errorf("decide: replay run %q: %w", run, err)
	}
	r := &replayer{}
	for i := range events {
		ev := &events[i]
		if ev.Kind != llmkit.KindDecision || ev.Decision == nil {
			continue
		}
		ids := make([]string, len(ev.Decision.Questions))
		for j, q := range ev.Decision.Questions {
			ids[j] = q.ID
		}
		sort.Strings(ids)
		r.records = append(r.records, &replayRecord{step: ev.Step, ids: ids, dec: ev.Decision})
	}
	return r, nil
}

type replayRecord struct {
	step int
	ids  []string // sorted
	dec  *llmkit.DecisionEvent
	used bool
}

type replayer struct {
	mu      sync.Mutex
	records []*replayRecord
}

func (r *replayer) Ask(ctx context.Context, state any, qs Questions) (Response, error) {
	if _, err := Normalize(state, qs); err != nil {
		return Response{}, err
	}
	step := llmkit.StepFromContext(ctx)
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	rec := r.take(step, ids)
	if rec == nil {
		return Response{}, fmt.Errorf("%w: step %d: no unconsumed recorded decision for questions %q", ErrReplayDiverged, step, ids)
	}
	if rec.dec.Err != "" {
		return Response{}, fmt.Errorf("%w: step %d: %s", ErrReplayedFailure, step, rec.dec.Err)
	}
	resp, err := rebuild(rec.dec, qs)
	if err != nil {
		return Response{}, fmt.Errorf("%w: step %d: %v", ErrReplayDiverged, step, err)
	}
	return resp, nil
}

// take consumes and returns the earliest unconsumed record at step with
// exactly the question ids ids (sorted), or nil.
func (r *replayer) take(step int, ids []string) *replayRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, rec := range r.records {
		if !rec.used && rec.step == step && slices.Equal(rec.ids, ids) {
			rec.used = true
			return rec
		}
	}
	return nil
}

// rebuild converts a recorded decision's answers back into a Response by
// the kind of each asked question, then checks it against qs.
func rebuild(dec *llmkit.DecisionEvent, qs Questions) (Response, error) {
	resp := Response{Model: dec.Model, Usage: dec.Usage}
	for _, a := range dec.Answers {
		switch qs[a.ID].(type) {
		case Noul:
			if a.Belief != nil {
				if resp.Nouls == nil {
					resp.Nouls = map[string]float64{}
				}
				resp.Nouls[a.ID] = *a.Belief
			}
		case Choice:
			if resp.Choices == nil {
				resp.Choices = map[string]ChoiceAnswer{}
			}
			resp.Choices[a.ID] = ChoiceAnswer{Choice: a.Choice, Probabilities: a.Probabilities, Confidence: a.Confidence}
		case Score:
			if a.Score != nil {
				if resp.Scores == nil {
					resp.Scores = map[string]ScoreAnswer{}
				}
				resp.Scores[a.ID] = ScoreAnswer{Score: *a.Score, Levels: a.Levels, Probabilities: a.LevelProbabilities, Confidence: a.Confidence}
			}
		default:
			return Response{}, fmt.Errorf("answers[%q]: answer for an unknown question id", a.ID)
		}
	}
	if err := Check(qs, resp); err != nil {
		return Response{}, err
	}
	return resp, nil
}

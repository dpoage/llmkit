package decide

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
//     ([llmkit.StepFromContext]), the same set of question ids, and the same
//     set of image hashes (SHA-256 of the bytes, so media type and number
//     of occurrences do not matter).
//  3. With no match, Ask returns the zero Response and an error wrapping
//     [ErrReplayDiverged] that names the step.
//  4. A matched record with Err set returns the zero Response and an error
//     wrapping [ErrReplayedFailure] with the recorded text.
//  5. Otherwise Ask rebuilds the Response from the recorded answers by the
//     asked question's kind and returns it if it passes [Check]; a Check
//     failure wraps [ErrReplayDiverged].
//
// A record's image hashes are those in its Images plus those of the valid
// image values left inline in its State and Questions, as logs written
// before images were recorded hold them. An inline image that is not valid
// adds no hash and never fails Replay. Scanning inline images is slow at
// pathological nesting depth, as [Normalize] is.
//
// Replay compares neither the state nor the Backend: a changed state still
// receives the recorded answer, unless an image differs. Recorded decisions
// the replay never asks for are not reported. A Decider from Replay is safe
// for concurrent use.
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
		r.records = append(r.records, &replayRecord{step: ev.Step, ids: questionIDs(ev.Decision), images: recordImageHashes(ev.Decision), dec: ev.Decision})
	}
	return r, nil
}

type replayRecord struct {
	step   int
	ids    []string // sorted
	images []string // sorted, distinct SHA-256 hex
	dec    *llmkit.DecisionEvent
	used   bool
}

type replayer struct {
	mu      sync.Mutex
	records []*replayRecord
}

func (r *replayer) Ask(ctx context.Context, state any, qs Questions) (Response, error) {
	req, err := Normalize(state, qs, ImageSupport{Placements: PlaceAll})
	if err != nil {
		return Response{}, err
	}
	step := llmkit.StepFromContext(ctx)
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	images := make([]string, len(req.Images))
	for i, img := range req.Images {
		images[i] = img.SHA256
	}
	sort.Strings(images)

	rec := r.take(step, ids, images)
	if rec == nil {
		return Response{}, fmt.Errorf("%w: step %d: no unconsumed recorded decision for questions %q with image hashes %q", ErrReplayDiverged, step, ids, images)
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

// take consumes the earliest unconsumed record at step whose sorted question
// ids and image hashes equal ids and images; nil if none.
func (r *replayer) take(step int, ids, images []string) *replayRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, rec := range r.records {
		if !rec.used && rec.step == step && slices.Equal(rec.ids, ids) && slices.Equal(rec.images, images) {
			rec.used = true
			return rec
		}
	}
	return nil
}

func questionIDs(dec *llmkit.DecisionEvent) []string {
	ids := make([]string, len(dec.Questions))
	for i, q := range dec.Questions {
		ids[i] = q.ID
	}
	sort.Strings(ids)
	return ids
}

func recordImageHashes(dec *llmkit.DecisionEvent) []string {
	set := map[string]struct{}{}
	for _, img := range dec.Images {
		set[img.SHA256] = struct{}{}
	}
	inlineImageHashes(dec.State, set)
	for _, q := range dec.Questions {
		inlineImageHashes(q.Instructions, set)
		inlineImageHashes(q.True, set)
		inlineImageHashes(q.False, set)
		for _, v := range q.Options {
			inlineImageHashes(v, set)
		}
		for _, v := range q.Levels {
			inlineImageHashes(v, set)
		}
	}
	out := make([]string, 0, len(set))
	for h := range set {
		out = append(out, h)
	}
	sort.Strings(out)
	return out
}

// inlineImageHashes adds to set the SHA-256 of every valid image value in
// raw, by Normalize's definition. It never refuses: an invalid one is skipped.
func inlineImageHashes(raw []byte, set map[string]struct{}) {
	if bytes.IndexByte(raw, '{') < 0 || !json.Valid(raw) {
		return
	}
	scanImageHashes(raw, skipWS(raw, 0), set)
}

func scanImageHashes(src []byte, i int, set map[string]struct{}) int {
	switch src[i] {
	case '[':
		i = skipWS(src, i+1)
		if src[i] == ']' {
			return i + 1
		}
		for {
			i = skipWS(src, scanImageHashes(src, i, set))
			if src[i] == ']' {
				return i + 1
			}
			i = skipWS(src, i+1)
		}
	case '{':
		return scanObjectImageHashes(src, i, set)
	}
	return skipValue(src, i)
}

func scanObjectImageHashes(src []byte, start int, set map[string]struct{}) int {
	i := skipWS(src, start+1)
	if src[i] == '}' {
		return i + 1
	}
	var values []int
	named := true
	for {
		ke := skipValue(src, i)
		named = named && blockField(src[i:ke]) >= 0
		vs := skipWS(src, skipWS(src, ke)+1)
		values = append(values, vs)
		i = skipWS(src, skipValue(src, vs))
		if src[i] == '}' {
			break
		}
		i = skipWS(src, i+1)
	}
	end := i + 1
	if named {
		var b llmkit.Block
		err := json.Unmarshal(src[start:end], &b)
		if b.Kind == llmkit.BlockImage {
			if checkImage(b, err) == "" {
				sum := sha256.Sum256(b.Data)
				set[hex.EncodeToString(sum[:])] = struct{}{}
			}
			return end
		}
	}
	for _, vs := range values {
		scanImageHashes(src, vs, set)
	}
	return end
}

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

package decide

import (
	"fmt"
	"strings"
	"testing"
)

func checkQuestions() Questions {
	return Questions{
		"n": Noul{Instructions: "i"},
		"c": Choice{Instructions: "i", Options: map[string]any{"yes": nil, "no": nil}},
		"s": Score{Instructions: "i", Levels: []any{"lo", "mid", "hi"}},
	}
}

func checkResponse() Response {
	return Response{
		Nouls:   map[string]float64{"n": 0.5},
		Choices: map[string]ChoiceAnswer{"c": {Choice: "yes", Probabilities: map[string]float64{"yes": 0.9, "no": 0.1}, Confidence: 0.8}},
		Scores:  map[string]ScoreAnswer{"s": {Score: 0.5, Levels: []string{"lo", "mid", "hi"}, Probabilities: []float64{0.2, 0.5, 0.3}, Confidence: 0.7}},
	}
}

func TestCheck_Boundaries(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(r *Response)
		wantID  string // empty: want nil
		wantErr bool
	}{
		{name: "well-formed", mutate: func(*Response) {}},
		{
			name: "distribution-less choice and score",
			mutate: func(r *Response) {
				r.Choices = map[string]ChoiceAnswer{"c": {Choice: "no"}}
				r.Scores = map[string]ScoreAnswer{"s": {Score: 0.5}}
			},
		},
		{
			name:    "choice outside the options",
			mutate:  func(r *Response) { r.Choices = map[string]ChoiceAnswer{"c": {Choice: "maybe"}} },
			wantID:  "c",
			wantErr: true,
		},
		{
			name:    "id in Nouls and Choices",
			mutate:  func(r *Response) { r.Nouls["c"] = 0.1 },
			wantID:  "c",
			wantErr: true,
		},
		{
			name: "noul id answered as a score",
			mutate: func(r *Response) {
				delete(r.Nouls, "n")
				r.Scores["n"] = ScoreAnswer{Score: 1}
			},
			wantID:  "n",
			wantErr: true,
		},
		{
			name: "score with 1 of 3 probabilities",
			mutate: func(r *Response) {
				r.Scores = map[string]ScoreAnswer{"s": {Probabilities: []float64{1}}}
			},
			wantID:  "s",
			wantErr: true,
		},
		{
			name: "score with 2 of 3 levels",
			mutate: func(r *Response) {
				r.Scores = map[string]ScoreAnswer{"s": {Levels: []string{"lo", "mid"}}}
			},
			wantID:  "s",
			wantErr: true,
		},
		{
			name:    "missing answer",
			mutate:  func(r *Response) { delete(r.Nouls, "n") },
			wantID:  "n",
			wantErr: true,
		},
		{
			name:    "answer for an unasked id",
			mutate:  func(r *Response) { r.Nouls["zz"] = 0.5 },
			wantID:  "zz",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := checkResponse()
			tt.mutate(&r)
			err := Check(checkQuestions(), r)
			if !tt.wantErr {
				if err != nil {
					t.Fatalf("Check = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Check = nil, want an error")
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("%q", tt.wantID)) {
				t.Errorf("error %q does not name id %q", err, tt.wantID)
			}
		})
	}
}

func TestCheck_NilScoreDistribution(t *testing.T) {
	qs := Questions{"s": Score{Instructions: "i", Levels: []any{"a", "b", "c"}}}
	r := Response{Scores: map[string]ScoreAnswer{"s": {Score: 0.5, Levels: nil, Probabilities: nil}}}
	if err := Check(qs, r); err != nil {
		t.Fatalf("Check = %v, want nil", err)
	}
}

// TestCheck_Property compares Check with a direct statement of its
// contract over every combination of a small question set and answer set.
func TestCheck_Property(t *testing.T) {
	type entry struct {
		put func(r *Response, id string)
	}
	entries := map[string]entry{
		"noul":         {func(r *Response, id string) { r.Nouls[id] = 0.5 }},
		"choice-yes":   {func(r *Response, id string) { r.Choices[id] = ChoiceAnswer{Choice: "yes"} }},
		"choice-maybe": {func(r *Response, id string) { r.Choices[id] = ChoiceAnswer{Choice: "maybe"} }},
		"score-bare":   {func(r *Response, id string) { r.Scores[id] = ScoreAnswer{} }},
		"score-full": {func(r *Response, id string) {
			r.Scores[id] = ScoreAnswer{Levels: make([]string, 3), Probabilities: make([]float64, 3)}
		}},
		"score-1prob":   {func(r *Response, id string) { r.Scores[id] = ScoreAnswer{Probabilities: make([]float64, 1)} }},
		"score-4levels": {func(r *Response, id string) { r.Scores[id] = ScoreAnswer{Levels: make([]string, 4)} }},
	}
	names := []string{"noul", "choice-yes", "choice-maybe", "score-bare", "score-full", "score-1prob", "score-4levels"}
	kinds := []Question{
		nil,
		Noul{Instructions: "i"},
		Choice{Instructions: "i", Options: map[string]any{"yes": nil, "no": nil}},
		Score{Instructions: "i", Levels: []any{1, 2, 3}},
	}
	// A placement is the set of answer kinds one id appears in (a bitmask over names).
	placements := []int{0}
	for i := range names {
		placements = append(placements, 1<<i)
	}
	for _, pair := range [][2]int{{0, 1}, {0, 3}, {1, 2}, {3, 4}, {0, 2}} {
		placements = append(placements, 1<<pair[0]|1<<pair[1])
	}
	ids := []string{"a", "b", "c"}

	want := func(qs Questions, r Response) bool {
		answered := map[string]bool{}
		for _, id := range ids {
			_, n := r.Nouls[id]
			_, c := r.Choices[id]
			_, s := r.Scores[id]
			if n || c || s {
				answered[id] = true
			}
		}
		for _, id := range ids {
			_, asked := qs[id]
			if asked != answered[id] {
				return false
			}
		}
		for id, q := range qs {
			_, n := r.Nouls[id]
			ch, c := r.Choices[id]
			sc, s := r.Scores[id]
			switch q := q.(type) {
			case Noul:
				if !n || c || s {
					return false
				}
			case Choice:
				if n || !c || s {
					return false
				}
				if _, ok := q.Options[ch.Choice]; !ok {
					return false
				}
			case Score:
				if n || c || !s {
					return false
				}
				for _, l := range []int{len(sc.Probabilities), len(sc.Levels)} {
					if l != 0 && l != len(q.Levels) {
						return false
					}
				}
			default:
				return false
			}
		}
		return true
	}

	cases := 0
	var rec func(i int, qs Questions, ps []int)
	rec = func(i int, qs Questions, ps []int) {
		if i < len(ids) {
			for _, k := range kinds {
				next := Questions{}
				for id, q := range qs {
					next[id] = q
				}
				if k != nil {
					next[ids[i]] = k
				}
				for _, p := range placements {
					rec(i+1, next, append(ps[:i:i], p))
				}
			}
			return
		}
		r := Response{Nouls: map[string]float64{}, Choices: map[string]ChoiceAnswer{}, Scores: map[string]ScoreAnswer{}}
		for j, p := range ps {
			for bit, name := range names {
				if p&(1<<bit) != 0 {
					entries[name].put(&r, ids[j])
				}
			}
		}
		cases++
		got := Check(qs, r) == nil
		if got != want(qs, r) {
			t.Fatalf("Check(%v, %+v) nil = %v, want %v", qs, r, got, !got)
		}
	}
	rec(0, Questions{}, nil)
	if cases < 1000 {
		t.Fatalf("only %d cases enumerated", cases)
	}
}

package decide

import (
	"errors"
	"strings"
	"testing"
)

func TestResponseAccessors(t *testing.T) {
	r := checkResponse()
	ids := []string{"n", "c", "s", "typo", ""}
	for _, id := range ids {
		_, nOK := r.Nouls[id]
		_, cOK := r.Choices[id]
		_, sOK := r.Scores[id]

		nv, nErr := r.Noul(id)
		cv, cErr := r.Choice(id)
		sv, sErr := r.Score(id)

		for _, c := range []struct {
			kind string
			ok   bool
			err  error
			zero bool
		}{
			{"noul", nOK, nErr, nv == 0},
			{"choice", cOK, cErr, cv.Choice == "" && cv.Probabilities == nil && cv.Confidence == 0},
			{"score", sOK, sErr, sv.Score == 0 && sv.Levels == nil && sv.Probabilities == nil && sv.Confidence == 0},
		} {
			if c.ok {
				if c.err != nil {
					t.Errorf("%s(%q) err = %v, want nil", c.kind, id, c.err)
				}
				continue
			}
			if !errors.Is(c.err, ErrNotAsked) {
				t.Errorf("%s(%q) err = %v, want ErrNotAsked", c.kind, id, c.err)
				continue
			}
			if !strings.Contains(c.err.Error(), `"`+id+`"`) {
				t.Errorf("%s(%q) error %q does not name the id", c.kind, id, c.err)
			}
			if !c.zero {
				t.Errorf("%s(%q) returned a non-zero value on a miss", c.kind, id)
			}
		}
	}

	if v, err := r.Noul("n"); err != nil || v != 0.5 {
		t.Errorf("Noul(n) = %v, %v", v, err)
	}
	if v, err := r.Choice("c"); err != nil || v.Choice != "yes" {
		t.Errorf("Choice(c) = %+v, %v", v, err)
	}
	if v, err := r.Score("s"); err != nil || v.Score != 0.5 {
		t.Errorf("Score(s) = %+v, %v", v, err)
	}
}

func TestResponseAccessors_ZeroResponse(t *testing.T) {
	var r Response
	if _, err := r.Noul("n"); !errors.Is(err, ErrNotAsked) {
		t.Errorf("Noul on zero Response err = %v", err)
	}
	if _, err := r.Choice("c"); !errors.Is(err, ErrNotAsked) {
		t.Errorf("Choice on zero Response err = %v", err)
	}
	if _, err := r.Score("s"); !errors.Is(err, ErrNotAsked) {
		t.Errorf("Score on zero Response err = %v", err)
	}
}

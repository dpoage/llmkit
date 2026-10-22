package decide

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

func raw(s string) json.RawMessage { return json.RawMessage(s) }

func TestNormalize_RootVocabularySortedByID(t *testing.T) {
	req, err := Normalize(map[string]any{"b": 1, "a": "x"}, Questions{
		"zeta":  Noul{Instructions: "Is it raining?", True: "wet", False: map[string]any{"streets": "dry"}},
		"alpha": Choice{Instructions: map[string]any{"prompt": "Pick."}, Options: map[string]any{"red": nil, "blue": "the sky"}},
		"mid":   Score{Instructions: "Rate.", Levels: []any{"bad", nil, "good"}},
		"bare":  Noul{Instructions: "Bare?"},
	}, ImageSupport{})
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	stateRaw, got := req.State, req.Questions
	if want := `{"a":"x","b":1}`; string(stateRaw) != want {
		t.Errorf("state = %s, want %s", stateRaw, want)
	}
	want := []llmkit.DecisionQuestion{
		{ID: "alpha", Kind: "choice", Instructions: raw(`{"prompt":"Pick."}`), Options: map[string]json.RawMessage{"blue": raw(`"the sky"`), "red": raw(`null`)}},
		{ID: "bare", Kind: "noul", Instructions: raw(`"Bare?"`)},
		{ID: "mid", Kind: "score", Instructions: raw(`"Rate."`), Levels: []json.RawMessage{raw(`"bad"`), raw(`null`), raw(`"good"`)}},
		{ID: "zeta", Kind: "noul", Instructions: raw(`"Is it raining?"`), True: raw(`"wet"`), False: raw(`{"streets":"dry"}`)},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("questions =\n%+v\nwant\n%+v", got, want)
	}
}

func TestNormalize_NullCriterionIsKept(t *testing.T) {
	req, err := Normalize("s", Questions{"q": Noul{Instructions: "i", True: (*int)(nil)}}, ImageSupport{})
	got := req.Questions
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	if string(got[0].True) != "null" || got[0].False != nil {
		t.Errorf("True = %s False = %s, want null and absent", got[0].True, got[0].False)
	}
}

func TestNormalize_FirstErrorIsDeterministic(t *testing.T) {
	qs := Questions{"b": Noul{Instructions: 2}, "a": Noul{Instructions: 1}, "c": Noul{}}
	for i := 0; i < 20; i++ {
		_, err := Normalize("s", qs, ImageSupport{})
		var apiErr *llmkit.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %v, want *llmkit.APIError", err)
		}
		if apiErr.Provider != "decide" || apiErr.Kind != llmkit.ErrInvalidRequest || apiErr.StatusCode != 0 {
			t.Fatalf("err = %+v, want Provider decide, ErrInvalidRequest, status 0", apiErr)
		}
		if want := `questions["a"].instructions: must be a string, object, or array, not a JSON number`; apiErr.Message != want {
			t.Fatalf("Message = %q, want %q", apiErr.Message, want)
		}
	}
}

func TestNormalize_ErrorMessagePaths(t *testing.T) {
	tests := []struct {
		name  string
		state any
		qs    Questions
		want  string
	}{
		{"state null", nil, Questions{"q": Noul{Instructions: "i"}}, "state: must be a string, object, or array, not null"},
		{"no questions", "s", nil, "questions: at least one question is required"},
		{"empty id", "s", Questions{"": Noul{Instructions: "i"}}, "questions: question id must be non-empty"},
		{"nil question", "s", Questions{"q": nil}, `questions["q"]: question is required`},
		{"choice no options", "s", Questions{"q": Choice{Instructions: "i"}}, `questions["q"].options: at least one option is required`},
		{"score one level", "s", Questions{"q": Score{Instructions: "i", Levels: []any{"a"}}}, `questions["q"].levels: at least 2 levels are required, got 1`},
		{"noul true bool", "s", Questions{"q": Noul{Instructions: "i", True: true}}, `questions["q"].criteria.true: must be a string, object, array, or null, not a JSON boolean`},
		{"score level number", "s", Questions{"q": Score{Instructions: "i", Levels: []any{"a", 2}}}, `questions["q"].levels[1]: must be a string, object, array, or null, not a JSON number`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Normalize(tt.state, tt.qs, ImageSupport{})
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) || apiErr.Message != tt.want {
				t.Errorf("err = %v, want Message %q", err, tt.want)
			}
		})
	}
}

func TestNormalize_UnmarshalableValueNamesPath(t *testing.T) {
	_, err := Normalize("s", Questions{"q": Noul{Instructions: make(chan int)}}, ImageSupport{})
	if err == nil || !strings.Contains(err.Error(), `questions["q"].instructions: value cannot be marshalled`) {
		t.Errorf("err = %v, want a marshal failure naming the instructions path", err)
	}
}

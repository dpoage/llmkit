package decide

import (
	"errors"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// The sealed set, pinned at compile time: these three are the only Question
// implementations in the package, and the unexported question() method keeps
// any others out.
var _ = []Question{Noul{}, Choice{}, Score{}}

func TestQuestion_PrewireValidation(t *testing.T) {
	noul := Noul{Instructions: "i"}
	tests := []struct {
		name      string
		state     any
		questions Questions
	}{
		{"nil state", nil, Questions{"q": noul}},
		{"state marshalling to null", (*int)(nil), Questions{"q": noul}},
		{"state number", 42, Questions{"q": noul}},
		{"state bool", true, Questions{"q": noul}},
		{"nil questions", "s", nil},
		{"empty questions", "s", Questions{}},
		{"empty question id", "s", Questions{"": noul}},
		{"nil question", "s", Questions{"q": nil}},
		{"noul nil instructions", "s", Questions{"q": Noul{}}},
		{"noul instructions number", "s", Questions{"q": Noul{Instructions: 7}}},
		{"noul instructions bool", "s", Questions{"q": Noul{Instructions: true}}},
		{"noul criteria number", "s", Questions{"q": Noul{Instructions: "i", True: 1}}},
		{"noul criteria bool", "s", Questions{"q": Noul{Instructions: "i", False: false}}},
		{"choice nil instructions", "s", Questions{"q": Choice{Options: map[string]any{"a": "x"}}}},
		{"choice nil options", "s", Questions{"q": Choice{Instructions: "i"}}},
		{"choice empty options", "s", Questions{"q": Choice{Instructions: "i", Options: map[string]any{}}}},
		{"choice option number", "s", Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": 3}}}},
		{"choice option bool", "s", Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": true}}}},
		{"score nil instructions", "s", Questions{"q": Score{Levels: []any{"a", "b"}}}},
		{"score zero levels", "s", Questions{"q": Score{Instructions: "i"}}},
		{"score one level", "s", Questions{"q": Score{Instructions: "i", Levels: []any{"a"}}}},
		{"score level number", "s", Questions{"q": Score{Instructions: "i", Levels: []any{"a", 2}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := Normalize(tt.state, tt.questions)
			if err == nil {
				t.Fatal("expected a pre-wire validation error")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("err = %v, want ErrInvalidRequest", err)
			}
			var apiErr *llmkit.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %T, want *llmkit.APIError", err)
			}
			if apiErr.StatusCode != 0 {
				t.Errorf("StatusCode = %d, want 0 (validation must not touch the network)", apiErr.StatusCode)
			}
			if apiErr.Provider != "decide" {
				t.Errorf("Provider = %q, want decide", apiErr.Provider)
			}
		})
	}
}

func TestQuestion_ErrorsNameFieldPaths(t *testing.T) {
	_, _, err := Normalize("s", Questions{"q": Noul{Instructions: 7}})
	if err == nil || !strings.Contains(err.Error(), `questions["q"].instructions`) {
		t.Errorf("err = %v, want a message naming the instructions path", err)
	}

	_, _, err = Normalize("s", Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": 1}}})
	if err == nil || !strings.Contains(err.Error(), `questions["q"].options["a"]`) {
		t.Errorf("err = %v, want a message naming the option path", err)
	}

	_, _, err = Normalize(42, Questions{"q": Noul{Instructions: "i"}})
	if err == nil || !strings.Contains(err.Error(), "state") {
		t.Errorf("err = %v, want a message naming state", err)
	}
}

// fakeQuestion proves the seal: an in-package implementation Normalize
// must reject rather than guess a kind for..
type fakeQuestion struct{}

func (fakeQuestion) question() {}

func TestQuestion_SealRejectsUnknownImplementation(t *testing.T) {
	_, _, err := Normalize("s", Questions{"q": fakeQuestion{}})
	if err == nil || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v, want ErrInvalidRequest for an unknown Question implementation", err)
	}
}

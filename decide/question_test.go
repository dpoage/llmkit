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

func TestQuestion_GoldenWire(t *testing.T) {
	tests := []struct {
		name string
		q    Question
		want string
	}{
		{
			name: "noul minimal omits criteria",
			q:    Noul{Instructions: "Is 2+2 equal to 4?"},
			want: `{"type":"noul","instructions":"Is 2+2 equal to 4?"}`,
		},
		{
			name: "noul with one criterion omits the nil side",
			q:    Noul{Instructions: "Is it hot?", True: "hot above 30C"},
			want: `{"type":"noul","instructions":"Is it hot?","criteria":{"true":"hot above 30C"}}`,
		},
		{
			name: "noul with both criteria",
			q:    Noul{Instructions: "Is it hot?", True: "hot", False: "cold"},
			want: `{"type":"noul","instructions":"Is it hot?","criteria":{"true":"hot","false":"cold"}}`,
		},
		{
			name: "choice with object instructions and null option description",
			q: Choice{
				Instructions: map[string]any{"prompt": "pick one"},
				Options:      map[string]any{"blue": "the sky", "red": nil},
			},
			want: `{"type":"choice","instructions":{"prompt":"pick one"},"criteria":{"blue":"the sky","red":null}}`,
		},
		{
			name: "score keeps level order",
			q:    Score{Instructions: "rate the answer", Levels: []any{"low", "mixed", map[string]any{"desc": "high"}}},
			want: `{"type":"score","instructions":"rate the answer","criteria":["low","mixed",{"desc":"high"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := buildQuestion("q1", tt.q)
			if err != nil {
				t.Fatalf("buildQuestion: %v", err)
			}
			if string(raw) != tt.want {
				t.Errorf("wire = %s, want %s", raw, tt.want)
			}
		})
	}
}

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
			_, err := buildRequest(tt.state, "m", tt.questions)
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
			if apiErr.Provider != "typesafe" {
				t.Errorf("Provider = %q, want typesafe", apiErr.Provider)
			}
		})
	}
}

// TestQuestion_NullDescriptionsAllowed pins the kind split: null is a legal
// description and criterion value (the vendor's string|object|array|null open
// set), while state and instructions reject it.
func TestQuestion_NullDescriptionsAllowed(t *testing.T) {
	raw, err := buildQuestion("q", Noul{Instructions: "i", False: (*int)(nil)})
	if err != nil {
		t.Fatalf("noul null criterion: %v", err)
	}
	if want := `{"type":"noul","instructions":"i","criteria":{"false":null}}`; string(raw) != want {
		t.Errorf("noul wire = %s, want %s", raw, want)
	}

	raw, err = buildQuestion("q", Score{Instructions: "i", Levels: []any{nil, "b"}})
	if err != nil {
		t.Fatalf("score null level: %v", err)
	}
	if want := `{"type":"score","instructions":"i","criteria":[null,"b"]}`; string(raw) != want {
		t.Errorf("score wire = %s, want %s", raw, want)
	}
}

func TestQuestion_ErrorsNameFieldPaths(t *testing.T) {
	_, err := buildRequest("s", "m", Questions{"q": Noul{Instructions: 7}})
	if err == nil || !strings.Contains(err.Error(), `questions["q"].instructions`) {
		t.Errorf("err = %v, want a message naming the instructions path", err)
	}

	_, err = buildRequest("s", "m", Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": 1}}})
	if err == nil || !strings.Contains(err.Error(), `questions["q"].options["a"]`) {
		t.Errorf("err = %v, want a message naming the option path", err)
	}

	_, err = buildRequest(42, "m", Questions{"q": Noul{Instructions: "i"}})
	if err == nil || !strings.Contains(err.Error(), "state") {
		t.Errorf("err = %v, want a message naming state", err)
	}
}

// fakeQuestion proves the seal: an in-package implementation the wire builder
// must reject rather than guess a type tag for.
type fakeQuestion struct{}

func (fakeQuestion) question() {}

func TestQuestion_SealRejectsUnknownImplementation(t *testing.T) {
	_, err := buildQuestion("q", fakeQuestion{})
	if err == nil || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v, want ErrInvalidRequest for an unknown Question implementation", err)
	}
}

package systemone

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

// Question-level wire shapes, moved from the former in-package validator:
// each question renders from decide.Normalize's output.
func TestQuestion_GoldenWire(t *testing.T) {
	tests := []struct {
		name string
		q    decide.Question
		want string
	}{
		{
			name: "noul minimal omits criteria",
			q:    decide.Noul{Instructions: "Is 2+2 equal to 4?"},
			want: `{"type":"noul","instructions":"Is 2+2 equal to 4?"}`,
		},
		{
			name: "noul with one criterion omits the nil side",
			q:    decide.Noul{Instructions: "Is it hot?", True: "hot above 30C"},
			want: `{"type":"noul","instructions":"Is it hot?","criteria":{"true":"hot above 30C"}}`,
		},
		{
			name: "noul with both criteria",
			q:    decide.Noul{Instructions: "Is it hot?", True: "hot", False: "cold"},
			want: `{"type":"noul","instructions":"Is it hot?","criteria":{"true":"hot","false":"cold"}}`,
		},
		{
			name: "choice with object instructions and null option description",
			q: decide.Choice{
				Instructions: map[string]any{"prompt": "pick one"},
				Options:      map[string]any{"blue": "the sky", "red": nil},
			},
			want: `{"type":"choice","instructions":{"prompt":"pick one"},"criteria":{"blue":"the sky","red":null}}`,
		},
		{
			name: "score keeps level order",
			q:    decide.Score{Instructions: "rate the answer", Levels: []any{"low", "mixed", map[string]any{"desc": "high"}}},
			want: `{"type":"score","instructions":"rate the answer","criteria":["low","mixed",{"desc":"high"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := marshalOne(t, "q1", tt.q)
			if string(raw) != tt.want {
				t.Errorf("wire = %s, want %s", raw, tt.want)
			}
		})
	}
}

// TestQuestion_NullDescriptionsAllowed pins the kind split: null is a legal
// description and criterion value, while state and instructions reject it.
func TestQuestion_NullDescriptionsAllowed(t *testing.T) {
	raw := marshalOne(t, "q", decide.Noul{Instructions: "i", False: (*int)(nil)})
	if want := `{"type":"noul","instructions":"i","criteria":{"false":null}}`; string(raw) != want {
		t.Errorf("noul wire = %s, want %s", raw, want)
	}

	raw = marshalOne(t, "q", decide.Score{Instructions: "i", Levels: []any{nil, "b"}})
	if want := `{"type":"score","instructions":"i","criteria":[null,"b"]}`; string(raw) != want {
		t.Errorf("score wire = %s, want %s", raw, want)
	}
}

func marshalOne(t *testing.T, id string, q decide.Question) []byte {
	t.Helper()
	req, err := decide.Normalize("s", decide.Questions{id: q}, decide.ImageSupport{})
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	raw, err := marshalQuestion(req.Questions[0])
	if err != nil {
		t.Fatalf("marshalQuestion: %v", err)
	}
	return raw
}

func TestMarshalRequest_BytesMatchLegacyEnvelope(t *testing.T) {
	tests := []struct {
		name  string
		state any
		model string
		want  string
		qs    decide.Questions
	}{
		{
			name:  "mixed object state",
			want:  `{"state":{"host":"db-1","load":[1,2.5],"nested":{"a":"first","z":"last"}},"model":"jev-latest","questions":{"alpha":{"type":"choice","instructions":{"prompt":"Pick a color."},"criteria":{"blue":"the sky","none":null,"red":{"hex":"#f00"}}},"mid":{"type":"score","instructions":["rate","quality"],"criteria":["bad","ok",{"desc":"good"}]},"zeta":{"type":"noul","instructions":"Is it \u003craining\u003e \u0026 cold?","criteria":{"true":"wet streets","false":{"streets":"dry"}}}}}`,
			state: map[string]any{"host": "db-1", "load": []any{1, 2.5}, "nested": map[string]any{"z": "last", "a": "first"}},
			model: "jev-latest",
			qs: decide.Questions{
				"zeta":  decide.Noul{Instructions: "Is it <raining> & cold?", True: "wet streets", False: map[string]any{"streets": "dry"}},
				"alpha": decide.Choice{Instructions: map[string]any{"prompt": "Pick a color."}, Options: map[string]any{"red": map[string]any{"hex": "#f00"}, "blue": "the sky", "none": nil}},
				"mid":   decide.Score{Instructions: []any{"rate", "quality"}, Levels: []any{"bad", "ok", map[string]any{"desc": "good"}}},
			},
		},
		{
			name:  "string state minimal noul",
			want:  `{"state":"a plain string state with é and \u2028","model":"m","questions":{"q":{"type":"noul","instructions":"Is 2+2 equal to 4?"}}}`,
			state: "a plain string state with \u00e9 and \u2028",
			model: "m",
			qs:    decide.Questions{"q": decide.Noul{Instructions: "Is 2+2 equal to 4?"}},
		},
		{
			name:  "noul one criterion and null criterion",
			want:  `{"state":["a","b"],"model":"m","questions":{"null":{"type":"noul","instructions":"i","criteria":{"false":null}},"one":{"type":"noul","instructions":"i","criteria":{"true":"t"}}}}`,
			state: []any{"a", "b"},
			model: "m",
			qs: decide.Questions{
				"one":  decide.Noul{Instructions: "i", True: "t"},
				"null": decide.Noul{Instructions: "i", False: (*int)(nil)},
			},
		},
		{
			name:  "score null level",
			want:  `{"state":"s","model":"m","questions":{"q":{"type":"score","instructions":"i","criteria":[null,"b"]}}}`,
			state: "s",
			model: "m",
			qs:    decide.Questions{"q": decide.Score{Instructions: "i", Levels: []any{nil, "b"}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := decide.Normalize(tt.state, tt.qs, decide.ImageSupport{})
			if err != nil {
				t.Fatalf("Normalize: %v", err)
			}
			body, err := marshalRequest(req.State, tt.model, req.Questions)
			if err != nil {
				t.Fatalf("marshalRequest: %v", err)
			}
			if string(body) != tt.want {
				t.Errorf("body = %s\nwant   %s", body, tt.want)
			}
		})
	}
}

func TestAsk_PrewireRefusalIsTaggedSystemOne(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	_, err := c.Ask(context.Background(), "s", decide.Questions{"q": decide.Noul{}})
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v, want ErrInvalidRequest *APIError", err)
	}
	if apiErr.Provider != "systemone" {
		t.Errorf("Provider = %q, want systemone", apiErr.Provider)
	}
	if apiErr.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0", apiErr.StatusCode)
	}
	if want := `questions["q"].instructions: must be a string, object, or array, not null`; apiErr.Message != want {
		t.Errorf("Message = %q, want %q", apiErr.Message, want)
	}
	if got := hits.Load(); got != 0 {
		t.Errorf("hits = %d, want 0", got)
	}
}

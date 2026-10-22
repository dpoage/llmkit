package systemone_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/decide/systemone"
	"github.com/dpoage/llmkit/internal/livetest"
)

// kevReadmeBody is the response from github.com/jaredpalmer/kev's README
// (@main, "Example response from Kev-4B"), verbatim, latency_ms included.
const kevReadmeBody = `{
  "model": "kev-latest",
  "answers": {
    "department":  { "type": "choice", "choice": "returns", "confidence": 0.21,
                     "probabilities": { "returns": 0.47, "shipping": 0.28, "billing": 0.25 } },
    "escalate":    { "type": "noul", "noul": 0.93 },
    "frustration": { "type": "score", "score": 1.44, "confidence": 0.34,
                     "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                     "probabilities": { "0": 0.00, "1": 0.56, "2": 0.44 } }
  },
  "usage": { "input_tokens": 101, "output_tokens": 161 },
  "latency_ms": 495
}`

// kevTicketState and kevTicketQuestions are the README's request, which
// both the README body and the captured fixture answer.
const kevTicketState = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."

func kevTicketQuestions() decide.Questions {
	return decide.Questions{
		"department": decide.Choice{
			Instructions: "Which team should handle this?",
			Options: map[string]any{
				"returns":  "Exchanges, refunds, wrong or damaged items",
				"shipping": "Delivery status, delays, lost packages",
				"billing":  "Charges, invoices, payment problems",
			},
		},
		"escalate": decide.Noul{Instructions: "Does this need urgent human attention?"},
		"frustration": decide.Score{
			Instructions: "How frustrated is the customer?",
			Levels:       []any{"Calm", "Frustrated", "Very angry"},
		},
	}
}

// askKev serves body from a Kev-shaped server at a non-TypeSafe BaseURL and
// returns what systemone.Ask makes of it.
func askKev(t *testing.T, body string, state any, qs decide.Questions) decide.Response {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/systemone" {
			t.Errorf("request = %s %s, want POST /v1/systemone", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer local" {
			t.Errorf("Authorization = %q, want the configured secret as a bearer token", got)
		}
		reqBody, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(reqBody), `"model":"kev-latest"`) {
			t.Errorf("request body %s does not carry model kev-latest", reqBody)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()
	if strings.Contains(srv.URL, "typesafe") {
		t.Fatalf("test server URL %q must not be a TypeSafe host", srv.URL)
	}

	cl, err := systemone.New(systemone.Config{
		Secret:  "local",
		Model:   "kev-latest",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("systemone.New: %v", err)
	}
	resp, err := cl.Ask(context.Background(), state, qs)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	return resp
}

// TestAsk_ParsesKevReadmeResponse: the README's response body, verbatim,
// parses to exactly the expected Response.
func TestAsk_ParsesKevReadmeResponse(t *testing.T) {
	got := askKev(t, kevReadmeBody, kevTicketState, kevTicketQuestions())
	want := decide.Response{
		Model: "kev-latest",
		Usage: llmkit.Usage{InputTokens: 101, OutputTokens: 161},
		Nouls: map[string]float64{"escalate": 0.93},
		Choices: map[string]decide.ChoiceAnswer{
			"department": {
				Choice:        "returns",
				Probabilities: map[string]float64{"returns": 0.47, "shipping": 0.28, "billing": 0.25},
				Confidence:    0.21,
			},
		},
		Scores: map[string]decide.ScoreAnswer{
			"frustration": {
				Score:         1.44,
				Levels:        []string{"Calm", "Frustrated", "Very angry"},
				Probabilities: []float64{0, 0.56, 0.44},
				Confidence:    0.34,
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Ask response mismatch\n got: %+v\nwant: %+v", got, want)
	}
}

// TestAsk_ParsesKevCapturedResponse: a response body captured from a local
// Kev-0.8B server (testdata/kev_ticket.json) parses to exactly the expected
// Response. The expected values are written here, not read from the fixture.
func TestAsk_ParsesKevCapturedResponse(t *testing.T) {
	f, err := livetest.ReadDecideFixture(filepath.Join("testdata", "kev_ticket.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Exchanges) != 1 {
		t.Fatalf("fixture has %d exchanges, want 1", len(f.Exchanges))
	}
	body := f.Exchanges[0].Response.Body
	if !strings.Contains(body, `"latency_ms"`) {
		t.Fatalf("captured body %s lacks the latency_ms extra field", body)
	}

	got := askKev(t, body, kevTicketState, kevTicketQuestions())
	want := decide.Response{
		Model: "kev-latest",
		Usage: llmkit.Usage{InputTokens: 101, OutputTokens: 179},
		Nouls: map[string]float64{"escalate": 0.5512},
		Choices: map[string]decide.ChoiceAnswer{
			"department": {
				Choice:        "shipping",
				Probabilities: map[string]float64{"billing": 0.1685, "returns": 0.3111, "shipping": 0.5204},
				Confidence:    0.2805,
			},
		},
		Scores: map[string]decide.ScoreAnswer{
			"frustration": {
				Score:         1.23,
				Levels:        []string{"Calm", "Frustrated", "Very angry"},
				Probabilities: []float64{0.0515, 0.667, 0.2815},
				Confidence:    0.5005,
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Ask response mismatch\n got: %+v\nwant: %+v", got, want)
	}
}

package livetest

// The decide-package twin of Fixture (fixture.go): the recorded artifact of
// one live decide case. The two formats share the sanitized Exchange wire
// truth and the write-time secret refusal; they differ on the input and
// outcome halves, which are Ask-shaped rather than llmkit.Request-shaped.
//
// A decide.Question is a sealed Go interface — it cannot be unmarshalled
// from wire JSON — so the recorder stores each question's kind and raw-JSON
// fields (DecideQuestionSpec) and replay rebuilds the Noul/Choice/Score
// values from them before re-issuing the Ask through decide.New.

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/dpoage/llmkit/decide"
)

//   - State and Questions are the Ask inputs; replay re-issues them through
//     decide.New.
//   - Exchanges is the sanitized wire truth — what the vendor actually
//     received and returned, one per HTTP request (a retried Ask records
//     several); replay serves them from an httptest server.
//   - Normalized is the Ask's final outcome: the normalized decide.Response,
//     or the sentinel error kind; replay asserts the client still produces
//     it from the recorded truth.
type DecideFixture struct {
	Case  string `json:"case"`
	Lane  string `json:"lane"`
	Model string `json:"model"`
	// State is the Ask's state as marshalled JSON (string, object, or
	// array); replay re-issues it verbatim.
	State json.RawMessage `json:"state"`
	// Questions records the Ask's questions in replayable form; see
	// DecideQuestionSpec.
	Questions map[string]DecideQuestionSpec `json:"questions"`
	// Exchanges is the sanitized wire truth in call order.
	Exchanges []Exchange `json:"exchanges"`
	// Normalized is the Ask's final outcome.
	Normalized DecideNormalized `json:"normalized"`
}

// DecideQuestionSpec is one recorded question: enough to rebuild the
// decide.Noul, decide.Choice, or decide.Score value the case issued. The
// bytes are the validated values the recorder captured, so re-marshalling
// them is byte-identical to the recorded request.
type DecideQuestionSpec struct {
	Kind         string                     `json:"kind"` // noul | choice | score
	Instructions json.RawMessage            `json:"instructions"`
	True         json.RawMessage            `json:"true,omitempty"`
	False        json.RawMessage            `json:"false,omitempty"`
	Options      map[string]json.RawMessage `json:"options,omitempty"`
	Levels       []json.RawMessage          `json:"levels,omitempty"`
}

// DecideNormalized is the recorded outcome of one Ask: the normalized
// decide.Response, or the sentinel error kind when the case expected a
// failure.
type DecideNormalized struct {
	ErrKind  string           `json:"err_kind,omitempty"`
	Response *decide.Response `json:"response,omitempty"`
}

// RecordDecideQuestions converts the questions an Ask issues into their
// replayable recorded form. decide.Question is sealed, so any other kind is
// impossible; an unrecognized value is an error naming the question id.
func RecordDecideQuestions(qs decide.Questions) (map[string]DecideQuestionSpec, error) {
	specs := make(map[string]DecideQuestionSpec, len(qs))
	for id, q := range qs {
		spec := DecideQuestionSpec{}
		switch q := q.(type) {
		case decide.Noul:
			spec.Kind = "noul"
			raw, err := json.Marshal(q.Instructions)
			if err != nil {
				return nil, fmt.Errorf("livetest: question %q: record instructions: %w", id, err)
			}
			spec.Instructions = raw
			if q.True != nil {
				raw, err := json.Marshal(q.True)
				if err != nil {
					return nil, fmt.Errorf("livetest: question %q: record true criterion: %w", id, err)
				}
				spec.True = raw
			}
			if q.False != nil {
				raw, err := json.Marshal(q.False)
				if err != nil {
					return nil, fmt.Errorf("livetest: question %q: record false criterion: %w", id, err)
				}
				spec.False = raw
			}
		case decide.Choice:
			spec.Kind = "choice"
			raw, err := json.Marshal(q.Instructions)
			if err != nil {
				return nil, fmt.Errorf("livetest: question %q: record instructions: %w", id, err)
			}
			spec.Instructions = raw
			spec.Options = make(map[string]json.RawMessage, len(q.Options))
			for opt, desc := range q.Options {
				raw, err := json.Marshal(desc)
				if err != nil {
					return nil, fmt.Errorf("livetest: question %q: record option %q: %w", id, opt, err)
				}
				spec.Options[opt] = raw
			}
		case decide.Score:
			spec.Kind = "score"
			raw, err := json.Marshal(q.Instructions)
			if err != nil {
				return nil, fmt.Errorf("livetest: question %q: record instructions: %w", id, err)
			}
			spec.Instructions = raw
			spec.Levels = make([]json.RawMessage, len(q.Levels))
			for i, level := range q.Levels {
				raw, err := json.Marshal(level)
				if err != nil {
					return nil, fmt.Errorf("livetest: question %q: record level %d: %w", id, i, err)
				}
				spec.Levels[i] = raw
			}
		default:
			return nil, fmt.Errorf("livetest: question %q: unknown kind %T", id, q)
		}
		specs[id] = spec
	}
	return specs, nil
}

// BuildDecideQuestions rebuilds the decide.Questions a fixture recorded, for
// replay. Every field is carried as raw JSON, so the rebuilt Ask marshals to
// the recorded wire bytes; unknown recorded kinds are an error naming the
// question id.
func BuildDecideQuestions(specs map[string]DecideQuestionSpec) (decide.Questions, error) {
	qs := make(decide.Questions, len(specs))
	for id, spec := range specs {
		switch spec.Kind {
		case "noul":
			q := decide.Noul{Instructions: json.RawMessage(spec.Instructions)}
			if len(spec.True) > 0 {
				q.True = json.RawMessage(spec.True)
			}
			if len(spec.False) > 0 {
				q.False = json.RawMessage(spec.False)
			}
			qs[id] = q
		case "choice":
			opts := make(map[string]any, len(spec.Options))
			for opt, desc := range spec.Options {
				opts[opt] = json.RawMessage(desc)
			}
			qs[id] = decide.Choice{Instructions: json.RawMessage(spec.Instructions), Options: opts}
		case "score":
			levels := make([]any, len(spec.Levels))
			for i, level := range spec.Levels {
				levels[i] = json.RawMessage(level)
			}
			qs[id] = decide.Score{Instructions: json.RawMessage(spec.Instructions), Levels: levels}
		default:
			return nil, fmt.Errorf("livetest: question %q: unknown recorded kind %q", id, spec.Kind)
		}
	}
	return qs, nil
}

// WriteDecideFixture serializes f to path with the same refusal WriteFixture
// applies: a recording containing the lane credential, an sk- key, or an
// apikey_ key prefix is not written at all. Fatal on any refusal or write
// error.
func WriteDecideFixture(t testing.TB, path string, f *DecideFixture, secret string) {
	if err := writeSecretFree(path, f, secret); err != nil {
		t.Fatalf("livetest: %v", err)
	}
}

// ReadDecideFixture loads a recorded decide fixture.
func ReadDecideFixture(path string) (*DecideFixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f DecideFixture
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("livetest: %s: %w", path, err)
	}
	return &f, nil
}

// NormalizeDecideFixture compacts every json.RawMessage embedded in f — the
// state and each recorded question's fields. Fixture files are
// pretty-printed, which re-indents embedded RawMessage bytes; anyone
// re-issuing a fixture's Ask MUST compact them back first, or the client's
// wire bytes drift from the recording.
func NormalizeDecideFixture(f *DecideFixture) {
	f.State = compactRaw(f.State)
	for id, spec := range f.Questions {
		spec.Instructions = compactRaw(spec.Instructions)
		spec.True = compactRaw(spec.True)
		spec.False = compactRaw(spec.False)
		for opt, desc := range spec.Options {
			spec.Options[opt] = compactRaw(desc)
		}
		for i := range spec.Levels {
			spec.Levels[i] = compactRaw(spec.Levels[i])
		}
		f.Questions[id] = spec
	}
}

package decide

import (
	"encoding/json"
	"fmt"
	"github.com/dpoage/llmkit"
	"strconv"
)

// Question is the sealed set of System One questions: [Noul], [Choice], and
// [Score]. The unexported method keeps the set closed — code outside this
// package cannot add a question kind — and the wire "type" discriminator is
// derived from the Go type, so callers never write it.
type Question interface{ question() }

// Noul asks a binary belief question: how strongly the state supports the
// True description over the False one. Instructions is required and must be
// a string, object, or array. True and False are optional criteria; a nil
// side omits its key, and both nil omit the whole criteria object.
type Noul struct {
	Instructions any
	True, False  any
}

// Choice asks which option best fits the state. Instructions is required.
// Options maps option id to description and is required and non-empty; a nil
// value serializes as the JSON null the vendor accepts.
type Choice struct {
	Instructions any
	Options      map[string]any
}

// Score asks for a rating against an ordered legend. Instructions is
// required. Levels is the ordered list of level descriptions, from lowest to
// highest; it is required and must hold at least two.
type Score struct {
	Instructions any
	Levels       []any
}

func (Noul) question()   {}
func (Choice) question() {}
func (Score) question()  {}

// wireQuestion is the JSON body of one question. Instructions and Criteria
// carry pre-validated raw JSON so the marshalled bytes are exactly what the
// kind checks approved.
type wireQuestion struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// noulCriteria is the optional noul criteria object; nil sides are omitted.
type noulCriteria struct {
	True  json.RawMessage `json:"true,omitempty"`
	False json.RawMessage `json:"false,omitempty"`
}

// buildRequest validates the state and every question pre-wire and returns
// the complete request body. No network I/O happens on this path: every
// failure is an *llmkit.APIError wrapping llmkit.ErrInvalidRequest with a
// message naming the field path.
func buildRequest(state any, model string, questions Questions) ([]byte, error) {
	stateRaw, err := marshalStructured(state, "state")
	if err != nil {
		return nil, err
	}
	if len(questions) == 0 {
		return nil, invalidRequest("questions", "at least one question is required")
	}
	wq := make(map[string]json.RawMessage, len(questions))
	for id, q := range questions {
		if id == "" {
			return nil, invalidRequest("questions", "question id must be non-empty")
		}
		raw, err := buildQuestion(id, q)
		if err != nil {
			return nil, err
		}
		wq[id] = raw
	}
	body, err := json.Marshal(wireRequest{State: stateRaw, Model: model, Questions: wq})
	if err != nil {
		return nil, fmt.Errorf("decide: marshal request: %w", err)
	}
	return body, nil
}

func buildQuestion(id string, q Question) (json.RawMessage, error) {
	path := "questions[" + strconv.Quote(id) + "]"
	switch t := q.(type) {
	case Noul:
		return buildNoul(path, t)
	case Choice:
		return buildChoice(path, t)
	case Score:
		return buildScore(path, t)
	case nil:
		return nil, invalidRequest(path, "question is required")
	default:
		return nil, invalidRequest(path, "unknown Question implementation (the set is sealed: Noul, Choice, Score)")
	}
}

func buildNoul(path string, q Noul) (json.RawMessage, error) {
	ins, err := marshalStructured(q.Instructions, path+".instructions")
	if err != nil {
		return nil, err
	}
	wq := wireQuestion{Type: "noul", Instructions: ins}
	if q.True != nil || q.False != nil {
		criteria := noulCriteria{}
		if q.True != nil {
			if criteria.True, err = marshalDescription(q.True, path+".criteria.true"); err != nil {
				return nil, err
			}
		}
		if q.False != nil {
			if criteria.False, err = marshalDescription(q.False, path+".criteria.false"); err != nil {
				return nil, err
			}
		}
		wq.Criteria = criteria
	}
	return json.Marshal(wq)
}

func buildChoice(path string, q Choice) (json.RawMessage, error) {
	ins, err := marshalStructured(q.Instructions, path+".instructions")
	if err != nil {
		return nil, err
	}
	if len(q.Options) == 0 {
		return nil, invalidRequest(path+".options", "at least one option is required")
	}
	criteria := make(map[string]json.RawMessage, len(q.Options))
	for opt, desc := range q.Options {
		raw, err := marshalDescription(desc, path+".options["+strconv.Quote(opt)+"]")
		if err != nil {
			return nil, err
		}
		criteria[opt] = raw
	}
	return json.Marshal(wireQuestion{Type: "choice", Instructions: ins, Criteria: criteria})
}

func buildScore(path string, q Score) (json.RawMessage, error) {
	ins, err := marshalStructured(q.Instructions, path+".instructions")
	if err != nil {
		return nil, err
	}
	if len(q.Levels) < 2 {
		return nil, invalidRequest(path+".levels", fmt.Sprintf("at least 2 levels are required, got %d", len(q.Levels)))
	}
	criteria := make([]json.RawMessage, len(q.Levels))
	for i, level := range q.Levels {
		raw, err := marshalDescription(level, path+".levels["+strconv.Itoa(i)+"]")
		if err != nil {
			return nil, err
		}
		criteria[i] = raw
	}
	return json.Marshal(wireQuestion{Type: "score", Instructions: ins, Criteria: criteria})
}

// marshalStructured validates that v marshals to one of the vendor's
// structured text kinds — string, object, or array — and returns the
// marshalled bytes. Numbers, booleans, and null are rejected: state and
// instructions carry prose, and a mis-typed value must fail in the caller's
// process, not as a vendor 422.
func marshalStructured(v any, path string) (json.RawMessage, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, invalidRequest(path, fmt.Sprintf("value cannot be marshalled: %v", err))
	}
	switch kindByte(raw) {
	case '"', '{', '[':
		return raw, nil
	case 'n':
		return nil, invalidRequest(path, "must be a string, object, or array, not null")
	case 't', 'f':
		return nil, invalidRequest(path, "must be a string, object, or array, not a JSON boolean")
	default:
		return nil, invalidRequest(path, "must be a string, object, or array, not a JSON number")
	}
}

// marshalDescription validates that v marshals to one of the vendor's
// description kinds — string, object, array, or null — and returns the
// marshalled bytes. Only numbers and booleans are rejected; null is a legal
// description.
func marshalDescription(v any, path string) (json.RawMessage, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, invalidRequest(path, fmt.Sprintf("value cannot be marshalled: %v", err))
	}
	switch kindByte(raw) {
	case '"', '{', '[', 'n':
		return raw, nil
	case 't', 'f':
		return nil, invalidRequest(path, "must be a string, object, array, or null, not a JSON boolean")
	default:
		return nil, invalidRequest(path, "must be a string, object, array, or null, not a JSON number")
	}
}

// kindByte returns the first non-whitespace byte of a JSON document — its
// kind: '"' string, '{' object, '[' array, 'n' null, 't'/'f' boolean,
// digit/'-' number.
func kindByte(raw json.RawMessage) byte {
	for _, b := range raw {
		switch b {
		case ' ', '\t', '\n', '\r':
			continue
		default:
			return b
		}
	}
	return 0
}

// invalidRequest builds the pre-wire validation error: an *llmkit.APIError
// wrapping llmkit.ErrInvalidRequest, StatusCode 0 (nothing reached the
// network), naming the offending field path.
func invalidRequest(path, problem string) error {
	return &llmkit.APIError{
		Kind:     llmkit.ErrInvalidRequest,
		Provider: providerName,
		Message:  path + ": " + problem,
	}
}

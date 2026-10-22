package decide

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/dpoage/llmkit"
)

// Normalize validates state and qs and returns the state's marshalled JSON
// and the questions in the root vocabulary, sorted by ID. state must
// marshal to a JSON string, object, or array; Instructions likewise;
// descriptions (True, False, Options, Levels) also accept null. At least
// one question is required, each with a non-empty id.
//
// A refusal is an *llmkit.APIError with Kind llmkit.ErrInvalidRequest,
// StatusCode 0, Provider "decide", and Message "<path>: <problem>". A
// backend re-tags Provider to its own name.
func Normalize(state any, qs Questions) (json.RawMessage, []llmkit.DecisionQuestion, error) {
	stateRaw, err := marshalStructured(state, "state")
	if err != nil {
		return nil, nil, err
	}
	if len(qs) == 0 {
		return nil, nil, invalidRequest("questions", "at least one question is required")
	}
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]llmkit.DecisionQuestion, 0, len(ids))
	for _, id := range ids {
		if id == "" {
			return nil, nil, invalidRequest("questions", "question id must be non-empty")
		}
		dq, err := normalizeQuestion(id, qs[id])
		if err != nil {
			return nil, nil, err
		}
		out = append(out, dq)
	}
	return stateRaw, out, nil
}

func normalizeQuestion(id string, q Question) (llmkit.DecisionQuestion, error) {
	path := "questions[" + strconv.Quote(id) + "]"
	switch t := q.(type) {
	case Noul:
		return normalizeNoul(id, path, t)
	case Choice:
		return normalizeChoice(id, path, t)
	case Score:
		return normalizeScore(id, path, t)
	case nil:
		return llmkit.DecisionQuestion{}, invalidRequest(path, "question is required")
	default:
		return llmkit.DecisionQuestion{}, invalidRequest(path, "unknown Question implementation (the set is sealed: Noul, Choice, Score)")
	}
}

func normalizeNoul(id, path string, q Noul) (llmkit.DecisionQuestion, error) {
	ins, err := marshalStructured(q.Instructions, path+".instructions")
	if err != nil {
		return llmkit.DecisionQuestion{}, err
	}
	dq := llmkit.DecisionQuestion{ID: id, Kind: "noul", Instructions: ins}
	if q.True != nil {
		if dq.True, err = marshalDescription(q.True, path+".criteria.true"); err != nil {
			return llmkit.DecisionQuestion{}, err
		}
	}
	if q.False != nil {
		if dq.False, err = marshalDescription(q.False, path+".criteria.false"); err != nil {
			return llmkit.DecisionQuestion{}, err
		}
	}
	return dq, nil
}

func normalizeChoice(id, path string, q Choice) (llmkit.DecisionQuestion, error) {
	ins, err := marshalStructured(q.Instructions, path+".instructions")
	if err != nil {
		return llmkit.DecisionQuestion{}, err
	}
	if len(q.Options) == 0 {
		return llmkit.DecisionQuestion{}, invalidRequest(path+".options", "at least one option is required")
	}
	opts := make(map[string]json.RawMessage, len(q.Options))
	for opt, desc := range q.Options {
		raw, err := marshalDescription(desc, path+".options["+strconv.Quote(opt)+"]")
		if err != nil {
			return llmkit.DecisionQuestion{}, err
		}
		opts[opt] = raw
	}
	return llmkit.DecisionQuestion{ID: id, Kind: "choice", Instructions: ins, Options: opts}, nil
}

func normalizeScore(id, path string, q Score) (llmkit.DecisionQuestion, error) {
	ins, err := marshalStructured(q.Instructions, path+".instructions")
	if err != nil {
		return llmkit.DecisionQuestion{}, err
	}
	if len(q.Levels) < 2 {
		return llmkit.DecisionQuestion{}, invalidRequest(path+".levels", fmt.Sprintf("at least 2 levels are required, got %d", len(q.Levels)))
	}
	levels := make([]json.RawMessage, len(q.Levels))
	for i, level := range q.Levels {
		raw, err := marshalDescription(level, path+".levels["+strconv.Itoa(i)+"]")
		if err != nil {
			return llmkit.DecisionQuestion{}, err
		}
		levels[i] = raw
	}
	return llmkit.DecisionQuestion{ID: id, Kind: "score", Instructions: ins, Levels: levels}, nil
}

// marshalStructured accepts a value that marshals to a JSON string, object,
// or array; numbers, booleans, and null are refused.
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

// marshalDescription accepts a string, object, array, or null; numbers and
// booleans are refused.
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

// kindByte returns the first non-whitespace byte of a JSON document.
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

func invalidRequest(path, problem string) error {
	return &llmkit.APIError{
		Kind:     llmkit.ErrInvalidRequest,
		Provider: "decide",
		Message:  path + ": " + problem,
	}
}

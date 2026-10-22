package systemone

import (
	"encoding/json"
	"fmt"

	"github.com/dpoage/llmkit"
)

// wireRequest is the JSON envelope POSTed to /v1/systemone.
type wireRequest struct {
	State     json.RawMessage            `json:"state"`
	Model     string                     `json:"model"`
	Questions map[string]json.RawMessage `json:"questions"`
}

// wireQuestion is the JSON body of one question. Criteria is a map, slice,
// or noulCriteria built from decide.Normalize's validated raw JSON.
type wireQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     any             `json:"criteria,omitempty"`
}

// noulCriteria omits any side that is empty.
type noulCriteria struct {
	True  json.RawMessage `json:"true,omitempty"`
	False json.RawMessage `json:"false,omitempty"`
}

// marshalRequest renders the request body from decide.Normalize's output.
func marshalRequest(stateRaw json.RawMessage, model string, dqs []llmkit.DecisionQuestion) ([]byte, error) {
	wq := make(map[string]json.RawMessage, len(dqs))
	for _, dq := range dqs {
		raw, err := marshalQuestion(dq)
		if err != nil {
			return nil, err
		}
		wq[dq.ID] = raw
	}
	body, err := json.Marshal(wireRequest{State: stateRaw, Model: model, Questions: wq})
	if err != nil {
		return nil, fmt.Errorf("systemone: marshal request: %w", err)
	}
	return body, nil
}

func marshalQuestion(dq llmkit.DecisionQuestion) (json.RawMessage, error) {
	w := wireQuestion{Type: dq.Kind, Instructions: dq.Instructions}
	switch dq.Kind {
	case "noul":
		if len(dq.True) > 0 || len(dq.False) > 0 {
			w.Criteria = noulCriteria{True: dq.True, False: dq.False}
		}
	case "choice":
		w.Criteria = dq.Options
	case "score":
		w.Criteria = dq.Levels
	default:
		return nil, fmt.Errorf("systemone: marshal request: question %q: unknown kind %q", dq.ID, dq.Kind)
	}
	raw, err := json.Marshal(w)
	if err != nil {
		return nil, fmt.Errorf("systemone: marshal request: %w", err)
	}
	return raw, nil
}

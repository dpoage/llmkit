package decide

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/dpoage/llmkit"
)

// Request is a validated state and question set in the root vocabulary.
// Each image in State and in the questions' Instructions, True, False,
// Options values and Levels is replaced by its label ([ImageSupport.Label]).
type Request struct {
	State     json.RawMessage
	Questions []llmkit.DecisionQuestion
	// Images holds the distinct images, deduplicated by bytes, in order of
	// first occurrence: the state in document order, then each question in
	// sorted id order, instructions before criteria. Nil when there are none.
	Images []Image
}

// Normalize validates state and qs and returns a Request: State is the
// state's marshalled JSON and Questions are the questions in the root
// vocabulary, sorted by ID. state and each Instructions must marshal to a
// JSON string, object, or array; descriptions (True, False, Options, Levels)
// also accept null. At least one question is required, each with a
// non-empty id. Questions are validated in sorted id order, so the first
// refusal is the same on every call.
//
// An image value is a JSON object whose member names all match Block JSON
// field names (case-insensitively, escapes decoded, a repeated member takes
// its last value) and whose "kind" is "image", however it was built. It is
// replaced whole by its label; its inner objects are not scanned.
//
// An image must carry inline Data and a MediaType beginning "image/"; URL,
// Text, Title, Provider and Raw must be empty, and every member must decode.
// An image object more than 10 000 levels deep, counting itself, does not
// decode and is refused. The same bytes under two media types are refused, as
// is an image whose placement or media type s does not carry; the zero s
// refuses every image. An object whose "kind" is "image", whose "data" is
// neither null nor "" and that has a member that is not a Block field is
// refused too. Any other object is left as marshalled and goes out as text.
//
// A refusal is an *llmkit.APIError with Kind llmkit.ErrInvalidRequest,
// StatusCode 0, Provider "decide", and Message "<path>: <problem>", where
// path names the JSON path, as in state["frames"][2]["frame"]. A backend
// re-tags Provider to its own name. A refusal for s alone also wraps
// [ErrImageUnsupported].
func Normalize(state any, qs Questions, s ImageSupport) (Request, error) {
	set := &imageSet{support: s}
	stateRaw, err := marshalStructured(state, "state")
	if err != nil {
		return Request{}, err
	}
	if stateRaw, err = set.rewrite(stateRaw, "state", PlaceState); err != nil {
		return Request{}, err
	}
	if len(qs) == 0 {
		return Request{}, invalidRequest("questions", "at least one question is required")
	}
	ids := make([]string, 0, len(qs))
	for id := range qs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]llmkit.DecisionQuestion, 0, len(ids))
	for _, id := range ids {
		if id == "" {
			return Request{}, invalidRequest("questions", "question id must be non-empty")
		}
		dq, err := normalizeQuestion(id, qs[id], set)
		if err != nil {
			return Request{}, err
		}
		out = append(out, dq)
	}
	return Request{State: stateRaw, Questions: out, Images: set.images}, nil
}

func normalizeQuestion(id string, q Question, set *imageSet) (llmkit.DecisionQuestion, error) {
	path := "questions[" + strconv.Quote(id) + "]"
	switch t := q.(type) {
	case Noul:
		return normalizeNoul(id, path, t, set)
	case Choice:
		return normalizeChoice(id, path, t, set)
	case Score:
		return normalizeScore(id, path, t, set)
	case nil:
		return llmkit.DecisionQuestion{}, invalidRequest(path, "question is required")
	default:
		return llmkit.DecisionQuestion{}, invalidRequest(path, "unknown Question implementation (the set is sealed: Noul, Choice, Score)")
	}
}

// instructions marshals, validates and image-scans a question's Instructions.
func instructions(v any, path string, set *imageSet) (json.RawMessage, error) {
	path += ".instructions"
	raw, err := marshalStructured(v, path)
	if err != nil {
		return nil, err
	}
	return set.rewrite(raw, path, PlaceInstructions)
}

// description marshals, validates and image-scans one criteria description.
func description(v any, path string, set *imageSet) (json.RawMessage, error) {
	raw, err := marshalDescription(v, path)
	if err != nil {
		return nil, err
	}
	return set.rewrite(raw, path, PlaceCriteria)
}

func normalizeNoul(id, path string, q Noul, set *imageSet) (llmkit.DecisionQuestion, error) {
	ins, err := instructions(q.Instructions, path, set)
	if err != nil {
		return llmkit.DecisionQuestion{}, err
	}
	dq := llmkit.DecisionQuestion{ID: id, Kind: "noul", Instructions: ins}
	if q.True != nil {
		if dq.True, err = description(q.True, path+".criteria.true", set); err != nil {
			return llmkit.DecisionQuestion{}, err
		}
	}
	if q.False != nil {
		if dq.False, err = description(q.False, path+".criteria.false", set); err != nil {
			return llmkit.DecisionQuestion{}, err
		}
	}
	return dq, nil
}

func normalizeChoice(id, path string, q Choice, set *imageSet) (llmkit.DecisionQuestion, error) {
	ins, err := instructions(q.Instructions, path, set)
	if err != nil {
		return llmkit.DecisionQuestion{}, err
	}
	if len(q.Options) == 0 {
		return llmkit.DecisionQuestion{}, invalidRequest(path+".options", "at least one option is required")
	}
	keys := make([]string, 0, len(q.Options))
	for opt := range q.Options {
		keys = append(keys, opt)
	}
	sort.Strings(keys)
	opts := make(map[string]json.RawMessage, len(q.Options))
	for _, opt := range keys {
		raw, err := description(q.Options[opt], path+".options["+strconv.Quote(opt)+"]", set)
		if err != nil {
			return llmkit.DecisionQuestion{}, err
		}
		opts[opt] = raw
	}
	return llmkit.DecisionQuestion{ID: id, Kind: "choice", Instructions: ins, Options: opts}, nil
}

func normalizeScore(id, path string, q Score, set *imageSet) (llmkit.DecisionQuestion, error) {
	ins, err := instructions(q.Instructions, path, set)
	if err != nil {
		return llmkit.DecisionQuestion{}, err
	}
	if len(q.Levels) < 2 {
		return llmkit.DecisionQuestion{}, invalidRequest(path+".levels", fmt.Sprintf("at least 2 levels are required, got %d", len(q.Levels)))
	}
	levels := make([]json.RawMessage, len(q.Levels))
	for i, level := range q.Levels {
		raw, err := description(level, path+".levels["+strconv.Itoa(i)+"]", set)
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

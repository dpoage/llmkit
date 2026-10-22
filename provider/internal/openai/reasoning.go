package openai

import (
	"bytes"
	"encoding/json"
	"net/url"
	"strings"

	"github.com/dpoage/llmkit"
	"github.com/openai/openai-go/v3/packages/respjson"
)

const (
	keyReasoningContent = "reasoning_content"
	keyReasoningDetails = "reasoning_details"
)

// thinkingTag is the BlockThinking.Provider tag for this adapter: "openai"
// first-party, "openai-compatible:<host of baseURL>" otherwise.
func thinkingTag(compatible bool, baseURL string) string {
	if !compatible {
		return "openai"
	}
	var host string
	if u, err := url.Parse(baseURL); err == nil {
		host = u.Host
	}
	return "openai-compatible:" + host
}

// reasoning is the reasoning payload of one assistant turn. Content is
// reasoning_content; Details is the canonical encoding of a non-empty
// reasoning_details array. A zero value carries nothing.
type reasoning struct {
	Content string          `json:"reasoning_content,omitempty"`
	Details json.RawMessage `json:"reasoning_details,omitempty"`
}

func (r reasoning) block(tag string) (llmkit.Block, bool) {
	if r.Content == "" && len(r.Details) == 0 {
		return llmkit.Block{}, false
	}
	raw, _ := json.Marshal(r) // cannot fail: Details is JSON from canonicalDetails
	return llmkit.Block{Kind: llmkit.BlockThinking, Provider: tag, Text: r.Content, Raw: raw}, true
}

// extraString returns the string value of an extra field; null, absent and
// non-string values yield "".
func extraString(extra map[string]respjson.Field, key string) string {
	f, ok := extra[key]
	if !ok {
		return ""
	}
	var s string
	if json.Unmarshal([]byte(f.Raw()), &s) != nil {
		return ""
	}
	return s
}

// extraDetails returns the elements of a reasoning_details extra field; null,
// absent and non-array values yield nil.
func extraDetails(extra map[string]respjson.Field) []any {
	f, ok := extra[keyReasoningDetails]
	if !ok {
		return nil
	}
	return decodeDetails([]byte(f.Raw()))
}

func decodeDetails(raw []byte) []any {
	var elems []any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if dec.Decode(&elems) != nil {
		return nil
	}
	return elems
}

// canonicalDetails encodes elems identically on Complete and Stream; numbers
// keep their literal form and unknown keys survive.
func canonicalDetails(elems []any) json.RawMessage {
	if len(elems) == 0 {
		return nil
	}
	b, _ := json.Marshal(elems) // cannot fail: elems were decoded from JSON
	return b
}

func reasoningFromMessage(extra map[string]respjson.Field) reasoning {
	return reasoning{
		Content: extraString(extra, keyReasoningContent),
		Details: canonicalDetails(extraDetails(extra)),
	}
}

// reasoningAcc accumulates the reasoning extra fields of every delta of
// choice 0: the SDK accumulator drops them.
type reasoningAcc struct {
	content strings.Builder
	details detailsMerger
}

func (a *reasoningAcc) add(extra map[string]respjson.Field) {
	a.content.WriteString(extraString(extra, keyReasoningContent))
	for _, e := range extraDetails(extra) {
		a.details.add(e)
	}
}

func (a *reasoningAcc) result() reasoning {
	return reasoning{Content: a.content.String(), Details: canonicalDetails(a.details.merged())}
}

// concatKeys are the detail fields whose string fragments concatenate;
// every other field takes its last non-empty value.
var concatKeys = map[string]bool{"text": true, "summary": true, "data": true}

// detailsMerger merges reasoning_details fragments by (index, type),
// ordering elements by first appearance.
type detailsMerger struct {
	order []*detailElem
	byKey map[detailKey]*detailElem
}

type detailKey struct{ index, typ string }

type detailElem struct {
	verbatim any // non-object fragment, never merged
	fields   map[string]any
	texts    map[string]*strings.Builder
}

func (m *detailsMerger) add(frag any) {
	obj, ok := frag.(map[string]any)
	if !ok {
		m.order = append(m.order, &detailElem{verbatim: frag})
		return
	}
	key := detailKey{index: fieldKey(obj["index"]), typ: fieldKey(obj["type"])}
	el := m.byKey[key]
	if el == nil {
		el = &detailElem{fields: map[string]any{}, texts: map[string]*strings.Builder{}}
		if m.byKey == nil {
			m.byKey = map[detailKey]*detailElem{}
		}
		m.byKey[key] = el
		m.order = append(m.order, el)
	}
	for k, v := range obj {
		if s, isStr := v.(string); isStr && concatKeys[k] {
			if el.texts[k] == nil {
				el.texts[k] = &strings.Builder{}
			}
			el.texts[k].WriteString(s)
			continue
		}
		if prev, seen := el.fields[k]; seen && isEmptyValue(v) && !isEmptyValue(prev) {
			continue
		}
		el.fields[k] = v
	}
}

func (m *detailsMerger) merged() []any {
	out := make([]any, 0, len(m.order))
	for _, el := range m.order {
		if el.fields == nil {
			out = append(out, el.verbatim)
			continue
		}
		for k, sb := range el.texts {
			el.fields[k] = sb.String()
		}
		out = append(out, el.fields)
	}
	return out
}

func fieldKey(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case json.Number:
		return x.String()
	}
	b, _ := json.Marshal(v)
	return string(b)
}

func isEmptyValue(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	case map[string]any:
		return len(x) == 0
	}
	return false
}

// emitFields validates an own-tag thinking block's Raw and returns the
// wire fields it carries. Raw must be a JSON object holding at least one
// included key; other keys are ignored.
func emitFields(raw json.RawMessage) (map[string]any, bool) {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, false
	}
	fields := map[string]any{}
	if rc, ok := obj[keyReasoningContent]; ok {
		var s string
		if json.Unmarshal(rc, &s) == nil && s != "" {
			fields[keyReasoningContent] = s
		}
	}
	if rd, ok := obj[keyReasoningDetails]; ok && len(decodeDetails(rd)) > 0 {
		fields[keyReasoningDetails] = rd
	}
	return fields, len(fields) > 0
}

package sqlite

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/dpoage/llmkit"
)

// maxPayloadDepth is SQLite's JSON nesting limit: a deeper document is
// malformed to every json_* function a view calls.
const maxPayloadDepth = 1000

// The stored* types shadow the Event paths that carry Request.Messages: a
// shallower field hides the embedded one of the same JSON name, so the
// payload holds each message's hash where the event holds the message.
type storedEvent struct {
	llmkit.Event
	Completion *storedCompletion `json:"completion,omitempty"`
	Attempt    *storedAttempt    `json:"attempt,omitempty"`
}

type storedCompletion struct {
	llmkit.CompletionEvent
	Request storedRequest `json:"request"`
}

type storedAttempt struct {
	llmkit.AttemptEvent
	Request storedRequest `json:"request"`
}

type storedRequest struct {
	llmkit.Request
	Messages []string `json:"messages,omitempty"`
}

// message is one deduplicated Request.Messages entry: the SHA-256 of its
// JSON encoding and that encoding.
type message struct {
	hash string
	body []byte
}

// encoded is an event ready to write: the payload with message hashes and
// the messages it refers to, in request order.
type encoded struct {
	payload  []byte
	messages []message
}

func encode(ev llmkit.Event) (encoded, error) {
	var (
		out encoded
		se  = storedEvent{Event: ev}
	)
	if c := ev.Completion; c != nil {
		req, msgs, err := hashRequest(c.Request)
		if err != nil {
			return encoded{}, err
		}
		out.messages = append(out.messages, msgs...)
		se.Completion = &storedCompletion{CompletionEvent: *c, Request: req}
	}
	if a := ev.Attempt; a != nil {
		req, msgs, err := hashRequest(a.Request)
		if err != nil {
			return encoded{}, err
		}
		out.messages = append(out.messages, msgs...)
		se.Attempt = &storedAttempt{AttemptEvent: *a, Request: req}
	}
	var err error
	if out.payload, err = json.Marshal(se); err != nil {
		return encoded{}, err
	}
	if d := jsonDepth(out.payload); d > maxPayloadDepth {
		return encoded{}, fmt.Errorf("payload nests %d levels deep, over the %d SQLite's JSON functions accept", d, maxPayloadDepth)
	}
	if err := out.readBack(); err != nil {
		return encoded{}, err
	}
	return out, nil
}

// readBack runs Events' decode over what enc would write, so an event Events
// could not read is refused instead of stored.
func (enc encoded) readBack() error {
	se, err := decode(enc.payload)
	if err != nil {
		return fmt.Errorf("payload would not read back: %w", err)
	}
	bodies := make(map[string][]byte, len(enc.messages))
	for _, m := range enc.messages {
		bodies[m.hash] = m.body
	}
	if _, err := se.assemble(bodies); err != nil {
		return fmt.Errorf("would not read back: %w", err)
	}
	return nil
}

func hashRequest(r llmkit.Request) (storedRequest, []message, error) {
	sr := storedRequest{Request: r}
	msgs := make([]message, len(r.Messages))
	if len(r.Messages) > 0 {
		sr.Messages = make([]string, len(r.Messages))
	}
	for i, m := range r.Messages {
		body, err := json.Marshal(m)
		if err != nil {
			return storedRequest{}, nil, fmt.Errorf("message %d: %w", i, err)
		}
		sum := sha256.Sum256(body)
		msgs[i] = message{hash: hex.EncodeToString(sum[:]), body: body}
		sr.Messages[i] = msgs[i].hash
	}
	return sr, msgs, nil
}

// decode parses a stored payload; its message hashes are resolved by assemble.
func decode(payload []byte) (storedEvent, error) {
	var se storedEvent
	err := json.Unmarshal(payload, &se)
	return se, err
}

// hashes lists the message hashes the payload names.
func (se storedEvent) hashes() []string {
	var hs []string
	if se.Completion != nil {
		hs = append(hs, se.Completion.Request.Messages...)
	}
	if se.Attempt != nil {
		hs = append(hs, se.Attempt.Request.Messages...)
	}
	return hs
}

// assemble rebuilds the event the payload was encoded from; bodies maps each
// message hash it names to its JSON encoding.
func (se storedEvent) assemble(bodies map[string][]byte) (llmkit.Event, error) {
	ev := se.Event
	if se.Completion != nil {
		c := se.Completion.CompletionEvent
		var err error
		if c.Request, err = restoreRequest(se.Completion.Request, bodies); err != nil {
			return llmkit.Event{}, err
		}
		ev.Completion = &c
	}
	if se.Attempt != nil {
		a := se.Attempt.AttemptEvent
		var err error
		if a.Request, err = restoreRequest(se.Attempt.Request, bodies); err != nil {
			return llmkit.Event{}, err
		}
		ev.Attempt = &a
	}
	return ev, nil
}

func restoreRequest(sr storedRequest, bodies map[string][]byte) (llmkit.Request, error) {
	r := sr.Request
	r.Messages = nil
	if len(sr.Messages) > 0 {
		r.Messages = make([]llmkit.Message, len(sr.Messages))
	}
	for i, h := range sr.Messages {
		body, ok := bodies[h]
		if !ok {
			return llmkit.Request{}, fmt.Errorf("message %s is missing", h)
		}
		if err := json.Unmarshal(body, &r.Messages[i]); err != nil {
			return llmkit.Request{}, fmt.Errorf("message %s: %w", h, err)
		}
	}
	return r, nil
}

// jsonDepth is the nesting depth of a well-formed JSON document as SQLite
// counts it: a bare scalar is 0, {"a":1} is 1.
func jsonDepth(b []byte) int {
	depth, deepest := 0, 0
	inString := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		switch {
		case inString:
			switch c {
			case '\\':
				i++
			case '"':
				inString = false
			}
		case c == '"':
			inString = true
		case c == '{' || c == '[':
			depth++
			deepest = max(deepest, depth)
		case c == '}' || c == ']':
			depth--
		}
	}
	return deepest
}

// unixNanos is t as UTC Unix nanoseconds, saturating outside int64's range
// (years 1678..2262) so the zero Time sorts first instead of wrapping.
func unixNanos(t time.Time) int64 {
	sec := t.Unix()
	const limit = math.MaxInt64 / int64(time.Second)
	switch {
	case sec >= limit:
		return math.MaxInt64
	case sec <= -limit:
		return math.MinInt64
	}
	return sec*int64(time.Second) + int64(t.Nanosecond())
}

func nullable[T ~string](s T) any {
	if s == "" {
		return nil
	}
	return string(s)
}

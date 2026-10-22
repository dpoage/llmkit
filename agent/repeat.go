package agent

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/dpoage/llmkit"
)

const (
	repeatWindow  = 12
	repeatMaxRuns = 1024
)

type callID [sha256.Size]byte

// RepetitionPolicy returns a [ToolPolicy] that denies a call repeating a periodic pattern of
// identical calls (same tool, JSON-equal arguments) in the run's last 12 calls: A,A,A or
// A,B,A,B,A. State is per run id and shared by every run using the policy; [Continue] resets it.
func RepetitionPolicy() ToolPolicy {
	return &repetitionPolicy{runs: map[llmkit.RunID]*list.Element{}, lru: list.New()}
}

type repetitionPolicy struct {
	mu   sync.Mutex
	runs map[llmkit.RunID]*list.Element
	lru  *list.List
}

type repeatRun struct {
	id   llmkit.RunID
	last []callID
}

func (p *repetitionPolicy) Authorize(ctx context.Context, call *llmkit.ToolCall) error {
	id := identifyCall(call.Name, call.Arguments)
	run := llmkit.RunFromContext(ctx)

	p.mu.Lock()
	defer p.mu.Unlock()
	el, ok := p.runs[run]
	if ok {
		p.lru.MoveToFront(el)
	} else {
		el = p.lru.PushFront(&repeatRun{id: run})
		p.runs[run] = el
		if p.lru.Len() > repeatMaxRuns {
			oldest := p.lru.Back()
			p.lru.Remove(oldest)
			delete(p.runs, oldest.Value.(*repeatRun).id)
		}
	}
	st := el.Value.(*repeatRun)
	if len(st.last) == repeatWindow {
		copy(st.last, st.last[1:])
		st.last = st.last[:repeatWindow-1]
	}
	st.last = append(st.last, id)
	if !periodicTail(st.last) {
		return nil
	}
	n := 0
	for _, t := range st.last {
		if t == id {
			n++
		}
	}
	return fmt.Errorf("repeated identical call: %s called %d times in the last %d calls without progress",
		call.Name, n, repeatWindow)
}

// periodicTail reports whether, for some p >= 1 with 2p+1 <= len(t), the last 2p+1 entries satisfy
// t[i] == t[i+p].
func periodicTail(t []callID) bool {
	for p := 1; 2*p+1 <= len(t); p++ {
		s := t[len(t)-2*p-1:]
		ok := true
		for i := 0; i <= p; i++ {
			if s[i] != s[i+p] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// identifyCall hashes the tool name with its arguments decoded under UseNumber and re-encoded
// (sorted keys, literal numbers). Arguments that do not parse as one JSON value hash their raw bytes.
func identifyCall(name string, args json.RawMessage) callID {
	h := sha256.New()
	h.Write([]byte(name))
	h.Write([]byte{0})
	if canon, ok := canonicalArgs(args); ok {
		h.Write([]byte{'j'})
		h.Write(canon)
	} else {
		h.Write([]byte{'r'})
		h.Write(args)
	}
	var id callID
	h.Sum(id[:0])
	return id
}

func canonicalArgs(args json.RawMessage) ([]byte, bool) {
	dec := json.NewDecoder(bytes.NewReader(args))
	dec.UseNumber()
	var v any
	if dec.Decode(&v) != nil {
		return nil, false
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, false
	}
	out, err := json.Marshal(v)
	return out, err == nil
}

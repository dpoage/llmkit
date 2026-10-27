package sqlite_test

// R1's Store leg: a tool run recorded into a store is cut at every event
// boundary, each cut is written into a fresh store, and agent.NewResume
// reads it back through the store as the llmkit.Source.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	sqlite "github.com/dpoage/llmkit/store/sqlite"
)

// resumeLiveClient forwards to a scripted client and records every request.
type resumeLiveClient struct {
	f    funcClient
	mu   sync.Mutex
	reqs []llmkit.Request
}

func (c *resumeLiveClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	c.mu.Lock()
	c.reqs = append(c.reqs, req)
	c.mu.Unlock()
	return c.f(req)
}

func (c *resumeLiveClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (c *resumeLiveClient) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.reqs)
}

func (c *resumeLiveClient) first() llmkit.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.reqs[0]
}

func resumeCanonRequest(t *testing.T, r llmkit.Request) string {
	t.Helper()
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var fresh llmkit.Request
	if err := json.Unmarshal(b, &fresh); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	b, err = json.Marshal(fresh)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	return string(b)
}

func resumeServable(evs []llmkit.Event) (reqs []llmkit.Request) {
	for _, ev := range evs {
		if ev.Kind != llmkit.KindCompletion || ev.Completion == nil || ev.Completion.Purpose != "" {
			continue
		}
		ce := ev.Completion
		if ce.Err != "" && ce.ErrKind != llmkit.ErrKindContextTooLong {
			continue
		}
		reqs = append(reqs, ce.Request)
	}
	return reqs
}

func TestResumeFromStoreAtEveryCut(t *testing.T) {
	ctx := context.Background()
	const system = "You test resume."
	const task = "call echo twice, then answer"
	runID := llmkit.RunID("store-resume")
	st, log, _ := openStore(t)
	origRunner := agent.NewRunner(loopClient(2), []agent.Tool{markerTool()}, system, agent.WithObserver(st))
	run(t, origRunner, task, agent.WithRunID(runID))
	noErrors(t, log)
	full, err := st.Events(ctx, runID)
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if len(resumeServable(full)) != 3 {
		t.Fatalf("record holds %d servable completions, want 3", len(resumeServable(full)))
	}
	wantReqs := resumeServable(full)
	var wantUsage llmkit.Usage
	for _, ev := range full {
		if ev.Kind == llmkit.KindCompletion && ev.Completion != nil && ev.Completion.Err == "" {
			wantUsage = wantUsage.Add(ev.Completion.Response.Usage)
		}
	}
	for cut := 1; cut <= len(full); cut++ {
		t.Run(fmt.Sprintf("cut=%d", cut), func(t *testing.T) {
			cutStore, err := sqlite.Open(filepath.Join(t.TempDir(), "cut.db"), nil)
			if err != nil {
				t.Fatalf("Open cut store: %v", err)
			}
			defer func() {
				if err := cutStore.Close(); err != nil {
					t.Errorf("close cut store: %v", err)
				}
			}()
			for _, ev := range full[:cut] {
				cutStore.Observe(ctx, ev)
			}
			got, err := cutStore.Events(ctx, runID)
			if err != nil {
				t.Fatalf("cut Events: %v", err)
			}
			k := len(resumeServable(got))
			live := &resumeLiveClient{f: loopClient(2)}
			template := agent.NewRunner(live, []agent.Tool{markerTool()}, system)
			res, err := agent.NewResume(ctx, cutStore, runID, template)
			if k == 0 {
				if !errors.Is(err, agent.ErrResumeMismatch) {
					t.Fatalf("record with no completion: err = %v, want ErrResumeMismatch", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewResume: %v", err)
			}
			if cut == len(full) && res.Usage() != wantUsage {
				t.Errorf("full-record Usage() = %+v, want %+v", res.Usage(), wantUsage)
			}
			out, err := res.Run(ctx, task)
			if err != nil {
				t.Fatalf("resumed run: %v", err)
			}
			if got, want := live.calls(), len(wantReqs)-k; got != want {
				t.Errorf("live client calls = %d, want %d", got, want)
			}
			if k < len(wantReqs) && live.calls() > 0 {
				if resumeCanonRequest(t, wantReqs[k]) != resumeCanonRequest(t, live.first()) {
					t.Errorf("first live request differs from the original's next request")
				}
			}
			if out.FinalText != "final answer" {
				t.Errorf("FinalText = %q, want %q", out.FinalText, "final answer")
			}
			for _, ev := range out.Transcript.Record {
				if ev.ParentRunID != runID {
					t.Fatalf("event %s lacks ParentRunID", ev.Kind)
				}
			}
		})
	}
}

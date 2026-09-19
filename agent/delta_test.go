package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// streamFakeClient wraps fakeClient with a Stream implementation, so a test
// can pin which delivery path the Runner chose. Stream consumes the same
// scripted steps as Complete (via the embedded fakeClient, which also keeps
// recording requests) and delivers one delta per content block before
// returning that response — the minimal native-stream shape; wire-level
// fragmentation is the adapters' concern, not the Runner's.
type streamFakeClient struct {
	*fakeClient

	mu            sync.Mutex
	streamCalls   int
	completeCalls int
}

func (s *streamFakeClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	s.mu.Lock()
	s.completeCalls++
	s.mu.Unlock()
	return s.fakeClient.Complete(ctx, req)
}

func (s *streamFakeClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	s.mu.Lock()
	s.streamCalls++
	s.mu.Unlock()
	resp, err := s.fakeClient.Complete(ctx, req)
	if err != nil {
		return llmkit.Response{}, err
	}
	// Native streams deliver text as it is generated, so an unblocked
	// scripted response streams its Text field; blocked responses stream
	// one delta per block, matching the normalized shape.
	if len(resp.Blocks) == 0 && resp.Text != "" {
		if err := fn(llmkit.Delta{Kind: llmkit.DeltaText, Text: resp.Text}); err != nil {
			return llmkit.Response{}, err
		}
	}
	for _, b := range resp.Blocks {
		var d llmkit.Delta
		switch b.Kind {
		case llmkit.BlockText:
			d = llmkit.Delta{Kind: llmkit.DeltaText, Text: b.Text}
		case llmkit.BlockThinking:
			d = llmkit.Delta{Kind: llmkit.DeltaThinking, Text: b.Text}
		default:
			continue
		}
		if err := fn(d); err != nil {
			return llmkit.Response{}, err
		}
	}
	for i, tc := range resp.ToolCalls {
		d := llmkit.Delta{Kind: llmkit.DeltaToolCall, Index: i, ID: tc.ID, Name: tc.Name, Arguments: string(tc.Arguments)}
		if err := fn(d); err != nil {
			return llmkit.Response{}, err
		}
	}
	return resp, nil
}

func (s *streamFakeClient) calls() (streams, completes int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.streamCalls, s.completeCalls
}

// deltaHit is one Hooks.Delta delivery: the step it reported and the delta.
type deltaHit struct {
	step int
	d    llmkit.Delta
}

// deltaRecorder collects the (step, Delta) pairs a Hooks.Delta receives. No
// lock: the Runner fires Delta synchronously on the loop goroutine and the
// tests read after Run returns.
type deltaRecorder struct {
	hits []deltaHit
}

func (rec *deltaRecorder) hooks() Hooks {
	return Hooks{Delta: func(_ context.Context, step int, d llmkit.Delta) {
		rec.hits = append(rec.hits, deltaHit{step: step, d: d})
	}}
}

// TestDelta_HookUnsetUsesComplete pins the zero-change contract: without
// Hooks.Delta the Runner calls Complete exactly as before and never touches
// Stream, even on a client that implements it.
func TestDelta_HookUnsetUsesComplete(t *testing.T) {
	fc := &streamFakeClient{fakeClient: newFakeClient(textResp("done", 5, 5))}
	r := NewRunner(fc, nil, "sys")

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	streams, completes := fc.calls()
	if streams != 0 {
		t.Errorf("Stream calls = %d, want 0 (hook unset must not stream)", streams)
	}
	if completes != 1 {
		t.Errorf("Complete calls = %d, want 1", completes)
	}
	if out.FinalText != "done" {
		t.Errorf("FinalText = %q, want %q", out.FinalText, "done")
	}
}

// TestDelta_StreamsEveryCompletionKind pins that the hook receives deltas in
// wire order with the step the Before/AfterCompletion pair reports, for every
// completion kind the loop issues: the main turn, a max-tokens continuation
// turn, a RunJSON forced-finalization turn, and a RunJSON repair turn.
func TestDelta_StreamsEveryCompletionKind(t *testing.T) {
	t.Run("main turn", func(t *testing.T) {
		fc := &streamFakeClient{fakeClient: newFakeClient(
			toolResp("c1", "echo", `{"v":"x"}`, 10, 4),
			textResp("all done", 5, 2),
		)}
		rec := &deltaRecorder{}
		// order pins the documented fire-point sequence: every completion's
		// deltas land after its BeforeCompletion and before its
		// AfterCompletion.
		var order []string
		hooks := Hooks{
			BeforeCompletion: func(context.Context, int, *llmkit.Request) { order = append(order, "before") },
			AfterCompletion: func(context.Context, int, *llmkit.Request, *llmkit.Response, error) {
				order = append(order, "after")
			},
		}
		base := rec.hooks()
		hooks.Delta = func(ctx context.Context, step int, d llmkit.Delta) {
			order = append(order, "delta")
			base.Delta(ctx, step, d)
		}
		r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(hooks))

		if _, err := r.Run(context.Background(), "task"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		want := []deltaHit{
			{1, llmkit.Delta{Kind: llmkit.DeltaToolCall, Index: 0, ID: "c1", Name: "echo", Arguments: `{"v":"x"}`}},
			{2, llmkit.Delta{Kind: llmkit.DeltaText, Text: "all done"}},
		}
		if !reflect.DeepEqual(rec.hits, want) {
			t.Errorf("deltas = %+v, want %+v", rec.hits, want)
		}
		wantOrder := []string{"before", "delta", "after", "before", "delta", "after"}
		if !reflect.DeepEqual(order, wantOrder) {
			t.Errorf("fire order = %v, want %v", order, wantOrder)
		}
	})

	t.Run("max-tokens continuation", func(t *testing.T) {
		fc := &streamFakeClient{fakeClient: newFakeClient(
			maxTokensResp("head", 5, 5),
			textResp("tail", 5, 2),
		)}
		rec := &deltaRecorder{}
		r := NewRunner(fc, nil, "sys", WithHooks(rec.hooks()))

		out, err := r.Run(context.Background(), "task")
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		want := []deltaHit{
			{1, llmkit.Delta{Kind: llmkit.DeltaText, Text: "head"}},
			{2, llmkit.Delta{Kind: llmkit.DeltaText, Text: "tail"}},
		}
		if !reflect.DeepEqual(rec.hits, want) {
			t.Errorf("deltas = %+v, want %+v", rec.hits, want)
		}
		if out.FinalText != "headtail" {
			t.Errorf("FinalText = %q, want stitched %q", out.FinalText, "headtail")
		}
	})

	t.Run("RunJSON finalization", func(t *testing.T) {
		// The iteration cap fires after the tool turn, reserving one
		// tool-less finalization completion at step 2; its JSON answer parses.
		fc := &streamFakeClient{fakeClient: newFakeClient(
			toolResp("c1", "echo", `{"v":"x"}`, 10, 4),
			textResp(`{"path":"f.go","note":"final"}`, 5, 2),
		)}
		rec := &deltaRecorder{}
		r := NewRunner(fc, []Tool{echoTool{name: "echo"}}, "sys",
			WithHooks(rec.hooks()), WithLimits(Limits{MaxIterations: 1}))

		var got item
		out, err := r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &got)
		if err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		want := []deltaHit{
			{1, llmkit.Delta{Kind: llmkit.DeltaToolCall, Index: 0, ID: "c1", Name: "echo", Arguments: `{"v":"x"}`}},
			{2, llmkit.Delta{Kind: llmkit.DeltaText, Text: `{"path":"f.go","note":"final"}`}},
		}
		if !reflect.DeepEqual(rec.hits, want) {
			t.Errorf("deltas = %+v, want %+v", rec.hits, want)
		}
		if !out.Finalized {
			t.Error("Finalized = false, want true (finalization turn was taken)")
		}
		if got.Path != "f.go" {
			t.Errorf("parsed = %+v, want the finalization turn's answer", got)
		}
	})

	t.Run("RunJSON repair", func(t *testing.T) {
		// First answer is not JSON; the single repair pass re-asks at the
		// parent's step count + 1 (step 2 here).
		fc := &streamFakeClient{fakeClient: newFakeClient(
			textResp("here is the answer: not json at all", 5, 5),
			textResp(`{"path":"c.go","note":"fixed"}`, 5, 5),
		)}
		rec := &deltaRecorder{}
		r := NewRunner(fc, nil, "sys", WithHooks(rec.hooks()))

		var got item
		if _, err := r.RunJSON(context.Background(), "task", json.RawMessage(`{"type":"object"}`), &got); err != nil {
			t.Fatalf("RunJSON: %v", err)
		}
		want := []deltaHit{
			{1, llmkit.Delta{Kind: llmkit.DeltaText, Text: "here is the answer: not json at all"}},
			{2, llmkit.Delta{Kind: llmkit.DeltaText, Text: `{"path":"c.go","note":"fixed"}`}},
		}
		if !reflect.DeepEqual(rec.hits, want) {
			t.Errorf("deltas = %+v, want %+v", rec.hits, want)
		}
		if got.Path != "c.go" {
			t.Errorf("parsed = %+v, want the repair's answer", got)
		}
	})
}

// TestDelta_StreamingTranscriptMatchesComplete pins that the delivery path
// never changes what the run records: two identical scripts, one driven with
// Hooks.Delta set (Stream path) and one without (Complete path), produce
// DeepEqual Outcome.Messages and DeepEqual transcripts modulo event
// timestamps, plus identical outcome accounting.
func TestDelta_StreamingTranscriptMatchesComplete(t *testing.T) {
	script := func() []scriptStep {
		return []scriptStep{
			toolResp("c1", "echo", `{"v":"x"}`, 10, 4),
			textResp("all done", 5, 2),
		}
	}
	tools := []Tool{echoTool{name: "echo"}}

	plain := newFakeClient(script()...)
	rp := NewRunner(plain, tools, "sys")
	outPlain, err := rp.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("plain Run: %v", err)
	}

	sf := &streamFakeClient{fakeClient: newFakeClient(script()...)}
	rec := &deltaRecorder{}
	rs := NewRunner(sf, tools, "sys", WithHooks(rec.hooks()))
	outStream, err := rs.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("streaming Run: %v", err)
	}
	if len(rec.hits) == 0 {
		t.Fatal("streaming run delivered no deltas")
	}

	if !reflect.DeepEqual(outPlain.Messages, outStream.Messages) {
		t.Errorf("Messages differ:\nplain=%+v\nstream=%+v", outPlain.Messages, outStream.Messages)
	}
	if !reflect.DeepEqual(eventsWithoutTimes(outPlain.Transcript), eventsWithoutTimes(outStream.Transcript)) {
		t.Errorf("transcripts differ:\nplain=%+v\nstream=%+v",
			eventsWithoutTimes(outPlain.Transcript), eventsWithoutTimes(outStream.Transcript))
	}
	if outPlain.FinalText != outStream.FinalText {
		t.Errorf("FinalText: plain %q, stream %q", outPlain.FinalText, outStream.FinalText)
	}
	if outPlain.Usage != outStream.Usage {
		t.Errorf("Usage: plain %+v, stream %+v", outPlain.Usage, outStream.Usage)
	}
	if outPlain.Iterations != outStream.Iterations {
		t.Errorf("Iterations: plain %d, stream %d", outPlain.Iterations, outStream.Iterations)
	}
	if outPlain.LastStopReason != outStream.LastStopReason {
		t.Errorf("LastStopReason: plain %q, stream %q", outPlain.LastStopReason, outStream.LastStopReason)
	}
}

// eventsWithoutTimes returns the transcript's events with timestamps zeroed,
// so two runs of the same script compare DeepEqual.
func eventsWithoutTimes(tr *Transcript) []Event {
	evs := slices.Clone(tr.Events)
	for i := range evs {
		evs[i].Time = time.Time{}
	}
	return evs
}

// TestDelta_SynthesizedFromCompleteOnlyClient pins the fallback: a client
// that implements only Complete still feeds the hook — llmkit.Stream calls
// Complete and synthesizes one delta per content block, so a non-streaming
// client is never a special case in the Runner.
func TestDelta_SynthesizedFromCompleteOnlyClient(t *testing.T) {
	// textResp leaves Blocks nil, so build the step with a text block: the
	// fallback synthesizes deltas from Blocks, not from the Text field.
	fc := newFakeClient(scriptStep{resp: llmkit.Response{
		Text:       "synthesized body",
		Blocks:     []llmkit.Block{llmkit.Text("synthesized body")},
		StopReason: llmkit.StopEndTurn,
		Usage:      llmkit.Usage{InputTokens: 5, OutputTokens: 5},
	}})
	rec := &deltaRecorder{}
	r := NewRunner(fc, nil, "sys", WithHooks(rec.hooks()))

	out, err := r.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := []deltaHit{{1, llmkit.Delta{Kind: llmkit.DeltaText, Text: "synthesized body"}}}
	if !reflect.DeepEqual(rec.hits, want) {
		t.Errorf("deltas = %+v, want %+v", rec.hits, want)
	}
	if out.FinalText != "synthesized body" {
		t.Errorf("FinalText = %q, want %q", out.FinalText, "synthesized body")
	}
}

// TestDelta_HookPanicPropagates pins that a panicking Delta hook is a harness
// bug like any other hook panic: it propagates out of Run with the ORIGINAL
// value — nothing recovers it inside Stream — and the run aborts instead of
// rendering the turn.
func TestDelta_HookPanicPropagates(t *testing.T) {
	sf := &streamFakeClient{fakeClient: newFakeClient(
		toolResp("c1", "echo", `{}`, 10, 4),
		textResp("never reached", 5, 2),
	)}
	hooks := Hooks{Delta: func(_ context.Context, _ int, _ llmkit.Delta) {
		panic("delta hook exploded")
	}}
	r := NewRunner(sf, []Tool{echoTool{name: "echo"}}, "sys", WithHooks(hooks))

	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		_, _ = r.Run(context.Background(), "task")
		done <- nil
	}()
	select {
	case v := <-done:
		if v != "delta hook exploded" {
			t.Fatalf("Run panic value = %v, want the original hook panic value", v)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not panic on a Delta hook panic")
	}
	streams, completes := sf.calls()
	if streams != 1 || completes != 0 {
		t.Errorf("calls after panic: Stream=%d Complete=%d, want 1 and 0 (the panic must abort the run mid-stream)", streams, completes)
	}
}

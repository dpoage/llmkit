package provider

// Stream-path retry/serialize tests exercise Wrap's stages over
// StreamingClient (and the Complete-only fallback) through llmkit.Stream.

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// scriptedStreamClient implements StreamingClient with a canned delta
// sequence, final response, and optional terminal error: it wraps a
// non-nil fn error, never routes through Complete, and can stall its
// leading calls until their context is done.
type scriptedStreamClient struct {
	caps   llmkit.Capabilities
	deltas []llmkit.Delta
	// blockAttempts is the number of leading Stream calls that deliver
	// deltas and then stall until their context is done; later calls return
	// resp/err immediately.
	blockAttempts int
	resp          llmkit.Response
	err           error
	calls         int
	completes     int
}

func (s *scriptedStreamClient) Capabilities() llmkit.Capabilities { return s.caps }

func (s *scriptedStreamClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	s.completes++
	return llmkit.Response{}, errors.New("scriptedStreamClient: unexpected Complete call")
}

func (s *scriptedStreamClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	s.calls++
	if fn != nil {
		for _, d := range s.deltas {
			if err := fn(d); err != nil {
				return llmkit.Response{}, err
			}
		}
	}
	if s.calls <= s.blockAttempts {
		<-ctx.Done()
		return llmkit.Response{}, &llmkit.APIError{
			Kind: llmkit.ErrServer, Provider: "fake", Message: ctx.Err().Error(), Err: ctx.Err(),
		}
	}
	if s.err != nil {
		return llmkit.Response{}, s.err
	}
	return s.resp, nil
}

// countedPolicy returns a fast retry.Config whose backoff sleeps are
// counted, so tests observe retry behavior without real waiting.
func countedPolicy(maxAttempts int, timeout time.Duration, sleeps *int) retry.Config {
	return retry.Config{
		MaxAttempts:    maxAttempts,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: timeout,
		Sleep:          func(context.Context, time.Duration) error { *sleeps++; return nil },
	}
}

func TestWrap_RetryStreamSucceedsAfterServerErrorBeforeFirstDelta(t *testing.T) {
	ok := llmkit.Response{Blocks: []llmkit.Block{llmkit.Text("recovered")}, Text: "recovered", StopReason: llmkit.StopEndTurn}
	inner := &fakeClient{
		errs:      []error{&llmkit.APIError{Kind: llmkit.ErrServer, StatusCode: 500, Provider: "fake", Message: "500"}},
		responses: []llmkit.Response{{}, ok},
	}
	sleeps := 0
	c := Wrap(inner, Options{Retry: countedPolicy(3, time.Second, &sleeps)})
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if inner.calls != 2 {
		t.Fatalf("attempts = %d, want 2", inner.calls)
	}
	if sleeps != 1 {
		t.Fatalf("backoff sleeps = %d, want 1", sleeps)
	}
	if !reflect.DeepEqual(out, ok) {
		t.Fatalf("Response = %+v, want %+v", out, ok)
	}
	want := []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "recovered"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
}

func TestWrap_RetryStreamErrorAfterDeliveredDeltaReturnsUnretried(t *testing.T) {
	mid := &llmkit.APIError{Kind: llmkit.ErrServer, StatusCode: 500, Provider: "fake", Message: "mid-stream drop"}
	inner := &scriptedStreamClient{
		deltas: []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "partial"}},
		err:    mid,
	}
	sleeps := 0
	c := Wrap(inner, Options{Retry: countedPolicy(4, time.Second, &sleeps)})
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || apiErr != mid {
		t.Fatalf("err = %v, want the mid-stream error returned unretried and unwrapped", err)
	}
	if inner.calls != 1 {
		t.Fatalf("attempts = %d, want 1: a delivered delta forbids retry", inner.calls)
	}
	if sleeps != 0 {
		t.Fatalf("backoff sleeps = %d, want 0", sleeps)
	}
	want := []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "partial"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(out, llmkit.Response{}) {
		t.Fatalf("Response = %+v, want zero", out)
	}
}

func TestWrap_RetryStreamRequestTimeoutBoundsStalledStreamAfterDelta(t *testing.T) {
	inner := &scriptedStreamClient{
		deltas:        []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "partial"}},
		blockAttempts: 99,
	}
	sleeps := 0
	c := Wrap(inner, Options{Retry: countedPolicy(3, 50*time.Millisecond, &sleeps)})
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded from the attempt deadline", err)
	}
	if inner.calls != 1 {
		t.Fatalf("attempts = %d, want 1: the delivered delta forbids retry", inner.calls)
	}
	if sleeps != 0 {
		t.Fatalf("backoff sleeps = %d, want 0", sleeps)
	}
	want := []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "partial"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(out, llmkit.Response{}) {
		t.Fatalf("Response = %+v, want zero", out)
	}
}

func TestWrap_RetryStreamTimeoutBeforeFirstDeltaIsRetried(t *testing.T) {
	ok := llmkit.Response{Text: "recovered", StopReason: llmkit.StopEndTurn}
	inner := &scriptedStreamClient{blockAttempts: 1, resp: ok}
	sleeps := 0
	c := Wrap(inner, Options{Retry: countedPolicy(3, 50*time.Millisecond, &sleeps)})
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if inner.calls != 2 {
		t.Fatalf("attempts = %d, want 2: a stalled attempt before any delta is retried", inner.calls)
	}
	if sleeps != 1 {
		t.Fatalf("backoff sleeps = %d, want 1", sleeps)
	}
	if !reflect.DeepEqual(out, ok) {
		t.Fatalf("Response = %+v, want %+v", out, ok)
	}
	if len(got) != 0 {
		t.Fatalf("deltas = %+v, want none", got)
	}
}

// TestWrap_RetryStreamFnErrorOnFirstDeltaReturnsUnretried pins the
// delivered flag against fn errors: an unclassified fn error would look
// retryable and replay the stream the caller already began consuming.
func TestWrap_RetryStreamFnErrorOnFirstDeltaReturnsUnretried(t *testing.T) {
	sentinel := errors.New("stop after first fragment")
	inner := &scriptedStreamClient{
		deltas: []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "first"}},
		err:    &llmkit.APIError{Kind: llmkit.ErrServer, StatusCode: 500, Provider: "fake", Message: "drop"},
	}
	sleeps := 0
	c := Wrap(inner, Options{Retry: countedPolicy(4, time.Second, &sleeps)})
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want it to wrap the fn error", err)
	}
	if inner.calls != 1 {
		t.Fatalf("attempts = %d, want 1: the fn error on the first delta is terminal", inner.calls)
	}
	if sleeps != 0 {
		t.Fatalf("backoff sleeps = %d, want 0", sleeps)
	}
	if len(got) != 1 {
		t.Fatalf("delivered %d deltas, want 1", len(got))
	}
	if !reflect.DeepEqual(out, llmkit.Response{}) {
		t.Fatalf("Response = %+v, want zero", out)
	}
}

func TestSerializeStage_StreamDropsToolCallsBeyondFirst(t *testing.T) {
	inner := &scriptedStreamClient{
		deltas: []llmkit.Delta{
			{Kind: llmkit.DeltaText, Text: "a"},
			{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{"x":`},
			{Kind: llmkit.DeltaToolCall, Index: 0, Arguments: `1}`},
			{Kind: llmkit.DeltaToolCall, Index: 1, ID: "call_2", Name: "g", Arguments: `{"y":2}`},
		},
		resp: llmkit.Response{
			Text: "a",
			ToolCalls: []llmkit.ToolCall{
				{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"x":1}`)},
				{ID: "call_2", Name: "g", Arguments: json.RawMessage(`{"y":2}`)},
			},
			StopReason: llmkit.StopToolUse,
		},
	}
	c := serializeStage(inner)
	want := []llmkit.Delta{
		{Kind: llmkit.DeltaText, Text: "a"},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{"x":`},
		{Kind: llmkit.DeltaToolCall, Index: 0, Arguments: `1}`},
	}
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v, want only call_1", out.ToolCalls)
	}
	if out.StopReason != llmkit.StopToolUse {
		t.Fatalf("StopReason = %q, want %q", out.StopReason, llmkit.StopToolUse)
	}
}

func TestSerializeStage_StreamFromCompleteOnlyClient(t *testing.T) {
	inner := &fakeClient{responses: []llmkit.Response{{
		Blocks: []llmkit.Block{llmkit.Text("a")},
		Text:   "a",
		ToolCalls: []llmkit.ToolCall{
			{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)},
			{ID: "call_2", Name: "g", Arguments: json.RawMessage(`{}`)},
		},
		StopReason: llmkit.StopToolUse,
	}}}
	c := serializeStage(inner)
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []llmkit.Delta{
		{Kind: llmkit.DeltaText, Text: "a"},
		{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{}`},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v, want only call_1", out.ToolCalls)
	}
}

func TestSerializeStage_StreamPassesThroughParallelCapableClient(t *testing.T) {
	resp := llmkit.Response{
		ToolCalls: []llmkit.ToolCall{
			{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)},
			{ID: "call_2", Name: "g", Arguments: json.RawMessage(`{}`)},
		},
		StopReason: llmkit.StopToolUse,
	}
	inner := &scriptedStreamClient{
		caps: llmkit.Capabilities{ParallelToolCalls: true},
		deltas: []llmkit.Delta{
			{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{}`},
			{Kind: llmkit.DeltaToolCall, Index: 1, ID: "call_2", Name: "g", Arguments: `{}`},
		},
		resp: resp,
	}
	c := serializeStage(inner)
	var got []llmkit.Delta
	out, err := llmkit.Stream(context.Background(), c, simpleRequest(), func(d llmkit.Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("delivered %d deltas, want both tool calls' fragments", len(got))
	}
	if !reflect.DeepEqual(out, resp) {
		t.Fatalf("Response = %+v, want untruncated %+v", out, resp)
	}
}

// TestWrap_NilFnThroughDecorators routes a nil fn through both stages over
// both inner kinds: without a guard, the wrappers would panic on the first
// fragment instead of returning Complete's response.
func TestWrap_NilFnThroughDecorators(t *testing.T) {
	completeResp := llmkit.Response{
		Blocks:     []llmkit.Block{llmkit.Text("once"), {Kind: llmkit.BlockThinking, Text: "hmm"}},
		Text:       "oncehmm",
		ToolCalls:  []llmkit.ToolCall{{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)}},
		StopReason: llmkit.StopToolUse,
	}
	nativeResp := llmkit.Response{Text: "native", StopReason: llmkit.StopEndTurn}

	t.Run("retry", func(t *testing.T) {
		t.Run("complete only", func(t *testing.T) {
			inner := &fakeClient{responses: []llmkit.Response{completeResp}}
			sleeps := 0
			c := Wrap(inner, Options{Retry: countedPolicy(3, time.Second, &sleeps)})
			out, err := llmkit.Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if !reflect.DeepEqual(out, completeResp) {
				t.Fatalf("Response = %+v, want Complete's %+v", out, completeResp)
			}
		})
		t.Run("native", func(t *testing.T) {
			inner := &scriptedStreamClient{
				deltas: []llmkit.Delta{{Kind: llmkit.DeltaText, Text: "native"}},
				resp:   nativeResp,
			}
			sleeps := 0
			c := Wrap(inner, Options{Retry: countedPolicy(3, time.Second, &sleeps)})
			out, err := llmkit.Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if inner.completes != 0 {
				t.Fatalf("Complete calls = %d, want 0", inner.completes)
			}
			if !reflect.DeepEqual(out, nativeResp) {
				t.Fatalf("Response = %+v, want %+v", out, nativeResp)
			}
		})
	})

	t.Run("serialize", func(t *testing.T) {
		t.Run("complete only", func(t *testing.T) {
			inner := &fakeClient{responses: []llmkit.Response{completeResp}}
			c := serializeStage(inner)
			out, err := llmkit.Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			want := completeResp
			want.ToolCalls = completeResp.ToolCalls[:1]
			if !reflect.DeepEqual(out, want) {
				t.Fatalf("Response = %+v, want truncated %+v", out, want)
			}
		})
		t.Run("native", func(t *testing.T) {
			inner := &scriptedStreamClient{
				deltas: []llmkit.Delta{
					{Kind: llmkit.DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{}`},
					{Kind: llmkit.DeltaToolCall, Index: 1, ID: "call_2", Name: "g", Arguments: `{}`},
				},
				resp: nativeResp,
			}
			c := serializeStage(inner)
			out, err := llmkit.Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if !reflect.DeepEqual(out, nativeResp) {
				t.Fatalf("Response = %+v, want %+v", out, nativeResp)
			}
		})
	})
}

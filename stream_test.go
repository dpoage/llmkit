package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"
)

// scriptedStreamClient implements StreamingClient with a canned delta
// sequence, final response, and optional terminal error: it wraps a
// non-nil fn error, never routes through Complete, and can stall its
// leading calls until their context is done.
type scriptedStreamClient struct {
	caps   Capabilities
	deltas []Delta
	// blockAttempts is the number of leading Stream calls that deliver
	// deltas and then stall until their context is done; later calls return
	// resp/err immediately.
	blockAttempts int
	resp          Response
	err           error
	calls         int
	completes     int
}

func (s *scriptedStreamClient) Capabilities() Capabilities { return s.caps }

func (s *scriptedStreamClient) Complete(ctx context.Context, req Request) (Response, error) {
	s.completes++
	return Response{}, errors.New("scriptedStreamClient: unexpected Complete call")
}

func (s *scriptedStreamClient) Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error) {
	s.calls++
	if fn != nil {
		for _, d := range s.deltas {
			if err := fn(d); err != nil {
				return Response{}, fmt.Errorf("llmkit: stream fn: %w", err)
			}
		}
	}
	if s.calls <= s.blockAttempts {
		<-ctx.Done()
		return Response{}, ctx.Err()
	}
	if s.err != nil {
		return Response{}, s.err
	}
	return s.resp, nil
}

// countedRetryConfig returns a fast RetryConfig whose backoff sleeps are
// counted, so tests observe retry behavior without real waiting.
func countedRetryConfig(maxAttempts int, timeout time.Duration, sleeps *int) RetryConfig {
	return RetryConfig{
		MaxAttempts:    maxAttempts,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: timeout,
		sleep:          func(context.Context, time.Duration) error { *sleeps++; return nil },
	}
}

func TestStream_SynthesizesFromCompleteOnlyClient(t *testing.T) {
	inner := &fakeClient{responses: []Response{{
		Blocks: []Block{
			Text("hello "),
			{Kind: BlockThinking, Text: "pondering"},
			Text("world"),
			Image("image/png", []byte{1}),
		},
		ToolCalls: []ToolCall{
			{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"a"}`)},
			{ID: "call_2", Name: "lookup", Arguments: json.RawMessage(`{"q":"b"}`)},
		},
		Usage:      Usage{InputTokens: 10, OutputTokens: 5},
		StopReason: StopToolUse,
	}}}
	var got []Delta
	out, err := Stream(context.Background(), inner, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if inner.calls != 1 {
		t.Fatalf("Complete calls = %d, want 1", inner.calls)
	}
	// Text/thinking in block order (other block kinds carry no delta), then
	// tool calls in order with their full arguments.
	want := []Delta{
		{Kind: DeltaText, Text: "hello "},
		{Kind: DeltaThinking, Text: "pondering"},
		{Kind: DeltaText, Text: "world"},
		{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "lookup", Arguments: `{"q":"a"}`},
		{Kind: DeltaToolCall, Index: 1, ID: "call_2", Name: "lookup", Arguments: `{"q":"b"}`},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(out, inner.responses[0]) {
		t.Fatalf("Response = %+v, want the Complete response %+v", out, inner.responses[0])
	}
	// The response is the one Complete returned, not a rebuild.
	if &out.ToolCalls[0] != &inner.responses[0].ToolCalls[0] {
		t.Fatal("Response.ToolCalls was copied; Stream must return Complete's response unchanged")
	}
}

func TestStream_PrefersStreamingClient(t *testing.T) {
	want := Response{Text: "streamed", Usage: Usage{OutputTokens: 3}, StopReason: StopEndTurn}
	inner := &scriptedStreamClient{
		deltas: []Delta{{Kind: DeltaText, Text: "streamed"}},
		resp:   want,
	}
	var got []Delta
	out, err := Stream(context.Background(), inner, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if inner.completes != 0 {
		t.Fatalf("Complete calls = %d, want 0: Stream must be preferred", inner.completes)
	}
	if inner.calls != 1 {
		t.Fatalf("Stream calls = %d, want 1", inner.calls)
	}
	if !reflect.DeepEqual(got, []Delta{{Kind: DeltaText, Text: "streamed"}}) {
		t.Fatalf("deltas = %+v, want the scripted fragment", got)
	}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("Response = %+v, want %+v", out, want)
	}
}

func TestStream_NilFnEqualsComplete(t *testing.T) {
	t.Run("complete only", func(t *testing.T) {
		want := Response{Text: "all at once", StopReason: StopEndTurn}
		inner := &fakeClient{responses: []Response{want}}
		out, err := Stream(context.Background(), inner, simpleRequest(), nil)
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if inner.calls != 1 {
			t.Fatalf("Complete calls = %d, want 1", inner.calls)
		}
		if !reflect.DeepEqual(out, want) {
			t.Fatalf("Response = %+v, want %+v", out, want)
		}
	})
	t.Run("streaming client", func(t *testing.T) {
		want := Response{Text: "streamed", StopReason: StopEndTurn}
		inner := &scriptedStreamClient{
			deltas: []Delta{{Kind: DeltaText, Text: "streamed"}},
			resp:   want,
		}
		out, err := Stream(context.Background(), inner, simpleRequest(), nil)
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if inner.calls != 1 || inner.completes != 0 {
			t.Fatalf("Stream calls = %d, Complete calls = %d, want 1 and 0", inner.calls, inner.completes)
		}
		if !reflect.DeepEqual(out, want) {
			t.Fatalf("Response = %+v, want %+v", out, want)
		}
	})
}

func TestStream_FnErrorCancelsAndWraps(t *testing.T) {
	sentinel := errors.New("stop streaming")
	t.Run("synthesized", func(t *testing.T) {
		inner := &fakeClient{responses: []Response{{
			Blocks:    []Block{Text("one"), {Kind: BlockThinking, Text: "two"}, Text("three")},
			ToolCalls: []ToolCall{{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)}},
		}}}
		var got []Delta
		out, err := Stream(context.Background(), inner, simpleRequest(), func(d Delta) error {
			got = append(got, d)
			if d.Kind == DeltaThinking {
				return sentinel
			}
			return nil
		})
		if !errors.Is(err, sentinel) {
			t.Fatalf("err = %v, want it to wrap the fn error", err)
		}
		if !reflect.DeepEqual(out, Response{}) {
			t.Fatalf("Response = %+v, want the zero Response on fn error", out)
		}
		// Delivery stops at the failing fragment; the rest never arrives.
		if len(got) != 2 {
			t.Fatalf("delivered %d deltas, want 2", len(got))
		}
	})
	t.Run("streaming client", func(t *testing.T) {
		inner := &scriptedStreamClient{
			deltas: []Delta{
				{Kind: DeltaText, Text: "one"},
				{Kind: DeltaText, Text: "two"},
				{Kind: DeltaText, Text: "three"},
			},
			resp: Response{Text: "onetwothree"},
		}
		var got []Delta
		out, err := Stream(context.Background(), inner, simpleRequest(), func(d Delta) error {
			got = append(got, d)
			if len(got) == 2 {
				return sentinel
			}
			return nil
		})
		if !errors.Is(err, sentinel) {
			t.Fatalf("err = %v, want it to wrap the fn error", err)
		}
		if !reflect.DeepEqual(out, Response{}) {
			t.Fatalf("Response = %+v, want the zero Response on fn error", out)
		}
		if len(got) != 2 {
			t.Fatalf("delivered %d deltas, want 2", len(got))
		}
	})
}

func TestRetryStream_SucceedsAfterServerErrorBeforeFirstDelta(t *testing.T) {
	ok := Response{Blocks: []Block{Text("recovered")}, Text: "recovered", StopReason: StopEndTurn}
	inner := &fakeClient{
		errs:      []error{&APIError{Kind: ErrServer, StatusCode: 500, Provider: "fake", Message: "500"}},
		responses: []Response{{}, ok},
	}
	sleeps := 0
	c := WithRetry(inner, countedRetryConfig(3, time.Second, &sleeps))
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
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
	want := []Delta{{Kind: DeltaText, Text: "recovered"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
}

func TestRetryStream_ErrorAfterDeliveredDeltaReturnsUnretried(t *testing.T) {
	mid := &APIError{Kind: ErrServer, StatusCode: 500, Provider: "fake", Message: "mid-stream drop"}
	inner := &scriptedStreamClient{
		deltas: []Delta{{Kind: DeltaText, Text: "partial"}},
		err:    mid,
	}
	sleeps := 0
	c := WithRetry(inner, countedRetryConfig(4, time.Second, &sleeps))
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr != mid {
		t.Fatalf("err = %v, want the mid-stream error returned unretried and unwrapped", err)
	}
	if inner.calls != 1 {
		t.Fatalf("attempts = %d, want 1: a delivered delta forbids retry", inner.calls)
	}
	if sleeps != 0 {
		t.Fatalf("backoff sleeps = %d, want 0", sleeps)
	}
	want := []Delta{{Kind: DeltaText, Text: "partial"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(out, Response{}) {
		t.Fatalf("Response = %+v, want zero", out)
	}
}

func TestRetryStream_RequestTimeoutBoundsStalledStreamAfterDelta(t *testing.T) {
	inner := &scriptedStreamClient{
		deltas:        []Delta{{Kind: DeltaText, Text: "partial"}},
		blockAttempts: 99,
	}
	sleeps := 0
	c := WithRetry(inner, countedRetryConfig(3, 50*time.Millisecond, &sleeps))
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
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
	want := []Delta{{Kind: DeltaText, Text: "partial"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(out, Response{}) {
		t.Fatalf("Response = %+v, want zero", out)
	}
}

func TestRetryStream_TimeoutBeforeFirstDeltaIsRetried(t *testing.T) {
	ok := Response{Text: "recovered", StopReason: StopEndTurn}
	inner := &scriptedStreamClient{blockAttempts: 1, resp: ok}
	sleeps := 0
	c := WithRetry(inner, countedRetryConfig(3, 50*time.Millisecond, &sleeps))
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
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

func TestRecorderStream_RecordsFinalUsage(t *testing.T) {
	want := Response{Text: "done", Usage: Usage{InputTokens: 7, OutputTokens: 2}, StopReason: StopEndTurn}
	inner := &fakeClient{responses: []Response{want}}
	var events []UsageEvent
	c := WithRecorder(inner, RecorderFunc(func(ev UsageEvent) { events = append(events, ev) }), "fake", "m-1")
	out, err := Stream(context.Background(), c, simpleRequest(), func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("records = %d, want 1", len(events))
	}
	if events[0] != (UsageEvent{Provider: "fake", Model: "m-1", Usage: want.Usage}) {
		t.Fatalf("event = %+v, want usage %+v tagged fake/m-1", events[0], want.Usage)
	}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("Response = %+v, want %+v", out, want)
	}
}

func TestRecorderStream_NoRecordOnStreamError(t *testing.T) {
	inner := &fakeClient{errs: []error{errors.New("boom")}}
	records := 0
	c := WithRecorder(inner, RecorderFunc(func(UsageEvent) { records++ }), "fake", "m-1")
	if _, err := Stream(context.Background(), c, simpleRequest(), nil); err == nil {
		t.Fatal("Stream: want error")
	}
	if records != 0 {
		t.Fatalf("records = %d, want 0", records)
	}
}

func TestSerializeStream_DropsToolCallsBeyondFirst(t *testing.T) {
	inner := &scriptedStreamClient{
		deltas: []Delta{
			{Kind: DeltaText, Text: "a"},
			{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{"x":`},
			{Kind: DeltaToolCall, Index: 0, Arguments: `1}`},
			{Kind: DeltaToolCall, Index: 1, ID: "call_2", Name: "g", Arguments: `{"y":2}`},
		},
		resp: Response{
			Text: "a",
			ToolCalls: []ToolCall{
				{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"x":1}`)},
				{ID: "call_2", Name: "g", Arguments: json.RawMessage(`{"y":2}`)},
			},
			StopReason: StopToolUse,
		},
	}
	c := WithSerializedToolCalls(inner)
	want := []Delta{
		{Kind: DeltaText, Text: "a"},
		{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{"x":`},
		{Kind: DeltaToolCall, Index: 0, Arguments: `1}`},
	}
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
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
	if out.StopReason != StopToolUse {
		t.Fatalf("StopReason = %q, want %q", out.StopReason, StopToolUse)
	}
}

func TestSerializeStream_FromCompleteOnlyClient(t *testing.T) {
	inner := &fakeClient{responses: []Response{{
		Blocks: []Block{Text("a")},
		Text:   "a",
		ToolCalls: []ToolCall{
			{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)},
			{ID: "call_2", Name: "g", Arguments: json.RawMessage(`{}`)},
		},
		StopReason: StopToolUse,
	}}}
	c := WithSerializedToolCalls(inner)
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []Delta{
		{Kind: DeltaText, Text: "a"},
		{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{}`},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v, want only call_1", out.ToolCalls)
	}
}

func TestSerializeStream_PassesThroughParallelCapableClient(t *testing.T) {
	resp := Response{
		ToolCalls: []ToolCall{
			{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)},
			{ID: "call_2", Name: "g", Arguments: json.RawMessage(`{}`)},
		},
		StopReason: StopToolUse,
	}
	inner := &scriptedStreamClient{
		caps: Capabilities{ParallelToolCalls: true},
		deltas: []Delta{
			{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{}`},
			{Kind: DeltaToolCall, Index: 1, ID: "call_2", Name: "g", Arguments: `{}`},
		},
		resp: resp,
	}
	c := WithSerializedToolCalls(inner)
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
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

// TestStream_NilFnThroughDecorators routes a nil fn through every
// decorator over both inner kinds: without a guard, the wrappers
// would panic on the first fragment instead of returning Complete's
// response.
func TestStream_NilFnThroughDecorators(t *testing.T) {
	completeResp := Response{
		Blocks:     []Block{Text("once"), {Kind: BlockThinking, Text: "hmm"}},
		Text:       "oncehmm",
		ToolCalls:  []ToolCall{{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`)}},
		StopReason: StopToolUse,
	}
	nativeResp := Response{Text: "native", StopReason: StopEndTurn}

	t.Run("retry", func(t *testing.T) {
		t.Run("complete only", func(t *testing.T) {
			inner := &fakeClient{responses: []Response{completeResp}}
			sleeps := 0
			c := WithRetry(inner, countedRetryConfig(3, time.Second, &sleeps))
			out, err := Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if !reflect.DeepEqual(out, completeResp) {
				t.Fatalf("Response = %+v, want Complete's %+v", out, completeResp)
			}
		})
		t.Run("native", func(t *testing.T) {
			inner := &scriptedStreamClient{
				deltas: []Delta{{Kind: DeltaText, Text: "native"}},
				resp:   nativeResp,
			}
			sleeps := 0
			c := WithRetry(inner, countedRetryConfig(3, time.Second, &sleeps))
			out, err := Stream(context.Background(), c, simpleRequest(), nil)
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

	t.Run("recorder", func(t *testing.T) {
		t.Run("complete only", func(t *testing.T) {
			inner := &fakeClient{responses: []Response{completeResp}}
			records := 0
			c := WithRecorder(inner, RecorderFunc(func(UsageEvent) { records++ }), "fake", "m-1")
			out, err := Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if records != 1 {
				t.Fatalf("records = %d, want 1", records)
			}
			if !reflect.DeepEqual(out, completeResp) {
				t.Fatalf("Response = %+v, want Complete's %+v", out, completeResp)
			}
		})
		t.Run("native", func(t *testing.T) {
			inner := &scriptedStreamClient{resp: nativeResp}
			records := 0
			c := WithRecorder(inner, RecorderFunc(func(UsageEvent) { records++ }), "fake", "m-1")
			out, err := Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if records != 1 {
				t.Fatalf("records = %d, want 1", records)
			}
			if !reflect.DeepEqual(out, nativeResp) {
				t.Fatalf("Response = %+v, want %+v", out, nativeResp)
			}
		})
	})

	t.Run("serialize", func(t *testing.T) {
		t.Run("complete only", func(t *testing.T) {
			inner := &fakeClient{responses: []Response{completeResp}}
			c := WithSerializedToolCalls(inner)
			out, err := Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			// One tool call, the way this decorator's Complete truncates.
			want := completeResp
			want.ToolCalls = completeResp.ToolCalls[:1]
			if !reflect.DeepEqual(out, want) {
				t.Fatalf("Response = %+v, want truncated %+v", out, want)
			}
		})
		t.Run("native", func(t *testing.T) {
			inner := &scriptedStreamClient{
				deltas: []Delta{
					{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "f", Arguments: `{}`},
					{Kind: DeltaToolCall, Index: 1, ID: "call_2", Name: "g", Arguments: `{}`},
				},
				resp: nativeResp,
			}
			c := WithSerializedToolCalls(inner)
			out, err := Stream(context.Background(), c, simpleRequest(), nil)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if !reflect.DeepEqual(out, nativeResp) {
				t.Fatalf("Response = %+v, want %+v", out, nativeResp)
			}
		})
	})
}

// TestRetryStream_FnErrorOnFirstDeltaReturnsUnretried pins the delivered
// flag against fn errors: an unclassified fn error would look retryable
// and replay the stream the caller already began consuming.
func TestRetryStream_FnErrorOnFirstDeltaReturnsUnretried(t *testing.T) {
	sentinel := errors.New("stop after first fragment")
	inner := &scriptedStreamClient{
		deltas: []Delta{{Kind: DeltaText, Text: "first"}},
		err:    &APIError{Kind: ErrServer, StatusCode: 500, Provider: "fake", Message: "drop"},
	}
	sleeps := 0
	c := WithRetry(inner, countedRetryConfig(4, time.Second, &sleeps))
	var got []Delta
	out, err := Stream(context.Background(), c, simpleRequest(), func(d Delta) error {
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
	if !reflect.DeepEqual(out, Response{}) {
		t.Fatalf("Response = %+v, want zero", out)
	}
}

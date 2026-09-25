package llmkit

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

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
	out, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
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

// TestStream_TextOnlyResponseEmitsOneDelta pins the synthesizer fallback: a
// Complete-only client may return Text set with Blocks empty (the Response
// contract permits it; shipped adapters always populate Blocks, but custom
// Clients need not). Stream must emit exactly one DeltaText with the full
// Text.
func TestStream_TextOnlyResponseEmitsOneDelta(t *testing.T) {
	inner := &fakeClient{responses: []Response{{
		Text:       "just text, no blocks",
		Usage:      Usage{InputTokens: 4, OutputTokens: 6},
		StopReason: StopEndTurn,
	}}}
	var got []Delta
	out, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []Delta{{Kind: DeltaText, Text: "just text, no blocks"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(out, inner.responses[0]) {
		t.Fatalf("Response = %+v, want the Complete response %+v", out, inner.responses[0])
	}
}

// TestStream_TextFallbackNotFiredWhenBlocksCarryText pins the other half of
// the fallback: when Blocks already contain a text block, Text must NOT be
// emitted a second time (no double emission).
func TestStream_TextFallbackNotFiredWhenBlocksCarryText(t *testing.T) {
	inner := &fakeClient{responses: []Response{{
		Text: "hello world",
		Blocks: []Block{
			Text("hello "),
			Text("world"),
			{Kind: BlockThinking, Text: "hmm"},
		},
	}}}
	var got []Delta
	if _, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
		got = append(got, d)
		return nil
	}); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	want := []Delta{
		{Kind: DeltaText, Text: "hello "},
		{Kind: DeltaText, Text: "world"},
		{Kind: DeltaThinking, Text: "hmm"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deltas = %+v, want %+v", got, want)
	}
}

// TestStream_TextFallbackOrderingAndEmpty pins the fallback's edges: a
// thinking-only Blocks list with Text set emits the text delta after the
// thinking delta; ToolCalls without Blocks put the text delta first; and a
// response with no Text and no Blocks emits zero deltas (never an empty
// DeltaText).
func TestStream_TextFallbackOrderingAndEmpty(t *testing.T) {
	tests := []struct {
		name string
		resp Response
		want []Delta
	}{
		{
			name: "thinking blocks then text fallback",
			resp: Response{
				Text: "afterthought",
				Blocks: []Block{
					{Kind: BlockThinking, Text: "hmm"},
					{Kind: BlockThinking, Text: "hah"},
				},
			},
			want: []Delta{
				{Kind: DeltaThinking, Text: "hmm"},
				{Kind: DeltaThinking, Text: "hah"},
				{Kind: DeltaText, Text: "afterthought"},
			},
		},
		{
			name: "text fallback precedes tool calls",
			resp: Response{
				Text: "look this up",
				ToolCalls: []ToolCall{
					{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"a"}`)},
				},
			},
			want: []Delta{
				{Kind: DeltaText, Text: "look this up"},
				{Kind: DeltaToolCall, Index: 0, ID: "call_1", Name: "lookup", Arguments: `{"q":"a"}`},
			},
		},
		{
			name: "empty text and blocks emit nothing",
			resp: Response{Usage: Usage{InputTokens: 1}},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &fakeClient{responses: []Response{tt.resp}}
			var got []Delta
			if _, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
				got = append(got, d)
				return nil
			}); err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("deltas = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// TestStream_TextFallbackFnError pins fn-error propagation on the fallback
// delta: the error wraps as "llmkit: stream fn: ..." and Stream returns a
// zero Response (the Complete result is dropped).
func TestStream_TextFallbackFnError(t *testing.T) {
	inner := &fakeClient{responses: []Response{{
		Text:       "doomed",
		StopReason: StopEndTurn,
	}}}
	sentinel := errors.New("stop")
	out, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
		return sentinel
	})
	if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), "llmkit: stream fn: ") {
		t.Fatalf("err = %v, want wrapped llmkit: stream fn error", err)
	}
	if !reflect.DeepEqual(out, Response{}) {
		t.Fatalf("Response = %+v, want zero Response on fn error", out)
	}
}

func TestStream_PrefersStreamingClient(t *testing.T) {
	want := Response{Text: "streamed", Usage: Usage{OutputTokens: 3}, StopReason: StopEndTurn}
	inner := &scriptedStreamClient{
		deltas: []Delta{{Kind: DeltaText, Text: "streamed"}},
		resp:   want,
	}
	var got []Delta
	out, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
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
		out, err := Stream(t.Context(), inner, simpleRequest(), nil)
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
		out, err := Stream(t.Context(), inner, simpleRequest(), nil)
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
		out, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
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
		out, err := Stream(t.Context(), inner, simpleRequest(), func(d Delta) error {
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

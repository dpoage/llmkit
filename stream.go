package llmkit

import (
	"context"
	"fmt"
)

// DeltaKind distinguishes the fragment kinds a streaming completion delivers.
type DeltaKind string

const (
	// DeltaText: Text holds a text fragment of the assistant reply.
	DeltaText DeltaKind = "text"
	// DeltaThinking: Text holds a thinking fragment.
	DeltaThinking DeltaKind = "thinking"
	// DeltaToolCall: one tool call's fragment. Index is the 0-based position
	// of the call in [Response.ToolCalls] — adapters renumber vendor wire
	// indices to this; ID and Name are set on the first fragment for that
	// Index (and may repeat); Arguments holds a raw JSON fragment to
	// concatenate with the other fragments of the same Index.
	DeltaToolCall DeltaKind = "tool_call"
)

// Delta is one incremental fragment of a streaming completion.
type Delta struct {
	// Kind selects which fields carry the payload.
	Kind DeltaKind
	// Text holds the fragment for DeltaText and DeltaThinking.
	Text string
	// Index is the 0-based position of the call in Response.ToolCalls
	// (DeltaToolCall); adapters renumber vendor wire indices to this.
	Index int
	// ID and Name carry the tool call's identity, set on its first fragment.
	ID, Name string
	// Arguments holds a raw JSON fragment to concatenate (DeltaToolCall).
	Arguments string
}

// StreamingClient is implemented by clients that can deliver a completion
// incrementally. A Client that cannot still works with [Stream], which
// synthesizes deltas from a plain Complete.
type StreamingClient interface {
	Client

	// Stream is Complete with incremental delivery: fn is called on the
	// calling goroutine, in wire order, once per fragment; a non-nil error
	// from fn cancels the stream and is returned (wrapped) — no Response.
	// fn execution time counts toward the per-attempt RequestTimeout a
	// wrapping WithRetry applies. The returned Response equals what Complete
	// returns for the same wire exchange: same normalization, Usage
	// convention, StopReason, thinking text and signatures, and ToolCalls
	// with concatenated Arguments. Block.Raw may differ in encoding (a
	// vendor accumulator re-serializes streamed blocks) but decodes to the
	// same content and re-emits the same wire block. A stream that ends
	// before the vendor's terminal event is an error wrapping ErrServer,
	// never a partial Response. A nil fn is allowed and makes Stream
	// equivalent to Complete.
	Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error)
}

// Stream uses c's Stream when c implements StreamingClient; otherwise it
// calls c.Complete and synthesizes deltas from the Response in block order
// (text blocks → DeltaText, thinking blocks → DeltaThinking, each ToolCall →
// one DeltaToolCall with ID/Name/Index and the full Arguments) before
// returning it. When Blocks carries no text block but Text is set, one
// DeltaText from Text is emitted after the blocks and before the tool-call
// deltas. fn runs on the calling goroutine, and its execution time
// counts toward the per-attempt RequestTimeout when Stream is wrapped in
// WithRetry.
func Stream(ctx context.Context, c Client, req Request, fn func(Delta) error) (Response, error) {
	if sc, ok := c.(StreamingClient); ok {
		return sc.Stream(ctx, req, fn)
	}
	resp, err := c.Complete(ctx, req)
	if err != nil {
		return Response{}, err
	}
	if fn == nil {
		return resp, nil
	}
	sawText := false
	for _, b := range resp.Blocks {
		var kind DeltaKind
		switch b.Kind {
		case BlockText:
			kind = DeltaText
		case BlockThinking:
			kind = DeltaThinking
		default:
			continue
		}
		if kind == DeltaText {
			sawText = true
		}
		if err := fn(Delta{Kind: kind, Text: b.Text}); err != nil {
			return Response{}, fmt.Errorf("llmkit: stream fn: %w", err)
		}
	}
	if !sawText && resp.Text != "" {
		if err := fn(Delta{Kind: DeltaText, Text: resp.Text}); err != nil {
			return Response{}, fmt.Errorf("llmkit: stream fn: %w", err)
		}
	}
	for i, tc := range resp.ToolCalls {
		d := Delta{Kind: DeltaToolCall, Index: i, ID: tc.ID, Name: tc.Name, Arguments: string(tc.Arguments)}
		if err := fn(d); err != nil {
			return Response{}, fmt.Errorf("llmkit: stream fn: %w", err)
		}
	}
	return resp, nil
}

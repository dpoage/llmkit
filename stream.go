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
	// DeltaToolCall: one tool call's fragment. Index identifies the call
	// within the response; ID and Name are set on the first fragment for
	// that Index (and may repeat); Arguments holds a raw JSON fragment to
	// concatenate with the other fragments of the same Index.
	DeltaToolCall DeltaKind = "tool_call"
)

// Delta is one incremental fragment of a streaming completion.
type Delta struct {
	// Kind selects which fields carry the payload.
	Kind DeltaKind
	// Text holds the fragment for DeltaText and DeltaThinking.
	Text string
	// Index identifies the tool call within the response (DeltaToolCall).
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
	// The returned Response is byte-for-byte what Complete would have
	// returned for the same wire exchange (same normalization: Usage
	// convention, StopReason, verbatim thinking blocks, ToolCalls with
	// concatenated Arguments). A nil fn is allowed and makes Stream
	// equivalent to Complete.
	Stream(ctx context.Context, req Request, fn func(Delta) error) (Response, error)
}

// Stream uses c's Stream when c implements StreamingClient; otherwise it
// calls c.Complete and synthesizes deltas from the Response in block order
// (text blocks → DeltaText, thinking blocks → DeltaThinking, each ToolCall →
// one DeltaToolCall with ID/Name/Index and the full Arguments) before
// returning it. Every caller that wants deltas uses this function, so a
// non-streaming client is never a special case.
func Stream(ctx context.Context, c Client, req Request, fn func(Delta) error) (Response, error) {
	if sc, ok := c.(StreamingClient); ok {
		return sc.Stream(ctx, req, fn)
	}
	resp, err := c.Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	if fn == nil {
		return resp, nil
	}
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
		if err := fn(Delta{Kind: kind, Text: b.Text}); err != nil {
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

package anthropic

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/dpoage/llmkit"
)

var _ llmkit.StreamingClient = (*anthropicAdapter)(nil)

// errEarlyStreamEnd marks a connection that closed cleanly before the
// vendor's terminal message_stop event: Complete would have failed
// decoding the equivalent truncated body.
var errEarlyStreamEnd = errors.New("anthropic: stream ended before message_stop")

// streamProtocolError is an unnormalized wire-sequence violation inside one
// stream event; Stream routes it through the same error path as the other
// server-side stream failures.
type streamProtocolError struct{ msg string }

func (e streamProtocolError) Error() string { return e.msg }

// Stream implements llmkit.StreamingClient: it builds params exactly like
// Complete, streams the vendor's SSE events via Messages.NewStreaming,
// accumulates them into an anthropic.Message with the SDK's Accumulate, and
// returns the same normalized Response Complete would return for the same
// wire exchange. A connection that ends without message_stop is an error,
// never a partial success.
//
// Fragment mapping:
//
//	text_delta       → DeltaText
//	thinking_delta   → DeltaThinking
//	signature_delta  → nothing (the signature rides the final thinking block)
//	input_json_delta → DeltaToolCall; for the synthetic structured-output
//	                   tool, DeltaText — finalize presents that call as
//	                   Response.Text, so the fragments stream as text too
//
// The first DeltaToolCall for a call is emitted at its content_block_start
// with empty Arguments; Index counts tool_use blocks (the call's 0-based
// position in Response.ToolCalls), not wire content-block indices — text and
// thinking blocks occupy wire indices too.
func (a *anthropicAdapter) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	params, err := a.buildParams(req)
	if err != nil {
		return llmkit.Response{}, err
	}

	stream := a.client.Messages.NewStreaming(ctx, params)
	// Close on every path: it releases the response body after message_stop
	// and cancels an in-flight stream when fn errors partway through.
	defer func() { _ = stream.Close() }()

	// The synthetic structured-output tool's call becomes Response.Text (see
	// finalize), so its argument fragments stream as DeltaText and it
	// produces no DeltaToolCall fragments.
	synthTool, hasSynthetic := structuredOutputToolName(req, a.caps)

	var (
		acc     anthropic.Message // Accumulate's output feeds toResponse
		toolSeq int               // tool_use blocks seen = next ToolCalls position
		sawStop bool              // the terminal message_stop event arrived
		started = map[int64]startedTool{}
	)

	// forward hands one fragment to fn; a nil fn consumes nothing, and a
	// non-nil fn error aborts the stream (wrapped, per the contract).
	forward := func(d llmkit.Delta) error {
		if fn == nil {
			return nil
		}
		if err := fn(d); err != nil {
			return fmt.Errorf("llmkit: stream fn: %w", err)
		}
		return nil
	}

	for stream.Next() {
		ev := stream.Current()
		// Accumulate first and unconditionally: its index checks rely on
		// seeing every event in order, and the final Message must reflect
		// the full wire exchange even when fn cancels.
		if err := acc.Accumulate(ev); err != nil {
			// Out-of-order indices or undecodable blocks are the same
			// server-class failure as a transport error.
			return llmkit.Response{}, a.normalizeErr(err)
		}

		switch v := ev.AsAny().(type) {
		case anthropic.MessageStopEvent:
			sawStop = true

		case anthropic.ContentBlockStartEvent:
			switch tb := v.ContentBlock.AsAny().(type) {
			case anthropic.ToolUseBlock:
				st := startedTool{pos: toolSeq, id: tb.ID, name: tb.Name}
				toolSeq++
				st.synthetic = hasSynthetic && tb.Name == synthTool
				started[v.Index] = st
				if st.synthetic {
					continue
				}
				// Identity first, empty Arguments: consumers can render
				// "calling get_weather(toolu_x)" before any argument byte
				// arrives.
				if err := forward(llmkit.Delta{Kind: llmkit.DeltaToolCall, Index: st.pos, ID: st.id, Name: st.name}); err != nil {
					return llmkit.Response{}, err
				}
			case anthropic.ServerToolUseBlock:
				// Server-side tool calls (e.g. web_search) also stream as
				// input_json_delta fragments. toResponse drops that block,
				// so its fragments are consumed silently to keep deltas
				// consistent with the final Response.
				started[v.Index] = startedTool{serverTool: true}
			}

		case anthropic.ContentBlockDeltaEvent:
			fnErr, protoErr := a.forwardBlockDelta(v, started, forward)
			if protoErr != nil {
				return llmkit.Response{}, a.normalizeErr(protoErr)
			}
			if fnErr != nil {
				return llmkit.Response{}, fnErr
			}
		}
	}
	if err := stream.Err(); err != nil {
		return llmkit.Response{}, a.normalizeErr(err)
	}
	if !sawStop {
		// Clean EOF before message_stop: a partial message would hand the
		// caller half-finished tool arguments as a success. Complete errors
		// on the equivalent truncated body; this is the same server-class
		// failure.
		return llmkit.Response{}, a.normalizeErr(errEarlyStreamEnd)
	}
	return a.finalize(req, a.toResponse(&acc)), nil
}

// startedTool remembers one in-flight input-bearing block: a tool_use
// block's position in Response.ToolCalls, its wire identity, whether it is
// the synthetic structured-output tool (whose fragments surface as text),
// or a server-side tool call whose fragments are dropped.
type startedTool struct {
	pos        int
	id, name   string
	synthetic  bool
	serverTool bool
}

// forwardBlockDelta maps one content_block_delta event onto Delta fragments.
// fn's error is returned verbatim (it is already wrapped per the contract);
// a wire-sequence violation comes back as an unnormalized streamProtocolError.
func (a *anthropicAdapter) forwardBlockDelta(ev anthropic.ContentBlockDeltaEvent, started map[int64]startedTool, forward func(llmkit.Delta) error) (fnErr, protoErr error) {
	switch d := ev.Delta.AsAny().(type) {
	case anthropic.TextDelta:
		return forward(llmkit.Delta{Kind: llmkit.DeltaText, Text: d.Text}), nil
	case anthropic.ThinkingDelta:
		return forward(llmkit.Delta{Kind: llmkit.DeltaThinking, Text: d.Thinking}), nil
	case anthropic.InputJSONDelta:
		// Empty partials carry nothing to concatenate; skip them.
		if d.PartialJSON == "" {
			return nil, nil
		}
		st, ok := started[ev.Index]
		if !ok {
			// A fragment for a block that never started, or started as a
			// non-input block (text, thinking, redacted), is the same
			// protocol-violation class Accumulate's index checks reject.
			return nil, streamProtocolError{fmt.Sprintf("anthropic: input_json_delta for content block %d with no input-bearing start event", ev.Index)}
		}
		if st.serverTool {
			return nil, nil
		}
		if st.synthetic {
			return forward(llmkit.Delta{Kind: llmkit.DeltaText, Text: d.PartialJSON}), nil
		}
		return forward(llmkit.Delta{
			Kind:      llmkit.DeltaToolCall,
			Index:     st.pos,
			ID:        st.id,
			Name:      st.name,
			Arguments: d.PartialJSON,
		}), nil
	}
	// signature_delta and citations_delta carry no llmkit fragment: the
	// signature arrives inside the final thinking block's Raw, and citations
	// are not surfaced on llmkit.Response.
	return nil, nil
}

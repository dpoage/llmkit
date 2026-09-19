package anthropic

import (
	"context"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/dpoage/llmkit"
)

// The adapter delivers completions incrementally through the same params
// builder and normalizer tail Complete uses.
var _ llmkit.StreamingClient = (*anthropicAdapter)(nil)

// Stream implements llmkit.StreamingClient: it builds params exactly like
// Complete, streams the vendor's SSE events via Messages.NewStreaming,
// accumulates them into an anthropic.Message with the SDK's Accumulate, and
// returns the same normalized Response Complete would return for the same
// wire exchange.
//
// Fragment mapping (wire event → llmkit.Delta):
//
//	text_delta       → DeltaText
//	thinking_delta   → DeltaThinking
//	signature_delta  → nothing (the signature rides the final thinking block)
//	input_json_delta → DeltaToolCall; for the synthetic structured-output
//	                   tool, DeltaText — finalize presents that call as
//	                   Response.Text, so the fragments stream as text too
//
// The first DeltaToolCall for a call is emitted at its content_block_start
// with empty Arguments, so consumers learn ID and Name before the first
// argument fragment; Index counts tool_use blocks (the call's 0-based
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
	// finalize), so its argument fragments are delivered as DeltaText and it
	// produces no DeltaToolCall fragments.
	synthTool, hasSynthetic := structuredOutputToolName(req, a.caps)

	var (
		acc     anthropic.Message // Accumulate's output feeds toResponse
		toolSeq int               // tool_use blocks seen = next ToolCalls position
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
		// the full wire exchange even when fn cancels (the Response is
		// discarded then, but the error path below returns before it).
		if err := acc.Accumulate(ev); err != nil {
			// Out-of-order indices or undecodable blocks mean the server
			// violated its own stream protocol; normalizeErr routes them to
			// the same server-class bucket as a transport failure.
			return llmkit.Response{}, a.normalizeErr(err)
		}

		switch v := ev.AsAny().(type) {
		case anthropic.ContentBlockStartEvent:
			tb, ok := v.ContentBlock.AsAny().(anthropic.ToolUseBlock)
			if !ok {
				continue
			}
			st := startedTool{pos: toolSeq, id: tb.ID, name: tb.Name}
			toolSeq++
			st.synthetic = hasSynthetic && tb.Name == synthTool
			started[v.Index] = st
			if st.synthetic {
				continue
			}
			// Identity first, empty Arguments: consumers can render "calling
			// get_weather(toolu_x)" before any argument byte arrives.
			if err := forward(llmkit.Delta{Kind: llmkit.DeltaToolCall, Index: st.pos, ID: st.id, Name: st.name}); err != nil {
				return llmkit.Response{}, err
			}

		case anthropic.ContentBlockDeltaEvent:
			if err := a.forwardBlockDelta(v, started, forward); err != nil {
				return llmkit.Response{}, err
			}
		}
	}
	if err := stream.Err(); err != nil {
		return llmkit.Response{}, a.normalizeErr(err)
	}
	return a.finalize(req, a.toResponse(&acc)), nil
}

// startedTool remembers one in-flight tool_use block: its position in
// Response.ToolCalls, its wire identity, and whether it is the synthetic
// structured-output tool (whose fragments surface as text).
type startedTool struct {
	pos       int
	id, name  string
	synthetic bool
}

// forwardBlockDelta maps one content_block_delta event onto Delta fragments.
func (a *anthropicAdapter) forwardBlockDelta(ev anthropic.ContentBlockDeltaEvent, started map[int64]startedTool, forward func(llmkit.Delta) error) error {
	switch d := ev.Delta.AsAny().(type) {
	case anthropic.TextDelta:
		return forward(llmkit.Delta{Kind: llmkit.DeltaText, Text: d.Text})
	case anthropic.ThinkingDelta:
		return forward(llmkit.Delta{Kind: llmkit.DeltaThinking, Text: d.Thinking})
	case anthropic.InputJSONDelta:
		// Empty partials carry nothing to concatenate; skip them instead of
		// emitting no-op fragments.
		if d.PartialJSON == "" {
			return nil
		}
		st, ok := started[ev.Index]
		if !ok {
			return nil
		}
		if st.synthetic {
			return forward(llmkit.Delta{Kind: llmkit.DeltaText, Text: d.PartialJSON})
		}
		return forward(llmkit.Delta{
			Kind:      llmkit.DeltaToolCall,
			Index:     st.pos,
			ID:        st.id,
			Name:      st.name,
			Arguments: d.PartialJSON,
		})
	}
	// signature_delta and citations_delta carry no llmkit fragment: the
	// signature arrives inside the final thinking block's Raw, and citations
	// are not surfaced on llmkit.Response.
	return nil
}

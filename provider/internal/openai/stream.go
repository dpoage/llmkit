package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/dpoage/llmkit"
	"github.com/openai/openai-go/v3"
)

// openaiAdapter satisfies llmkit.StreamingClient for both the openai and
// openai-compatible provider types.
var _ llmkit.StreamingClient = (*openaiAdapter)(nil)

// Stream implements [llmkit.StreamingClient] on top of Chat Completions
// streaming: the same params builder Complete uses, plus
// stream_options.include_usage so the final usage-only chunk arrives (a
// stream carries usage nowhere else). Fragments map onto llmkit deltas in
// wire order: delta.content → DeltaText, the delta's reasoning_content
// extra field (MiniMax/DeepSeek-style compatible endpoints) → DeltaThinking,
// and each delta.tool_calls entry → DeltaToolCall with the vendor's wire
// index renumbered to the call's position in Response.ToolCalls. Refusal
// deltas emit nothing — the final response normalizes them exactly as
// Complete does. The accumulated completion goes through the same
// toResponse normalizer Complete uses, so the returned Response is what
// Complete would have returned for the same wire exchange.
//
// A non-nil fn error cancels the stream: it is returned wrapped as
// "llmkit: stream fn: ..." with a zero Response. Open and transport errors
// go through the same error normalization as Complete.
func (o *openaiAdapter) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	params, err := o.buildParams(req)
	if err != nil {
		return llmkit.Response{}, err
	}
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	stream := o.client.Chat.Completions.NewStreaming(ctx, params)

	var acc openai.ChatCompletionAccumulator
	calls := newToolCallTracker()
	for stream.Next() {
		chunk := stream.Current()
		if !acc.AddChunk(chunk) {
			_ = stream.Close()
			return llmkit.Response{}, &llmkit.APIError{
				Kind:     llmkit.ErrServer,
				Provider: o.provider,
				Message:  "stream chunk does not extend the accumulated completion",
			}
		}
		if fn == nil {
			continue
		}
		for _, ch := range chunk.Choices {
			if ch.Index != 0 {
				// The normalizer surfaces choices[0] only; the deltas must
				// match the final Response, so other choices never emit.
				continue
			}
			for _, d := range deltas(ch.Delta, calls) {
				if err := fn(d); err != nil {
					_ = stream.Close()
					return llmkit.Response{}, fmt.Errorf("llmkit: stream fn: %w", err)
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		return llmkit.Response{}, o.normalizeErr(err)
	}
	// A clean stream end without a finish_reason is a truncated generation:
	// Complete over the same wire would fail, and returning the partial text
	// or tool arguments would mask that as success.
	if len(acc.Choices) == 0 || acc.Choices[0].FinishReason == "" {
		return llmkit.Response{}, o.normalizeErr(errors.New("stream ended before finish_reason"))
	}
	compactGhostToolCalls(&acc.ChatCompletion)
	return o.toResponse(&acc.ChatCompletion), nil
}

// compactGhostToolCalls drops never-populated entries the accumulator leaves
// when the wire's tool-call indices are non-contiguous: it places each
// fragment by the raw vendor index, so a gap becomes a zero entry. Such an
// entry is not a call — no fragment ever carried it an ID, name, or
// arguments — and dropping it keeps Response.ToolCalls aligned with the
// renumbered Delta.Index positions.
func compactGhostToolCalls(cc *openai.ChatCompletion) {
	if len(cc.Choices) == 0 {
		return
	}
	tcs := cc.Choices[0].Message.ToolCalls
	kept := tcs[:0]
	for _, tc := range tcs {
		if tc.ID == "" && tc.Function.Name == "" && tc.Function.Arguments == "" {
			continue
		}
		kept = append(kept, tc)
	}
	cc.Choices[0].Message.ToolCalls = kept
}

// deltas maps one choice's delta onto the llmkit fragments it carries;
// empty text and reasoning fragments emit nothing.
func deltas(d openai.ChatCompletionChunkChoiceDelta, calls *toolCallTracker) []llmkit.Delta {
	var out []llmkit.Delta
	// reasoning_content is not a typed SDK field: compatible endpoints send
	// it as an extra delta property, which the SDK keeps (raw JSON value) in
	// JSON.ExtraFields — present but flagged invalid, so Valid() is false
	// even for a value. Null and absent stay silent.
	if f, ok := d.JSON.ExtraFields["reasoning_content"]; ok {
		raw := f.Raw()
		if raw != "" && raw != "null" {
			var text string
			if json.Unmarshal([]byte(raw), &text) == nil && text != "" {
				out = append(out, llmkit.Delta{Kind: llmkit.DeltaThinking, Text: text})
			}
		}
	}
	if d.Content != "" {
		out = append(out, llmkit.Delta{Kind: llmkit.DeltaText, Text: d.Content})
	}
	for _, tc := range d.ToolCalls {
		out = append(out, calls.delta(tc))
	}
	return out
}

// toolCallTracker renumbers the vendor's per-fragment tool_calls index onto
// the 0-based position of the call in Response.ToolCalls (first-appearance
// order; well-behaved endpoints send 0,1,2,... so the numbers coincide) and
// remembers each call's ID and name, so every fragment of a call repeats
// them whether or not the wire does.
type toolCallTracker struct {
	pos   map[int64]int
	ids   map[int]string
	names map[int]string
}

func newToolCallTracker() *toolCallTracker {
	return &toolCallTracker{
		pos:   map[int64]int{},
		ids:   map[int]string{},
		names: map[int]string{},
	}
}

func (t *toolCallTracker) delta(tc openai.ChatCompletionChunkChoiceDeltaToolCall) llmkit.Delta {
	p, ok := t.pos[tc.Index]
	if !ok {
		p = len(t.pos)
		t.pos[tc.Index] = p
	}
	// IDs travel whole on the wire (never fragmented), so the latest wins;
	// names may fragment across chunks, so they concatenate exactly like the
	// accumulator does — every fragment then repeats the same accumulated
	// name the final Response carries.
	if tc.ID != "" {
		t.ids[p] = tc.ID
	}
	t.names[p] += tc.Function.Name
	return llmkit.Delta{
		Kind:      llmkit.DeltaToolCall,
		Index:     p,
		ID:        t.ids[p],
		Name:      t.names[p],
		Arguments: tc.Function.Arguments,
	}
}

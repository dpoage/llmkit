package google

import (
	"context"
	"fmt"

	"github.com/dpoage/llmkit"
	"google.golang.org/genai"
)

// Stream implements llmkit.StreamingClient: Complete with incremental
// delivery over the SDK's GenerateContentStream SSE iterator. The request
// goes through the same builder as Complete, per-chunk errors through the
// same normalizeErr, and the reassembled response through the same
// toResponse — the returned Response is what Complete would have returned
// for the same wire exchange. fn (when non-nil) runs on the calling
// goroutine, in wire order: one DeltaText per text part, one DeltaThinking
// per thought part, one DeltaToolCall per functionCall part with the full
// arguments (function calls arrive whole; Index is the call's arrival
// position among the response's calls). A fn error cancels the stream and
// is returned wrapped, with no Response.
func (g *googleAdapter) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	// Per-request transport-status recorder for normalizeErr's fallback,
	// mirroring Complete.
	ctx = context.WithValue(ctx, transportStatusKey{}, &transportStatus{})
	contents, cfg, err := g.buildRequest(req)
	if err != nil {
		return llmkit.Response{}, err
	}
	agg := &genai.GenerateContentResponse{}
	calls := 0
	var fnErr error
	for chunk, err := range g.client.Models.GenerateContentStream(ctx, g.model, contents, cfg) {
		if err != nil {
			return llmkit.Response{}, g.normalizeErr(ctx, err)
		}
		if fnErr = absorbChunk(agg, chunk, fn, &calls); fnErr != nil {
			break // cancels the iterator; the SDK closes the response body
		}
	}
	if fnErr != nil {
		return llmkit.Response{}, fmt.Errorf("llmkit: stream fn: %w", fnErr)
	}
	return g.toResponse(agg), nil
}

// absorbChunk folds one streamed chunk into agg and, when fn is non-nil,
// emits one delta per part in wire order. The reassembly rules are chosen
// so the aggregate normalizes to exactly what the single-shot response
// would have produced: consecutive plain-text parts merge into one part
// (contiguous text arrives as one part when the whole completion comes
// back at once), thought and functionCall parts are kept verbatim — their
// ThoughtSignature bytes must survive for the next turn — and finishReason,
// usageMetadata, and promptFeedback are each taken from the last chunk that
// carries them. The adapter never sets CandidateCount, so every chunk
// carries the single (index-0) candidate.
func absorbChunk(agg *genai.GenerateContentResponse, chunk *genai.GenerateContentResponse, fn func(llmkit.Delta) error, calls *int) error {
	if len(chunk.Candidates) > 0 && chunk.Candidates[0] != nil {
		if len(agg.Candidates) == 0 {
			agg.Candidates = []*genai.Candidate{{}}
		}
		cand := agg.Candidates[0]
		src := chunk.Candidates[0]
		if fr := src.FinishReason; fr != "" && fr != genai.FinishReasonUnspecified {
			cand.FinishReason = fr
		}
		if src.Content != nil {
			if cand.Content == nil {
				cand.Content = &genai.Content{Role: src.Content.Role}
			}
			for _, p := range src.Content.Parts {
				if p == nil {
					continue
				}
				switch {
				case p.FunctionCall != nil:
					cand.Content.Parts = append(cand.Content.Parts, p)
					if fn != nil {
						d := llmkit.Delta{
							Kind:      llmkit.DeltaToolCall,
							Index:     *calls,
							ID:        p.FunctionCall.ID,
							Name:      p.FunctionCall.Name,
							Arguments: string(functionCallArgs(p.FunctionCall.Args)),
						}
						*calls++
						if err := fn(d); err != nil {
							return err
						}
					}
				case p.Thought:
					cand.Content.Parts = append(cand.Content.Parts, p)
					if fn != nil {
						if err := fn(llmkit.Delta{Kind: llmkit.DeltaThinking, Text: p.Text}); err != nil {
							return err
						}
					}
				default:
					if p.Text == "" {
						continue
					}
					parts := cand.Content.Parts
					if n := len(parts); n > 0 && plainText(parts[n-1]) && plainText(p) {
						parts[n-1].Text += p.Text
					} else {
						cand.Content.Parts = append(parts, p)
					}
					if fn != nil {
						if err := fn(llmkit.Delta{Kind: llmkit.DeltaText, Text: p.Text}); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	if chunk.UsageMetadata != nil {
		agg.UsageMetadata = chunk.UsageMetadata
	}
	if chunk.PromptFeedback != nil {
		agg.PromptFeedback = chunk.PromptFeedback
	}
	return nil
}

// plainText reports whether p is nothing but a text part — the only kind
// that may merge with its predecessor in the aggregate.
func plainText(p *genai.Part) bool {
	return p.FunctionCall == nil && !p.Thought && len(p.ThoughtSignature) == 0 &&
		p.InlineData == nil && p.FileData == nil && p.ExecutableCode == nil &&
		p.CodeExecutionResult == nil && p.FunctionResponse == nil &&
		p.VideoMetadata == nil && p.ToolCall == nil && p.ToolResponse == nil &&
		p.MediaResolution == nil && len(p.PartMetadata) == 0
}

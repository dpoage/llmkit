package adapter

import "github.com/dpoage/llmkit"

// bytesPerToken is the estimate's heuristic: one token per four bytes.
const bytesPerToken = 4

// EstimateUsage returns the Usage for a response whose endpoint reported
// none, with Estimated true. Each count is ceil(bytes/4): InputTokens over
// the request body w saw sent; ReasoningTokens over every BlockThinking
// block's Text (its Raw when Text is empty); OutputTokens over resp.Text
// and every tool call's name and arguments, plus ReasoningTokens.
func EstimateUsage(w *Wire, resp llmkit.Response) llmkit.Usage {
	out := int64(len(resp.Text))
	for _, tc := range resp.ToolCalls {
		out += int64(len(tc.Name) + len(tc.Arguments))
	}
	var reasoning int64
	for _, b := range resp.Blocks {
		if b.Kind != llmkit.BlockThinking {
			continue
		}
		if b.Text != "" {
			reasoning += int64(len(b.Text))
		} else {
			reasoning += int64(len(b.Raw))
		}
	}
	var in int64
	if w != nil {
		in = w.RequestBytes()
	}
	reasoningTokens := ceilTokens(reasoning)
	return llmkit.Usage{
		InputTokens:     ceilTokens(in),
		OutputTokens:    ceilTokens(out) + reasoningTokens,
		ReasoningTokens: reasoningTokens,
		Estimated:       true,
	}
}

func ceilTokens(n int64) int64 { return (n + bytesPerToken - 1) / bytesPerToken }

package adapter

import "github.com/dpoage/llmkit"

// bytesPerToken is the estimate's heuristic: one token per four bytes.
const bytesPerToken = 4

// EstimateUsage is the Usage for a response whose endpoint reported none
// (Estimated is true). InputTokens is ceil(bytes of the request body w saw
// sent / 4); OutputTokens is ceil(bytes of resp.Text plus every tool call's
// name and arguments / 4). Every other count is 0.
func EstimateUsage(w *Wire, resp llmkit.Response) llmkit.Usage {
	out := int64(len(resp.Text))
	for _, tc := range resp.ToolCalls {
		out += int64(len(tc.Name) + len(tc.Arguments))
	}
	var in int64
	if w != nil {
		in = w.RequestBytes()
	}
	return llmkit.Usage{
		InputTokens:  ceilTokens(in),
		OutputTokens: ceilTokens(out),
		Estimated:    true,
	}
}

func ceilTokens(n int64) int64 { return (n + bytesPerToken - 1) / bytesPerToken }

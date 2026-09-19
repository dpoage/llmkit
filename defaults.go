package llmkit

// DefaultMaxTokens is the output-token cap every adapter applies when
// Request.MaxTokens is zero or negative. The value is deliberately
// conservative — safe for legacy 8k-window models and enough to bound
// a runaway completion when a caller forgets to set one. Callers
// wanting a different cap set Request.MaxTokens explicitly.
//
// The rule is uniform across providers: Anthropic requires max_tokens,
// and the OpenAI (max_completion_tokens) and Gemini (maxOutputTokens)
// adapters send the same default instead of relying on each backend's
// own unset behavior. See Request.MaxTokens.
const DefaultMaxTokens = 4096

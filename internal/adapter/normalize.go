// Package adapter holds the parsing/normalization helpers shared by the three
// first-party provider adapters (anthropic, openai, google). It is internal
// on purpose: these helpers are implementation detail, not public API.
package adapter

import (
	"net/http"
	"strings"

	"github.com/dpoage/llmkit"
)

// ClassifyStatus maps an HTTP status code to a sentinel error Kind. body is
// the (optional) error message, used to disambiguate 400s into ErrContextTooLong.
func ClassifyStatus(status int, body string) error {
	switch {
	case status == http.StatusTooManyRequests:
		return llmkit.ErrRateLimited
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return llmkit.ErrAuth
	case status == http.StatusRequestEntityTooLarge:
		return llmkit.ErrContextTooLong
	case status == http.StatusBadRequest:
		if looksLikeContextLength(body) {
			return llmkit.ErrContextTooLong
		}
		return llmkit.ErrInvalidRequest
	case status == 529:
		return llmkit.ErrOverloaded
	case status >= 500:
		return llmkit.ErrServer
	default:
		return llmkit.ErrInvalidRequest
	}
}

// looksLikeContextLength heuristically detects a context-window-exceeded
// error from a provider's 400 message.
func looksLikeContextLength(msg string) bool {
	m := strings.ToLower(msg)
	for _, frag := range []string{
		"context length",
		"context window",
		"maximum context",
		"too many tokens",
		"prompt is too long",
		"input is too long",
		"maximum number of tokens",
		"exceeds the maximum",
		"reduce the length",
	} {
		if strings.Contains(m, frag) {
			return true
		}
	}
	return false
}

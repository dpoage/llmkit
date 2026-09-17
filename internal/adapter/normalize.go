// Package adapter holds the parsing/normalization helpers shared by the three
// first-party provider adapters (anthropic, openai, google). It is internal
// on purpose: these helpers are implementation detail, not public API.
package adapter

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/dpoage/llmkit"
)

// ClassifyStatus maps an HTTP status code to a sentinel error Kind. body is the
// (optional) error message, used to disambiguate 400s into ErrContextTooLong.
func ClassifyStatus(status int, body string) error {
	switch {
	case status == http.StatusTooManyRequests:
		return llmkit.ErrRateLimited
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return llmkit.ErrAuth
	case status == http.StatusRequestEntityTooLarge:
		return llmkit.ErrContextTooLong
	case status == http.StatusBadRequest:
		if LooksLikeContextLength(body) {
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

// LooksLikeContextLength heuristically detects a context-window-exceeded error
// from a provider's 400 message. Providers phrase this differently, so we match
// a few well-known fragments.
func LooksLikeContextLength(msg string) bool {
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

// ParseRetryAfter reads the Retry-After header from an HTTP response, returning
// the suggested delay or 0 if absent / unparseable. Supports both integer
// seconds and HTTP-date forms.
func ParseRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	v := resp.Header.Get("Retry-After")
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

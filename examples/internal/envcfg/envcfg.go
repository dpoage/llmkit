// Package envcfg loads a provider.Spec from the LLMKIT_* environment
// variables shared by the example programs. It lives under internal/ so it
// stays example-only and is not part of the public API.
package envcfg

import (
	"errors"
	"fmt"
	"os"

	"github.com/dpoage/llmkit/provider"
)

// usagePreamble is the shared first half of every example's usage text:
// the LLMKIT_* variables the examples read. It lives here, once, so the
// four examples cannot drift apart.
const usagePreamble = `missing environment:
  LLMKIT_PROVIDER  anthropic | openai | openai-compatible | google
  LLMKIT_MODEL     model name, e.g. claude-sonnet-4-5 or gpt-4o-mini
  LLMKIT_API_KEY   provider API key (any placeholder for a local endpoint)
  LLMKIT_BASE_URL  required for openai-compatible, optional otherwise; e.g. a local endpoint
`

// Usage returns the full usage text an example prints when the LLMKIT_*
// environment is unset: the shared variable preamble followed by the
// example's own re-run line. program is the command line, e.g. "go run
// ./examples/basic"; flags, when non-empty, is appended after a space,
// e.g. "[--image path/to/photo.jpg]".
func Usage(program, flags string) string {
	reRun := program
	if flags != "" {
		reRun += " " + flags
	}
	return usagePreamble + "\nset the variables above, then re-run:\n  " + reRun
}

// Load builds a provider.Spec from the environment:
//
//	LLMKIT_PROVIDER  required; anthropic | openai | openai-compatible | google
//	LLMKIT_MODEL     required; the model identifier
//	LLMKIT_API_KEY   required; the provider API key (any placeholder for a local endpoint)
//	LLMKIT_BASE_URL  required for openai-compatible, optional otherwise; for testing, proxies, and self-hosted gateways
//
// When a required variable is missing, the provider name is unknown, or
// LLMKIT_BASE_URL is missing for the openai-compatible provider, the
// returned error carries the usage string the caller passed in, so each
// example prints instructions for itself.
func Load(usage string) (provider.Spec, error) {
	spec := provider.Spec{
		Model:   os.Getenv("LLMKIT_MODEL"),
		Secret:  os.Getenv("LLMKIT_API_KEY"),
		BaseURL: os.Getenv("LLMKIT_BASE_URL"),
	}
	if spec.Model == "" || spec.Secret == "" || os.Getenv("LLMKIT_PROVIDER") == "" {
		return provider.Spec{}, errors.New(usage)
	}
	typ, err := provider.ParseType(os.Getenv("LLMKIT_PROVIDER"))
	if err != nil {
		return provider.Spec{}, fmt.Errorf("%w\n\n%s", err, usage)
	}
	if typ == provider.TypeOpenAICompatible && spec.BaseURL == "" {
		return provider.Spec{}, errors.New(usage)
	}
	spec.Type = typ
	return spec, nil
}

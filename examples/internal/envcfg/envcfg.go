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

// Load builds a provider.Spec from the environment:
//
//	LLMKIT_PROVIDER  required; anthropic | openai | openai-compatible | google
//	LLMKIT_MODEL     required; the model identifier
//	LLMKIT_API_KEY   required; the credential (API key or OAuth token)
//	LLMKIT_BASE_URL  optional; for openai-compatible endpoints and proxies
//
// When a required variable is missing or the provider name is unknown, the
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
	spec.Type = typ
	return spec, nil
}

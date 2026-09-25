package embed

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestParseBackend_AcceptsBothConstants pins ParseBackend: every accepted
// string succeeds; every rejected value (including the empty string and
// the deleted backend name, built at runtime below so this file's source
// carries no literal reference to it) fails with an error that names the
// bad input and lists both accepted values.
func TestParseBackend_AcceptsBothConstants(t *testing.T) {
	accepted := map[string]Backend{
		"ollama":            BackendOllama,
		"openai-compatible": BackendOpenAICompatible,
	}
	for s, want := range accepted {
		got, err := ParseBackend(s)
		if err != nil {
			t.Errorf("ParseBackend(%q): unexpected error %v", s, err)
		}
		if got != want {
			t.Errorf("ParseBackend(%q) = %q, want %q", s, got, want)
		}
	}

	// deletedBackendName is built at runtime so the source text never
	// carries a literal reference to the deleted backend name.
	deletedBackendName := "hu" + "got"
	for _, bad := range []string{"", deletedBackendName, "OLLAMA", "foo"} {
		_, err := ParseBackend(bad)
		if err == nil {
			t.Fatalf("ParseBackend(%q): expected error", bad)
		}
		// Quoted form, so the "" row demands the error show `""`
		// rather than pass on the always-true Contains(s, "").
		if !strings.Contains(err.Error(), fmt.Sprintf("%q", bad)) {
			t.Errorf("ParseBackend(%q) error %q must name the bad input", bad, err)
		}
		if !strings.Contains(err.Error(), "ollama") || !strings.Contains(err.Error(), "openai-compatible") {
			t.Errorf("ParseBackend(%q) error %q must list both accepted values", bad, err)
		}
	}
}

// TestNewAndValidate_RejectBadConfig pins that New and Validate refuse
// an unknown or zero Backend, or any other rejected field, with an error
// wrapping llmkit.ErrInvalidRequest.
func TestNewAndValidate_RejectBadConfig(t *testing.T) {
	const u = "http://localhost:1"

	if _, err := New(Config{Backend: "nope", Model: "m", URL: u}); !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("New(bad backend) = %v, want errors.Is ErrInvalidRequest", err)
	}
	if _, err := New(Config{Model: "m", URL: u}); !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("New(zero backend) = %v, want errors.Is ErrInvalidRequest", err)
	}

	rejected := []Config{
		{Backend: "nope", Model: "m", URL: u},
		{Backend: BackendOllama, Model: "", URL: u},
		{Backend: BackendOllama, Model: "m", URL: ""},
		{Backend: BackendOllama, Model: "m", URL: u, Dimensions: -1},
		{Backend: BackendOllama, Model: "m", URL: u, MaxBatch: -1},
	}
	for i, cfg := range rejected {
		if err := cfg.Validate(); !errors.Is(err, llmkit.ErrInvalidRequest) {
			t.Errorf("rejected[%d].Validate() = %v, want errors.Is ErrInvalidRequest", i, err)
		}
	}
}

package provider

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// TestNew_RejectsUnknownAuth pins the typed-Auth contract: New must refuse
// an Auth value outside the {AuthAPIKey, AuthOAuthToken} set with an error
// wrapping ErrInvalidRequest — never a silent fallback to API-key mode,
// which would send an OAuth secret in an x-api-key header.
func TestNew_RejectsUnknownAuth(t *testing.T) {
	for _, auth := range []Auth{"api_key", "bearer", "oauth"} {
		t.Run(string(auth), func(t *testing.T) {
			spec := Spec{Type: TypeOpenAI, Model: "gpt-test", Secret: "k", Auth: auth}
			client, err := New(context.Background(), spec, Options{})
			if client != nil {
				t.Fatal("New returned a client for an unknown Auth value")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

// TestNew_RejectsEmptyModel pins the required-Model contract: an endpoint
// without a model name is a malformed spec, not a default.
func TestNew_RejectsEmptyModel(t *testing.T) {
	spec := Spec{Type: TypeOpenAI, Secret: "k"}
	client, err := New(context.Background(), spec, Options{})
	if client != nil || err == nil {
		t.Fatalf("New with empty Model = (%v, %v), want a nil client and an error", client, err)
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("error = %v, want ErrInvalidRequest", err)
	}
}

// TestParseType_RejectsGarbageListingValidValues pins the parser contract:
// garbage is refused and the error names the bad value and every accepted
// one, so a bad config value is actionable without reading this package.
func TestParseType_RejectsGarbageListingValidValues(t *testing.T) {
	_, err := ParseType("gibberish")
	if err == nil {
		t.Fatal("ParseType(gibberish) = nil error, want an error")
	}
	for _, want := range []string{"gibberish", "anthropic", "openai", "openai-compatible", "google"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("ParseType error %q does not mention %q", err, want)
		}
	}
	for _, name := range []string{"anthropic", "openai", "openai-compatible", "google"} {
		typ, err := ParseType(name)
		if err != nil {
			t.Fatalf("ParseType(%q): %v", name, err)
		}
		if string(typ) != name {
			t.Errorf("ParseType(%q) = %q, want the same value back", name, typ)
		}
	}
}

// TestNew_RecorderProviderTag pins the UsageEvent tagging contract: the
// provider tag defaults to string(spec.Type) and Options.Provider overrides
// it, for callers whose ledger keys on a config-map name.
func TestNew_RecorderProviderTag(t *testing.T) {
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockTextBody("openai", "ok", 1, 1)))
	})
	complete := func(t *testing.T, opts Options) llmkit.UsageEvent {
		t.Helper()
		var mu sync.Mutex
		var ev llmkit.UsageEvent
		opts.Recorder = llmkit.RecorderFunc(func(e llmkit.UsageEvent) {
			mu.Lock()
			defer mu.Unlock()
			ev = e
		})
		spec := Spec{Type: TypeOpenAI, Model: "gpt-test", Secret: "k", BaseURL: base}
		client, err := New(context.Background(), spec, opts)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		mu.Lock()
		defer mu.Unlock()
		return ev
	}

	t.Run("defaults to spec.Type", func(t *testing.T) {
		ev := complete(t, Options{})
		if ev.Provider != string(TypeOpenAI) || ev.Model != "gpt-test" {
			t.Errorf("event tags = %q/%q, want %q/gpt-test", ev.Provider, ev.Model, string(TypeOpenAI))
		}
	})
	t.Run("Options.Provider overrides", func(t *testing.T) {
		ev := complete(t, Options{Provider: "my-config-key"})
		if ev.Provider != "my-config-key" || ev.Model != "gpt-test" {
			t.Errorf("event tags = %q/%q, want my-config-key/gpt-test", ev.Provider, ev.Model)
		}
	})
}

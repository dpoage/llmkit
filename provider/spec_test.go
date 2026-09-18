package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

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
			for _, want := range []string{string(auth), "Auth"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
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

// TestNew_RejectsOAuthOnNonAnthropic pins the Auth-scope rule: OAuth
// bearer-token authentication is implemented only by the Anthropic adapter,
// so AuthOAuthToken on any other Type must be refused with
// ErrInvalidRequest — never silently downgraded to API-key mode (the
// failure this pins: the token would ride the other adapter's API-key
// header).
func TestNew_RejectsOAuthOnNonAnthropic(t *testing.T) {
	for _, typ := range []Type{TypeOpenAI, TypeOpenAICompatible, TypeGoogle} {
		t.Run(string(typ), func(t *testing.T) {
			spec := Spec{Type: typ, Model: "gpt-test", Secret: "bearer-secret", Auth: AuthOAuthToken}
			client, err := New(context.Background(), spec, Options{})
			if client != nil {
				t.Fatal("New returned a client for AuthOAuthToken on a non-Anthropic Type")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("error = %v, want ErrInvalidRequest", err)
			}
			for _, want := range []string{string(AuthOAuthToken), string(typ), string(TypeAnthropic)} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

// TestNew_RejectsEmptySecret pins the required-Secret contract: Secret is
// the resolved credential, so an empty or whitespace-only value is a caller
// bug. New must refuse it for every Type with an error wrapping
// ErrInvalidRequest — uniformly, because the adapters disagree on empty
// secrets (ambient-env fallback, empty Bearer, hard error) — and the error
// must never echo the secret. The unroutable BaseURL plus the elapsed-time
// bound prove the refusal returns before any network activity.
func TestNew_RejectsEmptySecret(t *testing.T) {
	for _, typ := range []Type{TypeAnthropic, TypeOpenAI, TypeOpenAICompatible, TypeGoogle} {
		for _, secret := range []string{"", "   ", "\t\n"} {
			t.Run(string(typ)+"/"+fmt.Sprintf("%q", secret), func(t *testing.T) {
				spec := Spec{Type: typ, Model: "test-model", Secret: secret, BaseURL: "http://10.255.255.1:1"}
				start := time.Now()
				client, err := New(context.Background(), spec, Options{})
				if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
					t.Fatalf("New took %s; a refusal path must not touch the network", elapsed)
				}
				if client != nil {
					t.Fatal("New returned a client for an empty Secret")
				}
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Errorf("error = %v, want ErrInvalidRequest", err)
				}
				if !strings.Contains(err.Error(), "secret") {
					t.Errorf("error %q must name the field", err)
				}
				if secret != "" && strings.Contains(err.Error(), secret) {
					t.Errorf("error %q must not echo the secret", err)
				}
			})
		}
	}
}

// TestNew_RejectsOpenAICompatibleEmptyBaseURL pins the BaseURL rule: an
// openai-compatible Type with an empty BaseURL would silently target the
// first-party OpenAI host, so New refuses it with ErrInvalidRequest and a
// message saying the endpoint must be given. The elapsed-time bound proves
// the refusal returns before any dial could begin.
func TestNew_RejectsOpenAICompatibleEmptyBaseURL(t *testing.T) {
	spec := Spec{Type: TypeOpenAICompatible, Model: "test-model", Secret: "k"}
	start := time.Now()
	client, err := New(context.Background(), spec, Options{})
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("New took %s; a refusal path must not touch the network", elapsed)
	}
	if client != nil {
		t.Fatal("New returned a client for an openai-compatible Spec with an empty BaseURL")
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("error = %v, want ErrInvalidRequest", err)
	}
	if !strings.Contains(err.Error(), "endpoint must be given") {
		t.Errorf("error %q must say the endpoint must be given", err)
	}
}

// TestNew_BaseURLRules pins the per-Type BaseURL semantics: openai-compatible
// requires an explicit endpoint, while the three vendor Types construct fine
// against their SDK defaults. New never dials — the unroutable BaseURL in
// the accepted case would hang visibly if it did.
func TestNew_BaseURLRules(t *testing.T) {
	t.Run("openai-compatible accepts explicit BaseURL", func(t *testing.T) {
		spec := Spec{Type: TypeOpenAICompatible, Model: "test-model", Secret: "k", BaseURL: "http://10.255.255.1:1"}
		client, err := New(context.Background(), spec, Options{})
		if err != nil {
			t.Fatalf("New with explicit BaseURL: %v", err)
		}
		if client == nil {
			t.Fatal("New returned a nil client")
		}
	})
	for _, typ := range []Type{TypeAnthropic, TypeOpenAI, TypeGoogle} {
		t.Run(string(typ)+" constructs with empty BaseURL", func(t *testing.T) {
			spec := Spec{Type: typ, Model: "test-model", Secret: "k"}
			client, err := New(context.Background(), spec, Options{})
			if err != nil {
				t.Fatalf("New with empty BaseURL: %v", err)
			}
			if client == nil {
				t.Fatal("New returned a nil client")
			}
		})
	}
}

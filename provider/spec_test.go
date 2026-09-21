package provider

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
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

// zeroDialTransport counts DialContext attempts; the test asserts none, so a
// refusal path must complete before any network activity.
type zeroDialTransport struct {
	dials int
}

func (z *zeroDialTransport) dialContext(_ context.Context, _, _ string) (net.Conn, error) {
	z.dials++
	return nil, errors.New("zeroDialTransport: dial attempted")
}

func (z *zeroDialTransport) client() *http.Client {
	return &http.Client{Transport: &http.Transport{DialContext: z.dialContext}}
}

// TestNew_RejectsEmptySecret pins the required-Secret contract: New must
// refuse an empty or whitespace-only Secret for every Type with
// ErrInvalidRequest, return before any network dial, and produce a
// message that does not vary with the secret's bytes (so it cannot carry
// them).
func TestNew_RejectsEmptySecret(t *testing.T) {
	secrets := []string{"", " ", "   ", "\t", "\n", "\t\n", " \t \n "}
	for _, typ := range []Type{TypeAnthropic, TypeOpenAI, TypeOpenAICompatible, TypeGoogle} {
		t.Run(string(typ), func(t *testing.T) {
			messages := make([]string, len(secrets))
			for i, secret := range secrets {
				t.Run(fmt.Sprintf("%q", secret), func(t *testing.T) {
					z := &zeroDialTransport{}
					spec := Spec{Type: typ, Model: "test-model", Secret: secret, BaseURL: "http://10.255.255.1:1"}
					client, err := New(context.Background(), spec, Options{HTTPClient: z.client()})
					if z.dials != 0 {
						t.Fatalf("New dialed %d time(s); a refusal path must not touch the network", z.dials)
					}
					if client != nil {
						t.Fatal("New returned a client for an empty Secret")
					}
					if !errors.Is(err, llmkit.ErrInvalidRequest) {
						t.Fatalf("error = %v, want ErrInvalidRequest", err)
					}
					if !strings.Contains(err.Error(), "secret") {
						t.Errorf("error %q must name the field", err)
					}
					messages[i] = err.Error()
				})
			}
			for i := 1; i < len(messages); i++ {
				if messages[i] != messages[0] {
					t.Errorf("error for secret %q = %q, want byte-identical to the error for %q (%q) — the message must not vary with the secret's content", secrets[i], messages[i], secrets[0], messages[0])
				}
			}
		})
	}
}

// TestNew_RejectsSecretWithSurroundingWhitespace pins the trimmed-Secret
// contract: New refuses a Secret that differs from its own
// strings.TrimSpace — for example "sk-abc\n", " sk-abc", "sk-abc ",
// "sk-abc\t", and "\tsk-abc\n" — with an error wrapping
// ErrInvalidRequest that never echoes the value, and rejects it before
// any adapter is constructed (zero dials).
func TestNew_RejectsSecretWithSurroundingWhitespace(t *testing.T) {
	for _, secret := range []string{"sk-abc\n", " sk-abc", "sk-abc ", "sk-abc\t", "\tsk-abc\n"} {
		t.Run(fmt.Sprintf("%q", secret), func(t *testing.T) {
			z := &zeroDialTransport{}
			spec := Spec{Type: TypeOpenAI, Model: "test-model", Secret: secret, BaseURL: "http://10.255.255.1:1"}
			client, err := New(context.Background(), spec, Options{HTTPClient: z.client()})
			if z.dials != 0 {
				t.Fatalf("New dialed %d time(s); a refusal path must not touch the network", z.dials)
			}
			if client != nil {
				t.Fatal("New returned a client for a Secret with surrounding whitespace")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("error = %v, want ErrInvalidRequest", err)
			}
			if !strings.Contains(err.Error(), "secret") {
				t.Errorf("error %q must name the field", err)
			}
			if strings.Contains(err.Error(), "sk-abc") {
				t.Errorf("error %q must not echo the secret", err)
			}
		})
	}
}

// TestNew_RejectsOpenAICompatibleEmptyBaseURL pins the BaseURL rule: an
// openai-compatible Type with an empty BaseURL would silently target the
// first-party OpenAI host, so New refuses it with ErrInvalidRequest and a
// message saying the endpoint must be given. The zero-dial transport
// proves the refusal returns before any dial could begin.
func TestNew_RejectsOpenAICompatibleEmptyBaseURL(t *testing.T) {
	z := &zeroDialTransport{}
	spec := Spec{Type: TypeOpenAICompatible, Model: "test-model", Secret: "k"}
	client, err := New(context.Background(), spec, Options{HTTPClient: z.client()})
	if z.dials != 0 {
		t.Fatalf("New dialed %d time(s); a refusal path must not touch the network", z.dials)
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

// hostCapturingTransport records the resolved req.URL and fails the round
// trip, so a test can bind a Type's vendor-default host with no network I/O.
type hostCapturingTransport struct {
	mu  sync.Mutex
	url *url.URL
}

func (t *hostCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.mu.Lock()
	t.url = req.URL
	t.mu.Unlock()
	return nil, errors.New("hostCapturingTransport: no network")
}

func (t *hostCapturingTransport) observed() *url.URL {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.url
}

// TestNew_VendorHosts pins the vendor-default endpoint each Type resolves
// to with an empty spec.BaseURL. The test unsets
// ANTHROPIC_BASE_URL, OPENAI_BASE_URL, and GOOGLE_GEMINI_BASE_URL because
// the Anthropic and OpenAI SDKs key off os.LookupEnv's ok result — an
// empty-but-present var still overrides the default, so t.Setenv to ""
// is not equivalent (t.Setenv has no unset form). A RoundTripper is
// bound and the resolved req.URL is read, so an SDK bump that silently
// changes the default host fails this test. Retry is capped at one
// attempt since the transport always fails and the shared retry wrapper
// would otherwise back off the induced failure.
func TestNew_VendorHosts(t *testing.T) {
	for _, name := range []string{"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_GEMINI_BASE_URL"} {
		if old, had := os.LookupEnv(name); had {
			if err := os.Unsetenv(name); err != nil {
				t.Fatalf("unsetenv %s: %v", name, err)
			}
			t.Cleanup(func() {
				if err := os.Setenv(name, old); err != nil {
					t.Errorf("restore %s: %v", name, err)
				}
			})
		}
	}

	for _, tc := range []struct {
		typ  Type
		host string
	}{
		{TypeAnthropic, "api.anthropic.com"},
		{TypeOpenAI, "api.openai.com"},
		{TypeGoogle, "generativelanguage.googleapis.com"},
	} {
		t.Run(string(tc.typ), func(t *testing.T) {
			rt := &hostCapturingTransport{}
			spec := Spec{Type: tc.typ, Model: "test-model", Secret: "k"}
			opts := Options{HTTPClient: &http.Client{Transport: rt}, Retry: retry.Config{MaxAttempts: 1}}
			client, err := New(context.Background(), spec, opts)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			_, _ = client.Complete(context.Background(), simpleRequest())
			u := rt.observed()
			if u == nil {
				t.Fatal("no request observed; the adapter never reached the transport")
			}
			if u.Host != tc.host {
				t.Errorf("resolved host = %q, want %q", u.Host, tc.host)
			}
		})
	}
}

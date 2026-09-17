package anthropic

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// mockAnthropicResponse returns a minimal valid Anthropic /v1/messages response
// so the test server does not need to serve elaborate JSON.
func mockAnthropicResponse() string {
	return mockTextBody("ok", 5, 2)
}

// TestAnthropicAuth_APIKeyMode asserts that an api_key-mode adapter sends the
// x-api-key header and no Authorization header containing "Bearer", and does
// not send the oauth beta header value.
func TestAnthropicAuth_APIKeyMode(t *testing.T) {
	var gotXAPIKey, gotAuth, gotBeta string

	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("authorization")
		gotBeta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockAnthropicResponse()))
	})

	// Build the adapter in api_key mode directly.
	adapter := New("claude-test", Options{
		APIKey:  "sk-ant-test-key",
		BaseURL: base,
	})

	_, err := adapter.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// x-api-key must be present.
	if gotXAPIKey == "" {
		t.Error("api_key mode: x-api-key header must be set")
	}
	// Authorization header must not contain a Bearer token.
	if strings.HasPrefix(strings.ToLower(gotAuth), "bearer ") {
		t.Errorf("api_key mode: Authorization header must not be a Bearer token, got %q", gotAuth)
	}
	// oauth-2025-04-20 beta value must NOT appear in api_key mode.
	if strings.Contains(gotBeta, "oauth-2025-04-20") {
		t.Errorf("api_key mode: anthropic-beta must not contain oauth-2025-04-20, got %q", gotBeta)
	}
}

// TestAnthropicAuth_OAuthMode asserts that an oauth-token-mode adapter:
//   - Sets Authorization: Bearer <token>
//   - Includes "oauth-2025-04-20" in the anthropic-beta header
//   - Does NOT set x-api-key
func TestAnthropicAuth_OAuthMode(t *testing.T) {
	var gotXAPIKey, gotAuth, gotBeta string

	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("authorization")
		gotBeta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockAnthropicResponse()))
	})

	// Build the adapter in oauth-token mode directly.
	adapter := New("claude-test", Options{
		AuthToken: "claude-oauth-bearer-token-xyz",
		BaseURL:   base,
	})

	_, err := adapter.Complete(context.Background(), simpleRequest())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// Authorization must be "Bearer <token>".
	wantAuth := "Bearer claude-oauth-bearer-token-xyz"
	if gotAuth != wantAuth {
		t.Errorf("oauth mode: Authorization = %q, want %q", gotAuth, wantAuth)
	}
	// anthropic-beta must contain the oauth value.
	if !strings.Contains(gotBeta, "oauth-2025-04-20") {
		t.Errorf("oauth mode: anthropic-beta = %q, must contain oauth-2025-04-20", gotBeta)
	}
	// x-api-key must be absent — the API rejects requests carrying both.
	if gotXAPIKey != "" {
		t.Errorf("oauth mode: x-api-key must be absent, got %q", gotXAPIKey)
	}
}

// TestAnthropicAuth_OAuthModeIgnoresHostAPIKeyEnv pins the env-poisoning guard:
// anthropic.NewClient applies env defaults BEFORE explicit options, and a host
// ANTHROPIC_API_KEY eagerly sets the X-Api-Key header. Without the adapter's
// WithHeaderDel, an oauth-mode request would carry BOTH credentials and the API
// would reject it. t.Setenv simulates the poisoned host.
func TestAnthropicAuth_OAuthModeIgnoresHostAPIKeyEnv(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-host-env-key")

	var gotXAPIKey, gotAuth string
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockAnthropicResponse()))
	})

	adapter := New("claude-test", Options{
		AuthToken: "oauth-test-token",
		BaseURL:   base,
	})
	if _, err := adapter.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if gotXAPIKey != "" {
		t.Errorf("oauth mode with host ANTHROPIC_API_KEY: x-api-key must be stripped, got %q", gotXAPIKey)
	}
	if gotAuth != "Bearer oauth-test-token" {
		t.Errorf("oauth mode: Authorization = %q, want Bearer oauth-test-token", gotAuth)
	}
}

// TestAnthropicAuth_APIKeyModeIgnoresHostAuthTokenEnv pins the symmetric guard:
// a host ANTHROPIC_AUTH_TOKEN (with no ANTHROPIC_API_KEY) makes the env
// defaults eagerly set Authorization; api_key mode must strip it so only
// x-api-key is sent.
func TestAnthropicAuth_APIKeyModeIgnoresHostAuthTokenEnv(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "host-env-bearer")

	var gotXAPIKey, gotAuth string
	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockAnthropicResponse()))
	})

	adapter := New("claude-test", Options{
		APIKey:  "sk-ant-config-key",
		BaseURL: base,
	})
	if _, err := adapter.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if gotAuth != "" {
		t.Errorf("api_key mode with host ANTHROPIC_AUTH_TOKEN: Authorization must be stripped, got %q", gotAuth)
	}
	if gotXAPIKey != "sk-ant-config-key" {
		t.Errorf("api_key mode: x-api-key = %q, want sk-ant-config-key", gotXAPIKey)
	}
}

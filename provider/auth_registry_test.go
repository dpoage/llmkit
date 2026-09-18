package provider

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// mockAnthropicResponse returns a minimal valid Anthropic /v1/messages response
// so the test server does not need to serve elaborate JSON.
func mockAnthropicResponse() string {
	return mockTextBody("anthropic", "ok", 5, 2)
}

// TestNewClient_AnthropicAPIKeyMode exercises the registry-level dispatch for
// api_key mode to confirm the correct header is sent through New.
func TestNewClient_AnthropicAPIKeyMode(t *testing.T) {
	var gotXAPIKey, gotAuth string

	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockAnthropicResponse()))
	})

	provider := Spec{
		Type:    TypeAnthropic,
		BaseURL: base,
		Model:   "claude-test",
		Secret:  "sk-ant-via-registry",
	}
	client, err := New(context.Background(), provider, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if gotXAPIKey == "" {
		t.Error("api_key mode via New: x-api-key must be set")
	}
	if strings.HasPrefix(strings.ToLower(gotAuth), "bearer ") {
		t.Errorf("api_key mode via New: Authorization must not be a Bearer token, got %q", gotAuth)
	}
}

// TestNewClient_AnthropicOAuthMode exercises the registry-level dispatch for
// oauth-token mode to confirm the token routes to the Authorization header.
func TestNewClient_AnthropicOAuthMode(t *testing.T) {
	var gotXAPIKey, gotAuth, gotBeta string

	base := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("authorization")
		gotBeta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockAnthropicResponse()))
	})

	provider := Spec{
		Type:    TypeAnthropic,
		Auth:    AuthOAuthToken,
		BaseURL: base,
		Model:   "claude-test",
		Secret:  "claude-oauth-registry-token",
	}
	// New routes Secret to AuthToken because provider.Auth == AuthOAuthToken.
	client, err := New(context.Background(), provider, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	wantAuth := "Bearer claude-oauth-registry-token"
	if gotAuth != wantAuth {
		t.Errorf("oauth mode via New: Authorization = %q, want %q", gotAuth, wantAuth)
	}
	if !strings.Contains(gotBeta, "oauth-2025-04-20") {
		t.Errorf("oauth mode via New: anthropic-beta = %q, must contain oauth-2025-04-20", gotBeta)
	}
	if gotXAPIKey != "" {
		t.Errorf("oauth mode via New: x-api-key must be absent, got %q", gotXAPIKey)
	}
}

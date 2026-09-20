package decide

import (
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

func TestNew_ConfigValidation(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"empty APIKey", Config{Model: "jev-latest"}},
		{"whitespace-only APIKey", Config{APIKey: "   ", Model: "jev-latest"}},
		{"padded APIKey", Config{APIKey: " key ", Model: "jev-latest"}},
		{"empty Model", Config{APIKey: "key"}},
		{"negative jitter", Config{APIKey: "key", Model: "jev-latest", Retry: llmkit.RetryConfig{Jitter: -0.1}}},
		{"jitter above 1", Config{APIKey: "key", Model: "jev-latest", Retry: llmkit.RetryConfig{Jitter: 1.5}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := New(tt.cfg)
			if c != nil {
				t.Fatal("New returned a client for an invalid config")
			}
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Fatalf("err = %v, want ErrInvalidRequest", err)
			}
		})
	}

	// A padded key must be refused without the key value ever reaching the
	// error text.
	_, err := New(Config{APIKey: " sk-super-secret-do-not-echo ", Model: "jev-latest"})
	if err == nil {
		t.Fatal("expected an error for a padded API key")
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("err = %v, want ErrInvalidRequest", err)
	}
	if strings.Contains(err.Error(), "sk-super-secret-do-not-echo") {
		t.Errorf("error echoes the API key: %v", err)
	}
}

func TestNew_Defaults(t *testing.T) {
	c, err := New(Config{APIKey: "key", Model: "jev-latest"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.endpoint != "https://api.typesafe.ai/v1/systemone" {
		t.Errorf("endpoint = %q, want the vendor default with the System One path", c.endpoint)
	}

	def := llmkit.DefaultRetryConfig()
	if c.retry.MaxAttempts != 3 {
		t.Errorf("MaxAttempts = %d, want 3 (decide default)", c.retry.MaxAttempts)
	}
	if c.retry.RequestTimeout != 30*time.Second {
		t.Errorf("RequestTimeout = %v, want 30s (decide default)", c.retry.RequestTimeout)
	}
	if c.retry.BaseDelay != def.BaseDelay || c.retry.MaxDelay != def.MaxDelay {
		t.Errorf("delays = %v/%v, want kit defaults %v/%v", c.retry.BaseDelay, c.retry.MaxDelay, def.BaseDelay, def.MaxDelay)
	}
	if c.retry.Jitter != 0 {
		t.Errorf("Jitter = %v, want 0 (literal: unset means no jitter until resolved)", c.retry.Jitter)
	}

	if got := (Config{}).httpClient().Timeout; got != 0 {
		t.Errorf("default client timeout = %v, want 0 (bounded via per-attempt ctx)", got)
	}
	injected := &http.Client{Timeout: time.Second}
	if got := (Config{HTTPClient: injected}).httpClient(); got != injected {
		t.Error("httpClient must return the injected client as-is")
	}
}

func TestNew_ExplicitRetryAndBaseURLKept(t *testing.T) {
	policy := llmkit.RetryConfig{MaxAttempts: 5, RequestTimeout: 10 * time.Second, BaseDelay: time.Second, MaxDelay: time.Minute, Jitter: 0.2}
	c, err := New(Config{APIKey: "key", Model: "jev-1.13.0", BaseURL: "http://localhost:9000/", Retry: policy})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.endpoint != "http://localhost:9000/v1/systemone" {
		t.Errorf("endpoint = %q, want trailing slash trimmed and path appended", c.endpoint)
	}
	if !reflect.DeepEqual(c.retry, policy) {
		t.Errorf("retry = %+v, want the explicit policy kept as-is", c.retry)
	}
}

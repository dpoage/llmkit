package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// TestAPIKey_BearerHeaderBothBackends pins: a non-empty Config.APIKey
// sends "Authorization: Bearer <key>" on every request on both backends;
// an empty APIKey sends no Authorization header on either.
func TestAPIKey_BearerHeaderBothBackends(t *testing.T) {
	for _, backend := range []Backend{BackendOllama, BackendOpenAICompatible} {
		t.Run(string(backend)+"/with_key", func(t *testing.T) {
			var gotAuth []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotAuth = append(gotAuth, r.Header.Get("Authorization"))
				writeEchoEmbeddings(w, backend, r)
			}))
			defer srv.Close()

			emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, APIKey: "k"})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := emb.Embed(context.Background(), "x"); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if _, err := emb.EmbedBatch(context.Background(), []string{"a", "b"}); err != nil {
				t.Fatalf("EmbedBatch: %v", err)
			}
			if len(gotAuth) != 2 {
				t.Fatalf("requests = %d, want 2", len(gotAuth))
			}
			for i, auth := range gotAuth {
				if auth != "Bearer k" {
					t.Errorf("request %d Authorization = %q, want %q", i, auth, "Bearer k")
				}
			}
		})

		t.Run(string(backend)+"/without_key", func(t *testing.T) {
			var sawAuth bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, sawAuth = r.Header["Authorization"]
				writeSuccessBody(w, string(backend))
			}))
			defer srv.Close()

			emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := emb.Embed(context.Background(), "x"); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if sawAuth {
				t.Error("request carries an Authorization header, want none")
			}
		})
	}
}

// TestAPIKey_WhitespaceRefused pins: New and Validate refuse an APIKey
// with leading or trailing whitespace, and the error never echoes the
// key, raw or trimmed.
func TestAPIKey_WhitespaceRefused(t *testing.T) {
	const padded = "  sk-r6-padded-7f3a  "
	const trimmed = "sk-r6-padded-7f3a"

	cfg := Config{Backend: BackendOllama, Model: "m", URL: "http://localhost:1", APIKey: padded}

	err := cfg.Validate()
	if err == nil || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("Validate() = %v, want errors.Is ErrInvalidRequest", err)
	}
	if strings.Contains(err.Error(), trimmed) || strings.Contains(err.Error(), padded) {
		t.Errorf("Validate() error %q must not echo the key, raw or trimmed", err)
	}

	_, err = New(cfg)
	if err == nil || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("New() = %v, want errors.Is ErrInvalidRequest", err)
	}
	if strings.Contains(err.Error(), trimmed) || strings.Contains(err.Error(), padded) {
		t.Errorf("New() error %q must not echo the key, raw or trimmed", err)
	}
}

// TestAPIKey_ControlCharacterRefused pins: New and Validate refuse an
// APIKey that net/http cannot send in the Authorization header value, on
// both backends, without echoing the key. Without the check such a key
// passes Validate and every attempt fails client-side as a retryable
// transport error. The sweep holds Validate to net/http's own rule: for
// every byte b, the key "sk"+b+"z" is accepted exactly when net/http
// sends it, and an accepted key embeds.
func TestAPIKey_ControlCharacterRefused(t *testing.T) {
	const head, tail = "sk-r6", "ctl-9c1e"
	rows := map[string]string{"LF": "\n", "CR": "\r", "NUL": "\x00"}
	for _, backend := range []Backend{BackendOllama, BackendOpenAICompatible} {
		for name, ctl := range rows {
			t.Run(string(backend)+"/"+name, func(t *testing.T) {
				cfg := Config{Backend: backend, Model: "m", URL: "http://localhost:1", APIKey: head + ctl + tail}
				_, newErr := New(cfg)
				for call, err := range map[string]error{"Validate": cfg.Validate(), "New": newErr} {
					if !errors.Is(err, llmkit.ErrInvalidRequest) {
						t.Errorf("%s() = %v, want errors.Is ErrInvalidRequest", call, err)
						continue
					}
					if strings.Contains(err.Error(), head) || strings.Contains(err.Error(), tail) {
						t.Errorf("%s() error %q must not echo the key", call, err)
					}
				}
			})
		}

		t.Run(string(backend)+"/sweep_matches_net_http", func(t *testing.T) {
			var gotAuth string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotAuth = r.Header.Get("Authorization")
				writeEchoEmbeddings(w, backend, r)
			}))
			defer srv.Close()

			for b := range 256 {
				key := "sk" + string([]byte{byte(b)}) + "z"
				emb, err := New(Config{Backend: backend, Model: "m", URL: srv.URL, APIKey: key, Retry: retry.Config{MaxAttempts: 1}})
				if err == nil {
					if _, err := emb.Embed(context.Background(), "x"); err != nil {
						t.Errorf("byte %#02x: New accepted the key but Embed failed: %v", b, err)
					} else if gotAuth != "Bearer "+key {
						t.Errorf("byte %#02x: Authorization = %q, want %q", b, gotAuth, "Bearer "+key)
					}
					continue
				}
				if !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Errorf("byte %#02x: New() = %v, want nil or errors.Is ErrInvalidRequest", b, err)
				}
				req, _ := http.NewRequest(http.MethodPost, srv.URL, nil)
				req.Header.Set("Authorization", "Bearer "+key)
				if resp, err := srv.Client().Do(req); err == nil {
					_ = resp.Body.Close()
					t.Errorf("byte %#02x: New refused a key net/http sends", b)
				}
			}
		})

		t.Run(string(backend)+"/normal_key_accepted", func(t *testing.T) {
			cfg := Config{Backend: backend, Model: "m", URL: "http://localhost:1", APIKey: "sk-r6-normal-key"}
			if err := cfg.Validate(); err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
			if _, err := New(cfg); err != nil {
				t.Errorf("New() = %v, want nil", err)
			}
		})
	}
}

// writeEchoEmbeddings decodes the request body and writes back a
// count-matching success response, so a batch call of any size gets a
// vector per input instead of writeSuccessBody's fixed single vector.
func writeEchoEmbeddings(w http.ResponseWriter, backend Backend, r *http.Request) {
	switch backend {
	case BackendOllama:
		var req ollamaRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		embs := make([][]float64, len(req.Input))
		for i := range embs {
			embs[i] = []float64{0.1}
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: embs})
	case BackendOpenAICompatible:
		var req openaiRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		data := make([]openaiEmbedding, len(req.Input))
		for i := range data {
			data[i] = openaiEmbedding{Embedding: []float64{0.1}, Index: i}
		}
		jsonEncode(w, openaiResponse{Data: data})
	}
}

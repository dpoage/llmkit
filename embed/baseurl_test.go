package embed_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/embed"
	"github.com/dpoage/llmkit/provider"
)

// pathServer answers every request with a response valid for both the
// chat-completions and embeddings wire shapes and records the exact
// request paths it saw.
type pathServer struct {
	*httptest.Server
	mu    sync.Mutex
	paths []string
}

func newPathServer(t *testing.T) *pathServer {
	t.Helper()
	ps := &pathServer{}
	ps.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ps.mu.Lock()
		ps.paths = append(ps.paths, r.URL.Path)
		ps.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
			"data": [{"embedding": [0.5, 0.25], "index": 0}],
			"embeddings": [[0.5, 0.25]]
		}`))
	}))
	t.Cleanup(ps.Close)
	return ps
}

// take returns the paths recorded so far and clears them.
func (ps *pathServer) take() []string {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	out := ps.paths
	ps.paths = nil
	return out
}

// TestConfigBaseURL_RequestPath pins the BaseURL join per backend against
// the exact path a fake server sees. The OpenAI-compatible backend
// appends "/embeddings" to a BaseURL that already carries "/v1"; a root
// BaseURL is not corrected, so it reaches "/embeddings". The Ollama
// backend appends "/api/embed" to the server root.
func TestConfigBaseURL_RequestPath(t *testing.T) {
	tests := []struct {
		name    string
		backend embed.Backend
		suffix  string // appended to the server URL to form Config.BaseURL
		want    string
	}{
		{"openai-compatible/v1", embed.BackendOpenAICompatible, "/v1", "/v1/embeddings"},
		{"openai-compatible/v1 trailing slash", embed.BackendOpenAICompatible, "/v1/", "/v1/embeddings"},
		{"openai-compatible/root", embed.BackendOpenAICompatible, "", "/embeddings"},
		{"openai-compatible/root trailing slash", embed.BackendOpenAICompatible, "/", "/embeddings"},
		{"ollama/root", embed.BackendOllama, "", "/api/embed"},
		{"ollama/root trailing slash", embed.BackendOllama, "/", "/api/embed"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := newPathServer(t)
			emb, err := embed.New(embed.Config{Backend: tc.backend, Model: "m", BaseURL: srv.URL + tc.suffix})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := emb.Embed(context.Background(), "hello"); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			got := srv.take()
			if len(got) != 1 || got[0] != tc.want {
				t.Fatalf("request paths = %q, want exactly [%q]", got, tc.want)
			}
		})
	}
}

// TestConfigBaseURL_SharedWithProviderSpec pins that one BaseURL string
// serves both a provider.Spec (openai-compatible) and an
// embed.Config (openai-compatible) against one server: the chat request
// lands on "/v1/chat/completions" and the embeddings request on
// "/v1/embeddings", and both succeed.
func TestConfigBaseURL_SharedWithProviderSpec(t *testing.T) {
	srv := newPathServer(t)
	baseURL := srv.URL + "/v1"

	client, err := provider.New(context.Background(), provider.Spec{
		Type: provider.TypeOpenAICompatible, Model: "m", BaseURL: baseURL, Secret: "k",
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}
	if _, err := client.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.UserMessage(llmkit.Text("hi"))},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	emb, err := embed.New(embed.Config{Backend: embed.BackendOpenAICompatible, Model: "m", BaseURL: baseURL, Secret: "k"})
	if err != nil {
		t.Fatalf("embed.New: %v", err)
	}
	if _, err := emb.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	got := srv.take()
	want := []string{"/v1/chat/completions", "/v1/embeddings"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("request paths = %q, want %q", got, want)
	}
}

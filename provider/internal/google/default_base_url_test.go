package google

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"
)

// hostRecorder records the host of every request and answers with a canned
// Gemini success body. It never dials.
type hostRecorder struct {
	mu    sync.Mutex
	hosts []string
}

func (h *hostRecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	h.mu.Lock()
	h.hosts = append(h.hosts, req.URL.Host)
	h.mu.Unlock()
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(mockTextBody("ok", 1, 1))),
		Request:    req,
	}, nil
}

// TestNew_BaseURLBeatsGenaiDefaultTier pins that New always passes genai an
// explicit base URL: with a process-global genai.SetDefaultBaseURLs tier
// pointing at a poison host, an empty Options.BaseURL still sends to the
// Gemini vendor host, and a set Options.BaseURL sends to that URL. It
// mutates genai's package state, so it must not run in parallel.
func TestNew_BaseURLBeatsGenaiDefaultTier(t *testing.T) {
	genai.SetDefaultBaseURLs(genai.BaseURLParameters{
		GeminiURL: "https://poison-global-tier.invalid/",
		VertexURL: "https://poison-global-tier.invalid/",
	})
	t.Cleanup(func() { genai.SetDefaultBaseURLs(genai.BaseURLParameters{}) })

	for _, tc := range []struct {
		name    string
		baseURL string
		want    string
	}{
		{"empty BaseURL", "", "generativelanguage.googleapis.com"},
		{"explicit BaseURL", "https://explicit.example.com/", "explicit.example.com"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &hostRecorder{}
			cl, err := New(context.Background(), "gemini-test", Options{
				APIKey: "k", BaseURL: tc.baseURL, HTTPClient: &http.Client{Transport: rec},
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := cl.Complete(context.Background(), simpleRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			rec.mu.Lock()
			defer rec.mu.Unlock()
			if len(rec.hosts) != 1 || rec.hosts[0] != tc.want {
				t.Errorf("request hosts = %q, want [%q]", rec.hosts, tc.want)
			}
		})
	}
}

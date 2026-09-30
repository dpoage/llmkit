package provider

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"
)

// defaultClientTap replaces http.DefaultClient.Transport for the duration of
// t with a forwarder that counts every request sent through the process-
// global client and passes it on to the original transport, so a client that
// wrongly falls back to http.DefaultClient still completes (and the count,
// not a transport error, is what shows the fallback). The swap is restored
// in t.Cleanup. The caller must not run in parallel: http.DefaultClient is
// process-global.
func defaultClientTap(t *testing.T) *atomic.Int64 {
	t.Helper()
	var hits atomic.Int64
	orig := http.DefaultClient.Transport
	next := orig
	if next == nil {
		next = http.DefaultTransport
	}
	http.DefaultClient.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		hits.Add(1)
		return next.RoundTrip(req)
	})
	t.Cleanup(func() { http.DefaultClient.Transport = orig })
	return &hits
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// vendorEnvSources lists the OpenAI and Google variables
// TestNew_HermeticConstruction scrubs, plus anthropicEnvSources.
var vendorEnvSources = append([]string{
	"OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
	"OPENAI_CUSTOM_HEADERS", "OPENAI_BASE_URL", "OPENAI_ADMIN_KEY",
	"OPENAI_WEBHOOK_SECRET",
	"GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
	"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION",
	"GOOGLE_GEMINI_BASE_URL",
}, anthropicEnvSources...)

// TestNew_NilHTTPClientNeverUsesDefaultClient pins, for every provider Type,
// that a client built by New with Options.HTTPClient nil sends its request
// to Spec.BaseURL and sends none through http.DefaultClient. Options.HTTPClient
// documents nil as "each adapter's own plain http.Client"; an adapter that
// fell back to the process-global client would let any package that swaps
// http.DefaultClient.Transport (a tracer, a test recorder, a proxy shim)
// intercept the caller's credentials. Not parallel: it swaps
// http.DefaultClient.Transport.
func TestNew_NilHTTPClientNeverUsesDefaultClient(t *testing.T) {
	cases := []struct {
		name string
		typ  Type
		wire string
	}{
		{"anthropic", TypeAnthropic, "anthropic"},
		{"openai", TypeOpenAI, "openai"},
		{"openai-compatible", TypeOpenAICompatible, "openai-compatible"},
		{"google", TypeGoogle, "google"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			isolateEnv(t, vendorEnvSources...)
			t.Setenv("HOME", t.TempDir())

			var served atomic.Int64
			baseURL := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				served.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(mockTextBody(tc.wire, "ok", 1, 1)))
			})
			viaDefault := defaultClientTap(t)

			spec := Spec{Type: tc.typ, Model: "test-model", Secret: "sk-real-secret", BaseURL: baseURL}
			client, err := New(context.Background(), spec, Options{})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if served.Load() == 0 {
				t.Error("no request reached the httptest server")
			}
			if n := viaDefault.Load(); n != 0 {
				t.Errorf("%d request(s) went through http.DefaultClient; a nil Options.HTTPClient must give the adapter its own http.Client", n)
			}
		})
	}
}

package google

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
)

// int32Server answers both endpoints (unary and SSE) with a one-chunk
// success, counting requests and keeping the last request body.
func int32Server(t *testing.T, hits *atomic.Int32, captured *map[string]any) string {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		body, _ := io.ReadAll(r.Body)
		*captured = nil
		_ = json.Unmarshal(body, captured)
		if strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+mockTextBody("ok", 1, 1)+"\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, mockTextBody("ok", 1, 1))
	})
}

// int32Entries runs one request through Complete and through Stream.
func int32Entries(t *testing.T, base string, req llmkit.Request) map[string]error {
	t.Helper()
	client, err := New(context.Background(), "gemini-test", Options{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, cerr := client.Complete(context.Background(), req)
	_, serr := client.(llmkit.StreamingClient).Stream(context.Background(), req, nil)
	return map[string]error{"Complete": cerr, "Stream": serr}
}

func generationConfig(t *testing.T, captured map[string]any) map[string]any {
	t.Helper()
	gen, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing or wrong type: %T", captured["generationConfig"])
	}
	return gen
}

// TestMaxTokensAboveInt32IsRefused: genai carries maxOutputTokens as int32,
// so a larger MaxTokens must be refused pre-wire rather than wrapping (int32
// wraps 1<<31 to -2147483648 and 1<<40 to 0). Complete and Stream.
func TestMaxTokensAboveInt32IsRefused(t *testing.T) {
	for _, mt := range []int64{math.MaxInt32 + 1, 1 << 40, math.MaxInt64} {
		if int64(int(mt)) != mt {
			continue // int is 32 bits here: the value is not representable
		}
		var hits atomic.Int32
		var captured map[string]any
		base := int32Server(t, &hits, &captured)
		req := simpleRequest()
		req.MaxTokens = int(mt)
		for entry, err := range int32Entries(t, base, req) {
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("MaxTokens=%d %s: err = %v; want ErrInvalidRequest", mt, entry, err)
			}
		}
		if n := hits.Load(); n != 0 {
			t.Errorf("MaxTokens=%d: server saw %d requests; a refused request must send none", mt, n)
		}
	}
}

// TestMaxTokensInt32MaxGoesOut: the boundary itself is legal and reaches the
// wire unchanged.
func TestMaxTokensInt32MaxGoesOut(t *testing.T) {
	var hits atomic.Int32
	var captured map[string]any
	base := int32Server(t, &hits, &captured)
	req := simpleRequest()
	req.MaxTokens = math.MaxInt32
	for entry, err := range int32Entries(t, base, req) {
		if err != nil {
			t.Fatalf("%s: %v", entry, err)
		}
		if got := generationConfig(t, captured)["maxOutputTokens"]; got != float64(math.MaxInt32) {
			t.Errorf("%s: maxOutputTokens = %v; want %d", entry, got, math.MaxInt32)
		}
	}
}

// TestSeedOutsideInt32IsRefused pins the Seed range refusal: genai carries
// seed as int32, so an out-of-range value must be refused, not truncated to
// a different deterministic seed. The int32 bounds themselves go out.
func TestSeedOutsideInt32IsRefused(t *testing.T) {
	for _, seed := range []int64{math.MaxInt32 + 1, math.MinInt32 - 1, math.MaxInt64, math.MinInt64} {
		var hits atomic.Int32
		var captured map[string]any
		base := int32Server(t, &hits, &captured)
		req := simpleRequest()
		req.Seed = &seed
		for entry, err := range int32Entries(t, base, req) {
			if !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("Seed=%d %s: err = %v; want ErrInvalidRequest", seed, entry, err)
			}
		}
		if n := hits.Load(); n != 0 {
			t.Errorf("Seed=%d: server saw %d requests; a refused request must send none", seed, n)
		}
	}
	for _, seed := range []int64{math.MaxInt32, math.MinInt32} {
		var hits atomic.Int32
		var captured map[string]any
		base := int32Server(t, &hits, &captured)
		req := simpleRequest()
		req.Seed = &seed
		for entry, err := range int32Entries(t, base, req) {
			if err != nil {
				t.Fatalf("Seed=%d %s: %v", seed, entry, err)
			}
			if got := generationConfig(t, captured)["seed"]; got != float64(seed) {
				t.Errorf("Seed=%d %s: wire seed = %v", seed, entry, got)
			}
		}
	}
}

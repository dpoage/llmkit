package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fastRetry is a deterministic, near-instant retry policy for tests.
var fastRetry = RetryConfig{
	MaxAttempts: 3,
	BaseDelay:   time.Millisecond,
	MaxDelay:    5 * time.Millisecond,
}

// =============================================================================
// Timeout Enforcement
// =============================================================================

func TestOllamaEmbedder_TimeoutEnforced(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		time.Sleep(500 * time.Millisecond)
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1}}})
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{
		Embedder: "ollama",
		Model:    "m",
		URL:      srv.URL,
		Timeout:  50 * time.Millisecond,
		Retry:    RetryConfig{MaxAttempts: 1},
	})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	start := time.Now()
	_, err = emb.Embed(context.Background(), "x")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("Embed took %v; timeout not enforced", elapsed)
	}
	var ne net.Error
	if !errors.As(err, &ne) || !ne.Timeout() {
		t.Errorf("expected timeout-classified net.Error, got %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("requests = %d, want 1 (retry disabled)", got)
	}
}

func TestOpenAIEmbedder_TimeoutEnforced(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		time.Sleep(500 * time.Millisecond)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	}))
	defer srv.Close()

	emb, err := NewOpenAICompatibleEmbedder(Config{
		Embedder: "openai-compatible",
		Model:    "m",
		URL:      srv.URL,
		Timeout:  50 * time.Millisecond,
		Retry:    RetryConfig{MaxAttempts: 1},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}

	start := time.Now()
	_, err = emb.Embed(context.Background(), "x")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("Embed took %v; timeout not enforced", elapsed)
	}
	var ne net.Error
	if !errors.As(err, &ne) || !ne.Timeout() {
		t.Errorf("expected timeout-classified net.Error, got %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("requests = %d, want 1 (retry disabled)", got)
	}
}

// =============================================================================
// Retry Policy
// =============================================================================

func TestOllamaEmbedder_RetryOn5xx_ThenSuccess(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("transient"))
			return
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1, 0.2}}})
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	got, err := emb.Embed(context.Background(), "x")
	if err != nil {
		t.Fatalf("Embed after retry: %v", err)
	}
	if len(got) != 2 || got[0] != 0.1 {
		t.Errorf("Embed = %v, want [0.1 0.2]", got)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("requests = %d, want 2 (one 500 then success)", got)
	}
}

func TestOllamaEmbedder_RetryGivesUpAfterMaxAttempts(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("down"))
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	_, err = emb.Embed(context.Background(), "x")
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("requests = %d, want 3 (MaxAttempts is total attempts)", got)
	}
}

func TestOllamaEmbedder_NoRetryOn400(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	if _, err := emb.Embed(context.Background(), "x"); err == nil {
		t.Fatal("expected error for 400")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("requests = %d, want 1 (4xx must not retry)", got)
	}
}

func TestOllamaEmbedder_RetryAfterSeconds_Honored(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1}}})
	}))
	defer srv.Close()

	// A wrong implementation that ignores Retry-After would back off for
	// BaseDelay=30s; honoring it retries immediately.
	emb, err := NewOllamaEmbedder(Config{
		Embedder: "ollama",
		Model:    "m",
		URL:      srv.URL,
		Retry:    RetryConfig{MaxAttempts: 3, BaseDelay: 30 * time.Second, MaxDelay: 30 * time.Second},
	})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	start := time.Now()
	if _, err := emb.Embed(context.Background(), "x"); err != nil {
		t.Fatalf("Embed after Retry-After: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("Embed took %v; Retry-After not honored (fell back to exponential backoff)", elapsed)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("requests = %d, want 2", got)
	}
}

func TestOpenAIEmbedder_RetryAfterHTTPDate_Honored(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", time.Now().Add(-time.Second).UTC().Format(http.TimeFormat))
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	}))
	defer srv.Close()

	emb, err := NewOpenAICompatibleEmbedder(Config{
		Embedder: "openai-compatible",
		Model:    "m",
		URL:      srv.URL,
		Retry:    RetryConfig{MaxAttempts: 3, BaseDelay: 30 * time.Second, MaxDelay: 30 * time.Second},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}

	start := time.Now()
	if _, err := emb.Embed(context.Background(), "x"); err != nil {
		t.Fatalf("Embed after Retry-After date: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("Embed took %v; HTTP-date Retry-After not honored", elapsed)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("requests = %d, want 2", got)
	}
}

func TestOpenAIEmbedder_NoRetryWhenContextCancelled(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	}))
	defer srv.Close()

	emb, err := NewOpenAICompatibleEmbedder(Config{Embedder: "openai-compatible", Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = emb.Embed(ctx, "x")
	if err == nil {
		t.Fatal("expected error with cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("requests = %d, want 0 (cancelled context must not reach or retry the server)", got)
	}
}

// =============================================================================
// Batch Chunking
// =============================================================================

func TestOllamaEmbedder_EmbedBatch_Chunking(t *testing.T) {
	inputs := []string{"a", "b", "c", "d", "e"}
	vectorFor := func(s string) []float64 { return []float64{float64(s[0]), 1} }

	var mu sync.Mutex
	var payloads [][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaRequest
		if err := decodeBody(r, &req); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		texts := anyToStrings(req.Input)
		mu.Lock()
		payloads = append(payloads, texts)
		mu.Unlock()

		embs := make([][]float64, len(texts))
		for i, s := range texts {
			embs[i] = vectorFor(s)
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: embs})
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, MaxBatch: 2, Retry: RetryConfig{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	results, err := emb.EmbedBatch(context.Background(), inputs)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(payloads) != 3 {
		t.Fatalf("requests = %d (%v), want 3 chunks of [2 2 1]", len(payloads), payloads)
	}
	wantPayloads := [][]string{{"a", "b"}, {"c", "d"}, {"e"}}
	for i, want := range wantPayloads {
		if fmt.Sprint(payloads[i]) != fmt.Sprint(want) {
			t.Errorf("chunk %d = %v, want %v", i, payloads[i], want)
		}
	}
	if len(results) != len(inputs) {
		t.Fatalf("results = %d, want %d (index-aligned)", len(results), len(inputs))
	}
	for i, in := range inputs {
		if len(results[i]) != 2 || results[i][0] != float32(in[0]) {
			t.Errorf("results[%d] = %v, want first value %v (order preserved)", i, results[i], float32(in[0]))
		}
	}
}

func TestOpenAIEmbedder_EmbedBatch_Chunking(t *testing.T) {
	inputs := []string{"a", "b", "c", "d", "e"}
	vectorFor := func(s string) []float64 { return []float64{float64(s[0]), 1} }

	var mu sync.Mutex
	var payloads [][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openaiRequest
		if err := decodeBody(r, &req); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		payloads = append(payloads, req.Input)
		mu.Unlock()

		data := make([]openaiEmbedding, len(req.Input))
		for i, s := range req.Input {
			data[i] = openaiEmbedding{Embedding: vectorFor(s), Index: i}
		}
		jsonEncode(w, openaiResponse{Data: data})
	}))
	defer srv.Close()

	emb, err := NewOpenAICompatibleEmbedder(Config{Embedder: "openai-compatible", Model: "m", URL: srv.URL, MaxBatch: 2, Retry: RetryConfig{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}

	results, err := emb.EmbedBatch(context.Background(), inputs)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(payloads) != 3 {
		t.Fatalf("requests = %d (%v), want 3 chunks of [2 2 1]", len(payloads), payloads)
	}
	wantPayloads := [][]string{{"a", "b"}, {"c", "d"}, {"e"}}
	for i, want := range wantPayloads {
		if fmt.Sprint(payloads[i]) != fmt.Sprint(want) {
			t.Errorf("chunk %d = %v, want %v", i, payloads[i], want)
		}
	}
	for i, in := range inputs {
		if len(results[i]) != 2 || results[i][0] != float32(in[0]) {
			t.Errorf("results[%d] = %v, want first value %v (order preserved)", i, results[i], float32(in[0]))
		}
	}
}

func TestOllamaEmbedder_EmbedBatch_ChunkFailureFailsWholeCall(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		var req ollamaRequest
		_ = decodeBody(r, &req)
		texts := anyToStrings(req.Input)
		embs := make([][]float64, len(texts))
		for i := range texts {
			embs[i] = []float64{0.1}
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: embs})
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, MaxBatch: 2, Retry: RetryConfig{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	results, err := emb.EmbedBatch(context.Background(), []string{"a", "b", "c", "d"})
	if err == nil {
		t.Fatal("expected error when one chunk fails")
	}
	if results != nil {
		t.Errorf("results = %v, want nil (no partial results)", results)
	}
}

// =============================================================================
// Cache: LRU eviction, counters, concurrency
// =============================================================================

func TestCachedEmbedder_LRU_EvictionOrderAndCounters(t *testing.T) {
	inner := &fakeEmbedder{dims: 1, model: "m", embedFn: func(text string) []float32 {
		return []float32{float32(text[0])}
	}}
	c := NewCachedEmbedder(inner, 2)

	assertStats := func(wantHits, wantMisses, wantSize int64) {
		t.Helper()
		st := c.Stats()
		if st.Hits != wantHits || st.Misses != wantMisses || st.Size != wantSize {
			t.Fatalf("Stats = {Hits:%d Misses:%d Size:%d}, want {%d %d %d}", st.Hits, st.Misses, st.Size, wantHits, wantMisses, wantSize)
		}
	}

	// Fill: a, b.
	if v, err := c.Embed(context.Background(), "a"); err != nil || v[0] != 'a' {
		t.Fatalf("Embed(a) = %v, %v", v, err)
	}
	if v, err := c.Embed(context.Background(), "b"); err != nil || v[0] != 'b' {
		t.Fatalf("Embed(b) = %v, %v", v, err)
	}
	assertStats(0, 2, 2)

	// Touch a: a becomes most recent, b is now LRU.
	if _, err := c.Embed(context.Background(), "a"); err != nil {
		t.Fatalf("Embed(a) again: %v", err)
	}
	assertStats(1, 2, 2)

	// Insert c: evicts b (the least recently USED entry), keeps a.
	if v, err := c.Embed(context.Background(), "c"); err != nil || v[0] != 'c' {
		t.Fatalf("Embed(c) = %v, %v", v, err)
	}
	assertStats(1, 3, 2)
	if c.Len() != 2 {
		t.Errorf("Len = %d, want 2 (bounded)", c.Len())
	}

	// a is a hit: eviction chose b (recently used) over a (older insert),
	// so the policy is LRU, not FIFO.
	if v, err := c.Embed(context.Background(), "a"); err != nil || v[0] != 'a' {
		t.Fatalf("Embed(a) after eviction = %v, %v", v, err)
	}
	assertStats(2, 3, 2)

	// b was evicted: a miss. This insert evicts c (now the LRU entry).
	if v, err := c.Embed(context.Background(), "b"); err != nil || v[0] != 'b' {
		t.Fatalf("Embed(b) after eviction = %v, %v", v, err)
	}

	// c was evicted in turn: a miss.
	if v, err := c.Embed(context.Background(), "c"); err != nil || v[0] != 'c' {
		t.Fatalf("Embed(c) after eviction = %v, %v", v, err)
	}
	assertStats(2, 5, 2)
}

func TestCachedEmbedder_ConcurrentAccess(t *testing.T) {
	inner := &fakeEmbedder{
		dims:  4,
		model: "m",
		embedFn: func(text string) []float32 {
			return []float32{float32(text[0]), float32(len(text)), 0, 0}
		},
		embedBatchFn: func(texts []string) [][]float32 {
			out := make([][]float32, len(texts))
			for i, s := range texts {
				out[i] = []float32{float32(s[0]), float32(len(s)), 0, 0}
			}
			return out
		},
	}
	c := NewCachedEmbedder(inner, 8)

	const goroutines, iters = 16, 50
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				key := fmt.Sprintf("text-%d", (g*i)%10)
				if i%3 == 0 {
					_, _ = c.EmbedBatch(context.Background(), []string{key, fmt.Sprintf("b-%d", i%5)})
				} else {
					_, _ = c.Embed(context.Background(), key)
				}
				if i%25 == 0 {
					_ = c.Stats()
					_ = c.Len()
				}
			}
		}(g)
	}
	wg.Wait()

	st := c.Stats()
	if st.Hits+st.Misses == 0 {
		t.Error("expected nonzero counters after concurrent use")
	}
	if st.Size > 8 {
		t.Errorf("Size = %d, want <= 8 (bound enforced under concurrency)", st.Size)
	}
	if int(st.Size) != c.Len() {
		t.Errorf("Stats.Size = %d, Len = %d; disagree", st.Size, c.Len())
	}

	c.Clear()
	if st := c.Stats(); st.Size != 0 {
		t.Errorf("Size after Clear = %d, want 0", st.Size)
	}
	if st := c.Stats(); st.Hits+st.Misses == 0 {
		t.Error("Clear must keep lifetime counters")
	}
}

// =============================================================================
// Dimension Consistency
// =============================================================================

func TestOllamaEmbedder_DimensionMismatch_Configured(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1, 0.2, 0.3, 0.4}}})
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, Dimensions: 3, Retry: RetryConfig{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	for i := 0; i < 2; i++ {
		_, err := emb.Embed(context.Background(), "x")
		if err == nil {
			t.Fatalf("call %d: expected dimension mismatch error", i+1)
		}
		if !strings.Contains(err.Error(), "got 4") || !strings.Contains(err.Error(), "want 3") {
			t.Errorf("call %d: error %q must name expected and actual dimensions", i+1, err)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("requests = %d, want 2 (mismatch is not retried, checked every call)", got)
	}
}

func TestOpenAIEmbedder_DimensionMismatch_AutoDetect(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dim := 3
		if calls.Add(1) > 1 {
			dim = 4
		}
		vec := make([]float64, dim)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: vec, Index: 0}}})
	}))
	defer srv.Close()

	emb, err := NewOpenAICompatibleEmbedder(Config{Embedder: "openai-compatible", Model: "m", URL: srv.URL, Retry: RetryConfig{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}

	if _, err := emb.Embed(context.Background(), "first"); err != nil {
		t.Fatalf("first Embed: %v", err)
	}
	if emb.Dimensions() != 3 {
		t.Fatalf("Dimensions() = %d, want 3 (auto-detected)", emb.Dimensions())
	}

	for i := 0; i < 2; i++ {
		_, err := emb.Embed(context.Background(), "later")
		if err == nil {
			t.Fatalf("call %d: expected mismatch vs detected dimensions", i+1)
		}
		if !strings.Contains(err.Error(), "got 4") || !strings.Contains(err.Error(), "want 3") {
			t.Errorf("call %d: error %q must name detected and actual dimensions", i+1, err)
		}
	}
}

// =============================================================================
// OpenAI index/count validation
// =============================================================================

func TestOpenAIEmbedder_EmbedBatch_CountMismatch(t *testing.T) {
	tests := []struct {
		name    string
		data    []openaiEmbedding
		wantErr string
	}{
		{
			name:    "too few entries",
			data:    []openaiEmbedding{{Embedding: []float64{1}, Index: 0}, {Embedding: []float64{2}, Index: 1}},
			wantErr: "expected 3 embeddings, got 2",
		},
		{
			name: "too many entries",
			data: []openaiEmbedding{
				{Embedding: []float64{1}, Index: 0},
				{Embedding: []float64{2}, Index: 1},
				{Embedding: []float64{3}, Index: 2},
				{Embedding: []float64{4}, Index: 3},
			},
			wantErr: "expected 3 embeddings, got 4",
		},
		{
			name: "duplicate index",
			data: []openaiEmbedding{
				{Embedding: []float64{1}, Index: 0},
				{Embedding: []float64{2}, Index: 0},
				{Embedding: []float64{3}, Index: 2},
			},
			wantErr: "duplicate embedding at index 0",
		},
		{
			name:    "no entries",
			data:    nil,
			wantErr: "expected 3 embeddings, got 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				jsonEncode(w, openaiResponse{Data: tt.data})
			}))
			defer srv.Close()

			emb, err := NewOpenAICompatibleEmbedder(Config{Embedder: "openai-compatible", Model: "m", URL: srv.URL, Retry: RetryConfig{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
			}

			_, err = emb.EmbedBatch(context.Background(), []string{"a", "b", "c"})
			if err == nil {
				t.Fatalf("expected error %q, got success", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err, tt.wantErr)
			}
		})
	}
}

// =============================================================================
// HTTPClient injection precedence
// =============================================================================

func TestOpenAIEmbedder_HTTPClientInjection(t *testing.T) {
	respond := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	})

	// A client with its own short timeout is used as-is: the slow server
	// must trip it.
	timedClient := &http.Client{Timeout: 50 * time.Millisecond}
	srv1 := httptest.NewServer(respond)
	defer srv1.Close()
	emb1, err := NewOpenAICompatibleEmbedder(Config{
		Embedder: "openai-compatible", Model: "m", URL: srv1.URL,
		HTTPClient: timedClient, Timeout: 5 * time.Second, Retry: RetryConfig{MaxAttempts: 1},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}
	start := time.Now()
	if _, err := emb1.Embed(context.Background(), "x"); err == nil {
		t.Error("expected the injected client's own timeout to fire")
	} else if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Embed took %v; injected client timeout not in effect", elapsed)
	}

	// A client without a timeout is NOT topped up by Config.Timeout: the
	// slow request must succeed despite Timeout=50ms.
	untimedClient := &http.Client{}
	srv2 := httptest.NewServer(respond)
	defer srv2.Close()
	emb2, err := NewOpenAICompatibleEmbedder(Config{
		Embedder: "openai-compatible", Model: "m", URL: srv2.URL,
		HTTPClient: untimedClient, Timeout: 50 * time.Millisecond, Retry: RetryConfig{MaxAttempts: 1},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleEmbedder: %v", err)
	}
	if _, err := emb2.Embed(context.Background(), "x"); err != nil {
		t.Errorf("Embed with injected untimed client: %v (Timeout must not be applied on top)", err)
	}
}

// =============================================================================
// Config: LoadConfig errors and new fields
// =============================================================================

func TestLoadConfig_InvalidValues(t *testing.T) {
	tests := []struct {
		name string
		varb string
		val  string
	}{
		{"bad dimensions", "_EMBED_DIMENSIONS", "abc"},
		{"bad timeout", "_EMBED_TIMEOUT", "fast"},
		{"bad max batch", "_EMBED_MAX_BATCH", "10x"},
		{"bad cache size", "_EMBED_CACHE_SIZE", "1.5"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("BADVALS"+tt.varb, tt.val)
			cfg, err := LoadConfig("badvals")
			if err == nil {
				t.Fatalf("LoadConfig with %s=%q: expected error, got config %+v", tt.varb, tt.val, cfg)
			}
			if !strings.Contains(err.Error(), tt.varb) {
				t.Errorf("error %q must name the offending variable %s", err, tt.varb)
			}
			if cfg != (Config{}) {
				t.Errorf("Config on error = %+v, want zero", cfg)
			}
		})
	}
}

func TestLoadConfig_NewFields(t *testing.T) {
	t.Setenv("NEWAPP_EMBED_TIMEOUT", "30s")
	t.Setenv("NEWAPP_EMBED_MAX_BATCH", "10")
	t.Setenv("NEWAPP_EMBED_CACHE_SIZE", "100")

	cfg, err := LoadConfig("newapp") // prefix case-insensitive
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Timeout != 30*time.Second {
		t.Errorf("Timeout = %v, want 30s", cfg.Timeout)
	}
	if cfg.MaxBatch != 10 {
		t.Errorf("MaxBatch = %d, want 10", cfg.MaxBatch)
	}
	if cfg.CacheSize != 100 {
		t.Errorf("CacheSize = %d, want 100", cfg.CacheSize)
	}
}

func TestConfig_Validate_Negatives(t *testing.T) {
	base := Config{Embedder: "ollama", Model: "m", URL: "http://localhost"}

	if err := base.Validate(); err != nil {
		t.Fatalf("base config invalid: %v", err)
	}

	negBatch := base
	negBatch.MaxBatch = -1
	if err := negBatch.Validate(); err == nil {
		t.Error("negative MaxBatch must be rejected")
	}

	negSize := base
	negSize.CacheSize = -1
	if err := negSize.Validate(); err == nil {
		t.Error("negative CacheSize must be rejected")
	}
}

// =============================================================================
// Test Helpers
// =============================================================================

// jsonEncode writes v as a JSON response body.
func jsonEncode(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// decodeBody decodes r's JSON body into v.
func decodeBody(r *http.Request, v any) error {
	return json.NewDecoder(r.Body).Decode(v)
}

// anyToStrings converts a decoded JSON array (from an `any` field) to strings.
func anyToStrings(v any) []string {
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, len(raw))
	for i, e := range raw {
		s, ok := e.(string)
		if !ok {
			return nil
		}
		out[i] = s
	}
	return out
}

// =============================================================================
// Fix round: response-count validation, policy normalization, defaults
// =============================================================================

func TestOllamaEmbedder_EmbedBatch_CountMismatch(t *testing.T) {
	tests := []struct {
		name    string
		inputs  []string
		embs    [][]float64
		wantErr string
	}{
		{
			name:    "too few vectors",
			inputs:  []string{"a", "b", "c"},
			embs:    [][]float64{{97}, {98}},
			wantErr: "expected 3 embeddings, got 2",
		},
		{
			name:    "too many vectors",
			inputs:  []string{"a", "b"},
			embs:    [][]float64{{97}, {98}, {99}, {100}},
			wantErr: "expected 2 embeddings, got 4",
		},
		{
			name:    "empty response",
			inputs:  []string{"a", "b"},
			embs:    nil,
			wantErr: "expected 2 embeddings, got 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				jsonEncode(w, ollamaResponse{Model: "m", Embeddings: tt.embs})
			}))
			defer srv.Close()

			emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, Retry: RetryConfig{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("NewOllamaEmbedder: %v", err)
			}

			results, err := emb.EmbedBatch(context.Background(), tt.inputs)
			if err == nil {
				t.Fatalf("expected error %q, got success with %d results", tt.wantErr, len(results))
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err, tt.wantErr)
			}
		})
	}

	// The single-text (string) request form must expect exactly one vector.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{97}, {98}}})
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{Embedder: "ollama", Model: "m", URL: srv.URL, Retry: RetryConfig{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}
	if _, err := emb.Embed(context.Background(), "a"); err == nil || !strings.Contains(err.Error(), "expected 1 embeddings, got 2") {
		t.Errorf("Embed overlong response: err = %v, want expected-1-got-2 error", err)
	}
}

func TestCachedEmbedder_EmbedBatch_ShortInnerResult(t *testing.T) {
	inner := &fakeEmbedder{
		dims:  1,
		model: "m",
		embedBatchFn: func(texts []string) [][]float32 {
			return [][]float32{{float32(texts[0][0])}} // one vector for two texts
		},
	}
	c := NewCachedEmbedder(inner, 0)

	results, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatalf("expected error for short inner result, got %v", results)
	}
	if !strings.Contains(err.Error(), "1 results for 2 texts") {
		t.Errorf("error = %q, want count-mismatch detail", err)
	}
	if results != nil {
		t.Errorf("results = %v, want nil", results)
	}
}

func TestRetry_MaxAttemptsOnlyPolicy_Bounded(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1}}})
	}))
	defer srv.Close()

	// MaxAttempts-only policy: unset knobs are normalized, so the call must
	// complete in bounded time. A policy that mapped zero BaseDelay to an
	// overflow value would sleep for years here.
	emb, err := NewOllamaEmbedder(Config{
		Embedder: "ollama",
		Model:    "m",
		URL:      srv.URL,
		Retry:    RetryConfig{MaxAttempts: 4},
	})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	start := time.Now()
	if _, err := emb.Embed(context.Background(), "x"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Fatalf("Embed took %v; retry policy not bounded", elapsed)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("requests = %d, want 3", got)
	}
}

func TestRetry_PolicyNormalization(t *testing.T) {
	p := Config{Retry: RetryConfig{MaxAttempts: 5}}.retryPolicy()
	def := DefaultRetryConfig()
	if p.Jitter != def.Jitter {
		t.Errorf("MaxAttempts-only policy Jitter = %v, want default %v", p.Jitter, def.Jitter)
	}
	if p.BaseDelay != def.BaseDelay || p.MaxDelay != def.MaxDelay {
		t.Errorf("MaxAttempts-only policy = %+v, want delays normalized from defaults", p)
	}
	// Literal pin: the default policy must include 20% jitter, so a
	// DefaultRetryConfig that drops Jitter fails here rather than silently
	// making the default unreachable via Config{}.
	if got := (Config{}).retryPolicy(); got != (RetryConfig{MaxAttempts: 3, BaseDelay: 500 * time.Millisecond, MaxDelay: 30 * time.Second, Jitter: 0.2}) {
		t.Errorf("Config{}.retryPolicy() = %+v, want the documented default {3, 500ms, 30s, 0.2}", got)
	}

	// Retry-After is capped at MaxDelay even under a normalized policy
	// whose explicit MaxDelay is zero.
	after := backoffDelay(p, 1, time.Hour, true)
	if after != def.MaxDelay {
		t.Errorf("Retry-After backoff = %v, want capped at %v", after, def.MaxDelay)
	}
	d := backoffDelay(p, 3, 0, false)
	if d <= 0 || d > def.MaxDelay {
		t.Errorf("exponential backoff = %v, want in (0, %v]", d, def.MaxDelay)
	}

	if got := (Config{Retry: RetryConfig{MaxAttempts: 3, Jitter: 5}}).retryPolicy().Jitter; got != 1 {
		t.Errorf("Jitter 5 clamped to %v, want 1", got)
	}
	if got := (Config{Retry: RetryConfig{MaxAttempts: 3, Jitter: -2}}).retryPolicy().Jitter; got != 0.2 {
		t.Errorf("Jitter -2 filled to %v, want default 0.2", got)
	}

	// Zero BaseDelay must stay zero (immediate retry), never be mistaken
	// for int64 overflow.
	if d := backoffDelay(RetryConfig{BaseDelay: 0}, 3, 0, false); d != 0 {
		t.Errorf("backoffDelay with zero BaseDelay = %v, want 0", d)
	}
}

func TestDefaultClientTimeout(t *testing.T) {
	// Literal, not DefaultEmbedTimeout: mutating the constant to 0 must
	// fail here.
	if DefaultEmbedTimeout != 60*time.Second {
		t.Errorf("DefaultEmbedTimeout = %v, want 60s", DefaultEmbedTimeout)
	}
	if got := (Config{}).httpClient().Timeout; got != 60*time.Second {
		t.Errorf("default client timeout = %v, want 60s", got)
	}
	if got := (Config{Timeout: -1}).httpClient().Timeout; got != 60*time.Second {
		t.Errorf("negative Timeout client = %v, want 60s", got)
	}
	injected := &http.Client{}
	if got := (Config{HTTPClient: injected, Timeout: 7 * time.Second}).httpClient(); got != injected {
		t.Error("httpClient must return the injected client as-is")
	}
}

func TestOllamaEmbedder_TimeoutRetriedAsTransient(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()

	emb, err := NewOllamaEmbedder(Config{
		Embedder: "ollama",
		Model:    "m",
		URL:      srv.URL,
		Timeout:  30 * time.Millisecond,
		Retry:    RetryConfig{MaxAttempts: 2, BaseDelay: time.Millisecond},
	})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}

	_, err = emb.Embed(context.Background(), "x")
	if err == nil {
		t.Fatal("expected timeout error")
	}
	var ne net.Error
	if !errors.As(err, &ne) || !ne.Timeout() {
		t.Errorf("expected timeout-classified error, got %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("requests = %d, want 2 (timeouts are transient)", got)
	}
}

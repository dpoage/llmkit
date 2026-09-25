package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit/retry"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var fastRetry = retry.Config{
	MaxAttempts: 3,
	BaseDelay:   time.Millisecond,
	MaxDelay:    5 * time.Millisecond,
}

func TestOllamaBackend_RequestTimeoutEnforced(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		time.Sleep(500 * time.Millisecond)
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1}}})
	}))
	defer srv.Close()

	emb, err := New(Config{
		Backend: BackendOllama,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 1, RequestTimeout: 50 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_RequestTimeoutEnforced(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		time.Sleep(500 * time.Millisecond)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	}))
	defer srv.Close()

	emb, err := New(Config{
		Backend: BackendOpenAICompatible,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 1, RequestTimeout: 50 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOllamaBackend_RetryOn5xx_ThenSuccess(t *testing.T) {
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

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOllamaBackend_RetryGivesUpAfterMaxAttempts(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("down"))
	}))
	defer srv.Close()

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = emb.Embed(context.Background(), "x")
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("requests = %d, want 3 (MaxAttempts is total attempts)", got)
	}
}

func TestOllamaBackend_NoRetryOn400(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	defer srv.Close()

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := emb.Embed(context.Background(), "x"); err == nil {
		t.Fatal("expected error for 400")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("requests = %d, want 1 (4xx must not retry)", got)
	}
}

func TestOllamaBackend_RetryAfterSeconds_Honored(t *testing.T) {
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

	emb, err := New(Config{
		Backend: BackendOllama,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 3, BaseDelay: 30 * time.Second, MaxDelay: 30 * time.Second},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_RetryAfterHTTPDate_Honored(t *testing.T) {
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

	emb, err := New(Config{
		Backend: BackendOpenAICompatible,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 3, BaseDelay: 30 * time.Second, MaxDelay: 30 * time.Second},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_NoRetryWhenContextCancelled(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	}))
	defer srv.Close()

	emb, err := New(Config{Backend: BackendOpenAICompatible, Model: "m", URL: srv.URL, Retry: fastRetry})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOllamaBackend_EmbedBatch_Chunking(t *testing.T) {
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
		texts := req.Input
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

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, MaxBatch: 2, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_EmbedBatch_Chunking(t *testing.T) {
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

	emb, err := New(Config{Backend: BackendOpenAICompatible, Model: "m", URL: srv.URL, MaxBatch: 2, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOllamaBackend_EmbedBatch_ChunkFailureFailsWholeCall(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		var req ollamaRequest
		_ = decodeBody(r, &req)
		embs := make([][]float64, len(req.Input))
		for i := range req.Input {
			embs[i] = []float64{0.1}
		}
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: embs})
	}))
	defer srv.Close()

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, MaxBatch: 2, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	results, err := emb.EmbedBatch(context.Background(), []string{"a", "b", "c", "d"})
	if err == nil {
		t.Fatal("expected error when one chunk fails")
	}
	if results != nil {
		t.Errorf("results = %v, want nil (no partial results)", results)
	}
}

func TestCachedEmbedder_LRU_EvictionOrderAndCounters(t *testing.T) {
	inner := &fakeEmbedder{dims: 1, model: "m", embedFn: func(text string) []float32 {
		return []float32{float32(text[0])}
	}}
	c := NewCachedEmbedder(inner, 2)

	assertStats := func(wantHits, wantMisses int64, wantLen int) {
		t.Helper()
		st := c.Stats()
		if st.Hits != wantHits || st.Misses != wantMisses {
			t.Fatalf("Stats = {Hits:%d Misses:%d}, want {%d %d}", st.Hits, st.Misses, wantHits, wantMisses)
		}
		if got := c.Len(); got != wantLen {
			t.Fatalf("Len = %d, want %d", got, wantLen)
		}
	}

	if v, err := c.Embed(context.Background(), "a"); err != nil || v[0] != 'a' {
		t.Fatalf("Embed(a) = %v, %v", v, err)
	}
	if v, err := c.Embed(context.Background(), "b"); err != nil || v[0] != 'b' {
		t.Fatalf("Embed(b) = %v, %v", v, err)
	}
	assertStats(0, 2, 2)

	if _, err := c.Embed(context.Background(), "a"); err != nil {
		t.Fatalf("Embed(a) again: %v", err)
	}
	assertStats(1, 2, 2)

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

	if v, err := c.Embed(context.Background(), "b"); err != nil || v[0] != 'b' {
		t.Fatalf("Embed(b) after eviction = %v, %v", v, err)
	}

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
	if got := c.Len(); got > 8 {
		t.Errorf("Len = %d, want <= 8 (bound enforced under concurrency)", got)
	}

	c.Clear()
	if got := c.Len(); got != 0 {
		t.Errorf("Len after Clear = %d, want 0", got)
	}
	if st := c.Stats(); st.Hits+st.Misses == 0 {
		t.Error("Clear must keep lifetime counters")
	}
}

func TestOllamaBackend_DimensionMismatch_Configured(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{0.1, 0.2, 0.3, 0.4}}})
	}))
	defer srv.Close()

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, Dimensions: 3, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_DimensionMismatch_AutoDetect(t *testing.T) {
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

	emb, err := New(Config{Backend: BackendOpenAICompatible, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_EmbedBatch_CountMismatch(t *testing.T) {
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

			emb, err := New(Config{Backend: BackendOpenAICompatible, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("New: %v", err)
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

func TestOpenAIBackend_HTTPClientInjection(t *testing.T) {
	respond := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		jsonEncode(w, openaiResponse{Data: []openaiEmbedding{{Embedding: []float64{0.1}, Index: 0}}})
	})

	timedClient := &http.Client{Timeout: 50 * time.Millisecond}
	srv1 := httptest.NewServer(respond)
	defer srv1.Close()
	emb1, err := New(Config{
		Backend: BackendOpenAICompatible, Model: "m", URL: srv1.URL,
		HTTPClient: timedClient, Retry: retry.Config{MaxAttempts: 1, RequestTimeout: 5 * time.Second},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	start := time.Now()
	if _, err := emb1.Embed(context.Background(), "x"); err == nil {
		t.Error("expected the injected client's own timeout to fire")
	} else if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Embed took %v; injected client timeout not in effect", elapsed)
	}

	untimedClient := &http.Client{}
	srv2 := httptest.NewServer(respond)
	defer srv2.Close()
	emb2, err := New(Config{
		Backend: BackendOpenAICompatible, Model: "m", URL: srv2.URL,
		HTTPClient: untimedClient, Retry: retry.Config{MaxAttempts: 1, RequestTimeout: 50 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	start = time.Now()
	if _, err := emb2.Embed(context.Background(), "x"); err == nil {
		t.Error("expected the per-attempt RequestTimeout to bound the injected untimed client")
	} else if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Embed took %v; RequestTimeout not enforced on injected client", elapsed)
	}
}

func TestConfig_Validate_Negatives(t *testing.T) {
	base := Config{Backend: BackendOllama, Model: "m", URL: "http://localhost"}

	if err := base.Validate(); err != nil {
		t.Fatalf("base config invalid: %v", err)
	}

	negBatch := base
	negBatch.MaxBatch = -1
	if err := negBatch.Validate(); err == nil {
		t.Error("negative MaxBatch must be rejected")
	}
}

func jsonEncode(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func decodeBody(r *http.Request, v any) error {
	return json.NewDecoder(r.Body).Decode(v)
}

func TestOllamaBackend_EmbedBatch_CountMismatch(t *testing.T) {
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

			emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
			if err != nil {
				t.Fatalf("New: %v", err)
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

	// A single-text Embed call must expect exactly one vector back.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{97}, {98}}})
	}))
	defer srv.Close()

	emb, err := New(Config{Backend: BackendOllama, Model: "m", URL: srv.URL, Retry: retry.Config{MaxAttempts: 1}})
	if err != nil {
		t.Fatalf("New: %v", err)
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
			return [][]float32{{float32(texts[0][0])}}
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

	emb, err := New(Config{
		Backend: BackendOllama,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 4},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
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

// TestRetry_PolicyNormalization pins retryPolicy's Or-based resolution:
// an explicit field is kept, an unset one (<= 0, and Jitter == 0) takes
// the embed/kit default. Jitter 0 counts as unset like every other
// schedule field under Or, so both an explicit zero and a fully zero
// Config resolve to the kit's default jitter.
func TestRetry_PolicyNormalization(t *testing.T) {
	p := Config{Retry: retry.Config{MaxAttempts: 5}}.retryPolicy()
	def := retry.Default()
	if p.MaxAttempts != 5 {
		t.Errorf("explicit MaxAttempts = %d, want 5 (kept)", p.MaxAttempts)
	}
	if p.BaseDelay != def.BaseDelay || p.MaxDelay != def.MaxDelay {
		t.Errorf("MaxAttempts-only policy = %+v, want unset delays filled from retry.Default", p)
	}
	if p.RequestTimeout != 60*time.Second {
		t.Errorf("RequestTimeout = %v, want 60s (embed default)", p.RequestTimeout)
	}
	if p.Jitter != def.Jitter {
		t.Errorf("Jitter = %v, want %v (explicit 0 counts as unset under Or)", p.Jitter, def.Jitter)
	}
	got := (Config{}).retryPolicy()
	if got.MaxAttempts != 3 || got.RequestTimeout != 60*time.Second {
		t.Errorf("zero policy attempts/timeout = %d/%v, want 3/60s (embed bounds)", got.MaxAttempts, got.RequestTimeout)
	}
	if got.BaseDelay != def.BaseDelay || got.MaxDelay != def.MaxDelay || got.Jitter != def.Jitter {
		t.Errorf("Config{}.retryPolicy() = %+v, want kit-default delays including Jitter", got)
	}
}

func TestHTTPClientConstruction(t *testing.T) {
	if got := (Config{}).httpClient().Timeout; got != 0 {
		t.Errorf("default client timeout = %v, want 0 (bounded via per-attempt ctx)", got)
	}
	injected := &http.Client{}
	if got := (Config{HTTPClient: injected}).httpClient(); got != injected {
		t.Error("httpClient must return the injected client as-is")
	}
}

func TestOllamaBackend_TimeoutRetriedAsTransient(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()

	emb, err := New(Config{
		Backend: BackendOllama,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 2, BaseDelay: time.Millisecond, RequestTimeout: 30 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
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

// TestCachedEmbedder_ReturnedVectorIsACopy pins copy-on-read: a caller that
// mutates a returned vector must never corrupt subsequent reads, on both the
// Embed and EmbedBatch paths.
func TestCachedEmbedder_ReturnedVectorIsACopy(t *testing.T) {
	inner := &fakeEmbedder{
		dims:  3,
		model: "m",
		embedFn: func(text string) []float32 {
			return []float32{1, 2, 3}
		},
		embedBatchFn: func(texts []string) [][]float32 {
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{float32(i), float32(i + 1), float32(i + 2)}
			}
			return out
		},
	}
	c := NewCachedEmbedder(inner, 0)

	v1, err := c.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	v1[0] = 99
	v2, err := c.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed (cached): %v", err)
	}
	if v2[0] != 1 {
		t.Fatalf("cache aliased the returned vector: re-read = %v, want [1 2 3]", v2)
	}

	b1, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	b1[0][0] = 99
	b1[1][0] = 99
	b2, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("EmbedBatch (cached): %v", err)
	}
	if b2[0][0] != 0 || b2[1][0] != 1 {
		t.Fatalf("cache aliased batch vectors: re-read [0][0]=%v [1][0]=%v, want 0 and 1", b2[0][0], b2[1][0])
	}
}

// TestOllamaBackend_EmptyVector_AutoDetectRejected pins the guard: with
// Dimensions unset, an empty first vector is ErrEmptyVector — it must not
// latch the embedder's dimensionality to zero.
func TestOllamaBackend_EmptyVector_AutoDetectRejected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jsonEncode(w, ollamaResponse{Model: "m", Embeddings: [][]float64{{}}})
	}))
	defer srv.Close()

	emb, err := New(Config{
		Backend: BackendOllama,
		Model:   "m",
		URL:     srv.URL,
		Retry:   retry.Config{MaxAttempts: 1},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = emb.Embed(context.Background(), "x")
	if !errors.Is(err, ErrEmptyVector) {
		t.Fatalf("Embed with empty first vector = %v, want ErrEmptyVector", err)
	}
	if emb.Dimensions() != 0 {
		t.Errorf("Dimensions() = %d, want 0 (an empty vector must not be latched)", emb.Dimensions())
	}
}

// TestCheckDimensions_EmptyVectorConfigured pins the error split: with
// configured dimensions an empty vector is a dimension mismatch, not
// ErrEmptyVector — that error is specific to the auto-detect path.
func TestCheckDimensions_EmptyVectorConfigured(t *testing.T) {
	var mu sync.RWMutex
	dims := 3
	err := checkDimensions("ollama", [][]float32{{}}, &dims, &mu)
	if err == nil || errors.Is(err, ErrEmptyVector) {
		t.Fatalf("configured dims with empty vector = %v, want a dimension-mismatch error", err)
	}
	if dims != 3 {
		t.Errorf("dims = %d, want 3 (unchanged)", dims)
	}
}

// TestConfig_JitterRangeAndDefault pins Jitter under Validate and Or:
// negative and above 1 are Validate errors; 0 (explicit or left unset) is
// valid and resolves to the kit default via retryPolicy's Or, because Or
// treats Jitter 0 as unset like every other schedule field — the
// deterministic escape is Retry.Rand, not a literal zero Jitter.
func TestConfig_JitterRangeAndDefault(t *testing.T) {
	base := Config{Backend: BackendOllama, Model: "m", URL: "http://localhost"}

	zero := base
	zero.Retry = retry.Config{MaxAttempts: 2}
	if err := zero.Validate(); err != nil {
		t.Fatalf("Validate with Jitter 0: %v", err)
	}
	if got := zero.retryPolicy().Jitter; got != retry.Default().Jitter {
		t.Errorf("retryPolicy Jitter = %v, want kit default %v (0 is unset under Or)", got, retry.Default().Jitter)
	}

	neg := base
	neg.Retry = retry.Config{Jitter: -0.5}
	if err := neg.Validate(); err == nil {
		t.Error("negative Jitter must be rejected by Validate")
	}

	big := base
	big.Retry = retry.Config{Jitter: 1.5}
	if err := big.Validate(); err == nil {
		t.Error("Jitter above 1 must be rejected by Validate")
	}
}

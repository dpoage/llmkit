package embed

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

// Stats reports cumulative cache counters. Hits+Misses counts every text
// requested through the cache; Size is the current entry count. Counters
// are lifetime values and are not reset by Clear.
type Stats struct {
	Hits   int64
	Misses int64
	Size   int64
}

// CachedEmbedder wraps any Embedder with an in-memory content-hash cache.
// Cache keys are SHA-256(model + "\x00" + text) so that different models
// produce distinct cache entries for the same text.
//
// When maxSize > 0 the cache is bounded: inserting past the bound evicts the
// least recently used entry, and cache hits refresh recency. maxSize <= 0
// means unbounded.
type CachedEmbedder struct {
	inner   Embedder
	maxSize int

	mu     sync.Mutex
	cache  map[string]*list.Element // element value: *lruEntry
	lru    *list.List               // front = most recently used
	hits   int64
	misses int64
}

// lruEntry is the value stored in each list element.
type lruEntry struct {
	key string
	emb []float32
}

// NewCachedEmbedder wraps an existing Embedder with caching, bounded to
// maxSize entries (maxSize <= 0 means unbounded).
func NewCachedEmbedder(inner Embedder, maxSize int) *CachedEmbedder {
	if maxSize < 0 {
		maxSize = 0
	}
	return &CachedEmbedder{
		inner:   inner,
		maxSize: maxSize,
		cache:   make(map[string]*list.Element),
		lru:     list.New(),
	}
}

// Embed returns a cached embedding or delegates to the inner Embedder.
func (c *CachedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	key := c.cacheKey(text)

	c.mu.Lock()
	if el, ok := c.cache[key]; ok {
		c.lru.MoveToFront(el)
		c.hits++
		emb := el.Value.(*lruEntry).emb
		c.mu.Unlock()
		return emb, nil
	}
	c.misses++
	c.mu.Unlock()

	emb, err := c.inner.Embed(ctx, text)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.insert(key, emb)
	c.mu.Unlock()

	return emb, nil
}

// EmbedBatch returns cached embeddings where available and batches the
// remaining texts through the inner Embedder. Results are index-aligned
// with the input.
func (c *CachedEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	var missingTexts []string
	var missingIndices []int

	c.mu.Lock()
	for i, text := range texts {
		key := c.cacheKey(text)
		if el, ok := c.cache[key]; ok {
			c.lru.MoveToFront(el)
			c.hits++
			results[i] = el.Value.(*lruEntry).emb
		} else {
			c.misses++
			missingTexts = append(missingTexts, text)
			missingIndices = append(missingIndices, i)
		}
	}
	c.mu.Unlock()

	if len(missingTexts) == 0 {
		return results, nil
	}

	computed, err := c.inner.EmbedBatch(ctx, missingTexts)
	if err != nil {
		return nil, err
	}
	if len(computed) != len(missingTexts) {
		return nil, fmt.Errorf("embed: inner embedder returned %d results for %d texts", len(computed), len(missingTexts))
	}

	c.mu.Lock()
	for j, idx := range missingIndices {
		results[idx] = computed[j]
		c.insert(c.cacheKey(missingTexts[j]), computed[j])
	}
	c.mu.Unlock()

	return results, nil
}

// Dimensions delegates to the inner Embedder.
func (c *CachedEmbedder) Dimensions() int {
	return c.inner.Dimensions()
}

// ModelName delegates to the inner Embedder.
func (c *CachedEmbedder) ModelName() string {
	return c.inner.ModelName()
}

// Stats returns cumulative hit/miss counters and the current entry count.
func (c *CachedEmbedder) Stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return Stats{Hits: c.hits, Misses: c.misses, Size: int64(len(c.cache))}
}

// Len returns the number of entries currently in the cache.
func (c *CachedEmbedder) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.cache)
}

// Clear removes all entries from the cache. Hit/miss counters are kept.
func (c *CachedEmbedder) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[string]*list.Element)
	c.lru.Init()
}

// insert stores emb at key as the most recent entry and evicts
// least-recently-used entries past the bound. c.mu must be held.
func (c *CachedEmbedder) insert(key string, emb []float32) {
	if el, ok := c.cache[key]; ok {
		el.Value.(*lruEntry).emb = emb
		c.lru.MoveToFront(el)
		return
	}
	c.cache[key] = c.lru.PushFront(&lruEntry{key: key, emb: emb})
	for c.maxSize > 0 && len(c.cache) > c.maxSize {
		oldest := c.lru.Back()
		if oldest == nil {
			break
		}
		delete(c.cache, oldest.Value.(*lruEntry).key)
		c.lru.Remove(oldest)
	}
}

// cacheKey produces a deterministic key from model name and text content.
func (c *CachedEmbedder) cacheKey(text string) string {
	h := sha256.Sum256([]byte(c.inner.ModelName() + "\x00" + text))
	return hex.EncodeToString(h[:])
}

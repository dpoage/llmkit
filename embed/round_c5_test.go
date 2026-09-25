package embed

import (
	"context"
	"reflect"
	"testing"
)

// poisonableEmbedder is a fake Embedder that returns its own backing
// slices directly (no defensive copy) and can later mutate them in
// place, simulating an inner embedder that reuses a scratch buffer after
// a call returns.
type poisonableEmbedder struct {
	vec  []float32
	rows [][]float32
}

func (p *poisonableEmbedder) Embed(context.Context, string) ([]float32, error) {
	return p.vec, nil
}

func (p *poisonableEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	copy(out, p.rows[:len(texts)])
	return out, nil
}

func (p *poisonableEmbedder) Dimensions() int   { return len(p.vec) }
func (p *poisonableEmbedder) ModelName() string { return "poison-model" }

// poison sets every element of every backing slice to -1, simulating the
// inner embedder mutating a buffer it keeps after Embed/EmbedBatch
// returned.
func (p *poisonableEmbedder) poison() {
	for i := range p.vec {
		p.vec[i] = -1
	}
	for _, r := range p.rows {
		for i := range r {
			r[i] = -1
		}
	}
}

// TestCachedEmbedder_InsertCopiesVector pins that the cache never stores
// a vector it did not copy: an inner Embedder whose Embed/EmbedBatch
// return a slice it keeps and later mutates must not be able to corrupt
// a cache hit.
func TestCachedEmbedder_InsertCopiesVector(t *testing.T) {
	t.Run("Embed", func(t *testing.T) {
		original := []float32{1, 2, 3}
		inner := &poisonableEmbedder{vec: append([]float32(nil), original...)}
		c := NewCachedEmbedder(inner, 0)

		first, err := c.Embed(context.Background(), "hello")
		if err != nil {
			t.Fatalf("first Embed: %v", err)
		}
		inner.poison()

		second, err := c.Embed(context.Background(), "hello") // cache hit
		if err != nil {
			t.Fatalf("cache-hit Embed: %v", err)
		}
		if !reflect.DeepEqual(first, original) {
			t.Errorf("first Embed = %v, want %v", first, original)
		}
		if !reflect.DeepEqual(second, original) {
			t.Errorf("cache-hit Embed = %v, want %v (insert must copy)", second, original)
		}
	})

	t.Run("EmbedBatch", func(t *testing.T) {
		original := [][]float32{{4, 5}, {6, 7}}
		inner := &poisonableEmbedder{rows: [][]float32{
			append([]float32(nil), original[0]...),
			append([]float32(nil), original[1]...),
		}}
		c := NewCachedEmbedder(inner, 0)

		first, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
		if err != nil {
			t.Fatalf("first EmbedBatch: %v", err)
		}
		inner.poison()

		second, err := c.EmbedBatch(context.Background(), []string{"a", "b"}) // both cache hits
		if err != nil {
			t.Fatalf("cache-hit EmbedBatch: %v", err)
		}
		if !reflect.DeepEqual(first, original) {
			t.Errorf("first EmbedBatch = %v, want %v", first, original)
		}
		if !reflect.DeepEqual(second, original) {
			t.Errorf("cache-hit EmbedBatch = %v, want %v (insert must copy)", second, original)
		}
	})
}

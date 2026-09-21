package embed

import (
	"context"
	"time"

	"github.com/dpoage/llmkit"
)

// Observe wraps e so every Embed and EmbedBatch call reports one
// llmkit.Event (llmkit.KindEmbed) to obs and otherwise behaves exactly like
// e: results, errors, Dimensions, and ModelName pass through unchanged. A
// nil obs returns e itself.
//
// The event inherits llmkit.Event.RunID, llmkit.Event.SpanID, and
// llmkit.Event.Step from the call's context (llmkit.NewEvent); Observe
// never mints a span — only Completion emitters do. Populated fields:
//
//   - Model: e.ModelName().
//   - Inputs: 1 for Embed, len(texts) for EmbedBatch — the requested input
//     count, error or not.
//   - Dimensions: the length of the returned vector on success; whenever
//     the call produced no vector — an error, an empty batch, or a success
//     returning none — e.Dimensions(), which is 0 while a backend's
//     dimensionality is still undetected.
//   - Duration: the call's wall time.
//   - Err: the error's text, non-empty exactly when the call failed.
//
// CacheHits and Usage always stay zero — the honest-capabilities rule. The
// Embedder interface carries no per-call cache attribution: a
// *CachedEmbedder exposes only lifetime counters (read CachedEmbedder.Stats
// yourself for hit counts), and a before/after delta would attribute a
// concurrent call's hit to this call. No Embedder in this module exposes
// token usage either. Input texts and vectors are not recorded; v1 embed
// events are a summary, so replaying this boundary is not possible today.
//
// Placement around a cache is the honest per-call attribution:
// NewCachedEmbedder(Observe(inner, obs), n) emits exactly one event per
// cache miss (a hit never reaches the observer);
// Observe(NewCachedEmbedder(inner, n), obs) emits one event per call, with
// CacheHits 0.
//
// Observers are synchronous data sinks: a panic in obs propagates to the
// caller and obs never affects the returned vectors.
func Observe(e Embedder, obs llmkit.Observer) Embedder {
	if obs == nil {
		return e
	}
	return &observedEmbedder{inner: e, obs: obs}
}

// observedEmbedder is the Embedder Observe returns. It holds no mutable
// state and is safe for concurrent use as long as inner is.
type observedEmbedder struct {
	inner Embedder
	obs   llmkit.Observer
}

var _ Embedder = (*observedEmbedder)(nil)

func (w *observedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	start := time.Now()
	vec, err := w.inner.Embed(ctx, text)

	ev := llmkit.NewEvent(ctx, llmkit.KindEmbed)
	ev.Duration = time.Since(start)
	ev.Embed = &llmkit.EmbedEvent{
		Model:      w.inner.ModelName(),
		Inputs:     1,
		Dimensions: eventDimensions(err == nil, vec, w.inner),
		Err:        errText(err),
	}
	w.obs.Observe(ctx, ev)
	return vec, err
}

func (w *observedEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	start := time.Now()
	vecs, err := w.inner.EmbedBatch(ctx, texts)

	ev := llmkit.NewEvent(ctx, llmkit.KindEmbed)
	ev.Duration = time.Since(start)
	ev.Embed = &llmkit.EmbedEvent{
		Model:      w.inner.ModelName(),
		Inputs:     len(texts),
		Dimensions: eventDimensions(err == nil, firstVector(vecs), w.inner),
		Err:        errText(err),
	}
	w.obs.Observe(ctx, ev)
	return vecs, err
}

func (w *observedEmbedder) Dimensions() int   { return w.inner.Dimensions() }
func (w *observedEmbedder) ModelName() string { return w.inner.ModelName() }

// eventDimensions derives EmbedEvent.Dimensions: the length of the vector
// the call returned when it succeeded, otherwise the embedder's own report
// — which is 0 while dimension auto-detection has not seen a vector yet.
func eventDimensions(succeeded bool, first []float32, e Embedder) int {
	if succeeded && first != nil {
		return len(first)
	}
	return e.Dimensions()
}

// firstVector returns the first of an EmbedBatch's vectors, or nil when the
// batch returned none.
func firstVector(vecs [][]float32) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	return vecs[0]
}

// errText renders an error for an event's Err field: "" on success.
func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

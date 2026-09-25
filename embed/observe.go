package embed

import (
	"context"
	"time"

	"github.com/dpoage/llmkit"
)

// Observe wraps e so every Embed and EmbedBatch call reports one
// llmkit.Event (llmkit.KindEmbed) to obs and otherwise passes results,
// errors, Dimensions, and ModelName through unchanged. A nil obs returns e.
//
// The event inherits llmkit.Event.RunID, llmkit.Event.SpanID, and
// llmkit.Event.Step from the call's context via llmkit.NewEvent; Observe
// never mints a span — only Completion emitters do.
//
// Populated EmbedEvent fields:
//
//   - Model: e.ModelName().
//   - Inputs: 1 for Embed, len(texts) for EmbedBatch — the requested count,
//     error or not.
//   - Dimensions: length of the returned vector on success; otherwise
//     e.Dimensions() (0 while auto-detection has not seen a vector yet).
//   - Duration: wall time of the call.
//   - Err: the error's text; "" on success.
//
// Input texts and vectors are not recorded; v1 embed events are a summary,
// so replaying this boundary is not possible today.
//
// Placement around a cache changes attribution:
// NewCachedEmbedder(Observe(inner, obs), n) emits one event per cache miss
// (a hit never reaches the observer); Observe(NewCachedEmbedder(inner, n),
// obs) emits one event per call, hit or miss alike — the cache is
// invisible to the event either way, since Embedder exposes no per-call
// cache attribution.
//
// Observers are synchronous: a panic in obs propagates and obs never
// affects the returned vectors.
func Observe(e Embedder, obs llmkit.Observer) Embedder {
	if obs == nil {
		return e
	}
	return &observedEmbedder{inner: e, obs: obs}
}

// observedEmbedder is the Embedder Observe returns; it holds no mutable
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

// eventDimensions is the EmbedEvent.Dimensions value: length of the vector
// the call returned on success, otherwise the embedder's own report.
func eventDimensions(succeeded bool, first []float32, e Embedder) int {
	if succeeded && first != nil {
		return len(first)
	}
	return e.Dimensions()
}

// firstVector returns vecs[0], or nil when vecs is empty.
func firstVector(vecs [][]float32) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	return vecs[0]
}

// errText returns err.Error(), or "" when err is nil.
func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

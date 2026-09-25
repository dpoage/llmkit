package embed_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"unsafe"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/embed"
)

// scriptedEmbedder is a fake [embed.Embedder] for observer tests; returns the
// scripted vector verbatim (or the scripted error).
type scriptedEmbedder struct {
	vec  []float32
	err  error
	dims int // reported by Dimensions; 0 falls back to len(vec)
}

func (f *scriptedEmbedder) Embed(context.Context, string) ([]float32, error) {
	return f.vec, f.err
}

func (f *scriptedEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = f.vec
	}
	return out, nil
}

func (f *scriptedEmbedder) Dimensions() int {
	if f.dims != 0 {
		return f.dims
	}
	return len(f.vec)
}

func (f *scriptedEmbedder) ModelName() string { return "fake-model" }

// eventLog is a concurrent-safe [llmkit.Observer] collecting events for assertions.

type eventLog struct {
	mu  sync.Mutex
	evs []llmkit.Event
}

func (l *eventLog) Observe(_ context.Context, ev llmkit.Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.evs = append(l.evs, ev)
}

func (l *eventLog) events() []llmkit.Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]llmkit.Event(nil), l.evs...)
}

func TestObserveEmbedEmitsOneEvent(t *testing.T) {
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1, 0.2, 0.3}}, log)

	runID := llmkit.NewRunID()
	ctx := llmkit.WithRun(context.Background(), runID)

	vec, err := emb.Embed(ctx, "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("len(vec) = %d, want 3", len(vec))
	}

	evs := log.events()
	if len(evs) != 1 {
		t.Fatalf("got %d events, want 1", len(evs))
	}
	ev := evs[0]
	if ev.Kind != llmkit.KindEmbed {
		t.Errorf("Kind = %q, want %q", ev.Kind, llmkit.KindEmbed)
	}
	if ev.RunID != runID {
		t.Errorf("RunID = %q, want %q", ev.RunID, runID)
	}
	if ev.SchemaVersion != llmkit.EventSchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", ev.SchemaVersion, llmkit.EventSchemaVersion)
	}
	if ev.Embed == nil {
		t.Fatalf("Embed payload is nil")
	}
	e := ev.Embed
	if e.Model != "fake-model" {
		t.Errorf("Model = %q, want %q", e.Model, "fake-model")
	}
	if e.Inputs != 1 {
		t.Errorf("Inputs = %d, want 1", e.Inputs)
	}
	if e.Dimensions != 3 {
		t.Errorf("Dimensions = %d, want 3", e.Dimensions)
	}
	if e.CacheHits != 0 {
		t.Errorf("CacheHits = %d, want 0 (the wrapper cannot observe per-call cache attribution)", e.CacheHits)
	}
	if e.Usage != (llmkit.Usage{}) {
		t.Errorf("Usage = %+v, want zero (no Embedder exposes usage)", e.Usage)
	}
	if e.Err != "" {
		t.Errorf("Err = %q, want empty on success", e.Err)
	}
	if ev.Duration <= 0 {
		t.Errorf("Duration = %v, want > 0", ev.Duration)
	}
}

func TestObserveEmbedErrorStillEmitsAndPassesThrough(t *testing.T) {
	boom := errors.New("boom")
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{err: boom, dims: 7}, log)

	ctx := llmkit.WithRun(context.Background(), llmkit.NewRunID())

	vec, err := emb.Embed(ctx, "hello")
	if err != boom || !errors.Is(err, boom) {
		t.Fatalf("Embed err = %v, want the identical scripted error", err)
	}
	if vec != nil {
		t.Fatalf("vec = %v, want nil on error", vec)
	}

	evs := log.events()
	if len(evs) != 1 {
		t.Fatalf("got %d events, want 1 (a failed call still emits)", len(evs))
	}
	e := evs[0].Embed
	if e == nil {
		t.Fatalf("Embed payload is nil")
	}
	if e.Err != boom.Error() {
		t.Errorf("Err = %q, want %q", e.Err, boom.Error())
	}
	if e.Dimensions != 7 {
		t.Errorf("Dimensions = %d, want 7 (the embedder's own report on error)", e.Dimensions)
	}
	if e.Inputs != 1 {
		t.Errorf("Inputs = %d, want 1", e.Inputs)
	}
}

func TestObserveEmbedBatchReportsInputs(t *testing.T) {
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1, 0.2}}, log)

	vecs, err := emb.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("len(vecs) = %d, want 3", len(vecs))
	}

	evs := log.events()
	if len(evs) != 1 {
		t.Fatalf("got %d events, want exactly one per call", len(evs))
	}
	e := evs[0].Embed
	if e.Inputs != 3 {
		t.Errorf("Inputs = %d, want 3", e.Inputs)
	}
	if e.Dimensions != 2 {
		t.Errorf("Dimensions = %d, want 2 (from the first returned vector)", e.Dimensions)
	}
}

func TestObserveEmbedBatchEmptyInput(t *testing.T) {
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1, 0.2}, dims: 2}, log)

	if _, err := emb.EmbedBatch(context.Background(), nil); err != nil {
		t.Fatalf("EmbedBatch(nil): %v", err)
	}

	e := log.events()[0].Embed
	if e.Inputs != 0 {
		t.Errorf("Inputs = %d, want 0", e.Inputs)
	}
	if e.Dimensions != 2 {
		t.Errorf("Dimensions = %d, want 2 (the embedder's report; no vector came back)", e.Dimensions)
	}
}

func TestObservePassthroughMethods(t *testing.T) {
	log := &eventLog{}
	inner := &scriptedEmbedder{vec: []float32{0.1}, dims: 9}
	emb := embed.Observe(inner, log)

	if emb.Dimensions() != 9 {
		t.Errorf("Dimensions() = %d, want 9", emb.Dimensions())
	}
	if emb.ModelName() != "fake-model" {
		t.Errorf("ModelName() = %q, want %q", emb.ModelName(), "fake-model")
	}
	if got := len(log.events()); got != 0 {
		t.Errorf("got %d events from passthrough methods, want 0", got)
	}
}

func TestObserveNilObserverReturnsInput(t *testing.T) {
	var e embed.Embedder = &scriptedEmbedder{vec: []float32{0.1}}
	if got := embed.Observe(e, nil); got != e {
		t.Fatalf("Observe(e, nil) returned a new wrapper, want e itself")
	}
}

func TestObserveObserverPanicPropagates(t *testing.T) {
	emb := embed.Observe(
		&scriptedEmbedder{vec: []float32{0.1}},
		llmkit.ObserverFunc(func(context.Context, llmkit.Event) { panic("observer bug") }),
	)

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("Embed recovered an observer panic, want it to propagate")
		}
	}()
	_, _ = emb.Embed(context.Background(), "hello")
}

func TestObserveConcurrentEmbeds(t *testing.T) {
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1, 0.2}}, log)
	ctx := llmkit.WithRun(context.Background(), llmkit.NewRunID())
	const n = 32
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := emb.Embed(ctx, "text"); err != nil {
				t.Errorf("Embed: %v", err)
			}
			if _, err := emb.EmbedBatch(ctx, []string{"a", "b"}); err != nil {
				t.Errorf("EmbedBatch: %v", err)
			}
		}()
	}
	wg.Wait()

	if got := len(log.events()); got != 2*n {
		t.Errorf("got %d events, want %d", got, 2*n)
	}
}

// ExampleObserve wraps a fake backend with an observer and reads the one
// Embed event back. The fake keeps the example hermetic; real use points
// Observe at [embed.New] or [embed.NewCachedEmbedder].

func ExampleObserve() {
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1, 0.2, 0.3}}, log)

	ctx := llmkit.WithRun(context.Background(), llmkit.NewRunID())
	_, _ = emb.Embed(ctx, "hello")

	for _, ev := range log.events() {
		fmt.Printf("%s model=%s inputs=%d dims=%d err=%q\n",
			ev.Kind, ev.Embed.Model, ev.Embed.Inputs, ev.Embed.Dimensions, ev.Embed.Err)
	}
	// Output:
	// embed model=fake-model inputs=1 dims=3 err=""
}

func TestObserveEmbedBatchErrorStillEmits(t *testing.T) {
	boom := errors.New("boom")
	log := &eventLog{}
	emb := embed.Observe(&scriptedEmbedder{err: boom, dims: 11}, log)

	vecs, err := emb.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err != boom || !errors.Is(err, boom) {
		t.Fatalf("EmbedBatch err = %v, want the identical scripted error", err)
	}
	if vecs != nil {
		t.Fatalf("vecs = %v, want nil on error", vecs)
	}

	evs := log.events()
	if len(evs) != 1 {
		t.Fatalf("got %d events, want exactly 1 (a failed batch still emits)", len(evs))
	}
	e := evs[0].Embed
	if e.Err != boom.Error() {
		t.Errorf("Err = %q, want %q", e.Err, boom.Error())
	}
	if e.Inputs != 3 {
		t.Errorf("Inputs = %d, want 3 (the requested input count)", e.Inputs)
	}
	if e.Dimensions != 11 {
		t.Errorf("Dimensions = %d, want 11 (the embedder's own report on error)", e.Dimensions)
	}
	if evs[0].Duration <= 0 {
		t.Errorf("Duration = %v, want > 0", evs[0].Duration)
	}
}

func TestObservePassThroughFidelity(t *testing.T) {
	log := &eventLog{}
	inner := &scriptedEmbedder{vec: []float32{0.1, 0.2, 0.3}}
	emb := embed.Observe(inner, log)

	vec, err := emb.Embed(context.Background(), "hello")
	if err != nil || len(vec) != 3 {
		t.Fatalf("Embed = %v, %v", vec, err)
	}
	if unsafe.SliceData(vec) != unsafe.SliceData(inner.vec) {
		t.Errorf("Embed returned a copied vector; want the inner embedder's slice itself")
	}

	vecs, err := emb.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil || len(vecs) != 2 {
		t.Fatalf("EmbedBatch = %v, %v", vecs, err)
	}
	for i, v := range vecs {
		if unsafe.SliceData(v) != unsafe.SliceData(inner.vec) {
			t.Errorf("batch row %d is a copy; want the inner embedder's vector", i)
		}
	}
}

func TestObservePassesCallerCtxToObserver(t *testing.T) {
	type marker struct{}
	var got any
	obs := llmkit.ObserverFunc(func(ctx context.Context, _ llmkit.Event) {
		got = ctx.Value(marker{})
	})
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1}}, obs)

	ctx := context.WithValue(context.Background(), marker{}, "present")
	_, _ = emb.Embed(ctx, "hello")
	if got != "present" {
		t.Errorf("observer saw ctx value %v, want \"present\" — the observer must receive the caller's context", got)
	}
}

func TestObserveEmbedDimensionsFromResult(t *testing.T) {
	log := &eventLog{}
	// dims 77 stands in for e.Dimensions(); a successful call must report
	// the returned vector's length, not the embedder's own report.
	emb := embed.Observe(&scriptedEmbedder{vec: []float32{0.1, 0.2, 0.3}, dims: 77}, log)

	if _, err := emb.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got := log.events()[0].Embed.Dimensions; got != 3 {
		t.Errorf("Dimensions = %d, want 3 (the returned vector's length, not e.Dimensions() = 77)", got)
	}
}

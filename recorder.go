package llmkit

import "context"

// UsageEvent is a single accounting record emitted after a completion. It binds
// the spend to the provider/model that incurred it so callers can ledger
// per-dimension.
type UsageEvent struct {
	// Provider is the provider tag: the provider type by default
	// ("anthropic", "openai", ...), or the caller's own config name via the
	// provider package's Options.Provider.
	Provider string
	// Model is the model identifier.
	Model string
	// Usage is the token consumption for this call.
	Usage Usage
}

// Recorder is a callback hook callers implement to ledger spend. Record is
// invoked once per successful Complete. Implementations must be safe for
// concurrent use and must not block for long, since they run on the request
// path.
type Recorder interface {
	Record(ev UsageEvent)
}

// RecorderFunc adapts a plain function to the Recorder interface.
type RecorderFunc func(ev UsageEvent)

// Record implements Recorder.
func (f RecorderFunc) Record(ev UsageEvent) { f(ev) }

// recordingClient wraps a Client, reporting usage to a Recorder after each
// successful completion.
type recordingClient struct {
	inner    Client
	rec      Recorder
	provider string
	model    string
}

// WithRecorder wraps c so that every successful Complete reports its usage to
// rec, tagged with provider and model. A nil rec returns c unchanged.
//
// Callers that want extra tags of their own (a pipeline role, a request ID)
// wrap their Recorder instead of expecting this package to carry the strings:
//
//	type tagged struct {
//		role string
//		next Recorder
//	}
//
//	func (t tagged) Record(ev UsageEvent) {
//		log.Printf("%s: %s/%s spent %+v", t.role, ev.Provider, ev.Model, ev.Usage)
//		t.next.Record(ev)
//	}
//	client := llmkit.WithRecorder(c, tagged{role: "finder", next: ledger}, "anthropic", "claude-sonnet-4-5")
func WithRecorder(c Client, rec Recorder, provider, model string) Client {
	if rec == nil {
		return c
	}
	return &recordingClient{inner: c, rec: rec, provider: provider, model: model}
}

func (r *recordingClient) Capabilities() Capabilities { return r.inner.Capabilities() }

func (r *recordingClient) Complete(ctx context.Context, req Request) (Response, error) {
	resp, err := r.inner.Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	r.rec.Record(UsageEvent{
		Provider: r.provider,
		Model:    r.model,
		Usage:    resp.Usage,
	})
	return resp, nil
}

package provider

import (
	"context"

	llmkit "github.com/dpoage/llmkit"
)

// stackClient is the outermost stage of every stack [New] and [Wrap]
// build. It remembers the client below every provider stage (base) and the
// stack's Identity, so [Wrap] over a stackClient rebuilds one stack from
// base instead of stacking a second retry loop on the first.
//
// It owns the stack's completion claim: a call whose ctx carries no span is
// claimed with [llmkit.BeginCompletion] above the retry stage, whether or
// not an observer is wired. With one, llmkit.Observe mints the span and
// emits the Completion; without one, stackClient mints the span and no
// emitter runs. Either way an emitter nested below the retry stage sees a
// claimed ctx on every attempt and stays silent, so one call yields at most
// one Completion event across the stack.
type stackClient struct {
	base     llmkit.Client
	identity llmkit.Identity
	next     llmkit.Client
	// claims is true when the stack has no completion emitter (Options.Observer nil).
	claims bool
}

func (s *stackClient) Capabilities() llmkit.Capabilities { return s.next.Capabilities() }

// Identity implements llmkit.IdentifiedClient with the Identity Wrap/New
// computed for the stack.
func (s *stackClient) Identity() llmkit.Identity { return s.identity }

func (s *stackClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	return s.next.Complete(s.claim(ctx), req)
}

func (s *stackClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	return llmkit.Stream(s.claim(ctx), s.next, req, fn)
}

// claim returns ctx marked as belonging to this call's logical completion:
// unchanged when ctx already carries a span (an enclosing emitter owns the
// call) or when the stack's own emitter will claim it; otherwise a fresh span.
func (s *stackClient) claim(ctx context.Context) context.Context {
	if !s.claims || llmkit.SpanFromContext(ctx) != "" {
		return ctx
	}
	return llmkit.BeginCompletion(ctx)
}

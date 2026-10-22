package provider

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"

	llmkit "github.com/dpoage/llmkit"
)

// Fallback returns a client that calls primary and, on a retryable failure,
// each fallback in order until one serves. It panics on a nil member.
// Each member is usually a [New] or [Wrap] stack with its own retry and
// Observer; Capabilities and Identity are the primary's, and a response
// whose Served is empty is stamped with the serving member's identity.
// Never [Wrap] the result, which multiplies retries and labels every call as
// the primary: observe it with [llmkit.Observe]. See docs/providers.md.
func Fallback(primary llmkit.Client, fallbacks ...llmkit.Client) llmkit.Client {
	members := append([]llmkit.Client{primary}, fallbacks...)
	for i, m := range members {
		if m == nil {
			panic(fmt.Sprintf("provider.Fallback: member %d is nil", i))
		}
	}
	return &fallbackClient{members: members}
}

type fallbackClient struct {
	members []llmkit.Client
}

func (f *fallbackClient) Capabilities() llmkit.Capabilities { return f.members[0].Capabilities() }

func (f *fallbackClient) Identity() llmkit.Identity { return llmkit.IdentityOf(f.members[0]) }

func (f *fallbackClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	var earlier []error
	for _, m := range f.members {
		resp, err := m.Complete(ctx, req)
		if err == nil {
			return stamp(resp, m), nil
		}
		earlier = append(earlier, err)
		if !fallsThrough(ctx, err) {
			break
		}
	}
	return llmkit.Response{}, joinMemberErrors(earlier)
}

// Stream falls through only while no delta has reached fn.
func (f *fallbackClient) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	var delivered atomic.Bool
	deliver := fn
	if fn != nil {
		deliver = func(d llmkit.Delta) error {
			delivered.Store(true)
			return fn(d)
		}
	}
	var earlier []error
	for _, m := range f.members {
		resp, err := llmkit.Stream(ctx, m, req, deliver)
		if err == nil {
			return stamp(resp, m), nil
		}
		earlier = append(earlier, err)
		if delivered.Load() || !fallsThrough(ctx, err) {
			break
		}
	}
	return llmkit.Response{}, joinMemberErrors(earlier)
}

func fallsThrough(ctx context.Context, err error) bool {
	_, _, retryable := llmkit.Classify(err)
	return retryable && ctx.Err() == nil
}

func stamp(resp llmkit.Response, m llmkit.Client) llmkit.Response {
	if resp.Served == (llmkit.Identity{}) {
		resp.Served = llmkit.IdentityOf(m)
	}
	return resp
}

// joinMemberErrors returns the final error as the single %w, so errors.Is and
// errors.As see only its chain; earlier errors appear in the message alone.
func joinMemberErrors(errs []error) error {
	last := errs[len(errs)-1]
	if len(errs) == 1 {
		return last
	}
	var earlier strings.Builder
	for i, e := range errs[:len(errs)-1] {
		if i > 0 {
			earlier.WriteString("; ")
		}
		fmt.Fprintf(&earlier, "member %d: %v", i, e)
	}
	return fmt.Errorf("provider: fallback member %d failed after earlier members failed (%s): %w", len(errs)-1, earlier.String(), last)
}

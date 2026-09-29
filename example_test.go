package llmkit_test

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/dpoage/llmkit"
)

// echoClient is a programmable Client that returns scripted responses and
// errors in order. It implements only llmkit.Client — no Stream method — so
// llmkit.Stream exercises its delta-synthesis path; this exists to make the
// examples hermetic.
type echoClient struct {
	responses []llmkit.Response
	errs      []error
	attempts  int
}

func (e *echoClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

func (e *echoClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	i := e.attempts
	e.attempts++
	if i < len(e.errs) && e.errs[i] != nil {
		return llmkit.Response{}, e.errs[i]
	}
	// No error scripted for this attempt: serve the first response not yet
	// consumed, or the last one when a retry exhausts the script.
	if len(e.responses) == 0 {
		return llmkit.Response{}, errors.New("llmkit: no scripted response")
	}
	idx := i - len(e.errs)
	if idx < 0 || idx >= len(e.responses) {
		idx = len(e.responses) - 1
	}
	return e.responses[idx], nil
}

func ExampleUserMessage() {
	msg := llmkit.UserMessage(
		llmkit.Text("What is in this picture?"),
		llmkit.Image("image/png", []byte{0x89, 'P', 'N', 'G'}),
	)
	fmt.Println(msg.Role)
	for _, b := range msg.Content {
		fmt.Println(b.Kind)
	}
	// Output:
	// user
	// text
	// image
}

func ExampleStream() {
	// echoClient has no Stream method, so llmkit.Stream calls Complete and
	// synthesizes deltas in block order: the thinking block arrives as a
	// DeltaThinking fragment in Delta.Thinking, each text block as a
	// DeltaText fragment in Delta.Text.
	c := &echoClient{responses: []llmkit.Response{{
		Blocks: []llmkit.Block{
			{Kind: llmkit.BlockThinking, Text: "The user greets me; I greet back."},
			llmkit.Text("Hello"),
			llmkit.Text(" world"),
		},
		Text:       "Hello world",
		Usage:      llmkit.Usage{InputTokens: 5, OutputTokens: 3},
		StopReason: llmkit.StopEndTurn,
	}}}

	// Printing Delta.Text leaves out this response's thinking block: its
	// DeltaThinking fragment carries the text in Delta.Thinking.
	_, err := llmkit.Stream(context.Background(), c, llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Say hello.")},
	}, func(d llmkit.Delta) error { fmt.Print(d.Text); return nil })
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println()
	// Output:
	// Hello world
}

func ExampleObserve() {
	c := &echoClient{
		responses: []llmkit.Response{
			{Text: "observed", StopReason: llmkit.StopEndTurn},
		},
	}
	// Observe emits one Completion event per logical call when wrapping a
	// bare client. A client built with provider.New or provider.Wrap also
	// emits Attempt events from its retry stage, joined to this Completion
	// by SpanID — see the provider package's Wrap example.
	var kinds []llmkit.EventKind
	obs := llmkit.ObserverFunc(func(ctx context.Context, ev llmkit.Event) {
		kinds = append(kinds, ev.Kind)
	})
	observed := llmkit.Observe(c, obs)
	_, err := observed.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for _, k := range kinds {
		fmt.Println(k)
	}
	// Output:
	// completion
}

func ExampleAPIError() {
	var err error = &llmkit.APIError{
		Kind:          llmkit.ErrRateLimited,
		StatusCode:    429,
		RetryAfter:    2 * time.Second,
		HasRetryAfter: true,
		Provider:      "anthropic",
		Message:       "rate limit exceeded",
	}
	fmt.Println(errors.Is(err, llmkit.ErrRateLimited))
	fmt.Println(errors.Is(err, llmkit.ErrAuth))

	var apiErr *llmkit.APIError
	if errors.As(err, &apiErr) {
		fmt.Println(apiErr.StatusCode)
		fmt.Println(apiErr.RetryAfter)
	}
	// Output:
	// true
	// false
	// 429
	// 2s
}

func ExampleStripThinkBlocks() {
	fmt.Println(llmkit.StripThinkBlocks("<think>2 + 2 is 4</think>The answer is 4."))
	// A tag embedded inside the body stays.
	fmt.Println(llmkit.StripThinkBlocks("A <think> tag mid-sentence stays."))
	// Output:
	// The answer is 4.
	// A <think> tag mid-sentence stays.
}

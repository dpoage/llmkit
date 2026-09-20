package llmkit_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// echoClient is a programmable Client that returns scripted responses and
// errors in order. It implements only llmkit.Client — no Stream method — so
// llmkit.Stream exercises its delta-synthesis path. A real client comes from
// the provider package; this one exists to make the examples hermetic.
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
	// synthesizes deltas: text blocks in order, then one tool-call delta.
	c := &echoClient{responses: []llmkit.Response{{
		Blocks: []llmkit.Block{
			llmkit.Text("Hello"),
			llmkit.Text(" world"),
		},
		Text: "Hello world",
		ToolCalls: []llmkit.ToolCall{
			{ID: "call_1", Name: "get_time", Arguments: json.RawMessage(`{"tz":"utc"}`)},
		},
		Usage:      llmkit.Usage{InputTokens: 5, OutputTokens: 3},
		StopReason: llmkit.StopEndTurn,
	}}}

	resp, err := llmkit.Stream(context.Background(), c, llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "What time is it?")},
	}, func(d llmkit.Delta) error {
		switch d.Kind {
		case llmkit.DeltaText, llmkit.DeltaThinking:
			fmt.Printf("%s: %q\n", d.Kind, d.Text)
		case llmkit.DeltaToolCall:
			fmt.Printf("%s: %s %s args=%s\n", d.Kind, d.ID, d.Name, d.Arguments)
		}
		return nil
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(resp.Text)
	// Output:
	// text: "Hello"
	// text: " world"
	// tool_call: call_1 get_time args={"tz":"utc"}
	// Hello world
}

func ExampleWithRetry() {
	c := &echoClient{
		errs: []error{
			&llmkit.APIError{Kind: llmkit.ErrRateLimited, StatusCode: 429, Provider: "demo", Message: "slow down"},
		},
		responses: []llmkit.Response{
			{Text: "recovered", StopReason: llmkit.StopEndTurn},
		},
	}
	wrapped := llmkit.WithRetry(c, retry.Config{
		MaxAttempts:    3,
		BaseDelay:      time.Millisecond,
		MaxDelay:       time.Millisecond,
		RequestTimeout: time.Minute,
	})
	resp, err := wrapped.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("attempts:", c.attempts)
	fmt.Println(resp.Text)
	// Output:
	// attempts: 2
	// recovered
}

func ExampleWithRecorder() {
	c := &echoClient{responses: []llmkit.Response{{
		Text:  "hi there",
		Usage: llmkit.Usage{InputTokens: 12, OutputTokens: 5},
	}}}
	var events []llmkit.UsageEvent
	wrapped := llmkit.WithRecorder(c, llmkit.RecorderFunc(func(ev llmkit.UsageEvent) {
		events = append(events, ev)
	}), "anthropic", "claude-sonnet-4-5")

	_, err := wrapped.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for _, ev := range events {
		fmt.Printf("%s/%s in=%d out=%d\n", ev.Provider, ev.Model, ev.Usage.InputTokens, ev.Usage.OutputTokens)
	}
	// Output:
	// anthropic/claude-sonnet-4-5 in=12 out=5
}

func ExampleAPIError() {
	var err error = &llmkit.APIError{
		Kind:       llmkit.ErrRateLimited,
		StatusCode: 429,
		RetryAfter: 2 * time.Second,
		Provider:   "anthropic",
		Message:    "rate limit exceeded",
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

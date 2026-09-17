// Command agent demonstrates the llmkit/agent tool-calling harness: a
// Runner with two small tools (now, add), synchronous lifecycle hooks
// logging ToolStart/ToolEnd/AfterCompletion, a per-tool timeout, and —
// behind --parallel — concurrent dispatch of the tool calls a single
// completion requests. It doubles as a compile-time contract check for the
// agent surface: agent.NewRunner, agent.Tool, agent.Hooks, WithToolTimeout,
// and WithParallelTools.
//
// Usage:
//
//	export LLMKIT_PROVIDER=openai   # anthropic | openai | openai-compatible | google
//	export LLMKIT_MODEL=gpt-4o-mini
//	export LLMKIT_API_KEY=sk-...
//	export LLMKIT_BASE_URL=...      # optional; e.g. a local OpenAI-compatible endpoint
//	go run ./examples/agent [--parallel] [--task "..."]
//
// Without the environment variables set, the program prints usage and exits
// non-zero without touching the network.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/provider"
)

var errUsage = errors.New(`missing environment:
  LLMKIT_PROVIDER  anthropic | openai | openai-compatible | google
  LLMKIT_MODEL     model name, e.g. claude-sonnet-4-5 or gpt-4o-mini
  LLMKIT_API_KEY   provider API key (any placeholder for a local endpoint)
  LLMKIT_BASE_URL  optional base URL for OpenAI-compatible endpoints

set the variables above, then re-run:
  go run ./examples/agent [--parallel] [--task "..."]`)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "agent:", err)
		os.Exit(1)
	}
}

func run() error {
	parallel := flag.Bool("parallel", false, "dispatch the tool calls of one turn concurrently (WithParallelTools)")
	task := flag.String("task", "What time is it right now, and what is 41 plus 58? Use the tools to answer both parts.",
		"the task to give the agent")
	flag.Parse()

	providerName := os.Getenv("LLMKIT_PROVIDER")
	model := os.Getenv("LLMKIT_MODEL")
	apiKey := os.Getenv("LLMKIT_API_KEY")
	baseURL := os.Getenv("LLMKIT_BASE_URL")
	if providerName == "" || model == "" || apiKey == "" {
		return errUsage
	}

	spec, err := specFor(providerName, baseURL)
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, providerName, model, apiKey, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	// Hooks are synchronous: each runs inline on the goroutine that reaches
	// the fire point (under --parallel, ToolStart/ToolEnd fire concurrently
	// from the per-call goroutines, so a real hook must synchronize its own
	// state).
	hooks := agent.Hooks{
		BeforeCompletion: func(_ context.Context, step int, req *llmkit.Request) {
			log.Printf("step %d: completion (%d message(s), %d tool(s))", step, len(req.Messages), len(req.Tools))
		},
		AfterCompletion: func(_ context.Context, step int, _ *llmkit.Request, resp *llmkit.Response, err error) {
			if err != nil {
				log.Printf("step %d: completion failed: %v", step, err)
				return
			}
			log.Printf("step %d: completion stop=%s input=%d output=%d", step, resp.StopReason, resp.Usage.InputTokens, resp.Usage.OutputTokens)
		},
		ToolStart: func(_ context.Context, ev agent.ToolEvent) {
			log.Printf("step %d: tool %s start args=%s", ev.Step, ev.Call.Name, ev.Call.Arguments)
		},
		ToolEnd: func(_ context.Context, ev agent.ToolEvent) {
			log.Printf("step %d: tool %s done in %s (error=%t) result=%s",
				ev.Step, ev.Call.Name, ev.Duration.Round(time.Microsecond), ev.IsError, ev.Result)
		},
	}

	opts := []agent.Option{
		agent.WithHooks(hooks),
		agent.WithToolTimeout(10 * time.Second),
	}
	if *parallel {
		opts = append(opts, agent.WithParallelTools())
	}

	runner := agent.NewRunner(client, []agent.Tool{nowTool{}, addTool{}},
		"You are a helpful assistant. Use the provided tools whenever they would help answer.",
		opts...)

	log.Printf("task: %s (parallel=%t)", *task, *parallel)
	outcome, err := runner.Run(context.Background(), *task)
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}

	fmt.Println()
	if outcome.FinalTextSet {
		fmt.Println("final:     ", outcome.FinalText)
	} else {
		fmt.Println("final:      (no assistant text produced)")
	}
	u := outcome.Usage
	fmt.Printf("usage:      input=%d output=%d over %d turn(s)\n", u.InputTokens, u.OutputTokens, outcome.Iterations)
	if outcome.Truncated {
		fmt.Println("truncated: ", outcome.TruncationReason)
	}
	return nil
}

// specFor maps the provider name onto provider.Type with a friendly error
// listing the accepted values. BaseURL is empty unless the caller set
// LLMKIT_BASE_URL (used by openai-compatible and useful for proxies).
func specFor(providerName, baseURL string) (provider.Spec, error) {
	var spec provider.Spec
	switch providerName {
	case "anthropic":
		spec.Type = provider.TypeAnthropic
	case "openai":
		spec.Type = provider.TypeOpenAI
	case "openai-compatible":
		spec.Type = provider.TypeOpenAICompatible
	case "google":
		spec.Type = provider.TypeGoogle
	default:
		return provider.Spec{}, fmt.Errorf("unknown LLMKIT_PROVIDER %q: expected anthropic, openai, openai-compatible, or google", providerName)
	}
	spec.BaseURL = baseURL
	return spec, nil
}

// nowTool reports the current local time; it takes no arguments.
type nowTool struct{}

func (nowTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{
		Name:        "now",
		Description: "returns the current local date and time",
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
	}
}

func (nowTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	return time.Now().Format(time.RFC3339), nil
}

// addTool adds two numbers, decoding the model's raw JSON arguments with
// agent.UnmarshalArgs.
type addTool struct{}

func (addTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{
		Name:        "add",
		Description: "adds two numbers",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}`),
	}
}

func (addTool) Run(_ context.Context, args json.RawMessage) (string, error) {
	var p struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := agent.UnmarshalArgs(args, &p); err != nil {
		return "", err
	}
	return fmt.Sprintf("%g", p.A+p.B), nil
}

// Command chat demonstrates multi-turn conversation through the llmkit/agent
// harness: a stdin REPL where each line becomes the next task via
// agent.Runner.Run with agent.Continue(prev), so the model keeps every
// earlier turn of the session instead of re-orienting from scratch. One
// trivial tool (now) keeps the tool-calling path exercised. When the model
// stops for a refusal/safety reason (agent.StopReasonError) a canned reply is
// printed and the refusal turn stays in the history: the attached
// err.Outcome is threaded into the next run's agent.Continue so the
// conversation continues from it. It doubles as a compile-time contract check
// for agent.Continue and Outcome.Messages threading.
//
// Usage:
//
//	export LLMKIT_PROVIDER=openai   # anthropic | openai | openai-compatible | google
//	export LLMKIT_MODEL=gpt-4o-mini
//	export LLMKIT_API_KEY=sk-...
//	export LLMKIT_BASE_URL=...      # optional; e.g. a local OpenAI-compatible endpoint
//	go run ./examples/chat
//
// Without the environment variables set, the program prints usage and exits
// non-zero without touching the network.
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/provider"
)

var errUsage = errors.New(`missing environment:
  LLMKIT_PROVIDER  anthropic | openai | openai-compatible | google
  LLMKIT_MODEL     model name, e.g. claude-sonnet-4-5 or gpt-4o-mini
  LLMKIT_API_KEY   provider API key (any placeholder for a local endpoint)
  LLMKIT_BASE_URL  optional base URL for OpenAI-compatible endpoints

set the variables above, then re-run:
  go run ./examples/chat`)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "chat:", err)
		os.Exit(1)
	}
}

func run() error {
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

	// now takes no arguments (struct{}); agent.Func derives its parameter
	// schema and replaces the hand-written ToolDef + Run pair.
	now := agent.Func[struct{}]("now", "returns the current local date and time",
		func(_ context.Context, _ struct{}) (string, error) {
			return time.Now().Format(time.RFC3339), nil
		})

	runner := agent.NewRunner(client, []agent.Tool{now},
		"You are a helpful assistant in a multi-turn chat. You remember every earlier turn of this conversation. Use the provided tool when it would help.")

	fmt.Println("llmkit chat — type a message (ctrl-d to exit).")
	sc := bufio.NewScanner(os.Stdin)
	var prev *agent.Outcome
	for {
		fmt.Print("you> ")
		if !sc.Scan() {
			break // EOF: exit cleanly
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		outcome, err := runner.Run(context.Background(), line, agent.Continue(prev))
		var stopErr *agent.StopReasonError
		if errors.As(err, &stopErr) {
			fmt.Printf("assistant> (the model declined: %s)\n", stopErr.StopReason)
			prev = stopErr.Outcome // keep the refusal turn in the history
			continue
		}
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
		if outcome.FinalTextSet {
			fmt.Println("assistant>", outcome.FinalText)
		} else {
			fmt.Println("assistant> (no assistant text produced)")
		}
		prev = outcome
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

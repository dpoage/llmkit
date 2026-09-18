// Command chat demonstrates multi-turn conversation through the llmkit/agent
// harness: a stdin REPL where each line becomes the next task via
// agent.Runner.RunContinue, so the model keeps every earlier turn of the
// session instead of re-orienting from scratch. One trivial tool (now) keeps
// the tool-calling path exercised. When the model stops for a refusal/safety
// reason (agent.ErrStopReason) a canned reply is printed and the refusal turn
// stays in the history: the attached err.Outcome is threaded into the next
// RunContinue so the conversation continues from it. It doubles as a
// compile-time contract check for agent.RunContinue and Outcome.Messages
// threading.
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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/examples/internal/envcfg"
	"github.com/dpoage/llmkit/provider"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "chat:", err)
		os.Exit(1)
	}
}

func run() error {
	spec, err := envcfg.Load(`missing environment:
  LLMKIT_PROVIDER  anthropic | openai | openai-compatible | google
  LLMKIT_MODEL     model name, e.g. claude-sonnet-4-5 or gpt-4o-mini
  LLMKIT_API_KEY   provider API key (any placeholder for a local endpoint)
  LLMKIT_BASE_URL  optional base URL for OpenAI-compatible endpoints

set the variables above, then re-run:
  go run ./examples/chat`)
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	runner := agent.NewRunner(client, []agent.Tool{nowTool{}},
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
		outcome, err := runner.RunContinue(context.Background(), prev, line)
		var stopErr *agent.ErrStopReason
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

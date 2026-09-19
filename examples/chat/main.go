// Command chat demonstrates multi-turn conversation through the llmkit/agent
// harness: a stdin REPL where each line becomes the next task via
// agent.Runner.Run with agent.Continue(prev), so the model keeps every
// earlier turn of the session instead of re-orienting from scratch. One
// trivial tool (now) keeps the tool-calling path exercised. When the model
// stops for a refusal/safety reason (agent.StopReasonError) a canned reply is
// printed and the refusal turn stays in the history: the attached
// err.Outcome is threaded into the next run's agent.Continue so the
// conversation continues from it. It doubles as a compile-time contract check
// for agent.Continue and Outcome.Messages threading. A /think command
// toggles extended thinking via an agent.RequestPolicy when the model
// reports thinking support.
//
// Usage:
//
//	# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
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
	"sync/atomic"
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
	spec, err := envcfg.Load(envcfg.Usage("go run ./examples/chat", ""))
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	// now takes no arguments (struct{}); agent.Func derives its parameter
	// schema and replaces the hand-written ToolDef + Run pair.
	now := agent.Func[struct{}]("now", "returns the current local date and time",
		func(_ context.Context, _ struct{}) (string, error) {
			return time.Now().Format(time.RFC3339), nil
		})

	// A RequestPolicy shapes every outgoing completion request just before
	// the wire call. The /think command flips this one's atomic flag; while
	// on, it stamps extended thinking onto each request. Gated on the
	// client's Thinking capability — without it the toggle reports and
	// stays off.
	think := &thinkPolicy{}
	thinkingSupported := client.Capabilities().Thinking
	if !thinkingSupported {
		fmt.Println("note: model reports no thinking support; /think stays off")
	}
	runner := agent.NewRunner(client, []agent.Tool{now},
		"You are a helpful assistant in a multi-turn chat. You remember every earlier turn of this conversation. Use the provided tool when it would help.",
		agent.WithRequestPolicy(think))

	fmt.Println("llmkit chat — type a message (/think toggles extended thinking, ctrl-d to exit).")
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
		if line == "/think" {
			if !thinkingSupported {
				fmt.Println("(extended thinking is not supported by this model)")
				continue
			}
			think.on.Store(!think.on.Load())
			if think.on.Load() {
				fmt.Println("(extended thinking on)")
			} else {
				fmt.Println("(extended thinking off)")
			}
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
		if outcome.FinalText != "" {
			fmt.Println("assistant>", outcome.FinalText)
		} else {
			fmt.Println("assistant> (no assistant text produced)")
		}
		prev = outcome
	}
	return nil
}

// thinkPolicy is the [agent.RequestPolicy] behind the /think command: while
// enabled it stamps Thinking on every outgoing completion request. The flag
// is atomic so toggling and running stay race-free.
type thinkPolicy struct {
	on atomic.Bool
}

func (p *thinkPolicy) PrepareRequest(_ context.Context, _ int, req *llmkit.Request) error {
	if p.on.Load() {
		req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 1024}
	}
	return nil
}

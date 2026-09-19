// Command agent demonstrates the llmkit/agent tool-calling harness: a
// Runner with two small tools (now, add), synchronous lifecycle hooks
// logging ToolStart/ToolEnd/AfterCompletion, a per-tool timeout, and —
// behind --parallel — concurrent dispatch of the tool calls a single
// completion requests. It doubles as a compile-time contract check for the
// agent surface: agent.NewRunner, agent.Func, agent.Hooks, WithToolTimeout,
// and WithParallelTools.
//
// Usage:
//
//	# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
//	go run ./examples/agent [--parallel] [--task "..."]
//
// Without the environment variables set, the program prints usage and exits
// non-zero without touching the network.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/examples/internal/envcfg"
	"github.com/dpoage/llmkit/provider"
)

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

	spec, err := envcfg.Load(envcfg.Usage("go run ./examples/agent",
		`[--parallel] [--task "..."]`))
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, provider.Options{})
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

	// now takes no arguments (struct{}); add's schema is derived from
	// addArgs. agent.Func replaces the hand-written ToolDef + Run pairs.
	now := agent.Func[struct{}]("now", "returns the current local date and time",
		func(_ context.Context, _ struct{}) (string, error) {
			return time.Now().Format(time.RFC3339), nil
		})
	add := agent.Func("add", "adds two numbers", func(_ context.Context, p addArgs) (string, error) {
		return fmt.Sprintf("%g", p.A+p.B), nil
	})

	runner := agent.NewRunner(client, []agent.Tool{now, add},
		"You are a helpful assistant. Use the provided tools whenever they would help answer.",
		opts...)

	log.Printf("task: %s (parallel=%t)", *task, *parallel)
	outcome, err := runner.Run(context.Background(), *task)
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}

	fmt.Println()
	if outcome.FinalText != "" {
		fmt.Println("final:     ", outcome.FinalText)
	} else {
		fmt.Println("final:      (no assistant text produced)")
	}
	u := outcome.Usage
	fmt.Printf("usage:      input=%d output=%d over %d turn(s)\n", u.InputTokens, u.OutputTokens, outcome.Iterations)
	if outcome.Truncated() {
		fmt.Println("truncated: ", outcome.TruncationReason)
	}
	return nil
}

// addArgs are the arguments of the add tool; agent.SchemaOf derives the
// advertised parameter schema from these fields.
type addArgs struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

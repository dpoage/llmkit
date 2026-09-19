// Command agent demonstrates the llmkit/agent tool-calling harness: a
// Runner with two small tools (now, add), synchronous lifecycle hooks
// logging ToolStart/ToolEnd/AfterCompletion, a per-tool timeout, a
// ToolPolicy denying any tool named in --deny (decisions logged alongside
// the hooks), and — behind --parallel — concurrent dispatch of the tool
// calls a single completion requests. Compile-time check for the agent
// surface: agent.NewRunner, agent.Func, agent.Hooks, agent.ToolPolicyFunc,
// WithToolTimeout, WithParallelTools.
//
// Usage:
//
//	# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
//	go run ./examples/agent [--parallel] [--task "..."] [--deny now,add]
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
	"strings"
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
	deny := flag.String("deny", "", "comma-separated tool names the ToolPolicy refuses to run (WithToolPolicy demo)")
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

	// ToolPolicy is the permission seam: consulted once per model-requested
	// call, in model order, on the loop goroutine before any Tool.Run of the
	// turn dispatches. A denial feeds the model
	// "ERROR: tool <name> denied: …" and the run continues; the decision is
	// logged with the same output as the hooks.
	denySet := map[string]bool{}
	for _, name := range strings.Split(*deny, ",") {
		if name = strings.TrimSpace(name); name != "" {
			denySet[name] = true
		}
	}
	policy := agent.ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if denySet[call.Name] {
			log.Printf("policy: tool %s DENIED (named in --deny)", call.Name)
			return fmt.Errorf("%s is disabled in this demo", call.Name)
		}
		log.Printf("policy: tool %s allowed", call.Name)
		return nil
	})

	opts := []agent.Option{
		agent.WithHooks(hooks),
		agent.WithToolTimeout(10 * time.Second),
		agent.WithToolPolicy(policy),
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

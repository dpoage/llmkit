// Command agent demonstrates the llmkit/agent tool-calling harness: a
// Runner with two small tools (now, add), synchronous lifecycle hooks
// logging ToolStart/ToolEnd/AfterCompletion, a per-tool timeout, a
// ToolPolicy denying any tool named in --deny (decisions logged alongside
// the hooks), and — behind --parallel — concurrent dispatch of the tool
// calls that a single completion requests. The provider client is built with a
// retry schedule (provider.Options.Retry) and an Observer that logs each
// provider attempt, so a retried completion is visible on stderr. Behind
// --record <dir>, the run is persisted as "<dir>/<run id>.jsonl" through
// agent.JSONL, registered with agent.WithObserver, and the run id is
// printed; examples/replay replays that record offline.
//
// Usage:
//
//	# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
//	go run ./examples/agent [--parallel] [--task "..."] [--deny now,add] [--record dir]
//
// Without the environment variables set, the program prints usage and exits
// non-zero without touching the network.
package main

import (
	"context"
	"errors"
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
	"github.com/dpoage/llmkit/retry"
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
	record := flag.String("record", "", "directory to persist the run in as <run id>.jsonl (agent.JSONL); examples/replay replays it")
	task := flag.String("task", "What time is it right now, and what is 41 plus 58? Use the tools to answer both parts.",
		"the task to give the agent")
	flag.Parse()

	spec, err := envcfg.Load(envcfg.Usage("go run ./examples/agent",
		`[--parallel] [--task "..."] [--deny now,add] [--record dir]`))
	if err != nil {
		return err
	}

	// Retry and Observer configure the provider stack: a failed attempt
	// that is retryable is retried up to MaxAttempts with backoff (unset
	// schedule fields take retry.Default's values), and the Observer sees
	// one Attempt event per try, failures included. Under the agent Runner
	// the Observer reports Attempts only; the Runner reports the logical
	// completion itself (the JSONL record below).
	client, err := provider.New(context.Background(), spec, provider.Options{
		Retry:    retry.Config{MaxAttempts: 3},
		Observer: llmkit.ObserverFunc(logAttempt),
	})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	// Hooks are synchronous: each runs inline on the goroutine that reaches
	// the fire point. Under --parallel, ToolStart/ToolEnd fire concurrently
	// from the per-call goroutines, so a real hook must synchronize its own state.
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
	// call, in model order, on the loop goroutine before any Tool.Run of
	// the turn dispatches. A denial feeds the model
	// "ERROR: tool <name> denied: …" and the run continues; the decision is
	// logged alongside the hooks.
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
	// The durable record: one "<run id>.jsonl" line per event. The sink
	// never fails the run; its refusals and write failures arrive on onErr.
	// The run id is pinned up front and printed before the run starts, so a
	// run that fails still names its record.
	var runOpts []agent.RunOption
	if *record != "" {
		opts = append(opts, agent.WithObserver(agent.JSONL(*record, func(err error) {
			log.Printf("record: %v", err)
		})))
		id := llmkit.NewRunID()
		runOpts = append(runOpts, agent.WithRunID(id))
		fmt.Println("run id:    ", id)
	}

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
	outcome, err := runner.Run(context.Background(), *task, runOpts...)
	var incomplete *agent.IncompleteError
	switch {
	case errors.As(err, &incomplete):
		// A limit stopped the run before the model finished: its last text
		// is not an answer.
		outcome = incomplete.Outcome
		fmt.Println()
		fmt.Println("stopped:   ", incomplete.Reason, "(no final answer)")
		if outcome.FinalText != "" {
			fmt.Println("last text: ", outcome.FinalText)
		}
	case err != nil:
		return fmt.Errorf("run: %w", err)
	default:
		fmt.Println()
		if outcome.FinalText != "" {
			fmt.Println("final:     ", outcome.FinalText)
		} else {
			fmt.Println("final:      (no assistant text produced)")
		}
	}
	u := outcome.Usage
	fmt.Printf("usage:      input=%d output=%d over %d turn(s)\n", u.InputTokens, u.OutputTokens, outcome.Iterations)
	return nil
}

// addArgs are the arguments of the add tool; agent.SchemaOf derives the
// advertised parameter schema from these fields.
type addArgs struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

// logAttempt is the provider Observer: it logs each provider attempt the
// retry stage makes, so a retried completion shows as several lines.
func logAttempt(_ context.Context, ev llmkit.Event) {
	if ev.Kind != llmkit.KindAttempt || ev.Attempt == nil {
		return
	}
	a := ev.Attempt
	if a.Err != "" {
		log.Printf("attempt %d failed (status %d): %s", a.Attempt, a.StatusCode, a.Err)
		return
	}
	log.Printf("attempt %d ok", a.Attempt)
}

// Command chat demonstrates multi-turn conversation through the llmkit/agent
// harness: a stdin REPL where each line becomes the next task via
// agent.Runner.Run with agent.Continue(prev), so the model keeps every
// earlier turn of the session. One trivial tool (now) keeps the tool-calling
// path exercised. While a run is in flight the prompt changes to "steered>"
// and a typed line becomes mid-run steering (agent.NewSteering +
// agent.WithSteering): the turn joins the running conversation before the
// next model call, or when the run would otherwise end. When the model stops
// for a refusal/safety reason (agent.StopReasonError) a canned reply is
// printed and the refusal turn stays in the history: the attached
// err.Outcome is threaded into the next run's agent.Continue so the
// conversation continues from it. A /think command toggles extended thinking
// via an agent.RequestPolicy when the model reports thinking support.
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

// runResult carries one finished run back to the input loop.
type runResult struct {
	outcome *agent.Outcome
	err     error
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

	now := agent.Func[struct{}]("now", "returns the current local date and time",
		func(_ context.Context, _ struct{}) (string, error) {
			return time.Now().Format(time.RFC3339), nil
		})

	// A RequestPolicy shapes every outgoing completion request just before
	// the wire call. The /think command flips this one's atomic flag.
	think := &thinkPolicy{}
	thinkingSupported := client.Capabilities().Thinking
	if !thinkingSupported {
		fmt.Println("note: model reports no thinking support; /think stays off")
	}
	runner := agent.NewRunner(client, []agent.Tool{now},
		"You are a helpful assistant in a multi-turn chat. You remember every earlier turn of this conversation. Use the provided tool when it would help.",
		agent.WithRequestPolicy(think))

	// stdin feeds a channel so the input loop keeps reading while a run
	// executes: each line typed mid-run becomes steering for that run.
	lines := make(chan string)
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		for sc.Scan() {
			lines <- sc.Text()
		}
		close(lines)
	}()

	fmt.Println("llmkit chat — type a message (/think toggles extended thinking, ctrl-d to exit).")

	var prev *agent.Outcome
	for {
		fmt.Print("you> ")
		line, ok := <-lines
		if !ok {
			break
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "/think" {
			toggleThink(think, thinkingSupported)
			continue
		}

		// Idle: start a run with a fresh steering handle. Lines typed after
		// the run finished are new tasks, not stale steering.
		steering := agent.NewSteering()
		result := make(chan runResult, 1)
		go func(task string, p *agent.Outcome) {
			o, e := runner.Run(context.Background(), task,
				agent.Continue(p), agent.WithSteering(steering))
			result <- runResult{outcome: o, err: e}
		}(line, prev)

		// Serve stdin while the run is in flight; the prompt marks it.
		for {
			fmt.Print("steered> ")
			select {
			case l, ok := <-lines:
				if !ok {
					// stdin closed mid-run: nothing more to steer; let the
					// run finish, print its answer, and exit.
					p, rerr := reportRun(<-result)
					prev = p
					if rerr != nil {
						return rerr
					}
					return nil
				}
				l = strings.TrimSpace(l)
				switch {
				case l == "":
				case l == "/think":
					toggleThink(think, thinkingSupported)
				default:
					steering.Steer(llmkit.Text(l))
				}
			case res := <-result:
				p, rerr := reportRun(res)
				prev = p
				if rerr != nil {
					return rerr
				}
				steering = nil
			}
			if steering == nil {
				break
			}
		}
	}
	return nil
}

// reportRun prints one finished run and returns the conversation state to
// continue from: the refusal outcome on a StopReasonError (the canned reply
// keeps the refusal turn in the history), or the run's outcome on success.
func reportRun(res runResult) (*agent.Outcome, error) {
	var stopErr *agent.StopReasonError
	if errors.As(res.err, &stopErr) {
		fmt.Printf("assistant> (the model declined: %s)\n", stopErr.StopReason)
		return stopErr.Outcome, nil
	}
	if res.err != nil {
		return nil, fmt.Errorf("run: %w", res.err)
	}
	if res.outcome.FinalText != "" {
		fmt.Println("assistant>", res.outcome.FinalText)
	} else {
		fmt.Println("assistant> (no assistant text produced)")
	}
	return res.outcome, nil
}

// toggleThink flips the extended-thinking flag and prints the new state.
func toggleThink(p *thinkPolicy, supported bool) {
	if !supported {
		fmt.Println("(extended thinking is not supported by this model)")
		return
	}
	p.on.Store(!p.on.Load())
	if p.on.Load() {
		fmt.Println("(extended thinking on)")
	} else {
		fmt.Println("(extended thinking off)")
	}
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

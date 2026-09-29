// Command replay replays a recorded agent run offline: it reads the run's
// "<record-dir>/<run-id>.jsonl" record (written by examples/agent --record),
// serves the recorded completions and tool results through
// agent.NewReplayClient, runs the same task through a fresh agent.Runner
// over that client, and prints the replayed final text. No provider is
// contacted and no LLMKIT_* variables are read, so it needs no credentials
// and no network.
//
// Usage:
//
//	go run ./examples/replay <record-dir> <run-id>
//
// The program exits 1 when ReplayClient.Err reports a divergence, whether or
// not Run returned an error, or when Run returns an error — including an
// *agent.IncompleteError, the result of replaying a record whose run
// stopped at a limit or without an answer. With the wrong number of
// arguments it prints usage and exits 1.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
)

const usage = `usage: go run ./examples/replay <record-dir> <run-id>

replays the run recorded in <record-dir>/<run-id>.jsonl offline; the run id is
printed by a recording run:
  go run ./examples/agent --record <record-dir>`

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "replay:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) != 2 {
		return errors.New(usage)
	}
	dir, id := args[0], llmkit.RunID(args[1])
	ctx := context.Background()

	// The JSONL sink is also a llmkit.Source: the read side of the record.
	src := agent.JSONL(dir, nil)
	task, err := recordedTask(ctx, src, id)
	if err != nil {
		return err
	}

	rc, err := agent.NewReplayClient(src, id, llmkit.Capabilities{})
	if err != nil {
		return err
	}
	// rc.Tools() are the run's recorded tools, bound to rc: each serves its
	// recorded result and is never executed live. rc.ToolPolicy() reproduces
	// the tool calls the recorded run's policy denied.
	runner := agent.NewRunner(rc, rc.Tools(), "Replay of a recorded run.",
		agent.WithToolPolicy(rc.ToolPolicy()))
	outcome, runErr := runner.Run(ctx, task)
	// Consult the client's divergence state whatever Run returned: a replay
	// that diverges and then stops at a limit gets the limit's error from
	// Run, and only Err names the divergence.
	if err := rc.Err(); err != nil {
		return fmt.Errorf("replay diverged: %w", err)
	}
	if runErr != nil {
		return fmt.Errorf("replay run: %w", runErr)
	}

	fmt.Println("task:      ", task)
	fmt.Println("final:     ", outcome.FinalText)
	u := outcome.Usage
	fmt.Printf("usage:      input=%d output=%d over %d turn(s)\n", u.InputTokens, u.OutputTokens, outcome.Iterations)
	return nil
}

// recordedTask returns the task the recorded run was given: the Task on the
// run's Start event.
func recordedTask(ctx context.Context, src llmkit.Source, id llmkit.RunID) (string, error) {
	events, err := src.Events(ctx, id)
	if err != nil {
		return "", err
	}
	for _, ev := range events {
		if ev.Kind == llmkit.KindStart && ev.Start != nil {
			return ev.Start.Task, nil
		}
	}
	return "", fmt.Errorf("run %s has no start event", id)
}

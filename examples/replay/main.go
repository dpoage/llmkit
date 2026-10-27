// Command replay replays a recorded agent run offline: it reads the run's
// events from an llmkit.Source — a store/sqlite database (--store, written
// by examples/agent --store) or a "<record-dir>/<run-id>.jsonl" record
// (written by examples/agent --record) — serves the recorded completions
// and tool results through agent.NewReplayClient, runs the same task
// through a fresh agent.Runner over that client, and prints the replayed
// final text. No provider is contacted and no LLMKIT_* variables are read,
// so it needs no credentials and no network.
//
// Usage:
//
//	go run ./examples/replay --store <db> --run <run-id>
//	go run ./examples/replay <record-dir> <run-id>
//
// The program exits 1 when ReplayClient.Err reports a divergence, whether or
// not Run returned an error, or when Run returns an error — including an
// *agent.IncompleteError, the result of replaying a record whose run
// stopped at a limit or without an answer. With the wrong arguments it
// prints usage and exits 1.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/store/sqlite"
)

const usage = `usage: go run ./examples/replay --store <db> --run <run-id>
       go run ./examples/replay <record-dir> <run-id>

replays a recorded run offline from a store/sqlite database:
  go run ./examples/agent --store <db>
or from a JSONL record:
  go run ./examples/agent --record <record-dir>

` + foreignStore

// foreignStore states what --store does to a SQLite file the store did not create (llmkit-rqmm).
const foreignStore = "--store on a foreign SQLite file at user_version 0 writes the store schema into it and switches it to WAL; a colliding name of any kind (table, view or index), or a foreign file at user_version 1, fails after the WAL switch; user_version 2 or greater fails with ErrSchemaTooNew, leaving the file unchanged"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "replay:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	var storePath, runFlag string
	flags := flag.NewFlagSet("replay", flag.ContinueOnError)
	flags.StringVar(&storePath, "store", "", "path to a store/sqlite database the run is read from\n"+foreignStore)
	flags.StringVar(&runFlag, "run", "", "the run id to replay (required with --store)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	args = flags.Args()

	var src llmkit.Source
	var id llmkit.RunID
	switch {
	case storePath != "":
		if runFlag == "" {
			return errors.New("--store needs --run <run-id>")
		}
		if len(args) != 0 {
			return errors.New(usage)
		}
		// A reader must not create the database Open would mint on a typo.
		if _, err := os.Stat(storePath); err != nil {
			return fmt.Errorf("store %s: %w", storePath, err)
		}
		st, err := sqlite.Open(storePath, nil)
		if err != nil {
			return err
		}
		defer func() { _ = st.Close() }()
		src, id = st, llmkit.RunID(runFlag)
	case runFlag != "":
		return errors.New("--run needs --store <db>")
	case len(args) == 2:
		// The JSONL sink is also a llmkit.Source: the read side of the record.
		src, id = agent.JSONL(args[0], nil), llmkit.RunID(args[1])
	default:
		return errors.New(usage)
	}
	ctx := context.Background()

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

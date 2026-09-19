// Command structured demonstrates schema-constrained output with the
// llmkit/agent harness: agent.RunJSONAs[book] drives the tool loop with a
// schema derived from the Go type via agent.SchemaOf, validates the model's
// answer against it, and unmarshals it into book (with one repair round-trip
// on failure). It doubles as a compile-time contract check for the
// RunJSONAs/SchemaOf surface.
//
// Usage:
//
//	# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
//	go run ./examples/structured [--task "..."]
//
// Without the environment variables set, the program prints usage and exits
// non-zero without touching the network.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/examples/internal/envcfg"
	"github.com/dpoage/llmkit/provider"
)

// book is the shape the final answer must have; agent.SchemaOf derives the
// JSON Schema from these fields and tags. No field has omitempty, so all
// three are required; unknown keys are rejected; title/author must be
// non-empty and publication_year at least 1000.
type book struct {
	Title           string `json:"title" jsonschema:"minLength=1"`
	Author          string `json:"author" jsonschema:"minLength=1"`
	PublicationYear int    `json:"publication_year" jsonschema:"minimum=1000"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "structured:", err)
		os.Exit(1)
	}
}

func run() error {
	task := flag.String("task", "Pick one classic science fiction novel and report its title, author, and publication year.",
		"the question whose answer must match the schema")
	flag.Parse()

	spec, err := envcfg.Load(envcfg.Usage("go run ./examples/structured",
		`[--task "..."]`))
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	// No tools: the run is a single schema-bearing completion. RunJSONAs
	// derives the schema from book (see above), attaches it to the request
	// when the client reports StructuredOutput capability, and deep-validates
	// the answer either way before unmarshaling into a fresh book.
	runner := agent.NewRunner(client, nil,
		"You are a precise reference assistant. Answer only in JSON matching the requested schema.",
		agent.WithMaxTokens(1024))

	out, outcome, err := agent.RunJSONAs[book](context.Background(), runner, *task)
	if err != nil {
		return fmt.Errorf("runjson: %w", err)
	}

	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal result: %w", err)
	}
	fmt.Println(string(b))
	fmt.Printf("parsed ok: %d turn(s), input=%d output=%d\n",
		outcome.Iterations, outcome.Usage.InputTokens, outcome.Usage.OutputTokens)
	return nil
}

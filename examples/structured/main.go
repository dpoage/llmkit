// Command structured demonstrates schema-constrained output with the
// llmkit/agent harness: Runner.RunJSON drives the tool loop with a JSON
// Schema, validates the model's answer against it, and unmarshals it into a
// Go struct (with one repair round-trip on failure). It doubles as a
// compile-time contract check for the RunJSON surface.
//
// Usage:
//
//	export LLMKIT_PROVIDER=openai   # anthropic | openai | openai-compatible | google
//	export LLMKIT_MODEL=gpt-4o-mini
//	export LLMKIT_API_KEY=sk-...
//	export LLMKIT_BASE_URL=...      # optional; e.g. a local OpenAI-compatible endpoint
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

// book is the shape the final answer must have. The schema below mirrors it.
type book struct {
	Title           string `json:"title"`
	Author          string `json:"author"`
	PublicationYear int    `json:"publication_year"`
}

var bookSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "title":            {"type": "string", "minLength": 1},
    "author":           {"type": "string", "minLength": 1},
    "publication_year": {"type": "integer", "minimum": 1000}
  },
  "required": ["title", "author", "publication_year"],
  "additionalProperties": false
}`)

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

	spec, err := envcfg.Load(`missing environment:
  LLMKIT_PROVIDER  anthropic | openai | openai-compatible | google
  LLMKIT_MODEL     model name, e.g. claude-sonnet-4-5 or gpt-4o-mini
  LLMKIT_API_KEY   provider API key (any placeholder for a local endpoint)
  LLMKIT_BASE_URL  optional base URL for OpenAI-compatible endpoints

set the variables above, then re-run:
  go run ./examples/structured [--task "..."]`)
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	// No tools: the run is a single schema-bearing completion. RunJSON
	// attaches bookSchema to the request when the client reports
	// StructuredOutput capability, and deep-validates the answer either way
	// before unmarshaling into out.
	runner := agent.NewRunner(client, nil,
		"You are a precise reference assistant. Answer only in JSON matching the requested schema.",
		agent.WithMaxTokens(1024))

	var out book
	outcome, err := runner.RunJSON(context.Background(), *task, bookSchema, &out)
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

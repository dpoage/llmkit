package provider_test

import (
	"context"
	"fmt"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

// ExampleNew builds a client for a local Ollama server through the
// openai-compatible provider. Construction performs no network I/O — it only
// validates the spec and wires the adapter and wrappers — so the example runs
// without a server listening on the endpoint.
func ExampleNew() {
	spec := provider.Spec{
		Type:    provider.TypeOpenAICompatible,
		Model:   "llama3.1",
		BaseURL: "http://localhost:11434/v1", // local Ollama
		Secret:  "ollama",                    // placeholder; New checks presence only
	}
	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	caps := client.Capabilities()
	fmt.Println("context window:", caps.ContextWindow, "(0 = unknown)")
	fmt.Println("parallel tool calls:", caps.ParallelToolCalls)
	fmt.Println("structured output:", caps.StructuredOutput)
	fmt.Println("thinking:", caps.Thinking)
	fmt.Println("tool choice:", caps.ToolChoice)
	fmt.Println("images:", caps.Images)
	// Output:
	// context window: 0 (0 = unknown)
	// parallel tool calls: false
	// structured output: false
	// thinking: false
	// tool choice: true
	// images: true
}

// ExampleSpec_capabilities pins the ContextWindow for a model the adapter's
// table cannot know. The override runs once at construction; its return value
// becomes the profile every caller sees.
func ExampleSpec_capabilities() {
	spec := provider.Spec{
		Type:    provider.TypeOpenAICompatible,
		Model:   "qwen3-custom",
		BaseURL: "http://localhost:11434/v1",
		Secret:  "ollama",
		Capabilities: func(c llmkit.Capabilities) llmkit.Capabilities {
			c.ContextWindow = 131072
			return c
		},
	}
	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("context window:", client.Capabilities().ContextWindow)
	// Output:
	// context window: 131072
}

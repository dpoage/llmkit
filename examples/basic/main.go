// Command basic demonstrates a minimal llmkit completion: build a client
// from the environment, send one user message (text, optionally with an
// image block), and print the normalized response. It doubles as a
// compile-time contract check for the root package surface —
// provider.New, TextMessage, content blocks, Capabilities, and Response.
//
// Usage:
//
//	export LLMKIT_PROVIDER=openai   # anthropic | openai | openai-compatible | google
//	export LLMKIT_MODEL=gpt-4o-mini
//	export LLMKIT_API_KEY=sk-...
//	export LLMKIT_BASE_URL=...      # optional; e.g. a local OpenAI-compatible endpoint
//	go run ./examples/basic [--image path/to/photo.jpg]
//
// With --image, the user message carries a second content block
// (llmkit.BlockImage with the file's bytes inline), guarded by the model's
// Capabilities.Images — an advisory flag: adapters do not read it, but this
// example refuses to send an image the model does not advertise. Without
// the environment variables set, the program prints usage and exits
// non-zero without touching the network.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"mime"
	"os"
	"path/filepath"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

var errUsage = errors.New(`missing environment:
  LLMKIT_PROVIDER  anthropic | openai | openai-compatible | google
  LLMKIT_MODEL     model name, e.g. claude-sonnet-4-5 or gpt-4o-mini
  LLMKIT_API_KEY   provider API key (any placeholder for a local endpoint)
  LLMKIT_BASE_URL  optional base URL for OpenAI-compatible endpoints

set the variables above, then re-run:
  go run ./examples/basic [--image path/to/photo.jpg]`)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "basic:", err)
		os.Exit(1)
	}
}

func run() error {
	imagePath := flag.String("image", "", "optional image file to include in the user message")
	flag.Parse()

	providerName := os.Getenv("LLMKIT_PROVIDER")
	model := os.Getenv("LLMKIT_MODEL")
	apiKey := os.Getenv("LLMKIT_API_KEY")
	baseURL := os.Getenv("LLMKIT_BASE_URL")
	if providerName == "" || model == "" || apiKey == "" {
		return errUsage
	}

	spec, err := specFor(providerName, baseURL)
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, providerName, model, apiKey, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	msg := llmkit.TextMessage(llmkit.RoleUser, "Say hello in one short sentence.")
	if *imagePath != "" {
		block, err := imageBlock(*imagePath)
		if err != nil {
			return err
		}
		caps := client.Capabilities()
		if !caps.Images {
			return fmt.Errorf("model %s reports Capabilities.Images=false — the flag is advisory (llmkit sends the block anyway and the provider may reject the request), so this example refuses to send it; re-run without --image or pick another model", model)
		}
		msg.Content = append(msg.Content, block)
	}

	resp, err := client.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{msg},
	})
	if err != nil {
		return fmt.Errorf("complete: %w", err)
	}

	fmt.Println("text:      ", resp.Text)
	u := resp.Usage
	fmt.Printf("usage:      input=%d output=%d (cache read=%d, cache creation=%d)\n",
		u.InputTokens, u.OutputTokens, u.CacheReadInputTokens, u.CacheCreationInputTokens)
	fmt.Println("stop reason:", resp.StopReason)
	return nil
}

// specFor maps the provider name onto provider.Type with a friendly error
// listing the accepted values. BaseURL is empty unless the caller set
// LLMKIT_BASE_URL (used by openai-compatible and useful for proxies).
func specFor(providerName, baseURL string) (provider.Spec, error) {
	var spec provider.Spec
	switch providerName {
	case "anthropic":
		spec.Type = provider.TypeAnthropic
	case "openai":
		spec.Type = provider.TypeOpenAI
	case "openai-compatible":
		spec.Type = provider.TypeOpenAICompatible
	case "google":
		spec.Type = provider.TypeGoogle
	default:
		return provider.Spec{}, fmt.Errorf("unknown LLMKIT_PROVIDER %q: expected anthropic, openai, openai-compatible, or google", providerName)
	}
	spec.BaseURL = baseURL
	return spec, nil
}

// imageBlock reads an image file into an inline BlockImage content block.
// Block.Data carries the raw bytes (never pre-encoded); adapters base64 them
// per provider wire format.
func imageBlock(path string) (llmkit.Block, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return llmkit.Block{}, fmt.Errorf("read image: %w", err)
	}
	mediaType := mime.TypeByExtension(filepath.Ext(path))
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return llmkit.Block{
		Kind:      llmkit.BlockImage,
		MediaType: mediaType,
		Data:      data,
	}, nil
}

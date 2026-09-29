// Command basic demonstrates a minimal llmkit completion: build a client
// from the environment, send one user message (text, optionally with an
// image block), and print the normalized response.
//
// Usage:
//
//	# export the shared LLMKIT_* variables (examples/internal/envcfg), then:
//	go run ./examples/basic [--image path/to/photo.jpg]
//
// With --image, the user message is llmkit.UserMessage(llmkit.Text(...),
// llmkit.Image(mediaType, data)): a second content block carrying the file's
// bytes inline, guarded by the model's Capabilities.Images — an advisory
// flag: adapters do not read it, but this example refuses to send an image
// the model does not advertise. Without the environment variables set, the
// program prints usage and exits non-zero without touching the network.
package main

import (
	"context"
	"flag"
	"fmt"
	"mime"
	"os"
	"path/filepath"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/examples/internal/envcfg"
	"github.com/dpoage/llmkit/provider"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "basic:", err)
		os.Exit(1)
	}
}

func run() error {
	imagePath := flag.String("image", "", "optional image file to include in the user message")
	flag.Parse()

	spec, err := envcfg.Load(envcfg.Usage("go run ./examples/basic",
		"[--image path/to/photo.jpg]"))
	if err != nil {
		return err
	}

	client, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	const prompt = "Say hello in one short sentence."
	var msg llmkit.Message
	if *imagePath == "" {
		msg = llmkit.TextMessage(llmkit.RoleUser, prompt)
	} else {
		mediaType, data, err := readImage(*imagePath)
		if err != nil {
			return err
		}
		caps := client.Capabilities()
		if !caps.Images {
			return fmt.Errorf("model %s reports Capabilities.Images=false — the flag is advisory (llmkit sends the block anyway and the provider may reject the request), so this example refuses to send it; re-run without --image or pick another model", spec.Model)
		}
		msg = llmkit.UserMessage(llmkit.Text(prompt), llmkit.Image(mediaType, data))
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

// readImage reads an image file and returns its MIME type (by extension)
// and raw bytes, ready for llmkit.Image. The bytes are never pre-encoded;
// adapters base64 them per provider wire format. It rejects an empty file
// itself, because llmkit.Image panics on empty data.
func readImage(path string) (mediaType string, data []byte, err error) {
	data, err = os.ReadFile(path)
	if err != nil {
		return "", nil, fmt.Errorf("read image: %w", err)
	}
	if len(data) == 0 {
		return "", nil, fmt.Errorf("read image: %s is empty", path)
	}
	mediaType = mime.TypeByExtension(filepath.Ext(path))
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return mediaType, data, nil
}

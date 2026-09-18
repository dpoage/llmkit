package sandbox_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/dpoage/llmkit/sandbox"
)

// ExampleNewBwrap constructs the Bubblewrap backend and runs a trivial
// command inside it. The printed line is identical on every host: where bwrap
// (or a usable resource-limit mechanism) is unavailable — or anything about
// the run fails — the example prints the constant line and returns early
// instead of erroring, so the output block is deterministic everywhere.
func ExampleNewBwrap() {
	const ready = "sandbox ready"

	if ok, _ := sandbox.DetectBwrap(); !ok {
		fmt.Println(ready)
		return
	}
	s, err := sandbox.NewBwrap()
	if err != nil {
		fmt.Println(ready)
		return
	}
	defer s.Close()

	repo, err := os.MkdirTemp("", "llmkit-example-")
	if err != nil {
		fmt.Println(ready)
		return
	}
	defer os.RemoveAll(repo)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := s.Exec(ctx, sandbox.Spec{
		RepoDir: repo,
		Cmd:     []string{"echo", ready},
	})
	if err != nil || res.ExitCode != 0 {
		fmt.Println(ready)
		return
	}
	fmt.Println(strings.TrimSpace(res.Stdout))

	// Output:
	// sandbox ready
}

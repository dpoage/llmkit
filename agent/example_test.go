package agent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strconv"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
)

// weatherArgs is the schema for the ExampleNewRunner weather tool.
type weatherArgs struct {
	City string `json:"city" jsonschema:"description=the city to look up"`
}

// ExampleNewRunner builds a Runner with one Func tool and drives it through a
// ReplayClient scripted with two responses: a tool call, then the final text.
func ExampleNewRunner() {
	weather := agent.Func("weather", "look up the current weather for a city",
		func(_ context.Context, a weatherArgs) (string, error) {
			return "18°C, clear", nil
		})

	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls: []llmkit.ToolCall{{
				ID:        "call-1",
				Name:      "weather",
				Arguments: json.RawMessage(`{"city":"Tokyo"}`),
			}},
			StopReason: llmkit.StopToolUse,
		},
		{
			Text:       "Tokyo is 18°C and clear.",
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	runner := agent.NewRunner(client, []agent.Tool{weather}, "Answer weather questions.")

	outcome, err := runner.Run(context.Background(), "What is the weather in Tokyo?")
	if err != nil {
		log.Fatal(err)
	}
	toolCalls := 0
	for _, ev := range outcome.Transcript.Events {
		if ev.Kind == agent.EventToolResult {
			toolCalls++
		}
	}
	fmt.Println(outcome.FinalText)
	fmt.Println("tool calls:", toolCalls)
	// Output:
	// Tokyo is 18°C and clear.
	// tool calls: 1
}

// exampleAddArgs is the schema for the ExampleFunc add tool.
type exampleAddArgs struct {
	A int `json:"a" jsonschema:"description=first addend"`
	B int `json:"b" jsonschema:"description=second addend"`
}

// ExampleFunc builds a Tool from a plain function. The struct argument is the
// schema: the runner validates the model's JSON arguments against it and
// decodes them before fn runs. This example invokes the tool directly to show
// the typed decoding.
func ExampleFunc() {
	tool := agent.Func("add", "add two integers",
		func(_ context.Context, a exampleAddArgs) (string, error) {
			return strconv.Itoa(a.A + a.B), nil
		})

	result, err := tool.Run(context.Background(), json.RawMessage(`{"a":2,"b":40}`))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
	// Output:
	// 42
}

// tripAnswer is the structured output schema for ExampleRunJSONAs.
type tripAnswer struct {
	City string `json:"city" jsonschema:"description=destination city"`
	Days int    `json:"days" jsonschema:"description=trip length in days"`
}

// ExampleRunJSONAs derives the answer schema from a Go type and returns the
// decoded value. The replayed response is the JSON the model would produce.
func ExampleRunJSONAs() {
	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			Text:       `{"city":"Paris","days":3}`,
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	runner := agent.NewRunner(client, nil, "You plan trips.")

	answer, _, err := agent.RunJSONAs[tripAnswer](context.Background(), runner, "Plan a 3-day trip to Paris.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s, %d days\n", answer.City, answer.Days)
	// Output:
	// Paris, 3 days
}

// ExampleWithToolPolicy denies a tool call. The model receives the denial as
// an "ERROR:"-prefixed tool result and recovers; the transcript records it.
func ExampleWithToolPolicy() {
	deleteFile := agent.Func("delete_file", "delete a file",
		func(_ context.Context, _ struct{}) (string, error) {
			return "deleted", nil
		})

	policy := agent.ToolPolicyFunc(func(_ context.Context, call *llmkit.ToolCall) error {
		if call.Name == "delete_file" {
			return errors.New("deletes require manual approval")
		}
		return nil
	})

	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls: []llmkit.ToolCall{{
				ID:        "call-1",
				Name:      "delete_file",
				Arguments: json.RawMessage(`{}`),
			}},
			StopReason: llmkit.StopToolUse,
		},
		{
			Text:       "I could not delete the file: deletes require manual approval.",
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	runner := agent.NewRunner(client, []agent.Tool{deleteFile}, "You manage files.",
		agent.WithToolPolicy(policy))

	outcome, err := runner.Run(context.Background(), "Delete /tmp/old.txt")
	if err != nil {
		log.Fatal(err)
	}
	for _, ev := range outcome.Transcript.Events {
		if ev.Kind == agent.EventToolResult {
			fmt.Printf("%s -> %s (isError: %v)\n", ev.ToolName, ev.Result, ev.IsError)
		}
	}
	// Output:
	// delete_file -> ERROR: tool delete_file denied: deletes require manual approval (isError: true)
}

// ExampleWithRequestPolicy caps output tokens on every request. The policy
// runs before Hooks.BeforeCompletion, so the hook observes the edited
// request — the exact request the client receives.
func ExampleWithRequestPolicy() {
	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			Text:       "4",
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	var maxTokensOnWire int
	runner := agent.NewRunner(client, nil, "Answer briefly.",
		agent.WithRequestPolicy(agent.RequestPolicyFunc(
			func(_ context.Context, _ int, req *llmkit.Request) error {
				req.MaxTokens = 128
				return nil
			})),
		agent.WithHooks(agent.Hooks{
			BeforeCompletion: func(_ context.Context, _ int, req *llmkit.Request) {
				maxTokensOnWire = req.MaxTokens
			},
		}))

	if _, err := runner.Run(context.Background(), "What is 2+2?"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("max tokens on the wire:", maxTokensOnWire)
	// Output:
	// max tokens on the wire: 128
}

// echoArgs is the schema for the ExampleWithHooks echo tool.
type echoArgs struct {
	Text string `json:"text" jsonschema:"description=the text to echo"`
}

// ExampleWithHooks observes tool calls with Hooks.ToolEnd. The Step field is
// the transcript step of the turn that requested the call.
func ExampleWithHooks() {
	echo := agent.Func("echo", "echo the given text",
		func(_ context.Context, a echoArgs) (string, error) {
			return a.Text, nil
		})

	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls: []llmkit.ToolCall{{
				ID:        "call-1",
				Name:      "echo",
				Arguments: json.RawMessage(`{"text":"hello"}`),
			}},
			StopReason: llmkit.StopToolUse,
		},
		{
			Text:       "done",
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	runner := agent.NewRunner(client, []agent.Tool{echo}, "You echo things.",
		agent.WithHooks(agent.Hooks{
			ToolEnd: func(_ context.Context, ev agent.ToolEvent) {
				fmt.Printf("step %d: %s -> %s\n", ev.Step, ev.Call.Name, ev.Result)
			},
		}))

	if _, err := runner.Run(context.Background(), "Echo hello"); err != nil {
		log.Fatal(err)
	}
	// Output:
	// step 1: echo -> hello
}

// ExampleSteering queues a follow-up turn that extends a run past its
// would-be finish. The run asks for a draft, the model finishes, the
// follow-up delivers, and the loop continues for a second draft. The
// follow-up appears as an ordinary user message in Outcome.Messages.
func ExampleSteering() {
	tool := agent.Func("echo", "echo the given text",
		func(_ context.Context, a echoArgs) (string, error) {
			return a.Text, nil
		})

	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls: []llmkit.ToolCall{{
				ID:        "call-1",
				Name:      "echo",
				Arguments: json.RawMessage(`{"text":"draft"}`),
			}},
			StopReason: llmkit.StopToolUse,
		},
		{
			Text:       "Draft: draft.",
			StopReason: llmkit.StopEndTurn,
		},
		{
			Text:       "Shorter draft: draft.",
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	steering := agent.NewSteering()
	steering.FollowUp(llmkit.Text("Make it shorter."))

	runner := agent.NewRunner(client, []agent.Tool{tool}, "You draft copy.")

	outcome, err := runner.Run(context.Background(), "Write a tagline.", agent.WithSteering(steering))
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range outcome.Messages {
		fmt.Println(m.Role)
	}
	fmt.Println("undelivered:", steering.Pending())
	// Output:
	// user
	// assistant
	// tool-result
	// assistant
	// user
	// assistant
	// undelivered: 0
}

// ExampleLoadJSONL saves a run's transcript as JSONL, loads it back, and
// walks the events. The record order per step is the request, the assistant
// turn, then each tool result.
func ExampleLoadJSONL() {
	tick := agent.Func("tick", "return one tick",
		func(_ context.Context, _ struct{}) (string, error) {
			return "tick", nil
		})

	client := agent.NewReplayClientFromResponses([]llmkit.Response{
		{
			ToolCalls: []llmkit.ToolCall{{
				ID:        "call-1",
				Name:      "tick",
				Arguments: json.RawMessage(`{}`),
			}},
			StopReason: llmkit.StopToolUse,
		},
		{
			Text:       "finished",
			StopReason: llmkit.StopEndTurn,
		},
	}, llmkit.Capabilities{})

	runner := agent.NewRunner(client, []agent.Tool{tick}, "Tick once.")

	outcome, err := runner.Run(context.Background(), "Tick")
	if err != nil {
		log.Fatal(err)
	}

	var buf bytes.Buffer
	if err := outcome.Transcript.SaveJSONL(&buf); err != nil {
		log.Fatal(err)
	}
	loaded, err := agent.LoadJSONL(&buf)
	if err != nil {
		log.Fatal(err)
	}
	for _, ev := range loaded.Events {
		fmt.Printf("step %d: %s\n", ev.Step, ev.Kind)
	}
	// Output:
	// step 1: request
	// step 1: assistant
	// step 1: tool_result
	// step 2: request
	// step 2: assistant
}

package sqlite_test

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/dpoage/llmkit"
)

func newEv(run llmkit.RunID, kind llmkit.EventKind) llmkit.Event {
	return llmkit.NewEvent(llmkit.WithRun(context.Background(), run), kind)
}

func f64(v float64) *float64 { return &v }

// richRequest exercises every Request and Message field.
func richRequest(extra ...llmkit.Message) llmkit.Request {
	seed, topk := int64(42), 40
	msgs := []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleSystem, "inline <system> & note"),
		{Role: llmkit.RoleUser, Content: []llmkit.Block{
			llmkit.Text("look at this"),
			llmkit.Image("image/png", []byte{0, 1, 2, 0xff, 0xfe}),
			llmkit.ImageURL("https://example.test/a.png"),
			llmkit.Document("application/pdf", "doc", []byte("%PDF-1.4")),
		}},
		{Role: llmkit.RoleAssistant, Content: []llmkit.Block{
			{Kind: llmkit.BlockThinking, Text: "hmm", Provider: "anthropic", Raw: json.RawMessage(`{"sig":"abc","n":[1,2,3]}`)},
			llmkit.Text("calling"),
		}, ToolCalls: []llmkit.ToolCall{
			{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"a": 1,  "b":"<x>"}`)},
			{ID: "c2", Name: "echo", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: llmkit.RoleToolResult, ToolCallID: "c1", Content: []llmkit.Block{llmkit.Text("ok")}},
		{Role: llmkit.RoleToolResult, ToolCallID: "c2", IsError: true, Content: []llmkit.Block{llmkit.Text("ERROR: boom")}},
	}
	return llmkit.Request{
		System:   "sys prompt \u2028 with <html> & unicode é",
		Messages: append(msgs, extra...),
		Tools: []llmkit.ToolDef{{
			Name: "echo", Description: "d", Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}}}`),
		}},
		MaxTokens:          512,
		Temperature:        f64(0.25),
		Thinking:           &llmkit.ThinkingConfig{BudgetTokens: 100, Effort: llmkit.EffortLow},
		ToolChoice:         llmkit.ToolChoice{Mode: llmkit.ToolChoiceTool, Name: "echo"},
		StopSequences:      []string{"STOP", "\n\n"},
		TopP:               f64(0.9),
		TopK:               &topk,
		Seed:               &seed,
		ResponseSchema:     json.RawMessage(`{"type":"object"}`),
		ResponseSchemaName: "answer",
	}
}

func richResponse() llmkit.Response {
	return llmkit.Response{
		Text: "answer text",
		Blocks: []llmkit.Block{
			{Kind: llmkit.BlockThinking, Provider: "openai", Raw: json.RawMessage(`{"reasoning_content":"r"}`)},
			llmkit.Text("answer text"),
		},
		ToolCalls: []llmkit.ToolCall{{ID: "c3", Name: "echo", Arguments: json.RawMessage(`{"z":null}`)}},
		Usage: llmkit.Usage{
			InputTokens: 120, OutputTokens: 30, CacheReadInputTokens: 50, CacheCreationInputTokens: 10,
			ReasoningTokens: 5, Estimated: true,
		},
		StopReason: llmkit.StopToolUse,
		Served:     llmkit.Identity{Provider: "anthropic", Model: "claude-x"},
	}
}

// allKinds returns one run holding an event of every Kind, in a plausible order.
func allKinds(run llmkit.RunID) []llmkit.Event {
	t0 := time.Date(2026, 10, 26, 9, 30, 0, 123456789, time.FixedZone("", -7*3600))
	stamp := func(e llmkit.Event, step int, d time.Duration) llmkit.Event {
		e.Time = t0.Add(time.Duration(step) * time.Second)
		e.Step = step
		e.Duration = d
		return e
	}
	req := richRequest()

	start := newEv(run, llmkit.KindStart)
	start.ParentRunID = "parent-run"
	start.Start = &llmkit.StartEvent{Task: "do the thing", Tools: []string{"echo", "other"}}

	comp := newEv(run, llmkit.KindCompletion)
	comp.SpanID = "span-1"
	comp.Completion = &llmkit.CompletionEvent{
		Request: req, Response: richResponse(), Provider: "anthropic", Model: "claude-x",
		Purpose: llmkit.CompletionPurposeCompaction,
	}

	failed := newEv(run, llmkit.KindCompletion)
	failed.Completion = &llmkit.CompletionEvent{
		Request: llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "x")}},
		Err:     "context too long", ErrKind: llmkit.ErrKindContextTooLong, Provider: "p", Model: "m",
	}

	att := newEv(run, llmkit.KindAttempt)
	att.SpanID = "span-1"
	att.Attempt = &llmkit.AttemptEvent{
		Attempt: 2, Request: req, Err: "429", StatusCode: 429,
		RetryAfter: 3 * time.Second, HasRetryAfter: true, Provider: "anthropic", Model: "claude-x",
	}

	tool := newEv(run, llmkit.KindToolRun)
	tool.ToolRun = &llmkit.ToolRunEvent{
		Call:                llmkit.ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"a":1}`)},
		DispatchedArguments: json.RawMessage(`null`),
		Result:              "[image: image/png, 5 bytes] done",
		Content:             []llmkit.Block{llmkit.Text("done"), llmkit.Image("image/png", []byte{1, 2})},
		Degraded:            1, OriginalBytes: 9000, SpillRef: "spill://1", SpillError: "disk full",
		IsError: true,
	}
	denied := newEv(run, llmkit.KindToolRun)
	denied.ToolRun = &llmkit.ToolRunEvent{
		Call:   llmkit.ToolCall{ID: "c2", Name: "rm", Arguments: json.RawMessage(`{"path":"/"}`)},
		Denied: true, DenyReason: "no rm",
	}

	compaction := newEv(run, llmkit.KindCompaction)
	compaction.Compaction = &llmkit.CompactionEvent{
		BeforeTokens: 9000, AfterTokens: 3000, Pruned: 4, Reason: llmkit.CompactionReasonOverflow, Summary: "earlier work",
	}

	steer := newEv(run, llmkit.KindSteer)
	steer.Steer = &llmkit.SteerEvent{Message: llmkit.TextMessage(llmkit.RoleUser, "also do y"), FollowUp: true}

	dec := newEv(run, llmkit.KindDecision)
	dec.Decision = &llmkit.DecisionEvent{
		Backend: "systemone", Model: "m1",
		State: json.RawMessage(`{"doc":"<a>","n":[1,2,{"k":null}]}`),
		Questions: []llmkit.DecisionQuestion{
			{ID: "q1", Kind: "noul", Instructions: json.RawMessage(`"is it?"`), True: json.RawMessage(`"yes"`), False: json.RawMessage(`"no"`)},
			{ID: "q2", Kind: "choice", Options: map[string]json.RawMessage{"a": json.RawMessage(`"A"`), "b": json.RawMessage(`{"d":1}`)}},
			{ID: "q3", Kind: "score", Levels: []json.RawMessage{json.RawMessage(`"low"`), json.RawMessage(`"high"`)}},
		},
		Answers: []llmkit.DecisionAnswer{
			{ID: "q1", Belief: f64(0.2)},
			{ID: "q2", Choice: "a", Probabilities: map[string]float64{"a": 0.7, "b": 0.3}, Confidence: 0.4},
			{ID: "q3", Score: f64(0), Levels: []string{"low", "high"}, LevelProbabilities: []float64{0.9, 0.1}},
		},
		Usage:  llmkit.Usage{InputTokens: 10, OutputTokens: 2},
		Images: []llmkit.DecisionImage{{SHA256: strings.Repeat("ab", 32), MediaType: "image/png", Placements: []string{"state", "criteria"}}},
	}
	decFailed := newEv(run, llmkit.KindDecision)
	decFailed.Decision = &llmkit.DecisionEvent{Backend: "systemone", Model: "m1", Err: "ask failed"}

	embed := newEv(run, llmkit.KindEmbed)
	embed.Embed = &llmkit.EmbedEvent{Model: "emb", Inputs: 3, Dimensions: 768}

	exec := newEv(run, llmkit.KindExec)
	exec.Exec = &llmkit.ExecEvent{Backend: "host", Command: []string{"ls", "-la", "a b"}, ExitCode: -1, StdoutBytes: 10, StderrBytes: 2, Truncated: true, Err: "killed"}

	fin := newEv(run, llmkit.KindFinalize)
	fin.Finalize = &llmkit.FinalizeEvent{
		TruncationReason: "max_iterations", Finalized: true, Usage: llmkit.Usage{InputTokens: 500, OutputTokens: 40},
		FinalText: "done", Status: llmkit.RunIncomplete, Err: "stopped",
	}

	evs := []llmkit.Event{start, comp, failed, att, tool, denied, compaction, steer, dec, decFailed, embed, exec, fin}
	for i := range evs {
		evs[i] = stamp(evs[i], i, time.Duration(i)*time.Millisecond)
	}
	return evs
}

package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

const (
	// conversationSummaryHeader starts the text of every summary message.
	conversationSummaryHeader = "[conversation summary]\n"
	// summaryToolResultChars bounds a tool result serialized for the summarizer.
	summaryToolResultChars = 2000
	// summaryMaxDepth bounds how many times an overflowing summary request is split.
	summaryMaxDepth = 3
)

// summaryInstructions is the summarizer's system prompt: the fixed summary format.
const summaryInstructions = `You summarize the conversation between a user and an AI agent that works with tools, so the agent can continue the work from your summary alone.

The conversation is inside <conversation> tags. It may start with a <previous-summary>, or hold <partial-summary> parts: fold their content into your summary and keep what is still true. Do not continue the conversation, call tools, or answer questions in it.

Reply with only the summary, in exactly this format:

## Goal
What the user wants done.

## Constraints
Requirements, preferences, and limits the user or the work imposes.

## Progress
What is done, what is in progress, and what is blocked.

## Key Decisions
Decisions made and why.

## Next Steps
What the agent should do next, in order.

## Files
Files read and files modified, one path per line, with a few words on each.`

// SummarizingCompactor returns a [Compactor] that replaces the history before
// a cut point with one model-written summary. Pass the run's own client unless
// a cheaper model will do; its calls are recorded as completion events with
// purpose "compaction" and their usage counts against the run's budgets.
//
// The cut is just before the last user message of the caller (the task or a
// steering message) that leaves at least one assistant turn after it, and
// never before a harness prompt or inside a tool call and its result. With no
// such cut it falls back to the default compactor. A summary that would not
// shrink the history is discarded. The summary is the user message
// "[conversation summary]\n" plus the summary text, and an earlier summary in
// the prefix is fed into the next one. If the summary request overflows the
// context, the prefix is summarized in halves and the halves merged, up to
// three splits deep; a single message too large for the context is an error
// wrapping [llmkit.ErrContextTooLong]. Request policies and hooks do not fire
// for the summary calls.
func SummarizingCompactor(client llmkit.Client) Compactor {
	return summarizer{client: client}
}

type summarizer struct{ client llmkit.Client }

func (s summarizer) Compact(ctx context.Context, in CompactInput) (CompactOutput, error) {
	lead, at := summaryCut(in.Messages)
	if at < 0 {
		return stubCompactor{}.Compact(ctx, in)
	}
	if s.client == nil {
		return CompactOutput{}, errors.New("agent: summarizing compactor has no client")
	}
	run := summaryRun{ctx: ctx, client: s.client}
	text, err := run.summarize(serializePrefix(in.Sent[lead:at]), 0)
	if err != nil {
		return CompactOutput{Usage: run.usage, calls: run.calls}, fmt.Errorf("agent: summarize history: %w", err)
	}
	out := summarized(in, lead, at, text)
	out.Usage, out.calls = run.usage, run.calls
	return out, nil
}

// compactCall is one model call a Compactor made, kept for its Completion event.
type compactCall struct {
	req      llmkit.Request
	resp     llmkit.Response
	err      error
	span     llmkit.SpanID
	provider string
	model    string
	duration time.Duration
}

type summaryRun struct {
	ctx    context.Context
	client llmkit.Client
	usage  llmkit.Usage
	calls  []compactCall
}

// summarize returns the summary of entries, splitting them when the request
// overflows the context.
func (r *summaryRun) summarize(entries []string, depth int) (string, error) {
	text, err := r.call(entries)
	if err == nil || !errors.Is(err, llmkit.ErrContextTooLong) || depth >= summaryMaxDepth || len(entries) < 2 {
		return text, err
	}
	mid := splitPoint(entries)
	left, err := r.summarize(entries[:mid], depth+1)
	if err != nil {
		return "", err
	}
	right, err := r.summarize(entries[mid:], depth+1)
	if err != nil {
		return "", err
	}
	return r.summarize([]string{partialSummary(left), partialSummary(right)}, depth+1)
}

// call makes one summary request on its own claimed span.
func (r *summaryRun) call(entries []string) (string, error) {
	req := llmkit.Request{
		System:   summaryInstructions,
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "<conversation>\n"+strings.Join(entries, "\n\n")+"\n</conversation>")},
	}
	cctx := llmkit.BeginCompletion(r.ctx)
	start := time.Now()
	resp, err := r.client.Complete(cctx, req)
	id := llmkit.IdentityOf(r.client)
	r.calls = append(r.calls, compactCall{
		req: req, resp: resp, err: err, span: llmkit.SpanFromContext(cctx),
		provider: id.Provider, model: id.Model, duration: time.Since(start),
	})
	if err != nil {
		return "", err
	}
	r.usage = r.usage.Add(resp.Usage)
	text := summaryText(resp)
	if text == "" {
		return "", errors.New("summarizer returned an empty summary")
	}
	return text, nil
}

// summaryText is the summary a summarizer response carries.
func summaryText(resp llmkit.Response) string {
	return strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text))
}

func partialSummary(text string) string {
	return "<partial-summary>\n" + text + "\n</partial-summary>"
}

// splitPoint returns the index that splits entries into two non-empty halves
// of about equal size in bytes.
func splitPoint(entries []string) int {
	total := 0
	for _, e := range entries {
		total += len(e)
	}
	sum := 0
	for i, e := range entries {
		sum += len(e)
		if sum*2 >= total {
			return min(max(i+1, 1), len(entries)-1)
		}
	}
	return len(entries) / 2
}

// summarized builds the compaction of in that replaces in.Messages[lead:at]
// with the summary text, or leaves the history unchanged when that would not
// shrink it.
func summarized(in CompactInput, lead, at int, text string) CompactOutput {
	msg := llmkit.TextMessage(llmkit.RoleUser, conversationSummaryHeader+text)
	join := func(msgs []llmkit.Message) []llmkit.Message {
		out := make([]llmkit.Message, 0, lead+1+len(msgs)-at)
		out = append(out, msgs[:lead]...)
		out = append(out, msg)
		return append(out, msgs[at:]...)
	}
	if estimateTokens(join(in.Sent)) >= estimateTokens(in.Sent) {
		return CompactOutput{Messages: in.Messages}
	}
	return CompactOutput{Messages: join(in.Messages), Pruned: at - lead, Summary: text}
}

// summaryCut returns the number of leading system messages and the index of
// the user message the summary ends before, or -1 for no cut point.
func summaryCut(msgs []llmkit.Message) (lead, at int) {
	for lead < len(msgs) && msgs[lead].Role == llmkit.RoleSystem {
		lead++
	}
	lastAssistant := -1
	for i, m := range msgs {
		if m.Role == llmkit.RoleAssistant {
			lastAssistant = i
		}
	}
	for i := lastAssistant - 1; i > lead; i-- {
		if !isCallerUser(msgs[i]) || i-lead == 1 && isSummaryMessage(msgs[lead]) || splitsToolCall(msgs, i) {
			continue
		}
		return lead, i
	}
	return lead, -1
}

func isSummaryMessage(m llmkit.Message) bool {
	return m.Role == llmkit.RoleUser && strings.HasPrefix(m.Text(), conversationSummaryHeader)
}

var finalizationPromptHead = finalizationPrompt(nil)

// isCallerUser reports whether m is a user turn of the caller: not a summary
// and not a prompt the Runner injected.
func isCallerUser(m llmkit.Message) bool {
	if m.Role != llmkit.RoleUser || isSummaryMessage(m) {
		return false
	}
	text := m.Text()
	return text != emptyTurnNudge && text != maxTokensContinuePrompt && !strings.HasPrefix(text, finalizationPromptHead)
}

// splitsToolCall reports whether a tool result at or after cut answers a call
// issued before it.
func splitsToolCall(msgs []llmkit.Message, cut int) bool {
	for j := cut; j < len(msgs); j++ {
		if msgs[j].Role != llmkit.RoleToolResult {
			continue
		}
		for k := j - 1; k >= 0; k-- {
			if msgs[k].Role == llmkit.RoleAssistant && issuesCall(msgs[k], msgs[j].ToolCallID) {
				if k < cut {
					return true
				}
				break
			}
		}
	}
	return false
}

func issuesCall(m llmkit.Message, id string) bool {
	for _, c := range m.ToolCalls {
		if c.ID == id {
			return true
		}
	}
	return false
}

// serializePrefix renders msgs as the summarizer's input, one entry per message.
func serializePrefix(msgs []llmkit.Message) []string {
	names := map[string]string{}
	entries := make([]string, 0, len(msgs))
	for _, m := range msgs {
		for _, c := range m.ToolCalls {
			names[c.ID] = c.Name
		}
		entries = append(entries, serializeMessage(m, names))
	}
	return entries
}

func serializeMessage(m llmkit.Message, toolNames map[string]string) string {
	var b strings.Builder
	switch m.Role {
	case llmkit.RoleToolResult:
		kind := "tool result"
		if m.IsError {
			kind = "tool error"
		}
		fmt.Fprintf(&b, "[%s: %s]\n%s", kind, toolNames[m.ToolCallID], truncateChars(blocksText(m.Content), summaryToolResultChars))
		return b.String()
	case llmkit.RoleUser:
		if isSummaryMessage(m) {
			return "<previous-summary>\n" + strings.TrimPrefix(m.Text(), conversationSummaryHeader) + "\n</previous-summary>"
		}
	}
	fmt.Fprintf(&b, "[%s]\n%s", m.Role, blocksText(m.Content))
	for _, c := range m.ToolCalls {
		fmt.Fprintf(&b, "\n[tool call: %s] %s", c.Name, c.Arguments)
	}
	return b.String()
}

// truncateChars cuts s to n characters and says how many it dropped.
func truncateChars(s string, n int) string {
	total := utf8.RuneCountInString(s)
	if total <= n {
		return s
	}
	cut := 0
	for range n {
		_, w := utf8.DecodeRuneInString(s[cut:])
		cut += w
	}
	return fmt.Sprintf("%s\n[... %d more characters truncated]", s[:cut], total-n)
}

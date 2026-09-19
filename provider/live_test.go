//go:build live

// Live acceptance suite for the provider package.
//
// TestLiveMatrix runs the capability-keyed case registry (liveCases in
// live_registry_test.go) against EVERY lane that has credentials, building
// each client through provider.New — the production construction path, retry
// and recorder wrappers included. A gated case runs only where the effective
// capability profile claims support and MUST pass there; otherwise it skips,
// naming the capability. TestLiveTableTruth checks the configured model
// against the vendor's own models listing. Fixtures are recorded into
// provider/testdata/<lane>/<case>.json under -update and replayed hermetically
// by fixture_replay_test.go.
//
// Environment (see AGENTS.md):
//
//	LLMKIT_LIVE_COMPAT_API_KEY / _BASE_URL / _MODEL   (all required)
//	LLMKIT_LIVE_ANTHROPIC_API_KEY [+_MODEL]           (default claude-haiku-4-5)
//	LLMKIT_LIVE_OPENAI_API_KEY [+_MODEL]              (default gpt-4o-mini)
//	LLMKIT_LIVE_GOOGLE_API_KEY [+_MODEL]              (default gemini-2.5-flash-lite)
//	LLMKIT_LIVE_COMPAT_CAPS                           (optional asserted capabilities)
//
// A keyless lane skips at the lane level, naming its missing variables.
// Prompts stay tiny and MaxTokens modest: this costs real money. Run with
//
//	go test -tags live -count=1 ./provider/ -v
//
// and add -update to re-record fixtures. Keys are never logged; every raw
// body or URL log goes through livetest's redacting logger.
package provider_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/livetest"
	"github.com/dpoage/llmkit/provider"
)

func TestMain(m *testing.M) {
	code := m.Run()
	livetest.DefaultTally().PrintSummary()
	os.Exit(code)
}

// defaultLiveMaxTokens bounds every case completion that does not set its
// own cap: enough room for a reasoning model's think block, small enough to
// keep the whole run cheap.
const defaultLiveMaxTokens = 768

// badCredential is the intentionally wrong Secret for the ErrAuth case. It
// deliberately avoids sk- prefixes so it can never collide with the
// fixture-writer's secret assertions.
const badCredential = "llmkit-live-intentionally-invalid-key"

// badModel is the intentionally unknown model for the ErrInvalidRequest case.
const badModel = "llmkit-no-such-model-x9"

// liveClient is the per-case handle handed to every case body: a fresh
// recording transport, a provider.New client, and the case's request log.
type liveClient struct {
	t        *testing.T
	sess     *livetest.Session
	tr       *livetest.Transport
	cl       llmkit.Client
	ctx      context.Context
	caseName string
	model    string
	reqs     []llmkit.Request
}

func newLiveClient(t *testing.T, sess *livetest.Session) *liveClient {
	ctx, _ := livetest.Ctx(t)
	lc := &liveClient{t: t, sess: sess, tr: livetest.NewTransport(sess.Key), ctx: ctx, model: sess.Model}
	lc.cl = sess.Client(ctx, t, lc.tr, nil)
	return lc
}

// variant returns a sibling client built from the same session with a
// mutated Spec (its own transport and request log), for the
// error-normalization cases.
func (lc *liveClient) variant(mutate func(*provider.Spec)) *liveClient {
	ctx, _ := livetest.Ctx(lc.t)
	v := &liveClient{t: lc.t, sess: lc.sess, tr: livetest.NewTransport(lc.sess.Key), ctx: ctx, caseName: lc.caseName}
	v.cl = lc.sess.Client(ctx, lc.t, v.tr, func(s *provider.Spec) {
		mutate(s)
		v.model = s.Model
	})
	return v
}

func (lc *liveClient) caps() llmkit.Capabilities { return lc.cl.Capabilities() }

// complete issues req, logs it for the fixture, and returns the outcome.
func (lc *liveClient) complete(req llmkit.Request) (llmkit.Response, error) {
	if req.MaxTokens == 0 {
		req.MaxTokens = defaultLiveMaxTokens
	}
	lc.reqs = append(lc.reqs, req)
	return lc.cl.Complete(lc.ctx, req)
}

// completeVia is complete through a caller-wrapped client (e.g. an extra
// WithRecorder layer) while still logging the request for the fixture.
func (lc *liveClient) completeVia(cl llmkit.Client, req llmkit.Request) (llmkit.Response, error) {
	if req.MaxTokens == 0 {
		req.MaxTokens = defaultLiveMaxTokens
	}
	lc.reqs = append(lc.reqs, req)
	return cl.Complete(lc.ctx, req)
}

// finish records the case fixture when -update is set: the logged inputs,
// the sanitized wire exchanges, and the final normalized outcome.
func (lc *liveClient) finish(resp llmkit.Response, err error) {
	if !livetest.Update() {
		return
	}
	f := &livetest.Fixture{
		Case:         lc.caseName,
		Lane:         lc.sess.Lane.Name,
		Model:        lc.model,
		Capabilities: lc.caps(),
		Inputs:       lc.reqs,
		Exchanges:    lc.tr.Exchanges(),
		Normalized: livetest.Normalized{
			ErrKind:    livetest.ErrKindOf(err),
			Text:       resp.Text,
			ToolCalls:  resp.ToolCalls,
			StopReason: resp.StopReason,
			Usage:      resp.Usage,
		},
	}
	if len(f.Inputs) != len(f.Exchanges) {
		lc.t.Fatalf("case %s: %d logged requests but %d wire exchanges; refusing to write a misaligned fixture",
			lc.caseName, len(f.Inputs), len(f.Exchanges))
	}
	livetest.WriteFixture(lc.t, filepath.Join("testdata", lc.sess.Lane.Name, lc.caseName+".json"), f, lc.sess.Key)
}

// redFatal fails the test with a message run through the session's
// redactor: lane URLs carry the credential on some vendors (google's ?key=).
func redFatal(t *testing.T, sess *livetest.Session, format string, args ...any) {
	t.Fatal(livetest.Redact(sess.Key, fmt.Sprintf(format, args...)))
}

// gateClaimed reports whether the effective profile claims the capability
// named by gate (an llmkit.Capabilities field name).
func gateClaimed(c llmkit.Capabilities, gate string) bool {
	v := reflect.ValueOf(c).FieldByName(gate)
	if !v.IsValid() {
		return false
	}
	switch v.Kind() {
	case reflect.Bool:
		return v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() > 0
	default:
		return false
	}
}

// TestLiveCaseParity is the tagged half of the registry contract: every
// descriptor has a body AND every body has a descriptor. A case name without
// a body must never pass silently.
func TestLiveCaseParity(t *testing.T) {
	descriptors := make(map[string]bool, len(liveCases))
	for _, c := range liveCases {
		descriptors[c.Name] = true
	}
	for name := range liveCaseBodies {
		if !descriptors[name] {
			t.Errorf("liveCaseBodies[%q] has no descriptor in liveCases", name)
		}
	}
	for _, c := range liveCases {
		if _, ok := liveCaseBodies[c.Name]; !ok {
			t.Errorf("live case %q has a descriptor but no body in liveCaseBodies", c.Name)
		}
	}
}

// TestLiveMatrix runs the whole case registry for every lane. A keyless lane
// skips once at the lane level; each gated case skips naming the capability
// it requires.
func TestLiveMatrix(t *testing.T) {
	for _, lane := range livetest.All() {
		t.Run(lane.Name, func(t *testing.T) {
			sess := livetest.Resolve(t, lane.Name)
			for _, c := range liveCases {
				c := c
				t.Run(c.Name, func(t *testing.T) {
					body, ok := liveCaseBodies[c.Name]
					if !ok {
						t.Fatalf("live case %q has a descriptor but no tagged body — add liveCaseBodies[%q]", c.Name, c.Name)
					}
					lc := newLiveClient(t, sess)
					lc.caseName = c.Name
					if c.Gate != "" && !gateClaimed(lc.caps(), c.Gate) {
						t.Skipf("capability %s not claimed by %s lane", c.Gate, sess.Lane.Name)
					}
					body(t, lc)
				})
			}
		})
	}
}

// --- case bodies ------------------------------------------------------------

var liveCaseBodies = map[string]func(t *testing.T, lc *liveClient){
	"text_usage":          caseTextUsage,
	"tool_round_trip":     caseToolRoundTrip,
	"max_tokens":          caseMaxTokens,
	"error_bad_key":       caseErrorBadKey,
	"error_bad_model":     caseErrorBadModel,
	"usage_recorded":      caseUsageRecorded,
	"context_window":      caseContextWindow,
	"parallel_tool_calls": caseParallelToolCalls,
	"prompt_caching":      casePromptCaching,
	"structured_output":   caseStructuredOutput,
	"thinking":            caseThinking,
	"tool_choice":         caseToolChoice,
	"images":              caseImages,
	"documents":           caseDocuments,
	"stop_sequences":      caseStopSequences,
	"top_p":               caseTopP,
	"top_k":               caseTopK,
	"seed":                caseSeed,
}

func caseTextUsage(t *testing.T, lc *liveClient) {
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		t.Fatalf("empty answer text: %q", resp.Text)
	}
	if resp.Usage.InputTokens <= 0 || resp.Usage.OutputTokens <= 0 {
		t.Fatalf("usage not accounted: %+v", resp.Usage)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("stop reason = %q, want %q", resp.StopReason, llmkit.StopEndTurn)
	}
	lc.finish(resp, err)
}

func caseToolRoundTrip(t *testing.T, lc *liveClient) {
	readFile := llmkit.ToolDef{
		Name:        "read_file",
		Description: "Reads a text file and returns its contents.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"file to read"}},"required":["path"]}`),
	}
	req := llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser,
			"Read the file poetry.txt using the read_file tool, then quote its second line verbatim as your final answer.")},
		Tools: []llmkit.ToolDef{readFile},
	}
	resp, err := lc.complete(req)
	if err != nil {
		t.Fatalf("tool request: %v", err)
	}
	if resp.StopReason != llmkit.StopToolUse || len(resp.ToolCalls) == 0 {
		t.Fatalf("stop = %q with %d tool call(s); want a tool request", resp.StopReason, len(resp.ToolCalls))
	}
	call := resp.ToolCalls[0]
	if call.Name != "read_file" {
		t.Fatalf("tool call %q, want read_file", call.Name)
	}
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		t.Fatalf("tool arguments %s do not parse: %v", call.Arguments, err)
	}
	if args.Path == "" {
		t.Fatalf("tool arguments %s carry no path", call.Arguments)
	}

	req.Messages = append(req.Messages,
		llmkit.Message{Role: llmkit.RoleAssistant, ToolCalls: resp.ToolCalls},
		llmkit.Message{
			Role:       llmkit.RoleToolResult,
			ToolCallID: call.ID,
			Content: []llmkit.Block{{Kind: llmkit.BlockText,
				Text: "line one\nshall I compare thee to a summer's day\nline three"}},
		},
	)
	final, err := lc.complete(req)
	if err != nil {
		t.Fatalf("final answer after tool result: %v", err)
	}
	if !strings.Contains(strings.ToLower(final.Text), "summer") {
		t.Fatalf("final answer %q does not quote the file's second line", final.Text)
	}
	lc.finish(final, err)
}

func caseMaxTokens(t *testing.T, lc *liveClient) {
	resp, err := lc.complete(llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Count from 1 to 100, one number per line. Do not skip any.")},
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.StopReason != llmkit.StopMaxTokens {
		t.Fatalf("stop reason = %q, want %q (MaxTokens 16 must truncate)", resp.StopReason, llmkit.StopMaxTokens)
	}
	lc.finish(resp, err)
}

func caseErrorBadKey(t *testing.T, lc *liveClient) {
	bad := lc.variant(func(s *provider.Spec) { s.Secret = badCredential })
	_, err := bad.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
	})
	if !errors.Is(err, llmkit.ErrAuth) {
		t.Fatalf("error = %v, want ErrAuth", err)
	}
	bad.finish(llmkit.Response{}, err)
}

func caseErrorBadModel(t *testing.T, lc *liveClient) {
	bad := lc.variant(func(s *provider.Spec) { s.Model = badModel })
	_, err := bad.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
	})
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("error = %v, want ErrInvalidRequest", err)
	}
	bad.finish(llmkit.Response{}, err)
}

type captureRecorder struct {
	mu     sync.Mutex
	events []llmkit.UsageEvent
}

func (c *captureRecorder) Record(ev llmkit.UsageEvent) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
}

func (c *captureRecorder) snapshot() []llmkit.UsageEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]llmkit.UsageEvent(nil), c.events...)
}

func caseUsageRecorded(t *testing.T, lc *liveClient) {
	rec := &captureRecorder{}
	wrapped := llmkit.WithRecorder(lc.cl, rec, "live-"+lc.sess.Lane.Name, lc.model)
	resp, err := lc.completeVia(wrapped, llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	events := rec.snapshot()
	if len(events) != 1 {
		t.Fatalf("WithRecorder emitted %d event(s), want 1", len(events))
	}
	if ev := events[0]; ev.Model != lc.model || ev.Usage.InputTokens <= 0 || ev.Usage.OutputTokens <= 0 {
		t.Fatalf("usage event = %+v, want this lane's model with accounted tokens", ev)
	}
	lc.finish(resp, err)
}

func caseContextWindow(t *testing.T, lc *liveClient) {
	window := lc.caps().ContextWindow
	if window <= 0 {
		t.Skipf("capability ContextWindow not claimed by %s lane", lc.sess.Lane.Name)
	}
	// An input larger than the reported window must be rejected by the
	// vendor and normalize to ErrContextTooLong. Each prompt word is at
	// least one token, so window+2048 repeats exceed the window under any
	// tokenizer; this is deliberately the case that fails when an adapter
	// table overstates or understates a window.
	prompt := strings.Repeat("llmkit ", window+2048)
	_, err := lc.complete(llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, prompt)},
		MaxTokens: 16,
	})
	if !errors.Is(err, llmkit.ErrContextTooLong) {
		t.Fatalf("oversize prompt (window %d): error = %v, want ErrContextTooLong", window, err)
	}
	lc.finish(llmkit.Response{}, err)
}

func caseParallelToolCalls(t *testing.T, lc *liveClient) {
	now := llmkit.ToolDef{
		Name: "now", Description: "Returns the current local date and time.",
		Parameters: json.RawMessage(`{"type":"object","properties":{}}`),
	}
	add := llmkit.ToolDef{
		Name: "add", Description: "Adds two numbers.",
		Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}`),
	}
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser,
			"Call BOTH the now tool and the add tool (a=41, b=58) in the SAME response, in parallel. Do not answer in words.")},
		Tools: []llmkit.ToolDef{now, add},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if len(resp.ToolCalls) < 2 {
		t.Fatalf("got %d tool call(s) in one response, want >= 2 (parallel tool calls)", len(resp.ToolCalls))
	}
	seen := make(map[string]bool, len(resp.ToolCalls))
	for _, c := range resp.ToolCalls {
		seen[c.Name] = true
	}
	if !seen["now"] || !seen["add"] {
		t.Fatalf("tool calls requested %v, want both now and add", seen)
	}
	lc.finish(resp, err)
}

func casePromptCaching(t *testing.T, lc *liveClient) {
	// A shared system prefix comfortably above the ~1024-token minimum.
	prefix := strings.Repeat(
		"The quick brown fox jumps over the lazy dog beside the riverbank at dusk. ", 120)
	req := llmkit.Request{System: prefix}
	req.Messages = []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Which animal is mentioned? One word.")}
	first, err := lc.complete(req)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	req.Messages = []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Which place is mentioned? One word.")}
	second, err := lc.complete(req)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if second.Usage.CacheReadInputTokens <= 0 {
		t.Fatalf("no cache read on the repeated prefix: first usage %+v, second usage %+v", first.Usage, second.Usage)
	}
	lc.finish(second, err)
}

func caseStructuredOutput(t *testing.T, lc *liveClient) {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {"answer": {"type": "string"}},
		"required": ["answer"],
		"additionalProperties": false
	}`)
	resp, err := lc.complete(llmkit.Request{
		Messages:           []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "What is the capital of France? Fill the schema.")},
		ResponseSchema:     schema,
		ResponseSchemaName: "capital_answer",
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	var out struct {
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal([]byte(llmkit.StripThinkBlocks(resp.Text)), &out); err != nil {
		t.Fatalf("output does not parse under the schema: %v\ntext: %s", err, resp.Text)
	}
	if !strings.Contains(strings.ToLower(out.Answer), "paris") {
		t.Fatalf("answer = %q, want Paris", out.Answer)
	}
	lc.finish(resp, err)
}

func caseThinking(t *testing.T, lc *liveClient) {
	req := llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "What is 17 times 24? Think it through, then answer.")},
		Thinking:  &llmkit.ThinkingConfig{BudgetTokens: 1024},
		MaxTokens: 2048,
	}
	resp, err := lc.complete(req)
	if err != nil {
		t.Fatalf("thinking completion: %v", err)
	}
	hasThink := false
	for _, b := range resp.Blocks {
		if b.Kind == llmkit.BlockThinking {
			hasThink = true
			break
		}
	}
	if !hasThink {
		t.Fatalf("no BlockThinking block in the response (Thinking capability claimed): %+v", resp.Blocks)
	}
	// The thinking turn must re-send cleanly on the next request.
	req.Messages = append(req.Messages,
		llmkit.Message{Role: llmkit.RoleAssistant, Content: resp.Blocks},
		llmkit.TextMessage(llmkit.RoleUser, "Now add ten."),
	)
	second, err := lc.complete(req)
	if err != nil {
		t.Fatalf("re-sending the thinking turn failed: %v", err)
	}
	lc.finish(second, err)
}

func caseToolChoice(t *testing.T, lc *liveClient) {
	add := llmkit.ToolDef{
		Name: "add", Description: "Adds two numbers.",
		Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}`),
	}
	base := llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Call the add tool with a=2 and b=3.")},
		Tools:    []llmkit.ToolDef{add},
	}

	// none forbids tool use.
	forbid := base
	forbid.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceNone}
	resp, err := lc.complete(forbid)
	if err != nil {
		t.Fatalf("tool_choice none: %v", err)
	}
	if len(resp.ToolCalls) != 0 {
		t.Fatalf("tool_choice none produced %d tool call(s), want 0", len(resp.ToolCalls))
	}

	// required forces SOME tool call.
	forced := base
	forced.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceRequired}
	resp, err = lc.complete(forced)
	if err != nil {
		t.Fatalf("tool_choice required: %v", err)
	}
	if len(resp.ToolCalls) == 0 {
		t.Fatal("tool_choice required produced no tool call")
	}

	// named forces THAT name.
	named := base
	named.ToolChoice = llmkit.ToolChoice{Mode: llmkit.ToolChoiceTool, Name: "add"}
	resp, err = lc.complete(named)
	if err != nil {
		t.Fatalf("tool_choice tool: %v", err)
	}
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "add" {
		t.Fatalf("tool_choice tool(add) produced %+v, want a call named add", resp.ToolCalls)
	}
	lc.finish(resp, err)
}

func caseImages(t *testing.T, lc *liveClient) {
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{{
			Role: llmkit.RoleUser,
			Content: []llmkit.Block{
				{Kind: llmkit.BlockImage, MediaType: "image/png", Data: tinyPNG()},
				{Kind: llmkit.BlockText, Text: "Reply with the single word OK if you received an image."},
			},
		}},
	})
	if err != nil {
		t.Fatalf("image completion: %v", err)
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		t.Fatalf("empty answer for image input: %q", resp.Text)
	}
	lc.finish(resp, err)
}

func caseDocuments(t *testing.T, lc *liveClient) {
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{{
			Role: llmkit.RoleUser,
			Content: []llmkit.Block{
				{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Title: "hello.pdf", Data: tinyPDF()},
				{Kind: llmkit.BlockText, Text: "Reply with the single word OK if you received a document."},
			},
		}},
	})
	if err != nil {
		t.Fatalf("document completion: %v", err)
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		t.Fatalf("empty answer for document input: %q", resp.Text)
	}
	lc.finish(resp, err)
}

func caseStopSequences(t *testing.T, lc *liveClient) {
	// The guard string is one the model was told to emit, so the ONLY way
	// the guarded word CHERRY can appear is generation running past the
	// stop sequence — including through a reasoning model's think block,
	// which quotes the guard before reaching CHERRY and therefore also
	// stops. StopEndTurn is the normalized reason for a sequence hit.
	const guard, tail = "XQ-END-77", "CHERRY"
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser,
			"Write the word BANANA. On a new line write exactly "+guard+". On a new line after that write the word "+tail+".")},
		StopSequences: []string{guard},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	// Record the exchange BEFORE the skip below: the single-exchange
	// fixture is strict-mode, so deleting the stop serialization from an
	// adapter makes hermetic replay fail on the request side even though
	// this vendor never honors the parameter.
	lc.finish(resp, err)
	if strings.Contains(resp.Text, tail) {
		// The stop sequence is SERVER-enforced: if the guard string and
		// anything after it are both in the output, the endpoint ignored
		// `stop` outright (verified against MiniMax-M3 at the raw wire —
		// the parameter is in their schema but not enforced). That is a
		// measured vendor fact, not a case bug: skip naming the capability.
		t.Skipf("capability StopSequences claimed but the %s endpoint ignored the stop sequence: output ran past %q",
			lc.sess.Lane.Name, guard)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("stop reason = %q, want %q", resp.StopReason, llmkit.StopEndTurn)
	}
}

func caseTopP(t *testing.T, lc *liveClient) {
	topP := 0.5
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
		TopP:     &topP,
	})
	if err != nil {
		t.Fatalf("TopP request rejected: %v", err)
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		t.Fatalf("empty answer: %q", resp.Text)
	}
	lc.finish(resp, err)
}

func caseTopK(t *testing.T, lc *liveClient) {
	topK := 40
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
		TopK:     &topK,
	})
	if err != nil {
		t.Fatalf("TopK request rejected: %v", err)
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		t.Fatalf("empty answer: %q", resp.Text)
	}
	lc.finish(resp, err)
}

func caseSeed(t *testing.T, lc *liveClient) {
	seed := int64(42)
	resp, err := lc.complete(llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
		Seed:     &seed,
	})
	if err != nil {
		t.Fatalf("Seed request rejected: %v", err)
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		t.Fatalf("empty answer: %q", resp.Text)
	}
	lc.finish(resp, err)
}

// --- table truth -------------------------------------------------------------

// TestLiveTableTruth checks the live vendor against our own tables: for each
// lane with credentials, the vendor's models listing must contain the
// configured model; on google the listing's inputTokenLimit must equal the
// adapter table's ContextWindow for every listed model the table knows.
func TestLiveTableTruth(t *testing.T) {
	for _, lane := range livetest.All() {
		t.Run(lane.Name, func(t *testing.T) {
			sess := livetest.Resolve(t, lane.Name)
			ctx, _ := livetest.Ctx(t)
			hc := &http.Client{Transport: livetest.NewTransport(sess.Key), Timeout: 30 * time.Second}

			url, headers := modelsRequest(sess)
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			if err != nil {
				redFatal(t, sess, "models request: %v", err)
			}
			for k, v := range headers {
				req.Header.Set(k, v)
			}
			resp, err := hc.Do(req)
			if err != nil {
				redFatal(t, sess, "models list GET: %v", err)
			}
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				redFatal(t, sess, "models list body: %v", err)
			}
			sess.Logf(t, "models endpoint returned status %d", resp.StatusCode)

			if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
				t.Skipf("models endpoint returned %d; skipping table truth for the %s lane", resp.StatusCode, sess.Lane.Name)
			}
			if resp.StatusCode != http.StatusOK {
				redFatal(t, sess, "models endpoint returned %d: %s", resp.StatusCode, livetest.Redact(sess.Key, string(body)))
			}

			var list struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
				Models []struct {
					Name            string `json:"name"`
					InputTokenLimit int    `json:"inputTokenLimit"`
				} `json:"models"`
			}
			if err := json.Unmarshal(body, &list); err != nil {
				if sess.Lane.Name == "compat" {
					t.Skipf("models endpoint returned non-JSON (status %d); skipping table truth for the compat lane", resp.StatusCode)
				}
				redFatal(t, sess, "models list did not parse as JSON: %v; body: %s", err, livetest.Redact(sess.Key, string(body)))
			}

			switch sess.Lane.Name {
			case "google":
				found := false
				for _, m := range list.Models {
					name := strings.TrimPrefix(m.Name, "models/")
					if name == sess.Model {
						found = true
					}
					window := capsFor(t, sess, name).ContextWindow
					if window == 0 {
						continue // not a table key; nothing to assert
					}
					if m.InputTokenLimit != window {
						t.Errorf("model %s: adapter table ContextWindow %d != vendor inputTokenLimit %d", name, window, m.InputTokenLimit)
					}
				}
				if !found {
					t.Errorf("configured model %q is not in the vendor listing", sess.Model)
				}
			default:
				found := false
				for _, m := range list.Data {
					if m.ID == sess.Model {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("configured model %q is not in the vendor listing", sess.Model)
				}
			}
		})
	}
}

// modelsRequest builds the vendor's models-listing request for a session.
// The URL is credential-carrying on google (?key=); anything that logs it
// must go through the redactor.
func modelsRequest(sess *livetest.Session) (string, map[string]string) {
	switch sess.Lane.Name {
	case "compat":
		return strings.TrimRight(sess.BaseURL, "/") + "/models",
			map[string]string{"Authorization": "Bearer " + sess.Key}
	case "openai":
		return "https://api.openai.com/v1/models",
			map[string]string{"Authorization": "Bearer " + sess.Key}
	case "anthropic":
		return "https://api.anthropic.com/v1/models",
			map[string]string{"x-api-key": sess.Key, "anthropic-version": "2023-06-01"}
	case "google":
		return "https://generativelanguage.googleapis.com/v1beta/models?key=" + sess.Key + "&pageSize=1000", nil
	default:
		return "", nil
	}
}

// capsFor builds a throwaway client for model through provider.New and
// reports the adapter table's profile for it.
func capsFor(t *testing.T, sess *livetest.Session, model string) llmkit.Capabilities {
	t.Helper()
	cl, err := provider.New(context.Background(), provider.Spec{
		Type:   sess.Lane.Type,
		Model:  model,
		Secret: sess.Key,
	}, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New(%q): %v", model, err)
	}
	return cl.Capabilities()
}

// --- fixture media -----------------------------------------------------------

// tinyPNG returns a valid 1x1 PNG's bytes.
func tinyPNG() []byte {
	const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
	b, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		panic("embedded PNG constant is invalid: " + err.Error())
	}
	return b
}

// tinyPDF builds a minimal, valid one-page PDF whose text reads "hello
// llmkit". The xref offsets are computed, so the bytes are deterministic and
// fixture-stable.
func tinyPDF() []byte {
	const stream = "BT /F1 24 Tf 72 720 Td (hello llmkit) Tj ET\n"
	objects := []string{
		"<</Type/Catalog/Pages 2 0 R>>",
		"<</Type/Pages/Kids[3 0 R]/Count 1>>",
		"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
		fmt.Sprintf("<</Length %d>>stream\n%sendstream", len(stream), stream),
		"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
	}
	var b bytes.Buffer
	b.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objects))
	for i, o := range objects {
		offsets[i] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj%s endobj\n", i+1, o)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n", len(objects)+1)
	b.WriteString("0000000000 65535 f \n")
	for _, off := range offsets {
		fmt.Fprintf(&b, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&b, "trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	return b.Bytes()
}

// TestLiveGoogleStream runs the google lane's streaming path against the
// live API through the same production construction as the matrix: llmkit.Stream
// must deliver at least one delta over the SSE endpoint, and the aggregated
// Response must carry text, accounted usage, and StopEndTurn — the same
// outcome the lane's text_usage case asserts for Complete. Skips without the
// google-lane credentials, like every matrix case.
func TestLiveGoogleStream(t *testing.T) {
	sess := livetest.Resolve(t, "google")
	lc := newLiveClient(t, sess)
	lc.caseName = "stream_text"

	var deltas int
	resp, err := llmkit.Stream(lc.ctx, lc.cl, llmkit.Request{
		Messages:  []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "Reply with exactly: OK")},
		MaxTokens: defaultLiveMaxTokens,
	}, func(llmkit.Delta) error {
		deltas++
		return nil
	})
	if err != nil {
		redFatal(t, sess, "stream: %v", err)
	}
	if deltas == 0 {
		redFatal(t, sess, "stream delivered no deltas")
	}
	if strings.TrimSpace(llmkit.StripThinkBlocks(resp.Text)) == "" {
		redFatal(t, sess, "empty streamed text: %q", resp.Text)
	}
	if resp.Usage.InputTokens <= 0 || resp.Usage.OutputTokens <= 0 {
		redFatal(t, sess, "streamed usage not accounted: %+v", resp.Usage)
	}
	if resp.StopReason != llmkit.StopEndTurn {
		redFatal(t, sess, "stop reason = %q, want %q", resp.StopReason, llmkit.StopEndTurn)
	}
	lc.finish(resp, err)
}

package sqlite_test

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/store/sqlite"
)

// errLog collects onErr calls.
type errLog struct {
	mu   sync.Mutex
	errs []error
}

func (l *errLog) add(err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.errs = append(l.errs, err)
}

func (l *errLog) all() []error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]error(nil), l.errs...)
}

func openStore(t *testing.T) (*sqlite.Store, *errLog, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "events.db")
	log := &errLog{}
	s, err := sqlite.Open(path, log.add)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, log, path
}

func noErrors(t testing.TB, log *errLog) {
	t.Helper()
	if errs := log.all(); len(errs) != 0 {
		t.Fatalf("onErr called %d times, first: %v", len(errs), errs[0])
	}
}

// canon is json.Marshal of the event after one JSON round trip.
func canon(t testing.TB, ev llmkit.Event) string {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal %s event: %v", ev.Kind, err)
	}
	var v llmkit.Event
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("unmarshal %s event: %v", ev.Kind, err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("re-marshal %s event: %v", ev.Kind, err)
	}
	return string(out)
}

// assertIdentical fails unless got has want's length and each event is
// json-identical to canon of the observed one.
func assertIdentical(t testing.TB, got, want []llmkit.Event) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("Events returned %d events, observed %d", len(got), len(want))
	}
	for i := range want {
		g, err := json.Marshal(got[i])
		if err != nil {
			t.Fatalf("event %d: marshal read event: %v", i, err)
		}
		if w := canon(t, want[i]); string(g) != w {
			t.Fatalf("event %d (%s) differs\n got: %s\nwant: %s", i, want[i].Kind, clip(string(g)), clip(w))
		}
	}
}

func clip(s string) string {
	if len(s) > 600 {
		return s[:600] + fmt.Sprintf("...(%d bytes)", len(s))
	}
	return s
}

// funcClient is a stateless scripted client: its reply depends only on the request.
type funcClient func(req llmkit.Request) (llmkit.Response, error)

func (f funcClient) Complete(_ context.Context, req llmkit.Request) (llmkit.Response, error) {
	return f(req)
}
func (funcClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

type fnTool struct {
	name string
	run  func(ctx context.Context, args json.RawMessage) ([]llmkit.Block, error)
}

func (t fnTool) Def() llmkit.ToolDef {
	return llmkit.ToolDef{Name: t.name, Description: "test tool", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t fnTool) Run(ctx context.Context, args json.RawMessage) ([]llmkit.Block, error) {
	return t.run(ctx, args)
}

// toolResults counts the tool-result messages in a request: the turns done.
func toolResults(req llmkit.Request) int {
	n := 0
	for _, m := range req.Messages {
		if m.Role == llmkit.RoleToolResult {
			n++
		}
	}
	return n
}

// loopClient asks for one echo call per turn for steps turns, then answers.
// Every reply and every call carries a marker unique to its turn.
func loopClient(steps int) funcClient {
	return func(req llmkit.Request) (llmkit.Response, error) {
		n := toolResults(req)
		usage := llmkit.Usage{InputTokens: int64(100 + n), OutputTokens: 7}
		if n >= steps {
			return llmkit.Response{Text: "final answer", StopReason: llmkit.StopEndTurn, Usage: usage}, nil
		}
		return llmkit.Response{
			Text:       fmt.Sprintf("assistant-%03d-mark", n),
			StopReason: llmkit.StopToolUse,
			ToolCalls:  []llmkit.ToolCall{{ID: fmt.Sprintf("call-%03d", n), Name: "echo", Arguments: json.RawMessage(fmt.Sprintf(`{"n":%d}`, n))}},
			Usage:      usage,
		}, nil
	}
}

// markerTool returns a result carrying a marker unique to its argument.
func markerTool() fnTool {
	return fnTool{name: "echo", run: func(_ context.Context, args json.RawMessage) ([]llmkit.Block, error) {
		var a struct{ N int }
		_ = json.Unmarshal(args, &a)
		return []llmkit.Block{llmkit.Text(fmt.Sprintf("result-%03d-mark", a.N))}, nil
	}}
}

// run executes one Runner.Run and returns the events the Runner observed.
func run(t testing.TB, r *agent.Runner, task string, opts ...agent.RunOption) []llmkit.Event {
	t.Helper()
	out, err := r.Run(context.Background(), task, opts...)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return out.Transcript.Record
}

func contains(s string, subs ...string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}

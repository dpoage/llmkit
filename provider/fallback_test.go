package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

var (
	_ llmkit.StreamingClient  = (*fallbackClient)(nil)
	_ llmkit.IdentifiedClient = (*fallbackClient)(nil)
)

type fbFake struct {
	id       llmkit.Identity
	caps     llmkit.Capabilities
	complete func(context.Context, llmkit.Request) (llmkit.Response, error)

	calls atomic.Int32
	mu    sync.Mutex
	reqs  []llmkit.Request
}

func (f *fbFake) Identity() llmkit.Identity         { return f.id }
func (f *fbFake) Capabilities() llmkit.Capabilities { return f.caps }
func (f *fbFake) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	f.calls.Add(1)
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	return f.complete(ctx, req)
}

type fbStreamFake struct {
	*fbFake
	stream func(context.Context, llmkit.Request, func(llmkit.Delta) error) (llmkit.Response, error)
}

func (f *fbStreamFake) Stream(ctx context.Context, req llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
	f.calls.Add(1)
	return f.stream(ctx, req, fn)
}

func fbID(model string) llmkit.Identity { return llmkit.Identity{Provider: "fake", Model: model} }

func fbFails(model string, err error) *fbFake {
	return &fbFake{id: fbID(model), complete: func(context.Context, llmkit.Request) (llmkit.Response, error) {
		return llmkit.Response{}, err
	}}
}

func fbServes(model, text string) *fbFake {
	return &fbFake{id: fbID(model), complete: func(context.Context, llmkit.Request) (llmkit.Response, error) {
		return llmkit.Response{Text: text}, nil
	}}
}

func fbOverloaded() error {
	return &llmkit.APIError{Kind: llmkit.ErrOverloaded, StatusCode: 529, Provider: "fake", Message: "primary overloaded"}
}

func fbReq() llmkit.Request {
	return llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}, MaxTokens: 64}
}

var fbNoSleep = func(context.Context, time.Duration) error { return nil }

type fbServer struct {
	url  string
	hits atomic.Int32
	mu   sync.Mutex
	body string
}

func (s *fbServer) lastBody() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.body
}

func fbHTTP(t *testing.T, status int, reply string) *fbServer {
	t.Helper()
	s := &fbServer{}
	s.url = newServer(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.body = string(b)
		s.mu.Unlock()
		s.hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	})
	return s
}

const (
	fbAnthropicOverloaded = `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`
	fbAnthropicThinking   = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"thinking","thinking":"plan it","signature":"sigA"},{"type":"tool_use","id":"toolu_01","name":"read","input":{"p":"a"}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":5}}`
	fbGoogleThinking      = `{"candidates":[{"content":{"role":"model","parts":[{"text":"g thoughts","thought":true,"thoughtSignature":"c2lnRw=="},{"functionCall":{"id":"call_g1","name":"read","args":{"p":"a"}},"thoughtSignature":"c2lnRkM="}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":5,"totalTokenCount":10}}`
)

func fbStack(t *testing.T, typ Type, model, url string, rc retry.Config) llmkit.Client {
	t.Helper()
	c, err := New(context.Background(), Spec{Type: typ, Model: model, BaseURL: url, Auth: AuthAPIKey, Secret: "k"}, Options{Retry: rc})
	if err != nil {
		t.Fatalf("New %s: %v", typ, err)
	}
	return c
}

func fbTools() []llmkit.ToolDef {
	return []llmkit.ToolDef{{Name: "read", Parameters: json.RawMessage(`{"type":"object","properties":{"p":{"type":"string"}}}`)}}
}

func fbHistory(t *testing.T, typ Type, model, reply string) []llmkit.Message {
	t.Helper()
	srv := fbHTTP(t, 200, reply)
	c := fbStack(t, typ, model, srv.url, retry.Config{MaxAttempts: 1})
	resp, err := c.Complete(context.Background(), llmkit.Request{
		Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "go")}, Tools: fbTools(), MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("history %s: %v", typ, err)
	}
	if len(resp.ToolCalls) != 1 || len(resp.Blocks) == 0 || resp.Blocks[0].Kind != llmkit.BlockThinking {
		t.Fatalf("history %s: want thinking + one tool call, got %+v", typ, resp)
	}
	return []llmkit.Message{
		llmkit.TextMessage(llmkit.RoleUser, "go"),
		resp.Message(),
		llmkit.ToolResult(resp.ToolCalls[0].ID, "contents"),
	}
}

func TestFallback_F1_PrimaryOverloadedSecondaryServes(t *testing.T) {
	primarySrv := fbHTTP(t, 529, fbAnthropicOverloaded)
	secondarySrv := fbHTTP(t, 200, mockTextBody("openai-compatible", "from secondary", 3, 4))
	primary := fbStack(t, TypeAnthropic, "claude-sonnet-4-5", primarySrv.url, retry.Config{MaxAttempts: 2, Sleep: fbNoSleep})
	secondary := fbStack(t, TypeOpenAICompatible, "sec-model", secondarySrv.url, retry.Config{MaxAttempts: 2, Sleep: fbNoSleep})

	resp, err := Fallback(primary, secondary).Complete(context.Background(), fbReq())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "from secondary" {
		t.Errorf("Text = %q, want the secondary's", resp.Text)
	}
	if got := primarySrv.hits.Load(); got != 2 {
		t.Errorf("primary server calls = %d, want 2 (MaxAttempts)", got)
	}
	if got := secondarySrv.hits.Load(); got != 1 {
		t.Errorf("secondary server calls = %d, want 1", got)
	}
	want := llmkit.IdentityOf(secondary)
	if want == (llmkit.Identity{}) || want.Model != "sec-model" {
		t.Fatalf("secondary identity = %+v, test setup is not distinguishing", want)
	}
	if resp.Served != want {
		t.Errorf("Served = %+v, want %+v", resp.Served, want)
	}
}

func TestFallback_F2_TerminalErrorsDoNotFallThrough(t *testing.T) {
	plain := errors.New("plain failure")
	deadCtx := func(cancel bool) func() (context.Context, context.CancelFunc) {
		return func() (context.Context, context.CancelFunc) {
			if cancel {
				ctx, c := context.WithCancel(context.Background())
				c()
				return ctx, c
			}
			return context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		}
	}
	live := func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) }

	rows := []struct {
		name    string
		ctx     func() (context.Context, context.CancelFunc)
		primary func(t *testing.T) llmkit.Client
		want    error
	}{
		{"auth 401", live, func(t *testing.T) llmkit.Client {
			return fbStack(t, TypeAnthropic, "m", fbHTTP(t, 401, errorBody("anthropic", 401, "bad key")).url, retry.Config{MaxAttempts: 3, Sleep: fbNoSleep})
		}, llmkit.ErrAuth},
		{"invalid request 400", live, func(t *testing.T) llmkit.Client {
			return fbStack(t, TypeAnthropic, "m", fbHTTP(t, 400, errorBody("anthropic", 400, "bad shape")).url, retry.Config{MaxAttempts: 3, Sleep: fbNoSleep})
		}, llmkit.ErrInvalidRequest},
		{"context too long 413", live, func(t *testing.T) llmkit.Client {
			return fbStack(t, TypeAnthropic, "m", fbHTTP(t, 413, errorBody("anthropic", 413, "too big")).url, retry.Config{MaxAttempts: 3, Sleep: fbNoSleep})
		}, llmkit.ErrContextTooLong},
		{"context too long typed", live, func(*testing.T) llmkit.Client {
			return fbFails("p", &llmkit.APIError{Kind: llmkit.ErrContextTooLong, Provider: "fake"})
		}, llmkit.ErrContextTooLong},
		{"plain non-APIError", live, func(*testing.T) llmkit.Client { return fbFails("p", plain) }, plain},
		{"canceled ctx, retryable error", deadCtx(true), func(*testing.T) llmkit.Client {
			return fbFails("p", fbOverloaded())
		}, llmkit.ErrOverloaded},
		{"expired deadline, retryable error", deadCtx(false), func(*testing.T) llmkit.Client {
			return fbFails("p", fbOverloaded())
		}, llmkit.ErrOverloaded},
		{"canceled ctx, real stack", deadCtx(true), func(t *testing.T) llmkit.Client {
			return fbStack(t, TypeAnthropic, "m", fbHTTP(t, 529, fbAnthropicOverloaded).url, retry.Config{MaxAttempts: 3, Sleep: fbNoSleep})
		}, context.Canceled},
		{"expired deadline, real stack", deadCtx(false), func(t *testing.T) llmkit.Client {
			return fbStack(t, TypeAnthropic, "m", fbHTTP(t, 529, fbAnthropicOverloaded).url, retry.Config{MaxAttempts: 3, Sleep: fbNoSleep})
		}, context.DeadlineExceeded},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			ctx, cancel := row.ctx()
			defer cancel()
			secondary := fbServes("s", "must not run")
			fb := Fallback(row.primary(t), secondary)
			if _, err := fb.Complete(ctx, fbReq()); !errors.Is(err, row.want) {
				t.Errorf("Complete error = %v, want errors.Is %v", err, row.want)
			}
			if _, err := llmkit.Stream(ctx, fb, fbReq(), func(llmkit.Delta) error { return nil }); !errors.Is(err, row.want) {
				t.Errorf("Stream error = %v, want errors.Is %v", err, row.want)
			}
			if n := secondary.calls.Load(); n != 0 {
				t.Errorf("secondary called %d times, want 0", n)
			}
		})
	}
}

func TestFallback_F3a_MembersReceiveRequestUnmodified(t *testing.T) {
	req := llmkit.Request{
		System: "sys",
		Messages: []llmkit.Message{
			llmkit.TextMessage(llmkit.RoleUser, "go"),
			{Role: llmkit.RoleAssistant, Content: []llmkit.Block{
				{Kind: llmkit.BlockThinking, Text: "t", Provider: "anthropic", Raw: json.RawMessage(`{"type":"thinking","thinking":"t","signature":"s"}`)},
				{Kind: llmkit.BlockThinking, Text: "g", Provider: "google", Raw: json.RawMessage(`{"text":"g","thought":true}`)},
				llmkit.Text("partial"),
			}, ToolCalls: []llmkit.ToolCall{{ID: "call_1", Name: "read", Arguments: json.RawMessage(`{"p":"a"}`)}}},
			llmkit.ToolResult("call_1", "contents"),
		},
		Tools:     fbTools(),
		MaxTokens: 128,
		Thinking:  &llmkit.ThinkingConfig{BudgetTokens: 1024},
	}
	want := req
	want.Messages = append([]llmkit.Message(nil), req.Messages...)

	primary := fbFails("p", fbOverloaded())
	secondary := fbServes("s", "ok")
	if _, err := Fallback(primary, secondary).Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	for name, m := range map[string]*fbFake{"primary": primary, "secondary": secondary} {
		if len(m.reqs) != 1 {
			t.Fatalf("%s saw %d requests, want 1", name, len(m.reqs))
		}
		if !reflect.DeepEqual(m.reqs[0], want) {
			t.Errorf("%s request differs from the caller's:\n got %+v\nwant %+v", name, m.reqs[0], want)
		}
	}
}

func TestFallback_F3b_DifferentTagThinkingNotSent(t *testing.T) {
	rows := []struct {
		name      string
		histTyp   Type
		histModel string
		histReply string
		secTyp    Type
		secModel  string
		secBody   string
		secret    string
	}{
		{"anthropic-tagged to openai-compatible", TypeAnthropic, "claude-sonnet-4-5", fbAnthropicThinking,
			TypeOpenAICompatible, "sec-model", mockTextBody("openai-compatible", "ok", 1, 1), "plan it"},
		{"google-tagged to anthropic", TypeGoogle, "gemini-2.5-pro", fbGoogleThinking,
			TypeAnthropic, "claude-sonnet-4-5", mockTextBody("anthropic", "ok", 1, 1), "g thoughts"},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			hist := fbHistory(t, row.histTyp, row.histModel, row.histReply)
			if tag := hist[1].Content[0].Provider; tag == "" || strings.Contains(string(row.secTyp), tag) {
				t.Fatalf("history thinking tag %q is not a different tag from the secondary %s", tag, row.secTyp)
			}
			primarySrv := fbHTTP(t, 529, fbAnthropicOverloaded)
			secondarySrv := fbHTTP(t, 200, row.secBody)
			fb := Fallback(
				fbStack(t, TypeAnthropic, "claude-sonnet-4-5", primarySrv.url, retry.Config{MaxAttempts: 1}),
				fbStack(t, row.secTyp, row.secModel, secondarySrv.url, retry.Config{MaxAttempts: 1}),
			)
			if _, err := fb.Complete(context.Background(), llmkit.Request{Messages: hist, Tools: fbTools(), MaxTokens: 64}); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if secondarySrv.hits.Load() != 1 {
				t.Fatalf("secondary hits = %d, want 1", secondarySrv.hits.Load())
			}
			wire := secondarySrv.lastBody()
			for _, banned := range []string{row.secret, "reasoning_content", `"thinking"`, "thoughtSignature", "sigA"} {
				if strings.Contains(wire, banned) {
					t.Errorf("secondary wire carries %q from the other provider's thinking:\n%s", banned, wire)
				}
			}
			if !strings.Contains(wire, "contents") {
				t.Errorf("secondary wire lost the tool result:\n%s", wire)
			}
		})
	}
}

func TestFallback_F3c_SameTagThinkingPassesThrough(t *testing.T) {
	hist := fbHistory(t, TypeAnthropic, "claude-sonnet-4-5", fbAnthropicThinking)
	primarySrv := fbHTTP(t, 529, fbAnthropicOverloaded)
	secondarySrv := fbHTTP(t, 200, mockTextBody("anthropic", "ok", 1, 1))
	fb := Fallback(
		fbStack(t, TypeAnthropic, "claude-sonnet-4-5", primarySrv.url, retry.Config{MaxAttempts: 1}),
		fbStack(t, TypeAnthropic, "claude-opus-4-1", secondarySrv.url, retry.Config{MaxAttempts: 1}),
	)
	if _, err := fb.Complete(context.Background(), llmkit.Request{Messages: hist, Tools: fbTools(), MaxTokens: 64}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	wire := secondarySrv.lastBody()
	for _, want := range []string{`"type":"thinking"`, "plan it", "sigA"} {
		if !strings.Contains(wire, want) {
			t.Errorf("same-tag secondary wire lacks %q:\n%s", want, wire)
		}
	}
}

func TestFallback_F4_Stream(t *testing.T) {
	text := func(s string) llmkit.Delta { return llmkit.Delta{Kind: llmkit.DeltaText, Text: s} }

	t.Run("failure before first delta falls through", func(t *testing.T) {
		primary := &fbStreamFake{fbFake: fbFails("p", nil), stream: func(context.Context, llmkit.Request, func(llmkit.Delta) error) (llmkit.Response, error) {
			return llmkit.Response{}, fbOverloaded()
		}}
		secondary := &fbStreamFake{fbFake: fbServes("s", "ab"), stream: func(_ context.Context, _ llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
			_ = fn(text("a"))
			_ = fn(text("b"))
			return llmkit.Response{Text: "ab"}, nil
		}}
		var got []string
		resp, err := llmkit.Stream(context.Background(), Fallback(primary, secondary), fbReq(), func(d llmkit.Delta) error {
			got = append(got, d.Text)
			return nil
		})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if !reflect.DeepEqual(got, []string{"a", "b"}) {
			t.Errorf("deltas = %v, want [a b]", got)
		}
		if resp.Served != fbID("s") {
			t.Errorf("Served = %+v, want %+v", resp.Served, fbID("s"))
		}
	})

	t.Run("non-streaming secondary is driven through llmkit.Stream", func(t *testing.T) {
		primary := &fbStreamFake{fbFake: fbFails("p", nil), stream: func(context.Context, llmkit.Request, func(llmkit.Delta) error) (llmkit.Response, error) {
			return llmkit.Response{}, fbOverloaded()
		}}
		secondary := fbServes("s", "whole")
		var got []string
		resp, err := Fallback(primary, secondary).(llmkit.StreamingClient).Stream(context.Background(), fbReq(), func(d llmkit.Delta) error {
			got = append(got, d.Text)
			return nil
		})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if !reflect.DeepEqual(got, []string{"whole"}) || resp.Served != fbID("s") {
			t.Errorf("deltas = %v served = %+v", got, resp.Served)
		}
	})

	t.Run("failure after a delta is terminal", func(t *testing.T) {
		primary := &fbStreamFake{fbFake: fbFails("p", nil), stream: func(_ context.Context, _ llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
			_ = fn(text("partial"))
			return llmkit.Response{}, fbOverloaded()
		}}
		secondary := fbServes("s", "must not run")
		n := 0
		_, err := llmkit.Stream(context.Background(), Fallback(primary, secondary), fbReq(), func(llmkit.Delta) error { n++; return nil })
		if !errors.Is(err, llmkit.ErrOverloaded) {
			t.Errorf("error = %v, want the primary's ErrOverloaded", err)
		}
		if secondary.calls.Load() != 0 {
			t.Errorf("secondary called %d times after a delivered delta, want 0", secondary.calls.Load())
		}
		if n != 1 {
			t.Errorf("fn saw %d deltas, want 1", n)
		}
	})

	t.Run("nil fn falls through", func(t *testing.T) {
		primary := &fbStreamFake{fbFake: fbFails("p", nil), stream: func(context.Context, llmkit.Request, func(llmkit.Delta) error) (llmkit.Response, error) {
			return llmkit.Response{}, fbOverloaded()
		}}
		resp, err := Fallback(primary, fbServes("s", "x")).(llmkit.StreamingClient).Stream(context.Background(), fbReq(), nil)
		if err != nil || resp.Served != fbID("s") {
			t.Errorf("resp = %+v err = %v", resp, err)
		}
	})
}

func TestFallback_F5_AllFailReturnsLastError(t *testing.T) {
	primarySrv := fbHTTP(t, 529, fbAnthropicOverloaded)
	secondarySrv := fbHTTP(t, 503, errorBody("openai-compatible", 503, "secondary is down"))
	fb := Fallback(
		fbStack(t, TypeAnthropic, "claude-sonnet-4-5", primarySrv.url, retry.Config{MaxAttempts: 1}),
		fbStack(t, TypeOpenAICompatible, "sec-model", secondarySrv.url, retry.Config{MaxAttempts: 1}),
	)
	_, err := fb.Complete(context.Background(), fbReq())
	if err == nil {
		t.Fatal("want an error")
	}
	var ae *llmkit.APIError
	if !errors.As(err, &ae) || ae.StatusCode != 503 || ae.Kind != llmkit.ErrServer {
		t.Fatalf("errors.As gave %+v, want the last member's 503 ErrServer", ae)
	}
	if !errors.Is(err, llmkit.ErrServer) {
		t.Error("errors.Is(err, ErrServer) = false, want true")
	}
	if errors.Is(err, llmkit.ErrOverloaded) {
		t.Error("errors.Is(err, ErrOverloaded) = true: the primary's kind leaked into the chain")
	}
	if !strings.Contains(err.Error(), "Overloaded") {
		t.Errorf("message does not mention the primary's error: %v", err)
	}
}

func TestFallback_TerminalAfterFallThroughKeepsEarlierErrorInMessage(t *testing.T) {
	primary := fbFails("p", fbOverloaded())
	secondary := fbFails("s", &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 400, Provider: "fake", Message: "secondary refuses"})
	third := fbServes("t", "must not run")
	_, err := Fallback(primary, secondary, third).Complete(context.Background(), fbReq())
	if !errors.Is(err, llmkit.ErrInvalidRequest) || errors.Is(err, llmkit.ErrOverloaded) {
		t.Errorf("chain = %v, want only the secondary's ErrInvalidRequest", err)
	}
	if !strings.Contains(err.Error(), "primary overloaded") {
		t.Errorf("message does not mention the primary's error: %v", err)
	}
	if third.calls.Load() != 0 {
		t.Errorf("third member called after a terminal error")
	}
}

func TestFallback_F6_CapabilitiesAndWireArePrimarys(t *testing.T) {
	req := llmkit.Request{
		Messages:       []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "x")},
		MaxTokens:      4096,
		Thinking:       &llmkit.ThinkingConfig{Effort: llmkit.EffortHigh},
		ResponseSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`),
	}
	primSrv := fbHTTP(t, 200, mockTextBody("openai", `{"a":"x"}`, 1, 1))
	secSrv := fbHTTP(t, 200, mockTextBody("openai-compatible", "x", 1, 1))
	primary := fbStack(t, TypeOpenAI, "o3", primSrv.url, retry.Config{MaxAttempts: 1})
	secondary, err := New(context.Background(), Spec{
		Type: TypeOpenAICompatible, Model: "weak", BaseURL: secSrv.url, Auth: AuthAPIKey, Secret: "k",
		Capabilities: func(c llmkit.Capabilities) llmkit.Capabilities {
			c.StructuredOutput, c.Thinking, c.ReasoningEffort = false, false, false
			return c
		},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	pc, sc := primary.Capabilities(), secondary.Capabilities()
	if !pc.StructuredOutput || !pc.ReasoningEffort || sc.StructuredOutput || sc.Thinking || sc.ReasoningEffort {
		t.Fatalf("setup: primary caps %+v, secondary caps %+v", pc, sc)
	}

	fb := Fallback(primary, secondary)
	if got := fb.Capabilities(); !reflect.DeepEqual(got, pc) {
		t.Errorf("Capabilities() = %+v, want the primary's %+v", got, pc)
	}

	if _, err := primary.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	alone := primSrv.lastBody()
	if _, err := fb.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := primSrv.lastBody(); got != alone {
		t.Errorf("primary wire through Fallback differs:\n got %s\nwant %s", got, alone)
	}
	if !strings.Contains(alone, "response_format") || !strings.Contains(alone, "reasoning_effort") {
		t.Errorf("primary wire carries neither schema nor effort, so the row proves nothing: %s", alone)
	}
	if secSrv.hits.Load() != 0 {
		t.Error("secondary called although the primary served")
	}
}

func TestFallback_F7_NestedServedNamesTheServingMember(t *testing.T) {
	cases := []struct {
		name    string
		a, b, c *fbFake
		want    string
	}{
		{"b serves", fbFails("a", fbOverloaded()), fbServes("b", "x"), fbServes("c", "x"), "b"},
		{"c serves", fbFails("a", fbOverloaded()), fbFails("b", fbOverloaded()), fbServes("c", "x"), "c"},
		{"a serves", fbServes("a", "x"), fbServes("b", "x"), fbServes("c", "x"), "a"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			outer := Fallback(Fallback(tc.a, tc.b), tc.c)
			resp, err := outer.Complete(context.Background(), fbReq())
			if err != nil {
				t.Fatal(err)
			}
			if resp.Served != fbID(tc.want) {
				t.Errorf("Complete Served = %+v, want %+v", resp.Served, fbID(tc.want))
			}
			resp, err = outer.(llmkit.StreamingClient).Stream(context.Background(), fbReq(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if resp.Served != fbID(tc.want) {
				t.Errorf("Stream Served = %+v, want %+v", resp.Served, fbID(tc.want))
			}
		})
	}
}

func TestFallback_R1_ServedNeverNamesAFailedMember(t *testing.T) {
	anon := func() llmkit.Client {
		foreign := &fbFake{complete: func(context.Context, llmkit.Request) (llmkit.Response, error) {
			return llmkit.Response{Text: "from b"}, nil
		}}
		return Wrap(foreign, Options{})
	}
	if got := llmkit.IdentityOf(anon()); got != (llmkit.Identity{}) {
		t.Fatalf("foreign member must be anonymous, got %+v", got)
	}
	build := map[string]func(a, b llmkit.Client) llmkit.Client{
		"flat":   func(a, b llmkit.Client) llmkit.Client { return Fallback(a, b) },
		"nested": func(a, b llmkit.Client) llmkit.Client { return Fallback(Fallback(a, b), fbServes("c", "from c")) },
	}
	for name, mk := range build {
		for _, mode := range []string{"Complete", "Stream"} {
			t.Run(name+"/"+mode, func(t *testing.T) {
				srv := fbHTTP(t, 529, fbAnthropicOverloaded)
				a := fbStack(t, TypeAnthropic, "claude-a", srv.url, retry.Config{MaxAttempts: 1})
				fb := mk(a, anon())
				var resp llmkit.Response
				var err error
				if mode == "Complete" {
					resp, err = fb.Complete(context.Background(), fbReq())
				} else {
					resp, err = fb.(llmkit.StreamingClient).Stream(context.Background(), fbReq(), nil)
				}
				if err != nil {
					t.Fatal(err)
				}
				if resp.Text != "from b" || srv.hits.Load() != 1 {
					t.Fatalf("b must serve after one failed call on a: text %q, a hits %d", resp.Text, srv.hits.Load())
				}
				if resp.Served == llmkit.IdentityOf(a) {
					t.Errorf("Served names the failed member a: %+v", resp.Served)
				}
				if resp.Served != (llmkit.Identity{}) {
					t.Errorf("Served = %+v, want zero for the anonymous serving member", resp.Served)
				}
			})
		}
	}
}

func TestFallback_R1_HiddenFallbackLimitIsTheDocumentedOne(t *testing.T) {
	srv := fbHTTP(t, 529, fbAnthropicOverloaded)
	a := fbStack(t, TypeAnthropic, "claude-a", srv.url, retry.Config{MaxAttempts: 1})
	foreign := &fbFake{complete: func(context.Context, llmkit.Request) (llmkit.Response, error) {
		return llmkit.Response{Text: "from b"}, nil
	}}
	hidden := llmkit.Observe(Fallback(a, Wrap(foreign, Options{})), llmkit.ObserverFunc(func(context.Context, llmkit.Event) {}))
	resp, err := Fallback(hidden, fbServes("c", "from c")).Complete(context.Background(), fbReq())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "from b" || resp.Served != llmkit.IdentityOf(hidden) {
		t.Errorf("hidden Fallback: text %q Served %+v, want the wrapper's Identity %+v", resp.Text, resp.Served, llmkit.IdentityOf(hidden))
	}
}

func TestFallback_F8_BudgetContract(t *testing.T) {
	stall := func(t *testing.T) *fbServer {
		s := &fbServer{}
		s.url = newServer(t, func(w http.ResponseWriter, r *http.Request) {
			s.hits.Add(1)
			_, _ = io.Copy(io.Discard, r.Body)
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
		})
		return s
	}
	build := func(t *testing.T, rc retry.Config) (llmkit.Client, *fbServer, *fbServer) {
		stalled := stall(t)
		secondary := fbHTTP(t, 200, mockTextBody("openai-compatible", "served", 1, 1))
		fb := Fallback(
			fbStack(t, TypeOpenAICompatible, "stalls", stalled.url, rc),
			fbStack(t, TypeOpenAICompatible, "serves", secondary.url, retry.Config{MaxAttempts: 1}),
		)
		return fb, stalled, secondary
	}

	t.Run("member budget inside the deadline falls through", func(t *testing.T) {
		fb, stalled, secondary := build(t, retry.Config{MaxAttempts: 2, RequestTimeout: 100 * time.Millisecond, BaseDelay: 10 * time.Millisecond, MaxDelay: 20 * time.Millisecond})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		start := time.Now()
		resp, err := fb.Complete(ctx, fbReq())
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if resp.Text != "served" || stalled.hits.Load() != 2 || secondary.hits.Load() != 1 {
			t.Errorf("text %q stalled hits %d secondary hits %d", resp.Text, stalled.hits.Load(), secondary.hits.Load())
		}
		if time.Since(start) >= 5*time.Second {
			t.Error("fall-through took the whole caller deadline")
		}
	})

	t.Run("member budget beyond the deadline returns the ctx error", func(t *testing.T) {
		fb, stalled, secondary := build(t, retry.Config{MaxAttempts: 4, RequestTimeout: 400 * time.Millisecond, BaseDelay: 10 * time.Millisecond, MaxDelay: 20 * time.Millisecond})
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		_, err := fb.Complete(ctx, fbReq())
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("error = %v, want errors.Is DeadlineExceeded", err)
		}
		if secondary.hits.Load() != 0 {
			t.Errorf("secondary called %d times after the caller deadline, want 0", secondary.hits.Load())
		}
		if stalled.hits.Load() == 0 {
			t.Error("primary never called")
		}
	})
}

func TestFallback_F9_Concurrent(t *testing.T) {
	const n = 16
	primary := fbFails("p", fbOverloaded())
	secondary := &fbStreamFake{fbFake: fbServes("s", "x"), stream: func(_ context.Context, _ llmkit.Request, fn func(llmkit.Delta) error) (llmkit.Response, error) {
		_ = fn(llmkit.Delta{Kind: llmkit.DeltaText, Text: "x"})
		return llmkit.Response{Text: "x"}, nil
	}}
	fb := Fallback(primary, secondary)

	var wg sync.WaitGroup
	errs := make(chan error, 2*n)
	for range n {
		wg.Add(2)
		go func() {
			defer wg.Done()
			resp, err := fb.Complete(context.Background(), fbReq())
			if err == nil && resp.Served != fbID("s") {
				err = fmt.Errorf("Complete Served = %+v", resp.Served)
			}
			errs <- err
		}()
		go func() {
			defer wg.Done()
			var got int
			resp, err := llmkit.Stream(context.Background(), fb, fbReq(), func(llmkit.Delta) error { got++; return nil })
			if err == nil && (resp.Served != fbID("s") || got != 1) {
				err = fmt.Errorf("Stream Served = %+v deltas = %d", resp.Served, got)
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Error(err)
		}
	}
	if got := primary.calls.Load(); got != 2*n {
		t.Errorf("primary calls = %d, want %d", got, 2*n)
	}
}

func TestFallback_IdentityAndNilMembers(t *testing.T) {
	p := fbServes("p", "x")
	fb := Fallback(p, fbServes("s", "x"))
	if got := llmkit.IdentityOf(fb); got != fbID("p") {
		t.Errorf("Identity = %+v, want the primary's", got)
	}

	for name, build := range map[string]func(){
		"nil primary":  func() { Fallback(nil, p) },
		"nil fallback": func() { Fallback(p, nil) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("want a panic on a nil member")
				}
			}()
			build()
		})
	}
}

func TestFallback_ServedKeptWhenMemberSetIt(t *testing.T) {
	set := llmkit.Identity{Provider: "inner", Model: "m"}
	member := &fbFake{id: fbID("outer-view"), complete: func(context.Context, llmkit.Request) (llmkit.Response, error) {
		return llmkit.Response{Served: set}, nil
	}}
	resp, err := Fallback(member).Complete(context.Background(), fbReq())
	if err != nil || resp.Served != set {
		t.Errorf("Served = %+v err = %v, want %+v kept", resp.Served, err, set)
	}
}

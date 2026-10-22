package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

const dummySig = "skip_thought_signature_validator"

type wireContent struct {
	Role  string           `json:"role"`
	Parts []map[string]any `json:"parts"`
}

// signedModelPath matches the request paths of gemini-3 and gemini-4 models, with the same name boundary as production ("-", "." or end of name, here ":" before the method).
var signedModelPath = regexp.MustCompile(`/models/gemini-[34](?:[-.]|:|$)`)

// gemini3Fake enforces, on gemini-3 and gemini-4 paths only, Google's rule that the first functionCall of each model step after the last user text is signed.
// It answers both generateContent and the SSE streamGenerateContent endpoint.
func gemini3Fake(t *testing.T) (string, func() []wireContent) {
	t.Helper()
	var mu sync.Mutex
	var last []wireContent
	url := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body struct {
			Contents []wireContent `json:"contents"`
		}
		_ = json.Unmarshal(b, &body)
		mu.Lock()
		last = body.Contents
		mu.Unlock()
		if signedModelPath.MatchString(r.URL.Path) {
			turn := -1
			for i, c := range body.Contents {
				for _, p := range c.Parts {
					if _, ok := p["text"]; ok && c.Role == "user" {
						turn = i
					}
				}
			}
			for _, c := range body.Contents[turn+1:] {
				for _, p := range c.Parts {
					if _, ok := p["functionCall"]; !ok {
						continue
					}
					if _, signed := p["thoughtSignature"]; !signed {
						w.Header().Set("Content-Type", "application/json")
						w.WriteHeader(http.StatusBadRequest)
						_, _ = io.WriteString(w, `{"error":{"code":400,"message":"Function call is missing a thought_signature in functionCall parts.","status":"INVALID_ARGUMENT"}}`)
						return
					}
					break
				}
			}
		}
		if strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+mockTextBody("ok", 1, 1)+"\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, mockTextBody("ok", 1, 1))
	})
	return url, func() []wireContent {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

func sendHistory(t *testing.T, model string, msgs []llmkit.Message) []wireContent {
	t.Helper()
	return sendHistoryVia(t, model, msgs, false)
}

// sendHistoryVia sends msgs through Complete, or through Stream when stream is set, and returns the contents on the wire.
func sendHistoryVia(t *testing.T, model string, msgs []llmkit.Message, stream bool) []wireContent {
	t.Helper()
	url, got := gemini3Fake(t)
	c, err := New(context.Background(), model, Options{APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := llmkit.Request{
		Messages:  msgs,
		MaxTokens: 64,
		Tools: []llmkit.ToolDef{{
			Name:       "read",
			Parameters: json.RawMessage(`{"type":"object","properties":{"p":{"type":"string"}}}`),
		}},
	}
	if stream {
		sc, ok := c.(llmkit.StreamingClient)
		if !ok {
			t.Fatalf("%T is not a StreamingClient", c)
		}
		_, err = sc.Stream(context.Background(), req, func(llmkit.Delta) error { return nil })
	} else {
		_, err = c.Complete(context.Background(), req)
	}
	if err != nil {
		t.Fatalf("send on %s (stream=%v): %v", model, stream, err)
	}
	return got()
}

func call(id string) llmkit.ToolCall {
	return llmkit.ToolCall{ID: id, Name: "read", Arguments: json.RawMessage(`{"p":"a"}`)}
}

// foreignStep is an assistant step without google signatures, plus its results.
func foreignStep(ids ...string) []llmkit.Message {
	m := llmkit.Message{Role: llmkit.RoleAssistant}
	var results []llmkit.Message
	for _, id := range ids {
		m.ToolCalls = append(m.ToolCalls, call(id))
		results = append(results, llmkit.ToolResult(id, "contents"))
	}
	return append([]llmkit.Message{m}, results...)
}

// googleStep is an assistant step whose call carries a google-tagged real signature.
func googleStep(id, sig string) []llmkit.Message {
	raw := `{"functionCall":{"id":"` + id + `","name":"read"},"thoughtSignature":"` + sig + `"}`
	m := llmkit.Message{
		Role:      llmkit.RoleAssistant,
		Content:   []llmkit.Block{{Kind: llmkit.BlockThinking, Provider: "google", Raw: json.RawMessage(raw)}},
		ToolCalls: []llmkit.ToolCall{call(id)},
	}
	return []llmkit.Message{m, llmkit.ToolResult(id, "contents")}
}

func history(parts ...[]llmkit.Message) []llmkit.Message {
	msgs := []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "go")}
	for _, p := range parts {
		msgs = append(msgs, p...)
	}
	return msgs
}

// sigsOf returns each functionCall part's thoughtSignature in wire order ("" when absent).
func sigsOf(t *testing.T, contents []wireContent) []string {
	t.Helper()
	var out []string
	for _, c := range contents {
		for _, p := range c.Parts {
			if _, ok := p["functionCall"]; !ok {
				continue
			}
			s, _ := p["thoughtSignature"].(string)
			out = append(out, s)
		}
	}
	return out
}

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func checkSigs(t *testing.T, contents []wireContent, want ...string) {
	t.Helper()
	got := sigsOf(t, contents)
	if len(got) != len(want) {
		t.Fatalf("functionCall signatures = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("functionCall[%d] signature = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestForeignCallSignature_G1_ForeignHistoryOnGemini3(t *testing.T) {
	contents := sendHistory(t, "gemini-3-pro", history(foreignStep("toolu_01")))
	checkSigs(t, contents, b64(dummySig))
	raw, err := base64.StdEncoding.DecodeString(sigsOf(t, contents)[0])
	if err != nil || string(raw) != dummySig {
		t.Errorf("signature decodes to %q (err %v), want %q", raw, err, dummySig)
	}
}

func TestForeignCallSignature_G2_RealSignatureKept(t *testing.T) {
	contents := sendHistory(t, "gemini-3-pro", history(googleStep("call_g", b64("real"))))
	checkSigs(t, contents, b64("real"))
}

func TestForeignCallSignature_G3_Gemini25Unchanged(t *testing.T) {
	url, got := gemini3Fake(t)
	c, err := New(context.Background(), "gemini-2.5-pro", Options{APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = c.Complete(context.Background(), llmkit.Request{
		Messages:  history(foreignStep("toolu_01")),
		MaxTokens: 64,
		Tools:     []llmkit.ToolDef{{Name: "read", Parameters: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	checkSigs(t, got(), "")
}

func TestForeignCallSignature_G4_MixedHistory(t *testing.T) {
	prior := foreignStep("old_1")
	tests := []struct {
		name string
		msgs []llmkit.Message
		want []string
	}{
		{
			name: "google step then foreign step",
			msgs: history(googleStep("g1", b64("real")), foreignStep("f1")),
			want: []string{b64("real"), b64(dummySig)},
		},
		{
			name: "foreign step then google step",
			msgs: history(foreignStep("f1"), googleStep("g1", b64("real"))),
			want: []string{b64(dummySig), b64("real")},
		},
		{
			name: "parallel foreign step signs the first call only",
			msgs: history(foreignStep("f1", "f2", "f3")),
			want: []string{b64(dummySig), "", ""},
		},
		{
			name: "calls before the current turn are untouched",
			msgs: append(append(history(prior), llmkit.TextMessage(llmkit.RoleAssistant, "done"), llmkit.TextMessage(llmkit.RoleUser, "again")), foreignStep("f1")...),
			want: []string{"", b64(dummySig)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			checkSigs(t, sendHistory(t, "gemini-3-pro", tc.msgs), tc.want...)
		})
	}
}

func TestForeignCallSignature_G6_Gemini4Bypass(t *testing.T) {
	for _, model := range []string{"gemini-4-argon", "gemini-4.1-pro", "gemini-4"} {
		t.Run(model, func(t *testing.T) {
			checkSigs(t, sendHistory(t, model, history(foreignStep("toolu_01"))), b64(dummySig))
		})
	}
}

// G6b: "gemini-40x" shares only the "gemini-4" prefix, with no name boundary, so it is no Gemini 4 model and gets no bypass.
func TestForeignCallSignature_G6b_Gemini4NoBoundary(t *testing.T) {
	for _, model := range []string{"gemini-40x", "gemini-30x"} {
		t.Run(model, func(t *testing.T) {
			checkSigs(t, sendHistory(t, model, history(foreignStep("toolu_01"))), "")
		})
	}
}

// G7: Stream signs foreign calls exactly as Complete does.
func TestForeignCallSignature_G7_StreamSigns(t *testing.T) {
	contents := sendHistoryVia(t, "gemini-3-pro", history(foreignStep("toolu_01")), true)
	checkSigs(t, contents, b64(dummySig))
}

package livetest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

// Fixture is the recorded artifact of one live case: everything needed to
// replay the case hermetically and detect adapter wire drift.
//
//   - Inputs are the llmkit.Requests the case issued, in order; replay
//     re-issues them verbatim through provider.New.
//   - Exchanges is the sanitized wire truth — what the vendor actually
//     received and returned — one per Input, in order; replay serves them
//     from an httptest server.
//   - Capabilities pins the effective profile at record time (including any
//     CAPS override), so replay re-issues under the same wire-affecting
//     profile regardless of the adapter's tables or the operator's env.
//   - Normalized is the FINAL call's outcome; replay asserts the adapter
//     still produces it from the recorded truth.
type Fixture struct {
	Case         string              `json:"case"`
	Lane         string              `json:"lane"`
	Model        string              `json:"model"`
	Capabilities llmkit.Capabilities `json:"capabilities"`
	// RequestCheck declares the replay's request-side mode:
	//
	//   - RequestCheckStrict: the replay asserts the adapter's outgoing
	//     request (method, path, body compared as parsed JSON; headers
	//     excluded) equals the recorded request, AND the normalized
	//     response matches.
	//   - RequestCheckResponseOnly: multi-turn fixtures whose later requests
	//     echo model-generated ids (tool-call ids, thinking signatures)
	//     compare response normalization only.
	//
	// The recorder sets strict for single-exchange cases. A fixture without
	// the field (legacy) replays as strict when it holds exactly one
	// exchange, response_only otherwise.
	RequestCheck string           `json:"request_check,omitempty"`
	Inputs       []llmkit.Request `json:"inputs"`
	Exchanges    []Exchange       `json:"exchanges"`
	Normalized   Normalized       `json:"normalized"`
}

// Replay request-side modes; see Fixture.RequestCheck.
const (
	RequestCheckStrict       = "strict"
	RequestCheckResponseOnly = "response_only"
)

// RequestCheckMode resolves a fixture's effective request-side mode,
// deriving the pre-request_check legacy default from the exchange count.
// Unknown declared values are returned unchanged so the caller's error can
// name them exactly once.
func (f *Fixture) RequestCheckMode() string {
	switch f.RequestCheck {
	case RequestCheckStrict, RequestCheckResponseOnly:
		return f.RequestCheck
	case "":
		if len(f.Exchanges) == 1 {
			return RequestCheckStrict
		}
		return RequestCheckResponseOnly
	default:
		return f.RequestCheck
	}
}

// Normalized is the recorded outcome of a case's final completion: the
// normalized Response fields replay asserts on, or the sentinel error kind.
type Normalized struct {
	ErrKind    string            `json:"err_kind,omitempty"`
	Text       string            `json:"text"`
	ToolCalls  []llmkit.ToolCall `json:"tool_calls,omitempty"`
	StopReason llmkit.StopReason `json:"stop_reason"`
	Usage      llmkit.Usage      `json:"usage"`
}

// ErrKindOf names the sentinel kind of a normalized error, "" for nil.
func ErrKindOf(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, llmkit.ErrAuth):
		return "ErrAuth"
	case errors.Is(err, llmkit.ErrRateLimited):
		return "ErrRateLimited"
	case errors.Is(err, llmkit.ErrContextTooLong):
		return "ErrContextTooLong"
	case errors.Is(err, llmkit.ErrInvalidRequest):
		return "ErrInvalidRequest"
	case errors.Is(err, llmkit.ErrServer):
		return "ErrServer"
	case errors.Is(err, llmkit.ErrOverloaded):
		return "ErrOverloaded"
	default:
		return "other"
	}
}

// SentinelByKind maps a recorded ErrKind back to the sentinel for
// errors.Is assertions in replay. Unknown and "" map to nil.
func SentinelByKind(kind string) error {
	switch kind {
	case "ErrRateLimited":
		return llmkit.ErrRateLimited
	case "ErrAuth":
		return llmkit.ErrAuth
	case "ErrContextTooLong":
		return llmkit.ErrContextTooLong
	case "ErrInvalidRequest":
		return llmkit.ErrInvalidRequest
	case "ErrServer":
		return llmkit.ErrServer
	case "ErrOverloaded":
		return llmkit.ErrOverloaded
	default:
		return nil
	}
}

// WriteFixture serializes f to path, asserting first that the recording
// contains neither the lane credential nor any sk- substring: a fixture that
// leaks is not written at all. Fatal on any refusal or write error.
func WriteFixture(t testing.TB, path string, f *Fixture, secret string) {
	if err := writeFixture(path, f, secret); err != nil {
		t.Fatalf("livetest: %v", err)
	}
}

func writeFixture(path string, f *Fixture, secret string) error {
	// The recorder rule: single-exchange fixtures carry the full request
	// side; multi-exchange fixtures default to response-only unless the
	// caller set the mode explicitly.
	if f.RequestCheck == "" {
		f.RequestCheck = RequestCheckStrict
		if len(f.Exchanges) != 1 {
			f.RequestCheck = RequestCheckResponseOnly
		}
	}
	b, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal fixture: %w", err)
	}
	b = append(b, '\n')
	if secret != "" && strings.Contains(string(b), secret) {
		return fmt.Errorf("refusing to write %s: the lane credential appears in the recording", path)
	}
	if strings.Contains(string(b), "sk-") {
		return fmt.Errorf("refusing to write %s: the recording contains an sk- substring", path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create fixture dir: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("write fixture: %w", err)
	}
	return nil
}

// ReadFixture loads a recorded fixture.
func ReadFixture(path string) (*Fixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("livetest: %s: %w", path, err)
	}
	return &f, nil
}

// NormalizeFixture compacts every json.RawMessage embedded in f — request
// inputs (tool parameters, tool-call arguments, response schemas, provider
// raw blocks) and the normalized outcome's tool calls. Fixture files are
// pretty-printed, which re-indents embedded RawMessage bytes; anyone
// re-issuing a fixture's requests or comparing its normalized outcome MUST
// compact them back first, or the adapter's wire bytes drift from the
// recording.
func NormalizeFixture(f *Fixture) {
	normalizeInputs(f.Inputs)
	for i := range f.Normalized.ToolCalls {
		f.Normalized.ToolCalls[i].Arguments = compactRaw(f.Normalized.ToolCalls[i].Arguments)
	}
}

func normalizeInputs(reqs []llmkit.Request) {
	for i := range reqs {
		r := &reqs[i]
		for j := range r.Tools {
			r.Tools[j].Parameters = compactRaw(r.Tools[j].Parameters)
		}
		for _, m := range r.Messages {
			for b := range m.Content {
				m.Content[b].Raw = compactRaw(m.Content[b].Raw)
			}
			for c := range m.ToolCalls {
				m.ToolCalls[c].Arguments = compactRaw(m.ToolCalls[c].Arguments)
			}
		}
		r.ResponseSchema = compactRaw(r.ResponseSchema)
	}
}

func compactRaw(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return raw // not valid JSON: leave it for the adapter to reject
	}
	return buf.Bytes()
}

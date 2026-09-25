package adapter

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/dpoage/llmkit"
)

func fullCaps() llmkit.Capabilities {
	return llmkit.Capabilities{
		StructuredOutput: true,
		Thinking:         true,
		ToolChoice:       true,
		StopSequences:    true,
		TopP:             true,
		TopK:             true,
		Seed:             true,
	}
}

func minimalRequest() llmkit.Request {
	return llmkit.Request{Messages: []llmkit.Message{llmkit.TextMessage(llmkit.RoleUser, "hi")}}
}

func wantRefused(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v, want ErrInvalidRequest", err)
	}
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v (%T), want *llmkit.APIError", err, err)
	}
	if apiErr.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (pre-wire)", apiErr.StatusCode)
	}
}

// TestPrepare_Thinking pins the Thinking gate: caps.Thinking=false drops the
// config with no error, even an invalid one; caps.Thinking=true validates
// BudgetTokens and refuses a non-positive value.
func TestPrepare_Thinking(t *testing.T) {
	req := minimalRequest()
	req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: -5}

	off, err := Prepare("test", llmkit.Capabilities{}, req)
	if err != nil {
		t.Fatalf("Prepare(caps.Thinking=false, invalid budget): %v, want silent drop", err)
	}
	if off.Thinking != nil {
		t.Errorf("Thinking = %+v, want nil when caps.Thinking is false", off.Thinking)
	}

	_, err = Prepare("test", fullCaps(), req)
	wantRefused(t, err)

	req.Thinking.BudgetTokens = 1024
	on, err := Prepare("test", fullCaps(), req)
	if err != nil {
		t.Fatalf("Prepare(valid budget): %v", err)
	}
	if on.Thinking == nil || on.Thinking.BudgetTokens != 1024 {
		t.Errorf("Thinking = %+v, want BudgetTokens 1024", on.Thinking)
	}
}

// TestPrepare_ResponseSchema pins the ResponseSchema gate: dropped to nil
// (no error, even malformed) when caps.StructuredOutput is false; decoded
// and validated the same way tool Parameters are when true, with Name
// carried verbatim from Request.ResponseSchemaName.
func TestPrepare_ResponseSchema(t *testing.T) {
	req := minimalRequest()
	req.ResponseSchema = json.RawMessage(`[1]`) // would be refused if validated
	req.ResponseSchemaName = "custom"

	off, err := Prepare("test", llmkit.Capabilities{}, req)
	if err != nil {
		t.Fatalf("Prepare(StructuredOutput=false, malformed schema): %v, want silent drop", err)
	}
	if off.ResponseSchema != nil {
		t.Errorf("ResponseSchema = %+v, want nil", off.ResponseSchema)
	}

	_, err = Prepare("test", fullCaps(), req)
	wantRefused(t, err)

	req.ResponseSchema = json.RawMessage(`{"type":"object"}`)
	on, err := Prepare("test", fullCaps(), req)
	if err != nil {
		t.Fatalf("Prepare(valid schema): %v", err)
	}
	if on.ResponseSchema == nil || on.ResponseSchema.Name != "custom" {
		t.Errorf("ResponseSchema = %+v, want Name %q", on.ResponseSchema, "custom")
	}
}

// TestRefuse pins the one pre-wire refusal shape: ErrInvalidRequest, the
// given Provider, StatusCode 0, and the cause chained when non-nil.
func TestRefuse(t *testing.T) {
	cause := errors.New("boom")
	err := Refuse("test-provider", "went wrong", cause)
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Refuse returned %T, want *llmkit.APIError", err)
	}
	if apiErr.Provider != "test-provider" || apiErr.Message != "went wrong" || apiErr.StatusCode != 0 {
		t.Errorf("Refuse = %+v, want Provider=test-provider Message=\"went wrong\" StatusCode=0", apiErr)
	}
	if !errors.Is(err, cause) {
		t.Error("Refuse did not chain the cause")
	}
}

// TestApplyOverride_NilOverrideIsCeilingChecked pins the nil-override arm:
// with no override the table profile itself is ceiling-checked, so a table
// reporting a wire-gated field above the ceiling is refused, not returned.
// No shipped table is above its ceiling, so the table here is synthetic.
func TestApplyOverride_NilOverrideIsCeilingChecked(t *testing.T) {
	ceiling := fullCaps()
	ceiling.TopK = false
	table := llmkit.Capabilities{TopK: true}
	_, err := ApplyOverride("test", ceiling, table, nil)
	wantRefused(t, err)
}

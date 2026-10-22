package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

var (
	servedA = llmkit.Identity{Provider: "alpha", Model: "a-1"}
	servedB = llmkit.Identity{Provider: "beta", Model: "b-1"}
)

func servedClient(step scriptStep) *identifiedFakeClient {
	return &identifiedFakeClient{fakeClient: fakeClient{steps: []scriptStep{step}}, identity: servedA}
}

func firstCompletion(t *testing.T, out *Outcome) *llmkit.CompletionEvent {
	t.Helper()
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindCompletion {
			return ev.Completion
		}
	}
	t.Fatal("no Completion event in the record")
	return nil
}

// TestRun_CompletionNamesServed pins I2: the Completion names Served, and
// the JSONL record carries it under lowercase provider/model keys and
// round-trips through LoadJSONL.
func TestRun_CompletionNamesServed(t *testing.T) {
	step := textResp("done", 1, 1)
	step.resp.Served = servedB
	out, err := NewRunner(servedClient(step), nil, "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	ce := firstCompletion(t, out)
	if ce.Provider != servedB.Provider || ce.Model != servedB.Model {
		t.Errorf("tags = %q/%q, want %q/%q (Served)", ce.Provider, ce.Model, servedB.Provider, servedB.Model)
	}

	var buf bytes.Buffer
	if err := out.Transcript.SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	var found bool
	for _, line := range bytes.Split(buf.Bytes(), []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		var rec struct {
			Kind       string `json:"kind"`
			Completion *struct {
				Response struct {
					Served map[string]string `json:"served"`
				} `json:"response"`
			} `json:"completion"`
		}
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatalf("decode %s: %v", line, err)
		}
		if rec.Completion == nil {
			continue
		}
		found = true
		if got := rec.Completion.Response.Served; len(got) != 2 || got["provider"] != "beta" || got["model"] != "b-1" {
			t.Errorf("served = %v, want exactly {provider: beta, model: b-1}", got)
		}
	}
	if !found {
		t.Fatal("no completion line in the JSONL")
	}

	loaded, err := LoadJSONL(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	if got := firstCompletion(t, &Outcome{Transcript: loaded}).Response.Served; got != servedB {
		t.Errorf("loaded Served = %+v, want %+v", got, servedB)
	}
}

// TestRun_ZeroServedKeepsClientIdentity pins I3: a zero Served names the
// client's Identity and the JSONL carries no served key.
func TestRun_ZeroServedKeepsClientIdentity(t *testing.T) {
	out, err := NewRunner(servedClient(textResp("done", 1, 1)), nil, "sys").Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	ce := firstCompletion(t, out)
	if ce.Provider != servedA.Provider || ce.Model != servedA.Model {
		t.Errorf("tags = %q/%q, want %q/%q (client identity)", ce.Provider, ce.Model, servedA.Provider, servedA.Model)
	}
	var buf bytes.Buffer
	if err := out.Transcript.SaveJSONL(&buf); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}
	if strings.Contains(buf.String(), `"served"`) {
		t.Errorf("JSONL carries a served key for a zero Served: %s", buf.String())
	}
}

// TestRun_FailedCallNamesClientIdentity pins I4: a failed call names the
// client's Identity even when the client returned a partial Response
// carrying Served.
func TestRun_FailedCallNamesClientIdentity(t *testing.T) {
	step := scriptStep{resp: llmkit.Response{Text: "half", Served: servedB}, err: errors.New("connection reset")}
	out, err := NewRunner(servedClient(step), nil, "sys").Run(context.Background(), "task")
	if err == nil {
		t.Fatal("Run succeeded; wanted the client error")
	}
	ce := firstCompletion(t, out)
	if ce.Err == "" {
		t.Fatal("completion carries no Err")
	}
	if ce.Provider != servedA.Provider || ce.Model != servedA.Model {
		t.Errorf("tags = %q/%q, want %q/%q (client identity on failure)", ce.Provider, ce.Model, servedA.Provider, servedA.Model)
	}
}

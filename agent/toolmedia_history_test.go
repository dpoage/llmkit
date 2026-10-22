package agent

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/provider"
)

// tmFinalOnly answers every request with w's final turn.
func tmFinalOnly(w *tmWire) *tmWire {
	w.tool = w.final
	return w
}

// tmAnthropicPNGRun is a finished anthropic run whose history holds a png tool result.
func tmAnthropicPNGRun(t *testing.T) *Outcome {
	t.Helper()
	out, run, _ := tmRun(t, tmAnthropicWire(), []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", tmPNG)})
	if run.Degraded != 0 || !slices.ContainsFunc(toolMessage(t, out.Messages).Content, isMedia) {
		t.Fatalf("anthropic run did not keep the png in history: %+v", out.Messages)
	}
	return out
}

func tmCloneMessages(msgs []llmkit.Message) []llmkit.Message {
	out := make([]llmkit.Message, len(msgs))
	for i, m := range msgs {
		m.Content = slices.Clone(m.Content)
		for j := range m.Content {
			m.Content[j].Data = bytes.Clone(m.Content[j].Data)
			m.Content[j].Raw = bytes.Clone(m.Content[j].Raw)
		}
		m.ToolCalls = slices.Clone(m.ToolCalls)
		for j := range m.ToolCalls {
			m.ToolCalls[j].Arguments = bytes.Clone(m.ToolCalls[j].Arguments)
		}
		out[i] = m
	}
	return out
}

// tmWireToolMessage returns the JSON of the body's first message whose role is role.
func tmWireToolMessage(t *testing.T, body map[string]any, role string) string {
	t.Helper()
	list, _ := body["messages"].([]any)
	for _, raw := range list {
		if m, _ := raw.(map[string]any); m["role"] == role {
			j, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			if role != "user" || strings.Contains(string(j), "tool_result") {
				return string(j)
			}
		}
	}
	t.Fatalf("wire body has no %s tool message: %v", role, body)
	return ""
}

func tmCompletions(evs []llmkit.Event) []llmkit.CompletionEvent {
	var out []llmkit.CompletionEvent
	for _, e := range evs {
		if e.Kind == llmkit.KindCompletion {
			out = append(out, *e.Completion)
		}
	}
	return out
}

func TestToolMedia_ContinuedHistoryDegradesPerProfile(t *testing.T) {
	data := base64.StdEncoding.EncodeToString(tmPNG)
	wantDegraded := []llmkit.Block{llmkit.Text("cap"), llmkit.Text(tmImageMarker)}
	cases := []struct {
		name    string
		wire    *tmWire
		role    string
		degrade bool
	}{
		{"openai", tmFinalOnly(tmOpenAIWire(provider.TypeOpenAI, "gpt-4o")), "tool", true},
		{"anthropic", tmFinalOnly(tmAnthropicWire()), "user", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			prev := tmAnthropicPNGRun(t)
			before := tmCloneMessages(prev.Messages)
			client := c.wire.client(t)

			out, err := NewRunner(client, nil, "sys").Run(context.Background(), "again", Continue(prev), WithRunID("tmc"))
			if err != nil {
				t.Fatalf("continued Run: %v", err)
			}
			if len(c.wire.bodies) != 1 {
				t.Fatalf("wire hits = %d, want 1", len(c.wire.bodies))
			}
			wire := tmWireToolMessage(t, c.wire.bodies[0], c.role)
			if c.degrade {
				if !strings.Contains(wire, "cap"+tmImageMarker) || strings.Contains(wire, data) || strings.Contains(wire, "image_url") {
					t.Errorf("wire tool message = %s, want %q and no image", wire, "cap"+tmImageMarker)
				}
			} else if !strings.Contains(wire, `"media_type":"image/png"`) || !strings.Contains(wire, data) || strings.Contains(wire, "omitted") {
				t.Errorf("wire tool message = %s, want the png verbatim", wire)
			}

			if !reflect.DeepEqual(prev.Messages, before) {
				t.Errorf("prev.Messages changed by the continued run:\n got %+v\nwant %+v", prev.Messages, before)
			}
			if !reflect.DeepEqual(out.Messages[:len(before)], before) {
				t.Errorf("continued Outcome.Messages rewrote the carried history:\n got %+v\nwant %+v", out.Messages[:len(before)], before)
			}

			comps := tmCompletions(out.Transcript.Record)
			if len(comps) != 1 {
				t.Fatalf("completion events = %d, want 1", len(comps))
			}
			wantContent := toolMessage(t, before).Content
			if c.degrade {
				wantContent = wantDegraded
			}
			if got := toolMessage(t, comps[0].Request.Messages).Content; !reflect.DeepEqual(got, wantContent) {
				t.Errorf("recorded request tool result = %+v, want %+v", got, wantContent)
			}

			var jsonl bytes.Buffer
			if err := out.Transcript.SaveJSONL(&jsonl); err != nil {
				t.Fatal(err)
			}
			loaded, err := LoadJSONL(&jsonl)
			if err != nil {
				t.Fatal(err)
			}
			for src, tr := range map[string]*Transcript{"memory": out.Transcript, "jsonl": loaded} {
				t.Run(src, func(t *testing.T) {
					rc, err := NewReplayClient(tr, "tmc", client.Capabilities())
					if err != nil {
						t.Fatal(err)
					}
					got, err := NewRunner(rc, nil, "sys").Run(context.Background(), "again", Continue(prev), WithRunID("tmc"))
					if err != nil || rc.Err() != nil {
						t.Fatalf("replay Run = %v, rc.Err = %v, want both nil", err, rc.Err())
					}
					if !reflect.DeepEqual(got.Messages, out.Messages) {
						t.Errorf("replayed history = %+v\nrecorded history = %+v", got.Messages, out.Messages)
					}
					gc := tmCompletions(got.Transcript.Record)
					if len(gc) != len(comps) {
						t.Fatalf("replayed completion events = %d, want %d", len(gc), len(comps))
					}
					for i := range comps {
						if !reflect.DeepEqual(gc[i].Request, comps[i].Request) || gc[i].Response.Text != comps[i].Response.Text {
							t.Errorf("replayed completion %d = %+v\nrecorded = %+v", i, gc[i], comps[i])
						}
					}
				})
			}
		})
	}
}

func TestToolMedia_PortableTypesReachMediaWires(t *testing.T) {
	blocks := map[string]llmkit.Block{
		"png":  llmkit.Image("image/png", tmPNG),
		"jpeg": llmkit.Image("image/jpeg", []byte("\xff\xd8\xff\xe0jpeg")),
		"webp": llmkit.Image("image/webp", []byte("RIFF\x00\x00\x00\x00WEBPVP8 ")),
		"pdf":  llmkit.Document("application/pdf", "t", []byte("%PDF-1.4 x")),
	}
	wires := map[string]struct {
		wire func() *tmWire
		key  string
		mime string
	}{
		"gemini-3":  {tmGoogleWire, "contents", `"mimeType":"%s"`},
		"anthropic": {tmAnthropicWire, "messages", `"media_type":"%s"`},
	}
	for wname, wc := range wires {
		for bname, b := range blocks {
			t.Run(wname+"/"+bname, func(t *testing.T) {
				w := wc.wire()
				_, run, _ := tmRun(t, w, []llmkit.Block{llmkit.Text("cap"), b})
				if run.Degraded != 0 {
					t.Errorf("Degraded = %d, want 0", run.Degraded)
				}
				wire := tmLastWireMessage(t, w.bodies[1], wc.key)
				if !strings.Contains(wire, fmt.Sprintf(wc.mime, b.MediaType)) ||
					!strings.Contains(wire, base64.StdEncoding.EncodeToString(b.Data)) || strings.Contains(wire, "omitted") {
					t.Errorf("wire = %s, want the %s bytes verbatim", wire, b.MediaType)
				}
			})
		}
	}
}

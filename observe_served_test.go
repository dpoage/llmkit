package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

var (
	servedA = Identity{Provider: "alpha", Model: "a-1"}
	servedB = Identity{Provider: "beta", Model: "b-1"}
)

func servedResponse(served Identity) Response {
	resp := okResponse()
	resp.Served = served
	return resp
}

// TestObserve_CompletionNamesServed pins I1: a successful call whose
// Response sets Served is attributed to Served, on Complete and on Stream
// (native and synthesized).
func TestObserve_CompletionNamesServed(t *testing.T) {
	cases := []struct {
		name string
		run  func(Client, Observer) error
		mk   func() Client
	}{
		{"complete", func(c Client, o Observer) error {
			_, err := Observe(c, o).Complete(context.Background(), simpleRequest())
			return err
		}, func() Client {
			return withIdentity{Client: &fakeClient{responses: []Response{servedResponse(servedB)}}, id: servedA}
		}},
		{"stream native", func(c Client, o Observer) error {
			_, err := Stream(context.Background(), Observe(c, o), simpleRequest(), func(Delta) error { return nil })
			return err
		}, func() Client {
			return withIdentity{Client: &scriptedStreamClient{resp: servedResponse(servedB)}, id: servedA}
		}},
		{"stream synthesized", func(c Client, o Observer) error {
			_, err := Stream(context.Background(), Observe(c, o), simpleRequest(), func(Delta) error { return nil })
			return err
		}, func() Client {
			return withIdentity{Client: &fakeClient{responses: []Response{servedResponse(servedB)}}, id: servedA}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obs := &captureObserver{}
			if err := tc.run(tc.mk(), obs); err != nil {
				t.Fatalf("call: %v", err)
			}
			events := obs.snapshot()
			if len(events) != 1 || events[0].Completion == nil {
				t.Fatalf("events = %+v, want exactly one Completion", events)
			}
			ce := events[0].Completion
			if ce.Provider != servedB.Provider || ce.Model != servedB.Model {
				t.Errorf("tags = %q/%q, want %q/%q (Served)", ce.Provider, ce.Model, servedB.Provider, servedB.Model)
			}
			if ce.Response.Served != servedB {
				t.Errorf("event Response.Served = %+v, want %+v", ce.Response.Served, servedB)
			}
		})
	}
}

// TestObserve_ZeroServedKeepsClientIdentity pins I3: a Response with zero
// Served is attributed to the client's Identity and serializes no served key.
func TestObserve_ZeroServedKeepsClientIdentity(t *testing.T) {
	obs := &captureObserver{}
	client := Observe(withIdentity{Client: &fakeClient{responses: []Response{okResponse()}}, id: servedA}, obs)
	if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	ce := obs.snapshot()[0].Completion
	if ce.Provider != servedA.Provider || ce.Model != servedA.Model {
		t.Errorf("tags = %q/%q, want %q/%q (client identity)", ce.Provider, ce.Model, servedA.Provider, servedA.Model)
	}
	raw, err := json.Marshal(ce)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(raw), `"served"`) {
		t.Errorf("completion JSON carries a served key for a zero Served: %s", raw)
	}
}

// TestObserve_FailedCallNamesClientIdentity pins I4: a failed call keeps the
// client's Identity even when the inner client returned a partial response
// carrying Served, on Complete and on Stream.
func TestObserve_FailedCallNamesClientIdentity(t *testing.T) {
	boom := errors.New("boom")
	cases := []struct {
		name string
		mk   func() Client
	}{
		{"complete", func() Client {
			return withIdentity{Client: &fakeClient{responses: []Response{servedResponse(servedB)}, errs: []error{boom}}, id: servedA}
		}},
		{"stream", func() Client {
			return withIdentity{Client: &scriptedStreamClient{resp: servedResponse(servedB), err: boom}, id: servedA}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obs := &captureObserver{}
			_, err := Stream(context.Background(), Observe(tc.mk(), obs), simpleRequest(), func(Delta) error { return nil })
			if !errors.Is(err, boom) {
				t.Fatalf("err = %v, want boom", err)
			}
			events := obs.snapshot()
			if len(events) != 1 || events[0].Completion == nil {
				t.Fatalf("events = %+v, want exactly one Completion", events)
			}
			ce := events[0].Completion
			if ce.Provider != servedA.Provider || ce.Model != servedA.Model {
				t.Errorf("tags = %q/%q, want %q/%q (client identity on failure)", ce.Provider, ce.Model, servedA.Provider, servedA.Model)
			}
		})
	}
}

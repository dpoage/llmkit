package provider_test

// Hermetic replay of the recorded live fixtures: for every file under
// provider/testdata/**, an httptest server serves the recorded exchanges in
// order and the recorded llmkit.Requests are re-issued through provider.New
// (the production construction path) with the fixture's pinned capability
// profile. The adapter's final normalized outcome must equal the recorded
// one — Text, ToolCalls, StopReason, Usage — and error cases must still
// errors.Is the recorded sentinel. Editing a fixture's recorded response text
// fails this test; so does adapter wire drift on the request side.
//
// No tag, no network, no credentials: this runs in plain `go test ./...`.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/livetest"
	"github.com/dpoage/llmkit/provider"
)

func TestFixtureReplay(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("testdata", "*", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Skip("no recorded fixtures yet — run the live suite with -update")
	}
	for _, p := range matches {
		p := p
		name := strings.TrimSuffix(filepath.Base(p), ".json")
		t.Run(path.Base(filepath.Dir(p))+"/"+name, func(t *testing.T) {
			f, err := livetest.ReadFixture(p)
			if err != nil {
				t.Fatal(err)
			}
			replayFixture(t, f)
		})
	}
}

// replayFixture rebuilds the vendor from the fixture and re-issues the
// recorded requests against it.
func replayFixture(t *testing.T, f *livetest.Fixture) {
	lane, ok := livetest.LaneByName(f.Lane)
	if !ok {
		t.Fatalf("fixture lane %q is not a known lane", f.Lane)
	}
	if len(f.Exchanges) != len(f.Inputs) {
		t.Fatalf("fixture is misaligned: %d inputs, %d exchanges", len(f.Inputs), len(f.Exchanges))
	}
	if len(f.Exchanges) == 0 {
		t.Fatal("fixture records no exchanges")
	}

	// Pretty-printed fixtures re-indent embedded RawMessage bytes (tool
	// arguments, schemas); compact them back so the re-issued wire bodies
	// match the recorded ones.
	livetest.NormalizeFixture(f)
	var mu sync.Mutex
	served := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		i := served
		served++
		mu.Unlock()

		if i >= len(f.Exchanges) {
			t.Errorf("replay step %d: unexpected extra request (fixture recorded %d exchanges)", i+1, len(f.Exchanges))
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		ex := f.Exchanges[i]
		if !sameEndpoint(r.URL.Path, ex.Request.Path) {
			t.Errorf("replay step %d: request path %q, fixture recorded %q", i+1, r.URL.Path, ex.Request.Path)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("replay step %d: read request body: %v", i+1, err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if !sameJSON(body, []byte(ex.Request.Body)) {
			t.Errorf("replay step %d: re-issued request body drifted from the recorded wire request\nrecorded: %s\nre-issued: %s",
				i+1, ex.Request.Body, body)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		for k, v := range ex.Response.Headers {
			if k == "content-length" {
				// Recomputed by net/http for the served body; echoing the
				// recorded length would truncate any body whose size differs.
				continue
			}
			w.Header().Set(k, v)
		}
		w.WriteHeader(ex.Response.Status)
		_, _ = w.Write([]byte(ex.Response.Body))
	}))
	defer srv.Close()

	// provider.New dispatches on the recorded lane type; the pinned
	// capability profile keeps wire behavior identical to record time even
	// if adapter tables or the operator's CAPS env change.
	spec := provider.Spec{
		Type:    lane.Type,
		Model:   f.Model,
		BaseURL: srv.URL,
		Secret:  "llmkit-replay-placeholder",
		Capabilities: func(llmkit.Capabilities) llmkit.Capabilities {
			return f.Capabilities
		},
	}
	cl, err := provider.New(context.Background(), spec, provider.Options{})
	if err != nil {
		t.Fatalf("provider.New: %v", err)
	}

	var resp llmkit.Response
	for i, in := range f.Inputs {
		resp, err = cl.Complete(context.Background(), in)
		if err != nil && i < len(f.Inputs)-1 {
			t.Fatalf("replay step %d failed before the final exchange: %v", i+1, err)
		}
	}

	n := f.Normalized
	if n.ErrKind != "" {
		want := livetest.SentinelByKind(n.ErrKind)
		if want == nil {
			t.Fatalf("fixture records unknown error kind %q", n.ErrKind)
		}
		if !errors.Is(err, want) {
			t.Fatalf("replay: recorded error kind %s, re-issued completion returned %v", n.ErrKind, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("replay: recorded success but re-issued completion returned %v", err)
	}
	if resp.Text != n.Text {
		t.Errorf("replay text drifted:\nrecorded: %q\nre-issued: %q", n.Text, resp.Text)
	}
	if resp.StopReason != n.StopReason {
		t.Errorf("replay stop reason = %q, recorded %q", resp.StopReason, n.StopReason)
	}
	if resp.Usage != n.Usage {
		t.Errorf("replay usage drifted: recorded %+v, re-issued %+v", n.Usage, resp.Usage)
	}
	if !reflect.DeepEqual(resp.ToolCalls, n.ToolCalls) {
		t.Errorf("replay tool calls drifted:\nrecorded: %+v\nre-issued: %+v", n.ToolCalls, resp.ToolCalls)
	}
}

// sameEndpoint compares request paths. The replay host has no vendor base
// path (the SDK's /v1 style prefix is part of the record-time base URL), so
// matching is on the final path segment plus the call sequence.
func sameEndpoint(got, recorded string) bool {
	if got == recorded {
		return true
	}
	recorded = strings.SplitN(recorded, "?", 2)[0]
	return path.Base(got) == path.Base(recorded)
}

// sameJSON reports semantic JSON equality: whitespace and key order do not
// matter, content does.
func sameJSON(a, b []byte) bool {
	var va, vb any
	if json.Unmarshal(a, &va) != nil {
		return false
	}
	if json.Unmarshal(b, &vb) != nil {
		return false
	}
	return reflect.DeepEqual(va, vb)
}

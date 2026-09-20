package decide_test

// Hermetic replay of the recorded decide fixtures: for every file under
// decide/testdata, an httptest server serves the recorded exchanges in
// order and the recorded Ask is re-issued through decide.New. The
// re-issued request must equal the recorded one — method, path, and body
// compared as parsed JSON; headers excluded — and the outcome must equal
// the recording: the whole normalized decide.Response, or the sentinel
// error kind. Editing a recorded probability fails this test; so does
// client wire drift on the request side.
//
// No tag, no network, no credentials: runs in plain `go test ./...`.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/internal/livetest"
)

func TestDecideFixtureReplay(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("testdata", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Skip("no recorded fixtures yet — run the live suite with -update")
	}
	for _, p := range matches {
		p := p
		name := strings.TrimSuffix(filepath.Base(p), ".json")
		t.Run(name, func(t *testing.T) {
			f, err := livetest.ReadDecideFixture(p)
			if err != nil {
				t.Fatal(err)
			}
			replayDecideFixture(t, f)
		})
	}
}

// replayDecideFixture rebuilds the vendor from the fixture and re-issues
// the recorded Ask against it.
func replayDecideFixture(t *testing.T, f *livetest.DecideFixture) {
	if len(f.Exchanges) == 0 {
		t.Fatal("fixture records no exchanges")
	}
	livetest.NormalizeDecideFixture(f)
	questions, err := livetest.BuildDecideQuestions(f.Questions)
	if err != nil {
		t.Fatal(err)
	}

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
		// An Ask's request side is fully caller-determined (no model-generated ids
		// are ever echoed back into a later request), so method, path, and body
		// are all gated.
		if ex.Request.Method != "" && r.Method != ex.Request.Method {
			t.Errorf("replay step %d: request method %q, fixture recorded %q", i+1, r.Method, ex.Request.Method)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if r.URL.Path != ex.Request.Path {
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
				// Recomputed by net/http for the served body; echoing the recorded length
				// would truncate any body whose size differs.
				continue
			}
			w.Header().Set(k, v)
		}
		w.WriteHeader(ex.Response.Status)
		_, _ = w.Write([]byte(ex.Response.Body))
	}))
	defer srv.Close()

	cl, err := decide.New(decide.Config{
		APIKey:     "llmkit-replay-placeholder",
		Model:      f.Model,
		BaseURL:    srv.URL,
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatalf("decide.New: %v", err)
	}
	resp, err := cl.Ask(context.Background(), f.State, questions)

	if n := f.Normalized; n.ErrKind != "" {
		want := livetest.SentinelByKind(n.ErrKind)
		if want == nil {
			t.Fatalf("fixture records unknown error kind %q", n.ErrKind)
		}
		if !errors.Is(err, want) {
			t.Fatalf("replay: recorded error kind %s, re-issued Ask returned %v", n.ErrKind, err)
		}
		return
	}
	if f.Normalized.Response == nil {
		t.Fatal("fixture records neither an error kind nor a response")
	}
	if err != nil {
		t.Fatalf("replay: recorded success but re-issued Ask returned %v", err)
	}
	if !reflect.DeepEqual(resp, *f.Normalized.Response) {
		t.Errorf("replay response drifted:\nrecorded: %+v\nre-issued: %+v", *f.Normalized.Response, resp)
	}
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

package adapter

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dpoage/llmkit"
)

func TestEstimateUsage_CeilBytesOverFour(t *testing.T) {
	cases := []struct {
		name        string
		text        string
		calls       []llmkit.ToolCall
		wantOut     int64
		wantInBytes int64
		wantIn      int64
		nilWire     bool
	}{
		{"empty", "", nil, 0, 0, 0, false},
		{"one byte", "a", nil, 1, 1, 1, false},
		{"exact multiple", "abcd", nil, 1, 4, 1, false},
		{"one over", "abcde", nil, 2, 5, 2, false},
		{"text plus call", "ab", []llmkit.ToolCall{{Name: "fn", Arguments: []byte(`{"k":1}`)}}, 3, 8, 2, false},
		{"two calls", "", []llmkit.ToolCall{{Name: "a", Arguments: []byte("{}")}, {Name: "b", Arguments: []byte("{}")}}, 2, 0, 0, false},
		{"nil wire", "abcde", nil, 2, 0, 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var w *Wire
			if !c.nilWire {
				w = &Wire{}
				w.body.Store(c.wantInBytes)
			}
			got := EstimateUsage(w, llmkit.Response{Text: c.text, ToolCalls: c.calls})
			want := llmkit.Usage{InputTokens: c.wantIn, OutputTokens: c.wantOut, Estimated: true}
			if got != want {
				t.Errorf("EstimateUsage = %+v, want %+v", got, want)
			}
		})
	}
}

func TestWire_RequestBytesCountsLastRequestBody(t *testing.T) {
	var redirected bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if r.URL.Path == "/start" {
			redirected = true
			http.Redirect(w, r, "/final", http.StatusTemporaryRedirect)
			return
		}
		_, _ = io.WriteString(w, "ok")
	}))
	defer srv.Close()

	ctx, w := WithWire(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/start", strings.NewReader("0123456789"))
	resp, err := WireClient(nil).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if !redirected || w.Requests() != 2 {
		t.Fatalf("redirected = %v, requests = %d, want the redirect followed", redirected, w.Requests())
	}
	if got := w.RequestBytes(); got != 10 {
		t.Errorf("RequestBytes = %d, want 10 (the resent body, not 20)", got)
	}
	if resp.Request.URL.Path != "/final" {
		t.Errorf("resp.Request.URL = %v, want the request the client sent last", resp.Request.URL)
	}
}

func TestWire_RequestBytesZeroWithoutBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	ctx, w := WithWire(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	resp, err := WireClient(nil).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if got := w.RequestBytes(); got != 0 {
		t.Errorf("RequestBytes = %d, want 0", got)
	}
}

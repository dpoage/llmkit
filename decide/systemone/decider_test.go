package systemone

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

func askVia(d decide.Decider, qs decide.Questions) (decide.Response, error) {
	return d.Ask(context.Background(), "state", qs)
}

func TestAsk_ChoiceOutsideOptionsRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusOK, `{"model":"m","answers":{"pick":{"type":"choice","choice":"green"}},"usage":{}}`)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	qs := decide.Questions{"pick": decide.Choice{Instructions: "Pick.", Options: map[string]any{"blue": nil, "red": nil}}}
	resp, err := askVia(c, qs)
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) {
		t.Fatalf("err = %v, want ErrServer *APIError", err)
	}
	if apiErr.StatusCode != http.StatusOK {
		t.Errorf("StatusCode = %d, want 200", apiErr.StatusCode)
	}
	if !strings.Contains(apiErr.Message, `"pick"`) {
		t.Errorf("Message %q does not name id \"pick\"", apiErr.Message)
	}
	if !reflect.DeepEqual(resp, decide.Response{}) {
		t.Errorf("resp = %+v, want zero Response", resp)
	}
	if got := hits.Load(); got != int32(fastRetry.MaxAttempts) {
		t.Errorf("hits = %d, want %d (retried like any ErrServer)", got, fastRetry.MaxAttempts)
	}
}

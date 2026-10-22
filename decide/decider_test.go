package decide

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
)

type fakeDecider struct{ resp Response }

func (f fakeDecider) Ask(context.Context, any, Questions) (Response, error) { return f.resp, nil }

func askVia(d Decider, qs Questions) (Response, error) {
	return d.Ask(context.Background(), "state", qs)
}

func TestDecider_FakeAcceptedWhereClientWas(t *testing.T) {
	got, err := askVia(fakeDecider{resp: checkResponse()}, checkQuestions())
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if err := Check(checkQuestions(), got); err != nil {
		t.Errorf("Check: %v", err)
	}
}

// TestAsk_ChoiceOutsideOptionsRetried pins a 200 answering a Choice with an option that was not asked is an ErrServer@200, retried like any ErrServer.
func TestAsk_ChoiceOutsideOptionsRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusOK, `{"model":"m","answers":{"pick":{"type":"choice","choice":"green"}},"usage":{}}`)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL, fastRetry)
	qs := Questions{"pick": Choice{Instructions: "Pick.", Options: map[string]any{"blue": nil, "red": nil}}}
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
	if !reflect.DeepEqual(resp, Response{}) {
		t.Errorf("resp = %+v, want zero Response", resp)
	}
	if got := hits.Load(); got != int32(fastRetry.MaxAttempts) {
		t.Errorf("hits = %d, want %d (retried like any ErrServer)", got, fastRetry.MaxAttempts)
	}
}

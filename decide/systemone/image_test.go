package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

func TestAsk_ImageIsRefusedNamingConfigImagesWithoutAHit(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		writeStatus(w, http.StatusOK, `{}`)
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL, fastRetry)

	img := json.RawMessage(`{"kind":"image","data":"AQID","media_type":"image/png"}`)
	tests := []struct {
		name  string
		state any
		qs    decide.Questions
		want  string
	}{
		{"state", map[string]any{"photo": img}, decide.Questions{"q": decide.Noul{Instructions: "i"}},
			`state["photo"]: image in state is not supported by this backend; set Config.Images to send images`},
		{"instructions", "s", decide.Questions{"q": decide.Noul{Instructions: []any{img}}},
			`questions["q"].instructions[0]: image in instructions is not supported by this backend; set Config.Images to send images`},
		{"criteria", "s", decide.Questions{"q": decide.Choice{Instructions: "i", Options: map[string]any{"a": img}}},
			`questions["q"].options["a"]: image in criteria is not supported by this backend; set Config.Images to send images`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := c.Ask(context.Background(), tt.state, tt.qs)
			var ae *llmkit.APIError
			if !errors.As(err, &ae) {
				t.Fatalf("err = %v, want *llmkit.APIError", err)
			}
			if ae.Kind != llmkit.ErrInvalidRequest || ae.Provider != "systemone" || ae.Message != tt.want {
				t.Errorf("err = %+v, want ErrInvalidRequest, Provider systemone, Message %q", ae, tt.want)
			}
			if !errors.Is(err, decide.ErrImageUnsupported) || !errors.Is(err, llmkit.ErrInvalidRequest) {
				t.Errorf("err = %v, want both decide.ErrImageUnsupported and llmkit.ErrInvalidRequest", err)
			}
		})
	}
	if n := hits.Load(); n != 0 {
		t.Errorf("HTTP hits = %d, want 0", n)
	}
}

func TestAsk_InvalidImageIsRefusedWithoutTheConfigImagesHint(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	defer srv.Close()
	c := newTestClient(t, srv.URL, fastRetry)

	_, err := c.Ask(context.Background(), map[string]any{"p": json.RawMessage(`{"kind":"image","url":"https://example.com/a.png"}`)}, decide.Questions{"q": decide.Noul{Instructions: "i"}})
	var ae *llmkit.APIError
	if !errors.As(err, &ae) || ae.Provider != "systemone" || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("err = %v, want a systemone ErrInvalidRequest", err)
	}
	if want := `state["p"]: invalid image: url is set; an image carries only data and media_type`; ae.Message != want {
		t.Errorf("Message = %q, want %q", ae.Message, want)
	}
	if errors.Is(err, decide.ErrImageUnsupported) || hits.Load() != 0 {
		t.Errorf("ErrImageUnsupported = %v, hits = %d, want false and 0", errors.Is(err, decide.ErrImageUnsupported), hits.Load())
	}
}

func TestClient_ImageSupportIsTheZeroValue(t *testing.T) {
	c := newTestClient(t, "http://127.0.0.1:1", fastRetry)
	var s decide.ImageSupporter = c
	if got := s.ImageSupport(); got.Placements != 0 || len(got.MediaTypes) != 0 || got.Label != decide.LabelRef {
		t.Errorf("ImageSupport() = %+v, want the zero value", got)
	}
}

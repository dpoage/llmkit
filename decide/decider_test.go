package decide

import (
	"context"
	"testing"
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

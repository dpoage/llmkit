package llmkit

import (
	"encoding/json"
	"testing"
)

func TestUsage_AddOrsEstimatedBothOrders(t *testing.T) {
	cases := []struct{ u, v, want bool }{
		{false, false, false},
		{true, false, true},
		{false, true, true},
		{true, true, true},
	}
	for _, c := range cases {
		got := Usage{InputTokens: 1, Estimated: c.u}.Add(Usage{OutputTokens: 2, Estimated: c.v})
		if got.Estimated != c.want || got.InputTokens != 1 || got.OutputTokens != 2 {
			t.Errorf("Add(Estimated %v, %v) = %+v, want Estimated %v with counts summed", c.u, c.v, got, c.want)
		}
	}
}

func TestUsage_EstimatedJSON(t *testing.T) {
	b, _ := json.Marshal(Usage{InputTokens: 1})
	if string(b) != `{"input_tokens":1}` {
		t.Errorf("unestimated Usage JSON = %s, want estimated omitted", b)
	}
	b, _ = json.Marshal(Usage{Estimated: true})
	if string(b) != `{"estimated":true}` {
		t.Errorf("estimated Usage JSON = %s", b)
	}
}

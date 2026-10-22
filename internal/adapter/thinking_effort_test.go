package adapter

import (
	"testing"

	"github.com/dpoage/llmkit"
)

// TestPrepare_ThinkingSurvivors pins what Prepared.Thinking keeps for every
// (Thinking, ReasoningEffort) profile class and ThinkingConfig shape.
func TestPrepare_ThinkingSurvivors(t *testing.T) {
	profiles := []struct {
		name             string
		thinking, effort bool
	}{
		{"neither", false, false},
		{"thinking", true, false},
		{"effort", false, true},
		{"both", true, true},
	}
	cfgs := []struct {
		name string
		cfg  llmkit.ThinkingConfig
		// want per profile in the order above; nil = no thinking survives.
		want [4]*llmkit.ThinkingConfig
	}{
		{"budget", llmkit.ThinkingConfig{BudgetTokens: 7}, [4]*llmkit.ThinkingConfig{
			nil, {BudgetTokens: 7}, nil, {BudgetTokens: 7}}},
		{"effort", llmkit.ThinkingConfig{Effort: llmkit.EffortMedium}, [4]*llmkit.ThinkingConfig{
			nil, nil, {Effort: llmkit.EffortMedium}, {Effort: llmkit.EffortMedium}}},
		{"both", llmkit.ThinkingConfig{BudgetTokens: 7, Effort: llmkit.EffortLow}, [4]*llmkit.ThinkingConfig{
			nil, {BudgetTokens: 7}, {Effort: llmkit.EffortLow}, {BudgetTokens: 7, Effort: llmkit.EffortLow}}},
	}
	for _, c := range cfgs {
		for i, p := range profiles {
			t.Run(c.name+"/"+p.name, func(t *testing.T) {
				req := minimalRequest()
				cfg := c.cfg
				req.Thinking = &cfg
				got, err := Prepare("test", llmkit.Capabilities{Thinking: p.thinking, ReasoningEffort: p.effort}, req)
				if err != nil {
					t.Fatalf("Prepare: %v", err)
				}
				switch want := c.want[i]; {
				case want == nil && got.Thinking != nil:
					t.Errorf("Thinking = %+v, want nil", *got.Thinking)
				case want != nil && (got.Thinking == nil || *got.Thinking != *want):
					t.Errorf("Thinking = %+v, want %+v", got.Thinking, *want)
				}
			})
		}
	}
}

// TestPrepare_ThinkingEffortEnum pins the closed set: an Effort outside it is
// refused whatever the profile.
func TestPrepare_ThinkingEffortEnum(t *testing.T) {
	for _, caps := range []llmkit.Capabilities{{}, {Thinking: true}, {ReasoningEffort: true}, fullCaps()} {
		for _, e := range []llmkit.ReasoningEffort{"max", "LOW", "minimal", " low"} {
			req := minimalRequest()
			req.Thinking = &llmkit.ThinkingConfig{BudgetTokens: 5, Effort: e}
			_, err := Prepare("test", caps, req)
			wantRefused(t, err)
		}
	}
}

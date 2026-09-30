package adapter

import (
	"reflect"
	"sort"
	"testing"

	"github.com/dpoage/llmkit"
)

// advisoryCapabilities is the explicit list of Capabilities bool fields no
// adapter puts on the wire, so ApplyOverride never ceiling-checks them:
// ParallelToolCalls is a decorator switch (provider.New installs the
// tool-call serializer when false), PromptCaching, Images and Documents are
// advisory (callers read them; adapters pass the blocks through).
// ContextWindow is advisory too but is an int, so it is not in a bool set.
var advisoryCapabilities = []string{"ParallelToolCalls", "PromptCaching", "Images", "Documents"}

// TestWireGatesCoverCapabilityBools ties the wireGate table to the
// Capabilities field set: every bool field of llmkit.Capabilities must be
// exactly one of wire-gated (a wireGates row) or listed advisory/decorator.
// A new bool field in neither set, a wireGates row for no field, and a
// field claimed by both all fail — so a capability cannot be added without
// deciding whether an override may claim it above an adapter's ceiling.
func TestWireGatesCoverCapabilityBools(t *testing.T) {
	var bools []string
	rt := reflect.TypeOf(llmkit.Capabilities{})
	for i := range rt.NumField() {
		if f := rt.Field(i); f.Type.Kind() == reflect.Bool {
			bools = append(bools, f.Name)
		}
	}

	gated := map[string]bool{}
	for _, g := range wireGates(llmkit.Capabilities{}, llmkit.Capabilities{}) {
		gated[g.name] = true
	}
	advisory := map[string]bool{}
	for _, n := range advisoryCapabilities {
		advisory[n] = true
	}

	for n := range gated {
		if advisory[n] {
			t.Errorf("Capabilities.%s is both wire-gated and on the advisory list", n)
		}
	}
	union := map[string]bool{}
	for n := range gated {
		union[n] = true
	}
	for n := range advisory {
		union[n] = true
	}
	fields := map[string]bool{}
	for _, n := range bools {
		fields[n] = true
		if !union[n] {
			t.Errorf("Capabilities.%s is a bool in neither wireGates nor advisoryCapabilities: add a wireGates row (an adapter can put it on the wire) or list it advisory", n)
		}
	}
	var stale []string
	for n := range union {
		if !fields[n] {
			stale = append(stale, n)
		}
	}
	sort.Strings(stale)
	if len(stale) > 0 {
		t.Errorf("wireGates/advisoryCapabilities name %v, which are not bool fields of Capabilities", stale)
	}
}

//go:build live

// Live acceptance suite for the decide package: TypeSafe's Jev (System One)
// decision API through decide.New. Every case asserts structural invariants
// only (ranges, argmax consistency, legend alignment, usage accounting); exact
// probabilities are the vendor's business. Fixtures are recorded under
// -update into decide/testdata/<case>.json and replayed hermetically by
// fixture_replay_test.go.
//
// Environment (see docs/testing.md):
//
//	LLMKIT_LIVE_TYPESAFE_API_KEY   (required)
//	LLMKIT_LIVE_TYPESAFE_MODEL     (optional, default jev-latest)
//	LLMKIT_LIVE_TYPESAFE_BASE_URL  (optional gateway override)
//
// A keyless lane skips at the lane level, naming its variable. TypeSafe
// bills input tokens only; the whole lane costs a handful. Run with
//
//	go test -tags live -count=1 ./decide/ -v
//
// adding -update to re-record fixtures. Keys are never logged; raw body
// logs go through livetest's redacting logger.
package decide_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	llmkit "github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/internal/livetest"
)

func TestMain(m *testing.M) {
	code := m.Run()
	livetest.DefaultTally().PrintSummary()
	os.Exit(code)
}

// probSumTol is the tolerance for the assertion that an n-way probability
// distribution sums to 1. The vendor (System One) reports probabilities
// rounded to 2 decimals — e.g. an observed [0 0.01 0.93 0.05] scores 0.99 —
// so each term contributes up to 0.005 of error and the sum can land within
// n*0.005 of 1; 1e-9 covers float accumulation. A 4-way score therefore
// admits |sum-1| <= 0.02, while sums of 0.9 or 1.1 still fail.
func probSumTol(n int) float64 {
	return 0.005*float64(n) + 1e-9
}

// badKey is the intentionally wrong credential for the ErrAuth case; no
// sk- prefix so it cannot collide with the fixture writer's secret refusals.
const badKey = "llmkit-live-intentionally-invalid-key"

// badModel is the intentionally unknown model for the ErrInvalidRequest case.
const badModel = "llmkit-no-such-model-x9"

// ticketState is the prose ticket summary used as the state across cases.
const ticketState = "Customer reports a paper jam error on a networked office printer. " +
	"They removed the jammed sheet and restarted the device, but the error returns within " +
	"minutes and the duplex unit sounds like it is grinding."

// ticketLevels is the score question's ordered legend, lowest to highest.
var ticketLevels = []any{
	"Cosmetic or minor inconvenience; no user impact.",
	"Annoying but a workaround exists.",
	"Blocking work for the reporting user.",
	"Blocking a team, with data loss or a security angle.",
}

// ticketQuestions returns the shared question set: one Noul, one Choice,
// one Score, keyed by stable ids the response echoes back.
func ticketQuestions() decide.Questions {
	return decide.Questions{
		"hardware_fault": decide.Noul{
			Instructions: "True if the evidence points at a physical defect rather than consumables, configuration, or software.",
			True:         "a physical or firmware defect is the likely cause",
			False:        "consumables, configuration, or software explain the symptom",
		},
		"next_action": decide.Choice{
			Instructions: "Which single next step best serves the customer?",
			Options: map[string]any{
				"dispatch_technician": "Send a field technician; likely hardware replacement.",
				"guided_reclean":      "Walk the customer through a duplex-roller cleaning session.",
				"replace_device":      "The unit is under warranty; ship a replacement directly.",
			},
		},
		"severity": decide.Score{
			Instructions: "Rate the operational severity of this ticket.",
			Levels:       ticketLevels,
		},
	}
}

// decideCase holds a fresh recording transport, the client, and the Ask
// inputs a fixture needs.
type decideCase struct {
	t         *testing.T
	sess      *livetest.Session
	tr        *livetest.Transport
	cl        *decide.Client
	ctx       context.Context
	caseName  string
	model     string
	state     any
	questions decide.Questions
}

func newDecideCase(t *testing.T, sess *livetest.Session, name string) *decideCase {
	ctx, _ := livetest.Ctx(t)
	dc := &decideCase{t: t, sess: sess, tr: livetest.NewTransport(sess.Key), ctx: ctx, caseName: name, model: sess.Model}
	dc.cl = sess.DecideClient(ctx, t, dc.tr, nil)
	return dc
}

// variant returns a sibling case with a mutated Config and its own
// transport/input log. The effective model is captured for the fixture, so
// mutating Config.Model records the model actually sent.
func (dc *decideCase) variant(mutate func(*decide.Config)) *decideCase {
	ctx, _ := livetest.Ctx(dc.t)
	v := &decideCase{t: dc.t, sess: dc.sess, tr: livetest.NewTransport(dc.sess.Key), ctx: ctx, caseName: dc.caseName, model: dc.model}
	v.cl = dc.sess.DecideClient(ctx, dc.t, v.tr, func(c *decide.Config) {
		mutate(c)
		v.model = c.Model
	})
	return v
}

// ask issues the Ask and logs its inputs for the fixture.
func (dc *decideCase) ask(state any, questions decide.Questions) (decide.Response, error) {
	dc.state, dc.questions = state, questions
	return dc.cl.Ask(dc.ctx, state, questions)
}

// finish records the Ask inputs, sanitized wire exchanges, and the
// normalized outcome (or error kind) when -update is set.
func (dc *decideCase) finish(resp decide.Response, err error) {
	if !livetest.Update() {
		return
	}
	state, err2 := json.Marshal(dc.state)
	if err2 != nil {
		dc.t.Fatalf("case %s: marshal state: %v", dc.caseName, err2)
	}
	qs, err3 := livetest.RecordDecideQuestions(dc.questions)
	if err3 != nil {
		dc.t.Fatalf("case %s: %v", dc.caseName, err3)
	}
	if got := len(dc.tr.Exchanges()); got == 0 {
		dc.t.Fatalf("case %s: no wire exchanges recorded; pre-wire validation must fail the case, not finish it", dc.caseName)
	}
	f := &livetest.DecideFixture{
		Case:      dc.caseName,
		Lane:      dc.sess.Lane.Name,
		Model:     dc.model,
		State:     state,
		Questions: qs,
		Exchanges: dc.tr.Exchanges(),
		Normalized: livetest.DecideNormalized{
			ErrKind: livetest.ErrKindOf(err),
		},
	}
	if err == nil {
		r := resp
		f.Normalized.Response = &r
	}
	livetest.WriteDecideFixture(dc.t, filepath.Join("testdata", dc.caseName+".json"), f, dc.sess.Key)
}

// TestLiveDecide resolves the typesafe lane once and runs every case under
// it; the lane-level skip names LLMKIT_LIVE_TYPESAFE_API_KEY.
func TestLiveDecide(t *testing.T) {
	sess := livetest.ResolveDecide(t)

	t.Run("mixed_questions", func(t *testing.T) { caseMixedQuestions(t, sess) })
	t.Run("structured_state", func(t *testing.T) { caseStructuredState(t, sess) })
	t.Run("error_bad_key", func(t *testing.T) { caseErrorBadKey(t, sess) })
	t.Run("error_bad_model", func(t *testing.T) { caseErrorBadModel(t, sess) })
	t.Run("usage_recorded", func(t *testing.T) { caseUsageRecorded(t, sess) })
}

// caseMixedQuestions asks one Noul, one Choice, and one Score in a single
// Ask and asserts the structural invariants of every answer kind.
func caseMixedQuestions(t *testing.T, sess *livetest.Session) {
	dc := newDecideCase(t, sess, "mixed_questions")
	resp, err := dc.ask(ticketState, ticketQuestions())
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if resp.Model == "" {
		t.Fatal("empty Model: the vendor's versioned id must be reported")
	}
	if resp.Usage.InputTokens <= 0 {
		t.Fatalf("usage not accounted: %+v", resp.Usage)
	}
	// Noul: a belief in [0, 1].
	if got, ok := resp.Nouls["hardware_fault"]; !ok || got < 0 || got > 1 {
		t.Fatalf("noul hardware_fault = %v (present=%t), want a value in [0, 1]", got, ok)
	}
	// Choice: one probability per asked option; the selected option carries
	// the maximum probability (compared by value: at a tied maximum the id is
	// ambiguous, and iterating the map for it would race Go's random order);
	// the distribution sums to 1; confidence is a probability.
	ch := resp.Choices["next_action"]
	if ch.Choice == "" {
		t.Fatalf("no answer for next_action: %+v", resp.Choices)
	}
	if len(ch.Probabilities) != len(ticketQuestions()["next_action"].(decide.Choice).Options) {
		t.Fatalf("choice probabilities %v, want one entry per asked option", ch.Probabilities)
	}
	var sum float64
	for _, p := range ch.Probabilities {
		sum += p
	}
	chosen, ok := ch.Probabilities[ch.Choice]
	if !ok {
		t.Fatalf("choice %q is not a reported option (%v)", ch.Choice, ch.Probabilities)
	}
	best := 0.0
	for _, p := range ch.Probabilities {
		if p > best {
			best = p
		}
	}
	if chosen != best {
		t.Fatalf("choice %q probability %v, want the maximum %v (%v)", ch.Choice, chosen, best, ch.Probabilities)
	}
	if math.Abs(sum-1) > probSumTol(len(ch.Probabilities)) {
		t.Fatalf("choice probabilities sum %.3f, want within %.3g of 1 (%v)", sum, probSumTol(len(ch.Probabilities)), ch.Probabilities)
	}
	if ch.Confidence < 0 || ch.Confidence > 1 {
		t.Fatalf("choice confidence %f outside [0, 1]", ch.Confidence)
	}
	// Score: the legend equals the request's levels in order; probabilities
	// are index-aligned and sum to 1; confidence is a probability.
	sc := resp.Scores["severity"]
	if len(sc.Levels) != len(ticketLevels) || len(sc.Probabilities) != len(ticketLevels) {
		t.Fatalf("score severity: %d levels, %d probabilities, want %d of each",
			len(sc.Levels), len(sc.Probabilities), len(ticketLevels))
	}
	for i, want := range ticketLevels {
		if sc.Levels[i] != want {
			t.Fatalf("score legend drifted at %d: got %q, want %q", i, sc.Levels[i], want)
		}
	}
	sum = 0
	for _, p := range sc.Probabilities {
		sum += p
	}
	if math.Abs(sum-1) > probSumTol(len(sc.Probabilities)) {
		t.Fatalf("score probabilities sum %.3f, want within %.3g of 1 (%v)", sum, probSumTol(len(sc.Probabilities)), sc.Probabilities)
	}
	if sc.Confidence < 0 || sc.Confidence > 1 {
		t.Fatalf("score confidence %f outside [0, 1]", sc.Confidence)
	}
	dc.finish(resp, err)
}

// caseStructuredState proves object-shaped state and instructions — the
// docs' "Advanced: structure" forms — reach the vendor and normalize.
func caseStructuredState(t *testing.T, sess *livetest.Session) {
	state := map[string]any{
		"ticket": map[string]any{
			"id":      "T-1024",
			"subject": "Printer drops off the network after firmware update",
			"body":    "Since yesterday's firmware update the device disappears from the network every few minutes. A power cycle helps for about an hour. Duplex printing always triggers the drop.",
		},
		"environment": map[string]any{
			"os":         "Windows 11",
			"connection": "Wi-Fi",
		},
	}
	questions := decide.Questions{
		"hardware_issue": decide.Noul{
			Instructions: map[string]any{
				"question": "Does this describe a hardware or firmware defect?",
				"signal":   "post-update connectivity loss that duplex printing reproducibly triggers",
			},
			True:  "firmware or hardware defect",
			False: "environment or configuration problem",
		},
	}
	dc := newDecideCase(t, sess, "structured_state")
	resp, err := dc.ask(state, questions)
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if got, ok := resp.Nouls["hardware_issue"]; !ok || got < 0 || got > 1 {
		t.Fatalf("noul hardware_issue = %v (present=%t), want a value in [0, 1]", got, ok)
	}
	if resp.Model == "" || resp.Usage.InputTokens <= 0 {
		t.Fatalf("model %q / usage %+v not reported", resp.Model, resp.Usage)
	}
	dc.finish(resp, err)
}

func caseErrorBadKey(t *testing.T, sess *livetest.Session) {
	base := newDecideCase(t, sess, "error_bad_key")
	bad := base.variant(func(c *decide.Config) { c.APIKey = badKey })
	_, err := bad.ask(ticketState, ticketQuestions())
	if !errors.Is(err, llmkit.ErrAuth) {
		t.Fatalf("error = %v, want ErrAuth", err)
	}
	bad.finish(decide.Response{}, err)
}

// caseErrorBadModel asserts the Kind only: an unknown model is vendor
// validation (422 or another 4xx), so docs/providers.md's status table maps
// it to ErrInvalidRequest whatever the exact status.
func caseErrorBadModel(t *testing.T, sess *livetest.Session) {
	base := newDecideCase(t, sess, "error_bad_model")
	bad := base.variant(func(c *decide.Config) { c.Model = badModel })
	_, err := bad.ask(ticketState, ticketQuestions())
	var ae *llmkit.APIError
	if !errors.As(err, &ae) {
		t.Fatalf("error = %v, want an *llmkit.APIError", err)
	}
	if !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Fatalf("error = %v (status %d), want ErrInvalidRequest per docs/providers.md's status table", err, ae.StatusCode)
	}
	t.Logf("observed vendor status for an unknown model: %d", ae.StatusCode)
	bad.finish(decide.Response{}, err)
}

// captureRecorder captures the usage events one case produces while still
// feeding the run-wide tally, so the LIVE_TOKENS line stays complete.
type captureRecorder struct {
	tally  *livetest.Tally
	mu     sync.Mutex
	events []llmkit.UsageEvent
}

func (c *captureRecorder) Record(ev llmkit.UsageEvent) {
	c.tally.Record(ev)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
}

func (c *captureRecorder) snapshot() []llmkit.UsageEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]llmkit.UsageEvent(nil), c.events...)
}

// caseUsageRecorded proves Config.Recorder fires exactly once per successful
// Ask, with the provider tag and the response's reported model.
func caseUsageRecorded(t *testing.T, sess *livetest.Session) {
	base := newDecideCase(t, sess, "usage_recorded")
	rec := &captureRecorder{tally: livetest.DefaultTally()}
	via := base.variant(func(c *decide.Config) { c.Recorder = rec })
	resp, err := via.ask(ticketState, ticketQuestions())
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	events := rec.snapshot()
	if len(events) != 1 {
		t.Fatalf("Config.Recorder fired %d time(s), want exactly once per successful Ask", len(events))
	}
	ev := events[0]
	if ev.Provider != "typesafe" {
		t.Fatalf("usage event provider %q, want typesafe", ev.Provider)
	}
	if ev.Model != resp.Model {
		t.Fatalf("usage event model %q, want the response's reported model %q", ev.Model, resp.Model)
	}
	if ev.Usage != resp.Usage {
		t.Fatalf("usage event %+v does not match the response's %+v", ev.Usage, resp.Usage)
	}
	via.finish(resp, err)
}

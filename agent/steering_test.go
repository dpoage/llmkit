package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
)

// steerGateClient is a scripted llmkit.Client whose completions the test
// releases one by one: Complete records the request, signals served[i], and
// blocks until release[i] closes. Holding a response blocks the loop inside
// client.Complete, so the test can queue steering turns at a deterministic
// point — before the loop sees the response and reaches the next drain.
type steerGateClient struct {
	mu      sync.Mutex
	steps   []scriptStep
	reqs    []llmkit.Request
	served  []chan struct{}
	release []chan struct{}
}

func newSteerGate(steps ...scriptStep) *steerGateClient {
	return &steerGateClient{steps: steps}
}

func (f *steerGateClient) Capabilities() llmkit.Capabilities { return llmkit.Capabilities{} }

// gate returns (creating lazily) the signal channels for completion i.
func (f *steerGateClient) gate(i int) (served, release chan struct{}) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for len(f.served) <= i {
		f.served = append(f.served, make(chan struct{}))
		f.release = append(f.release, make(chan struct{}))
	}
	return f.served[i], f.release[i]
}

func (f *steerGateClient) Complete(ctx context.Context, req llmkit.Request) (llmkit.Response, error) {
	if err := ctx.Err(); err != nil {
		return llmkit.Response{}, err
	}
	f.mu.Lock()
	i := len(f.reqs)
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	served, release := f.gate(i)
	close(served)
	select {
	case <-release:
		return f.step(i)
	case <-ctx.Done():
		return llmkit.Response{}, ctx.Err()
	}
}

func (f *steerGateClient) step(i int) (llmkit.Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i >= len(f.steps) {
		// Default to a benign end-turn so over-running tests fail on
		// assertions, not panics.
		return llmkit.Response{Text: "(unscripted)", StopReason: llmkit.StopEndTurn}, nil
	}
	return f.steps[i].resp, f.steps[i].err
}

// requests returns a copy of the recorded requests.
func (f *steerGateClient) requests() []llmkit.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llmkit.Request(nil), f.reqs...)
}

// waitServed blocks until the loop's i-th completion has recorded its
// request, failing the test on timeout instead of hanging.
func (f *steerGateClient) waitServed(t *testing.T, i int) {
	t.Helper()
	served, _ := f.gate(i)
	select {
	case <-served:
	case <-time.After(10 * time.Second):
		t.Fatalf("steerGateClient: completion %d never served", i)
	}
}

// releaseNow lets the loop's i-th completion return its scripted response.
func (f *steerGateClient) releaseNow(i int) {
	_, release := f.gate(i)
	close(release)
}

// waitDone waits for a run goroutine, failing the test on timeout.
func waitDone(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("run did not finish")
	}
}

// lastUserText returns the text of the final user message of a request.
func lastUserText(req llmkit.Request) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if m := req.Messages[i]; m.Role == llmkit.RoleUser {
			return m.Text()
		}
	}
	return ""
}

// hasUserText reports whether msgs carries a user message with exactly the
// given text.
func hasUserText(msgs []llmkit.Message, want string) bool {
	for _, m := range msgs {
		if m.Role == llmkit.RoleUser && m.Text() == want {
			return true
		}
	}
	return false
}

func TestSteeringSteerDeliveredBeforeNextCompletion(t *testing.T) {
	cl := newSteerGate(
		toolResp("call1", "echo", `"first"`, 10, 5),
		textResp("done", 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, []Tool{echoTool{name: "echo"}}, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	cl.waitServed(t, 0)
	if s.Pending() != 0 {
		t.Fatalf("Pending = %d before any queueing, want 0", s.Pending())
	}
	s.Steer(llmkit.Text("STEER-1"))
	if s.Pending() != 1 {
		t.Fatalf("Pending = %d after Steer, want 1", s.Pending())
	}
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	// The steer turn must sit after the tool results and before the next
	// completion — the request is recorded before the loop can proceed.
	reqs := cl.requests()
	if len(reqs) != 2 {
		t.Fatalf("completions = %d, want 2", len(reqs))
	}
	if got := lastUserText(reqs[1]); got != "STEER-1" {
		t.Fatalf("last user text of request 2 = %q, want STEER-1", got)
	}
	tail := reqs[1].Messages[len(reqs[1].Messages)-2:]
	if tail[0].Role != llmkit.RoleToolResult {
		t.Fatalf("message before the steer turn is %s, want tool result", tail[0].Role)
	}
	cl.releaseNow(1)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "done" {
		t.Fatalf("FinalText = %q, want done", outcome.FinalText)
	}
	if s.Pending() != 0 {
		t.Fatalf("Pending = %d after delivery, want 0", s.Pending())
	}
}

func TestSteeringFollowUpContinuesFinishedRun(t *testing.T) {
	cl := newSteerGate(
		textResp("answer-1", 10, 5),
		textResp("answer-2", 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, nil, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	cl.waitServed(t, 0)
	s.FollowUp(llmkit.Text("FOLLOW-UP"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	// The follow-up turned the would-be final turn into another turn.
	reqs := cl.requests()
	if len(reqs) != 2 {
		t.Fatalf("completions = %d, want 2", len(reqs))
	}
	if got := lastUserText(reqs[1]); got != "FOLLOW-UP" {
		t.Fatalf("last user text of request 2 = %q, want FOLLOW-UP", got)
	}
	cl.releaseNow(1)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "answer-2" {
		t.Fatalf("FinalText = %q, want answer-2", outcome.FinalText)
	}
	assistants := 0
	for _, m := range outcome.Messages {
		if m.Role == llmkit.RoleAssistant {
			assistants++
		}
	}
	if assistants != 2 {
		t.Fatalf("assistant turns = %d, want 2", assistants)
	}
	if !hasUserText(outcome.Messages, "FOLLOW-UP") {
		t.Fatal("Outcome.Messages lost the follow-up user turn")
	}
}

func TestSteeringSteerContinuesFinishedRun(t *testing.T) {
	cl := newSteerGate(
		textResp("answer-1", 10, 5),
		textResp("answer-2", 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, nil, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	cl.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	if got := lastUserText(cl.requests()[1]); got != "STEER" {
		t.Fatalf("last user text of request 2 = %q, want STEER", got)
	}
	cl.releaseNow(1)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "answer-2" {
		t.Fatalf("FinalText = %q, want answer-2", outcome.FinalText)
	}
}

func TestSteeringFollowUpWaitsAtTurnBoundary(t *testing.T) {
	cl := newSteerGate(
		toolResp("call1", "echo", `"x"`, 10, 5),
		textResp("mid", 10, 5),
		textResp("final", 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, []Tool{echoTool{name: "echo"}}, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	// The follow-up is queued BEFORE the steer, yet only the steer may
	// deliver at the turn boundary; the older follow-up waits for the
	// would-be finish.
	cl.waitServed(t, 0)
	s.FollowUp(llmkit.Text("FIRST-FOLLOWUP"))
	s.Steer(llmkit.Text("LATE-STEER"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	reqs := cl.requests()
	if got := lastUserText(reqs[1]); got != "LATE-STEER" {
		t.Fatalf("last user text of request 2 = %q, want LATE-STEER", got)
	}
	if hasUserText(reqs[1].Messages, "FIRST-FOLLOWUP") {
		t.Fatal("follow-up delivered at the turn boundary; it must wait for the finish drain")
	}
	cl.releaseNow(1)
	cl.waitServed(t, 2)

	if got := lastUserText(cl.requests()[2]); got != "FIRST-FOLLOWUP" {
		t.Fatalf("last user text of request 3 = %q, want FIRST-FOLLOWUP", got)
	}
	cl.releaseNow(2)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "final" {
		t.Fatalf("FinalText = %q, want final", outcome.FinalText)
	}
	steerAt, followAt := -1, -1
	for i, m := range outcome.Messages {
		if m.Role != llmkit.RoleUser {
			continue
		}
		switch m.Text() {
		case "LATE-STEER":
			steerAt = i
		case "FIRST-FOLLOWUP":
			followAt = i
		}
	}
	if steerAt < 0 || followAt < 0 {
		t.Fatalf("Outcome.Messages lost a queued turn (steer at %d, follow-up at %d)", steerAt, followAt)
	}
	if steerAt > followAt {
		t.Fatalf("steer (index %d) delivered after the older follow-up (index %d)", steerAt, followAt)
	}
}

func TestSteeringEmptyQueueKeepsWireIdentical(t *testing.T) {
	script := []scriptStep{
		toolResp("call1", "echo", `"a"`, 10, 5),
		textResp("mid", 10, 5),
		textResp("final", 10, 5),
	}
	base := newFakeClient(script...)
	steered := newFakeClient(script...)

	outBase, errBase := NewRunner(base, []Tool{echoTool{name: "echo"}}, "sys").
		Run(context.Background(), "task")
	if errBase != nil {
		t.Fatalf("base run: %v", errBase)
	}
	outSteered, errSteered := NewRunner(steered, []Tool{echoTool{name: "echo"}}, "sys").
		Run(context.Background(), "task", WithSteering(NewSteering()))
	if errSteered != nil {
		t.Fatalf("steered run: %v", errSteered)
	}

	if !reflect.DeepEqual(base.requests, steered.requests) {
		t.Fatalf("an empty steering handle changed the wire:\nbase: %+v\nsteered: %+v",
			base.requests, steered.requests)
	}
	if outBase.FinalText != outSteered.FinalText || outBase.Iterations != outSteered.Iterations {
		t.Fatalf("outcome drifted: base (%q, %d) vs steered (%q, %d)",
			outBase.FinalText, outBase.Iterations, outSteered.FinalText, outSteered.Iterations)
	}
}

func TestSteeringPendingAfterLimitStop(t *testing.T) {
	cl := newSteerGate(toolResp("call1", "echo", `"a"`, 10, 5))
	s := NewSteering()
	runner := NewRunner(cl, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{MaxIterations: 1}))

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	cl.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER"))
	cl.releaseNow(0)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.TruncationReason != TruncMaxIterations {
		t.Fatalf("TruncationReason = %q, want %q", outcome.TruncationReason, TruncMaxIterations)
	}
	if s.Pending() != 1 {
		t.Fatalf("Pending = %d after the limit stop, want 1 (undelivered turn stays queued)", s.Pending())
	}
	if hasUserText(outcome.Messages, "STEER") {
		t.Fatal("undelivered steer turn leaked into Outcome.Messages")
	}
}

func TestSteeringContinueDeliversPending(t *testing.T) {
	cl1 := newSteerGate(toolResp("call1", "echo", `"a"`, 10, 5))
	s := NewSteering()
	runner1 := NewRunner(cl1, []Tool{echoTool{name: "echo"}}, "sys",
		WithLimits(Limits{MaxIterations: 1}))

	done1 := make(chan struct{})
	var out1 *Outcome
	var err1 error
	go func() {
		defer close(done1)
		out1, err1 = runner1.Run(context.Background(), "task", WithSteering(s))
	}()
	cl1.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER-PEND"))
	s.FollowUp(llmkit.Text("FOLLOW-PEND"))
	cl1.releaseNow(0)
	waitDone(t, done1)
	if err1 != nil {
		t.Fatalf("run 1: %v", err1)
	}
	if s.Pending() != 2 {
		t.Fatalf("Pending = %d after the limit stop, want 2", s.Pending())
	}

	// The continued run delivers the pending steer before its FIRST
	// completion and the pending follow-up at its first would-be finish.
	cl2 := newSteerGate(
		textResp("resumed-1", 10, 5),
		textResp("resumed-2", 10, 5),
	)
	runner2 := NewRunner(cl2, []Tool{echoTool{name: "echo"}}, "sys")

	done2 := make(chan struct{})
	var out2 *Outcome
	var err2 error
	go func() {
		defer close(done2)
		out2, err2 = runner2.Run(context.Background(), "resume", Continue(out1), WithSteering(s))
	}()
	cl2.waitServed(t, 0)

	reqs := cl2.requests()
	if !hasUserText(reqs[0].Messages, "STEER-PEND") {
		t.Fatal("the continued run's first completion lost the pending steer")
	}
	if hasUserText(reqs[0].Messages, "FOLLOW-PEND") {
		t.Fatal("pending follow-up delivered at the turn boundary; it must wait for the finish drain")
	}
	if s.Pending() != 1 {
		t.Fatalf("Pending = %d after the steer drain, want 1", s.Pending())
	}
	cl2.releaseNow(0)
	cl2.waitServed(t, 1)

	if got := lastUserText(cl2.requests()[1]); got != "FOLLOW-PEND" {
		t.Fatalf("last user text of request 2 = %q, want FOLLOW-PEND", got)
	}
	cl2.releaseNow(1)
	waitDone(t, done2)

	if err2 != nil {
		t.Fatalf("run 2: %v", err2)
	}
	if out2.FinalText != "resumed-2" {
		t.Fatalf("FinalText = %q, want resumed-2", out2.FinalText)
	}
	if s.Pending() != 0 {
		t.Fatalf("Pending = %d after the continued run, want 0", s.Pending())
	}
}

func TestSteeringConcurrentProducers(t *testing.T) {
	s := NewSteering()
	cl := newSteerGate(
		toolResp("call1", "echo", `"a"`, 10, 5),
		textResp("mid", 10, 5),
		textResp("done", 10, 5),
	)
	done := make(chan struct{})
	var outcome *Outcome
	runner := NewRunner(cl, []Tool{echoTool{name: "echo"}}, "sys")
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()
	cl.waitServed(t, 0)

	const n = 64
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			if i%2 == 0 {
				s.Steer(llmkit.Text(fmt.Sprintf("steer-%d", i)))
			} else {
				s.FollowUp(llmkit.Text(fmt.Sprintf("follow-%d", i)))
			}
		}(i)
	}
	wg.Wait()
	if s.Pending() != n {
		t.Fatalf("Pending = %d with all producers done, want %d", s.Pending(), n)
	}

	cl.releaseNow(0)
	cl.waitServed(t, 1)
	reqs := cl.requests()
	steers := map[string]bool{}
	for _, m := range reqs[1].Messages {
		if m.Role == llmkit.RoleUser && len(m.Text()) > 6 && m.Text()[:6] == "steer-" {
			steers[m.Text()] = true
		}
		if m.Role == llmkit.RoleUser && len(m.Text()) > 7 && m.Text()[:7] == "follow-" {
			t.Fatalf("follow-up %q delivered at the turn boundary", m.Text())
		}
	}
	if len(steers) != n/2 {
		t.Fatalf("steer turns delivered = %d, want %d", len(steers), n/2)
	}
	for i := 0; i < n; i += 2 {
		if !steers[fmt.Sprintf("steer-%d", i)] {
			t.Fatalf("steer-%d lost at the turn boundary", i)
		}
	}
	cl.releaseNow(1)
	cl.waitServed(t, 2)
	cl.releaseNow(2)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "done" {
		t.Fatalf("FinalText = %q, want done", outcome.FinalText)
	}
	lastFollow := -1
	for i, m := range outcome.Messages {
		if m.Role != llmkit.RoleUser {
			continue
		}
		if len(m.Text()) > 7 && m.Text()[:7] == "follow-" {
			lastFollow = i
		}
	}
	if lastFollow < 0 {
		t.Fatal("Outcome.Messages lost the follow-up turns")
	}
	if s.Pending() != 0 {
		t.Fatalf("Pending = %d after the run, want 0", s.Pending())
	}
}

func TestSteeringRunJSONHonorsDrains(t *testing.T) {
	cl := newSteerGate(
		toolResp("call1", "echo", `"x"`, 10, 5),
		textResp(`{"answer":"final"}`, 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, []Tool{echoTool{name: "echo"}}, "sys")

	schema := json.RawMessage(`{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}}}`)
	type answer struct {
		Answer string `json:"answer"`
	}
	var out answer

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.RunJSON(context.Background(), "task", schema, &out, WithSteering(s))
	}()

	cl.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	if !hasUserText(cl.requests()[1].Messages, "STEER") {
		t.Fatal("RunJSON's second completion lost the steer turn")
	}
	cl.releaseNow(1)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if out.Answer != "final" {
		t.Fatalf("answer = %q, want final (the parse must apply to the LAST completion)", out.Answer)
	}
	if !hasUserText(outcome.Messages, "STEER") {
		t.Fatal("Outcome.Messages lost the steer turn")
	}
	if s.Pending() != 0 {
		t.Fatalf("Pending = %d after the run, want 0", s.Pending())
	}
}

func TestSteeringRefusalKeepsQueuePending(t *testing.T) {
	cl := newSteerGate(scriptStep{
		resp: llmkit.Response{Text: "I cannot help with that", StopReason: llmkit.StopRefusal},
	})
	s := NewSteering()
	runner := NewRunner(cl, nil, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	cl.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER"))
	cl.releaseNow(0)
	waitDone(t, done)

	var stopErr *StopReasonError
	if !errors.As(err, &stopErr) || stopErr.StopReason != llmkit.StopRefusal {
		t.Fatalf("err = %v, want a StopReasonError with stop reason %q", err, llmkit.StopRefusal)
	}
	if s.Pending() != 1 {
		t.Fatalf("Pending = %d after the refusal stop, want 1", s.Pending())
	}
	if hasUserText(outcome.Messages, "STEER") {
		t.Fatal("queued steer turn was delivered at a refusal stop")
	}
}

func TestSteeringRejectsSecondRunWhileBound(t *testing.T) {
	cl := newSteerGate(textResp("slow", 10, 5))
	s := NewSteering()
	runner := NewRunner(cl, nil, "sys")

	done := make(chan struct{})
	var err1 error
	go func() {
		defer close(done)
		_, err1 = runner.Run(context.Background(), "one", WithSteering(s))
	}()
	cl.waitServed(t, 0)

	// A short-cancel context keeps a mutated build (bind check removed)
	// failing fast on the ctx error instead of hanging on the gate.
	ctx2, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	// A second concurrent run on the same handle fails at Run entry.
	_, err2 := runner.Run(ctx2, "two", WithSteering(s))
	if !errors.Is(err2, ErrSteeringInUse) {
		t.Fatalf("second run err = %v, want ErrSteeringInUse", err2)
	}

	cl.releaseNow(0)
	waitDone(t, done)
	if err1 != nil {
		t.Fatalf("first run: %v", err1)
	}
}

func TestSteeringQueuedTurnReplacesEmptyTurnNudge(t *testing.T) {
	cl := newSteerGate(
		scriptStep{resp: llmkit.Response{StopReason: llmkit.StopEndTurn}},
		textResp("answered", 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, nil, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	cl.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	reqs := cl.requests()
	if got := lastUserText(reqs[1]); got != "STEER" {
		t.Fatalf("last user text of request 2 = %q, want STEER (queued content replaces the nudge)", got)
	}
	if hasUserText(reqs[1].Messages, emptyTurnNudge) {
		t.Fatal("the synthetic nudge was appended although queued content was delivered")
	}
	cl.releaseNow(1)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "answered" {
		t.Fatalf("FinalText = %q, want answered", outcome.FinalText)
	}
}

func TestSteeringSteerRescueDoesNotConsumeNudgeBudget(t *testing.T) {
	empty := scriptStep{resp: llmkit.Response{StopReason: llmkit.StopEndTurn}}
	cl := newSteerGate(empty, empty, empty, empty, empty)
	s := NewSteering()
	runner := NewRunner(cl, nil, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	// Two steer-rescued empty turns, then genuinely empty turns that ride
	// the nudge path.
	cl.waitServed(t, 0)
	s.Steer(llmkit.Text("STEER-A"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)
	s.Steer(llmkit.Text("STEER-B"))
	cl.releaseNow(1)
	for i := 2; i <= 4; i++ {
		cl.waitServed(t, i)
		cl.releaseNow(i)
	}
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	reqs := cl.requests()
	// The request in which a steer turn FIRST appears must be free of the
	// synthetic nudge: a steer rescue replaces the nudge for that turn.
	// (Later requests carry both, as persisted history.)
	for _, steer := range []string{"STEER-A", "STEER-B"} {
		first := -1
		for i, req := range reqs {
			if hasUserText(req.Messages, steer) {
				first = i
				break
			}
		}
		if first < 0 {
			t.Fatalf("%s never reached the wire", steer)
		}
		if hasUserText(reqs[first].Messages, emptyTurnNudge) {
			t.Fatalf("request %d delivered %s together with the synthetic nudge", first, steer)
		}
	}
	// The nudge turns persist in the final history: exactly the two
	// budgeted nudges may be there, none consumed by the steer rescues.
	nudges := 0
	for _, m := range outcome.Messages {
		if m.Role == llmkit.RoleUser && m.Text() == emptyTurnNudge {
			nudges++
		}
	}
	if nudges != 2 {
		t.Fatalf("nudge turns in the final history = %d, want 2 (steer rescues must not consume nudge attempts)", nudges)
	}
	if !hasUserText(outcome.Messages, "STEER-A") || !hasUserText(outcome.Messages, "STEER-B") {
		t.Fatal("Outcome.Messages lost a steer-rescued turn")
	}
}

func TestSteeringDrainKeepsEnqueueOrder(t *testing.T) {
	cl := newSteerGate(
		toolResp("call1", "echo", `"x"`, 10, 5),
		textResp("mid", 10, 5),
		textResp("final", 10, 5),
	)
	s := NewSteering()
	runner := NewRunner(cl, []Tool{echoTool{name: "echo"}}, "sys")

	done := make(chan struct{})
	var outcome *Outcome
	var err error
	go func() {
		defer close(done)
		outcome, err = runner.Run(context.Background(), "task", WithSteering(s))
	}()

	// Interleave the kinds while the loop is blocked: the FIFO across both
	// is what each drain must preserve.
	cl.waitServed(t, 0)
	s.FollowUp(llmkit.Text("F1"))
	s.Steer(llmkit.Text("S1"))
	s.Steer(llmkit.Text("S2"))
	s.FollowUp(llmkit.Text("F2"))
	cl.releaseNow(0)
	cl.waitServed(t, 1)

	// Turn-boundary drain: the steers deliver in enqueue order, contiguous
	// after the tool results; the follow-ups wait.
	reqs := cl.requests()
	toolIdx := -1
	for i, m := range reqs[1].Messages {
		if m.Role == llmkit.RoleToolResult {
			toolIdx = i
		}
	}
	if toolIdx < 0 {
		t.Fatal("request 2 lost the tool result")
	}
	tail := reqs[1].Messages[toolIdx+1:]
	if len(tail) != 2 || tail[0].Text() != "S1" || tail[1].Text() != "S2" {
		t.Fatalf("steer drain delivered [%s, %s], want [S1, S2] contiguous after the tool result",
			tail[0].Text(), tail[1].Text())
	}
	if hasUserText(reqs[1].Messages, "F1") || hasUserText(reqs[1].Messages, "F2") {
		t.Fatal("follow-ups delivered at the turn boundary")
	}
	cl.releaseNow(1)
	cl.waitServed(t, 2)

	// Finish drain: the follow-ups deliver in enqueue order, directly after
	// the assistant turn that would have ended the run.
	msgs := cl.requests()[2].Messages
	if n := len(msgs); n < 3 || msgs[n-2].Text() != "F1" || msgs[n-1].Text() != "F2" {
		t.Fatalf("finish drain tail = [%s, %s], want [F1, F2]",
			msgs[n-2].Text(), msgs[n-1].Text())
	}
	cl.releaseNow(2)
	waitDone(t, done)

	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if outcome.FinalText != "final" {
		t.Fatalf("FinalText = %q, want final", outcome.FinalText)
	}
	order := map[string]int{"S1": -1, "S2": -1, "F1": -1, "F2": -1}
	for i, m := range outcome.Messages {
		if m.Role == llmkit.RoleUser {
			if _, ok := order[m.Text()]; ok {
				order[m.Text()] = i
			}
		}
	}
	if order["S1"] < 0 || order["S1"] > order["S2"] || order["S2"] > order["F1"] || order["F1"] > order["F2"] {
		t.Fatalf("delivery order broken: %v", order)
	}
}

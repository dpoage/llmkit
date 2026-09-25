// Package livetest is the shared machinery behind llmkit's live acceptance
// tests (the `live` build tag). It is plain Go — no build tag of its own —
// and is imported only from _test files, so it never ships in the public
// API and never runs outside a test binary.
//
// It owns three decisions no individual live test may re-make:
//
//   - the lane environment contract: which LLMKIT_LIVE_* variables each
//     vendor lane needs, what the defaults are, and what a skip message
//     must name;
//   - secret hygiene: the recording transport's header sanitization, the
//     redacting logger, and the assertion that no written fixture contains
//     the lane credential or any sk- substring;
//   - the run-wide token tally and the fixture format (see fixture.go).
//
// The -update flag is registered here once per test binary, so a single
//
//	go test -tags live ./provider/ ./agent/ ./examples/... -update
//
// re-records every package's fixtures in one invocation.
package livetest

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/provider"
)

// update is registered at package init; every live-tagged test binary
// imports livetest, so -update is understood uniformly.
var update = flag.Bool("update", false, "re-record live fixtures into provider/testdata")

// Update reports whether the run was invoked with -update.
func Update() bool { return *update }

// Lane describes one vendor lane: which environment variables it reads
// (EnvPrefix + _API_KEY / _MODEL / _BASE_URL), which model to assume when
// the operator set none, and — for the chat lanes — which provider.Type
// its clients build through provider.New.
type Lane struct {
	Name      string
	EnvPrefix string
	// Type is the provider.Type the lane's clients build through
	// provider.New. It is empty for lanes that construct a different
	// client kind (DecideLane): Jev is not an llmkit.Client, so its lane
	// is resolved by ResolveDecide and its clients built by
	// Session.DecideClient.
	Type provider.Type
	// DefaultModel fills Model when <prefix>_MODEL is unset. Empty means the
	// model is required (the compat lane: an arbitrary endpoint has no
	// sensible default).
	DefaultModel string
	// OptionalBaseURL marks lanes whose <prefix>_BASE_URL is an optional
	// override (the typesafe lane can target a gateway) rather than a
	// requirement (the compat lane).
	OptionalBaseURL bool
}

var lanes = []Lane{
	{Name: "compat", EnvPrefix: "LLMKIT_LIVE_COMPAT", Type: provider.TypeOpenAICompatible},
	{Name: "anthropic", EnvPrefix: "LLMKIT_LIVE_ANTHROPIC", Type: provider.TypeAnthropic, DefaultModel: "claude-haiku-4-5"},
	{Name: "openai", EnvPrefix: "LLMKIT_LIVE_OPENAI", Type: provider.TypeOpenAI, DefaultModel: "gpt-4o-mini"},
	{Name: "google", EnvPrefix: "LLMKIT_LIVE_GOOGLE", Type: provider.TypeGoogle, DefaultModel: "gemini-2.5-flash-lite"},
}

// DecideLane is the TypeSafe Jev decision lane. It stays out of lanes:
// every lane there is dispatched by provider's live matrix through
// provider.New(Lane.Type), and Jev has no provider.Type — its live tests
// go through ResolveDecide and Session.DecideClient instead.
var DecideLane = Lane{
	Name:            "typesafe",
	EnvPrefix:       "LLMKIT_LIVE_TYPESAFE",
	DefaultModel:    "jev-latest",
	OptionalBaseURL: true,
}

// All returns every lane in stable declaration order.
func All() []Lane { return append([]Lane(nil), lanes...) }

// LaneByName looks a lane up by its Name.
func LaneByName(name string) (Lane, bool) {
	for _, l := range lanes {
		if l.Name == name {
			return l, true
		}
	}
	return Lane{}, false
}

// Session is a lane's resolved environment.
type Session struct {
	Lane  Lane
	Model string
	Key   string
	// BaseURL is an explicit endpoint override: required for the compat
	// lane (an arbitrary endpoint has no default), optional for lanes with
	// OptionalBaseURL, empty for the rest.
	BaseURL string
}

// resolve reads the lane's environment. The returned slice names each missing
// required variable exactly, so callers can print actionable skips.
func (l Lane) resolve() (Session, []string) {
	var missing []string
	s := Session{Lane: l}
	s.Key = os.Getenv(l.EnvPrefix + "_API_KEY")
	if s.Key == "" {
		missing = append(missing, l.EnvPrefix+"_API_KEY")
	}
	s.Model = os.Getenv(l.EnvPrefix + "_MODEL")
	if s.Model == "" {
		if l.DefaultModel == "" {
			missing = append(missing, l.EnvPrefix+"_MODEL")
		} else {
			s.Model = l.DefaultModel
		}
	}
	if l.Type == provider.TypeOpenAICompatible {
		s.BaseURL = os.Getenv(l.EnvPrefix + "_BASE_URL")
		if s.BaseURL == "" {
			missing = append(missing, l.EnvPrefix+"_BASE_URL")
		}
	} else if l.OptionalBaseURL {
		s.BaseURL = os.Getenv(l.EnvPrefix + "_BASE_URL")
	}
	return s, missing
}

// Resolve resolves the named lane from the environment, skipping the test
// with a message that names the lane and every missing variable. Use it at
// the lane level so a keyless lane skips once, not per case.
func Resolve(t testing.TB, name string) *Session {
	lane, ok := LaneByName(name)
	if !ok {
		t.Fatalf("livetest: unknown lane %q (want one of: compat, anthropic, openai, google)", name)
		return nil
	}
	s, missing := lane.resolve()
	if len(missing) > 0 {
		t.Skipf("live lane %s: skipping — missing %s", lane.Name, strings.Join(missing, ", "))
		return nil
	}
	return &s
}

// ResolveDecide resolves the typesafe decide lane (DecideLane) from the
// environment, skipping the test with a message that names every missing
// variable. With the lane's default model and optional base URL, only
// LLMKIT_LIVE_TYPESAFE_API_KEY can be missing.
func ResolveDecide(t testing.TB) *Session {
	s, missing := DecideLane.resolve()
	if len(missing) > 0 {
		t.Skipf("live lane %s: skipping — missing %s", DecideLane.Name, strings.Join(missing, ", "))
		return nil
	}
	return &s
}

// CapsEnvVar optionally lists the capabilities the operator asserts the
// compat endpoint supports (snake_case, comma-separated). Listed caps are
// forced true via Spec.Capabilities, so the gated cases run — and must pass.
const CapsEnvVar = "LLMKIT_LIVE_COMPAT_CAPS"

// ParseCaps parses a comma-separated snake_case list of llmkit.Capabilities
// bool fields into field-name → true. Unknown names are an error naming the
// valid ones; ContextWindow is rejected explicitly (a number, not a claim).
func ParseCaps(list string) (map[string]bool, error) {
	boolFields := map[string]string{} // snake_case name -> field name
	typ := reflect.TypeOf(llmkit.Capabilities{})
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if f.Type.Kind() == reflect.Bool {
			boolFields[toSnake(f.Name)] = f.Name
		}
	}
	out := make(map[string]bool)
	for _, item := range strings.Split(list, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if item == "context_window" {
			return nil, fmt.Errorf("livetest: %q is not a boolean capability (ContextWindow is a number; the live case runs only when the profile reports a known window)", item)
		}
		field, ok := boolFields[item]
		if !ok {
			valid := make([]string, 0, len(boolFields))
			for name := range boolFields {
				valid = append(valid, name)
			}
			sort.Strings(valid)
			return nil, fmt.Errorf("livetest: unknown capability %q (valid: %s)", item, strings.Join(valid, ", "))
		}
		out[field] = true
	}
	return out, nil
}

// toSnake converts PascalCase to snake_case: "ParallelToolCalls" ->
// "parallel_tool_calls".
func toSnake(s string) string {
	var b strings.Builder
	for i, r := range s {
		if r >= 'A' && r <= 'Z' {
			if i > 0 {
				b.WriteByte('_')
			}
			b.WriteRune(r - 'A' + 'a')
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// CapsOverride parses CapsEnvVar for the compat lane and returns a
// Spec.Capabilities override that forces the listed capabilities true.
// Returns nil when the lane is not compat or the variable is unset; a
// malformed list is t.Fatal.
func (s *Session) CapsOverride(t testing.TB) func(llmkit.Capabilities) llmkit.Capabilities {
	if s.Lane.Name != "compat" {
		return nil
	}
	raw := os.Getenv(CapsEnvVar)
	if raw == "" {
		return nil
	}
	caps, err := ParseCaps(raw)
	if err != nil {
		t.Fatal(err)
		return nil
	}
	return func(c llmkit.Capabilities) llmkit.Capabilities {
		rv := reflect.ValueOf(&c).Elem()
		for field := range caps {
			rv.FieldByName(field).SetBool(true)
		}
		return c
	}
}

// Client builds a live client for this session through provider.New — the
// production construction path, wrapper stack included — with the
// recording transport injected via Options.HTTPClient and the run-wide
// token tally as Options.Observer. mutate, when non-nil, adjusts the Spec
// last (e.g. an intentionally bad key or model for error-normalization
// cases); it runs after CapsOverride so an assertion-aware override can
// still be replaced.
func (s *Session) Client(ctx context.Context, t testing.TB, tr *Transport, mutate func(*provider.Spec)) llmkit.Client {
	spec := provider.Spec{
		Type:    s.Lane.Type,
		Model:   s.Model,
		BaseURL: s.BaseURL,
		Secret:  s.Key,
	}
	if spec.Secret == "" {
		// Resolve skips keyless lanes before any client is built; the
		// placeholder only keeps a logic error from surfacing as provider
		// New's empty-secret refusal instead of a clear lane failure.
		spec.Secret = "llmkit-live-placeholder"
	}
	if o := s.CapsOverride(t); o != nil {
		spec.Capabilities = o
	}
	if mutate != nil {
		mutate(&spec)
	}
	cl, err := provider.New(ctx, spec, provider.Options{
		HTTPClient: &http.Client{Transport: tr},
		Observer:   DefaultTally(),
	})
	if err != nil {
		t.Fatalf("livetest: provider.New for lane %s: %v", s.Lane.Name, err)
		return nil
	}
	return cl
}

// DecideClient builds the typesafe lane's decide client through
// decide.New, with the recording transport injected via Config.HTTPClient
// and the run-wide token tally as Config.Observer. mutate, when non-nil,
// adjusts the Config last.
func (s *Session) DecideClient(ctx context.Context, t testing.TB, tr *Transport, mutate func(*decide.Config)) *decide.Client {
	cfg := decide.Config{
		APIKey:     s.Key,
		Model:      s.Model,
		BaseURL:    s.BaseURL,
		HTTPClient: &http.Client{Transport: tr},
		Observer:   DefaultTally(),
	}
	if mutate != nil {
		mutate(&cfg)
	}
	cl, err := decide.New(cfg)
	if err != nil {
		t.Fatalf("livetest: decide.New for lane %s: %v", s.Lane.Name, err)
		return nil
	}
	return cl
}

// Ctx returns a context with a bounded deadline, so a hung vendor fails the
// test rather than blocking the run; the cancel is registered via t.Cleanup.
func Ctx(t testing.TB) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)
	return ctx, cancel
}

var (
	skRE     = regexp.MustCompile(`sk-[A-Za-z0-9_-]{8,}`)
	bearerRE = regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._~+/=-]+`)
)

// Redact replaces the given secret and any sk- key or bearer token with
// "[REDACTED]". Every raw-body log line must pass through it.
func Redact(secret, s string) string {
	if secret != "" {
		s = strings.ReplaceAll(s, secret, "[REDACTED]")
	}
	s = skRE.ReplaceAllString(s, "[REDACTED]")
	return bearerRE.ReplaceAllString(s, "Bearer [REDACTED]")
}

// Logf logs to t with the session's credential redacted. Use it for any log
// of a raw body or header set.
func (s *Session) Logf(t testing.TB, format string, args ...any) {
	t.Logf("%s", Redact(s.Key, fmt.Sprintf(format, args...)))
}

// Tally is the run-wide token accumulator. It implements llmkit.Observer
// and is wired into every client livetest builds (Options.Observer,
// Config.Observer, agent.WithObserver on every live Runner), so a whole
// test binary's spend is reported by one line at exit. It sums
// Completion.Response.Usage and DecisionEvent.Usage with [llmkit.Usage.Add]
// on success — Completion and Decision are the spend ledger; Attempt and
// every other event kind are ignored, so a stack that also reports
// Attempts is never double-counted.
type Tally struct {
	mu    sync.Mutex
	usage llmkit.Usage
	calls int
}

var defaultTally = new(Tally)

// DefaultTally returns the run-wide tally wired into every livetest client.
func DefaultTally() *Tally { return defaultTally }

// Observe implements llmkit.Observer.
func (t *Tally) Observe(_ context.Context, ev llmkit.Event) {
	var u llmkit.Usage
	switch ev.Kind {
	case llmkit.KindCompletion:
		if ev.Completion == nil || ev.Completion.Err != "" {
			return
		}
		u = ev.Completion.Response.Usage
	case llmkit.KindDecision:
		if ev.Decision == nil || ev.Decision.Err != "" {
			return
		}
		u = ev.Decision.Usage
	default:
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.usage = t.usage.Add(u)
	t.calls++
}

// PrintSummary prints the single machine-readable spend line for this test
// binary. Call it from TestMain after m.Run.
func (t *Tally) PrintSummary() {
	t.mu.Lock()
	defer t.mu.Unlock()
	fmt.Printf("LIVE_TOKENS total=%d input=%d output=%d cache_read=%d cache_creation=%d calls=%d\n",
		t.usage.InputTokens+t.usage.OutputTokens, t.usage.InputTokens, t.usage.OutputTokens,
		t.usage.CacheReadInputTokens, t.usage.CacheCreationInputTokens, t.calls)
}

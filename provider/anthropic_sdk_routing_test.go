package provider

import (
	"context"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/config"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// guardToken is the bearer token defaultTransportGuard mints for every
// federation token exchange it answers.
const guardToken = "tok-guard-minted"

// defaultTransportGuard stands in for http.DefaultTransport.
// anthropic.NewClient builds its default HTTP client from
// http.DefaultTransport, and the env-federation token exchange stays bound
// to that client after option.WithHTTPClient replaces it for API requests.
// The guard records every request it is handed and never dials: it answers
// a token exchange (POST to config.TokenEndpoint) with guardToken and any
// other request as recordingTransport does.
type defaultTransportGuard struct {
	api recordingTransport

	mu   sync.Mutex
	reqs []*http.Request
}

func (g *defaultTransportGuard) RoundTrip(req *http.Request) (*http.Response, error) {
	g.mu.Lock()
	g.reqs = append(g.reqs, req)
	g.mu.Unlock()
	if !isTokenExchange(req) {
		return g.api.RoundTrip(req)
	}
	if req.Body != nil {
		_, _ = io.Copy(io.Discard, req.Body)
		_ = req.Body.Close()
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"access_token":"` + guardToken + `","token_type":"Bearer","expires_in":3600}`)),
		Request:    req,
	}, nil
}

func (g *defaultTransportGuard) requests() []*http.Request {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]*http.Request(nil), g.reqs...)
}

func (rt *recordingTransport) requests() []*http.Request {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return append([]*http.Request(nil), rt.reqs...)
}

func isTokenExchange(req *http.Request) bool {
	return req.Method == http.MethodPost && strings.HasSuffix(req.URL.Path, config.TokenEndpoint)
}

func requestTargets(reqs []*http.Request) []string {
	targets := make([]string, 0, len(reqs))
	for _, req := range reqs {
		targets = append(targets, req.Method+" "+req.URL.Host+req.URL.Path)
	}
	return targets
}

// sdkRefusesBeforeSending names the anthropicProfileRows on which
// anthropic-sdk-go's Messages.New must fail with neither recorder having
// seen a request: an ANTHROPIC_PROFILE naming a missing or unparsable
// profile, an empty ANTHROPIC_PROFILE, and env federation whose identity
// token is read from an empty ANTHROPIC_IDENTITY_TOKEN_FILE path.
var sdkRefusesBeforeSending = map[string]bool{
	"ANTHROPIC_PROFILE naming a missing file plus dotfile":                                                            true,
	"ANTHROPIC_PROFILE naming an unparsable file plus dotfile":                                                        true,
	"empty ANTHROPIC_PROFILE plus dotfile":                                                                            true,
	"env federation selected by ANTHROPIC_IDENTITY_TOKEN reads the empty ANTHROPIC_IDENTITY_TOKEN_FILE, plus dotfile": true,
}

// writeAnthropicCredentials gives every profile under the SDK's default
// config directory a credentials/<profile>.json the SDK will accept, so a
// row whose SDK routing lands on a profile actually sends its request
// instead of failing on missing credentials before the transport.
func writeAnthropicCredentials(t *testing.T) {
	t.Helper()
	dir := config.DefaultDir()
	profiles, err := filepath.Glob(filepath.Join(dir, "configs", "*.json"))
	if err != nil {
		t.Fatalf("glob profiles: %v", err)
	}
	for _, p := range profiles {
		name := strings.TrimSuffix(filepath.Base(p), ".json")
		writeFile(t, config.ProfileCredentialsPath(dir, name),
			[]byte(`{"type":"oauth_token","access_token":"tok-fixture"}`))
	}
}

// TestAnthropicProfileBaseURLFile_MatchesSDKRouting drives anthropic-sdk-go's
// own default routing — anthropic.NewClient with no options but a recording
// transport and no retries — over every anthropicProfileRows environment and
// asserts, row by row, that anthropicProfileBaseURLFile names a profile file
// exactly when a request the SDK made, to the routing transport or to the
// http.DefaultTransport guard, targeted the profile's base_url host.
//
// Hermetic: HOME and the SDK config directory are per-row temp dirs with
// credentials-file fixtures, and http.DefaultTransport is swapped for
// defaultTransportGuard before the client is built. Neither the routing
// transport nor the guard dials. On every row outside
// sdkRefusesBeforeSending, Messages.New must return no error and one of the
// two must have recorded its /v1/messages request. The rows in
// sdkRefusesBeforeSending must fail with nothing recorded. At least one
// row's token exchange must reach the guard. Not parallel: it
// swaps http.DefaultTransport and the process environment.
func TestAnthropicProfileBaseURLFile_MatchesSDKRouting(t *testing.T) {
	isolateEnv(t, anthropicEnvSources...)

	rowNames := map[string]bool{}
	for _, row := range anthropicProfileRows {
		rowNames[row.name] = true
	}
	for name := range sdkRefusesBeforeSending {
		if !rowNames[name] {
			t.Errorf("sdkRefusesBeforeSending names %q, which is not an anthropicProfileRows row", name)
		}
	}

	var routed, direct, exchanged int
	for _, row := range anthropicProfileRows {
		t.Run(row.name, func(t *testing.T) {
			home := t.TempDir()
			t.Setenv("HOME", home)
			wantFile := row.setup(t, home)
			writeAnthropicCredentials(t)

			guard := &defaultTransportGuard{api: recordingTransport{wireProvider: "anthropic"}}
			orig := http.DefaultTransport
			http.DefaultTransport = guard
			t.Cleanup(func() { http.DefaultTransport = orig })

			rt := &recordingTransport{wireProvider: "anthropic"}
			client := anthropic.NewClient(
				option.WithHTTPClient(&http.Client{Transport: rt}),
				option.WithMaxRetries(0),
			)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_, err := client.Messages.New(ctx, anthropic.MessageNewParams{
				Model:     "claude-test",
				MaxTokens: 1,
				Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
			})
			viaClient, viaGuard := rt.requests(), guard.requests()
			t.Logf("routing transport %v, http.DefaultTransport guard %v, err %v",
				requestTargets(viaClient), requestTargets(viaGuard), err)
			recorded := append(viaClient, viaGuard...)

			if sdkRefusesBeforeSending[row.name] {
				if err == nil || len(recorded) != 0 {
					t.Errorf("want Messages.New to fail before sending anything; err %v, recorded %v", err, requestTargets(recorded))
				}
			} else {
				if err != nil {
					t.Errorf("Messages.New: %v; the request whose result Messages.New returns must reach the routing transport or the http.DefaultTransport guard, and both answer it", err)
				}
				sentMessages := false
				for _, req := range recorded {
					if strings.HasSuffix(req.URL.Path, "/v1/messages") {
						sentMessages = true
					}
				}
				if !sentMessages {
					t.Errorf("neither recorder saw the /v1/messages request; recorded %v", requestTargets(recorded))
				}
			}
			for _, req := range viaGuard {
				if isTokenExchange(req) {
					exchanged++
					break
				}
			}

			sdkRoutedToProfile := false
			for _, req := range recorded {
				if req.URL.Hostname() == profilePoisonHost {
					sdkRoutedToProfile = true
				}
			}
			file := anthropicProfileBaseURLFile()
			if (file != "") != sdkRoutedToProfile {
				t.Errorf("anthropicProfileBaseURLFile() = %q but the SDK routed to the profile base_url = %v", file, sdkRoutedToProfile)
			}
			// The row's own expectation must agree with the SDK too, so a
			// fixture the SDK cannot use (and therefore never routes) cannot
			// make the comparison above vacuous.
			if (wantFile != "") != sdkRoutedToProfile {
				t.Errorf("row expects a refusal file %q but the SDK routed to the profile base_url = %v", wantFile, sdkRoutedToProfile)
			}
			if sdkRoutedToProfile {
				routed++
			} else {
				direct++
			}
		})
	}
	if routed == 0 || direct == 0 {
		t.Errorf("table exercised %d profile-routed and %d directly routed rows; a differential needs both", routed, direct)
	}
	if exchanged == 0 {
		t.Error("no row's token exchange reached the http.DefaultTransport guard")
	}
}

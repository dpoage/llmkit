package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
)

// anthropicTextSSE is a minimal successful Anthropic Messages stream.
const anthropicTextSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// recordingTransport records every outgoing *http.Request a client under
// test sends and answers each with a canned success body in wireProvider's
// format ("anthropic", "openai", "openai-compatible", or "google") — an SSE
// stream when an anthropic request asks for one. It never dials.
type recordingTransport struct {
	wireProvider string

	mu   sync.Mutex
	reqs []*http.Request
}

func (rt *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	body, contentType := mockTextBody(rt.wireProvider, "ok", 1, 1), "application/json"
	if req.Body != nil {
		sent, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		if rt.wireProvider == "anthropic" && bytes.Contains(sent, []byte(`"stream":true`)) {
			body, contentType = anthropicTextSSE, "text/event-stream"
		}
	}
	rt.mu.Lock()
	rt.reqs = append(rt.reqs, req)
	rt.mu.Unlock()
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     http.Header{"Content-Type": {contentType}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}, nil
}

// observed returns the most recent request, or nil when none was sent.
func (rt *recordingTransport) observed() *http.Request {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if len(rt.reqs) == 0 {
		return nil
	}
	return rt.reqs[len(rt.reqs)-1]
}

func (rt *recordingTransport) sent() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return len(rt.reqs)
}

// unsetEnvForTest ensures name is absent for the duration of t, restoring
// whatever value (or absence) the host process had. t.Setenv has no unset
// form, and several reads key off os.LookupEnv's ok result rather than the
// value, so an empty-but-present variable is not equivalent to an absent one.
func unsetEnvForTest(t *testing.T, name string) {
	t.Helper()
	old, had := os.LookupEnv(name)
	if had {
		if err := os.Unsetenv(name); err != nil {
			t.Fatalf("unsetenv %s: %v", name, err)
		}
		t.Cleanup(func() {
			if err := os.Setenv(name, old); err != nil {
				t.Errorf("restore %s: %v", name, err)
			}
		})
	}
}

// isolateBaseURLSources removes, for the duration of t, every ambient source
// New refuses an empty Spec.BaseURL on: the three base-URL variables and the
// Anthropic SDK profile files. A test that builds a vendor Type with an empty
// Spec.BaseURL calls it so it does not depend on the calling shell or its
// dotfiles. ANTHROPIC_CONFIG_DIR takes precedence over XDG_CONFIG_HOME and
// $HOME in the SDK's profile lookup, and the empty directory holds no profile.
func isolateBaseURLSources(t *testing.T) {
	t.Helper()
	for _, name := range []string{"OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "GOOGLE_GEMINI_BASE_URL"} {
		unsetEnvForTest(t, name)
	}
	t.Setenv("ANTHROPIC_CONFIG_DIR", t.TempDir())
}

// anthropicEnvSources lists every variable the Anthropic SDK's default
// options consult, plus XDG_CONFIG_HOME for its config directory.
var anthropicEnvSources = []string{
	"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS",
	"ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_BASE_URL",
	"ANTHROPIC_WEBHOOK_SIGNING_KEY", "ANTHROPIC_FEDERATION_RULE_ID",
	"ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_WORKSPACE_ID",
	"ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_IDENTITY_TOKEN",
	"ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_SCOPE", "XDG_CONFIG_HOME",
}

// writeAnthropicProfile writes a minimal SDK profile config file
// (configs/<profile>.json under dir). baseURL and workspaceID are written
// only when non-empty. AuthenticationInfo.Type "user_oauth" needs no other
// field.
func writeAnthropicProfile(t *testing.T, dir, profile, baseURL, workspaceID string) string {
	t.Helper()
	cfg := map[string]any{"authentication": map[string]any{"type": "user_oauth"}}
	if baseURL != "" {
		cfg["base_url"] = baseURL
	}
	if workspaceID != "" {
		cfg["workspace_id"] = workspaceID
	}
	body, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal profile: %v", err)
	}
	return writeFile(t, filepath.Join(dir, "configs", profile+".json"), body)
}

func writeFile(t *testing.T, path string, body []byte) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

// wireExpect is what one recorded request must show: the host, the
// credential header carrying exactly the Spec secret, the denied header
// names, and a poison marker no header name or value may contain.
type wireExpect struct {
	host       string
	credHeader string
	wantCred   string
	denied     []string
	marker     string
}

func (w wireExpect) check(t *testing.T, req *http.Request) {
	t.Helper()
	if req == nil {
		t.Fatal("no request observed; the adapter never reached the transport")
	}
	if req.URL.Host != w.host {
		t.Errorf("host = %q, want %q", req.URL.Host, w.host)
	}
	if got := req.Header.Get(w.credHeader); got != w.wantCred {
		t.Errorf("%s = %q, want %q (the Spec secret only)", w.credHeader, got, w.wantCred)
	}
	for _, name := range w.denied {
		if v := req.Header.Get(name); v != "" {
			t.Errorf("denied header %s = %q, must be absent", name, v)
		}
	}
	if w.marker == "" {
		return
	}
	for name, vals := range req.Header {
		if strings.Contains(strings.ToLower(name), strings.ToLower(w.marker)) {
			t.Errorf("header name %q contains poison marker %q", name, w.marker)
		}
		for _, v := range vals {
			if strings.Contains(v, w.marker) {
				t.Errorf("header %s = %q contains poison marker %q", name, v, w.marker)
			}
		}
	}
}

// poisonRow is one ambient-poison row: a single poison source, plus the
// substring that must not appear in any recorded header name or value.
type poisonRow struct {
	name   string
	setup  func(t *testing.T, home string)
	marker string
}

// hermeticGroup is one Type/Auth combination: how to build the Spec, the
// expected host, the credential header and its expected value, the group's
// denied header names, and its poison rows.
type hermeticGroup struct {
	name         string
	typ          Type
	auth         Auth
	secret       string
	baseURL      string // "" picks the vendor default; TypeOpenAICompatible always sets one
	host         string
	wireProvider string
	credHeader   string
	wantCred     string
	denied       []string
	rows         []poisonRow
}

// anthropicPoisonRows are the poison sources for both Anthropic Auth modes.
// The profile files carry a workspace_id and no base_url: a profile
// base_url is refused by New, which TestNew_RefusesAnthropicProfileBaseURL
// covers.
var anthropicPoisonRows = []poisonRow{
	{
		name:   "ANTHROPIC_API_KEY",
		marker: "poisoned-api-key-xyz",
		setup: func(t *testing.T, home string) {
			t.Setenv("ANTHROPIC_API_KEY", "poisoned-api-key-xyz")
		},
	},
	{
		name:   "ANTHROPIC_AUTH_TOKEN",
		marker: "poisoned-bearer-xyz",
		setup: func(t *testing.T, home string) {
			t.Setenv("ANTHROPIC_AUTH_TOKEN", "poisoned-bearer-xyz")
		},
	},
	{
		name:   "ANTHROPIC_CUSTOM_HEADERS",
		marker: "custom-header-poison",
		setup: func(t *testing.T, home string) {
			t.Setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Leak: custom-header-poison")
		},
	},
	{
		name:   "ANTHROPIC_PROFILE in ANTHROPIC_CONFIG_DIR",
		marker: "profile-poison",
		setup: func(t *testing.T, home string) {
			dir := t.TempDir()
			writeAnthropicProfile(t, dir, "leaky", "", "wrkspc-profile-poison")
			t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
			t.Setenv("ANTHROPIC_PROFILE", "leaky")
		},
	},
	{
		name:   "ANTHROPIC_CONFIG_DIR default profile",
		marker: "configdir-poison",
		setup: func(t *testing.T, home string) {
			dir := t.TempDir()
			writeAnthropicProfile(t, dir, "default", "", "wrkspc-configdir-poison")
			t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
		},
	},
	{
		name:   "ANTHROPIC_CONFIG_DIR active_config",
		marker: "active-poison",
		setup: func(t *testing.T, home string) {
			dir := t.TempDir()
			writeAnthropicProfile(t, dir, "staging", "", "wrkspc-active-poison")
			writeFile(t, filepath.Join(dir, "active_config"), []byte("staging\n"))
			t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
		},
	},
	{
		name:   "XDG_CONFIG_HOME default profile",
		marker: "xdg-poison",
		setup: func(t *testing.T, home string) {
			xdg := t.TempDir()
			writeAnthropicProfile(t, filepath.Join(xdg, "anthropic"), "default", "", "wrkspc-xdg-poison")
			t.Setenv("XDG_CONFIG_HOME", xdg)
		},
	},
	{
		name:   "HOME dotfile default profile",
		marker: "dotfile-poison",
		setup: func(t *testing.T, home string) {
			writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", "", "wrkspc-dotfile-poison")
		},
	},
}

// openAIPoisonRows are the poison sources for openai and openai-compatible.
var openAIPoisonRows = []poisonRow{
	{
		name:   "OPENAI_API_KEY",
		marker: "sk-poisoned-xyz",
		setup: func(t *testing.T, home string) {
			t.Setenv("OPENAI_API_KEY", "sk-poisoned-xyz")
		},
	},
	{
		name:   "OPENAI_ORG_ID",
		marker: "org-poison-xyz",
		setup: func(t *testing.T, home string) {
			t.Setenv("OPENAI_ORG_ID", "org-poison-xyz")
		},
	},
	{
		name:   "OPENAI_PROJECT_ID",
		marker: "proj-poison-xyz",
		setup: func(t *testing.T, home string) {
			t.Setenv("OPENAI_PROJECT_ID", "proj-poison-xyz")
		},
	},
	{
		name:   "OPENAI_CUSTOM_HEADERS",
		marker: "poison",
		setup: func(t *testing.T, home string) {
			t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Leak: poison")
		},
	},
}

// TestNew_HermeticConstruction pins, for every provider Type/Auth
// combination, that a client built by provider.New sends to Spec.BaseURL's
// host (or the vendor default when empty) with exactly the Spec credential
// and no header whose name or value carries an ambient environment or
// profile poison marker — one poison source at a time, HOME pointed at a
// fresh empty temp dir on every row.
func TestNew_HermeticConstruction(t *testing.T) {
	for _, name := range append([]string{
		"OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
		"OPENAI_CUSTOM_HEADERS", "OPENAI_BASE_URL", "OPENAI_ADMIN_KEY",
		"OPENAI_WEBHOOK_SECRET",
		"GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
		"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION",
		"GOOGLE_GEMINI_BASE_URL",
	}, anthropicEnvSources...) {
		unsetEnvForTest(t, name)
	}

	groups := []hermeticGroup{
		{
			name: "anthropic api-key", typ: TypeAnthropic, auth: AuthAPIKey,
			secret: "sk-ant-real-secret", host: "api.anthropic.com",
			wireProvider: "anthropic", credHeader: "X-Api-Key", wantCred: "sk-ant-real-secret",
			denied: []string{"Authorization", "Anthropic-Workspace-Id", "X-Leak"},
			rows:   anthropicPoisonRows,
		},
		{
			name: "anthropic oauth", typ: TypeAnthropic, auth: AuthOAuthToken,
			secret: "oauth-real-secret-token", host: "api.anthropic.com",
			wireProvider: "anthropic", credHeader: "Authorization", wantCred: "Bearer oauth-real-secret-token",
			denied: []string{"X-Api-Key", "Anthropic-Workspace-Id", "X-Leak"},
			rows:   anthropicPoisonRows,
		},
		{
			name: "openai", typ: TypeOpenAI, auth: AuthAPIKey,
			secret: "sk-openai-real-secret", host: "api.openai.com",
			wireProvider: "openai", credHeader: "Authorization", wantCred: "Bearer sk-openai-real-secret",
			denied: []string{"OpenAI-Organization", "OpenAI-Project", "X-Leak"},
			rows:   openAIPoisonRows,
		},
		{
			name: "openai-compatible", typ: TypeOpenAICompatible, auth: AuthAPIKey,
			secret: "compat-real-secret", baseURL: "https://compat.example.com", host: "compat.example.com",
			wireProvider: "openai-compatible", credHeader: "Authorization", wantCred: "Bearer compat-real-secret",
			denied: []string{"OpenAI-Organization", "OpenAI-Project", "X-Leak"},
			rows:   openAIPoisonRows,
		},
		{
			name: "google", typ: TypeGoogle, auth: AuthAPIKey,
			secret: "google-real-secret", host: "generativelanguage.googleapis.com",
			wireProvider: "google", credHeader: "X-Goog-Api-Key", wantCred: "google-real-secret",
			rows: []poisonRow{
				{
					name:   "GOOGLE_API_KEY",
					marker: "goog-poisoned-xyz",
					setup: func(t *testing.T, home string) {
						t.Setenv("GOOGLE_API_KEY", "goog-poisoned-xyz")
					},
				},
				{
					name:   "GEMINI_API_KEY",
					marker: "gemini-poisoned-xyz",
					setup: func(t *testing.T, home string) {
						t.Setenv("GEMINI_API_KEY", "gemini-poisoned-xyz")
					},
				},
				{
					name:   "GOOGLE_GENAI_USE_VERTEXAI",
					marker: "poison-project",
					setup: func(t *testing.T, home string) {
						t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
						t.Setenv("GOOGLE_CLOUD_PROJECT", "poison-project")
						t.Setenv("GOOGLE_CLOUD_LOCATION", "poison-location")
					},
				},
			},
		},
	}

	for _, g := range groups {
		t.Run(g.name, func(t *testing.T) {
			for _, row := range g.rows {
				t.Run(row.name, func(t *testing.T) {
					home := t.TempDir()
					t.Setenv("HOME", home)
					row.setup(t, home)

					rt := &recordingTransport{wireProvider: g.wireProvider}
					spec := Spec{Type: g.typ, Model: "test-model", Secret: g.secret, Auth: g.auth, BaseURL: g.baseURL}
					client, err := New(context.Background(), spec, Options{HTTPClient: &http.Client{Transport: rt}})
					if err != nil {
						t.Fatalf("New: %v", err)
					}
					if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
						t.Fatalf("Complete: %v", err)
					}
					wireExpect{
						host: g.host, credHeader: g.credHeader, wantCred: g.wantCred,
						denied: g.denied, marker: row.marker,
					}.check(t, rt.observed())
				})
			}
		})
	}
}

// TestNew_RefusesEmptyBaseURLWithEnvVarSet pins, for each Type with a
// vendor default, that an empty Spec.BaseURL is refused when the Type's
// own base-URL variable is present, an empty value included — the
// refusal names the variable and Spec.BaseURL, and sends nothing; an
// explicit Spec.BaseURL is honored and the variable ignored; an absent
// variable reaches the vendor default; and another Type's variable never
// refuses.
func TestNew_RefusesEmptyBaseURLWithEnvVarSet(t *testing.T) {
	cases := []struct {
		typ          Type
		envVar       string
		vendorHost   string
		wireProvider string
	}{
		{TypeOpenAI, "OPENAI_BASE_URL", "api.openai.com", "openai"},
		{TypeAnthropic, "ANTHROPIC_BASE_URL", "api.anthropic.com", "anthropic"},
		{TypeGoogle, "GOOGLE_GEMINI_BASE_URL", "generativelanguage.googleapis.com", "google"},
	}
	isolateBaseURLSources(t)

	for _, tc := range cases {
		t.Run(string(tc.typ), func(t *testing.T) {
			for _, value := range []string{"https://env-routed-gateway.invalid", ""} {
				t.Run(fmt.Sprintf("var set to %q, BaseURL empty: refused, nothing sent", value), func(t *testing.T) {
					t.Setenv(tc.envVar, value)
					rt := &recordingTransport{wireProvider: tc.wireProvider}
					spec := Spec{Type: tc.typ, Model: "test-model", Secret: "k"}
					client, err := New(context.Background(), spec, Options{HTTPClient: &http.Client{Transport: rt}})
					if client != nil {
						t.Error("New returned a non-nil client")
					}
					if !errors.Is(err, llmkit.ErrInvalidRequest) {
						t.Fatalf("error = %v, want ErrInvalidRequest", err)
					}
					if !strings.Contains(err.Error(), tc.envVar) {
						t.Errorf("error %q must name %q", err, tc.envVar)
					}
					if !strings.Contains(err.Error(), "Spec.BaseURL") {
						t.Errorf("error %q must name Spec.BaseURL", err)
					}
					if rt.sent() != 0 {
						t.Error("a request reached the transport; New must send nothing on refusal")
					}
				})
			}

			t.Run("var set, BaseURL also set: succeeds, goes to Spec.BaseURL", func(t *testing.T) {
				t.Setenv(tc.envVar, "https://env-routed-gateway.invalid")
				rt := &recordingTransport{wireProvider: tc.wireProvider}
				spec := Spec{Type: tc.typ, Model: "test-model", Secret: "k", BaseURL: "https://explicit.example.com"}
				client, err := New(context.Background(), spec, Options{HTTPClient: &http.Client{Transport: rt}})
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
					t.Fatalf("Complete: %v", err)
				}
				req := rt.observed()
				if req == nil {
					t.Fatal("no request observed")
				}
				if req.URL.Host != "explicit.example.com" {
					t.Errorf("host = %q, want explicit.example.com (Spec.BaseURL, the env var ignored)", req.URL.Host)
				}
			})

			t.Run("var unset, BaseURL empty: succeeds, goes to vendor default", func(t *testing.T) {
				unsetEnvForTest(t, tc.envVar)
				rt := &recordingTransport{wireProvider: tc.wireProvider}
				spec := Spec{Type: tc.typ, Model: "test-model", Secret: "k"}
				client, err := New(context.Background(), spec, Options{HTTPClient: &http.Client{Transport: rt}})
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
					t.Fatalf("Complete: %v", err)
				}
				req := rt.observed()
				if req == nil {
					t.Fatal("no request observed")
				}
				if req.URL.Host != tc.vendorHost {
					t.Errorf("host = %q, want %q", req.URL.Host, tc.vendorHost)
				}
			})
		})
	}

	t.Run("another Type's variable never refuses", func(t *testing.T) {
		t.Setenv("OPENAI_BASE_URL", "https://env-routed-gateway.invalid")
		rt := &recordingTransport{wireProvider: "anthropic"}
		spec := Spec{Type: TypeAnthropic, Model: "test-model", Secret: "k"}
		client, err := New(context.Background(), spec, Options{HTTPClient: &http.Client{Transport: rt}})
		if err != nil {
			t.Fatalf("New: %v (OPENAI_BASE_URL must not refuse TypeAnthropic)", err)
		}
		if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		req := rt.observed()
		if req == nil {
			t.Fatal("no request observed")
		}
		if req.URL.Host != "api.anthropic.com" {
			t.Errorf("host = %q, want api.anthropic.com", req.URL.Host)
		}
	})
}

const (
	profilePoisonHost = "poison-profile.invalid"
	profilePoisonURL  = "https://" + profilePoisonHost
)

// anthropicProfileRows are the profile-baseURL rows. Each setup returns
// the profile file New must name, or "" when New must not refuse.
var anthropicProfileRows = []struct {
	name  string
	setup func(t *testing.T, home string) string
}{
	{"ANTHROPIC_PROFILE in ANTHROPIC_CONFIG_DIR", func(t *testing.T, home string) string {
		dir := t.TempDir()
		t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
		t.Setenv("ANTHROPIC_PROFILE", "work")
		return writeAnthropicProfile(t, dir, "work", profilePoisonURL, "wrkspc-poison")
	}},
	{"ANTHROPIC_CONFIG_DIR default profile", func(t *testing.T, home string) string {
		dir := t.TempDir()
		t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
		return writeAnthropicProfile(t, dir, "default", profilePoisonURL, "wrkspc-poison")
	}},
	{"ANTHROPIC_CONFIG_DIR active_config", func(t *testing.T, home string) string {
		dir := t.TempDir()
		t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
		writeAnthropicProfile(t, dir, "default", "", "")
		writeFile(t, filepath.Join(dir, "active_config"), []byte("  staging\n"))
		return writeAnthropicProfile(t, dir, "staging", profilePoisonURL, "wrkspc-poison")
	}},
	{"XDG_CONFIG_HOME default profile", func(t *testing.T, home string) string {
		xdg := t.TempDir()
		t.Setenv("XDG_CONFIG_HOME", xdg)
		return writeAnthropicProfile(t, filepath.Join(xdg, "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
	}},
	{"HOME dotfile default profile", func(t *testing.T, home string) string {
		return writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
	}},
	{"empty ANTHROPIC_API_KEY plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_API_KEY", "")
		return writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
	}},
	{"partial env federation plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl-poison")
		t.Setenv("ANTHROPIC_ORGANIZATION_ID", "org-poison")
		return writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
	}},
	{"dotfile without base_url", func(t *testing.T, home string) string {
		writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", "", "wrkspc-poison")
		return ""
	}},
	{"ANTHROPIC_API_KEY plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_API_KEY", "sk-env-poison")
		writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
		return ""
	}},
	{"ANTHROPIC_AUTH_TOKEN plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "bearer-env-poison")
		writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
		return ""
	}},
	{"complete env federation plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl-poison")
		t.Setenv("ANTHROPIC_ORGANIZATION_ID", "org-poison")
		t.Setenv("ANTHROPIC_IDENTITY_TOKEN", "jwt-poison")
		writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
		return ""
	}},
	{"ANTHROPIC_PROFILE naming a missing file plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_PROFILE", "missing")
		writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
		return ""
	}},
	{"ANTHROPIC_PROFILE naming an unparsable file plus dotfile", func(t *testing.T, home string) string {
		dir := filepath.Join(home, ".config", "anthropic")
		t.Setenv("ANTHROPIC_PROFILE", "broken")
		writeFile(t, filepath.Join(dir, "configs", "broken.json"), []byte(`{"base_url": "`+profilePoisonURL+`"`))
		writeAnthropicProfile(t, dir, "default", profilePoisonURL, "wrkspc-poison")
		return ""
	}},
	{"empty ANTHROPIC_PROFILE plus dotfile", func(t *testing.T, home string) string {
		t.Setenv("ANTHROPIC_PROFILE", "")
		writeAnthropicProfile(t, filepath.Join(home, ".config", "anthropic"), "default", profilePoisonURL, "wrkspc-poison")
		return ""
	}},
}

// TestNew_RefusesAnthropicProfileBaseURL pins, with an empty
// Spec.BaseURL, that TypeAnthropic is refused exactly when
// anthropic-sdk-go's default options would have applied a profile file's
// base_url. The refusal wraps ErrInvalidRequest, names the profile file
// and Spec.BaseURL, never echoes the base_url, and sends nothing. Every
// other row sends, from Complete and from Stream, to the vendor host with
// the Spec secret and no Anthropic-Workspace-Id. With Spec.BaseURL set, no
// row is refused and every request goes to Spec.BaseURL.
func TestNew_RefusesAnthropicProfileBaseURL(t *testing.T) {
	for _, name := range anthropicEnvSources {
		unsetEnvForTest(t, name)
	}
	modes := []struct {
		name       string
		auth       Auth
		credHeader string
		wantCred   string
		denied     []string
	}{
		{"api-key", AuthAPIKey, "X-Api-Key", "sk-ant-real-secret", []string{"Authorization", "Anthropic-Workspace-Id"}},
		{"oauth", AuthOAuthToken, "Authorization", "Bearer sk-ant-real-secret", []string{"X-Api-Key", "Anthropic-Workspace-Id"}},
	}

	for _, mode := range modes {
		for _, baseURL := range []string{"", "https://explicit.example.com"} {
			for _, row := range anthropicProfileRows {
				name := mode.name + "/BaseURL empty/" + row.name
				if baseURL != "" {
					name = mode.name + "/BaseURL set/" + row.name
				}
				t.Run(name, func(t *testing.T) {
					home := t.TempDir()
					t.Setenv("HOME", home)
					file := row.setup(t, home)
					if baseURL != "" {
						file = ""
					}

					rt := &recordingTransport{wireProvider: "anthropic"}
					spec := Spec{Type: TypeAnthropic, Model: "test-model", Secret: "sk-ant-real-secret", Auth: mode.auth, BaseURL: baseURL}
					client, err := New(context.Background(), spec, Options{HTTPClient: &http.Client{Transport: rt}})
					if file != "" {
						if client != nil {
							t.Error("New returned a non-nil client")
						}
						if !errors.Is(err, llmkit.ErrInvalidRequest) {
							t.Fatalf("error = %v, want ErrInvalidRequest", err)
						}
						for _, want := range []string{file, "Spec.BaseURL"} {
							if !strings.Contains(err.Error(), want) {
								t.Errorf("error %q must name %q", err, want)
							}
						}
						if strings.Contains(err.Error(), profilePoisonHost) {
							t.Errorf("error %q echoes the profile's base_url", err)
						}
						if rt.sent() != 0 {
							t.Errorf("%d request(s) reached the transport; New must send nothing on refusal", rt.sent())
						}
						return
					}
					if err != nil {
						t.Fatalf("New: %v", err)
					}
					want := wireExpect{
						host: "api.anthropic.com", credHeader: mode.credHeader, wantCred: mode.wantCred,
						denied: mode.denied, marker: "poison",
					}
					if baseURL != "" {
						want.host = "explicit.example.com"
					}
					if _, err := client.Complete(context.Background(), simpleRequest()); err != nil {
						t.Fatalf("Complete: %v", err)
					}
					want.check(t, rt.observed())
					if _, err := llmkit.Stream(context.Background(), client, simpleRequest(), func(llmkit.Delta) error { return nil }); err != nil {
						t.Fatalf("Stream: %v", err)
					}
					want.check(t, rt.observed())
					if rt.sent() != 2 {
						t.Errorf("requests sent = %d, want 2 (one Complete, one Stream)", rt.sent())
					}
				})
			}
		}
	}
}

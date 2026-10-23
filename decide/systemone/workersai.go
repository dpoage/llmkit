package systemone

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/retry"
)

// defaultWorkersAIBaseURL is the Cloudflare API root, the default for
// WorkersAIConfig.BaseURL.
const defaultWorkersAIBaseURL = "https://api.cloudflare.com"

// accountIDLen is the length of a Cloudflare account id: lowercase hex.
const accountIDLen = 32

// WorkersAIConfig configures a System One client served by Cloudflare
// Workers AI. NewWorkersAI validates it.
type WorkersAIConfig struct {
	// Secret is the Cloudflare API token, sent as "Authorization: Bearer".
	// The same rule as [Config.Secret].
	Secret string

	// AccountID is the Cloudflare account id: exactly 32 lowercase
	// hexadecimal characters. Required.
	AccountID string

	// Model is the Workers AI model name under @cf/cloudflare: "clef",
	// "clef-flash" or "clef-omni". Required; it matches
	// ^[a-z0-9][a-z0-9.-]*$.
	Model string

	// BaseURL overrides the Cloudflare API root. Empty selects
	// "https://api.cloudflare.com". The same rules as [Config.BaseURL].
	BaseURL string

	// HTTPClient is optional; the same rule as [Config.HTTPClient].
	HTTPClient *http.Client

	// Retry tunes the shared retry policy; the same rule as [Config.Retry].
	Retry retry.Config
}

// NewWorkersAI validates cfg and returns a Client for Cloudflare's hosted
// System One models, POSTing to
// <BaseURL>/client/v4/accounts/<AccountID>/ai/run/@cf/cloudflare/<Model>.
// Images always ride in the [ImagesArray] shape. A bad cfg returns an error
// wrapping llmkit.ErrInvalidRequest that never echoes Secret or AccountID.
func NewWorkersAI(cfg WorkersAIConfig) (*Client, error) {
	if !validAccountID(cfg.AccountID) {
		return nil, fmt.Errorf("systemone: WorkersAIConfig.AccountID must be exactly %d lowercase hexadecimal characters: %w", accountIDLen, llmkit.ErrInvalidRequest)
	}
	if !validWorkersAIModel(cfg.Model) {
		return nil, fmt.Errorf("systemone: WorkersAIConfig.Model must match ^[a-z0-9][a-z0-9.-]*$: %w", llmkit.ErrInvalidRequest)
	}
	base := cfg.BaseURL
	if base == "" {
		base = defaultWorkersAIBaseURL
	}
	c, err := newClient(Config{
		Secret:     cfg.Secret,
		Model:      cfg.Model,
		BaseURL:    base,
		HTTPClient: cfg.HTTPClient,
		Retry:      cfg.Retry,
		Images:     ImagesArray,
	}, "/client/v4/accounts/"+cfg.AccountID+"/ai/run/@cf/cloudflare/"+cfg.Model)
	if err != nil {
		return nil, err
	}
	c.cloudflare = true
	return c, nil
}

func validAccountID(s string) bool {
	if len(s) != accountIDLen {
		return false
	}
	for i := range len(s) {
		if c := s[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func validWorkersAIModel(s string) bool {
	if s == "" {
		return false
	}
	for i := range len(s) {
		c := s[i]
		if c >= 'a' && c <= 'z' || c >= '0' && c <= '9' {
			continue
		}
		if i > 0 && (c == '.' || c == '-') {
			continue
		}
		return false
	}
	return true
}

// cfEnvelope is Cloudflare's response wrapper; Success is a pointer so an
// absent member differs from false.
type cfEnvelope struct {
	Result  json.RawMessage `json:"result"`
	Success *bool           `json:"success"`
	Errors  []cfError       `json:"errors"`
}

type cfError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// describeCFErrors renders each error as "code <n>: <message>", joined by "; ".
func describeCFErrors(errs []cfError) string {
	parts := make([]string, len(errs))
	for i, e := range errs {
		parts[i] = fmt.Sprintf("code %d: %s", e.Code, e.Message)
	}
	return strings.Join(parts, "; ")
}

// cloudflareErrorMessage is the Message source for a non-200 reply: the
// envelope's errors, or body itself when it is not an envelope with errors.
func cloudflareErrorMessage(body []byte) string {
	var env cfEnvelope
	if err := json.Unmarshal(body, &env); err != nil || len(env.Errors) == 0 {
		return string(body)
	}
	return describeCFErrors(env.Errors)
}

// unwrapResult returns the "result" member of a 200 reply for
// parseResponse. An explicit success:false, or a null or missing result, is
// an ErrServer naming the Cloudflare errors.
func unwrapResult(status int, body []byte) ([]byte, error) {
	var env cfEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, serverError(status, fmt.Sprintf("response: malformed JSON: %s", err))
	}
	failed := env.Success != nil && !*env.Success
	if failed || len(env.Result) == 0 || bytes.Equal(env.Result, []byte("null")) {
		msg := "cloudflare: reply has no result"
		if failed {
			msg = "cloudflare: success is false"
		}
		if len(env.Errors) > 0 {
			msg += ": " + describeCFErrors(env.Errors)
		}
		return nil, serverError(status, msg)
	}
	return env.Result, nil
}

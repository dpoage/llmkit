package decide

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/internal/adapter"
	"github.com/dpoage/llmkit/retry"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Questions indexes one Ask's question set by id. Ids are echoed back in the
// response's answer maps.
type Questions map[string]Question

// Response is the normalized result of one Ask.
type Response struct {
	// Model is the versioned id the server actually used; it can differ from
	// a requested alias ("jev-latest" -> "jev-1.13.0"). Ledger on this.
	Model string
	// Usage is token consumption. TypeSafe bills input tokens only.
	Usage llmkit.Usage
	// Nouls holds one entry per Noul question. nil when no Nouls were asked.
	Nouls map[string]float64
	// Choices holds one entry per Choice question. nil when none were asked.
	Choices map[string]ChoiceAnswer
	// Scores holds one entry per Score question. nil when none were asked.
	Scores map[string]ScoreAnswer
}

// ChoiceAnswer is the answer to a Choice question. Probabilities map option
// id to probability, passed through exactly as reported.
type ChoiceAnswer struct {
	Choice        string
	Probabilities map[string]float64
	Confidence    float64
}

// ScoreAnswer is the answer to a Score question. Levels is the server's
// legend ordered by index ("0".."n-1", with n equal to the levels asked
// for), and Probabilities is index-aligned with Levels.
type ScoreAnswer struct {
	Score         float64
	Levels        []string
	Probabilities []float64
	Confidence    float64
}

// wireRequest is the JSON envelope POSTed to /v1/systemone.
type wireRequest struct {
	State     json.RawMessage            `json:"state"`
	Model     string                     `json:"model"`
	Questions map[string]json.RawMessage `json:"questions"`
}

// wireResponse is the JSON envelope the endpoint returns. Model and Usage
// are vendor-required; a 200 body missing either is a server contract
// violation, not a silent zero.
type wireResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   *wireUsage                 `json:"usage"`
}

type wireUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// answerBody decodes any answer kind. The pointer fields distinguish
// present from absent so a missing value is a named contract violation.
type answerBody struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul"`
	Choice        *string            `json:"choice"`
	Score         *float64           `json:"score"`
	Probabilities map[string]float64 `json:"probabilities"`
	Legend        map[string]string  `json:"legend"`
	Confidence    float64            `json:"confidence"`
}

// statusAttempt carries one HTTP attempt's classified APIError plus the
// Retry-After presence bit: llmkit.APIError records the parsed delay but
// not presence, and presence — including a 0 or past-date HTTP-date, which
// ParseRetryAfter clamps to 0 — is the embed-parity signal that the retry
// loop should ignore exponential backoff. Ask unwraps the APIError.
type statusAttempt struct {
	*llmkit.APIError
	hasRetryAfter bool
}

// Ask evaluates state against questions in one request: validates and
// marshals the envelope, POSTs to <BaseURL>/v1/systemone with an
// "Authorization: Bearer" header, retries transient failures, validates the
// response, and reports usage to the configured Recorder — once, on success.
// ctx bounds the call; its cancellation is never retried. Every returned
// error is an *llmkit.APIError with Provider "typesafe" (see the package doc
// for the status mapping).
func (c *Client) Ask(ctx context.Context, state any, questions Questions) (Response, error) {
	body, err := buildRequest(state, c.model, questions)
	if err != nil {
		return Response{}, err
	}
	var resp Response
	err = retry.Do(ctx, c.retry, classifyRetryable, func(actx context.Context) error {
		r, err := c.attempt(actx, body, questions)
		if err != nil {
			return err
		}
		resp = r
		return nil
	})
	if err != nil {
		var st *statusAttempt
		if errors.As(err, &st) {
			return Response{}, st.APIError
		}
		return Response{}, err
	}
	if c.recorder != nil {
		c.recorder.Record(llmkit.UsageEvent{
			Provider: providerName,
			Model:    resp.Model,
			Usage:    resp.Usage,
		})
	}
	return resp, nil
}

// attempt performs one HTTP round trip and response parse. body is built
// once per Ask and reused across attempts; questions is read-only.
func (c *Client) attempt(ctx context.Context, body []byte, questions Questions) (Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return Response{}, serverError(0, fmt.Sprintf("build request: %s", err))
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	httpResp, err := c.client.Do(req)
	if err != nil {
		// url.Error text carries the URL and cause, never the Authorization
		// header, so the API key cannot leak through it.
		return Response{}, &llmkit.APIError{Kind: llmkit.ErrServer, StatusCode: 0, Provider: providerName, Message: "request failed", Err: err}
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return Response{}, &llmkit.APIError{Kind: llmkit.ErrServer, StatusCode: 0, Provider: providerName, Message: "read response", Err: err}
	}

	if httpResp.StatusCode != http.StatusOK {
		return Response{}, newAPIStatusError(httpResp, respBody)
	}
	return parseResponse(httpResp.StatusCode, respBody, questions)
}

// newAPIStatusError maps a non-200 response to a sentinel-classified
// *llmkit.APIError, wrapped with the Retry-After presence bit. The header
// is parsed for every status; the retry loop honors it whenever the status
// is retried. The Message carries the truncated vendor body text — never
// the request's credential.
func newAPIStatusError(resp *http.Response, body []byte) error {
	kind := adapter.ClassifyStatus(resp.StatusCode, string(body))
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = fmt.Sprintf("HTTP %d", resp.StatusCode)
	}
	apiErr := &llmkit.APIError{
		Kind:       kind,
		StatusCode: resp.StatusCode,
		Provider:   providerName,
		Message:    truncate(msg, 200),
	}
	after, hasAfter := retry.ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	apiErr.RetryAfter = after
	return &statusAttempt{APIError: apiErr, hasRetryAfter: hasAfter}
}

// parseResponse validates the answer set against the questions and converts
// it to a Response. A violation of the server's own answer contract — a
// missing model or usage block, a missing or extra answer, a type that
// does not match its question, or a sparse legend — is an ErrServer-class
// APIError at the serving status: the server broke the protocol.
func parseResponse(status int, body []byte, questions Questions) (Response, error) {
	var wire wireResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return Response{}, serverError(status, fmt.Sprintf("response: malformed JSON: %s", err))
	}
	if wire.Model == "" {
		return Response{}, serverError(status, "model: missing model id")
	}
	if wire.Usage == nil {
		return Response{}, serverError(status, "usage: missing usage block")
	}

	nouls := make(map[string]float64)
	choices := make(map[string]ChoiceAnswer)
	scores := make(map[string]ScoreAnswer)
	for id, raw := range wire.Answers {
		q, ok := questions[id]
		if !ok {
			return Response{}, serverError(status, fmt.Sprintf("answers[%q]: answer for an unknown question id", id))
		}
		var ab answerBody
		if err := json.Unmarshal(raw, &ab); err != nil {
			return Response{}, serverError(status, fmt.Sprintf("answers[%q]: malformed answer: %s", id, err))
		}
		switch t := q.(type) {
		case Noul:
			if ab.Type != "noul" {
				return Response{}, serverError(status, fmt.Sprintf("answers[%q].type: want \"noul\", got %q", id, ab.Type))
			}
			if ab.Noul == nil {
				return Response{}, serverError(status, fmt.Sprintf("answers[%q].noul: missing noul value", id))
			}
			nouls[id] = *ab.Noul
		case Choice:
			if ab.Type != "choice" {
				return Response{}, serverError(status, fmt.Sprintf("answers[%q].type: want \"choice\", got %q", id, ab.Type))
			}
			if ab.Choice == nil {
				return Response{}, serverError(status, fmt.Sprintf("answers[%q].choice: missing choice value", id))
			}
			choices[id] = ChoiceAnswer{Choice: *ab.Choice, Probabilities: ab.Probabilities, Confidence: ab.Confidence}
		case Score:
			if ab.Type != "score" {
				return Response{}, serverError(status, fmt.Sprintf("answers[%q].type: want \"score\", got %q", id, ab.Type))
			}
			if ab.Score == nil {
				return Response{}, serverError(status, fmt.Sprintf("answers[%q].score: missing score value", id))
			}
			levels, err := denseSlice(status, id, ".legend", ab.Legend, len(t.Levels))
			if err != nil {
				return Response{}, err
			}
			probs, err := denseSlice(status, id, ".probabilities", ab.Probabilities, len(t.Levels))
			if err != nil {
				return Response{}, err
			}
			scores[id] = ScoreAnswer{Score: *ab.Score, Levels: levels, Probabilities: probs, Confidence: ab.Confidence}
		case nil:
			return Response{}, serverError(status, fmt.Sprintf("answers[%q]: answer for an unknown question id", id))
		}
	}
	for id := range questions {
		if _, ok := wire.Answers[id]; !ok {
			return Response{}, serverError(status, fmt.Sprintf("answers[%q]: missing answer for question", id))
		}
	}

	return Response{
		Model:   wire.Model,
		Usage:   llmkit.Usage{InputTokens: wire.Usage.InputTokens, OutputTokens: wire.Usage.OutputTokens},
		Nouls:   nonEmptyOrNil(nouls),
		Choices: nonEmptyOrNil(choices),
		Scores:  nonEmptyOrNil(scores),
	}, nil
}

// denseSlice converts a server map keyed by stringified legend index into
// the index-aligned slice the Response carries. Keys must be exactly
// "0".."n-1" — dense, in range, matching the levels asked for.
func denseSlice[T any](status int, id, field string, m map[string]T, n int) ([]T, error) {
	fail := func(problem string) ([]T, error) {
		return nil, serverError(status, fmt.Sprintf("answers[%q]%s: %s (want exactly %d entries keyed \"0\"..\"%d\")", id, field, problem, n, n-1))
	}
	if len(m) != n {
		return fail(fmt.Sprintf("got %d entries", len(m)))
	}
	out := make([]T, n)
	for i := range out {
		v, ok := m[strconv.Itoa(i)]
		if !ok {
			return fail(fmt.Sprintf("missing key %q", strconv.Itoa(i)))
		}
		out[i] = v
	}
	return out, nil
}

// classifyRetryable reports whether an Ask attempt is worth retrying,
// mirroring embed's boundary exactly: HTTP 429, every 5xx (529 included),
// and transport failures. Everything else — auth, invalid request,
// context-too-long, a 200 that violates the body contract — is terminal: a
// deterministic server-side violation must not burn the retry budget. When a
// retried status carried a Retry-After header, its delay replaces the
// computed backoff; presence alone — even the 0 a past HTTP-date clamps to
// — means an immediate retry.
func classifyRetryable(err error) (time.Duration, bool, bool) {
	var st *statusAttempt
	if errors.As(err, &st) {
		if !retryWorthy(st.APIError) {
			return 0, false, false
		}
		return st.RetryAfter, st.hasRetryAfter, true
	}
	var apiErr *llmkit.APIError
	if errors.As(err, &apiErr) {
		if !retryWorthy(apiErr) {
			return 0, false, false
		}
		return apiErr.RetryAfter, apiErr.RetryAfter > 0, true
	}
	return 0, false, false
}

// retryWorthy is the status boundary for retries: 429, any 5xx (529
// included), and transport failures — the only StatusCode-0 errors attempt
// produces, guarded by their ErrServer kind so a pre-wire validation error
// can never look transient.
func retryWorthy(apiErr *llmkit.APIError) bool {
	switch {
	case apiErr.StatusCode == http.StatusTooManyRequests, apiErr.StatusCode >= 500:
		return true
	case apiErr.StatusCode == 0:
		return apiErr.Kind == llmkit.ErrServer
	default:
		return false
	}
}

// serverError builds the ErrServer-class APIError: transport failures carry
// StatusCode 0; response contract violations carry the serving status.
func serverError(status int, message string) error {
	return &llmkit.APIError{
		Kind:       llmkit.ErrServer,
		StatusCode: status,
		Provider:   providerName,
		Message:    message,
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// nonEmptyOrNil returns nil for an empty map, so absence reads as absence.
func nonEmptyOrNil[K comparable, V any](m map[K]V) map[K]V {
	if len(m) == 0 {
		return nil
	}
	return m
}

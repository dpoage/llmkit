package decide

import (
	"bytes"
	"context"
	"encoding/json"
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

// Ask evaluates state against questions in one request: validates and
// marshals the envelope, POSTs to <BaseURL>/v1/systemone with an
// "Authorization: Bearer" header, retries transient failures, and
// validates the response. ctx bounds the call. Every error from Ask is an
// *llmkit.APIError with Provider "typesafe", except the caller's
// cancellation or a deadline it set, which is an error chaining the
// context error (errors.Is(err, ctx.Err()); never retryable) — see the
// package doc for the status mapping.
//
// When [Config.Observer] is set, Ask emits exactly one
// [llmkit.DecisionEvent] for every Ask that passes pre-wire validation:
// on success, Backend "typesafe", Model the response's reported model,
// Usage, State and Questions as validated (sorted by ID), Answers
// sorted by ID; on failure, Err set, Answers nil, Model the requested
// alias. A pre-wire buildRequest refusal emits nothing; a caller's
// already-cancelled ctx emits one event with Err set and zero wire
// hits. The event rides the Ask's ctx — decide never mints a span.
func (c *Client) Ask(ctx context.Context, state any, questions Questions) (Response, error) {
	body, stateRaw, err := buildRequest(state, c.model, questions)
	if err != nil {
		return Response{}, err
	}
	start := time.Now()
	var resp Response
	err = retry.Do(ctx, c.retry, llmkit.Classify, func(actx context.Context) error {
		r, err := c.attempt(actx, body, questions)
		if err != nil {
			return err
		}
		resp = r
		return nil
	})
	c.emitDecision(ctx, stateRaw, questions, resp, err, time.Since(start))
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

// emitDecision converts and emits one DecisionEvent for an Ask that
// passed pre-wire validation, gated on c.observer != nil. askErr is
// the Ask's own outcome (nil on success).
func (c *Client) emitDecision(ctx context.Context, stateRaw json.RawMessage, questions Questions, resp Response, askErr error, d time.Duration) {
	if c.observer == nil {
		return
	}
	de := &llmkit.DecisionEvent{
		Backend:   providerName,
		State:     stateRaw,
		Questions: toDecisionQuestions(questions),
	}
	if askErr != nil {
		de.Model = c.model
		de.Err = askErr.Error()
	} else {
		de.Model = resp.Model
		de.Answers = toDecisionAnswers(resp)
		de.Usage = resp.Usage
	}
	ev := llmkit.NewEvent(ctx, llmkit.KindDecision)
	ev.Duration = d
	ev.Decision = de
	c.observer.Observe(ctx, ev)
}

// attempt performs one HTTP round trip and response parse. body is built
// once per Ask and reused across attempts; questions is read-only.
func (c *Client) attempt(ctx context.Context, body []byte, questions Questions) (Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return Response{}, &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 0, Provider: providerName, Message: fmt.Sprintf("build request: %s", err), Err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	httpResp, err := c.client.Do(req)
	if err != nil {
		// url.Error text carries the URL and cause, never the Authorization
		// header, so the API key cannot leak through it.
		return Response{}, adapter.TransportError(providerName, ctx, err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return Response{}, adapter.TransportError(providerName, ctx, err)
	}

	if httpResp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(string(respBody))
		if msg == "" {
			msg = fmt.Sprintf("HTTP %d", httpResp.StatusCode)
		}
		// Classify on the full body: a context-length phrase can sit past
		// byte 200. Only the Message the caller sees is truncated.
		err := adapter.NormalizeSDKError(providerName, adapter.VendorError{
			Status:  httpResp.StatusCode,
			Message: msg,
			Header:  httpResp.Header,
		})
		if apiErr, ok := err.(*llmkit.APIError); ok {
			apiErr.Message = truncate(apiErr.Message, 200)
		}
		return Response{}, err
	}
	return parseResponse(httpResp.StatusCode, respBody, questions)
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

// serverError builds the ErrServer-class APIError for a response that
// violates the answer contract; it carries the serving status.
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

package systemone

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
	"github.com/dpoage/llmkit/internal/adapter"
	"github.com/dpoage/llmkit/retry"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

var (
	_ decide.Decider        = (*Client)(nil)
	_ decide.ImageSupporter = (*Client)(nil)
)

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

// Ask sends state and questions to <BaseURL>/v1/systemone as one logical
// request, retrying transient failures; ctx bounds the call. Every error is
// an *llmkit.APIError with Provider "systemone", except the caller's
// cancellation or deadline, which chains the context error and is never
// retried. The package doc has the status mapping. Ask emits no event; wrap
// the Client in [decide.Observe] to record it.
func (c *Client) Ask(ctx context.Context, state any, questions decide.Questions) (decide.Response, error) {
	req, err := decide.Normalize(state, questions, c.support)
	if err != nil {
		return decide.Response{}, retagProvider(err, c.images)
	}
	body, err := marshalRequest(req, c.model, c.images)
	if err != nil {
		return decide.Response{}, err
	}
	imageHeader := c.images == ImagesAssets && len(req.Images) > 0
	// Every attempt's error is scrubbed of the Secret and the BaseURL
	// password before retry.Do or the caller sees it.
	redact := adapter.NewRedactor(c.apiKey, c.endpoint)
	var resp decide.Response
	err = retry.Do(ctx, c.retry, llmkit.Classify, func(actx context.Context) error {
		r, err := c.attempt(actx, redact, body, questions, imageHeader)
		if err != nil {
			return redact.Scrub(err)
		}
		resp = r
		return nil
	})
	if err != nil {
		return decide.Response{}, err
	}
	return resp, nil
}

// ImageSupport reports what the Client's [Config.Images] dialect sends; the
// zero [ImagesNone] carries no image. Mutating the result does not affect
// the Client.
func (c *Client) ImageSupport() decide.ImageSupport {
	s := c.support
	s.MediaTypes = slices.Clone(s.MediaTypes)
	return s
}

// retagProvider returns err with Provider set to this backend when err is
// the *llmkit.APIError decide.Normalize returns. Under ImagesNone a refusal
// for unsupported images names Config.Images; the caller's value is not
// mutated.
func retagProvider(err error, d ImageDialect) error {
	var ae *llmkit.APIError
	if !errors.As(err, &ae) {
		return err
	}
	tagged := *ae
	tagged.Provider = providerName
	if d == ImagesNone && errors.Is(err, decide.ErrImageUnsupported) {
		tagged.Message += "; set Config.Images to send images"
	}
	return &tagged
}

// Identity returns the Client's backend identity: Provider "systemone" and
// the configured Model alias. [decide.Observe] reads it to tag its events.
func (c *Client) Identity() llmkit.Identity {
	return llmkit.Identity{Provider: providerName, Model: c.model}
}

// attempt performs one HTTP round trip and response parse. body is built
// once per Ask and reused across attempts; questions is read-only.
// imageHeader adds the TypeSafe images opt-in header.
func (c *Client) attempt(ctx context.Context, redact *adapter.Redactor, body []byte, questions decide.Questions, imageHeader bool) (decide.Response, error) {
	req, err := adapter.NewRequest(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return decide.Response{}, &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 0, Provider: providerName, Message: fmt.Sprintf("build request: %s", err), Err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if imageHeader {
		req.Header.Set("TypeSafe-Danger-Zone-Images", "true")
	}

	httpResp, err := c.client.Do(req)
	if err != nil {
		// url.Error text carries the URL, its username included, and the
		// cause; Ask scrubs both.
		return decide.Response{}, adapter.DoError(providerName, ctx, httpResp, err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return decide.Response{}, adapter.TransportError(providerName, ctx, err)
	}

	if httpResp.StatusCode != http.StatusOK {
		msg := string(respBody)
		if strings.TrimSpace(msg) == "" {
			msg = fmt.Sprintf("HTTP %d", httpResp.StatusCode)
		}
		// Classify on the full body: a context-length phrase can sit past
		// byte 200. Only the Message the caller sees is truncated.
		return decide.Response{}, adapter.NormalizeCapped(providerName, adapter.VendorError{
			Status:  httpResp.StatusCode,
			Message: msg,
			Header:  httpResp.Header,
			Redact:  redact,
		})
	}
	return parseResponse(httpResp.StatusCode, respBody, questions)
}

// parseResponse converts a 200 body to a Response. A violation of the answer
// contract, including a failed [decide.Check], is an ErrServer APIError at
// the serving status.
func parseResponse(status int, body []byte, questions decide.Questions) (decide.Response, error) {
	var wire wireResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return decide.Response{}, serverError(status, fmt.Sprintf("response: malformed JSON: %s", err))
	}
	if wire.Model == "" {
		return decide.Response{}, serverError(status, "model: missing model id")
	}
	if wire.Usage == nil {
		return decide.Response{}, serverError(status, "usage: missing usage block")
	}

	nouls := make(map[string]float64)
	choices := make(map[string]decide.ChoiceAnswer)
	scores := make(map[string]decide.ScoreAnswer)
	for id, raw := range wire.Answers {
		q, ok := questions[id]
		if !ok {
			return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q]: answer for an unknown question id", id))
		}
		var ab answerBody
		if err := json.Unmarshal(raw, &ab); err != nil {
			return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q]: malformed answer: %s", id, err))
		}
		switch t := q.(type) {
		case decide.Noul:
			if ab.Type != "noul" {
				return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q].type: want \"noul\", got %q", id, ab.Type))
			}
			if ab.Noul == nil {
				return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q].noul: missing noul value", id))
			}
			nouls[id] = *ab.Noul
		case decide.Choice:
			if ab.Type != "choice" {
				return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q].type: want \"choice\", got %q", id, ab.Type))
			}
			if ab.Choice == nil {
				return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q].choice: missing choice value", id))
			}
			choices[id] = decide.ChoiceAnswer{Choice: *ab.Choice, Probabilities: ab.Probabilities, Confidence: ab.Confidence}
		case decide.Score:
			if ab.Type != "score" {
				return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q].type: want \"score\", got %q", id, ab.Type))
			}
			if ab.Score == nil {
				return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q].score: missing score value", id))
			}
			levels, err := denseSlice(status, id, ".legend", ab.Legend, len(t.Levels))
			if err != nil {
				return decide.Response{}, err
			}
			probs, err := denseSlice(status, id, ".probabilities", ab.Probabilities, len(t.Levels))
			if err != nil {
				return decide.Response{}, err
			}
			scores[id] = decide.ScoreAnswer{Score: *ab.Score, Levels: levels, Probabilities: probs, Confidence: ab.Confidence}
		case nil:
			return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q]: answer for an unknown question id", id))
		}
	}
	for id := range questions {
		if _, ok := wire.Answers[id]; !ok {
			return decide.Response{}, serverError(status, fmt.Sprintf("answers[%q]: missing answer for question", id))
		}
	}
	resp := decide.Response{
		Model:   wire.Model,
		Usage:   llmkit.Usage{InputTokens: wire.Usage.InputTokens, OutputTokens: wire.Usage.OutputTokens},
		Nouls:   nonEmptyOrNil(nouls),
		Choices: nonEmptyOrNil(choices),
		Scores:  nonEmptyOrNil(scores),
	}
	if err := decide.Check(questions, resp); err != nil {
		return decide.Response{}, serverError(status, err.Error())
	}
	return resp, nil
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

// nonEmptyOrNil returns nil for an empty map, so absence reads as absence.
func nonEmptyOrNil[K comparable, V any](m map[K]V) map[K]V {
	if len(m) == 0 {
		return nil
	}
	return m
}

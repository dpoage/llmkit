package anthropic

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llmkit "github.com/dpoage/llmkit"
)

func noCompletionClient(t *testing.T, h http.HandlerFunc) llmkit.StreamingClient {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return newStreamAdapter(t, srv.URL)
}

const emptyMessage = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],` +
	`"stop_reason":%s,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}`

// TestNoCompletion_EmptyContentWithStopReasonIsACompletion: the check is
// "empty content AND empty stop_reason". A message with no content blocks
// but stop_reason end_turn is a turn that produced nothing, not a missing
// completion, and stays a success on Complete and Stream.
func TestNoCompletion_EmptyContentWithStopReasonIsACompletion(t *testing.T) {
	c := noCompletionClient(t, jsonHandler(strings.Replace(emptyMessage, "%s", `"end_turn"`, 1)))
	resp, err := c.Complete(context.Background(), simpleRequest())
	if err != nil || resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("Complete = (%+v, %v), want StopEndTurn, nil", resp, err)
	}

	c = noCompletionClient(t, sseHandler([]sseEvent{
		{"message_start", `{"type":"message_start","message":` + strings.Replace(emptyMessage, "%s", "null", 1) + `}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}))
	resp, err = c.Stream(context.Background(), simpleRequest(), nil)
	if err != nil || resp.StopReason != llmkit.StopEndTurn {
		t.Fatalf("Stream = (%+v, %v), want StopEndTurn, nil", resp, err)
	}
}

// TestNoCompletion_StreamWithoutContentOrStopReasonIsAPIError: a stream
// that reaches message_stop with no content and no stop_reason carries no
// completion; Stream returns the same *APIError Complete does for the
// equivalent message, never an empty success.
func TestNoCompletion_StreamWithoutContentOrStopReasonIsAPIError(t *testing.T) {
	c := noCompletionClient(t, sseHandler([]sseEvent{
		{"message_start", `{"type":"message_start","message":` + strings.Replace(emptyMessage, "%s", "null", 1) + `}`},
		{"message_stop", `{"type":"message_stop"}`},
	}))
	_, err := c.Stream(context.Background(), simpleRequest(), nil)
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != http.StatusOK {
		t.Fatalf("Stream err = %v (%T), want *APIError{ErrServer, 200}", err, err)
	}

	c = noCompletionClient(t, jsonHandler(strings.Replace(emptyMessage, "%s", "null", 1)))
	_, err = c.Complete(context.Background(), simpleRequest())
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != http.StatusOK {
		t.Fatalf("Complete err = %v (%T), want *APIError{ErrServer, 200}", err, err)
	}
}

// TestNoCompletion_ContentWithoutStopReasonIsACompletion: the check needs
// BOTH halves empty. A message whose content carries text but whose
// stop_reason is empty is a completion, not the no-completion error, on
// Complete and Stream.
func TestNoCompletion_ContentWithoutStopReasonIsACompletion(t *testing.T) {
	c := noCompletionClient(t, jsonHandler(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test",`+
		`"content":[{"type":"text","text":"hello"}],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1}}`))
	resp, err := c.Complete(context.Background(), simpleRequest())
	if err != nil || resp.Text != "hello" {
		t.Fatalf("Complete = (%+v, %v), want Text %q, nil", resp, err, "hello")
	}

	c = noCompletionClient(t, sseHandler([]sseEvent{
		{"message_start", `{"type":"message_start","message":` + strings.Replace(emptyMessage, "%s", "null", 1) + `}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_stop", `{"type":"message_stop"}`},
	}))
	resp, err = c.Stream(context.Background(), simpleRequest(), nil)
	if err != nil || resp.Text != "hello" {
		t.Fatalf("Stream = (%+v, %v), want Text %q, nil", resp, err, "hello")
	}
}

// TestNoCompletion_NullBodyIsAPIError: a 200 JSON body of `null` decodes to
// no message at all. Complete returns the no-completion *APIError
// (ErrServer, the response status), never a panic or a nil error.
func TestNoCompletion_NullBodyIsAPIError(t *testing.T) {
	c := noCompletionClient(t, jsonHandler(`null`))
	_, err := c.Complete(context.Background(), simpleRequest())
	var apiErr *llmkit.APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, llmkit.ErrServer) || apiErr.StatusCode != http.StatusOK {
		t.Fatalf("Complete err = %v (%T), want *APIError{ErrServer, 200}", err, err)
	}
}

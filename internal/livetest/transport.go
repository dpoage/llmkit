package livetest

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"sync"
)

// droppedHeaders are never written into a fixture: they carry the lane
// credential or session state, not wire truth. Matching is case-insensitive.
var droppedHeaders = []string{"Authorization", "X-Api-Key", "X-Goog-Api-Key", "Cookie", "Set-Cookie"}

// Exchange is one recorded HTTP request/response pair.
type Exchange struct {
	Request  WireMessage `json:"request"`
	Response WireMessage `json:"response"`
}

// WireMessage is one side of a recorded exchange: the wire truth, after
// header sanitization. Headers are lowercased; repeated values are joined
// with ", ". For responses only Status/Headers/Body are meaningful.
type WireMessage struct {
	Method  string            `json:"method,omitempty"`
	Path    string            `json:"path,omitempty"`
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body"`
	Status  int               `json:"status,omitempty"`
}

// Transport is a recording http.RoundTripper. It captures every exchange —
// method, path (+query), sanitized request headers, request body, response
// status, sanitized response headers, response body — in call order, and
// passes the exchange through to the real transport unmodified so the
// client's own behavior is never altered.
type Transport struct {
	secret string
	base   http.RoundTripper

	mu        sync.Mutex
	exchanges []Exchange
}

// NewTransport returns a recording transport whose sanitizer knows the lane
// credential (any header value containing it is dropped).
func NewTransport(secret string) *Transport {
	return &Transport{secret: secret, base: http.DefaultTransport}
}

// RoundTrip implements http.RoundTripper.
func (tr *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	var reqBody []byte
	if req.Body != nil {
		b, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		reqBody = b
		req.Body = io.NopCloser(bytes.NewReader(reqBody))
	}

	resp, err := tr.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	resp.Body = io.NopCloser(bytes.NewReader(respBody))

	tr.mu.Lock()
	tr.exchanges = append(tr.exchanges, Exchange{
		Request: WireMessage{
			Method:  req.Method,
			Path:    RequestPath(req),
			Headers: SanitizeHeaders(req.Header, tr.secret),
			Body:    string(reqBody),
		},
		Response: WireMessage{
			Headers: SanitizeHeaders(resp.Header, tr.secret),
			Body:    string(respBody),
			Status:  resp.StatusCode,
		},
	})
	tr.mu.Unlock()
	return resp, nil
}

// Exchanges returns the recorded exchanges in call order.
func (tr *Transport) Exchanges() []Exchange {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return append([]Exchange(nil), tr.exchanges...)
}

// RequestPath renders the request's URL path plus query, the identity stored
// in fixtures.
func RequestPath(req *http.Request) string {
	p := req.URL.Path
	if req.URL.RawQuery != "" {
		p += "?" + req.URL.RawQuery
	}
	return p
}

// SanitizeHeaders copies h into a plain map, dropping droppedHeaders
// (case-insensitively) and any header whose value contains secret.
func SanitizeHeaders(h http.Header, secret string) map[string]string {
	out := make(map[string]string, len(h))
	for k, vs := range h {
		drop := false
		for _, d := range droppedHeaders {
			if strings.EqualFold(k, d) {
				drop = true
				break
			}
		}
		if !drop && secret != "" {
			for _, v := range vs {
				if strings.Contains(v, secret) {
					drop = true
					break
				}
			}
		}
		if drop || len(vs) == 0 {
			continue
		}
		out[strings.ToLower(k)] = strings.Join(vs, ", ")
	}
	return out
}

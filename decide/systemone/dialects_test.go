package systemone

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

const okNoulResponse = `{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}`

const imagesHeader = "TypeSafe-Danger-Zone-Images"

// wireServer answers every request with a valid noul answer for "q" and
// keeps the last request.
type wireServer struct {
	*httptest.Server
	hits   atomic.Int32
	mu     sync.Mutex
	body   []byte
	header http.Header
}

func newWireServer(t *testing.T) *wireServer {
	t.Helper()
	ws := &wireServer{}
	ws.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		ws.mu.Lock()
		ws.body, ws.header = b, r.Header.Clone()
		ws.mu.Unlock()
		ws.hits.Add(1)
		writeStatus(w, http.StatusOK, okNoulResponse)
	}))
	t.Cleanup(ws.Close)
	return ws
}

func (ws *wireServer) last() ([]byte, http.Header) {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	return ws.body, ws.header
}

func (ws *wireServer) decoded(t *testing.T) map[string]any {
	t.Helper()
	b, _ := ws.last()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("request body is not JSON: %v\n%s", err, b)
	}
	return m
}

func dialectClient(t *testing.T, url string, d ImageDialect) *Client {
	t.Helper()
	pol := fastRetry
	pol.Rand = func() float64 { return 0.5 }
	c, err := New(Config{Secret: testSecret, Model: "jev-latest", BaseURL: url, Retry: pol, Images: d})
	if err != nil {
		t.Fatalf("New(Images: %v): %v", d, err)
	}
	return c
}

func img(mediaType, seed string) llmkit.Block { return llmkit.Image(mediaType, []byte(seed)) }

func sha(b llmkit.Block) [sha256.Size]byte { return sha256.Sum256(b.Data) }

func refOf(b llmkit.Block) string {
	s := sha(b)
	return "<image:" + hex.EncodeToString(s[:16]) + ">"
}

func b64(b llmkit.Block) string { return base64.StdEncoding.EncodeToString(b.Data) }

func dataURL(b llmkit.Block) string { return "data:" + b.MediaType + ";base64," + b64(b) }

func asset(format string, b llmkit.Block) map[string]any {
	return map[string]any{"type": "image", "format": format, "data_base64": b64(b)}
}

func noulQ(instructions any, criteria map[string]any) map[string]any {
	q := map[string]any{"type": "noul", "instructions": instructions}
	if criteria != nil {
		q["criteria"] = criteria
	}
	return q
}

// refusedWithoutHit asserts err is the pre-wire image refusal with message
// msg, matching both sentinels, and that the server saw nothing.
func refusedWithoutHit(t *testing.T, ws *wireServer, err error, msg string) {
	t.Helper()
	var ae *llmkit.APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %v, want *llmkit.APIError", err)
	}
	if ae.Kind != llmkit.ErrInvalidRequest || ae.Provider != "systemone" || ae.Message != msg {
		t.Errorf("err = %+v, want ErrInvalidRequest, Provider systemone, Message %q", ae, msg)
	}
	if !errors.Is(err, decide.ErrImageUnsupported) || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("err = %v, want both decide.ErrImageUnsupported and llmkit.ErrInvalidRequest", err)
	}
	if n := ws.hits.Load(); n != 0 {
		t.Errorf("HTTP hits = %d, want 0", n)
	}
}

func TestImagesAssets_Wire(t *testing.T) {
	png1, png2, jpg1 := img("image/png", "png-one"), img("image/png", "png-two"), img("image/jpeg", "jpg-one")
	tests := []struct {
		name       string
		state      any
		q          decide.Noul
		wantState  any
		wantQ      map[string]any
		wantAssets map[string]any
	}{
		{
			name:       "state image",
			state:      map[string]any{"photo": png1},
			q:          decide.Noul{Instructions: "Is it red?"},
			wantState:  map[string]any{"photo": refOf(png1)},
			wantQ:      noulQ("Is it red?", nil),
			wantAssets: map[string]any{refOf(png1): asset("png", png1)},
		},
		{
			name:      "instructions and criteria images, jpeg maps to jpg",
			state:     "s",
			q:         decide.Noul{Instructions: []any{"same as", jpg1}, True: png2},
			wantState: "s",
			wantQ:     noulQ([]any{"same as", refOf(jpg1)}, map[string]any{"true": refOf(png2)}),
			wantAssets: map[string]any{
				refOf(jpg1): asset("jpg", jpg1),
				refOf(png2): asset("png", png2),
			},
		},
		{
			name:       "equal bytes are one asset",
			state:      []any{png1, png1},
			q:          decide.Noul{Instructions: "i"},
			wantState:  []any{refOf(png1), refOf(png1)},
			wantQ:      noulQ("i", nil),
			wantAssets: map[string]any{refOf(png1): asset("png", png1)},
		},
		{
			name:      "no image",
			state:     map[string]any{"a": "b"},
			q:         decide.Noul{Instructions: "i"},
			wantState: map[string]any{"a": "b"},
			wantQ:     noulQ("i", nil),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := newWireServer(t)
			c := dialectClient(t, ws.URL, ImagesAssets)
			if _, err := c.Ask(context.Background(), tt.state, decide.Questions{"q": tt.q}); err != nil {
				t.Fatalf("Ask: %v", err)
			}
			got := ws.decoded(t)
			assets, hasAssets := got["assets"]
			delete(got, "assets")
			base := map[string]any{"state": tt.wantState, "model": "jev-latest", "questions": map[string]any{"q": tt.wantQ}}
			if !reflect.DeepEqual(got, base) {
				t.Errorf("body without assets = %v\nwant %v", got, base)
			}
			if tt.wantAssets == nil {
				if hasAssets {
					t.Errorf("assets = %v, want no assets key", assets)
				}
			} else if !reflect.DeepEqual(assets, tt.wantAssets) {
				t.Errorf("assets = %v\nwant   %v", assets, tt.wantAssets)
			}
			_, hdr := ws.last()
			vals := hdr.Values(imagesHeader)
			present := len(vals) > 0
			if want := tt.wantAssets != nil; present != want || (want && !reflect.DeepEqual(vals, []string{"true"})) {
				t.Errorf("%s = %v (present %t), want present=%t with value true", imagesHeader, vals, present, want)
			}
			if _, ok := got["images"]; ok {
				t.Errorf("assets dialect sent an images key")
			}
		})
	}
}

// reverseShaPair returns two images whose first-occurrence order (a, b) is
// the reverse of their SHA-256 order.
func reverseShaPair(t *testing.T, mediaType string) (a, b llmkit.Block) {
	t.Helper()
	for i := range 64 {
		x, y := img(mediaType, fmt.Sprintf("pair-%d", i)), img(mediaType, fmt.Sprintf("pair-%d", i+1))
		if sx, sy := sha(x), sha(y); bytes.Compare(sx[:], sy[:]) > 0 {
			return x, y
		}
	}
	t.Fatal("no image pair with reversed SHA order")
	return
}

func TestImagesArray_Wire(t *testing.T) {
	a, b := reverseShaPair(t, "image/png")
	webp := img("image/webp", "webp-one")
	jpg := img("image/jpeg", "jpg-two")

	tests := []struct {
		name       string
		state      any
		wantState  any
		wantImages []any
	}{
		{
			name:       "two images numbered by occurrence, not by hash",
			state:      map[string]any{"a": a, "b": b},
			wantState:  map[string]any{"a": "[image 1]", "b": "[image 2]"},
			wantImages: []any{dataURL(a), dataURL(b)},
		},
		{
			name:       "user strings equal to a ref or a label survive; equal bytes share a number",
			state:      map[string]any{"a": a, "again": a, "b": b, "note": refOf(a), "lbl": "[image 2]"},
			wantState:  map[string]any{"a": "[image 1]", "again": "[image 1]", "b": "[image 2]", "note": refOf(a), "lbl": "[image 2]"},
			wantImages: []any{dataURL(a), dataURL(b)},
		},
		{
			name:       "png, webp and jpeg in a list",
			state:      []any{webp, "x", jpg, a},
			wantState:  []any{"[image 1]", "x", "[image 2]", "[image 3]"},
			wantImages: []any{dataURL(webp), dataURL(jpg), dataURL(a)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := newWireServer(t)
			c := dialectClient(t, ws.URL, ImagesArray)
			if _, err := c.Ask(context.Background(), tt.state, decide.Questions{"q": decide.Noul{Instructions: "i"}}); err != nil {
				t.Fatalf("Ask: %v", err)
			}
			got := ws.decoded(t)
			want := map[string]any{
				"state": tt.wantState, "model": "jev-latest",
				"questions": map[string]any{"q": noulQ("i", nil)},
				"images":    tt.wantImages,
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("body = %v\nwant %v", got, want)
			}
			if _, hdr := ws.last(); len(hdr.Values(imagesHeader)) > 0 {
				t.Errorf("array dialect sent %s: %v", imagesHeader, hdr.Values(imagesHeader))
			}
		})
	}
}

func TestImagesArray_RefusesAllButState(t *testing.T) {
	a := img("image/png", "array-a")
	tests := []struct {
		name  string
		state any
		q     decide.Question
		want  string
	}{
		{"instructions", "s", decide.Noul{Instructions: []any{"see", a}},
			`questions["q"].instructions[1]: image in instructions is not supported by this backend`},
		{"criteria", "s", decide.Noul{Instructions: "i", True: a},
			`questions["q"].criteria.true: image in criteria is not supported by this backend`},
		{"same bytes in state and criteria", map[string]any{"p": a}, decide.Noul{Instructions: "i", True: a},
			`questions["q"].criteria.true: image in criteria is not supported by this backend`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := newWireServer(t)
			c := dialectClient(t, ws.URL, ImagesArray)
			_, err := c.Ask(context.Background(), tt.state, decide.Questions{"q": tt.q})
			refusedWithoutHit(t, ws, err, tt.want)
		})
	}
}

func TestImages_MediaTypesPerDialect(t *testing.T) {
	tests := []struct {
		name    string
		dialect ImageDialect
		image   llmkit.Block
		want    string
	}{
		{"webp under assets", ImagesAssets, img("image/webp", "w"),
			`state["p"]: image media_type "image/webp" is not supported by this backend (supported: image/png, image/jpeg)`},
		{"gif under array", ImagesArray, img("image/gif", "g"),
			`state["p"]: image media_type "image/gif" is not supported by this backend (supported: image/png, image/jpeg, image/webp)`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := newWireServer(t)
			c := dialectClient(t, ws.URL, tt.dialect)
			_, err := c.Ask(context.Background(), map[string]any{"p": tt.image}, decide.Questions{"q": decide.Noul{Instructions: "i"}})
			refusedWithoutHit(t, ws, err, tt.want)
		})
	}
}

func TestImages_NoneRefusesEveryPlacementNamingConfigImages(t *testing.T) {
	a := img("image/png", "none-a")
	tests := []struct {
		name  string
		state any
		q     decide.Question
		want  string
	}{
		{"state", map[string]any{"p": a}, decide.Noul{Instructions: "i"},
			`state["p"]: image in state is not supported by this backend; set Config.Images to send images`},
		{"instructions", "s", decide.Noul{Instructions: []any{a}},
			`questions["q"].instructions[0]: image in instructions is not supported by this backend; set Config.Images to send images`},
		{"criteria", "s", decide.Noul{Instructions: "i", False: a},
			`questions["q"].criteria.false: image in criteria is not supported by this backend; set Config.Images to send images`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := newWireServer(t)
			c := dialectClient(t, ws.URL, ImagesNone)
			_, err := c.Ask(context.Background(), tt.state, decide.Questions{"q": tt.q})
			refusedWithoutHit(t, ws, err, tt.want)
		})
	}
}

func TestImages_ImageFreeAskIsTheSameBytesInEveryDialect(t *testing.T) {
	const golden = `{"state":{"a":"b"},"model":"jev-latest","questions":{"q":{"type":"noul","instructions":"i"}}}`
	var baseHeader http.Header
	for _, d := range []ImageDialect{ImagesNone, ImagesAssets, ImagesArray} {
		t.Run(d.String(), func(t *testing.T) {
			ws := newWireServer(t)
			c := dialectClient(t, ws.URL, d)
			if _, err := c.Ask(context.Background(), map[string]any{"a": "b"}, decide.Questions{"q": decide.Noul{Instructions: "i"}}); err != nil {
				t.Fatalf("Ask: %v", err)
			}
			body, hdr := ws.last()
			if string(body) != golden {
				t.Errorf("body = %s\nwant   %s", body, golden)
			}
			if d == ImagesNone {
				baseHeader = hdr
			} else if !reflect.DeepEqual(hdr, baseHeader) {
				t.Errorf("headers = %v, want the ImagesNone headers %v", hdr, baseHeader)
			}
		})
	}
}

// A Block-shaped state member whose text or data member sits beside a
// foreign member must still leave no image bytes in the state or questions.
func TestImages_BlockShapedStatesCarryNoBase64Outside(t *testing.T) {
	blk := img("image/png", "m3-image-bytes")
	data := b64(blk)
	states := []struct {
		name  string
		state any
	}{
		{"text member", map[string]any{"text": blk, "zzz": 1}},
		{"data member", map[string]any{"data": blk, "meta": 1}},
	}
	wantLabel := map[ImageDialect]string{ImagesAssets: refOf(blk), ImagesArray: "[image 1]"}
	for _, st := range states {
		for _, d := range []ImageDialect{ImagesNone, ImagesAssets, ImagesArray} {
			t.Run(st.name+"/"+d.String(), func(t *testing.T) {
				ws := newWireServer(t)
				c := dialectClient(t, ws.URL, d)
				_, err := c.Ask(context.Background(), st.state, decide.Questions{"q": decide.Noul{Instructions: "i"}})
				if d == ImagesNone {
					if !errors.Is(err, decide.ErrImageUnsupported) || ws.hits.Load() != 0 {
						t.Fatalf("err = %v, hits = %d, want a refusal with 0 hits", err, ws.hits.Load())
					}
					return
				}
				if err != nil {
					t.Fatalf("Ask: %v", err)
				}
				got := ws.decoded(t)
				delete(got, "assets")
				delete(got, "images")
				rest, _ := json.Marshal(got)
				if strings.Contains(string(rest), data) {
					t.Errorf("state or questions carry the image base64: %s", rest)
				}
				state := got["state"].(map[string]any)
				for k, v := range state {
					if k == "text" || k == "data" {
						if v != wantLabel[d] {
							t.Errorf("state[%q] = %v, want %q", k, v, wantLabel[d])
						}
					}
				}
				body, _ := ws.last()
				if n := strings.Count(string(body), data); n != 1 {
					t.Errorf("image base64 occurs %d times in the body, want once (in the %s entry)", n, map[ImageDialect]string{ImagesAssets: "assets", ImagesArray: "images"}[d])
				}
			})
		}
	}
}

func TestNew_ImagesDialect(t *testing.T) {
	_, err := New(Config{Secret: testSecret, Model: "m", Images: ImagesArray + 1})
	if !errors.Is(err, llmkit.ErrInvalidRequest) || err == nil || !strings.Contains(err.Error(), "Config.Images") {
		t.Fatalf("err = %v, want an ErrInvalidRequest naming Config.Images", err)
	}
	if strings.Contains(err.Error(), testSecret) {
		t.Errorf("error echoes the Secret: %v", err)
	}
}

func TestClient_ImageSupportPerDialect(t *testing.T) {
	want := map[ImageDialect]decide.ImageSupport{
		ImagesNone:   {},
		ImagesAssets: {Placements: decide.PlaceAll, MediaTypes: []string{"image/png", "image/jpeg"}, Label: decide.LabelRef},
		ImagesArray:  {Placements: decide.PlaceState, MediaTypes: []string{"image/png", "image/jpeg", "image/webp"}, Label: decide.LabelPosition},
	}
	for d, w := range want {
		t.Run(d.String(), func(t *testing.T) {
			c := dialectClient(t, "http://127.0.0.1:1", d)
			got := c.ImageSupport()
			if got.Placements != w.Placements || got.Label != w.Label || !reflect.DeepEqual(append([]string(nil), got.MediaTypes...), append([]string(nil), w.MediaTypes...)) {
				t.Errorf("ImageSupport() = %+v, want %+v", got, w)
			}
			for i := range got.MediaTypes {
				got.MediaTypes[i] = "image/x-mutated"
			}
			if again := c.ImageSupport(); !reflect.DeepEqual(append([]string(nil), again.MediaTypes...), append([]string(nil), w.MediaTypes...)) {
				t.Errorf("mutating the result changed the Client: %+v", again)
			}
		})
	}
}

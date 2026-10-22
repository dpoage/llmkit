package decide

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/dpoage/llmkit"
)

// Placement is a bitset of the places in a request an image can sit.
type Placement uint8

// The places an image can sit. PlaceState is anywhere in the state;
// PlaceInstructions is anywhere in a question's Instructions; PlaceCriteria
// is anywhere in a Noul's True or False, a Choice option's description, or a
// Score level.
const (
	PlaceState Placement = 1 << iota
	PlaceInstructions
	PlaceCriteria

	PlaceAll = PlaceState | PlaceInstructions | PlaceCriteria
)

// ImageLabel selects the text that stands in for an image in
// [Request.State] and [Request.Questions].
type ImageLabel uint8

const (
	// LabelRef (the zero value) renders [Image.Ref], "<image:hex>".
	LabelRef ImageLabel = iota
	// LabelPosition renders "[image N]", N being the image's 1-based index
	// in [Request.Images].
	LabelPosition
)

// ImageSupport is what a backend can carry. The zero value refuses every
// image.
type ImageSupport struct {
	// Placements is the set of places the backend carries images in.
	Placements Placement
	// MediaTypes lists the accepted media types, matched exactly. Empty
	// accepts any "image/*".
	MediaTypes []string
	// Label is the text that replaces each image in the request.
	Label ImageLabel
}

// ImageSupporter is the optional interface a [Decider] implements to
// declare the [ImageSupport] its Ask normalizes with.
type ImageSupporter interface{ ImageSupport() ImageSupport }

// Image is one distinct image of a [Request].
type Image struct {
	// Ref is "<image:" + lowercase hex of the first 16 bytes of the SHA-256
	// of Data + ">".
	Ref string
	// SHA256 is the lowercase hex of the full SHA-256 of Data.
	SHA256    string
	MediaType string
	Data      []byte
	// Placements is the union of the places the image occurs in.
	Placements Placement
}

// ErrImageUnsupported is the Err of a refusal because the [ImageSupport]
// does not carry an image's placement or media type. A refusal because the
// image itself is invalid leaves Err nil. Either way the error is an
// *llmkit.APIError with Kind llmkit.ErrInvalidRequest.
var ErrImageUnsupported = errors.New("decide: image not supported by this backend")

// imageSet collects the distinct images across one Normalize call.
type imageSet struct {
	support ImageSupport
	images  []Image
	paths   []string // first occurrence path of images[i]
	index   map[[sha256.Size]byte]int
}

// rewrite replaces every image value in raw, a part of the request at
// path in place, with its label, and records the images. A raw without an
// image is returned as is.
func (s *imageSet) rewrite(raw json.RawMessage, path string, place Placement) (json.RawMessage, error) {
	if bytes.IndexByte(raw, '{') < 0 {
		return raw, nil
	}
	w := &walker{set: s, src: raw, base: path, place: place}
	if _, err := w.value(skipWS(raw, 0)); err != nil {
		return nil, err
	}
	if w.out == nil {
		return raw, nil
	}
	return append(w.out, raw[w.last:]...), nil
}

// add records one valid image at path and returns its label as a JSON
// string.
func (s *imageSet) add(b llmkit.Block, path string, place Placement) ([]byte, error) {
	sum := sha256.Sum256(b.Data)
	idx, seen := s.index[sum]
	if seen && s.images[idx].MediaType != b.MediaType {
		return nil, invalidRequest(path, fmt.Sprintf("image has media_type %q, but the same bytes at %s have %q", b.MediaType, s.paths[idx], s.images[idx].MediaType))
	}
	if s.support.Placements&place == 0 {
		return nil, unsupportedImage(path, fmt.Sprintf("image in %s is not supported by this backend", placementName(place)))
	}
	if mts := s.support.MediaTypes; len(mts) > 0 && !slices.Contains(mts, b.MediaType) {
		return nil, unsupportedImage(path, fmt.Sprintf("image media_type %q is not supported by this backend (supported: %s)", b.MediaType, strings.Join(mts, ", ")))
	}
	if !seen {
		full := hex.EncodeToString(sum[:])
		idx = len(s.images)
		s.images = append(s.images, Image{
			Ref:       "<image:" + full[:32] + ">",
			SHA256:    full,
			MediaType: b.MediaType,
			Data:      b.Data,
		})
		s.paths = append(s.paths, path)
		if s.index == nil {
			s.index = make(map[[sha256.Size]byte]int)
		}
		s.index[sum] = idx
	}
	s.images[idx].Placements |= place
	if s.support.Label == LabelPosition {
		return []byte(`"[image ` + strconv.Itoa(idx+1) + `]"`), nil
	}
	return []byte(`"\u003cimage:` + s.images[idx].SHA256[:32] + `\u003e"`), nil
}

func placementName(p Placement) string {
	switch p {
	case PlaceState:
		return "state"
	case PlaceInstructions:
		return "instructions"
	default:
		return "criteria"
	}
}

func unsupportedImage(path, problem string) error {
	return &llmkit.APIError{
		Kind:     llmkit.ErrInvalidRequest,
		Provider: "decide",
		Message:  path + ": " + problem,
		Err:      ErrImageUnsupported,
	}
}

// checkImage reports why b, an image-kind Block decoded with decodeErr, is
// not a valid image, or "".
func checkImage(b llmkit.Block, decodeErr error) string {
	if decodeErr != nil {
		return "invalid image: member does not decode: " + decodeErr.Error()
	}
	for _, f := range []struct {
		name string
		set  bool
	}{
		{"url", b.URL != ""},
		{"text", b.Text != ""},
		{"title", b.Title != ""},
		{"provider", b.Provider != ""},
		{"raw", len(b.Raw) != 0},
	} {
		if f.set {
			return "invalid image: " + f.name + " is set; an image carries only data and media_type"
		}
	}
	switch {
	case len(b.Data) == 0:
		return "invalid image: no inline data"
	case !strings.HasPrefix(b.MediaType, "image/"):
		return fmt.Sprintf(`invalid image: media_type %q must begin "image/"`, b.MediaType)
	}
	return ""
}

// blockNames are llmkit.Block's JSON member names and blockProbe is a
// struct with the same names and tags whose fields are all bool, so that
// encoding/json itself decides which member name folds to which field.
var blockNames, blockProbe = blockShape()

func blockShape() ([]string, reflect.Type) {
	t := reflect.TypeFor[llmkit.Block]()
	names := make([]string, 0, t.NumField())
	fields := make([]reflect.StructField, 0, t.NumField())
	for i := range t.NumField() {
		f := t.Field(i)
		name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
		if name == "" {
			name = f.Name
		}
		names = append(names, name)
		fields = append(fields, reflect.StructField{Name: f.Name, Type: reflect.TypeFor[bool](), Tag: f.Tag})
	}
	return names, reflect.StructOf(fields)
}

// isBlockName reports whether key, a JSON string token with its quotes,
// names a llmkit.Block member under encoding/json's matching rule.
func isBlockName(key []byte) bool {
	inner := key[1 : len(key)-1]
	plain := true
	for _, c := range inner {
		if c == '\\' || c >= 0x80 {
			plain = false
			break
		}
	}
	if plain {
		for _, n := range blockNames {
			if asciiEqualFold(inner, n) {
				return true
			}
		}
		return false
	}
	buf := make([]byte, 0, len(key)+8)
	buf = append(buf, '{')
	buf = append(buf, key...)
	buf = append(buf, ":true}"...)
	probe := reflect.New(blockProbe)
	if json.Unmarshal(buf, probe.Interface()) != nil {
		return false
	}
	for i := range probe.Elem().NumField() {
		if probe.Elem().Field(i).Bool() {
			return true
		}
	}
	return false
}

func asciiEqualFold(b []byte, s string) bool {
	if len(b) != len(s) {
		return false
	}
	for i := range b {
		c, d := b[i], s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if 'A' <= d && d <= 'Z' {
			d += 'a' - 'A'
		}
		if c != d {
			return false
		}
	}
	return true
}

// walker scans one marshalled JSON part for image values, splicing labels
// into out; bytes before last are already in out.
type walker struct {
	set     *imageSet
	src     []byte
	base    string
	place   Placement
	segs    []seg
	members []member
	out     []byte
	last    int
}

// seg is one path step: an object member (key is its JSON string token) or
// an array index.
type seg struct {
	key []byte
	idx int
}

type member struct {
	key    []byte
	vs, ve int
}

func (w *walker) path() string {
	var sb strings.Builder
	sb.WriteString(w.base)
	for _, s := range w.segs {
		if s.key == nil {
			sb.WriteString("[" + strconv.Itoa(s.idx) + "]")
			continue
		}
		var k string
		_ = json.Unmarshal(s.key, &k)
		sb.WriteString("[" + strconv.Quote(k) + "]")
	}
	return sb.String()
}

// value scans the value starting at i and returns the index after it.
func (w *walker) value(i int) (int, error) {
	switch w.src[i] {
	case '{':
		return w.object(i)
	case '[':
		return w.array(i)
	default:
		return skipValue(w.src, i), nil
	}
}

func (w *walker) array(i int) (int, error) {
	i = skipWS(w.src, i+1)
	if w.src[i] == ']' {
		return i + 1, nil
	}
	for n := 0; ; n++ {
		w.segs = append(w.segs, seg{idx: n})
		end, err := w.value(i)
		if err != nil {
			return 0, err
		}
		w.segs = w.segs[:len(w.segs)-1]
		i = skipWS(w.src, end)
		if w.src[i] == ']' {
			return i + 1, nil
		}
		i = skipWS(w.src, i+1)
	}
}

// object scans the object at start. It records members while every name
// is a Block name; if the object then closes and decodes as an image kind
// it is spliced, otherwise the recorded members are scanned.
func (w *walker) object(start int) (int, error) {
	src := w.src
	i := skipWS(src, start+1)
	if src[i] == '}' {
		return i + 1, nil
	}
	base := len(w.members)
	var (
		key   []byte
		vs    int
		named = true
	)
	for {
		ke := skipValue(src, i)
		key = src[i:ke]
		vs = skipWS(src, skipWS(src, ke)+1)
		if !isBlockName(key) {
			named = false
			break
		}
		ve := skipValue(src, vs)
		w.members = append(w.members, member{key, vs, ve})
		i = skipWS(src, ve)
		if src[i] == '}' {
			break
		}
		i = skipWS(src, i+1)
	}
	if named {
		end := i + 1
		spliced, err := w.image(start, end)
		if err != nil {
			return 0, err
		}
		if !spliced {
			if err := w.walkRecorded(base); err != nil {
				return 0, err
			}
		}
		w.members = w.members[:base]
		return end, nil
	}
	if err := w.walkRecorded(base); err != nil {
		return 0, err
	}
	w.members = w.members[:base]
	for {
		end, err := w.member(key, vs)
		if err != nil {
			return 0, err
		}
		i = skipWS(src, end)
		if src[i] == '}' {
			return i + 1, nil
		}
		i = skipWS(src, i+1)
		ke := skipValue(src, i)
		key = src[i:ke]
		vs = skipWS(src, skipWS(src, ke)+1)
	}
}

func (w *walker) walkRecorded(base int) error {
	for j := base; j < len(w.members); j++ {
		m := w.members[j]
		if _, err := w.member(m.key, m.vs); err != nil {
			return err
		}
	}
	return nil
}

func (w *walker) member(key []byte, vs int) (int, error) {
	w.segs = append(w.segs, seg{key: key})
	end, err := w.value(vs)
	if err != nil {
		return 0, err
	}
	w.segs = w.segs[:len(w.segs)-1]
	return end, nil
}

// image splices the object src[start:end], whose member names are all
// Block names, when it decodes as an image kind.
func (w *walker) image(start, end int) (bool, error) {
	var b llmkit.Block
	decodeErr := json.Unmarshal(w.src[start:end], &b)
	if b.Kind != llmkit.BlockImage {
		return false, nil
	}
	if problem := checkImage(b, decodeErr); problem != "" {
		return false, invalidRequest(w.path(), problem)
	}
	label, err := w.set.add(b, w.path(), w.place)
	if err != nil {
		return false, err
	}
	w.out = append(w.out, w.src[w.last:start]...)
	w.out = append(w.out, label...)
	w.last = end
	return true, nil
}

func skipWS(src []byte, i int) int {
	for i < len(src) {
		switch src[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// skipString returns the index after the string token starting at i.
func skipString(src []byte, i int) int {
	for i++; i < len(src); i++ {
		switch src[i] {
		case '\\':
			i++
		case '"':
			return i + 1
		}
	}
	return len(src)
}

// skipValue returns the index after the JSON value starting at i.
func skipValue(src []byte, i int) int {
	switch src[i] {
	case '"':
		return skipString(src, i)
	case '{', '[':
		for depth := 0; i < len(src); i++ {
			switch src[i] {
			case '"':
				i = skipString(src, i) - 1
			case '{', '[':
				depth++
			case '}', ']':
				if depth--; depth == 0 {
					return i + 1
				}
			}
		}
		return len(src)
	default:
		for i < len(src) && !strings.ContainsRune(",]} \t\r\n", rune(src[i])) {
			i++
		}
		return i
	}
}

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

// Image is one distinct image of a [Request]. MediaType begins "image/" and
// Data holds the image's decoded bytes, not base64.
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
	paths   []pathRef // first occurrence of images[i]
	index   map[[sha256.Size]byte]int
}

// rewrite replaces each image in raw, the request part at path, with its
// label and records it; a raw with no image is returned as is.
func (s *imageSet) rewrite(raw json.RawMessage, path string, place Placement) (json.RawMessage, error) {
	if bytes.IndexByte(raw, '{') < 0 {
		return raw, nil
	}
	w := &walker{set: s, src: raw, base: path, place: place, cur: -1}
	if err := w.run(); err != nil {
		return nil, err
	}
	if w.out == nil {
		return raw, nil
	}
	return append(w.out, raw[w.last:]...), nil
}

// inlineImageHashes adds to set the SHA-256 of every valid image value in
// raw, by Normalize's definition. It never refuses: an invalid one is skipped.
func inlineImageHashes(raw []byte, set map[string]struct{}) {
	if bytes.IndexByte(raw, '{') < 0 || !json.Valid(raw) {
		return
	}
	w := &walker{src: raw, collect: set, cur: -1}
	_ = w.run() // collect mode never refuses
}

func hashImage(data []byte) ([sha256.Size]byte, string) {
	sum := sha256.Sum256(data)
	return sum, hex.EncodeToString(sum[:])
}

// add records b and returns its label as a JSON string.
func (s *imageSet) add(b llmkit.Block, at pathRef, place Placement) ([]byte, error) {
	sum, full := hashImage(b.Data)
	idx, seen := s.index[sum]
	if seen && s.images[idx].MediaType != b.MediaType {
		return nil, invalidRequest(at.String(), fmt.Sprintf("image has media_type %q, but the same bytes at %s have %q", b.MediaType, s.paths[idx], s.images[idx].MediaType))
	}
	if s.support.Placements&place == 0 {
		return nil, unsupportedImage(at.String(), fmt.Sprintf("image in %s is not supported by this backend", placementName(place)))
	}
	if mts := s.support.MediaTypes; len(mts) > 0 && !slices.Contains(mts, b.MediaType) {
		return nil, unsupportedImage(at.String(), fmt.Sprintf("image media_type %q is not supported by this backend (supported: %s)", b.MediaType, strings.Join(mts, ", ")))
	}
	if !seen {
		idx = len(s.images)
		s.images = append(s.images, Image{
			Ref:       "<image:" + full[:32] + ">",
			SHA256:    full,
			MediaType: b.MediaType,
			Data:      b.Data,
		})
		s.paths = append(s.paths, at)
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

// pathRef names a value's JSON path; the string is built only when a refusal
// message needs it.
type pathRef struct {
	w    *walker
	node int
}

func (r pathRef) String() string { return r.w.path(r.node) }

// placementNames is the single Placement→name table, in report order.
var placementNames = [...]struct {
	place Placement
	name  string
}{
	{PlaceState, "state"},
	{PlaceInstructions, "instructions"},
	{PlaceCriteria, "criteria"},
}

func placementName(p Placement) string {
	for _, pn := range placementNames {
		if pn.place == p {
			return pn.name
		}
	}
	return placementNames[len(placementNames)-1].name
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

// blockNames are llmkit.Block's JSON member names; blockProbe has the same
// names and tags with bool fields, so encoding/json decides the name fold.
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

// blockField returns the index in blockNames of the Block member that key,
// a quoted JSON string token, names under encoding/json's matching, or -1.
func blockField(key []byte) int {
	inner := key[1 : len(key)-1]
	plain := true
	for _, c := range inner {
		if c == '\\' || c >= 0x80 {
			plain = false
			break
		}
	}
	if plain {
		for i, n := range blockNames {
			if asciiEqualFold(inner, n) {
				return i
			}
		}
		return -1
	}
	buf := make([]byte, 0, len(key)+8)
	buf = append(buf, '{')
	buf = append(buf, key...)
	buf = append(buf, ":true}"...)
	probe := reflect.New(blockProbe)
	if json.Unmarshal(buf, probe.Interface()) != nil {
		return -1
	}
	for i := range probe.Elem().NumField() {
		if probe.Elem().Field(i).Bool() {
			return i
		}
	}
	return -1
}

var kindField, dataField = slices.Index(blockNames, "kind"), slices.Index(blockNames, "data")

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

// walker scans one marshalled JSON part for image values in one pass. Strict
// mode splices labels into out; collect mode records hashes and never refuses.
type walker struct {
	set     *imageSet
	collect map[string]struct{} // collect mode's hash sink; nil is strict mode
	src     []byte
	base    string
	place   Placement
	nodes   []pathNode // strict mode only
	cur     int        // index in nodes of the value being scanned, -1 at the root
	events  []event
	out     []byte
	last    int
}

// pathNode is one path step, parent-linked so an event keeps its path in O(1).
type pathNode struct {
	parent int
	key    []byte // an object member's JSON string token; nil for an array index
	idx    int
}

// event is an image object (foreign nil) or, in strict mode, an image-kind
// object with a non-Block member (foreign, its name token) that commit refuses.
type event struct {
	start, end int
	node       int
	foreign    []byte
}

func (w *walker) run() error {
	w.value(skipWS(w.src, 0))
	return w.commit()
}

func (w *walker) path(node int) string {
	var chain []int
	for n := node; n >= 0; n = w.nodes[n].parent {
		chain = append(chain, n)
	}
	var sb strings.Builder
	sb.WriteString(w.base)
	for j := len(chain) - 1; j >= 0; j-- {
		nd := w.nodes[chain[j]]
		if nd.key == nil {
			sb.WriteString("[" + strconv.Itoa(nd.idx) + "]")
			continue
		}
		var k string
		_ = json.Unmarshal(nd.key, &k)
		sb.WriteString("[" + strconv.Quote(k) + "]")
	}
	return sb.String()
}

func (w *walker) value(i int) int {
	switch w.src[i] {
	case '{':
		return w.object(i)
	case '[':
		return w.array(i)
	default:
		return skipValue(w.src, i)
	}
}

// step scans the value at vs as the child key (or, when key is nil, index idx)
// of the enclosing container and returns its end.
func (w *walker) step(key []byte, idx, vs int) int {
	if c := w.src[vs]; w.collect != nil || (c != '{' && c != '[') {
		return w.value(vs)
	}
	parent, events, nodes := w.cur, len(w.events), len(w.nodes)
	w.nodes = append(w.nodes, pathNode{parent: parent, key: key, idx: idx})
	w.cur = nodes
	end := w.value(vs)
	w.cur = parent
	if len(w.events) == events {
		w.nodes = w.nodes[:nodes]
	}
	return end
}

func (w *walker) array(i int) int {
	i = skipWS(w.src, i+1)
	if w.src[i] == ']' {
		return i + 1
	}
	for n := 0; ; n++ {
		i = skipWS(w.src, w.step(nil, n, i))
		if w.src[i] == ']' {
			return i + 1
		}
		i = skipWS(w.src, i+1)
	}
}

// object scans the object at start, each member once. An image object, or in
// strict mode a foreign image, replaces the events its members logged.
func (w *walker) object(start int) int {
	src := w.src
	i := skipWS(src, start+1)
	if src[i] == '}' {
		return i + 1
	}
	events, nodes := len(w.events), len(w.nodes)
	named := true
	var foreign, kind, strKind, data []byte
	for {
		ke := skipValue(src, i)
		key := src[i:ke]
		vs := skipWS(src, skipWS(src, ke)+1)
		ve := w.step(key, 0, vs)
		if named || w.collect == nil {
			switch blockField(key) {
			case -1:
				named = false
				if foreign == nil {
					foreign = key
				}
			case kindField:
				kind = src[vs:ve]
				if kind[0] == '"' {
					strKind = kind
				}
			case dataField:
				data = src[vs:ve]
			}
		}
		i = skipWS(src, ve)
		if src[i] == '}' {
			break
		}
		i = skipWS(src, i+1)
	}
	end := i + 1
	switch {
	case named && isImageKind(strKind):
		w.events = append(w.events[:events], event{start: start, end: end, node: w.cur})
		w.nodes = w.nodes[:nodes]
	case !named && w.collect == nil && isForeignImage(foreign, kind, data):
		w.events = append(w.events[:events], event{start: start, end: end, node: w.cur, foreign: foreign})
		w.nodes = w.nodes[:nodes]
	}
	return end
}

// isImageKind reports whether tok, the last string-valued kind member of a
// Block-named object, decodes to "image".
func isImageKind(tok []byte) bool {
	var k string
	return tok != nil && json.Unmarshal(tok, &k) == nil && k == string(llmkit.BlockImage)
}

// isForeignImage reports whether an object with a non-Block member (foreign,
// its first name token) has kind "image" and data other than null or "".
func isForeignImage(foreign, kind, data []byte) bool {
	if foreign == nil || data == nil || string(data) == "null" || string(data) == `""` || len(kind) == 0 || kind[0] != '"' {
		return false
	}
	var k string
	return json.Unmarshal(kind, &k) == nil && k == string(llmkit.BlockImage)
}

// commit resolves the events in document order. An image that does not decode
// is refused, never demoted to text; collect mode never returns an error.
func (w *walker) commit() error {
	for _, e := range w.events {
		if e.foreign != nil {
			var name string
			_ = json.Unmarshal(e.foreign, &name)
			return invalidRequest(pathRef{w, e.node}.String(), "invalid image: member "+strconv.Quote(name)+" is not an llmkit.Block field")
		}
		var b llmkit.Block
		decodeErr := json.Unmarshal(w.src[e.start:e.end], &b)
		problem := checkImage(b, decodeErr)
		if w.collect != nil {
			if problem == "" {
				_, hash := hashImage(b.Data)
				w.collect[hash] = struct{}{}
			}
			continue
		}
		at := pathRef{w, e.node}
		if problem != "" {
			return invalidRequest(at.String(), problem)
		}
		label, err := w.set.add(b, at, w.place)
		if err != nil {
			return err
		}
		w.out = append(w.out, w.src[w.last:e.start]...)
		w.out = append(w.out, label...)
		w.last = e.end
	}
	return nil
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

// skipValue returns the end of the string or scalar at i.
func skipValue(src []byte, i int) int {
	if src[i] == '"' {
		return skipString(src, i)
	}
	for i < len(src) && !strings.ContainsRune(",]} \t\r\n", rune(src[i])) {
		i++
	}
	return i
}

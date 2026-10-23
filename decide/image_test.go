package decide

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"testing"

	"github.com/dpoage/llmkit"
)

const (
	imgJSON = `{"kind":"image","data":"AQID","media_type":"image/png"}`
	// refOf([]byte{1,2,3}) and its JSON string form.
	ref123     = "<image:039058c6f2c0cb492c533b0a4d14ef77>"
	ref123JSON = `"\u003cimage:039058c6f2c0cb492c533b0a4d14ef77\u003e"`
)

func pngBlock(data ...byte) llmkit.Block {
	return llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: data}
}

func refOf(data []byte) string {
	sum := sha256.Sum256(data)
	return "<image:" + hex.EncodeToString(sum[:16]) + ">"
}

func mustMarshal(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	return string(b)
}

func oneQuestion(instructions any) Questions {
	return Questions{"q": Noul{Instructions: instructions}}
}

func apiError(t *testing.T, err error) *llmkit.APIError {
	t.Helper()
	var ae *llmkit.APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %v, want *llmkit.APIError", err)
	}
	if ae.Kind != llmkit.ErrInvalidRequest || ae.Provider != "decide" || ae.StatusCode != 0 {
		t.Fatalf("err = %+v, want Kind ErrInvalidRequest, Provider decide, status 0", ae)
	}
	return ae
}

func TestNormalize_ImageFreeInputKeepsEveryByte(t *testing.T) {
	for _, w := range imageFreeWitnesses() {
		t.Run(w.name, func(t *testing.T) {
			qs := Questions{
				"n": Noul{Instructions: w.value, True: w.value, False: w.value},
				"c": Choice{Instructions: w.value, Options: map[string]any{"a": w.value}},
				"s": Score{Instructions: w.value, Levels: []any{w.value, w.value}},
			}
			for _, s := range []ImageSupport{{}, {Placements: PlaceAll}, {Placements: PlaceAll, Label: LabelPosition}} {
				req, err := Normalize(w.value, qs, s)
				if err != nil {
					t.Fatalf("Normalize: %v", err)
				}
				if string(req.State) != w.want {
					t.Errorf("state = %s, want %s", req.State, w.want)
				}
				if req.Images != nil {
					t.Errorf("Images = %v, want nil", req.Images)
				}
				got := map[string][]string{}
				for _, q := range req.Questions {
					fields := []string{string(q.Instructions), string(q.True), string(q.False)}
					for _, o := range q.Options {
						fields = append(fields, string(o))
					}
					for _, l := range q.Levels {
						fields = append(fields, string(l))
					}
					got[q.ID] = fields
				}
				want := map[string][]string{
					"n": {w.want, w.want, w.want},
					"c": {w.want, "", "", w.want},
					"s": {w.want, "", "", w.want, w.want},
				}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("questions = %q, want %q", got, want)
				}
			}
		})
	}
}

// imageTree builds one state and question set holding an image at every
// supported shape; at(i) is the stand-in for the i-th occurrence and raw(i)
// its form inside a json.RawMessage.
func imageTree(at func(i int) any, raw func(i int) string, msg any) (any, Questions) {
	state := map[string]any{
		"title":  "t",
		"pic":    at(0),
		"frames": []any{1, map[string]any{"frame": at(1)}, "x"},
		"msg":    msg,
		"raw":    json.RawMessage(`{"before":1.50,"img":` + raw(2) + `,"after":[9007199254740993]}`),
		"decoy":  ref123,
		"decoy2": "[image 1]",
		"nested": struct {
			A   int
			Img any
			Z   string
		}{1, at(3), "z"},
	}
	return state, Questions{
		"n": Noul{Instructions: []any{"see", at(4)}, True: map[string]any{"img": at(5)}, False: at(6)},
		"c": Choice{Instructions: at(7), Options: map[string]any{"b": []any{at(8)}, "a": at(9)}},
		"s": Score{Instructions: "i", Levels: []any{"low", map[string]any{"img": at(10)}}},
	}
}

func TestNormalize_ImageIsReplacedByItsLabel(t *testing.T) {
	blk := pngBlock(1, 2, 3)
	for _, tt := range []struct {
		name  string
		s     ImageSupport
		label string
	}{
		{"ref", ImageSupport{Placements: PlaceAll}, ref123},
		{"position", ImageSupport{Placements: PlaceAll, Label: LabelPosition}, "[image 1]"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rawImg := `{"media_type":"image/png","kind":"image","data":"AQID"}`
			msg := llmkit.Message{Role: llmkit.RoleUser, Content: []llmkit.Block{{Kind: llmkit.BlockText, Text: "look"}, blk}}
			state, qs := imageTree(func(int) any { return blk }, func(int) string { return rawImg }, msg)
			req, err := Normalize(state, qs, tt.s)
			if err != nil {
				t.Fatalf("Normalize: %v", err)
			}

			labelJSON := mustMarshal(t, tt.label)
			wantMsg := json.RawMessage(`{"role":"user","content":[{"kind":"text","text":"look"},` + labelJSON + `]}`)
			wantState, wantQs := imageTree(func(int) any { return tt.label }, func(int) string { return labelJSON }, wantMsg)
			want := mustMarshal(t, wantState)
			if string(req.State) != want {
				t.Errorf("state =\n%s\nwant\n%s", req.State, want)
			}
			wantReq, err := Normalize(wantState, wantQs, ImageSupport{})
			if err != nil {
				t.Fatalf("Normalize(labels): %v", err)
			}
			if !reflect.DeepEqual(req.Questions, wantReq.Questions) {
				t.Errorf("questions =\n%+v\nwant\n%+v", req.Questions, wantReq.Questions)
			}
			if len(req.Images) != 1 || req.Images[0].Ref != ref123 {
				t.Fatalf("Images = %+v, want the one image", req.Images)
			}
		})
	}
}

func TestNormalize_ImagesAreDistinctInFirstOccurrenceOrder(t *testing.T) {
	// ds[i] is the i-th occurrence; SHA-256 descends with i, the reverse of
	// the order a hash sort would give.
	const n = 10
	ds := make([][]byte, n)
	for i := range ds {
		ds[i] = []byte("image-bytes-" + strconv.Itoa(i))
	}
	sort.Slice(ds, func(i, j int) bool {
		a, b := sha256.Sum256(ds[i]), sha256.Sum256(ds[j])
		return bytes.Compare(a[:], b[:]) > 0
	})
	blk := func(i int) any { return llmkit.Block{Kind: llmkit.BlockImage, MediaType: "image/png", Data: ds[i]} }
	tree := func(at func(i int) any) (any, Questions) {
		state := []any{at(0), map[string]any{"k": at(1)}, at(0)}
		return state, Questions{
			"n": Noul{Instructions: at(5), True: at(6), False: at(7)},
			"c": Choice{Instructions: at(2), Options: map[string]any{"b": at(4), "a": at(3), "c": at(1)}},
			"s": Score{Instructions: at(8), Levels: []any{at(9), at(2), at(9)}},
		}
	}
	// Occurrence order: state 0,1,0; c: ins 2, a 3, b 4, c 1; n: ins 5, true 6,
	// false 7; s: ins 8, levels 9,2,9. Index 5 is first met after question c.
	wantPlace := []Placement{
		PlaceState, PlaceState | PlaceCriteria, PlaceInstructions | PlaceCriteria, PlaceCriteria, PlaceCriteria,
		PlaceInstructions, PlaceCriteria, PlaceCriteria, PlaceInstructions, PlaceCriteria,
	}
	for _, tt := range []struct {
		name  string
		label ImageLabel
		at    func(i int) string
	}{
		{"ref", LabelRef, func(i int) string { return refOf(ds[i]) }},
		{"position", LabelPosition, func(i int) string { return "[image " + strconv.Itoa(i+1) + "]" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			state, qs := tree(blk)
			wantState, wantQs := tree(func(i int) any { return tt.at(i) })
			for range 20 {
				req, err := Normalize(state, qs, ImageSupport{Placements: PlaceAll, Label: tt.label})
				if err != nil {
					t.Fatalf("Normalize: %v", err)
				}
				if got, want := string(req.State), mustMarshal(t, wantState); got != want {
					t.Fatalf("state = %s, want %s", got, want)
				}
				wantReq, err := Normalize(wantState, wantQs, ImageSupport{})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(req.Questions, wantReq.Questions) {
					t.Fatalf("questions =\n%+v\nwant\n%+v", req.Questions, wantReq.Questions)
				}
				if len(req.Images) != n {
					t.Fatalf("len(Images) = %d, want %d", len(req.Images), n)
				}
				for i, im := range req.Images {
					sum := sha256.Sum256(ds[i])
					want := Image{Ref: refOf(ds[i]), SHA256: hex.EncodeToString(sum[:]), MediaType: "image/png", Data: ds[i], Placements: wantPlace[i]}
					if !reflect.DeepEqual(im, want) {
						t.Errorf("Images[%d] = %+v, want %+v", i, im, want)
					}
				}
			}
		})
	}
}

func TestNormalize_UserStringEqualToALabelIsNotRewritten(t *testing.T) {
	state := map[string]any{"a": "[image 1]", "b": pngBlock(1, 2, 3), "c": ref123}
	req, err := Normalize(state, oneQuestion("i"), ImageSupport{Placements: PlaceAll, Label: LabelPosition})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"a":"[image 1]","b":"[image 1]","c":` + ref123JSON + `}`; string(req.State) != want {
		t.Errorf("state = %s, want %s", req.State, want)
	}
}

func TestNormalize_ImageValueDetection(t *testing.T) {
	const (
		pngMember = `"data":"AQID","media_type":"image/png"`
		kelvin    = "\u212a"
	)
	tests := []struct {
		name  string
		state string
		want  string // the state after Normalize
		image bool
	}{
		{"canonical", imgJSON, ref123JSON, true},
		{"foreign member stays", `{"kind":"image","caption":"x"}`, `{"kind":"image","caption":"x"}`, false},
		{"foreign member, empty data, stays", `{"kind":"image","data":"","caption":"x"}`, `{"kind":"image","data":"","caption":"x"}`, false},
		{"foreign member, null data, stays", `{"kind":"image","data":null,"caption":"x"}`, `{"kind":"image","data":null,"caption":"x"}`, false},
		{"foreign member, kind text, stays", `{"kind":"text","data":"AQID","caption":"x"}`, `{"kind":"text","data":"AQID","caption":"x"}`, false},
		{"foreign member, last kind text, stays", `{"kind":"image","KIND":"text","data":"AQID","caption":"x"}`, `{"kind":"image","KIND":"text","data":"AQID","caption":"x"}`, false},
		{"foreign member, last data empty, stays", `{"kind":"image","data":"AQID","DATA":"","caption":"x"}`, `{"kind":"image","data":"AQID","DATA":"","caption":"x"}`, false},
		{"kind Image stays", `{"kind":"Image",` + pngMember + `}`, `{"kind":"Image",` + pngMember + `}`, false},
		{"kind text stays", `{"kind":"text","text":"x"}`, `{"kind":"text","text":"x"}`, false},
		{"case variant names", `{"KIND":"image","DATA":"AQID","Media_Type":"image/png"}`, ref123JSON, true},
		{"duplicate kind, last image", `{"kind":"text","kind":"image",` + pngMember + `}`, ref123JSON, true},
		{"duplicate kind, last text", `{"kind":"image","kind":"text",` + pngMember + `}`, `{"kind":"image","kind":"text",` + pngMember + `}`, false},
		{"duplicate case variants", `{"kind":"text","KIND":"image",` + pngMember + `}`, ref123JSON, true},
		{"escaped values", `{"kind":"\u0069mage","data":"AQID","media_type":"\u0069mage/png"}`, ref123JSON, true},
		{"escaped names and values", `{"\u006bind":"\u0069mage","\u0064ata":"AQID","\u006dedia_type":"\u0069mage/png"}`, ref123JSON, true},
		{"escaped upper-case name", `{"\u004bIND":"image",` + pngMember + `}`, ref123JSON, true},
		{"escaped object nested in a raw value", `{"x":{"\u006bind":"\u0069mage","\u0064ata":"AQID","\u006dedia_type":"\u0069mage/png"}}`, `{"x":` + ref123JSON + `}`, true},
		{"escaped upper-case name nested", `{"x":[{"\u004bIND":"image",` + pngMember + `}]}`, `{"x":[` + ref123JSON + `]}`, true},
		{"kelvin sign name", `{"` + kelvin + `ind":"image",` + pngMember + `}`, ref123JSON, true},
		{"escaped kelvin sign name", `{"\u212aind":"image",` + pngMember + `}`, ref123JSON, true},
		{"member order", `{"media_type":"image/png","kind":"image","data":"AQID"}`, ref123JSON, true},
		{"empty text is valid", `{"kind":"image","text":"",` + pngMember + `}`, ref123JSON, true},
		{"image under a non-image block object", `{"kind":"text","text":` + imgJSON + `}`, `{"kind":"text","text":` + ref123JSON + `}`, true},
		{"image under a foreign object", `{"x":1,"y":[` + imgJSON + `,2]}`, `{"x":1,"y":[` + ref123JSON + `,2]}`, true},
		{"braces in strings are not objects", `{"s":"{\"kind\":\"image\"}"}`, `{"s":"{\"kind\":\"image\"}"}`, false},
		{"siblings keep their bytes", `{"a":1.50,"i":` + imgJSON + `,"z":"<"}`, `{"a":1.50,"i":` + ref123JSON + `,"z":"\u003c"}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := Normalize(json.RawMessage(tt.state), oneQuestion("i"), ImageSupport{Placements: PlaceAll})
			if err != nil {
				t.Fatalf("Normalize: %v", err)
			}
			if string(req.State) != tt.want {
				t.Errorf("state = %s, want %s", req.State, tt.want)
			}
			if got := len(req.Images) == 1; got != tt.image {
				t.Errorf("Images = %+v, want an image: %v", req.Images, tt.image)
			}
		})
	}
}

func TestNormalize_RefusesAnInvalidImage(t *testing.T) {
	tests := []struct {
		name  string
		state string
		want  string
	}{
		{"member fails to decode", `{"kind":"image","data":"AQID","media_type":5}`, `state: invalid image: member does not decode: json: cannot unmarshal number into Go struct field Block.media_type of type string`},
		{"member fails to decode, case variant name", `{"KIND":"image","DATA":"AQID","MEDIA_TYPE":5}`, `state: invalid image: member does not decode: json: cannot unmarshal number into Go struct field Block.media_type of type string`},
		{"duplicate kind, bad last", `{"kind":"image","kind":5,` + `"data":"AQID","media_type":"image/png"}`, `state: invalid image: member does not decode: json: cannot unmarshal number into Go struct field Block.kind of type llmkit.BlockKind`},
		{"data is not base64", `{"kind":"image","data":"!!","media_type":"image/png"}`, `state: invalid image: member does not decode: illegal base64 data at input byte 0`},
		{"no data", `{"kind":"image","media_type":"image/png"}`, `state: invalid image: no inline data`},
		{"empty data", `{"kind":"image","data":"","media_type":"image/png"}`, `state: invalid image: no inline data`},
		{"no media type", `{"kind":"image","data":"AQID"}`, `state: invalid image: media_type "" must begin "image/"`},
		{"document media type", `{"kind":"image","data":"AQID","media_type":"application/pdf"}`, `state: invalid image: media_type "application/pdf" must begin "image/"`},
		{"upper-case media type", `{"kind":"image","data":"AQID","media_type":"IMAGE/png"}`, `state: invalid image: media_type "IMAGE/png" must begin "image/"`},
		{"url image", `{"kind":"image","url":"https://example.com/a.png"}`, `state: invalid image: url is set; an image carries only data and media_type`},
		{"url and data", `{"kind":"image","url":"https://example.com/a.png","data":"AQID","media_type":"image/png"}`, `state: invalid image: url is set; an image carries only data and media_type`},
		{"text", `{"kind":"image","text":"a cat","data":"AQID","media_type":"image/png"}`, `state: invalid image: text is set; an image carries only data and media_type`},
		{"title", `{"kind":"image","title":"t","data":"AQID","media_type":"image/png"}`, `state: invalid image: title is set; an image carries only data and media_type`},
		{"provider", `{"kind":"image","provider":"p","data":"AQID","media_type":"image/png"}`, `state: invalid image: provider is set; an image carries only data and media_type`},
		{"raw", `{"kind":"image","raw":{},"data":"AQID","media_type":"image/png"}`, `state: invalid image: raw is set; an image carries only data and media_type`},
		{"raw null", `{"kind":"image","raw":null,"data":"AQID","media_type":"image/png"}`, `state: invalid image: raw is set; an image carries only data and media_type`},
		{"image inside a matched raw is not counted", `{"kind":"image","data":"AQID","media_type":"image/png","raw":` + imgJSON + `}`, `state: invalid image: raw is set; an image carries only data and media_type`},
		{"escaped invalid image", `{"\u006bind":"\u0069mage","\u0075rl":"u"}`, `state: invalid image: url is set; an image carries only data and media_type`},
		{"path of an array element", `{"frames":[0,1,{"frame":{"kind":"image"}}]}`, `state["frames"][2]["frame"]: invalid image: no inline data`},
		{"path key is quoted", `{"a\"b":{"kind":"image"}}`, `state["a\"b"]: invalid image: no inline data`},
		{"path of a nested escaped key", `{"\u0061":[[{"kind":"image"}]]}`, `state["a"][0][0]: invalid image: no inline data`},
		{"foreign member", `{"kind":"image","data":"AQID","media_type":"image/png","caption":"x"}`, `state: invalid image: member "caption" is not an llmkit.Block field`},
		{"foreign member, data not a string", `{"kind":"image","data":5,"source":"cam"}`, `state: invalid image: member "source" is not an llmkit.Block field`},
		{"foreign member, escaped names", `{"\u006bind":"image","\u0064ata":"AQID","zzz":1}`, `state: invalid image: member "zzz" is not an llmkit.Block field`},
		{"foreign member, escaped foreign name", `{"kind":"image","data":"AQID","\u007a":1,"y":2}`, `state: invalid image: member "z" is not an llmkit.Block field`},
		{"foreign member first, case variants, last duplicate wins", `{"x":1,"KIND":"text","Kind":"image","DATA":"","data":"AQID"}`, `state: invalid image: member "x" is not an llmkit.Block field`},
		{"foreign member, kelvin kind", `{"\u212aind":"image","data":"AQID","x":1}`, `state: invalid image: member "x" is not an llmkit.Block field`},
		{"foreign member, nested path", `{"a":[{"kind":"image","data":"AQID","x":1}]}`, `state["a"][0]: invalid image: member "x" is not an llmkit.Block field`},
		{"foreign member is not descended into", `{"kind":"image","data":"AQID","x":` + imgJSON + `}`, `state: invalid image: member "x" is not an llmkit.Block field`},
		{"same bytes, two media types", `[` + imgJSON + `,{"kind":"image","data":"AQID","media_type":"image/jpeg"}]`, `state[1]: image has media_type "image/jpeg", but the same bytes at state[0] have "image/png"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Normalize(json.RawMessage(tt.state), oneQuestion("i"), ImageSupport{Placements: PlaceAll})
			ae := apiError(t, err)
			if ae.Message != tt.want {
				t.Errorf("Message = %q, want %q", ae.Message, tt.want)
			}
			if errors.Is(err, ErrImageUnsupported) {
				t.Errorf("err wraps ErrImageUnsupported, want an invalid-image refusal without it")
			}
		})
	}
}

func TestNormalize_RefusesWhatTheSupportDoesNotCarry(t *testing.T) {
	img := json.RawMessage(imgJSON)
	in := func(slot string) (any, Questions) {
		switch slot {
		case "state":
			return map[string]any{"img": img}, oneQuestion("i")
		case "instructions":
			return "s", oneQuestion(map[string]any{"img": img})
		case "true":
			return "s", Questions{"q": Noul{Instructions: "i", True: img}}
		case "false":
			return "s", Questions{"q": Noul{Instructions: "i", False: []any{0, img}}}
		case "option":
			return "s", Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": img}}}
		default:
			return "s", Questions{"q": Score{Instructions: "i", Levels: []any{"lo", img}}}
		}
	}
	slots := []struct {
		name  string
		place Placement
		path  string
	}{
		{"state", PlaceState, `state["img"]`},
		{"instructions", PlaceInstructions, `questions["q"].instructions["img"]`},
		{"true", PlaceCriteria, `questions["q"].criteria.true`},
		{"false", PlaceCriteria, `questions["q"].criteria.false[1]`},
		{"option", PlaceCriteria, `questions["q"].options["a"]`},
		{"level", PlaceCriteria, `questions["q"].levels[1]`},
	}
	for _, sl := range slots {
		t.Run(sl.name, func(t *testing.T) {
			state, qs := in(sl.name)
			for _, s := range []ImageSupport{{}, {Placements: PlaceAll &^ sl.place}} {
				_, err := Normalize(state, qs, s)
				ae := apiError(t, err)
				want := sl.path + ": image in " + placementName(sl.place) + " is not supported by this backend"
				if ae.Message != want {
					t.Errorf("support %+v: Message = %q, want %q", s, ae.Message, want)
				}
				if !errors.Is(err, ErrImageUnsupported) || !errors.Is(err, llmkit.ErrInvalidRequest) {
					t.Errorf("support %+v: err = %v, want both ErrImageUnsupported and ErrInvalidRequest", s, err)
				}
			}
			req, err := Normalize(state, qs, ImageSupport{Placements: sl.place})
			if err != nil || len(req.Images) != 1 || req.Images[0].Placements != sl.place {
				t.Errorf("only %s carried: req = %+v, err = %v, want the image placed", sl.name, req.Images, err)
			}
		})
	}
}

func TestNormalize_MediaTypeSupport(t *testing.T) {
	webp := json.RawMessage(`{"kind":"image","data":"AQID","media_type":"image/webp"}`)
	s := ImageSupport{Placements: PlaceAll, MediaTypes: []string{"image/png", "image/jpeg"}}
	_, err := Normalize(map[string]any{"img": webp}, oneQuestion("i"), s)
	ae := apiError(t, err)
	if want := `state["img"]: image media_type "image/webp" is not supported by this backend (supported: image/png, image/jpeg)`; ae.Message != want {
		t.Errorf("Message = %q, want %q", ae.Message, want)
	}
	if !errors.Is(err, ErrImageUnsupported) || !errors.Is(err, llmkit.ErrInvalidRequest) {
		t.Errorf("err = %v, want both ErrImageUnsupported and ErrInvalidRequest", err)
	}
	if _, err := Normalize(map[string]any{"img": webp}, oneQuestion("i"), ImageSupport{Placements: PlaceAll}); err != nil {
		t.Errorf("empty MediaTypes refused image/webp: %v", err)
	}
	if _, err := Normalize(map[string]any{"img": json.RawMessage(imgJSON)}, oneQuestion("i"), s); err != nil {
		t.Errorf("listed media type refused: %v", err)
	}
}

func TestNormalize_ZeroSupportRefusesEveryImageAndFirstRefusalIsDeterministic(t *testing.T) {
	img := json.RawMessage(imgJSON)
	qs := Questions{
		"b": Noul{Instructions: img},
		"a": Choice{Instructions: "i", Options: map[string]any{"z": img, "y": img}},
	}
	for range 20 {
		_, err := Normalize("s", qs, ImageSupport{})
		if want := `questions["a"].options["y"]: image in criteria is not supported by this backend`; apiError(t, err).Message != want {
			t.Fatalf("Message = %q, want %q", apiError(t, err).Message, want)
		}
	}
}

func TestNormalize_StateIsScannedBeforeQuestions(t *testing.T) {
	bad := json.RawMessage(`{"kind":"image"}`)
	_, err := Normalize(map[string]any{"a": bad}, Questions{"q": Noul{Instructions: 7}}, ImageSupport{Placements: PlaceAll})
	if want := `state["a"]: invalid image: no inline data`; apiError(t, err).Message != want {
		t.Errorf("Message = %q, want %q", apiError(t, err).Message, want)
	}
}

func TestImageSupport_ZeroValueIsRefuseAllAndLabelRef(t *testing.T) {
	var s ImageSupport
	if s.Placements != 0 || len(s.MediaTypes) != 0 || s.Label != LabelRef {
		t.Errorf("zero ImageSupport = %+v", s)
	}
	if PlaceAll != PlaceState|PlaceInstructions|PlaceCriteria || PlaceState&PlaceInstructions != 0 || PlaceInstructions&PlaceCriteria != 0 || PlaceState&PlaceCriteria != 0 {
		t.Errorf("Placement bits overlap or PlaceAll is not their union")
	}
}

func TestBlockNamesFollowLlmkitBlock(t *testing.T) {
	want := []string{"kind", "text", "media_type", "data", "url", "title", "provider", "raw"}
	if !reflect.DeepEqual(blockNames, want) {
		t.Errorf("blockNames = %q, want %q: review image detection for the changed Block", blockNames, want)
	}
}

func TestNormalize_ImageLabelIsMarshalOfTheLabel(t *testing.T) {
	req, err := Normalize(pngBlock(1, 2, 3), oneQuestion("i"), ImageSupport{Placements: PlaceAll})
	if err != nil {
		t.Fatal(err)
	}
	if want := mustMarshal(t, req.Images[0].Ref); string(req.State) != want {
		t.Errorf("state = %s, want %s", req.State, want)
	}
	if got := fmt.Sprint(req.Images[0].SHA256); got != "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81" {
		t.Errorf("SHA256 = %s", got)
	}
}

func TestNormalize_ForeignMemberImageIsRefusedWhateverTheSupport(t *testing.T) {
	type captioned struct {
		llmkit.Block
		Caption string `json:"caption"`
	}
	state := map[string]any{"photo": captioned{Block: llmkit.Image("image/png", []byte{1, 2, 3}), Caption: "x"}}
	for _, s := range []ImageSupport{{}, {Placements: PlaceAll}, {Placements: PlaceAll, Label: LabelPosition}} {
		req, err := Normalize(state, oneQuestion("i"), s)
		if err == nil {
			t.Fatalf("support %+v: State = %s, want a refusal", s, req.State)
		}
		if want := `state["photo"]: invalid image: member "caption" is not an llmkit.Block field`; apiError(t, err).Message != want {
			t.Errorf("support %+v: Message = %q, want %q", s, apiError(t, err).Message, want)
		}
		if errors.Is(err, ErrImageUnsupported) {
			t.Errorf("support %+v: err wraps ErrImageUnsupported", s)
		}
	}
	_, err := Normalize("s", Questions{"q": Choice{Instructions: "i", Options: map[string]any{"a": state["photo"]}}}, ImageSupport{Placements: PlaceAll})
	if want := `questions["q"].options["a"]: invalid image: member "caption" is not an llmkit.Block field`; apiError(t, err).Message != want {
		t.Errorf("criteria: Message = %q, want %q", apiError(t, err).Message, want)
	}
}

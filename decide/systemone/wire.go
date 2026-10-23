package systemone

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

// wireRequest is the JSON envelope POSTed to /v1/systemone. Assets and
// Images are the two image dialects' top-level members; both are omitted
// when empty, so an image-free request is the same bytes in every dialect.
type wireRequest struct {
	State     json.RawMessage            `json:"state"`
	Model     string                     `json:"model"`
	Questions map[string]json.RawMessage `json:"questions"`
	Assets    map[string]wireAsset       `json:"assets,omitempty"`
	Images    []string                   `json:"images,omitempty"`
}

// wireAsset is one ImagesAssets entry. The host requires data_base64 to be
// standard padded base64 without a "data:" prefix.
type wireAsset struct {
	Type       string `json:"type"`
	Format     string `json:"format"`
	DataBase64 string `json:"data_base64"`
}

// wireQuestion is the JSON body of one question. Criteria is a map, slice,
// or noulCriteria built from decide.Normalize's validated raw JSON.
type wireQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     any             `json:"criteria,omitempty"`
}

// noulCriteria omits any side that is empty.
type noulCriteria struct {
	True  json.RawMessage `json:"true,omitempty"`
	False json.RawMessage `json:"false,omitempty"`
}

// marshalRequest renders the request body from a decide.Request, which
// already carries each image's label in State and Questions; d decides how
// req.Images ride along.
func marshalRequest(req decide.Request, model string, d ImageDialect) ([]byte, error) {
	wq := make(map[string]json.RawMessage, len(req.Questions))
	for _, dq := range req.Questions {
		raw, err := marshalQuestion(dq)
		if err != nil {
			return nil, err
		}
		wq[dq.ID] = raw
	}
	w := wireRequest{State: req.State, Model: model, Questions: wq}
	if err := addImages(&w, req.Images, d); err != nil {
		return nil, err
	}
	body, err := json.Marshal(w)
	if err != nil {
		return nil, fmt.Errorf("systemone: marshal request: %w", err)
	}
	return body, nil
}

// addImages puts images on w in dialect d's shape. images is non-empty only
// when decide.Normalize accepted them, so d is Assets or Array and every
// media type is one d carries; anything else is an error, never a guess.
func addImages(w *wireRequest, images []decide.Image, d ImageDialect) error {
	if len(images) == 0 {
		return nil
	}
	switch d {
	case ImagesAssets:
		w.Assets = make(map[string]wireAsset, len(images))
		for _, img := range images {
			format, ok := assetFormat(img.MediaType)
			if !ok {
				return fmt.Errorf("systemone: marshal request: image %s: media_type %q has no assets format", img.Ref, img.MediaType)
			}
			w.Assets[img.Ref] = wireAsset{Type: "image", Format: format, DataBase64: base64.StdEncoding.EncodeToString(img.Data)}
		}
	case ImagesArray:
		w.Images = make([]string, len(images))
		for i, img := range images {
			w.Images[i] = "data:" + img.MediaType + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
		}
	default:
		return fmt.Errorf("systemone: marshal request: %d image(s) with Config.Images %v", len(images), d)
	}
	return nil
}

// assetFormat maps a media type to the assets "format" member.
func assetFormat(mediaType string) (string, bool) {
	switch mediaType {
	case "image/png":
		return "png", true
	case "image/jpeg":
		return "jpg", true
	}
	return "", false
}

func marshalQuestion(dq llmkit.DecisionQuestion) (json.RawMessage, error) {
	w := wireQuestion{Type: dq.Kind, Instructions: dq.Instructions}
	switch dq.Kind {
	case "noul":
		if len(dq.True) > 0 || len(dq.False) > 0 {
			w.Criteria = noulCriteria{True: dq.True, False: dq.False}
		}
	case "choice":
		w.Criteria = dq.Options
	case "score":
		w.Criteria = dq.Levels
	default:
		return nil, fmt.Errorf("systemone: marshal request: question %q: unknown kind %q", dq.ID, dq.Kind)
	}
	raw, err := json.Marshal(w)
	if err != nil {
		return nil, fmt.Errorf("systemone: marshal request: %w", err)
	}
	return raw, nil
}

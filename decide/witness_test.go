package decide

import (
	"encoding/json"

	"github.com/dpoage/llmkit"
)

// imageFreeWitness is a value with no image value in it and the bytes
// encoding/json writes for it.
type imageFreeWitness struct {
	name  string
	value any
	want  string
}

type unsortedFields struct {
	Zebra string
	Apple int
	Mango []string
}

func imageFreeWitnesses() []imageFreeWitness {
	return []imageFreeWitness{
		{"string", "plain", `"plain"`},
		{"unsorted struct fields", unsortedFields{"z", 1, []string{"b", "a"}}, `{"Zebra":"z","Apple":1,"Mango":["b","a"]}`},
		{"nested maps sort", map[string]any{"b": 1, "a": map[string]any{"d": nil, "c": []int{3, 1}}}, `{"a":{"c":[3,1],"d":null},"b":1}`},
		{"raw number forms", json.RawMessage(`{ "n": 1.50 , "big": 12345678901234567890, "i": 9007199254740993, "e": 1E+2 }`), `{"n":1.50,"big":12345678901234567890,"i":9007199254740993,"e":1E+2}`},
		{"raw keeps member order", json.RawMessage(`{"z":1,"a":2}`), `{"z":1,"a":2}`},
		{"html escapes", "<a&b>", `"\u003ca\u0026b\u003e"`},
		{"raw html and unicode escapes", json.RawMessage(`{"s":"\u00e9\u0041","t":"<x>&","u":"\u2028"}`), `{"s":"\u00e9\u0041","t":"\u003cx\u003e\u0026","u":"\u2028"}`},
		{"user string shaped like a ref", "<image:039058c6f2c0cb492c533b0a4d14ef77>", `"\u003cimage:039058c6f2c0cb492c533b0a4d14ef77\u003e"`},
		{"user string shaped like a label", []string{"[image 1]"}, `["[image 1]"]`},
		{"text block", llmkit.Block{Kind: llmkit.BlockText, Text: "hi"}, `{"kind":"text","text":"hi"}`},
		{"document block", llmkit.Block{Kind: llmkit.BlockDocument, MediaType: "application/pdf", Data: []byte{1, 2, 3}}, `{"kind":"document","media_type":"application/pdf","data":"AQID"}`},
		{"kind image with a foreign member", json.RawMessage(`{"kind":"image","caption":"x"}`), `{"kind":"image","caption":"x"}`},
		{"kind Image", json.RawMessage(`{"kind":"Image"}`), `{"kind":"Image"}`},
		{"kind not a string", json.RawMessage(`{"kind":["image"],"data":"AQID"}`), `{"kind":["image"],"data":"AQID"}`},
		{"escaped kind text", json.RawMessage(`{"\u006bind":"\u0074ext","text":"x"}`), `{"\u006bind":"\u0074ext","text":"x"}`},
		{"empty object and array", []any{map[string]any{}, []any{}}, `[{},[]]`},
		{"foreign object in a block-named one", json.RawMessage(`{"kind":"text","text":{"x":{"kind":"image","caption":1}}}`), `{"kind":"text","text":{"x":{"kind":"image","caption":1}}}`},
		{"braces in strings", json.RawMessage(`{"s":"{\"kind\":\"image\"}","t":"}]"}`), `{"s":"{\"kind\":\"image\"}","t":"}]"}`},
	}
}

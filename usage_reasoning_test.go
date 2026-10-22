package llmkit

import (
	"reflect"
	"testing"
)

func TestUsage_ChargeableTokensIgnoresReasoning(t *testing.T) {
	bases := []Usage{
		{},
		{InputTokens: 1000, OutputTokens: 100},
		{InputTokens: 10000, OutputTokens: 500, CacheReadInputTokens: 8000},
		{InputTokens: 100, CacheReadInputTokens: 500, CacheCreationInputTokens: 7},
		{InputTokens: 7, OutputTokens: 2050, CacheCreationInputTokens: 3},
	}
	for _, u := range bases {
		for _, w := range []float64{0, 0.1, 0.5, 1} {
			want := u.ChargeableTokens(w)
			for _, k := range []int64{0, 1, 64, 2000, 1 << 40} {
				v := u
				v.ReasoningTokens = k
				if got := v.ChargeableTokens(w); got != want {
					t.Errorf("ChargeableTokens(%+v, w=%v) = %d, want %d (ReasoningTokens is a subset of OutputTokens)", v, w, got, want)
				}
			}
		}
	}
}

// TestUsage_AddSumsEveryField walks the struct by reflection so a field
// added to Usage but missed in Add fails here.
func TestUsage_AddSumsEveryField(t *testing.T) {
	typ := reflect.TypeOf(Usage{})
	var u, v Usage
	uv, vv := reflect.ValueOf(&u).Elem(), reflect.ValueOf(&v).Elem()
	for i := range typ.NumField() {
		switch typ.Field(i).Type.Kind() {
		case reflect.Int64:
			uv.Field(i).SetInt(int64(100 + 10*i + 1))
			vv.Field(i).SetInt(int64(1000 + 100*i + 7))
		case reflect.Bool:
			uv.Field(i).SetBool(false)
			vv.Field(i).SetBool(true)
		default:
			t.Fatalf("Usage.%s has kind %s: teach this test its Add rule", typ.Field(i).Name, typ.Field(i).Type.Kind())
		}
	}
	got := reflect.ValueOf(u.Add(v))
	for i := range typ.NumField() {
		name := typ.Field(i).Name
		switch typ.Field(i).Type.Kind() {
		case reflect.Int64:
			if want := uv.Field(i).Int() + vv.Field(i).Int(); got.Field(i).Int() != want {
				t.Errorf("Add dropped or mis-summed Usage.%s: got %d, want %d", name, got.Field(i).Int(), want)
			}
		case reflect.Bool:
			if !got.Field(i).Bool() {
				t.Errorf("Add did not OR Usage.%s", name)
			}
		}
	}
}

func TestEventSchemaVersionIs3(t *testing.T) {
	if EventSchemaVersion != 3 {
		t.Fatalf("EventSchemaVersion = %d, want 3 (google OutputTokens now includes thoughts)", EventSchemaVersion)
	}
}

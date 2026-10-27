package decide

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// depthShape builds a string leaf nested depth levels, size bytes in all
// (size is a floor of one leaf byte).
func depthShape(open, shut string, depth, size int) []byte {
	pad := max(size-depth*(len(open)+len(shut)), 1)
	return []byte(strings.Repeat(open, depth) + `"` + strings.Repeat("x", pad) + `"` + strings.Repeat(shut, depth))
}

// Bounds t(depth 4096)/t(depth 16) at equal input size.
func TestWalk_CostIsIndependentOfNestingDepth(t *testing.T) {
	const (
		bound  = 32
		rounds = 5
		giveUp = 4 * bound // after 2 rounds, a ratio this far over fails early
	)
	shapes := []struct {
		name       string
		open, shut string
		size       int
	}{
		{"raw", `{"raw":`, `}`, 100_000},
		{"text", `{"text":`, `}`, 100_000},
		{"a", `{"a":`, `}`, 100_000},
		{"kindobj", `{"data":"x","z":1,"kind":`, `}`, 150_000},
	}
	entries := []struct {
		name string
		run  func(t *testing.T, raw []byte)
	}{
		{"Normalize", func(t *testing.T, raw []byte) {
			if _, err := Normalize(json.RawMessage(raw), oneQuestion("Is it?"), ImageSupport{Placements: PlaceAll}); err != nil {
				t.Fatalf("Normalize: %v", err)
			}
		}},
		{"inlineImageHashes", func(t *testing.T, raw []byte) {
			set := map[string]struct{}{}
			inlineImageHashes(raw, set)
			if len(set) != 0 {
				t.Fatalf("hashes = %v, want none", set)
			}
		}},
	}
	for _, sh := range shapes {
		for _, en := range entries {
			t.Run(sh.name+"/"+en.name, func(t *testing.T) {
				shallow := depthShape(sh.open, sh.shut, 16, sh.size)
				deep := depthShape(sh.open, sh.shut, 4096, sh.size)
				if !json.Valid(shallow) || !json.Valid(deep) {
					t.Fatal("shape is not valid JSON")
				}
				best := func(raw []byte, prev time.Duration) time.Duration {
					start := time.Now()
					en.run(t, raw)
					return min(prev, time.Since(start))
				}
				tShallow, tDeep := time.Duration(1<<62), time.Duration(1<<62)
				for round := 1; round <= rounds; round++ {
					tShallow = best(shallow, tShallow)
					tDeep = best(deep, tDeep)
					ratio := float64(tDeep) / float64(tShallow)
					if ratio > giveUp && round >= 2 {
						t.Fatalf("depth 4096 / depth 16 = %.0f after %d rounds (%v / %v), bound %d", ratio, round, tDeep, tShallow, bound)
					}
					if round == rounds {
						t.Logf("depth 4096 / depth 16 = %.1f (%v / %v)", ratio, tDeep, tShallow)
						if ratio > bound {
							t.Fatalf("depth 4096 / depth 16 = %.1f, bound %d", ratio, bound)
						}
					}
				}
			})
		}
	}
}

func nestedArrays(depth int) any {
	var v any = 1
	for range depth {
		v = []any{v}
	}
	return v
}

// Depth 10001 is past encoding/json's decode limit; 9990 is not.
func TestNormalize_ImageKindObjectWithDeepMemberIsRefused(t *testing.T) {
	supports := map[string]ImageSupport{
		"zero":     {},
		"all":      {Placements: PlaceAll},
		"state":    {Placements: PlaceState},
		"media":    {Placements: PlaceAll, MediaTypes: []string{"image/png"}},
		"position": {Placements: PlaceAll, Label: LabelPosition},
	}
	for _, depth := range []int{9990, 10001} {
		state := map[string]any{"img": map[string]any{
			"kind": "image", "data": []byte{1}, "media_type": "image/png",
			"text": nestedArrays(depth),
		}}
		for name, sup := range supports {
			t.Run(fmt.Sprintf("%d/%s", depth, name), func(t *testing.T) {
				_, err := Normalize(state, oneQuestion("Is it?"), sup)
				if err == nil {
					t.Fatal("Normalize returned nil; the object went out as text")
				}
				ae := apiError(t, err)
				if want := `state["img"]: invalid image: member does not decode: `; !strings.HasPrefix(ae.Message, want) {
					t.Fatalf("message = %q, want prefix %q", ae.Message, want)
				}
				if errors.Is(err, ErrImageUnsupported) {
					t.Fatalf("err = %v wraps ErrImageUnsupported", err)
				}
			})
		}
	}
}

// Depth 10001 is past json.Valid's limit; 9990 is not.
func TestInlineImageHashes_DeepImageKindObjectAddsNothing(t *testing.T) {
	for _, depth := range []int{9990, 10001} {
		raw, err := json.Marshal(map[string]any{
			"kind": "image", "data": []byte{1}, "media_type": "image/png",
			"text": nestedArrays(depth),
		})
		if err != nil {
			t.Fatal(err)
		}
		set := map[string]struct{}{}
		inlineImageHashes(raw, set)
		if len(set) != 0 {
			t.Fatalf("depth %d: hashes = %v, want none", depth, set)
		}
	}
}

package decide_test

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/agent"
	"github.com/dpoage/llmkit/decide"
)

func sha256hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// beliefDecider answers the Noul question "q" with its belief.
type beliefDecider struct{ belief float64 }

func (d beliefDecider) Ask(context.Context, any, decide.Questions) (decide.Response, error) {
	return decide.Response{Model: "m", Nouls: map[string]float64{"q": d.belief}}, nil
}

var imgQ = decide.Questions{"q": decide.Noul{Instructions: "Is it red?"}}

func img(media, data string) llmkit.Block { return llmkit.Image(media, []byte(data)) }

func stepCtx(run llmkit.RunID, step int) context.Context {
	return llmkit.WithStep(llmkit.WithRun(context.Background(), run), step)
}

// recordedRun observes one Ask per state at consecutive steps from 1, each
// answering belief 0.1*step, and returns the replay Decider.
func recordedRun(t *testing.T, states ...any) (decide.Decider, llmkit.RunID) {
	t.Helper()
	var mu sync.Mutex
	var events []llmkit.Event
	sink := llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, ev)
	})
	run := llmkit.RunID("img-run")
	for i, s := range states {
		d := decide.Observe(beliefDecider{belief: float64(i+1) / 10}, sink)
		if _, err := d.Ask(stepCtx(run, i+1), s, imgQ); err != nil {
			t.Fatal(err)
		}
	}
	return replayOf(t, run, events), run
}

func replayOf(t *testing.T, run llmkit.RunID, events []llmkit.Event) decide.Decider {
	t.Helper()
	d, err := decide.Replay(&agent.Transcript{Record: events, RunID: run}, run)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	return d
}

func wantBelief(t *testing.T, d decide.Decider, run llmkit.RunID, step int, state any, want float64) {
	t.Helper()
	got, err := d.Ask(stepCtx(run, step), state, imgQ)
	if err != nil {
		t.Fatalf("step %d: Ask: %v", step, err)
	}
	if got.Nouls["q"] != want {
		t.Errorf("step %d: belief = %v, want %v", step, got.Nouls["q"], want)
	}
}

func wantDiverged(t *testing.T, d decide.Decider, run llmkit.RunID, step int, state any) {
	t.Helper()
	_, err := d.Ask(stepCtx(run, step), state, imgQ)
	if !errors.Is(err, decide.ErrReplayDiverged) {
		t.Fatalf("step %d: err = %v, want ErrReplayDiverged", step, err)
	}
	if want := "step " + string(rune('0'+step)); !strings.Contains(err.Error(), want) {
		t.Errorf("step %d: error %q does not name %q", step, err, want)
	}
}

func TestReplay_ImageAskMatchesOnTheHashSet(t *testing.T) {
	x, y := img("image/png", "image-X"), img("image/png", "image-Y")
	record := func(t *testing.T) (decide.Decider, llmkit.RunID) {
		return recordedRun(t,
			map[string]any{"frame": x},
			map[string]any{"a": x, "b": y},
			"no image",
		)
	}

	t.Run("same image answers whatever the state shape", func(t *testing.T) {
		d, run := record(t)
		wantBelief(t, d, run, 1, []any{"other", img("image/jpeg", "image-X")}, 0.1)
	})
	t.Run("different image diverges", func(t *testing.T) {
		d, run := record(t)
		wantDiverged(t, d, run, 1, map[string]any{"frame": y})
	})
	t.Run("a diverged Ask consumes nothing", func(t *testing.T) {
		d, run := record(t)
		wantDiverged(t, d, run, 1, map[string]any{"frame": y})
		wantBelief(t, d, run, 1, map[string]any{"frame": x}, 0.1)
	})
	t.Run("image-free Ask against an image record diverges", func(t *testing.T) {
		d, run := record(t)
		wantDiverged(t, d, run, 1, "no image here")
	})
	t.Run("image Ask against an image-free record diverges", func(t *testing.T) {
		d, run := record(t)
		wantDiverged(t, d, run, 3, map[string]any{"frame": x})
	})
	t.Run("two images match as a set, in any order", func(t *testing.T) {
		d, run := record(t)
		wantBelief(t, d, run, 2, []any{y, x}, 0.2)
	})
	t.Run("a subset of the images diverges", func(t *testing.T) {
		d, run := record(t)
		wantDiverged(t, d, run, 2, map[string]any{"a": x})
	})
	t.Run("a superset of the images diverges", func(t *testing.T) {
		d, run := record(t)
		wantDiverged(t, d, run, 1, map[string]any{"a": x, "b": y})
	})
	t.Run("the same image twice is one hash", func(t *testing.T) {
		d, run := record(t)
		wantBelief(t, d, run, 1, map[string]any{"a": x, "b": x}, 0.1)
	})
	t.Run("image-free Ask against an image-free record matches", func(t *testing.T) {
		d, run := record(t)
		wantBelief(t, d, run, 3, "another text", 0.3)
	})
}

func TestReplay_ImageInQuestionMatchesOnHash(t *testing.T) {
	x, y := img("image/png", "image-X"), img("image/png", "image-Y")
	var events []llmkit.Event
	sink := llmkit.ObserverFunc(func(_ context.Context, ev llmkit.Event) { events = append(events, ev) })
	ask := func(i llmkit.Block) decide.Questions {
		return decide.Questions{"q": decide.Noul{Instructions: []any{"compare", i}, True: i}}
	}
	if _, err := decide.Observe(beliefDecider{0.7}, sink).Ask(stepCtx("r", 1), "s", ask(x)); err != nil {
		t.Fatal(err)
	}
	d := replayOf(t, "r", events)
	if _, err := d.Ask(stepCtx("r", 1), "s", ask(y)); !errors.Is(err, decide.ErrReplayDiverged) {
		t.Errorf("different image in the question: err = %v, want ErrReplayDiverged", err)
	}
	got, err := d.Ask(stepCtx("r", 1), "s", ask(x))
	if err != nil || got.Nouls["q"] != 0.7 {
		t.Errorf("same image in the question: (%v, %v), want belief 0.7", got.Nouls, err)
	}
}

// legacyImages are the bytes the v0.6.0 generator recorded.
const (
	legacyA = "\x89PNG\r\n\x1a\nlegacy-image-A"
	legacyB = "\x89PNG\r\n\x1a\nlegacy-image-B"
	legacyC = "legacy-image-C-bytes"
	legacyD = "legacy-image-D-other"
)

func loadLegacy(t *testing.T) (decide.Decider, llmkit.RunID) {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "legacy_v060_images.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	tr, err := agent.LoadJSONL(f)
	if err != nil {
		t.Fatal(err)
	}
	d, err := decide.Replay(tr, tr.RunID)
	if err != nil {
		t.Fatalf("Replay of the v0.6.0 log: %v", err)
	}
	return d, tr.RunID
}

func TestReplay_LegacyV060Log(t *testing.T) {
	// The fixture holds five decisions, steps 1-5, answering 0.1..0.5:
	// (1) an inline image A in state; (2) A twice; (3) a URL image object;
	// (4) the same bytes C as image/png and image/jpeg; (5) image B in a
	// question's True.
	t.Run("(i) inline image answers the same image and diverges on another", func(t *testing.T) {
		d, run := loadLegacy(t)
		wantDiverged(t, d, run, 1, map[string]any{"frame": img("image/png", legacyD)})
		wantBelief(t, d, run, 1, map[string]any{"anything": img("image/png", legacyA)}, 0.1)
	})
	t.Run("(ii) the same image twice matches a one-hash Ask", func(t *testing.T) {
		d, run := loadLegacy(t)
		wantBelief(t, d, run, 2, map[string]any{"frame": img("image/png", legacyA)}, 0.2)
	})
	t.Run("(ii) and not an Ask with a second image", func(t *testing.T) {
		d, run := loadLegacy(t)
		wantDiverged(t, d, run, 2, []any{img("image/png", legacyA), img("image/png", legacyD)})
	})
	t.Run("(iii) a URL image adds no hash: an image-free Ask matches", func(t *testing.T) {
		d, run := loadLegacy(t)
		wantBelief(t, d, run, 3, "text only", 0.3)
	})
	t.Run("(iii) and an image Ask diverges", func(t *testing.T) {
		d, run := loadLegacy(t)
		wantDiverged(t, d, run, 3, map[string]any{"frame": img("image/png", legacyA)})
	})
	t.Run("(iv) one image under two media types is one hash", func(t *testing.T) {
		d, run := loadLegacy(t)
		wantBelief(t, d, run, 4, map[string]any{"frame": img("image/png", legacyC)}, 0.4)
	})
	t.Run("(v) an image in a question's criteria is a hash", func(t *testing.T) {
		d, run := loadLegacy(t)
		qs := func(b llmkit.Block) decide.Questions {
			return decide.Questions{"q": decide.Noul{Instructions: "Does it match the reference?", True: b}}
		}
		if _, err := d.Ask(stepCtx(run, 5), "plain text state", qs(img("image/png", legacyD))); !errors.Is(err, decide.ErrReplayDiverged) {
			t.Errorf("different criteria image: err = %v, want ErrReplayDiverged", err)
		}
		got, err := d.Ask(stepCtx(run, 5), "plain text state", qs(img("image/png", legacyB)))
		if err != nil || got.Nouls["q"] != 0.5 {
			t.Errorf("same criteria image: (%v, %v), want belief 0.5", got.Nouls, err)
		}
	})
}

// decisionRecord builds a recorded decision answering belief 0.9 whose
// State and Questions are the given raw JSON.
func decisionRecord(state string, q llmkit.DecisionQuestion, images ...llmkit.DecisionImage) llmkit.Event {
	ev := llmkit.NewEvent(stepCtx("hand", 1), llmkit.KindDecision)
	belief := 0.9
	q.ID, q.Kind = "q", "noul"
	ev.Decision = &llmkit.DecisionEvent{
		State:     []byte(state),
		Questions: []llmkit.DecisionQuestion{q},
		Answers:   []llmkit.DecisionAnswer{{ID: "q", Belief: &belief}},
		Images:    images,
	}
	return ev
}

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func TestReplay_InlineImagesInEveryRecordedPlace(t *testing.T) {
	x := `{"kind":"image","media_type":"image/png","data":"` + b64("image-X") + `"}`
	tests := []struct {
		name  string
		state string
		q     llmkit.DecisionQuestion
	}{
		{"state", `{"k":[1,{"deep":` + x + `}]}`, llmkit.DecisionQuestion{}},
		{"instructions", `"s"`, llmkit.DecisionQuestion{Instructions: []byte(`["see",` + x + `]`)}},
		{"true", `"s"`, llmkit.DecisionQuestion{True: []byte(x)}},
		{"false", `"s"`, llmkit.DecisionQuestion{False: []byte(x)}},
		{"option", `"s"`, llmkit.DecisionQuestion{Options: map[string]json.RawMessage{"a": []byte(x)}}},
		{"level", `"s"`, llmkit.DecisionQuestion{Levels: []json.RawMessage{[]byte(`"lo"`), []byte(x)}}},
		{"case and escape variants", `{"f":{"\u004bIND":"\u0069mage","Media_Type":"image/png","DATA":"` + b64("image-X") + `"}}`, llmkit.DecisionQuestion{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := replayOf(t, "hand", []llmkit.Event{decisionRecord(tt.state, tt.q)})
			if _, err := d.Ask(stepCtx("hand", 1), map[string]any{"f": img("image/png", "image-Y")}, imgQ); !errors.Is(err, decide.ErrReplayDiverged) {
				t.Errorf("other image: err = %v, want ErrReplayDiverged", err)
			}
			if _, err := d.Ask(stepCtx("hand", 1), "text", imgQ); !errors.Is(err, decide.ErrReplayDiverged) {
				t.Errorf("no image: err = %v, want ErrReplayDiverged", err)
			}
			got, err := d.Ask(stepCtx("hand", 1), map[string]any{"f": img("image/png", "image-X")}, imgQ)
			if err != nil || got.Nouls["q"] != 0.9 {
				t.Errorf("same image: (%v, %v), want belief 0.9", got.Nouls, err)
			}
		})
	}
}

func TestReplay_InvalidInlineImagesAddNoHashAndNeverFail(t *testing.T) {
	invalid := []string{
		`{"kind":"image","url":"https://example.invalid/x.png"}`,
		`{"kind":"image","media_type":"image/png","data":"` + b64("x") + `","text":"caption"}`,
		`{"kind":"image","media_type":"image/png","data":123}`,
		`{"kind":"image","media_type":"text/plain","data":"` + b64("x") + `"}`,
		`{"kind":"image","media_type":"image/png"}`,
		`{"kind":"image","media_type":"image/png","data":"` + b64("x") + `","caption":"foreign"}`,
	}
	for _, v := range invalid {
		d := replayOf(t, "hand", []llmkit.Event{decisionRecord(`{"f":`+v+`}`, llmkit.DecisionQuestion{})})
		got, err := d.Ask(stepCtx("hand", 1), "text only", imgQ)
		if err != nil || got.Nouls["q"] != 0.9 {
			t.Errorf("%s: image-free Ask = (%v, %v), want the record's answer", v, got.Nouls, err)
		}
	}
	// Malformed JSON in a record neither panics nor fails Replay.
	bad := decisionRecord(`{"f":{"kind":"image",`, llmkit.DecisionQuestion{})
	bad.Decision.Questions[0].Instructions = []byte(`{{`)
	d := replayOf(t, "hand", []llmkit.Event{bad})
	if _, err := d.Ask(stepCtx("hand", 1), "text only", imgQ); err != nil {
		t.Errorf("record with malformed JSON: %v", err)
	}
}

func TestReplay_RecordedImagesAndLegacyInlineImagesUnion(t *testing.T) {
	x := `{"kind":"image","media_type":"image/png","data":"` + b64("image-X") + `"}`
	hy := llmkit.DecisionImage{SHA256: sha256hex("image-Y"), MediaType: "image/png", Placements: []string{"state"}}
	// Images names Y, the inline State image is X, a ref string is not an image.
	d := replayOf(t, "hand", []llmkit.Event{decisionRecord(`{"x":`+x+`,"y":"<image:abc>"}`, llmkit.DecisionQuestion{}, hy, hy)})
	if _, err := d.Ask(stepCtx("hand", 1), []any{img("image/png", "image-X")}, imgQ); !errors.Is(err, decide.ErrReplayDiverged) {
		t.Errorf("X alone: err = %v, want ErrReplayDiverged", err)
	}
	got, err := d.Ask(stepCtx("hand", 1), []any{img("image/png", "image-X"), img("image/png", "image-Y")}, imgQ)
	if err != nil || got.Nouls["q"] != 0.9 {
		t.Errorf("X and Y: (%v, %v), want belief 0.9", got.Nouls, err)
	}
}

func TestObserveReplay_OneMiBImageRoundTripsThroughJSONLInUnder4KiB(t *testing.T) {
	data := make([]byte, 1<<20)
	rng := rand.New(rand.NewChaCha8([32]byte{1}))
	for i := range data {
		data[i] = byte(rng.Uint32())
	}
	other := append([]byte(nil), data...)
	other[len(other)/2] ^= 0xff

	dir := t.TempDir()
	sink := agent.JSONL(dir, func(err error) { t.Errorf("sink: %v", err) })
	run := llmkit.RunID("big-image")
	ctx := stepCtx(run, 1)
	start := llmkit.NewEvent(ctx, llmkit.KindStart)
	start.Start = &llmkit.StartEvent{Task: "big image"}
	sink.Observe(ctx, start)

	state := map[string]any{"frame": llmkit.Image("image/png", data)}
	if _, err := decide.Observe(beliefDecider{0.8}, sink).Ask(ctx, state, imgQ); err != nil {
		t.Fatal(err)
	}
	fin := llmkit.NewEvent(ctx, llmkit.KindFinalize)
	fin.Finalize = &llmkit.FinalizeEvent{Status: llmkit.RunCompleted}
	sink.Observe(ctx, fin)

	f, err := os.Open(filepath.Join(dir, string(run)+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 4<<20)
	lines := 0
	for sc.Scan() {
		if strings.Contains(sc.Text(), `"kind":"decision"`) {
			lines++
			if n := len(sc.Bytes()); n >= 4096 {
				t.Errorf("decision line is %d bytes, want under 4096", n)
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if lines != 1 {
		t.Fatalf("decision lines = %d, want 1", lines)
	}

	d, err := decide.Replay(sink, run)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if _, err := d.Ask(ctx, map[string]any{"frame": llmkit.Image("image/png", other)}, imgQ); !errors.Is(err, decide.ErrReplayDiverged) {
		t.Errorf("one byte different: err = %v, want ErrReplayDiverged", err)
	}
	got, err := d.Ask(ctx, state, imgQ)
	if err != nil || got.Nouls["q"] != 0.8 {
		t.Errorf("same image: (%v, %v), want belief 0.8", got.Nouls, err)
	}
}

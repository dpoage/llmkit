package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

type spillCall struct {
	call llmkit.ToolCall
	full []llmkit.Block
}

type recSink struct {
	mu    sync.Mutex
	calls []spillCall
	ref   string
	err   error
}

func (s *recSink) Spill(_ context.Context, call llmkit.ToolCall, full []llmkit.Block) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, spillCall{call, full})
	if s.err != nil {
		return "", s.err
	}
	if s.ref != "" {
		return s.ref, nil
	}
	return fmt.Sprintf("mem://%s", call.ID), nil
}

func (s *recSink) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.calls)
}

func spillBig(n int) string {
	var b strings.Builder
	for i := 0; b.Len() < n; i++ {
		fmt.Fprintf(&b, "line %06d\n", i)
	}
	return b.String()[:n]
}

type spillRun struct {
	out  *Outcome
	fc   *fakeClient
	run  llmkit.ToolRunEvent
	hook ToolEvent
	msg  llmkit.Message
}

func runSpill(t *testing.T, caps llmkit.Capabilities, blocks []llmkit.Block, opts ...Option) spillRun {
	t.Helper()
	fc := newFakeClient(toolResp("c1", "big", `{}`, 1, 1), textResp("done", 1, 1))
	fc.caps = caps
	var sr spillRun
	opts = append(opts, WithHooks(Hooks{ToolEnd: func(_ context.Context, ev ToolEvent) { sr.hook = ev }}))
	out, err := NewRunner(fc, []Tool{blocksTool{"big", blocks}}, "sys", opts...).Run(context.Background(), "task", WithRunID("sp"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fc.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fc.requests))
	}
	var runs []llmkit.ToolRunEvent
	for _, ev := range out.Transcript.Record {
		if ev.Kind == llmkit.KindToolRun {
			runs = append(runs, *ev.ToolRun)
		}
	}
	if len(runs) != 1 {
		t.Fatalf("ToolRun events = %d, want 1", len(runs))
	}
	msgs := fc.requests[1].Messages
	sr.out, sr.fc, sr.run, sr.msg = out, fc, runs[0], msgs[len(msgs)-1]
	return sr
}

func TestSpill_SP1_OversizedTextSpills(t *testing.T) {
	big := spillBig(100 << 10)
	sink := &recSink{ref: "mem://one"}
	sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(big)}, WithToolOutputLimit(16<<10, sink))

	if sink.count() != 1 {
		t.Fatalf("sink calls = %d, want 1", sink.count())
	}
	got := sink.calls[0]
	if got.call.ID != "c1" || len(got.full) != 1 || got.full[0].Text != big {
		t.Errorf("sink received call %q and %d blocks (first %d bytes), want c1 and the full %d bytes", got.call.ID, len(got.full), len(got.full[0].Text), len(big))
	}
	want := "[tool output spilled: mem://one, 102400 bytes]\n" + big[:spillPreviewBytes]
	if sr.msg.Role != llmkit.RoleToolResult || sr.msg.Text() != want || len(sr.msg.Content) != 1 {
		t.Errorf("tool-result message = %q (%d blocks), want exactly marker+preview (%d bytes)", clip(sr.msg.Text()), len(sr.msg.Content), len(want))
	}
	if len(sr.msg.Text()) > 16<<10 {
		t.Errorf("tool-result message is %d bytes, over the limit", len(sr.msg.Text()))
	}
	if sr.run.SpillRef != "mem://one" || sr.run.OriginalBytes != len(big) || sr.run.SpillError != "" || sr.run.Result != want {
		t.Errorf("ToolRun = {SpillRef:%q OriginalBytes:%d SpillError:%q Result:%q}", sr.run.SpillRef, sr.run.OriginalBytes, sr.run.SpillError, clip(sr.run.Result))
	}
	if sr.hook.Result != want {
		t.Errorf("ToolEnd hook Result = %q, want the model-visible text", clip(sr.hook.Result))
	}
	j, _ := json.Marshal(sr.run)
	if !strings.Contains(string(j), `"spill_ref":"mem://one"`) || !strings.Contains(string(j), `"original_bytes":102400`) {
		t.Errorf("event JSON lacks spill_ref/original_bytes: %.200s", j)
	}
}

func clip(s string) string {
	if len(s) > 80 {
		return s[:80] + "..."
	}
	return s
}

func TestSpill_SP2_ReplayServesRecordedVerbatim(t *testing.T) {
	big := spillBig(100 << 10)
	records := map[string]struct {
		blocks []llmkit.Block
		opts   []Option
	}{
		"spilled":           {[]llmkit.Block{llmkit.Text(big)}, []Option{WithToolOutputLimit(16<<10, &recSink{})}},
		"truncated":         {[]llmkit.Block{llmkit.Text(big)}, []Option{WithToolOutputLimit(16<<10, nil)}},
		"sink failed":       {[]llmkit.Block{llmkit.Text(big)}, []Option{WithToolOutputLimit(16<<10, &recSink{err: errors.New("disk full")})}},
		"unlimited replays": {[]llmkit.Block{llmkit.Text("small")}, nil},
	}
	for name, rec := range records {
		t.Run(name, func(t *testing.T) {
			recorded := runSpill(t, llmkit.Capabilities{}, rec.blocks, rec.opts...)
			rc, err := NewReplayClient(recorded.out.Transcript, recorded.out.RunID, llmkit.Capabilities{})
			if err != nil {
				t.Fatal(err)
			}
			replaySink := &recSink{}
			out, err := NewRunner(rc, rc.Tools(), "sys", WithToolOutputLimit(64, replaySink)).Run(context.Background(), "task", WithRunID("sp"))
			if err != nil || rc.Err() != nil {
				t.Fatalf("replay: err = %v, rc.Err() = %v", err, rc.Err())
			}
			if replaySink.count() != 0 {
				t.Errorf("replay called the sink %d times", replaySink.count())
			}
			if !reflect.DeepEqual(out.Messages, recorded.out.Messages) {
				t.Errorf("replayed history differs from recorded")
			}
			var rep llmkit.ToolRunEvent
			for _, ev := range out.Transcript.Record {
				if ev.Kind == llmkit.KindToolRun {
					rep = *ev.ToolRun
				}
			}
			if !reflect.DeepEqual(rep, recorded.run) {
				t.Errorf("replayed ToolRun = %+v, recorded %+v", rep, recorded.run)
			}
		})
	}
}

func TestSpill_SP3_TruncatesWithoutSink(t *testing.T) {
	big := strings.Repeat("€", 40000)
	for limit := 1000; limit < 1012; limit++ {
		sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(big)}, WithToolOutputLimit(limit, nil))
		got := sr.msg.Text()
		size := len(big)
		i := strings.LastIndex(got, "[tool output truncated: ")
		if i < 0 {
			t.Fatalf("limit %d: no truncation marker in %q", limit, clip(got))
		}
		head := got[:i]
		if want := fmt.Sprintf("[tool output truncated: %d of %d bytes shown]", len(head), size); got[i:] != want {
			t.Errorf("limit %d: marker = %q, want %q", limit, got[i:], want)
		}
		if !utf8.ValidString(got) || !strings.HasPrefix(big, head) {
			t.Errorf("limit %d: head is not a rune-aligned prefix of the text", limit)
		}
		if len(got) > limit {
			t.Errorf("limit %d: stand-in is %d bytes", limit, len(got))
		}
		if len(got)+3 <= limit {
			t.Errorf("limit %d: stand-in is %d bytes, wasted a whole rune", limit, len(got))
		}
		if sr.run.OriginalBytes != size || sr.run.SpillRef != "" || sr.run.Result != got {
			t.Errorf("limit %d: ToolRun = {OriginalBytes:%d SpillRef:%q}", limit, sr.run.OriginalBytes, sr.run.SpillRef)
		}
	}
}

func TestSpill_SP4_OffByDefault(t *testing.T) {
	big := spillBig(1 << 20)
	sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(big)})
	if sr.msg.Text() != big || sr.run.OriginalBytes != 0 || sr.run.SpillRef != "" || sr.run.Result != big {
		t.Errorf("no option: result changed (%d bytes, OriginalBytes %d)", len(sr.msg.Text()), sr.run.OriginalBytes)
	}
	for _, max := range []int{0, -1} {
		sink := &recSink{}
		sr = runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(big)}, WithToolOutputLimit(max, sink))
		if sr.msg.Text() != big || sink.count() != 0 {
			t.Errorf("limit %d: result changed or sink called", max)
		}
	}
}

func TestSpill_SP5_SinkFailureFallsBackAndIsRecorded(t *testing.T) {
	big := spillBig(100 << 10)
	rows := map[string]SpillSink{
		"error":          &recSink{err: errors.New("disk full")},
		"empty ref":      emptyRefSink{},
		"newline in ref": &recSink{ref: "a\nb"},
	}
	for name, sink := range rows {
		t.Run(name, func(t *testing.T) {
			sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(big)}, WithToolOutputLimit(16<<10, sink))
			got := sr.msg.Text()
			i := strings.LastIndex(got, "[tool output truncated: ")
			if i < 0 || got[i:] != truncationMarker(i, len(big)) || !strings.HasPrefix(big, got[:i]) {
				t.Errorf("message is not a head of the text plus its truncation marker: ...%q", got[max(0, len(got)-80):])
			}
			if len(got) > 16<<10 || sr.run.SpillRef != "" || sr.run.OriginalBytes != len(big) || sr.run.SpillError == "" || sr.run.Result != got {
				t.Errorf("ToolRun = {SpillRef:%q OriginalBytes:%d SpillError:%q}, len %d", sr.run.SpillRef, sr.run.OriginalBytes, sr.run.SpillError, len(got))
			}
			if name == "error" && !strings.Contains(sr.run.SpillError, "disk full") {
				t.Errorf("SpillError = %q, want the sink's error", sr.run.SpillError)
			}
		})
	}
}

type emptyRefSink struct{}

func (emptyRefSink) Spill(context.Context, llmkit.ToolCall, []llmkit.Block) (string, error) {
	return "", nil
}

func TestSpill_SP6_CompactionStubKeepsRef(t *testing.T) {
	big := spillBig(100 << 10)
	for _, ref := range []string{"mem://c1", "s3://b/k, with comma]"} {
		sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(big)}, WithToolOutputLimit(16<<10, &recSink{ref: ref}))
		msgs := []llmkit.Message{sr.msg, llmkit.ToolResult("c2", "x"), llmkit.ToolResult("c3", "x")}
		msgs[0].ToolCallID = "c1"
		out, pruned := compactHistory(msgs, msgs, 2, map[string]string{"c1": "big"})
		if pruned != 1 {
			t.Fatalf("pruned = %d, want 1", pruned)
		}
		want := fmt.Sprintf("[tool result pruned to save context: big, %d bytes; full output spilled: %s]", len(big), ref)
		if out[0].Text() != want {
			t.Errorf("stub = %q, want %q", out[0].Text(), want)
		}
		if !isCompactStub(out[0].Text()) {
			t.Errorf("a spilled stub must be recognized as a stub")
		}
	}
}

func TestSpill_SP7_MediaCountsTowardSize(t *testing.T) {
	img := make([]byte, 50<<10)
	copy(img, "\x89PNG\r\n\x1a\n")
	blocks := []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", img)}
	size := 3 + len(img)
	caps := llmkit.Capabilities{ToolResultMedia: true}

	sink := &recSink{ref: "mem://img"}
	sr := runSpill(t, caps, blocks, WithToolOutputLimit(16<<10, sink))
	wantText := fmt.Sprintf("[tool output spilled: mem://img, %d bytes]\ncap", size)
	if sink.count() != 1 || len(sink.calls[0].full) != 2 || !reflect.DeepEqual(sink.calls[0].full[1].Data, img) {
		t.Fatalf("sink did not receive the text and the image: %d calls", sink.count())
	}
	if sr.msg.Text() != wantText || len(sr.msg.Content) != 1 || sr.run.OriginalBytes != size || sr.run.SpillRef != "mem://img" {
		t.Errorf("spilled: message %q (%d blocks), ToolRun %+v", clip(sr.msg.Text()), len(sr.msg.Content), sr.run)
	}

	sr = runSpill(t, caps, blocks, WithToolOutputLimit(16<<10, nil))
	wantText = fmt.Sprintf("cap[tool output truncated: 3 of %d bytes shown]", size)
	if sr.msg.Text() != wantText || len(sr.msg.Content) != 1 || sr.run.OriginalBytes != size {
		t.Errorf("no sink: message %q (%d blocks), ToolRun %+v", sr.msg.Text(), len(sr.msg.Content), sr.run)
	}
}

func TestSpill_MediaDegradedBeforeSizing(t *testing.T) {
	img := make([]byte, 50<<10)
	sink := &recSink{}
	sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text("cap"), llmkit.Image("image/png", img)}, WithToolOutputLimit(16<<10, sink))
	if sink.count() != 0 || sr.run.OriginalBytes != 0 || sr.run.Degraded != 1 {
		t.Errorf("a degraded image must not count toward size: sink calls %d, ToolRun %+v", sink.count(), sr.run)
	}
}

func TestSpill_BoundaryAndTinyLimits(t *testing.T) {
	text := strings.Repeat("é", 500)
	size := len(text)
	sink := &recSink{ref: "r"}
	if sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(text)}, WithToolOutputLimit(size, sink)); sr.msg.Text() != text || sink.count() != 0 {
		t.Errorf("size == limit must pass untouched")
	}
	if sr := runSpill(t, llmkit.Capabilities{}, []llmkit.Block{llmkit.Text(text)}, WithToolOutputLimit(size-1, sink)); sink.count() != 1 || sr.run.OriginalBytes != size {
		t.Errorf("size == limit+1 must spill")
	}

	for limit := 1; limit <= size; limit++ {
		marker := spillMarker("r", size)
		sp := spillStandIn("r", size, text, limit)
		tr := truncatedStandIn(text, size, limit)
		for name, s := range map[string]string{"spill": sp, "truncate": tr} {
			if !utf8.ValidString(s) {
				t.Fatalf("%s limit %d: invalid UTF-8", name, limit)
			}
			if len(s) > limit && len(s) != len(marker) && name == "spill" {
				t.Fatalf("spill limit %d: %d bytes, neither within the limit nor the bare marker", limit, len(s))
			}
			if len(s) > limit && name == "truncate" && s != truncationMarker(0, size) {
				t.Fatalf("truncate limit %d: %d bytes and more than the bare marker", limit, len(s))
			}
		}
		if limit < len(marker) && sp != marker {
			t.Fatalf("spill limit %d below the marker: %q, want the bare marker", limit, sp)
		}
	}
}

func TestSpill_MarkerRoundTrip(t *testing.T) {
	for _, ref := range []string{"mem://x", "a, b", "[c]", "9 bytes]"} {
		got, n, ok := parseSpillMarker(spillMarker(ref, 123) + "\npreview\n[x, 5 bytes]")
		if !ok || got != ref || n != 123 {
			t.Errorf("ref %q parsed as %q, %d, %v", ref, got, n, ok)
		}
	}
	for _, s := range []string{"", "plain", "[tool output spilled: nonsense]", "[tool output spilled: r, x bytes]", "[tool output truncated: 1 of 2 bytes shown]"} {
		if _, _, ok := parseSpillMarker(s); ok {
			t.Errorf("%q parsed as a spill marker", s)
		}
	}
}

func TestSpill_ParallelCallsSpillEachOnce(t *testing.T) {
	big := spillBig(40 << 10)
	sink := &recSink{}
	fc := newFakeClient(toolTurn(mkCall("c1", "big", `{}`), mkCall("c2", "big", `{}`)), textResp("done", 1, 1))
	_, err := NewRunner(fc, []Tool{blocksTool{"big", []llmkit.Block{llmkit.Text(big)}}}, "sys",
		WithParallelTools(), WithToolOutputLimit(4<<10, sink)).Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]int{}
	for _, c := range sink.calls {
		ids[c.call.ID]++
	}
	if !reflect.DeepEqual(ids, map[string]int{"c1": 1, "c2": 1}) {
		t.Errorf("sink calls by call ID = %v, want one each", ids)
	}
}

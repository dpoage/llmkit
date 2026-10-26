package agent

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/dpoage/llmkit"
)

// SpillSink stores a tool result too large for the conversation and returns a
// reference the model is shown instead. full holds the result's blocks after
// media degradation, not the tool's raw output, and is read-only. Under
// [WithParallelTools] Spill is called concurrently. A non-empty valid-UTF-8
// ref with no '\n' or '\r' is required; anything else counts as a sink failure. A panic in
// Spill is not recovered: it unwinds out of Run like a hook panic.
type SpillSink interface {
	Spill(ctx context.Context, call llmkit.ToolCall, full []llmkit.Block) (ref string, err error)
}

// WithToolOutputLimit bounds each successful tool result entering history to
// maxBytes: the sum of its blocks' Text, Data and URL lengths after media
// degradation. A larger result goes whole to sink and the model sees
// "[tool output spilled: <ref>, <size> bytes]" and, when the head preview of
// the text is not empty, a newline and that preview. With a nil sink, or when
// the sink fails, the text is head-truncated and ends with
// "[tool output truncated: <shown> of <size> bytes shown]"; media blocks are
// dropped and counted in size. Either stand-in fits in maxBytes, except that a
// limit smaller than its marker still sends the marker alone. A sink failure
// is recorded as [llmkit.ToolRunEvent].SpillError. maxBytes <= 0 disables the
// limit, the default. Errors, denials and timeouts are not limited, and a
// [ReplayClient] serves recorded results unchanged.
func WithToolOutputLimit(maxBytes int, sink SpillSink) Option {
	return func(r *Runner) { r.outputLimit = outputLimit{max: maxBytes, sink: sink} }
}

type outputLimit struct {
	max  int
	sink SpillSink
}

// spillPreviewBytes is the head preview length sent with a spill marker.
const spillPreviewBytes = 2048

const (
	spillMarkerPrefix = "[tool output spilled: "
	spillMarkerSuffix = " bytes]"
)

func spillMarker(ref string, size int) string {
	return spillMarkerPrefix + ref + ", " + strconv.Itoa(size) + spillMarkerSuffix
}

// parseSpillMarker reads the ref and original size from the first line of a
// model-visible result produced by a spill.
func parseSpillMarker(text string) (ref string, size int, ok bool) {
	line, _, _ := strings.Cut(text, "\n")
	rest, ok := strings.CutPrefix(line, spillMarkerPrefix)
	if !ok {
		return "", 0, false
	}
	rest, ok = strings.CutSuffix(rest, spillMarkerSuffix)
	if !ok {
		return "", 0, false
	}
	i := strings.LastIndex(rest, ", ")
	if i < 0 {
		return "", 0, false
	}
	n, err := strconv.Atoi(rest[i+2:])
	if err != nil {
		return "", 0, false
	}
	return rest[:i], n, true
}

func truncationMarker(shown, size int) string {
	return fmt.Sprintf("[tool output truncated: %d of %d bytes shown]", shown, size)
}

func blocksSize(blocks []llmkit.Block) int {
	n := 0
	for _, b := range blocks {
		n += len(b.Text) + len(b.Data) + len(b.URL)
	}
	return n
}

func blocksPlainText(blocks []llmkit.Block) string {
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Kind == llmkit.BlockText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// headBytes returns the longest prefix of s that is at most n bytes and ends
// on a rune boundary.
func headBytes(s string, n int) string {
	if n >= len(s) {
		return s
	}
	if n <= 0 {
		return ""
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

func spillStandIn(ref string, size int, text string, limit int) string {
	marker := spillMarker(ref, size)
	room := min(spillPreviewBytes, limit-len(marker)-1)
	preview := headBytes(text, room)
	if preview == "" {
		return marker
	}
	return marker + "\n" + preview
}

// truncatedStandIn is the head of text with the truncation marker appended,
// the whole at most limit bytes unless the marker alone is longer.
func truncatedStandIn(text string, size, limit int) string {
	fixed := len(truncationMarker(0, size)) - 1
	// The marker names shown, so try each digit count d of shown, fewest
	// first (most room), and take the first head whose length has at most d digits.
	for d := 1; ; d++ {
		head := headBytes(text, min(len(text), limit-fixed-d))
		if len(strconv.Itoa(len(head))) <= d {
			return head + truncationMarker(len(head), size)
		}
	}
}

// validSpillRef rejects a ref a JSON record cannot carry byte-for-byte.
func validSpillRef(ref string) bool {
	return ref != "" && !strings.ContainsAny(ref, "\r\n") && utf8.ValidString(ref)
}

// limitToolOutput applies the Runner's output limit to a placed result.
func (r *Runner) limitToolOutput(ctx context.Context, call llmkit.ToolCall, res toolResult) toolResult {
	lim := r.outputLimit
	if lim.max <= 0 {
		return res
	}
	blocks := res.content
	size := blocksSize(blocks)
	if size <= lim.max {
		return res
	}
	text := blocksPlainText(blocks)
	res.originalBytes = size
	if lim.sink != nil {
		ref, err := lim.sink.Spill(ctx, call, blocks)
		if err == nil && !validSpillRef(ref) {
			err = fmt.Errorf("sink returned an unusable ref %q", ref)
		}
		if err == nil {
			res.spillRef = ref
			res.setText(spillStandIn(ref, size, text, lim.max))
			return res
		}
		res.spillErr = err.Error()
	}
	res.setText(truncatedStandIn(text, size, lim.max))
	return res
}

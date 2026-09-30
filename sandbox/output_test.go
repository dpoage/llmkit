package sandbox

import (
	"context"
	"fmt"
	"math"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestCappedBufferTruncates(t *testing.T) {
	b := newCappedBuffer(10)
	n, err := b.Write([]byte("0123456789ABCDEF")) // 16 bytes into a 10-byte cap
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if n != 16 {
		t.Errorf("Write should report full consumption (16) to keep the pipe draining, got %d", n)
	}

	got, truncated := b.result()
	if !truncated {
		t.Fatal("expected truncated=true")
	}
	if !strings.HasPrefix(got, "0123456789") {
		t.Errorf("retained prefix wrong: %q", got)
	}
	if !strings.Contains(got, "truncated") {
		t.Errorf("expected truncation marker in output, got %q", got)
	}
}

func TestCappedBufferUnderCap(t *testing.T) {
	b := newCappedBuffer(100)
	_, _ = b.Write([]byte("hello"))
	got, truncated := b.result()
	if truncated {
		t.Error("did not expect truncation")
	}
	if got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

func TestCappedBufferMultipleWritesCrossCap(t *testing.T) {
	b := newCappedBuffer(8)
	_, _ = b.Write([]byte("abcd"))
	_, _ = b.Write([]byte("efgh"))
	_, _ = b.Write([]byte("ijkl")) // pushes over the cap
	got, truncated := b.result()
	if !truncated {
		t.Fatal("expected truncation after crossing cap")
	}
	if !strings.HasPrefix(got, "abcdefgh") {
		t.Errorf("retained prefix wrong: %q", got)
	}
}

func TestCappedBufferZeroCapRetainsAll(t *testing.T) {
	b := newCappedBuffer(0)
	_, _ = b.Write([]byte("anything goes"))
	got, truncated := b.result()
	if truncated {
		t.Error("zero cap should not truncate")
	}
	if got != "anything goes" {
		t.Errorf("got %q", got)
	}
}

// TestCappedBufferRetainsTail_HeadTailWindow verifies dual-window retention:
// when the stream overflows the ring, bytes from BOTH the head and the tail
// are retained. Test runners print failure summaries at the END of output, so
// a pure head-only buffer would lose them.
func TestCappedBufferRetainsTail_HeadTailWindow(t *testing.T) {
	// cap = headBytes+32 (32-byte ring). Write headBytes + 64 bytes.
	// Ring receives 64 bytes but only holds 32 → overwrites first 32.
	// Last 32 bytes (tailLast) must survive; first write (tailFirst) is elided.
	ringSize := 32
	cap := headBytes + ringSize
	b := newCappedBuffer(cap)

	headContent := make([]byte, headBytes)
	for i := range headContent {
		headContent[i] = 'A'
	}
	tailFirst := make([]byte, 32)
	for i := range tailFirst {
		tailFirst[i] = 'B'
	}
	tailLast := []byte("--- FAIL: TestBug 00000000000000\n")
	tailLast = tailLast[:32]

	_, _ = b.Write(headContent)
	_, _ = b.Write(tailFirst) // fills ring; ringFull set; truncated=true
	_, _ = b.Write(tailLast)  // overwrites ring with FAIL marker

	got, truncated := b.result()
	if !truncated {
		t.Fatal("expected truncated=true: ring overwrote earlier tail bytes")
	}
	if got[0] != 'A' {
		t.Errorf("head content missing: first byte = %q", got[0])
	}
	if !strings.Contains(got, "--- FAIL") {
		t.Errorf("tail FAIL marker missing from result; got suffix=%q", got[maxInt(0, len(got)-60):])
	}
	if !strings.Contains(got, "elided") {
		t.Errorf("expected elision gap marker in result; got=%q", got[:minInt(len(got), 300)])
	}
}

// TestCappedBufferHeadTailGapMarker verifies the gap marker appears and the tail
// survives when ring overflows.
func TestCappedBufferHeadTailGapMarker(t *testing.T) {
	// Ring = 8 bytes. Write headBytes + 100 bytes. Ring keeps last 8 = "TAILDATA".
	cap := headBytes + 8
	b := newCappedBuffer(cap)

	headContent := make([]byte, headBytes)
	for i := range headContent {
		headContent[i] = 'H'
	}
	overflow := make([]byte, 100)
	for i := range overflow {
		overflow[i] = 'M'
	}
	copy(overflow[92:], "TAILDATA") // last 8 bytes

	_, _ = b.Write(headContent)
	_, _ = b.Write(overflow)

	got, truncated := b.result()
	if !truncated {
		t.Fatal("expected truncated=true when ring overflows")
	}
	if !strings.Contains(got, "elided") {
		t.Errorf("expected elision gap marker; got=%q", got[:minInt(len(got), 300)])
	}
	if !strings.HasSuffix(got, "TAILDATA") {
		t.Errorf("tail content missing; got suffix=%q", got[maxInt(0, len(got)-20):])
	}
	if got[0] != 'H' {
		t.Errorf("head content missing")
	}
}

// TestCappedBufferFailSummaryAtTail verifies the acceptance criterion:
// a stream exceeding the cap whose final lines contain '--- FAIL' still exposes
// that text after capping so interpret() can classify demonstrated.
func TestCappedBufferFailSummaryAtTail(t *testing.T) {
	// Ring = 1024. Write headBytes + 2048 filler + failSummary.
	// Filler overflows ring; failSummary survives in the last ring bytes.
	ringSize := 1024
	cap := headBytes + ringSize
	b := newCappedBuffer(cap)

	headContent := make([]byte, headBytes)
	for i := range headContent {
		headContent[i] = 'X'
	}
	filler := make([]byte, 2048)
	for i := range filler {
		filler[i] = 'Y'
	}
	failSummary := "--- FAIL: TestRaceCondition (0.123s)\nFAIL\tgithub.com/example/pkg\n"

	_, _ = b.Write(headContent)
	_, _ = b.Write(filler)
	_, _ = b.Write([]byte(failSummary))

	got, truncated := b.result()
	if !truncated {
		t.Fatal("expected truncated=true: ring overflowed")
	}
	if !strings.Contains(got, "--- FAIL") {
		t.Errorf("failure summary lost after capping; got tail=%q", got[maxInt(0, len(got)-200):])
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TestCappedBufferPartialRing_TailRetained is the regression guard for the
// data-loss bug: when a stream overflows the head window but its tail fits in
// the ring WITHOUT wrapping (total in (head, max]), result() must still return
// the ring/tail content. The prior implementation returned head-only on
// !truncated, silently dropping the tail — reintroducing the very
// false-negative a head-only cap has (a trailing "--- FAIL" past the
// 256KB head was lost for the whole 256KB..1MiB band under the 1MiB cap).
func TestCappedBufferPartialRing_TailRetained(t *testing.T) {
	ringSize := 4096
	b := newCappedBuffer(headBytes + ringSize) // 256KB head + 4KB ring

	headContent := make([]byte, headBytes)
	for i := range headContent {
		headContent[i] = 'H'
	}
	// Tail is smaller than the ring, so the ring never wraps: truncated stays
	// false, yet these bytes MUST survive.
	tail := "leading tail bytes ...\n--- FAIL: TestLate (0.10s)\nFAIL\tgithub.com/example/pkg\n"

	_, _ = b.Write(headContent)
	_, _ = b.Write([]byte(tail))

	got, truncated := b.result()
	if truncated {
		t.Error("no bytes were lost (tail fit the ring without wrapping); truncated must be false")
	}
	if got[0] != 'H' {
		t.Errorf("head content missing: first byte = %q", got[0])
	}
	if !strings.HasSuffix(got, tail) {
		t.Errorf("tail content dropped by result(); got suffix=%q", got[maxInt(0, len(got)-80):])
	}
	if !strings.Contains(got, "--- FAIL") {
		t.Error("trailing FAIL marker lost after capping — interpret() would misclassify not_demonstrated")
	}
	// No elision marker: head and tail are contiguous, nothing was dropped.
	if strings.Contains(got, "elided") {
		t.Errorf("no bytes were elided, but result() emitted an elision marker; got=%q", got[maxInt(0, len(got)-120):])
	}
	// Full fidelity: exactly head + tail, byte for byte.
	if len(got) != headBytes+len(tail) {
		t.Errorf("result length = %d, want %d (head+tail, no loss)", len(got), headBytes+len(tail))
	}
}

// TestCappedBufferRingGrowsWithBytesWritten pins that the tail ring's
// allocation tracks the bytes written, not the configured cap: after 1000
// bytes land in the tail of a 1 GiB-cap buffer the ring holds about that
// many, never the cap (llmkit-bk8.1.20).
func TestCappedBufferRingGrowsWithBytesWritten(t *testing.T) {
	c := newCappedBuffer(1 << 30)
	if _, err := c.Write(make([]byte, headBytes+1000)); err != nil {
		t.Fatal(err)
	}
	if len(c.ring) != 1000 {
		t.Fatalf("len(ring) = %d, want 1000 tail bytes", len(c.ring))
	}
	if cap(c.ring) > 2*len(c.ring) {
		t.Errorf("cap(ring) = %d for %d live bytes; capture memory must be proportional to bytes written, not to the 1 GiB cap", cap(c.ring), len(c.ring))
	}
}

// TestCappedBufferChunkingIsInvisible pins that the lazily grown ring
// retains the same head and tail for any write chunking as the stream it
// models: first headBytes, then the last tailSize bytes, with the elided
// count between, across streams that end before the ring fills, exactly as
// it fills, and long after it wraps.
func TestCappedBufferChunkingIsInvisible(t *testing.T) {
	const tailSize = 1000
	max := headBytes + tailSize
	for _, total := range []int{
		headBytes - 1, headBytes, headBytes + 1, headBytes + tailSize - 1,
		headBytes + tailSize, headBytes + tailSize + 1, headBytes + 2*tailSize + 37, headBytes + 5*tailSize,
	} {
		stream := make([]byte, total)
		for i := range stream {
			stream[i] = byte(i*31 + i>>8)
		}
		var want string
		var wantTrunc bool
		switch {
		case total <= headBytes:
			want = string(stream)
		case total == headBytes+tailSize:
			// The ring filling exactly reports truncated (the pre-lazy behavior).
			want, wantTrunc = string(stream), true
		case total < headBytes+tailSize:
			want = string(stream)
		default:
			elided := total - headBytes - tailSize
			want = string(stream[:headBytes]) + fmt.Sprintf("\n... [%d bytes elided by sandbox] ...\n", elided) + string(stream[total-tailSize:])
			wantTrunc = true
		}
		for _, chunk := range []int{1, 7, 999, 1000, 1001, 4096, headBytes + 3, total} {
			c := newCappedBuffer(max)
			for off := 0; off < total; off += chunk {
				end := off + chunk
				if end > total {
					end = total
				}
				if _, err := c.Write(stream[off:end]); err != nil {
					t.Fatal(err)
				}
			}
			got, trunc := c.result()
			if got != want || trunc != wantTrunc {
				t.Errorf("total=%d chunk=%d: result differs from the stream model (len got=%d want=%d, truncated got=%v want=%v)", total, chunk, len(got), len(want), trunc, wantTrunc)
			}
		}
	}
}

// TestExecCaptureAllocationIsIndependentOfCap pins bead llmkit-bk8.1.20:
// one supervised run that writes 10 bytes under a 1 GiB per-stream cap
// allocates a few MiB at most, and a math.MaxInt cap neither panics nor
// changes the captured bytes. TotalAlloc is the measure: an allocation
// counter sees the eager make([]byte, cap) that AllocsPerRun cannot.
func TestExecCaptureAllocationIsIndependentOfCap(t *testing.T) {
	run := func(maxOutput int) Result {
		res, err := runSupervised(context.Background(), runSpec{
			spec:           Spec{Workspace: t.TempDir(), Cmd: []string{"sh", "-c", "printf 0123456789; printf abcdefghij >&2"}},
			timeout:        30 * time.Second,
			maxOutputBytes: maxOutput,
			hooks: runHooks{
				buildCmd: func(_ string, runCtx context.Context) (*exec.Cmd, error) {
					return exec.CommandContext(runCtx, "sh", "-c", "printf 0123456789; printf abcdefghij >&2"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("runSupervised(maxOutput=%d): %v", maxOutput, err)
		}
		return res
	}

	run(1 << 30) // warm one-time allocations out of the measured window
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	res := run(1 << 30)
	runtime.ReadMemStats(&after)
	if grew := after.TotalAlloc - before.TotalAlloc; grew >= 16<<20 {
		t.Errorf("TotalAlloc grew %d bytes for a 10-byte run under WithMaxOutputBytes(1<<30); want < 16 MiB", grew)
	}
	if res.Stdout != "0123456789" || res.Stderr != "abcdefghij" {
		t.Errorf("captured (%q, %q), want the 10 bytes written to each stream", res.Stdout, res.Stderr)
	}

	huge := run(math.MaxInt)
	if huge.Stdout != "0123456789" || huge.Stderr != "abcdefghij" || huge.StdoutTruncated || huge.StderrTruncated {
		t.Errorf("MaxInt cap captured (%q, %q) truncated=%v/%v, want the 10 bytes untruncated", huge.Stdout, huge.Stderr, huge.StdoutTruncated, huge.StderrTruncated)
	}
}

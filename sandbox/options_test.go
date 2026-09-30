package sandbox

import (
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// mbInt converts an int64 MB count to int at run time, so rows naming a
// count above 2^31-1 compile on 32-bit targets; every such row runs only
// when strconv.IntSize is 64.
func mbInt(v int64) int { return int(v) }

// assertNumericRefusal fails unless err is non-nil and names both optName
// and backend, matching checkSupported's error shape.
func assertNumericRefusal(t *testing.T, err error, optName, backend string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected %s to be refused on the %s backend, got nil error", optName, backend)
	}
	if !strings.Contains(err.Error(), optName) || !strings.Contains(err.Error(), backend) {
		t.Errorf("error %q must name %s and the %s backend", err, optName, backend)
	}
}

// TestNumericOptionsRefused pins that WithCPUs, WithMemoryMB, WithTimeout,
// WithScratchSizeMB, and WithMaxOutputBytes refuse a value that names no
// usable limit: <= 0 for all five, NaN or ±Inf for WithCPUs (neither
// backend renders a working CPU cap from one), a WithCPUs outside
// [0.01, 214748.3647], and a WithMemoryMB/WithScratchSizeMB above the
// largest count whose byte conversion fits an int64. NewBwrap/NewCLI are called
// DIRECTLY and refuse before any runtime lookup (DetectBwrap/Detect never
// run for a refused row), so the check is hermetic on every host.
func TestNumericOptionsRefused(t *testing.T) {
	cases := []struct {
		name  string
		value string // for the subtest name
		bwrap Option
		cli   Option
	}{
		{"WithCPUs", "0", WithCPUs(0), WithCPUs(0)},
		{"WithCPUs", "-1", WithCPUs(-1), WithCPUs(-1)},
		{"WithCPUs", "NaN", WithCPUs(math.NaN()), WithCPUs(math.NaN())},
		{"WithCPUs", "+Inf", WithCPUs(math.Inf(1)), WithCPUs(math.Inf(1))},
		{"WithCPUs", "-Inf", WithCPUs(math.Inf(-1)), WithCPUs(math.Inf(-1))},
		{"WithMemoryMB", "0", WithMemoryMB(0), WithMemoryMB(0)},
		{"WithMemoryMB", "-1", WithMemoryMB(-1), WithMemoryMB(-1)},
		{"WithTimeout", "0", WithTimeout(0), WithTimeout(0)},
		{"WithTimeout", "-1", WithTimeout(-time.Second), WithTimeout(-time.Second)},
		{"WithScratchSizeMB", "0", WithScratchSizeMB(0), WithScratchSizeMB(0)},
		{"WithScratchSizeMB", "-1", WithScratchSizeMB(-1), WithScratchSizeMB(-1)},
		{"WithMaxOutputBytes", "0", WithMaxOutputBytes(0), WithMaxOutputBytes(0)},
		{"WithMaxOutputBytes", "-1", WithMaxOutputBytes(-1), WithMaxOutputBytes(-1)},
		// llmkit-bk8.1.45: a positive value below the 0.01 floor
		// yields no cap (1e-6) or a runtime refusal (0.001), and one above
		// the systemd ceiling fails at run time on the systemd-run path.
		{"WithCPUs", "1e-6", WithCPUs(1e-6), WithCPUs(1e-6)},
		{"WithCPUs", "0.001", WithCPUs(0.001), WithCPUs(0.001)},
		{"WithCPUs", "0.0099", WithCPUs(0.0099), WithCPUs(0.0099)},
		{"WithCPUs", "1e6", WithCPUs(1e6), WithCPUs(1e6)},
		{"WithCPUs", "just above ceiling", WithCPUs(math.Nextafter(maxCPUs, math.Inf(1))), WithCPUs(math.Nextafter(maxCPUs, math.Inf(1)))},
	}
	if strconv.IntSize == 64 {
		// llmkit-bk8.1.48: byte-converting MB options refuse what
		// overflows the int64 byte conversion.
		for _, name := range []string{"WithMemoryMB", "WithScratchSizeMB"} {
			mk := map[string]func(int) Option{"WithMemoryMB": WithMemoryMB, "WithScratchSizeMB": WithScratchSizeMB}[name]
			for _, v := range []int{math.MaxInt, mbInt(1 << 44), mbInt(maxMB + 1)} {
				cases = append(cases, struct {
					name  string
					value string
					bwrap Option
					cli   Option
				}{name, strconv.Itoa(v), mk(v), mk(v)})
			}
		}
	}
	for _, c := range cases {
		c := c
		t.Run(c.name+"("+c.value+")/bwrap", func(t *testing.T) {
			_, err := NewBwrap(c.bwrap)
			assertNumericRefusal(t, err, c.name, "bwrap")
		})
		t.Run(c.name+"("+c.value+")/cli", func(t *testing.T) {
			_, err := NewCLI(WithImage("img"), c.cli)
			assertNumericRefusal(t, err, c.name, "cli")
		})
	}
}

// TestExplicitDisableOptionsRefuseOnlyNegative pins the three options that
// keep 0 as their documented explicit-disable value (WithPidsLimit,
// WithWorkspaceGrowthCeilingMB, WithIdleTimeout): only a negative value is
// refused, never 0.
func TestExplicitDisableOptionsRefuseOnlyNegative(t *testing.T) {
	negCases := []struct {
		name  string
		bwrap Option
		cli   Option
	}{
		{"WithPidsLimit", WithPidsLimit(-1), WithPidsLimit(-1)},
		{"WithWorkspaceGrowthCeilingMB", WithWorkspaceGrowthCeilingMB(-1), WithWorkspaceGrowthCeilingMB(-1)},
		{"WithIdleTimeout", WithIdleTimeout(-time.Second), WithIdleTimeout(-time.Second)},
	}
	for _, c := range negCases {
		c := c
		t.Run(c.name+"(-1)/bwrap", func(t *testing.T) {
			_, err := NewBwrap(c.bwrap)
			assertNumericRefusal(t, err, c.name, "bwrap")
		})
		t.Run(c.name+"(-1)/cli", func(t *testing.T) {
			_, err := NewCLI(WithImage("img"), c.cli)
			assertNumericRefusal(t, err, c.name, "cli")
		})
	}

	// 0 is accepted (the documented explicit disable) and the field
	// actually lands at 0 on the real backend. These need a working
	// backend to construct, so they skip when one is unavailable.
	t.Run("WithPidsLimit(0)/bwrap", func(t *testing.T) {
		if ok, _ := DetectBwrap(); !ok {
			t.Skip("bwrap unavailable")
		}
		bw, err := NewBwrap(WithPidsLimit(0))
		if err != nil {
			t.Fatalf("NewBwrap(WithPidsLimit(0)): %v", err)
		}
		defer func() { _ = bw.Close() }()
		if bw.pidsLimit != 0 {
			t.Errorf("pidsLimit = %d, want 0 (explicit disable)", bw.pidsLimit)
		}
	})
	t.Run("WithPidsLimit(0)/cli", func(t *testing.T) {
		rt, ok := Detect()
		if !ok {
			t.Skip("no container runtime detected")
		}
		s, err := NewCLI(WithRuntime(rt), WithImage("img"), WithPidsLimit(0))
		if err != nil {
			t.Fatalf("NewCLI(WithPidsLimit(0)): %v", err)
		}
		if s.pidsLimit != 0 {
			t.Errorf("pidsLimit = %d, want 0 (explicit disable)", s.pidsLimit)
		}
	})
	t.Run("WithWorkspaceGrowthCeilingMB(0)/bwrap", func(t *testing.T) {
		if ok, _ := DetectBwrap(); !ok {
			t.Skip("bwrap unavailable")
		}
		bw, err := NewBwrap(WithWorkspaceGrowthCeilingMB(0))
		if err != nil {
			t.Fatalf("NewBwrap(WithWorkspaceGrowthCeilingMB(0)): %v", err)
		}
		defer func() { _ = bw.Close() }()
		if bw.defaultGrowthCeilingBytes != 0 {
			t.Errorf("defaultGrowthCeilingBytes = %d, want 0 (explicit disable)", bw.defaultGrowthCeilingBytes)
		}
	})
	t.Run("WithWorkspaceGrowthCeilingMB(0)/cli", func(t *testing.T) {
		rt, ok := Detect()
		if !ok {
			t.Skip("no container runtime detected")
		}
		s, err := NewCLI(WithRuntime(rt), WithImage("img"), WithWorkspaceGrowthCeilingMB(0))
		if err != nil {
			t.Fatalf("NewCLI(WithWorkspaceGrowthCeilingMB(0)): %v", err)
		}
		if s.defaultGrowthCeilingBytes != 0 {
			t.Errorf("defaultGrowthCeilingBytes = %d, want 0 (explicit disable)", s.defaultGrowthCeilingBytes)
		}
	})
	t.Run("WithIdleTimeout(0)/bwrap", func(t *testing.T) {
		if ok, _ := DetectBwrap(); !ok {
			t.Skip("bwrap unavailable")
		}
		bw, err := NewBwrap(WithIdleTimeout(0))
		if err != nil {
			t.Fatalf("NewBwrap(WithIdleTimeout(0)): %v", err)
		}
		defer func() { _ = bw.Close() }()
		if bw.defaultIdleTimeout != 0 {
			t.Errorf("defaultIdleTimeout = %v, want 0 (explicit disable)", bw.defaultIdleTimeout)
		}
	})
	t.Run("WithIdleTimeout(0)/cli", func(t *testing.T) {
		rt, ok := Detect()
		if !ok {
			t.Skip("no container runtime detected")
		}
		s, err := NewCLI(WithRuntime(rt), WithImage("img"), WithIdleTimeout(0))
		if err != nil {
			t.Fatalf("NewCLI(WithIdleTimeout(0)): %v", err)
		}
		if s.defaultIdleTimeout != 0 {
			t.Errorf("defaultIdleTimeout = %v, want 0 (explicit disable)", s.defaultIdleTimeout)
		}
	})
}

// TestNumericOptionBoundsAdmit pins the admitted side of the bounds: the
// floor and ceiling themselves, the largest byte-convertible MB count for
// every byte-converting option, and WithMaxOutputBytes(math.MaxInt), which
// lazy capture makes safe. checkNumericOptions is called directly:
// it is the single refusal point both constructors run first.
func TestNumericOptionBoundsAdmit(t *testing.T) {
	ok := []struct {
		name string
		opt  Option
	}{
		{"WithCPUs(floor)", WithCPUs(minCPUs)},
		{"WithCPUs(ceiling)", WithCPUs(maxCPUs)},
		{"WithMemoryMB(max)", WithMemoryMB(mbInt(maxMB))},
		{"WithScratchSizeMB(max)", WithScratchSizeMB(mbInt(maxMB))},
		{"WithWorkspaceGrowthCeilingMB(max)", WithWorkspaceGrowthCeilingMB(mbInt(maxMB))},
		{"WithMaxOutputBytes(MaxInt)", WithMaxOutputBytes(math.MaxInt)},
	}
	for _, c := range ok {
		if strconv.IntSize < 64 && strings.HasSuffix(c.name, "(max)") {
			continue
		}
		for _, backend := range []string{"bwrap", "cli"} {
			o := newOptions([]Option{c.opt})
			if err := o.checkNumericOptions(backend); err != nil {
				t.Errorf("%s on %s: %v, want admitted", c.name, backend, err)
			}
		}
	}
}

// p63FractionalCPUs are fractional WithCPUs rows (llmkit-bk8.1.45) that
// carry float noise or more than two decimals of percent when rendered as
// a CPUQuota.
var p63FractionalCPUs = []float64{0.07, 0.29, 0.333, 1.1, 2.3}

// TestFractionalCPUsAdmitted pins the admission half of llmkit-bk8.1.45 on
// every host, with no skip: the fractional rows, the floor, and the ceiling pass
// checkNumericOptions and both constructors. NewCLI gets the test binary as
// its runtime, so it constructs wherever the test runs. NewBwrap must return
// exactly what NewBwrap() without WithCPUs returns on this host (nil where
// bwrap works, the same availability error where it does not), so a refusal
// that depends on the value fails here.
func TestFractionalCPUsAdmitted(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	outcome := func(err error) string {
		if err == nil {
			return "<nil>"
		}
		return err.Error()
	}
	baseline, baseErr := NewBwrap()
	_ = baseline.Close()
	want := outcome(baseErr)
	for _, cpus := range append([]float64{minCPUs, maxCPUs}, p63FractionalCPUs...) {
		t.Run(strconv.FormatFloat(cpus, 'f', -1, 64), func(t *testing.T) {
			for _, backend := range []string{"bwrap", "cli"} {
				o := newOptions([]Option{WithCPUs(cpus)})
				if err := o.checkNumericOptions(backend); err != nil {
					t.Errorf("checkNumericOptions(%s): %v, want admitted", backend, err)
				}
			}
			cli, err := NewCLI(WithRuntime(exe), WithImage("img"), WithCPUs(cpus))
			if err != nil {
				t.Errorf("NewCLI(WithCPUs(%v)): %v, want admitted", cpus, err)
			} else {
				if cli.defaultCPUs != cpus {
					t.Errorf("NewCLI(WithCPUs(%v)).defaultCPUs = %v", cpus, cli.defaultCPUs)
				}
				_ = cli.Close()
			}
			bw, err := NewBwrap(WithCPUs(cpus))
			if got := outcome(err); got != want {
				t.Errorf("NewBwrap(WithCPUs(%v)) error = %s, want %s (the NewBwrap() outcome on this host)", cpus, got, want)
			}
			if err == nil {
				if bw.defaultCPUs != cpus {
					t.Errorf("NewBwrap(WithCPUs(%v)).defaultCPUs = %v", cpus, bw.defaultCPUs)
				}
				_ = bw.Close()
			}
		})
	}
}

// TestGrowthCeilingOverflowRefused pins that WithWorkspaceGrowthCeilingMB
// refuses a value whose byte conversion overflows int64: it would wrap to a
// negative or tiny ceiling (MaxInt64 MB -> disabled).
func TestGrowthCeilingOverflowRefused(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("MB counts that overflow int64 bytes are unrepresentable in a 32-bit int")
	}
	for _, v := range []int{math.MaxInt, mbInt(1 << 44), mbInt(maxMB + 1)} {
		for _, backend := range []string{"bwrap", "cli"} {
			o := newOptions([]Option{WithWorkspaceGrowthCeilingMB(v)})
			assertNumericRefusal(t, o.checkNumericOptions(backend), "WithWorkspaceGrowthCeilingMB", backend)
		}
	}
}

// TestLargestAdmittedMBRendersPositiveBytes pins that the bound is exactly
// the point where the byte conversion stays positive: the largest admitted
// MB count renders a positive byte count through the real cgroup, argv,
// and default converters, and one more MB is refused.
func TestLargestAdmittedMBRendersPositiveBytes(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("the largest byte-convertible MB count is unrepresentable in a 32-bit int")
	}
	top := mbInt(maxMB)
	if b := int64(top) * 1024 * 1024; b <= 0 {
		t.Fatalf("largest admitted MB %d converts to %d bytes", top, b)
	}
	if b := (int64(top) + 1) * 1024 * 1024; b > 0 {
		t.Fatalf("maxMB is not the overflow boundary: %d MB still converts to %d bytes", top+1, b)
	}
	memory, _, _, _, _, _ := cgroupV2Limits(1, top, 0)
	if n, err := strconv.ParseInt(memory, 10, 64); err != nil || n <= 0 {
		t.Errorf("cgroupV2Limits memory = %q (err %v), want a positive int64 byte count", memory, err)
	}
	var d defaults
	o := newOptions([]Option{WithWorkspaceGrowthCeilingMB(top)})
	o.applyDefaults(&d)
	if d.defaultGrowthCeilingBytes <= 0 {
		t.Errorf("growth ceiling for %d MB = %d bytes, want positive", top, d.defaultGrowthCeilingBytes)
	}
}

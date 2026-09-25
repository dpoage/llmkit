package sandbox

import (
	"math"
	"strings"
	"testing"
	"time"
)

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
// usable limit: <= 0 for all five, and NaN or ±Inf for WithCPUs (neither
// backend renders a working CPU cap from one). NewBwrap/NewCLI are called
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

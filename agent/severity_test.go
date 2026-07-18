package agent

import "testing"

func TestParseSeverity(t *testing.T) {
	cases := []struct {
		in     string
		want   Severity
		wantOK bool
	}{
		{"critical", SeverityCritical, true},
		{"high", SeverityHigh, true},
		{"medium", SeverityMedium, true},
		{"low", SeverityLow, true},
		{"  HIGH  ", SeverityHigh, true}, // case-insensitive + trimmed
		{"Critical", SeverityCritical, true},
		{"", "", false},
		{"blocker", "", false},
		{"none", "", false},
	}
	for _, c := range cases {
		got, ok := ParseSeverity(c.in)
		if got != c.want || ok != c.wantOK {
			t.Errorf("ParseSeverity(%q) = (%q, %v), want (%q, %v)", c.in, got, ok, c.want, c.wantOK)
		}
	}
}

func TestSeverityRankAndAtLeast(t *testing.T) {
	// Higher rank = more severe; unknown sorts last.
	if SeverityCritical.Rank() <= SeverityHigh.Rank() ||
		SeverityHigh.Rank() <= SeverityMedium.Rank() ||
		SeverityMedium.Rank() <= SeverityLow.Rank() ||
		SeverityLow.Rank() <= Severity("").Rank() {
		t.Fatalf("severity rank ordering broken: crit=%d high=%d med=%d low=%d unknown=%d",
			SeverityCritical.Rank(), SeverityHigh.Rank(), SeverityMedium.Rank(),
			SeverityLow.Rank(), Severity("").Rank())
	}
	if !SeverityHigh.AtLeast(SeverityHigh) {
		t.Error("High should be AtLeast High (inclusive)")
	}
	if !SeverityCritical.AtLeast(SeverityLow) {
		t.Error("Critical should be AtLeast Low")
	}
	if SeverityLow.AtLeast(SeverityHigh) {
		t.Error("Low should not be AtLeast High")
	}
}

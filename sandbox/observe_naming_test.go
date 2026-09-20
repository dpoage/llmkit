package sandbox

import "testing"

// TestBackendNamePinned pins the llmkit.ExecEvent.Backend names for the
// package's own backends. cli, bwrap, and host must keep matching
// UnsupportedSpecError.Backend; mock is this package's own name for the
// backend that refuses nothing. Typed nils resolve the type switch
// without touching a receiver.
func TestBackendNamePinned(t *testing.T) {
	cases := []struct {
		s    Sandbox
		want string
	}{
		{(*CLI)(nil), "cli"},
		{(*Bwrap)(nil), "bwrap"},
		{(*HostExec)(nil), "host"},
		{(*Mock)(nil), "mock"},
		{&observedSandbox{inner: (*Mock)(nil)}, "mock"},
	}
	for _, tc := range cases {
		if got := backendName(tc.s); got != tc.want {
			t.Errorf("backendName(%T) = %q, want %q", tc.s, got, tc.want)
		}
	}
}

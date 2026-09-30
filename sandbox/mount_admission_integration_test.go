//go:build integration

package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// specMountMarker returns a host dir holding marker and the ROMount of it at
// /opt/x, a destination no backend reserves.
func specMountMarker(t *testing.T) ROMount {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "marker"), []byte("MOUNTED\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return ROMount{HostPath: dir, ContainerPath: "/opt/x", Shared: true}
}

// TestIntegrationSpecMountAtUnreservedPathRuns: reserving the backends' own
// destinations leaves an ordinary Spec mount admitted and visible on both
// container backends.
func TestIntegrationSpecMountAtUnreservedPathRuns(t *testing.T) {
	t.Run("cli", func(t *testing.T) {
		s := newTestCLI(t)
		res, err := s.Exec(context.Background(), Spec{
			RepoDir:  t.TempDir(),
			ROMounts: []ROMount{specMountMarker(t)},
			Cmd:      []string{"cat", "/opt/x/marker"},
		})
		if err != nil || res.ExitCode != 0 || strings.TrimSpace(res.Stdout) != "MOUNTED" {
			t.Fatalf("Exec = %+v, %v; want exit 0 printing MOUNTED", res, err)
		}
	})
	t.Run("bwrap", func(t *testing.T) {
		s := newTestBwrap(t)
		t.Cleanup(func() { _ = s.Close() })
		shMounts, sh := shForTest(t)
		res, err := s.Exec(context.Background(), Spec{
			RepoDir:  t.TempDir(),
			ROMounts: append(shMounts, specMountMarker(t)),
			Cmd:      []string{sh, "-c", "read line < /opt/x/marker && echo $line"},
		})
		if err != nil || res.ExitCode != 0 || strings.TrimSpace(res.Stdout) != "MOUNTED" {
			t.Fatalf("Exec = %+v, %v; want exit 0 printing MOUNTED", res, err)
		}
	})
}

// TestIntegrationBwrapRelativeToolchainEntryDoesNotExposeHome: with the
// working directory at $HOME, the relative toolchain entry "bin/bt" mounts
// <HOME>/bin, so bt is visible in the sandbox and $HOME/.ssh is not.
func TestIntegrationBwrapRelativeToolchainEntryDoesNotExposeHome(t *testing.T) {
	cwdHomeWithRelativeTools(t)
	res := ResolveHostToolchains([]string{"bin/bt"})
	s := newTestBwrap(t, WithHostToolchains(res))
	t.Cleanup(func() { _ = s.Close() })
	shMounts, sh := shForTest(t)
	root := hostToolchainMountRoot + "/bt"
	out, err := s.Exec(context.Background(), Spec{
		RepoDir:  t.TempDir(),
		ROMounts: shMounts,
		Cmd: []string{sh, "-c", "if read line < " + root + "/.ssh/id_secret; then echo LEAK:$line; fi; " +
			"if [ -f " + root + "/bt ]; then echo BT; fi"},
	})
	if err != nil || out.ExitCode != 0 || strings.Contains(out.Stdout, "SECRET") || strings.TrimSpace(out.Stdout) != "BT" {
		t.Fatalf("Exec = %+v, %v; want exit 0 printing only BT (bt mounted, $HOME/.ssh not)", out, err)
	}
}

// TestIntegrationBwrapRelativeToolchainEntryRunsKernelPath: a relative
// toolchain entry puts on the sandbox's PATH the file the host kernel would
// have executed for it from the working directory (see
// TestResolveHostToolchains_RelativeEntryIsKernelPath for the layouts).
func TestIntegrationBwrapRelativeToolchainEntryRunsKernelPath(t *testing.T) {
	r := symlinkedCwdLayout(t)
	for _, tc := range []struct{ name, cwd, entry, tool, want string }{
		{"cwd entered through a symlink", filepath.Join(r, "logical", "proj"), "../tools/bin/tt", "tt", "tt-PHYS"},
		{"dotdot after a symlinked component", filepath.Join(r, "c"), "link/../bin/ft", "ft", "ft-FAR"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Chdir(tc.cwd) // also sets PWD to it
			s := newTestBwrap(t, WithHostToolchains(ResolveHostToolchains([]string{tc.entry})))
			t.Cleanup(func() { _ = s.Close() })
			shMounts, sh := shForTest(t)
			out, err := s.Exec(context.Background(), Spec{
				RepoDir:  t.TempDir(),
				ROMounts: shMounts,
				Cmd:      []string{sh, "-c", sh + ` "$(command -v ` + tc.tool + `)"`},
			})
			if err != nil || out.ExitCode != 0 || strings.TrimSpace(out.Stdout) != tc.want {
				t.Fatalf("%s from %s: Exec = %+v, %v; want exit 0 printing %s", tc.entry, tc.cwd, out, err, tc.want)
			}
		})
	}
}

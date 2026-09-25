package sandbox

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

// fakeToolchainHost builds a synthetic toolchain layout under a temp dir:
//
//	<root>/versions/1.2.3/bin/<name>   (the "real" executable, a tiny shell script)
//	<root>/shim/<name>                 -> symlink to the versioned bin (nvm/asdf shim pattern)
//
// and returns (root, shimDir) so a test can point PATH at shimDir to exercise
// the symlink-closure resolution in resolveToolchainRoot.
func fakeToolchainHost(t *testing.T, name string) (root, shimDir string) {
	t.Helper()
	root = t.TempDir()
	versionedBin := filepath.Join(root, "versions", "1.2.3", "bin")
	if err := os.MkdirAll(versionedBin, 0o755); err != nil {
		t.Fatal(err)
	}
	realExe := filepath.Join(versionedBin, name)
	script := "#!/bin/sh\necho fake-" + name + " version 1.2.3\n"
	if err := os.WriteFile(realExe, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	shimDir = filepath.Join(root, "shim")
	if err := os.MkdirAll(shimDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(realExe, filepath.Join(shimDir, name)); err != nil {
		t.Fatal(err)
	}
	return root, shimDir
}

// TestResolveHostToolchains_RefusesHomeBinAscent guards against over-mounting
// $HOME when a toolchain lives directly at $HOME/bin/<name> (a common manual
// install layout): ascending past bin/ here would RO-mount the ENTIRE home
// directory (SSH keys, git credentials, unrelated dotfiles) into the sandbox.
// The mount must stay narrowed to the bin/ directory itself.
func TestResolveHostToolchains_RefusesHomeBinAscent(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("HOME/bin layout assumed POSIX")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	binDir := filepath.Join(home, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(binDir, "fakenode")
	if err := os.WriteFile(exe, []byte("#!/bin/sh\necho fake-node version 9.9.9\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)

	res := ResolveHostToolchains([]string{"fakenode"})
	if len(res.mounts) != 1 {
		t.Fatalf("want 1 mount, got %d: %+v", len(res.mounts), res.mounts)
	}
	if res.mounts[0].HostPath != binDir {
		t.Errorf("mount HostPath = %q, want the narrow bin dir %q (must NOT ascend to $HOME %q)",
			res.mounts[0].HostPath, binDir, home)
	}
}

// TestResolveHostToolchains_RefusesLocalBinAscent covers the ~/.local/bin/<name>
// layout the same way.
func TestResolveHostToolchains_RefusesLocalBinAscent(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("~/.local/bin layout assumed POSIX")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	localDir := filepath.Join(home, ".local")
	binDir := filepath.Join(localDir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(binDir, "fakenode")
	if err := os.WriteFile(exe, []byte("#!/bin/sh\necho fake-node version 9.9.9\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)

	res := ResolveHostToolchains([]string{"fakenode"})
	if len(res.mounts) != 1 || res.mounts[0].HostPath != binDir {
		t.Fatalf("mount HostPath = %+v, want the narrow bin dir %q (must NOT ascend to ~/.local %q)",
			res.mounts, binDir, localDir)
	}
}

// TestResolveHostToolchains_StillAscendsForNarrowVersionedRoot verifies the
// guard does NOT block the legitimate nvm/asdf case: ascending out of a
// bin/ directory that sits under a narrow, versioned toolchain root (not
// $HOME or a broad catch-all) still happens, exactly as
// TestResolveHostToolchains_SymlinkClosure already pins — this test just
// makes the "guard does not over-trigger" property explicit against a
// $HOME-adjacent-but-not-equal path.
func TestResolveHostToolchains_StillAscendsForNarrowVersionedRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("versioned bin layout assumed POSIX")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	versionedRoot := filepath.Join(home, ".nvm", "versions", "node", "v18.0.0")
	binDir := filepath.Join(versionedRoot, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(binDir, "fakenode")
	if err := os.WriteFile(exe, []byte("#!/bin/sh\necho fake-node version 18.0.0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)

	res := ResolveHostToolchains([]string{"fakenode"})
	if len(res.mounts) != 1 || res.mounts[0].HostPath != versionedRoot {
		t.Fatalf("mount HostPath = %+v, want the versioned root %q (narrow ascent must still happen for legitimate nvm/asdf layouts)",
			res.mounts, versionedRoot)
	}
}

func TestResolveHostToolchains_SymlinkClosure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink + shebang semantics assumed POSIX")
	}
	root, shimDir := fakeToolchainHost(t, "fakenode")
	t.Setenv("PATH", shimDir)

	res := ResolveHostToolchains([]string{"fakenode"})
	if len(res.mounts) != 1 {
		t.Fatalf("want 1 mount, got %d: %+v", len(res.mounts), res.mounts)
	}
	wantRoot := filepath.Join(root, "versions", "1.2.3")
	if res.mounts[0].HostPath != wantRoot {
		t.Errorf("mount HostPath = %q, want %q (the versioned toolchain root, not the shim dir)", res.mounts[0].HostPath, wantRoot)
	}
	if !res.mounts[0].Shared {
		t.Error("host toolchain mount must have Shared=true (host-owned dir, no :Z relabel)")
	}
	if res.mounts[0].ContainerPath != hostToolchainMountRoot+"/fakenode" {
		t.Errorf("ContainerPath = %q, want %s/fakenode", res.mounts[0].ContainerPath, hostToolchainMountRoot)
	}
	wantPathDir := hostToolchainMountRoot + "/fakenode/bin"
	if res.pathPrepend != wantPathDir {
		t.Errorf("pathPrepend = %q, want %q", res.pathPrepend, wantPathDir)
	}
	if len(res.Fingerprints) != 1 || res.Fingerprints[0].Name != "fakenode" {
		t.Fatalf("Fingerprints = %+v, want one entry named fakenode", res.Fingerprints)
	}
	if res.Fingerprints[0].Path != wantRoot {
		t.Errorf("fingerprint Path = %q, want %q", res.Fingerprints[0].Path, wantRoot)
	}
	if res.Fingerprints[0].Version == "" {
		t.Error("fingerprint Version should be populated from the fake --version script output")
	}
	if res.Unresolved != nil {
		t.Errorf("Unresolved = %+v, want nil (fakenode resolved)", res.Unresolved)
	}
}

// TestResolveHostToolchains_UnresolvedNamesReported pins that an entry that
// does not resolve contributes no mount, no PATH entry, and no fingerprint,
// and its trimmed name appears in Unresolved, in request order. A blank
// entry (empty or whitespace-only) is neither resolved nor unresolved and
// never appears in either slice.
func TestResolveHostToolchains_UnresolvedNamesReported(t *testing.T) {
	t.Setenv("PATH", t.TempDir()) // empty PATH: no bare name resolves

	res := ResolveHostToolchains([]string{"node-that-does-not-exist-xyz", "", "  ", "/no/such/dir"})
	want := []string{"node-that-does-not-exist-xyz", "/no/such/dir"}
	if !slices.Equal(res.Unresolved, want) {
		t.Errorf("Unresolved = %+v, want %+v", res.Unresolved, want)
	}
	if len(res.Fingerprints) != 0 || len(res.mounts) != 0 || res.pathPrepend != "" {
		t.Errorf("unresolved entries must contribute nothing: Fingerprints=%+v mounts=%+v pathPrepend=%q",
			res.Fingerprints, res.mounts, res.pathPrepend)
	}

	_, shimDir := fakeToolchainHost(t, "fakenode")
	t.Setenv("PATH", shimDir)
	res2 := ResolveHostToolchains([]string{"fakenode"})
	if res2.Unresolved != nil {
		t.Errorf("Unresolved = %+v, want nil (fakenode resolved)", res2.Unresolved)
	}
	if len(res2.Fingerprints) != 1 {
		t.Errorf("Fingerprints = %+v, want one entry", res2.Fingerprints)
	}
}

func TestResolveHostToolchains_ExplicitDir(t *testing.T) {
	dir := t.TempDir()

	res := ResolveHostToolchains([]string{dir})
	if len(res.mounts) != 1 || res.mounts[0].HostPath != dir {
		t.Fatalf("want one mount at %q, got %+v", dir, res.mounts)
	}
	if res.pathPrepend != "" {
		t.Errorf("an explicit dir with no resolved executable should not contribute a PATH entry, got %q", res.pathPrepend)
	}
	// An explicit dir still gets a fingerprint (path only; version probing
	// against a bare directory fails silently).
	if len(res.Fingerprints) != 1 || res.Fingerprints[0].Path != dir {
		t.Errorf("Fingerprints = %+v, want one entry for %q", res.Fingerprints, dir)
	}
}

func TestResolveHostToolchains_DedupesContainerPaths(t *testing.T) {
	_, shimDir := fakeToolchainHost(t, "fakenode")
	t.Setenv("PATH", shimDir)

	res := ResolveHostToolchains([]string{"fakenode", "fakenode"})
	if len(res.mounts) != 1 {
		t.Errorf("duplicate entries should collapse to one mount, got %d: %+v", len(res.mounts), res.mounts)
	}
	if res.Unresolved != nil {
		t.Errorf("Unresolved = %+v, want nil (a deduplicated entry still resolved)", res.Unresolved)
	}
}

func TestResolveHostToolchains_EmptyAndBlankEntriesSkipped(t *testing.T) {
	res := ResolveHostToolchains([]string{"", "   "})
	if len(res.mounts) != 0 {
		t.Errorf("blank entries should produce no mounts, got %+v", res.mounts)
	}
	if res.Unresolved != nil {
		t.Errorf("Unresolved = %+v, want nil (a blank entry is neither resolved nor unresolved)", res.Unresolved)
	}
}

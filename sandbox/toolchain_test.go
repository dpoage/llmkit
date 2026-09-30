package sandbox

import (
	"os"
	"os/exec"
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
	if len(res.Fingerprints) != 1 {
		t.Errorf("a deduplicated entry has no fingerprint of its own: want 1 fingerprint, got %+v", res.Fingerprints)
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

// TestSanitizeToolchainSegment pins that, whatever the entry,
// the segment is one real path component, so the mount's ContainerPath is a
// direct child of hostToolchainMountRoot (never the root itself or its
// parent, which "..", "." or "/" would produce after Clean).
func TestSanitizeToolchainSegment(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"node", "node"},
		{"/nix/store/xxx-nodejs-18", "xxx-nodejs-18"},
		{"/opt/tool/", "tool"},
		{"/x/..", "toolchain"},
		{"/x/y/../..", "toolchain"},
		{"..", "toolchain"},
		{".", "toolchain"},
		{"a/..", "toolchain"},
		{"/", "toolchain"},
		{"/x/../y", "y"},
	} {
		got := sanitizeToolchainSegment(tc.in)
		if got != tc.want {
			t.Errorf("sanitizeToolchainSegment(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if ctr := filepath.Clean(hostToolchainMountRoot + "/" + got); filepath.Dir(ctr) != hostToolchainMountRoot {
			t.Errorf("entry %q mounts at %q, not a direct child of %s", tc.in, ctr, hostToolchainMountRoot)
		}
	}
}

// TestResolveHostToolchains_AbsoluteEntryWithDotDot: an absolute entry
// `<existing dir>/..` is used cleaned and mounts at a direct child of
// hostToolchainMountRoot, never at the root itself.
func TestResolveHostToolchains_AbsoluteEntryWithDotDot(t *testing.T) {
	tmp := t.TempDir()
	res := ResolveHostToolchains([]string{tmp + "/.."})
	if len(res.mounts) != 1 {
		t.Fatalf("want 1 mount, got %+v (unresolved %v)", res.mounts, res.Unresolved)
	}
	m := res.mounts[0]
	if want := filepath.Dir(tmp); m.HostPath != want {
		t.Errorf("HostPath = %q, want the cleaned %q", m.HostPath, want)
	}
	if filepath.Dir(m.ContainerPath) != hostToolchainMountRoot || filepath.Base(m.ContainerPath) == ".." {
		t.Errorf("ContainerPath = %q, want a direct child of %s", m.ContainerPath, hostToolchainMountRoot)
	}
}

// TestIsOverbroadToolchainRoot: the shared roots the bin/ ascent must never
// land on, independent of $HOME.
func TestIsOverbroadToolchainRoot(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	for _, dir := range []string{"/", "/usr", "/usr/", "/usr/local", "/opt", "/opt/.", home, filepath.Join(home, ".local"), filepath.Join(home, ".ssh")} {
		if !isOverbroadToolchainRoot(dir) {
			t.Errorf("isOverbroadToolchainRoot(%q) = false, want true", dir)
		}
	}
	for _, dir := range []string{"/opt/node-v20", "/usr/local/node", "/usr/lib", "/nix/store/xxx-nodejs-18", filepath.Join(home, ".nvm", "versions", "node", "v20")} {
		if isOverbroadToolchainRoot(dir) {
			t.Errorf("isOverbroadToolchainRoot(%q) = true, want false (narrow)", dir)
		}
	}
	t.Setenv("HOME", "")
	if !isOverbroadToolchainRoot("/usr") {
		t.Error("/usr must be overbroad even when $HOME is unresolvable")
	}
}

// TestResolveHostToolchains_FHSBinDoesNotAscendToUsr: a bare name resolving
// into /usr/bin mounts /usr/bin, not the whole host /usr.
func TestResolveHostToolchains_FHSBinDoesNotAscendToUsr(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("FHS layout assumed POSIX")
	}
	var name string
	for _, cand := range []string{"env", "sh", "ls", "cat", "true"} {
		p, err := exec.LookPath(cand)
		if err != nil {
			continue
		}
		if r, err := filepath.EvalSymlinks(p); err == nil && filepath.Dir(r) == "/usr/bin" {
			name = cand
			break
		}
	}
	if name == "" {
		t.Skip("no candidate utility resolves into /usr/bin on this host")
	}
	res := ResolveHostToolchains([]string{name})
	if len(res.mounts) != 1 {
		t.Fatalf("want 1 mount, got %+v", res.mounts)
	}
	if res.mounts[0].HostPath != "/usr/bin" {
		t.Errorf("HostPath = %q, want /usr/bin (must NOT ascend to /usr)", res.mounts[0].HostPath)
	}
}

// writeFakeExe writes a trivial executable name into dir (created as needed).
func writeFakeExe(t *testing.T, dir, name string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
}

// mustSymlink creates link -> target.
func mustSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
}

// TestResolveHostToolchains_AbsoluteEntryChecksCleanedPath: the path
// checked for being an existing directory is the cleaned path that is
// mounted and fingerprinted. `<S>/a/b/../c` with b a symlink to
// <S>/real/b exists through the kernel's walk (<S>/real/c) but its cleaned
// form <S>/a/c does not, so the entry is Unresolved and nothing is mounted;
// once <S>/a/c exists, that cleaned path is what mounts.
func TestResolveHostToolchains_AbsoluteEntryChecksCleanedPath(t *testing.T) {
	s, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range []string{"real/b", "real/c", "a"} {
		if err := os.MkdirAll(filepath.Join(s, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	mustSymlink(t, filepath.Join(s, "real", "b"), filepath.Join(s, "a", "b"))
	entry := s + "/a/b/../c"

	res := ResolveHostToolchains([]string{entry})
	if len(res.mounts) != 0 || len(res.Fingerprints) != 0 || !slices.Equal(res.Unresolved, []string{entry}) {
		t.Fatalf("cleaned %s/a/c does not exist: mounts=%+v fingerprints=%+v unresolved=%q, want no mount and Unresolved=[%q]", s, res.mounts, res.Fingerprints, res.Unresolved, entry)
	}

	if err := os.Mkdir(filepath.Join(s, "a", "c"), 0o755); err != nil {
		t.Fatal(err)
	}
	res = ResolveHostToolchains([]string{entry})
	want := filepath.Join(s, "a", "c")
	if len(res.mounts) != 1 || res.mounts[0].HostPath != want || res.Fingerprints[0].Path != want {
		t.Fatalf("mounts=%+v fingerprints=%+v, want the cleaned %s mounted and fingerprinted", res.mounts, res.Fingerprints, want)
	}
}

// TestResolveHostToolchains_OverbroadRootThroughSymlink: the ascent guard
// compares symlink-resolved paths, so a shared root reached through a
// symlink is still refused — a symlinked $HOME whose ~/.local/bin holds the
// tool, and a symlinked /usr/local (Fedora Atomic: /usr/local ->
// /var/usrlocal), injected here as a shared root because the host's own
// /usr/local cannot be relinked in a hermetic test.
func TestResolveHostToolchains_OverbroadRootThroughSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinked-root layouts assumed POSIX")
	}
	tmp, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	t.Run("symlinked HOME", func(t *testing.T) {
		realHome := filepath.Join(tmp, "realhome")
		writeFakeExe(t, filepath.Join(realHome, ".local", "bin"), "tltool")
		link := filepath.Join(tmp, "linkhome")
		mustSymlink(t, realHome, link)
		t.Setenv("HOME", link)
		t.Setenv("PATH", filepath.Join(link, ".local", "bin"))

		res := ResolveHostToolchains([]string{"tltool"})
		if want := filepath.Join(realHome, ".local", "bin"); len(res.mounts) != 1 || res.mounts[0].HostPath != want {
			t.Fatalf("mounts=%+v unresolved=%q, want HostPath %s (never ~/.local)", res.mounts, res.Unresolved, want)
		}
	})

	t.Run("symlinked /usr/local", func(t *testing.T) {
		target := filepath.Join(tmp, "var-usrlocal")
		writeFakeExe(t, filepath.Join(target, "bin"), "ultool")
		link := filepath.Join(tmp, "usrlocal")
		mustSymlink(t, target, link)
		prev := sharedToolchainRoots
		sharedToolchainRoots = []string{"/", "/usr", link, "/opt"}
		t.Cleanup(func() { sharedToolchainRoots = prev })
		t.Setenv("HOME", filepath.Join(tmp, "otherhome"))
		t.Setenv("PATH", filepath.Join(link, "bin"))

		res := ResolveHostToolchains([]string{"ultool"})
		if want := filepath.Join(target, "bin"); len(res.mounts) != 1 || res.mounts[0].HostPath != want {
			t.Fatalf("mounts=%+v unresolved=%q, want HostPath %s (never the whole symlinked root)", res.mounts, res.Unresolved, want)
		}
	})
}

// TestResolveHostToolchains_BinaryDirectlyInHomeUnresolved: a bare name
// whose executable sits directly in $HOME would otherwise mount all of
// $HOME (no ascent is involved); it lands in Unresolved instead.
func TestResolveHostToolchains_BinaryDirectlyInHomeUnresolved(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("HOME layout assumed POSIX")
	}
	home := filepath.Join(t.TempDir(), "home")
	writeFakeExe(t, home, "hometool")
	t.Setenv("HOME", home)
	t.Setenv("PATH", home)

	res := ResolveHostToolchains([]string{"hometool"})
	if len(res.mounts) != 0 || !slices.Equal(res.Unresolved, []string{"hometool"}) {
		t.Fatalf("mounts=%+v unresolved=%q, want no mount and Unresolved=[hometool]", res.mounts, res.Unresolved)
	}
}

// cwdHomeWithRelativeTools makes a fresh $HOME holding hometool, bin/bt and
// .ssh/id_secret (content "SECRET"), points HOME at it and makes it the
// working directory, so a relative toolchain entry resolves against $HOME.
func cwdHomeWithRelativeTools(t *testing.T) string {
	t.Helper()
	tmp, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	home := filepath.Join(tmp, "home")
	writeFakeExe(t, home, "hometool")
	writeFakeExe(t, filepath.Join(home, "bin"), "bt")
	if err := os.MkdirAll(filepath.Join(home, ".ssh"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, ".ssh", "id_secret"), []byte("SECRET\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Chdir(home)
	return home
}

// TestResolveHostToolchains_RelativeEntryResolvedAbsolute: a relative entry
// containing a slash resolves against the working directory, and what is
// guarded, mounted and fingerprinted is its absolute path — so with the
// working directory at $HOME, "./hometool" (directly in $HOME) is
// Unresolved and "bin/bt" mounts <HOME>/bin, never "." (= $HOME).
func TestResolveHostToolchains_RelativeEntryResolvedAbsolute(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("HOME layout assumed POSIX")
	}
	home := cwdHomeWithRelativeTools(t)

	res := ResolveHostToolchains([]string{"./hometool"})
	if len(res.mounts) != 0 || len(res.Fingerprints) != 0 || !slices.Equal(res.Unresolved, []string{"./hometool"}) {
		t.Errorf("./hometool: mounts=%+v fingerprints=%+v unresolved=%q, want no mount and Unresolved=[./hometool]", res.mounts, res.Fingerprints, res.Unresolved)
	}
	for _, entry := range []string{"bin/bt", "./bin/bt"} {
		res := ResolveHostToolchains([]string{entry})
		want := filepath.Join(home, "bin")
		if len(res.mounts) != 1 || res.mounts[0].HostPath != want || len(res.Fingerprints) != 1 || res.Fingerprints[0].Path != want {
			t.Errorf("%s: mounts=%+v fingerprints=%+v unresolved=%q, want one mount and fingerprint of %s", entry, res.mounts, res.Fingerprints, res.Unresolved, want)
		}
	}

	// A narrow relative layout still ascends past bin/, to an absolute root.
	writeFakeExe(t, filepath.Join(home, "tc", "v1", "bin"), "tctool")
	res = ResolveHostToolchains([]string{"tc/v1/bin/tctool"})
	if want := filepath.Join(home, "tc", "v1"); len(res.mounts) != 1 || res.mounts[0].HostPath != want || res.Fingerprints[0].Path != want {
		t.Errorf("tc/v1/bin/tctool: mounts=%+v fingerprints=%+v, want HostPath and fingerprint %s", res.mounts, res.Fingerprints, want)
	}
}

// symlinkedCwdLayout builds, under a fresh symlink-free root R (returned),
// the relative-entry layouts where the kernel's path walk and a lexical
// clean disagree:
//
//	R/phys/proj/                   the physical working directory
//	R/phys/tools/bin/tt            prints tt-PHYS
//	R/logical/proj -> R/phys/proj
//	R/logical/tools/bin/tt         prints tt-LOGICAL (lexical "../" from R/logical/proj)
//	R/c/link -> R/far/x
//	R/far/bin/ft                   prints ft-FAR
//
// HOME points at the fresh R/home, so the overbroad guard never sees the
// user's own home.
func symlinkedCwdLayout(t *testing.T) string {
	t.Helper()
	r, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	writeMarkerExe := func(dir, name, marker string) {
		t.Helper()
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\necho "+marker+"\n"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range []string{"phys/proj", "logical", "c", "far/x", "home"} {
		if err := os.MkdirAll(filepath.Join(r, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	writeMarkerExe(filepath.Join(r, "phys", "tools", "bin"), "tt", "tt-PHYS")
	writeMarkerExe(filepath.Join(r, "logical", "tools", "bin"), "tt", "tt-LOGICAL")
	mustSymlink(t, filepath.Join(r, "phys", "proj"), filepath.Join(r, "logical", "proj"))
	writeMarkerExe(filepath.Join(r, "far", "bin"), "ft", "ft-FAR")
	mustSymlink(t, filepath.Join(r, "far", "x"), filepath.Join(r, "c", "link"))
	t.Setenv("HOME", filepath.Join(r, "home"))
	return r
}

// TestResolveHostToolchains_RelativeEntryIsKernelPath: a relative entry
// resolves to the file the kernel would execute from the working
// directory, so its ".." names the physical parent — after a symlinked
// component, and out of a working directory entered through a symlink
// whether PWD names that symlinked path or is unset. Root, fingerprint and
// PATH entry all come from that physical file.
func TestResolveHostToolchains_RelativeEntryIsKernelPath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinked working directory layouts assumed POSIX")
	}
	r := symlinkedCwdLayout(t)
	logicalCwd := filepath.Join(r, "logical", "proj")
	check := func(t *testing.T, entry, wantRoot, wantVersion, wantPath string) {
		t.Helper()
		res := ResolveHostToolchains([]string{entry})
		if res.Unresolved != nil || len(res.mounts) != 1 || res.mounts[0].HostPath != wantRoot ||
			len(res.Fingerprints) != 1 || res.Fingerprints[0].Path != wantRoot || res.Fingerprints[0].Version != wantVersion ||
			res.pathPrepend != wantPath {
			t.Fatalf("%s: mounts=%+v fingerprints=%+v pathPrepend=%q unresolved=%q; want HostPath and fingerprint Path %s, Version %s, pathPrepend %s",
				entry, res.mounts, res.Fingerprints, res.pathPrepend, res.Unresolved, wantRoot, wantVersion, wantPath)
		}
	}
	physTools := filepath.Join(r, "phys", "tools")
	ttPath := hostToolchainMountRoot + "/tt/bin"

	t.Run("cwd entered through a symlink, PWD names it", func(t *testing.T) {
		t.Chdir(logicalCwd) // also sets PWD to the symlinked path
		check(t, "../tools/bin/tt", physTools, "tt-PHYS", ttPath)
	})
	t.Run("same cwd, PWD unset", func(t *testing.T) {
		t.Chdir(logicalCwd)
		if err := os.Unsetenv("PWD"); err != nil {
			t.Fatal(err)
		}
		check(t, "../tools/bin/tt", physTools, "tt-PHYS", ttPath)
	})
	t.Run("lexical parent's tools absent", func(t *testing.T) {
		if err := os.RemoveAll(filepath.Join(r, "logical", "tools")); err != nil {
			t.Fatal(err)
		}
		t.Chdir(logicalCwd)
		check(t, "../tools/bin/tt", physTools, "tt-PHYS", ttPath)
	})
	t.Run("dotdot after a symlinked component", func(t *testing.T) {
		t.Chdir(filepath.Join(r, "c"))
		check(t, "link/../bin/ft", filepath.Join(r, "far"), "ft-FAR", hostToolchainMountRoot+"/ft/bin")
	})
}

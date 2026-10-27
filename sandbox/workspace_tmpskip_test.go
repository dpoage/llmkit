package sandbox

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestForeignTmpHelperProcess is the child half of runForeignTmpChild.
func TestForeignTmpHelperProcess(t *testing.T) {
	mode := os.Getenv("LLMKIT_FOREIGN_MODE")
	if mode == "" {
		t.Skip("child half of runForeignTmpChild")
	}
	repoB := os.Getenv("LLMKIT_FOREIGN_REPO")
	repoA := os.Getenv("LLMKIT_FOREIGN_REPO_A")
	src := os.Getenv("LLMKIT_FOREIGN_SRC")

	// Seed TMPDIR with another repo's leftovers before the entry under test.
	var shared wsCache
	var err error
	switch src {
	case "none":
	case "sameCache":
		if mode != "cached" {
			t.Fatalf("src sameCache needs mode cached, got %q", mode)
		}
		var wsA string
		if wsA, _, err = prepareWorkspaceCached(&shared, repoA, nil); err == nil {
			_ = os.RemoveAll(wsA)
		}
	case "otherCache":
		var cForeign wsCache
		var wsA string
		if wsA, _, err = prepareWorkspaceCached(&cForeign, repoA, nil); err == nil {
			_ = os.RemoveAll(wsA)
		}
	case "liveWs":
		var cForeign wsCache
		// The other run's workspace (and its cache) stay live in TMPDIR.
		_, _, err = prepareWorkspaceCached(&cForeign, repoA, nil)
	default:
		t.Fatalf("unknown src %q", src)
	}
	if err != nil {
		t.Fatalf("seed %s: %v", src, err)
	}

	var (
		ws       string
		listRoot string
	)
	switch mode {
	case "prepare":
		ws, err = prepareWorkspace(repoB, nil)
	case "cached":
		if src == "sameCache" {
			ws, _, err = prepareWorkspaceCached(&shared, repoB, nil)
		} else {
			var c wsCache
			ws, _, err = prepareWorkspaceCached(&c, repoB, nil)
		}
	case "gitlink":
		// The destination sits inside a gitlink directory entry that
		// copyFileListSkipping copies recursively.
		if ws, err = os.MkdirTemp("", "llmkit-sandbox-"); err == nil {
			var wsInfo os.FileInfo
			if wsInfo, err = os.Stat(ws); err == nil {
				err = copyFileListSkipping(repoB, ws, []string{filepath.Join("vendor", "gtest")}, wsInfo)
			}
			listRoot = filepath.Join(ws, "vendor", "gtest")
		}
	case "movedCache":
		// The cache dir is created inside repoB, then TMPDIR moves in-process;
		// the forced miss re-materializes with the temp clause inert.
		var c wsCache
		var ws1 string
		if ws1, _, err = prepareWorkspaceCached(&c, repoB, nil); err != nil {
			t.Fatalf("movedCache first clone: %v", err)
		}
		_ = os.RemoveAll(ws1)
		if err = os.WriteFile(filepath.Join(repoB, "b.txt"), []byte("b2\n"), 0o644); err != nil {
			t.Fatalf("movedCache dirty repo: %v", err)
		}
		if err = os.Setenv("TMPDIR", os.Getenv("LLMKIT_FOREIGN_TMP2")); err != nil {
			t.Fatalf("movedCache set TMPDIR: %v", err)
		}
		var hit bool
		if ws, hit, err = prepareWorkspaceCached(&c, repoB, nil); err == nil && hit {
			_ = os.RemoveAll(ws)
			t.Fatalf("movedCache: second clone must be a cache miss")
		}
	default:
		t.Fatalf("unknown mode %q", mode)
	}
	if err != nil {
		if ws != "" {
			_ = os.RemoveAll(ws)
		}
		t.Fatalf("%s/%s: %v", mode, src, err)
	}
	if listRoot == "" {
		listRoot = ws
	}
	lines, lerr := treeListing(listRoot)
	_ = os.RemoveAll(ws)
	if lerr != nil {
		t.Fatalf("list workspace: %v", lerr)
	}
	for _, l := range lines {
		_, _ = os.Stdout.WriteString("FOREIGN " + l + "\n")
	}
}

// runForeignTmpChild runs the helper test in a child with TMPDIR=tmpdir and returns
// the workspace listing; the wall-clock bound stops a self-copy loop growing disk.
func runForeignTmpChild(t *testing.T, mode, src, repoB, repoA, tmpdir string) []string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestForeignTmpHelperProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(),
		"LLMKIT_FOREIGN_MODE="+mode,
		"LLMKIT_FOREIGN_SRC="+src,
		"LLMKIT_FOREIGN_REPO="+repoB,
		"LLMKIT_FOREIGN_REPO_A="+repoA,
		"TMPDIR="+tmpdir,
	)
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("child %s/%s did not terminate within 60s", mode, src)
	}
	if err != nil {
		t.Fatalf("child %s/%s failed: %v\n%s", mode, src, err, out)
	}
	var got []string
	for _, line := range strings.Split(string(out), "\n") {
		if rest, ok := strings.CutPrefix(line, "FOREIGN "); ok {
			got = append(got, rest)
		}
	}
	sort.Strings(got)
	return got
}

// llmkitPrefixedName mirrors the package's os.MkdirTemp prefixes independently
// of the production constants, so the expectation cannot follow a prefix edit.
func llmkitPrefixedName(name string) bool {
	return strings.HasPrefix(name, "llmkit-sandbox-") ||
		strings.HasPrefix(name, "llmkit-wscache-") ||
		strings.HasPrefix(name, "llmkit-mock-")
}

// dropLlmkitTempDirEntries removes entries under an llmkit-prefixed direct child
// of tmpRel ("." = the listing root), which the copy must leave out.
func dropLlmkitTempDirEntries(listing []string, tmpRel string) []string {
	var out []string
	for _, l := range listing {
		_, rel, _ := strings.Cut(l, " ")
		parts := strings.Split(rel, "/")
		if tmpRel != "." {
			if len(parts) < 2 || parts[0] != tmpRel {
				out = append(out, l)
				continue
			}
			parts = parts[1:]
		}
		if llmkitPrefixedName(parts[0]) {
			continue
		}
		out = append(out, l)
	}
	return out
}

// With TMPDIR at or under RepoDir, the workspace is the repo minus llmkit temp dirs.
func TestWorkspaceLeavesLlmkitTempDirsOut(t *testing.T) {
	type row struct {
		name    string
		gitB    bool   // RepoDir is a git work tree
		entry   string // prepare | cached | gitlink
		tmp     string // inside | alias | eqR
		src     string // none | sameCache | otherCache | liveWs
		userDir string // pre-created inside TMPDIR; left out (accepted cost)
	}
	rows := []row{
		{"prepare/nongit/inside/otherCache", false, "prepare", "inside", "otherCache", ""},
		{"prepare/nongit/inside/liveWs", false, "prepare", "inside", "liveWs", ""},
		{"prepare/git/inside/liveWs", true, "prepare", "inside", "liveWs", ""},
		{"cached/nongit/inside/sameCache", false, "cached", "inside", "sameCache", ""},
		{"cached/nongit/inside/otherCache", false, "cached", "inside", "otherCache", ""},
		{"cached/nongit/inside/liveWs", false, "cached", "inside", "liveWs", ""},
		{"cached/git/inside/sameCache", true, "cached", "inside", "sameCache", ""},
		{"cached/git/inside/otherCache", true, "cached", "inside", "otherCache", ""},
		{"cached/git/inside/liveWs", true, "cached", "inside", "liveWs", ""},
		{"prepare/nongit/alias/liveWs", false, "prepare", "alias", "liveWs", ""},
		{"cached/git/alias/otherCache", true, "cached", "alias", "otherCache", ""},
		{"prepare/nongit/eqR/liveWs", false, "prepare", "eqR", "liveWs", ""},
		{"cached/git/eqR/otherCache", true, "cached", "eqR", "otherCache", ""},
		{"gitlink/inside/liveWs", false, "gitlink", "inside", "liveWs", ""},
		{"prepare/nongit/inside/userdir", false, "prepare", "inside", "none", "llmkit-mock-zzz"},
		{"cached/git/inside/userdir", true, "cached", "inside", "otherCache", "llmkit-sandbox-user"},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			if r.gitB || r.src != "none" {
				if _, err := exec.LookPath("git"); err != nil {
					t.Skip("git not on PATH")
				}
			}
			if r.tmp == "alias" && runtime.GOOS == "windows" {
				t.Skip("symlink alias needs unix symlinks")
			}

			repoB := t.TempDir()
			mustWrite(t, filepath.Join(repoB, "b.txt"), "b\n", 0o644)
			mustMkdir(t, filepath.Join(repoB, "tmp"))
			mustWrite(t, filepath.Join(repoB, "tmp", "keep.txt"), "keep\n", 0o644)
			mustMkdir(t, filepath.Join(repoB, "tmp", "other-dir"))
			mustWrite(t, filepath.Join(repoB, "tmp", "other-dir", "od.txt"), "od\n", 0o644)
			// Prefixed but not a child of TMPDIR on inside/alias rows: kept.
			mustMkdir(t, filepath.Join(repoB, "llmkit-sandbox-notes"))
			mustWrite(t, filepath.Join(repoB, "llmkit-sandbox-notes", "x.txt"), "notes\n", 0o644)
			if r.userDir != "" {
				mustMkdir(t, filepath.Join(repoB, "tmp", r.userDir))
				mustWrite(t, filepath.Join(repoB, "tmp", r.userDir, "user.txt"), "user\n", 0o644)
			}

			listRoot := repoB
			if r.entry == "gitlink" {
				gt := filepath.Join(repoB, "vendor", "gtest")
				mustMkdir(t, filepath.Join(gt, "src"))
				mustMkdir(t, filepath.Join(gt, "tmp"))
				mustWrite(t, filepath.Join(gt, "CMakeLists.txt"), "project(gtest)\n", 0o644)
				mustWrite(t, filepath.Join(gt, ".git"), "gitdir: ../../.git/modules/gtest\n", 0o644)
				mustWrite(t, filepath.Join(gt, "src", "a.cc"), "int a;\n", 0o644)
				mustWrite(t, filepath.Join(gt, "tmp", "keep.txt"), "keep\n", 0o644)
				mustMkdir(t, filepath.Join(gt, "tmp", "other-dir"))
				mustWrite(t, filepath.Join(gt, "tmp", "other-dir", "od.txt"), "od\n", 0o644)
				listRoot = gt
			}
			if r.gitB {
				gitInit(t, repoB)
			}

			repoA := ""
			if r.src != "none" {
				repoA = t.TempDir()
				mustWrite(t, filepath.Join(repoA, "a.txt"), "SECRET-A\n", 0o644)
				gitInit(t, repoA)
			}

			pre, err := treeListing(listRoot)
			if err != nil {
				t.Fatal(err)
			}

			tmpdir := filepath.Join(repoB, "tmp")
			tmpRel := "tmp"
			if r.entry == "gitlink" {
				tmpdir = filepath.Join(listRoot, "tmp")
			}
			switch r.tmp {
			case "alias":
				alias := filepath.Join(t.TempDir(), "alias")
				if err := os.Symlink(tmpdir, alias); err != nil {
					t.Fatal(err)
				}
				tmpdir = alias
			case "eqR":
				tmpdir = repoB
				tmpRel = "."
			}

			want := dropLlmkitTempDirEntries(pre, tmpRel)
			got := runForeignTmpChild(t, r.entry, r.src, repoB, repoA, tmpdir)
			diffListings(t, got, want)
		})
	}
}

// runMovedTmpChild is runForeignTmpChild in movedCache mode: TMPDIR=tmpdir for
// the first cache use, tmp2 for the second.
func runMovedTmpChild(t *testing.T, repo, tmpdir, tmp2 string) []string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestForeignTmpHelperProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(),
		"LLMKIT_FOREIGN_MODE=movedCache",
		"LLMKIT_FOREIGN_SRC=none",
		"LLMKIT_FOREIGN_REPO="+repo,
		"LLMKIT_FOREIGN_REPO_A=",
		"LLMKIT_FOREIGN_TMP2="+tmp2,
		"TMPDIR="+tmpdir,
	)
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("movedCache child did not terminate within 60s")
	}
	if err != nil {
		t.Fatalf("movedCache child failed: %v\n%s", err, out)
	}
	var got []string
	for _, line := range strings.Split(string(out), "\n") {
		if rest, ok := strings.CutPrefix(line, "FOREIGN "); ok {
			got = append(got, rest)
		}
	}
	sort.Strings(got)
	return got
}

// A prefixed file, symlink, or TMPDIR itself copies; only a prefixed directory is left out.
func TestWorkspaceKeepsPrefixedEntriesThatAreNotTempDirs(t *testing.T) {
	type row struct {
		name  string
		shape string // filelink | eqR | tmpIsNested
		git   bool
		entry string // prepare | cached
	}
	rows := []row{
		{"filelink/git/prepare", "filelink", true, "prepare"},
		{"filelink/git/cached", "filelink", true, "cached"},
		{"filelink/nongit/prepare", "filelink", false, "prepare"},
		{"filelink/nongit/cached", "filelink", false, "cached"},
		{"eqR/git/prepare", "eqR", true, "prepare"},
		{"eqR/git/cached", "eqR", true, "cached"},
		{"eqR/nongit/prepare", "eqR", false, "prepare"},
		{"eqR/nongit/cached", "eqR", false, "cached"},
		{"tmpIsNested/git/prepare", "tmpIsNested", true, "prepare"},
		{"tmpIsNested/git/cached", "tmpIsNested", true, "cached"},
		{"tmpIsNested/nongit/prepare", "tmpIsNested", false, "prepare"},
		{"tmpIsNested/nongit/cached", "tmpIsNested", false, "cached"},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			if r.git || r.shape == "tmpIsNested" {
				if _, err := exec.LookPath("git"); err != nil {
					t.Skip("git not on PATH")
				}
			}
			if r.shape == "filelink" && runtime.GOOS == "windows" {
				t.Skip("symlink creation needs privileges on windows")
			}
			repo := t.TempDir()
			mustWrite(t, filepath.Join(repo, "b.txt"), "b\n", 0o644)
			var tmpdir string
			switch r.shape {
			case "filelink":
				mustMkdir(t, filepath.Join(repo, "tmp"))
				mustWrite(t, filepath.Join(repo, "tmp", "keep.txt"), "keep\n", 0o644)
				// Prefixed file and symlink inside TMPDIR: kept.
				mustWrite(t, filepath.Join(repo, "tmp", "llmkit-sandbox-notes.txt"), "n\n", 0o644)
				if err := os.Symlink("../b.txt", filepath.Join(repo, "tmp", "llmkit-mock-link")); err != nil {
					t.Fatal(err)
				}
				tmpdir = filepath.Join(repo, "tmp")
			case "eqR":
				mustMkdir(t, filepath.Join(repo, "tmp"))
				mustWrite(t, filepath.Join(repo, "tmp", "keep.txt"), "keep\n", 0o644)
				mustWrite(t, filepath.Join(repo, "llmkit-wscache-readme.md"), "r\n", 0o644)
				// A prefixed directory directly inside TMPDIR: left out.
				mustMkdir(t, filepath.Join(repo, "llmkit-sandbox-userdir"))
				mustWrite(t, filepath.Join(repo, "llmkit-sandbox-userdir", "u.txt"), "u\n", 0o644)
				tmpdir = repo
			case "tmpIsNested":
				// The outer gitInit runs first so the nested repo stays
				// untracked and git lists it as "llmkit-sandbox-nested/".
				if r.git {
					gitInit(t, repo)
				}
				nested := filepath.Join(repo, "llmkit-sandbox-nested")
				mustMkdir(t, nested)
				mustWrite(t, filepath.Join(nested, "v.txt"), "v\n", 0o644)
				gitInit(t, nested) // nested is TMPDIR itself: kept whole
				tmpdir = nested
			}
			if r.git && r.shape != "tmpIsNested" {
				gitInit(t, repo)
			}
			pre, err := treeListing(repo)
			if err != nil {
				t.Fatal(err)
			}
			want := pre
			if r.shape == "eqR" {
				want = nil
				for _, l := range pre {
					if strings.HasPrefix(l, "d llmkit-sandbox-userdir") ||
						strings.HasPrefix(l, "f llmkit-sandbox-userdir/") {
						continue
					}
					want = append(want, l)
				}
			}
			got := runForeignTmpChild(t, r.entry, "none", repo, "", tmpdir)
			diffListings(t, got, want)
		})
	}
}

// After TMPDIR moves, only the identity clause keeps the cache dir out of the copy.
func TestWorkspaceNeverCopiesCacheDirAfterTmpdirMove(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	mustWrite(t, filepath.Join(repo, "b.txt"), "b\n", 0o644)
	gitInit(t, repo)
	vend := filepath.Join(repo, "vend")
	mustMkdir(t, vend)
	mustWrite(t, filepath.Join(vend, "v.txt"), "v\n", 0o644)
	// Untracked nested repo: the git file list carries "vend/" as a directory
	// entry whose recursive copy walks past the cache dir.
	gitInit(t, vend)
	tmpdir := filepath.Join(vend, "tmp")
	mustMkdir(t, tmpdir)
	want, err := treeListing(repo)
	if err != nil {
		t.Fatal(err)
	}
	got := runMovedTmpChild(t, repo, tmpdir, t.TempDir())
	diffListings(t, got, want)
}

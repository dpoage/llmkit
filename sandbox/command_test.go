package sandbox

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestBuildRunArgsSecurityFlags(t *testing.T) {
	args := buildRunArgs(runParams{
		containerName: "llmkit-abc",
		workspace:     "/tmp/ws",
		image:         "alpine:latest",
		network:       "none",
		cpus:          1.5,
		memoryMB:      512,
		pidsLimit:     128,
		env:           []string{"FOO=bar", "BAZ=qux"},
		cmd:           []string{"sh", "-c", "echo hi"},
	})

	joined := strings.Join(args, " ")

	// Subcommand and security-relevant flags must all be present.
	mustContainSeq(t, args, "run")
	mustContainSeq(t, args, "--rm")
	mustContainSeq(t, args, "--network=none")
	mustContainSeq(t, args, "--read-only")
	mustContainSeq(t, args, "--cap-drop", "ALL")
	mustContainSeq(t, args, "--security-opt", "no-new-privileges")
	mustContainSeq(t, args, "--name", "llmkit-abc")
	mustContainSeq(t, args, "--workdir", WorkspaceMount)
	mustContainSeq(t, args, "--pids-limit", "128")
	mustContainSeq(t, args, "--memory", "512m")
	mustContainSeq(t, args, "--cpus", "1.5")
	mustContainSeq(t, args, "-v", "/tmp/ws:/workspace:rw,Z")
	mustContainSeq(t, args, "--env", "FOO=bar")
	mustContainSeq(t, args, "--env", "BAZ=qux")

	// Image must appear after the flags and before the command.
	imgIdx := slices.Index(args, "alpine:latest")
	cmdIdx := slices.Index(args, "echo hi")
	if imgIdx < 0 || cmdIdx < 0 || imgIdx > cmdIdx {
		t.Fatalf("image must precede command; args=%q", joined)
	}
	// Command tail must be exactly the spec command, in order, at the end.
	tail := args[len(args)-3:]
	if !slices.Equal(tail, []string{"sh", "-c", "echo hi"}) {
		t.Errorf("command tail = %q, want sh -c 'echo hi'", tail)
	}
}

func TestBuildRunArgsOmitsZeroLimits(t *testing.T) {
	args := buildRunArgs(runParams{
		containerName: "llmkit-x",
		workspace:     "/ws",
		image:         "img",
		network:       "none",
		cpus:          0,
		memoryMB:      0,
		pidsLimit:     0,
		cmd:           []string{"true"},
	})
	for _, flag := range []string{"--cpus", "--memory", "--pids-limit"} {
		if slices.Contains(args, flag) {
			t.Errorf("expected %s to be omitted when its value is zero; args=%q", flag, args)
		}
	}
}

// TestBuildRunArgsSizesScratchTmpfs pins the sized-tmpfs acceptance for the
// container backend: a configured scratchSizeMB is rendered as the --tmpfs
// size=... mount option.
func TestBuildRunArgsSizesScratchTmpfs(t *testing.T) {
	args := buildRunArgs(runParams{
		containerName: "llmkit-scratch",
		workspace:     "/ws",
		image:         "img",
		network:       "none",
		scratchSizeMB: 1024,
		cmd:           []string{"true"},
	})
	mustContainSeq(t, args, "--tmpfs", "/tmp:rw,exec,nosuid,size=1024m")
}

// TestBuildRunArgsDefaultScratchSize pins the unconfigured scratch size: a
// CLI with no options renders /tmp at the documented 512 MB.
func TestBuildRunArgsDefaultScratchSize(t *testing.T) {
	s := &CLI{runtime: "podman", defaultImage: "img", defaults: baseDefaults()}
	var o options
	o.applyDefaults(&s.defaults)
	p, err := s.resolveParams(Spec{Cmd: []string{"true"}})
	if err != nil {
		t.Fatalf("resolveParams: %v", err)
	}
	mustContainSeq(t, buildRunArgs(p), "--tmpfs", "/tmp:rw,exec,nosuid,size=512m")
}

// TestBuildRunArgsNoPathOverrideWithoutToolchains pins that a CLI with no
// host toolchains renders no --env PATH=, so the image's own ENV PATH
// stays in effect.
func TestBuildRunArgsNoPathOverrideWithoutToolchains(t *testing.T) {
	s := &CLI{runtime: "podman", defaultImage: "img", defaults: baseDefaults()}
	p, err := s.resolveParams(Spec{Cmd: []string{"true"}, Env: []string{"FOO=bar"}})
	if err != nil {
		t.Fatalf("resolveParams: %v", err)
	}
	args := buildRunArgs(p)
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--env" && strings.HasPrefix(args[i+1], "PATH=") {
			t.Errorf("args = %q, want no --env PATH= without host toolchains", args)
		}
	}
}

// TestBuildRunArgsRendersToolchainBinds pins S1-2's CLI-side rendering:
// WithHostToolchains' resolved mounts render as plain `-v host:ctr:ro`
// (never :Z, regardless of ROMount.Shared — these are always host-owned
// installs), and its PATH prefix renders as an --env PATH override BEFORE
// any Spec.Env entry, so an explicit Spec.Env PATH still wins.
func TestBuildRunArgsRendersToolchainBinds(t *testing.T) {
	args := buildRunArgs(runParams{
		containerName: "llmkit-toolchains",
		workspace:     "/ws",
		image:         "img",
		network:       "none",
		scratchSizeMB: 512,
		cmd:           []string{"true"},
		toolchainBinds: []ROMount{
			{HostPath: "/host/node", ContainerPath: "/opt/llmkit-toolchains/node", Shared: true},
			{HostPath: "/host/py", ContainerPath: "/opt/llmkit-toolchains/py", Shared: true},
		},
		toolchainPathPrepend: "/opt/llmkit-toolchains/node/bin:/opt/llmkit-toolchains/py/bin",
		env:                  []string{"PATH=/operator/bin"},
	})
	mustContainSeq(t, args, "-v", "/host/node:/opt/llmkit-toolchains/node:ro")
	mustContainSeq(t, args, "-v", "/host/py:/opt/llmkit-toolchains/py:ro")
	wantToolchainPath := "--env PATH=/opt/llmkit-toolchains/node/bin:/opt/llmkit-toolchains/py/bin:" + defaultContainerPath
	gotToolchainPath := false
	gotOperatorPath := false
	for i := 0; i+1 < len(args); i++ {
		if args[i] != "--env" {
			continue
		}
		if args[i+1] == "PATH=/opt/llmkit-toolchains/node/bin:/opt/llmkit-toolchains/py/bin:"+defaultContainerPath {
			gotToolchainPath = true
		}
		if args[i+1] == "PATH=/operator/bin" {
			gotOperatorPath = true
			if !gotToolchainPath {
				t.Errorf("Spec.Env PATH rendered before the toolchain PATH override; want %q first", wantToolchainPath)
			}
		}
	}
	if !gotToolchainPath {
		t.Errorf("args = %q, want an --env %s", args, wantToolchainPath)
	}
	if !gotOperatorPath {
		t.Errorf("args = %q, want Spec.Env's --env PATH=/operator/bin after the toolchain override", args)
	}
}

func TestBuildRunArgsRendersReadOnlyMounts(t *testing.T) {
	args := buildRunArgs(runParams{
		containerName: "llmkit-ro",
		workspace:     "/tmp/ws",
		image:         "img",
		network:       "none",
		cmd:           []string{"true"},
		roMounts: []ROMount{
			// Shared=false (default): tool-owned dir gets :ro,Z for SELinux isolation.
			{HostPath: "/host/modcache", ContainerPath: "/modcache"},
			{HostPath: "/host/other", ContainerPath: "/other"},
		},
	})

	// Non-shared RO mounts are rendered :ro,Z and never rw.
	mustContainSeq(t, args, "-v", "/host/modcache:/modcache:ro,Z")
	mustContainSeq(t, args, "-v", "/host/other:/other:ro,Z")

	// The workspace must still be the only rw mount.
	mustContainSeq(t, args, "-v", "/tmp/ws:/workspace:rw,Z")
	for i, a := range args {
		if a == "-v" && i+1 < len(args) {
			val := args[i+1]
			if strings.Contains(val, "modcache") && strings.Contains(val, "rw,") {
				t.Errorf("modcache mount must be read-only, got %q", val)
			}
		}
	}

	// Ordering: RO mounts render immediately after the workspace mount.
	wsIdx := slices.Index(args, "/tmp/ws:/workspace:rw,Z")
	roIdx := slices.Index(args, "/host/modcache:/modcache:ro,Z")
	if wsIdx < 0 || roIdx < 0 || roIdx < wsIdx {
		t.Errorf("RO mount must follow workspace mount; ws=%d ro=%d args=%q", wsIdx, roIdx, args)
	}
	// And RO mounts must come before the image/command.
	imgIdx := slices.Index(args, "img")
	if imgIdx < 0 || roIdx > imgIdx {
		t.Errorf("RO mount must precede image; ro=%d img=%d", roIdx, imgIdx)
	}
}

func TestBuildRunArgsSharedMountNoRelabel(t *testing.T) {
	// Shared=true (host Go module cache): must render :ro with NO :Z suffix.
	// :Z on a shared host dir relabels it to a container-private MCS label,
	// which is slow on large caches and can break the host go toolchain.
	args := buildRunArgs(runParams{
		containerName: "llmkit-shared",
		workspace:     "/tmp/ws",
		image:         "img",
		network:       "none",
		cmd:           []string{"true"},
		roMounts: []ROMount{
			{HostPath: "/home/user/go/pkg/mod", ContainerPath: "/modcache", Shared: true},
		},
	})

	// Shared mount must render :ro with NO ,Z suffix.
	mustContainSeq(t, args, "-v", "/home/user/go/pkg/mod:/modcache:ro")

	// Verify :Z is absent from the shared mount entry.
	for i, a := range args {
		if a == "-v" && i+1 < len(args) {
			val := args[i+1]
			if strings.Contains(val, "/modcache") && strings.Contains(val, ",Z") {
				t.Errorf("shared mount must not have ,Z relabel suffix, got %q", val)
			}
		}
	}
}

func TestBuildRunArgsNonSharedMountRelabel(t *testing.T) {
	// Shared=false (default, tool-owned dir): must render :ro,Z.
	args := buildRunArgs(runParams{
		containerName: "llmkit-owned",
		workspace:     "/tmp/ws",
		image:         "img",
		network:       "none",
		cmd:           []string{"true"},
		roMounts: []ROMount{
			{HostPath: "/home/user/.cache/llmkit/modcache/abc123", ContainerPath: "/modcache", Shared: false},
		},
	})
	mustContainSeq(t, args, "-v", "/home/user/.cache/llmkit/modcache/abc123:/modcache:ro,Z")
}

func TestBuildRunArgsRendersWritableMounts(t *testing.T) {
	args := buildRunArgs(runParams{
		containerName: "llmkit-rw",
		workspace:     "/tmp/ws",
		image:         "img",
		network:       "bridge",
		cmd:           []string{"go", "mod", "download"},
		rwMounts: []ROMount{
			// Tool-managed prefetch cache: private, gets the :Z relabel.
			{HostPath: "/host/cache", ContainerPath: "/modcache"},
			// Operator-configured writable mount: host-owned
			// (Shared=true), must NOT be relabeled — a private container
			// context would break host-side management of the same tree.
			{HostPath: "/data/vendor", ContainerPath: "/bazel-vendor", Shared: true},
		},
	})
	mustContainSeq(t, args, "-v", "/host/cache:/modcache:rw,Z")
	mustContainSeq(t, args, "-v", "/data/vendor:/bazel-vendor:rw")
	mustContainSeq(t, args, "--network=bridge")
}

func TestValidateMounts(t *testing.T) {
	tests := []struct {
		name    string
		ro, rw  []ROMount
		wantErr bool
	}{
		{"empty", nil, nil, false},
		{"valid ro", []ROMount{{HostPath: "/h", ContainerPath: "/c"}}, nil, false},
		{"valid rw", nil, []ROMount{{HostPath: "/h", ContainerPath: "/c"}}, false},
		{"empty host", []ROMount{{HostPath: "", ContainerPath: "/c"}}, nil, true},
		{"empty ctr", []ROMount{{HostPath: "/h", ContainerPath: ""}}, nil, true},
		{"relative host", []ROMount{{HostPath: "rel", ContainerPath: "/c"}}, nil, true},
		{"relative ctr", []ROMount{{HostPath: "/h", ContainerPath: "rel"}}, nil, true},
		{"dup within ro", []ROMount{{HostPath: "/a", ContainerPath: "/c"}, {HostPath: "/b", ContainerPath: "/c"}}, nil, true},
		{"dup across ro/rw", []ROMount{{HostPath: "/a", ContainerPath: "/c"}}, []ROMount{{HostPath: "/b", ContainerPath: "/c"}}, true},
		{"dup by trailing slash", []ROMount{{HostPath: "/a", ContainerPath: "/opt/a"}}, []ROMount{{HostPath: "/b", ContainerPath: "/opt/a/"}}, true},
		{"dup by dot element", []ROMount{{HostPath: "/a", ContainerPath: "/opt/a"}, {HostPath: "/b", ContainerPath: "/opt/./a"}}, nil, true},
		{"dup by double slash and dotdot", []ROMount{{HostPath: "/a", ContainerPath: "/opt//a"}, {HostPath: "/b", ContainerPath: "/opt/x/../a"}}, nil, true},
		{"container root", []ROMount{{HostPath: "/h", ContainerPath: "/"}}, nil, true},
		{"container root by dotdot", nil, []ROMount{{HostPath: "/h", ContainerPath: "/opt/.."}}, true},
		{"workspace", []ROMount{{HostPath: "/h", ContainerPath: WorkspaceMount}}, nil, true},
		{"workspace with trailing slash", nil, []ROMount{{HostPath: "/h", ContainerPath: WorkspaceMount + "/"}}, true},
		{"distinct siblings", []ROMount{{HostPath: "/a", ContainerPath: "/opt/a"}}, []ROMount{{HostPath: "/b", ContainerPath: "/opt/ab"}}, false},
		{"path nested under workspace is a different path", []ROMount{{HostPath: "/h", ContainerPath: WorkspaceMount + "/sub"}}, nil, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMounts(tc.ro, tc.rw)
			if (err != nil) != tc.wantErr {
				t.Errorf("validateMounts(%v,%v) err=%v wantErr=%v", tc.ro, tc.rw, err, tc.wantErr)
			}
		})
	}
}

// TestRemoveArgs pins the removal argv per runtime: podman needs `--time 0`
// to kill at once (its `rm -f` otherwise waits its 10s stop timeout), and
// docker has no such flag on `rm` (exit 125, unknown flag).
func TestRemoveArgs(t *testing.T) {
	tests := []struct {
		runtime string
		want    []string
	}{
		{"podman", []string{"rm", "-f", "--time", "0", "llmkit-x"}},
		{"/usr/bin/podman", []string{"rm", "-f", "--time", "0", "llmkit-x"}},
		{"docker", []string{"rm", "-f", "llmkit-x"}},
		{"/usr/local/bin/docker", []string{"rm", "-f", "llmkit-x"}},
	}
	for _, tc := range tests {
		if got := removeArgs(tc.runtime, "llmkit-x"); !slices.Equal(got, tc.want) {
			t.Errorf("removeArgs(%q) = %q, want %q", tc.runtime, got, tc.want)
		}
	}
}

// mustContainSeq asserts that the contiguous subsequence seq appears in args.
func mustContainSeq(t *testing.T, args []string, seq ...string) {
	t.Helper()
	for i := 0; i+len(seq) <= len(args); i++ {
		if slices.Equal(args[i:i+len(seq)], seq) {
			return
		}
	}
	t.Errorf("expected args to contain sequence %q; got %q", seq, args)
}

func TestShellQuote(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		// Plain arguments need no escaping beyond the outer quotes.
		{"hello", "'hello'"},
		{"hello world", "'hello world'"},
		// Embedded single quote: close, escape, reopen.
		{"it's", "'it'\\''s'"},
		// Dollar sign must not be expanded (stays inside single quotes).
		{"$HOME", "'$HOME'"},
		// Semicolon injection attempt: harmless inside single quotes.
		{"; rm -rf /", "'; rm -rf /'"},
		// Empty string: two adjacent single quotes.
		{"", "''"},
		// Multiple single quotes in a row.
		{"a'b'c", "'a'\\''b'\\''c'"},
		// Newline inside arg.
		{"foo\nbar", "'foo\nbar'"},
		// Backslash: no special meaning inside single quotes.
		{`a\b`, `'a\b'`},
		// Command substitution attempts: inert inside single quotes.
		{"$(rm -rf /)", "'$(rm -rf /)'"},
		{"`rm -rf /`", "'`rm -rf /`'"},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got := shellQuote(tc.in)
			if got != tc.want {
				t.Errorf("shellQuote(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestBuildSetupScript(t *testing.T) {
	cmds := [][]string{
		{"npm", "ci", "--offline"},
		{"echo", "hello world"},
	}
	script := buildSetupScript(cmds)
	if !strings.Contains(script, "'npm' 'ci' '--offline' || exit 125") {
		t.Errorf("script missing npm cmd with exit 125: %q", script)
	}
	if !strings.Contains(script, "'echo' 'hello world' || exit 125") {
		t.Errorf("script missing echo cmd with exit 125: %q", script)
	}
	if !strings.HasSuffix(script, `exec "$@"`) {
		t.Errorf("script does not end with exec \"$@\": %q", script)
	}

	// An empty argv must be skipped entirely: rendering it would produce a
	// bare "|| exit 125" line, which sh treats as a successful no-op — a
	// silently dead guard.
	withEmpty := buildSetupScript([][]string{{}, {"true"}})
	if strings.Contains(withEmpty, "\n || exit 125") || strings.HasPrefix(withEmpty, " || exit 125") {
		t.Errorf("empty argv must be skipped, got %q", withEmpty)
	}
	if !strings.Contains(withEmpty, "'true' || exit 125") {
		t.Errorf("non-empty argv after an empty one must still render, got %q", withEmpty)
	}
}

// setupScriptShells returns the POSIX shells found on this host that can run
// the rendered wrapper: argv prefixes that take `-c <script> sh <cmd...>`.
// An absent shell is skipped, so a host without dash or busybox runs the rest.
func setupScriptShells(t *testing.T) map[string][]string {
	t.Helper()
	shells := map[string][]string{}
	for _, name := range []string{"bash", "dash", "sh"} {
		if p, err := exec.LookPath(name); err == nil {
			shells[name] = []string{p}
		}
	}
	if p, err := exec.LookPath("busybox"); err == nil {
		shells["busybox"] = []string{p, "sh"}
	}
	if len(shells) == 0 {
		t.Skip("no POSIX shell on PATH")
	}
	return shells
}

// runSetupScript runs the rendered wrapper the way both backends do —
// `<shell> -c <script> sh <cmd...>` — with PATH set to pathDir plus the
// system dirs, and returns stdout and the shell's exit code (-1: killed by a
// signal).
func runSetupScript(t *testing.T, shell []string, setup [][]string, pathDir string, cmd ...string) (string, int) {
	t.Helper()
	argv := append(slices.Clone(shell[1:]), "-c", buildSetupScript(setup), "sh")
	c := exec.Command(shell[0], append(argv, cmd...)...)
	c.Env = []string{"PATH=" + pathDir + ":/usr/bin:/bin"}
	out, err := c.Output()
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			t.Fatalf("run %v: %v", shell, err)
		}
		return string(out), ee.ExitCode()
	}
	return string(out), 0
}

// TestBuildSetupScriptRunsCmdAsName pins that the wrapper runs Spec.Cmd[0]
// as a command name under every shell on the host, never as an exec option
// (`exec -a true /bin/false` runs /bin/false under bash), while a Cmd[0]
// that does not start with '-' still gets exec's exit code and signal
// disposition.
func TestBuildSetupScriptRunsCmdAsName(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX sh wrapper")
	}
	// A command literally named "-dash-cmd" that reports its arguments.
	bin := t.TempDir()
	dashCmd := filepath.Join(bin, "-dash-cmd")
	if err := os.WriteFile(dashCmd, []byte("#!/bin/sh\nprintf '<%s>' \"$@\"\nexit 7\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	sigCmd := filepath.Join(bin, "-sig-cmd")
	if err := os.WriteFile(sigCmd, []byte("#!/bin/sh\nkill -TERM $$\nsleep 5\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	for name, shell := range setupScriptShells(t) {
		t.Run(name, func(t *testing.T) {
			for _, setup := range [][][]string{nil, {{"true"}}} {
				// -a would be an exec option, "true" the exec'd
				// name, /bin/false its argument. As a command name, "-a" is
				// not found.
				if _, code := runSetupScript(t, shell, setup, bin, "-a", "true", "/bin/false"); code != 127 {
					t.Errorf("setup=%v Cmd [-a true /bin/false]: exit %d, want 127", setup, code)
				}
				for _, opt := range []string{"-c", "-l", "-", "--", "-x/y"} {
					if _, code := runSetupScript(t, shell, setup, bin, opt, "true"); code != 127 {
						t.Errorf("setup=%v Cmd [%s true]: exit %d, want 127", setup, opt, code)
					}
				}
				// A command that really is named with a leading '-' runs, its
				// arguments intact and its exit code forwarded.
				out, code := runSetupScript(t, shell, setup, bin, "-dash-cmd", "a b", "-c")
				if out != "<a b><-c>" || code != 7 {
					t.Errorf("setup=%v Cmd [-dash-cmd]: out %q exit %d, want <a b><-c> and 7", setup, out, code)
				}
				// A '-' command killed by signal N is reported as exit 128+N.
				if _, code := runSetupScript(t, shell, setup, bin, "-sig-cmd"); code != 128+15 {
					t.Errorf("setup=%v Cmd [-sig-cmd]: exit %d, want 143", setup, code)
				}
				// Any other command keeps exec semantics: its exit code, and
				// its own signal death (the wrapper is replaced, not a
				// parent that turns the signal into an exit code).
				if _, code := runSetupScript(t, shell, setup, bin, "sh", "-c", "exit 3"); code != 3 {
					t.Errorf("setup=%v Cmd [sh -c exit 3]: exit %d, want 3", setup, code)
				}
				if _, code := runSetupScript(t, shell, setup, bin, "sh", "-c", "kill -TERM $$"); code != -1 {
					t.Errorf("setup=%v Cmd [sh -c kill]: exit %d, want -1 (signal death)", setup, code)
				}
				if _, code := runSetupScript(t, shell, setup, bin, "no-such-cmd-xyz"); code != 127 {
					t.Errorf("setup=%v Cmd [no-such-cmd-xyz]: exit %d, want 127", setup, code)
				}
			}
			// A failing setup command still ends the wrapper with 125,
			// whichever kind of Cmd[0] follows.
			for _, cmd := range []string{"-a", "true"} {
				if _, code := runSetupScript(t, shell, [][]string{{"false"}}, bin, cmd, "true"); code != 125 {
					t.Errorf("failing setup, Cmd[0] %q: exit %d, want 125", cmd, code)
				}
			}
		})
	}
}

func TestBuildRunArgsSetupCmds(t *testing.T) {
	// When SetupCmds are present, the argv must use /bin/sh -c <script> sh
	// followed by the original command (as positional params to exec "$@").
	args := buildRunArgs(runParams{
		containerName: "llmkit-setup",
		workspace:     "/tmp/ws",
		image:         "node:20-alpine",
		network:       "none",
		cmd:           []string{"node", "--version"},
		setupCmds: [][]string{
			{"npm", "ci", "--offline"},
		},
	})

	// /bin/sh -c <script> sh must appear before the original command.
	shIdx := slices.Index(args, "/bin/sh")
	if shIdx < 0 {
		t.Fatalf("expected /bin/sh in args; got %q", args)
	}
	if shIdx+1 >= len(args) || args[shIdx+1] != "-c" {
		t.Fatalf("expected -c after /bin/sh; got %q", args)
	}
	// The script arg follows -c.
	script := args[shIdx+2]
	if !strings.Contains(script, "'npm' 'ci' '--offline' || exit 125") {
		t.Errorf("script missing npm cmd: %q", script)
	}
	if !strings.HasSuffix(script, `exec "$@"`) {
		t.Errorf("script missing exec trailer: %q", script)
	}
	// The sh $0 placeholder follows the script.
	if shIdx+3 >= len(args) || args[shIdx+3] != "sh" {
		t.Fatalf("expected sh ($0) after script; got %q", args)
	}
	// The original command must appear at the tail.
	tail := args[len(args)-2:]
	if !slices.Equal(tail, []string{"node", "--version"}) {
		t.Errorf("original cmd not at tail; got %q", tail)
	}
}

func TestBuildRunArgsNoSetupCmds(t *testing.T) {
	// When SetupCmds is empty, no /bin/sh wrapping occurs and the cmd
	// appears directly after the image — existing behavior is unchanged.
	args := buildRunArgs(runParams{
		containerName: "llmkit-plain",
		workspace:     "/tmp/ws",
		image:         "golang:1.23",
		network:       "none",
		cmd:           []string{"go", "test", "./..."},
	})
	if slices.Contains(args, "/bin/sh") {
		t.Errorf("/bin/sh must not appear when SetupCmds is empty; got %q", args)
	}
	tail := args[len(args)-3:]
	if !slices.Equal(tail, []string{"go", "test", "./..."}) {
		t.Errorf("cmd not at tail; got %q", tail)
	}
}

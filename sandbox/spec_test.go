package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// spec_test.go pins the ONE spec-admission contract (bead llmkit-bk8.7.1 /
// llmkit-bk8.1.8 wave): every backend — the Mock included — refuses a
// malformed Spec with *InvalidSpecError naming the offending field, a
// backend that cannot honor a well-formed field refuses with
// *UnsupportedSpecError, and every refusal happens BEFORE any filesystem
// effect (no temp dirs, no workspace writes).

// execFor returns the backend named by the honor matrix through its
// constructor or zero value. CLI and Bwrap are zero-value structs: Spec
// admission (validateSpec) and the workspace step precede any runtime or
// bwrap contact, so refused Specs never touch a runtime (hermetic).
func execFor(t *testing.T, backend string) Sandbox {
	t.Helper()
	switch backend {
	case backendCLI:
		return &CLI{}
	case backendBwrap:
		return &Bwrap{}
	case backendHost:
		return NewHostExec()
	case backendMock:
		return NewMock(MockResponse{Result: Result{ExitCode: 0}})
	default:
		t.Fatalf("unknown backend %q", backend)
		return nil
	}
}

// allBackends is every Exec-implementing backend, in matrix-column order.
var allBackends = []string{backendCLI, backendBwrap, backendHost, backendMock}

// universalInvalidRows is the universal-invalid set: each row names the
// Spec field the InvalidSpecError must carry. Each builder returns a Spec
// that fails ONLY that class on every backend.
func universalInvalidRows() []struct {
	name  string
	field string
	spec  func(base Spec) Spec
} {
	return []struct {
		name  string
		field string
		spec  func(base Spec) Spec
	}{
		{"empty Cmd", "Cmd", func(base Spec) Spec { base.Cmd = nil; return base }},
		{"no RepoDir and no Workspace", "RepoDir", func(base Spec) Spec { base.RepoDir = ""; base.Workspace = ""; return base }},
		{"relative Workspace", "Workspace", func(base Spec) Spec { base.Workspace = "rel/ative"; return base }},
		{"escaping WriteFiles key", "WriteFiles", func(base Spec) Spec { base.WriteFiles = map[string][]byte{"../escape.txt": []byte("x")}; return base }},
		{"escaping CaptureFiles entry", "CaptureFiles", func(base Spec) Spec { base.CaptureFiles = []string{"/abs/capture.txt"}; return base }},
		{"relative mount HostPath", "ROMounts", func(base Spec) Spec {
			base.ROMounts = []ROMount{{HostPath: "rel/host", ContainerPath: "/ctr"}}
			return base
		}},
		{"duplicate ContainerPath", "RWMounts", func(base Spec) Spec {
			base.ROMounts = []ROMount{{HostPath: "/host/ro", ContainerPath: "/ctr/dup"}}
			base.RWMounts = []ROMount{{HostPath: "/host/rw", ContainerPath: "/ctr/dup"}}
			return base
		}},
		{"Env entry without =", "Env", func(base Spec) Spec { base.Env = []string{"HOSTONLYVAR"}; return base }},
		{"Env entry with an empty key", "Env", func(base Spec) Spec { base.Env = []string{"=VALUE"}; return base }},
	}
}

// baseValidSpec is well-formed for every backend.
func baseValidSpec() Spec {
	return Spec{
		RepoDir: "/repo",
		Cmd:     []string{"/bin/sh", "-c", "exit 0"},
	}
}

// TestValidateSpecUniversalInvalidOnEveryBackend runs each universal-invalid
// class through every backend's Exec and pins the typed refusal
// (S0-2 part ii).
func TestValidateSpecUniversalInvalidOnEveryBackend(t *testing.T) {
	for _, row := range universalInvalidRows() {
		for _, backend := range allBackends {
			t.Run(row.name+"/"+backend, func(t *testing.T) {
				sb := execFor(t, backend)
				_, err := sb.Exec(context.Background(), row.spec(baseValidSpec()))
				var inv *InvalidSpecError
				if !errors.As(err, &inv) {
					t.Fatalf("Exec err = %v, want *InvalidSpecError", err)
				}
				if inv.Field != row.field {
					t.Errorf("Field = %q, want %q (err=%v)", inv.Field, row.field, err)
				}
			})
		}
	}
}

// TestRefusedSpecHasNoFilesystemEffect pins S0-1: every refusal happens
// before ANY filesystem effect — no temp directory is created (TMPDIR is
// pointed at an empty dir) and no WriteFiles marker lands in the workspace.
// Runs on all four backends hermetically: HostExec and Mock via their
// constructors; CLI and Bwrap via the zero-value structs. Bwrap's cap-method
// detection is stubbed to "none" (a host with neither systemd-run >= 254
// nor a delegated cgroup): the Workspace refusals must still win, because
// the supervisor checks the Workspace before the backend's admission.
func TestRefusedSpecHasNoFilesystemEffect(t *testing.T) {
	stubCapMethod(t, bwrapCapNone)
	empty := t.TempDir()
	t.Setenv("TMPDIR", empty)

	run := func(t *testing.T, backend string, spec Spec) {
		t.Helper()
		sb := execFor(t, backend)
		res, err := sb.Exec(context.Background(), spec)
		if err == nil {
			t.Fatalf("%s: Exec err = nil (res=%+v), want a typed refusal", backend, res)
		}
		var inv *InvalidSpecError
		var uns *UnsupportedSpecError
		if !errors.As(err, &inv) && !errors.As(err, &uns) {
			t.Fatalf("%s: err = %v, want *InvalidSpecError or *UnsupportedSpecError", backend, err)
		}
		if entries, readErr := os.ReadDir(empty); readErr != nil || len(entries) != 0 {
			t.Fatalf("%s: TMPDIR gained entries %v (readErr=%v) — the refusal must precede every write", backend, entries, readErr)
		}
		if spec.Workspace != "" {
			if _, statErr := os.Stat(filepath.Join(spec.Workspace, "marker")); !os.IsNotExist(statErr) {
				t.Fatalf("%s: workspace marker exists (statErr=%v) — the refusal must precede WriteFiles", backend, statErr)
			}
		}
	}

	// Backend-specific refusals, each with a workspace + WriteFiles marker
	// that must never be written.
	ws := t.TempDir()
	unsupported := []struct {
		backend string
		field   string
		spec    Spec
	}{
		{backendBwrap, "Image", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, Image: "quay.io/example/img:latest"}},
		{backendHost, "Image", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, Image: "quay.io/example/img:latest"}},
		{backendHost, "ROMounts", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, ROMounts: []ROMount{{HostPath: "/host", ContainerPath: "/ctr"}}}},
		{backendHost, "RWMounts", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, RWMounts: []ROMount{{HostPath: "/host", ContainerPath: "/ctr"}}}},
		{backendHost, "SetupCmds", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, SetupCmds: [][]string{{"npm", "ci"}}}},
		{backendCLI, "Network", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, Network: NetworkMode("bogus")}},
		{backendBwrap, "Network", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, Network: NetworkBridge}},
		{backendHost, "Network", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, Network: NetworkNone}},
		{backendBwrap, "ROMounts", Spec{RepoDir: "/repo", Workspace: ws, Cmd: []string{"true"}, ROMounts: []ROMount{{HostPath: "/host", ContainerPath: "/usr"}}}},
	}
	for _, row := range unsupported {
		t.Run("unsupported/"+row.backend+"/"+row.field, func(t *testing.T) {
			row.spec.WriteFiles = map[string][]byte{"marker": []byte("x")}
			run(t, row.backend, row.spec)
			// The typed error names the backend and field.
			sb := execFor(t, row.backend)
			_, err := sb.Exec(context.Background(), row.spec)
			var uns *UnsupportedSpecError
			if !errors.As(err, &uns) || uns.Backend != row.backend || uns.Field != row.field {
				t.Fatalf("err = %v, want UnsupportedSpecError{backend=%s field=%s}", err, row.backend, row.field)
			}
		})
	}

	// Universal-invalid refusals on all four backends, same no-write rule.
	for _, urow := range universalInvalidRows() {
		for _, backend := range allBackends {
			t.Run("universal/"+backend+"/"+urow.name, func(t *testing.T) {
				spec := urow.spec(baseValidSpec())
				spec.Workspace = ws
				spec.WriteFiles = map[string][]byte{"marker": []byte("x")}
				// Re-apply the mutation that may have been overwritten by
				// the WriteFiles/Workspace assignment above.
				spec = urow.spec(spec)
				run(t, backend, spec)
			})
		}
	}

	// The supervisor's workspace checks refuse before any write too: a
	// nonexistent, non-directory, or non-searchable Workspace is
	// InvalidSpecError{Field: "Workspace"} on the three real backends (the
	// Mock never stats). wsRoot is created here, by the parent test, so it
	// lives outside the watched TMPDIR.
	wsRoot := t.TempDir()
	for _, backend := range []string{backendCLI, backendBwrap, backendHost} {
		t.Run("workspace-missing/"+backend, func(t *testing.T) {
			spec := baseValidSpec()
			spec.Workspace = filepath.Join(t.TempDir(), "does-not-exist")
			sb := execFor(t, backend)
			_, err := sb.Exec(context.Background(), spec)
			var inv *InvalidSpecError
			if !errors.As(err, &inv) || inv.Field != "Workspace" {
				t.Fatalf("%s: err = %v, want InvalidSpecError{Field: Workspace}", backend, err)
			}
		})
		t.Run("workspace-not-a-dir/"+backend, func(t *testing.T) {
			file := filepath.Join(t.TempDir(), "file")
			if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
			spec := baseValidSpec()
			spec.Workspace = file
			sb := execFor(t, backend)
			_, err := sb.Exec(context.Background(), spec)
			var inv *InvalidSpecError
			if !errors.As(err, &inv) || inv.Field != "Workspace" {
				t.Fatalf("%s: err = %v, want InvalidSpecError{Field: Workspace}", backend, err)
			}
		})
		t.Run("workspace-not-searchable/"+backend, func(t *testing.T) {
			if os.Geteuid() == 0 {
				t.Skip("root searches any directory; the refusal is unreachable")
			}
			ws, err := os.MkdirTemp(wsRoot, "ws")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(ws, 0); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = os.Chmod(ws, 0o700) })
			spec := baseValidSpec()
			spec.Workspace = ws
			spec.Cmd = []string{"/bin/true"}
			sb := execFor(t, backend)
			// Without WriteFiles the child's chdir would be the first
			// thing to fail (HostExec used to report it as exit 126).
			res, err := sb.Exec(context.Background(), spec)
			var inv *InvalidSpecError
			if !errors.As(err, &inv) || inv.Field != "Workspace" {
				t.Fatalf("%s: err = %v (res=%+v), want InvalidSpecError{Field: Workspace}", backend, err, res)
			}
			// With a marker: still refused, and nothing written anywhere.
			spec.WriteFiles = map[string][]byte{"marker": []byte("x")}
			_, err = sb.Exec(context.Background(), spec)
			if !errors.As(err, &inv) || inv.Field != "Workspace" {
				t.Fatalf("%s with WriteFiles: err = %v, want InvalidSpecError{Field: Workspace}", backend, err)
			}
			if err := os.Chmod(ws, 0o700); err != nil {
				t.Fatal(err)
			}
			if _, statErr := os.Stat(filepath.Join(ws, "marker")); !os.IsNotExist(statErr) {
				t.Fatalf("%s: workspace marker exists (statErr=%v)", backend, statErr)
			}
			if entries, readErr := os.ReadDir(empty); readErr != nil || len(entries) != 0 {
				t.Fatalf("%s: TMPDIR gained entries %v (readErr=%v)", backend, entries, readErr)
			}
		})
	}
}

// TestValidateSpecPure pins the admission order contract on the pure
// validator directly: universal classes are checked before backend-specific
// ones, so a Spec with both problems reports the universal one.
func TestValidateSpecPure(t *testing.T) {
	// Universal wins over backend-specific: empty Cmd on bwrap with an
	// Image set reports Cmd.
	err := validateSpec(backendBwrap, Spec{Image: "img"})
	var inv *InvalidSpecError
	if !errors.As(err, &inv) || inv.Field != "Cmd" {
		t.Fatalf("err = %v, want InvalidSpecError{Cmd}", err)
	}

	// The validator never touches the filesystem: RepoDir need not exist.
	if err := validateSpec(backendCLI, baseValidSpec()); err != nil {
		t.Fatalf("validateSpec(valid) = %v, want nil", err)
	}

	// Backend-specific: bwrap fixed-bind collision, network sets.
	err = validateSpec(backendBwrap, Spec{Cmd: []string{"true"}, RepoDir: "/repo", ROMounts: []ROMount{{HostPath: "/host", ContainerPath: "/usr"}}})
	var uns *UnsupportedSpecError
	if !errors.As(err, &uns) || uns.Backend != backendBwrap {
		t.Fatalf("err = %v, want bwrap UnsupportedSpecError for the fixed-bind collision", err)
	}

	// The Mock has no backend-specific refusals: everything well-formed
	// records.
	if err := validateSpec(backendMock, Spec{Cmd: []string{"true"}, RepoDir: "/repo", Image: "img", ROMounts: []ROMount{{HostPath: "/h", ContainerPath: "/c"}}, SetupCmds: [][]string{{"npm"}}, Network: NetworkBridge, Env: []string{"A=B"}}); err != nil {
		t.Fatalf("validateSpec(mock, well-formed) = %v, want nil", err)
	}

	// Error text shape: "sandbox: invalid Spec.<Field>: <Reason>".
	err = validateSpec(backendMock, Spec{})
	if got := err.Error(); !strings.HasPrefix(got, "sandbox: invalid Spec.") {
		t.Fatalf("Error() = %q, want the sandbox: invalid Spec.<Field> shape", got)
	}
}

// --- S0-2: the ONE honor matrix, tied to docs/sandbox.md -----------------

// matrixCell is one Spec-field x backend cell: the class the code exhibits
// for a representative WELL-FORMED value of the field.
type matrixCell struct {
	field string
	// value fills the field into a base Spec with a well-formed
	// representative.
	value func(base Spec) Spec
	// class per backend: "honored" (validateSpec admits it),
	// "refused" (Exec returns *UnsupportedSpecError), "recorded" (the Mock
	// stores it; validateSpec admits it).
	class map[string]string
}

// honorMatrix is the code side of docs/sandbox.md's "Spec-field honor
// matrix". The test below parses the same table out of the docs and fails
// when the two disagree, so a docs edit and a behavior change cannot drift
// apart.
func honorMatrix() []matrixCell {
	abs := func(base Spec) Spec { return base }
	return []matrixCell{
		{"RepoDir", abs, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"Workspace", func(base Spec) Spec { base.Workspace = "/abs/ws"; return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"Cmd", abs, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"Env", func(base Spec) Spec { base.Env = []string{"K=V"}; return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"Image", func(base Spec) Spec { base.Image = "example.com/repo/img:1"; return base }, map[string]string{backendCLI: "honored", backendBwrap: "refused", backendHost: "refused", backendMock: "recorded"}},
		{"Timeout", func(base Spec) Spec { return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"Network", func(base Spec) Spec { return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"WriteFiles", func(base Spec) Spec { base.WriteFiles = map[string][]byte{"out.txt": []byte("x")}; return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
		{"ROMounts", func(base Spec) Spec {
			base.ROMounts = []ROMount{{HostPath: "/host/a", ContainerPath: "/ctr/a"}}
			return base
		}, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "refused", backendMock: "recorded"}},
		{"RWMounts", func(base Spec) Spec {
			base.RWMounts = []ROMount{{HostPath: "/host/b", ContainerPath: "/ctr/b"}}
			return base
		}, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "refused", backendMock: "recorded"}},
		{"SetupCmds", func(base Spec) Spec { base.SetupCmds = [][]string{{"npm", "ci", "--offline"}}; return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "refused", backendMock: "recorded"}},
		{"CaptureFiles", func(base Spec) Spec { base.CaptureFiles = []string{"out.txt"}; return base }, map[string]string{backendCLI: "honored", backendBwrap: "honored", backendHost: "honored", backendMock: "recorded"}},
	}
}

// docColumnOrder is docs/sandbox.md's column order for the matrix table.
var docColumnOrder = []string{"CLI", "Bwrap", "HostExec", "Mock"}

// parseDocsHonorMatrix parses the "Spec-field honor matrix" markdown table
// in ../docs/sandbox.md (go test's cwd is the package directory).
func parseDocsHonorMatrix(t *testing.T) map[string]map[string]string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "docs", "sandbox.md"))
	if err != nil {
		t.Fatalf("read docs: %v", err)
	}
	m, err := parseHonorMatrix(string(raw))
	if err != nil {
		t.Fatalf("docs/sandbox.md: %v", err)
	}
	return m
}

// parseHonorMatrix parses the honor-matrix markdown table in doc into
// field -> docs column -> class, deriving each cell's class from the cell
// text's leading word (honored / refused / recorded). The header row must
// name exactly docColumnOrder (a header edit alone would otherwise
// reattribute every column), and each field may appear once (a second,
// contradicting row would otherwise override the first).
func parseHonorMatrix(doc string) (map[string]map[string]string, error) {
	splitRow := func(trimmed string) []string {
		cells := strings.Split(strings.Trim(trimmed, "|"), "|")
		for i := range cells {
			cells[i] = strings.TrimSpace(cells[i])
		}
		return cells
	}
	inMatrix := false
	out := map[string]map[string]string{}
	for _, line := range strings.Split(doc, "\n") {
		trimmed := strings.TrimSpace(line)
		if !inMatrix {
			if strings.HasPrefix(trimmed, "| Spec field |") {
				header := splitRow(trimmed)
				if !slices.Equal(header[1:], docColumnOrder) {
					return nil, fmt.Errorf("honor matrix header columns = %q, want %q", header[1:], docColumnOrder)
				}
				inMatrix = true
			}
			continue
		}
		if !strings.HasPrefix(trimmed, "|") {
			if len(out) == 0 {
				return nil, errors.New("honor matrix table is empty")
			}
			return out, nil
		}
		cells := splitRow(trimmed)
		if strings.HasPrefix(cells[0], "-") {
			continue // separator
		}
		field := strings.Trim(cells[0], "`")
		if len(cells) != 1+len(docColumnOrder) {
			return nil, fmt.Errorf("honor matrix row %q has %d cells, want %d", field, len(cells), 1+len(docColumnOrder))
		}
		if _, dup := out[field]; dup {
			return nil, fmt.Errorf("honor matrix row %q appears more than once", field)
		}
		out[field] = map[string]string{}
		for i, col := range docColumnOrder {
			switch cell := strings.ToLower(cells[i+1]); {
			case strings.HasPrefix(cell, "honored"):
				out[field][col] = "honored"
			case strings.HasPrefix(cell, "refused"):
				out[field][col] = "refused"
			case strings.HasPrefix(cell, "recorded"):
				out[field][col] = "recorded"
			default:
				return nil, fmt.Errorf("honor matrix cell %q has no honored/refused/recorded leading word", cells[i+1])
			}
		}
	}
	return nil, errors.New("honor matrix table never ends")
}

// TestParseHonorMatrixRejectsDrift pins the parser against the two docs
// edits that would silently misattribute cells: a header whose columns are
// reordered while the cells stay put, and a second, contradicting row for
// a field.
func TestParseHonorMatrixRejectsDrift(t *testing.T) {
	const rows = "|---|---|---|---|---|\n" +
		"| `ROMounts` | honored | honored | refused | recorded |\n"
	good := "| Spec field | CLI | Bwrap | HostExec | Mock |\n" + rows + "\n"
	if m, err := parseHonorMatrix(good); err != nil || m["ROMounts"]["HostExec"] != "refused" {
		t.Fatalf("well-formed table: m=%v err=%v, want ROMounts/HostExec refused", m, err)
	}
	swappedHeader := "| Spec field | CLI | HostExec | Bwrap | Mock |\n" + rows + "\n"
	if _, err := parseHonorMatrix(swappedHeader); err == nil {
		t.Error("a header with HostExec and Bwrap swapped (cells unchanged) parsed; it must be refused")
	}
	duplicate := "| Spec field | CLI | Bwrap | HostExec | Mock |\n" +
		"|---|---|---|---|---|\n" +
		"| `ROMounts` | honored | honored | honored | recorded |\n" +
		"| `ROMounts` | honored | honored | refused | recorded |\n\n"
	if _, err := parseHonorMatrix(duplicate); err == nil {
		t.Error("a table with a repeated ROMounts row parsed; it must be refused")
	}
}

// docsBackendName maps a docs column to the backend constant.
func docsBackendName(col string) string {
	switch col {
	case "CLI":
		return backendCLI
	case "Bwrap":
		return backendBwrap
	case "HostExec":
		return backendHost
	case "Mock":
		return backendMock
	}
	panic("unknown docs column " + col)
}

// TestHonorMatrixMatchesDocs pins S0-2 part (i): for every Spec field x
// backend, the class the CODE exhibits for a representative well-formed
// value must equal the class the DOCS table declares. Honored/recorded
// cells are exercised against validateSpec directly (nil expected);
// refused cells go through Exec and must produce *UnsupportedSpecError
// naming the backend and field.
func TestHonorMatrixMatchesDocs(t *testing.T) {
	docs := parseDocsHonorMatrix(t)

	byField := map[string]matrixCell{}
	for _, cell := range honorMatrix() {
		byField[cell.field] = cell
	}
	// Every docs row must exist in the code table and vice versa.
	for field := range docs {
		if _, ok := byField[field]; !ok {
			t.Errorf("docs matrix row %q has no code-side expectation", field)
		}
	}
	for field := range byField {
		if _, ok := docs[field]; !ok {
			t.Errorf("code matrix row %q is missing from docs/sandbox.md", field)
		}
	}
	if t.Failed() {
		t.FailNow()
	}

	for _, cell := range honorMatrix() {
		for _, col := range docColumnOrder {
			backend := docsBackendName(col)
			want := cell.class[backend]
			got := docs[cell.field][col]
			if got != want {
				t.Errorf("docs cell [%s][%s] = %q, code behaves %q", cell.field, col, got, want)
				continue
			}
			spec := cell.value(baseValidSpec())
			switch want {
			case "honored", "recorded":
				if err := validateSpec(backend, spec); err != nil {
					t.Errorf("validateSpec(%s, %s) = %v, want nil (docs says %q)", backend, cell.field, err, want)
				}
			case "refused":
				sb := execFor(t, backend)
				_, err := sb.Exec(context.Background(), spec)
				var uns *UnsupportedSpecError
				if !errors.As(err, &uns) || uns.Backend != backend || uns.Field != cell.field {
					t.Errorf("Exec(%s, %s set) = %v, want UnsupportedSpecError{backend=%s field=%s}", backend, cell.field, err, backend, cell.field)
				}
			}
		}
	}
}

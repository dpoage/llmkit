package examples_test

// Hermetic tests for the store composition: examples/agent --store records
// with the store as the run's only sink, examples/replay --store replays it,
// examples/audit answers the cookbook queries, and the doc contracts hold
// (docs/store.md quotes the package examples and the cookbook verbatim; the
// AGENTS.md and CLAUDE.md tails stay byte-identical and carry the
// store/sqlite row). No build tag, no credentials, no network beyond the
// httptest server.

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/store/sqlite"
)

// storeRun runs the agent binary against srv with --store and returns the
// database path and the run id it printed.
func storeRun(t *testing.T, agentBin, srvURL string) (db, runID string) {
	t.Helper()
	dir := t.TempDir()
	db = filepath.Join(dir, "run.db")
	code, stdout, stderr := runExampleArgs(t, agentBin, []string{"--store", db}, map[string]string{
		"LLMKIT_PROVIDER": "openai-compatible",
		"LLMKIT_MODEL":    "m",
		"LLMKIT_API_KEY":  "k",
		"LLMKIT_BASE_URL": srvURL + "/v1",
	})
	if code != 0 {
		t.Fatalf("agent exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	m := runIDLine.FindStringSubmatch(stdout)
	if m == nil {
		t.Fatalf("agent printed no run id line:\n%s", stdout)
	}
	if _, err := os.Stat(db); err != nil {
		t.Fatalf("store database: %v\nstdout:\n%s\nstderr:\n%s", err, stdout, stderr)
	}
	return db, m[1]
}

// runAudit executes the audit binary with no environment and returns its
// stdout and exit code.
func runAudit(t *testing.T, bin, db, query string, extra ...string) (string, int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, append([]string{db, query}, extra...)...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	err := cmd.Run()
	if err == nil {
		return stdout.String(), 0
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok || ctx.Err() != nil {
		t.Fatalf("run audit %s %v: %v\nstdout:\n%s\nstderr:\n%s", query, extra, err, stdout.String(), stderr.String())
	}
	return stdout.String(), exitErr.ExitCode()
}

// seedRun stores a Start and a Finalize event for id through the store's
// public surface, so the runs and lineage views have a row.
func seedRun(t *testing.T, db string, id, parent llmkit.RunID, at time.Time, status llmkit.RunStatus) {
	t.Helper()
	st, err := sqlite.Open(db, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = st.Close() }()
	ctx := context.Background()
	st.Observe(ctx, llmkit.Event{
		Kind: llmkit.KindStart, RunID: id, ParentRunID: parent, Time: at,
		SchemaVersion: llmkit.EventSchemaVersion,
		Start:         &llmkit.StartEvent{Task: "seeded"},
	})
	st.Observe(ctx, llmkit.Event{
		Kind: llmkit.KindFinalize, RunID: id, Time: at.Add(time.Second), Step: 1,
		SchemaVersion: llmkit.EventSchemaVersion,
		Finalize:      &llmkit.FinalizeEvent{Status: status, Usage: llmkit.Usage{InputTokens: 1, OutputTokens: 1}},
	})
}

func TestExamplesAgentStoreIsTheOnlySinkAndReplays(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	root := moduleRoot(t)
	tmp := t.TempDir()
	agentBin := buildExample(t, root, tmp, "agent")
	replayBin := buildExample(t, root, tmp, "replay")

	srv, served := fakeCompat(t, false)
	dir := t.TempDir()
	db := filepath.Join(dir, "run.db")
	code, stdout, stderr := runExampleArgs(t, agentBin, []string{"--store", db}, map[string]string{
		"LLMKIT_PROVIDER": "openai-compatible",
		"LLMKIT_MODEL":    "m",
		"LLMKIT_API_KEY":  "k",
		"LLMKIT_BASE_URL": srv.URL + "/v1",
	})
	if code != 0 {
		t.Fatalf("agent exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	if got := served.Load(); got != 2 {
		t.Fatalf("recording run made %d completions, want 2 (a tool call, then the answer)", got)
	}
	m := runIDLine.FindStringSubmatch(stdout)
	if m == nil {
		t.Fatalf("agent printed no run id line:\n%s", stdout)
	}
	runID := m[1]

	var jsonl []string
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".jsonl") {
			jsonl = append(jsonl, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(jsonl) > 0 {
		t.Fatalf("the store was the only sink, but a .jsonl file appeared: %v", jsonl)
	}

	// Replay with the server gone and no LLMKIT_* variables.
	srv.Close()
	code, stdout, stderr = runExampleArgs(t, replayBin, []string{"--store", db, "--run", runID}, nil)
	if code != 0 {
		t.Fatalf("replay exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	m = finalLine.FindStringSubmatch(stdout)
	if m == nil || m[1] != scriptedAnswer {
		t.Fatalf("replay \"final:\" line = %q, want %q\nstdout:\n%s", m, scriptedAnswer, stdout)
	}
}

func TestExamplesAuditCookbook(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	root := moduleRoot(t)
	tmp := t.TempDir()
	agentBin := buildExample(t, root, tmp, "agent")
	auditBin := buildExample(t, root, tmp, "audit")

	srv, _ := fakeCompat(t, false)
	db, runID := storeRun(t, agentBin, srv.URL)
	parent := llmkit.RunID(runID)

	cases := []struct {
		query  string
		extra  []string
		wantIn string
	}{
		{"Q1", []string{"--since", "0"}, "openai-compatible"},
		{"Q2", []string{"--run", runID}, "add"},
		{"Q3", []string{"--since", "0"}, ""},
		{"Q4", []string{"--since", "0", "--threshold", "0.5"}, ""},
		{"Q5", []string{"--run", runID}, ""},
		{"Q6", []string{"--run", runID}, "finalize"},
		{"Q7", []string{"--since", "0"}, ""},
	}
	for _, c := range cases {
		out, code := runAudit(t, auditBin, db, c.query, c.extra...)
		if code != 0 {
			t.Fatalf("audit %s %v exited %d", c.query, c.extra, code)
		}
		if c.wantIn != "" && !strings.Contains(out, c.wantIn) {
			t.Fatalf("audit %s output lacks %q:\n%s", c.query, c.wantIn, out)
		}
	}

	// Lineage: seed a child run of the stored run, then Q5 must answer in
	// both directions.
	child := llmkit.RunID("seeded-child-run")
	seedRun(t, db, child, parent, time.Now(), llmkit.RunCompleted)
	out, code := runAudit(t, auditBin, db, "Q5", "--run", runID)
	if code != 0 {
		t.Fatalf("audit Q5 of the stored run exited %d", code)
	}
	if !strings.Contains(out, "descendant") || !strings.Contains(out, string(child)) {
		t.Fatalf("audit Q5 of the stored run lacks its descendant %q:\n%s", child, out)
	}
	out, code = runAudit(t, auditBin, db, "Q5", "--run", string(child))
	if code != 0 {
		t.Fatalf("audit Q5 of the child exited %d", code)
	}
	if !strings.Contains(out, "ancestor") || !strings.Contains(out, runID) {
		t.Fatalf("audit Q5 of the child lacks its ancestor %q:\n%s", runID, out)
	}
}

func TestExamplesAuditSinceTOrdersByInstant(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skips the example builds")
	}
	root := moduleRoot(t)
	auditBin := buildExample(t, root, t.TempDir(), "audit")

	db := filepath.Join(t.TempDir(), "zones.db")
	st, err := sqlite.Open(db, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.Close(); err != nil {
		t.Fatal(err)
	}
	// A's local text sorts after T, its instant is before T; B's local text
	// sorts before T, its instant is after T. Only instant order may decide.
	aTextAfter := llmkit.RunID("run-a-plus14")
	bTextBefore := llmkit.RunID("run-b-minus12")
	sinceT := time.Date(2026, 10, 26, 8, 30, 0, 0, time.UTC)
	seedRun(t, db, aTextAfter, "", time.Date(2026, 10, 26, 22, 15, 0, 0, time.FixedZone("", 14*3600)), llmkit.RunFailed)
	seedRun(t, db, bTextBefore, "", time.Date(2026, 10, 26, 5, 30, 0, 0, time.FixedZone("", -12*3600)), llmkit.RunFailed)

	out, code := runAudit(t, auditBin, db, "Q7", "--since", sinceT.Format(time.RFC3339))
	if code != 0 {
		t.Fatalf("audit Q7 since T exited %d", code)
	}
	if !strings.Contains(out, string(bTextBefore)) {
		t.Fatalf("Q7 since T (an instant after T) must return run %q:\n%s", bTextBefore, out)
	}
	if strings.Contains(out, string(aTextAfter)) {
		t.Fatalf("Q7 since T (an instant after T) must not return run %q, whose instant is before T:\n%s", aTextAfter, out)
	}

	out, code = runAudit(t, auditBin, db, "Q7", "--since", "0")
	if code != 0 {
		t.Fatalf("audit Q7 since 0 exited %d", code)
	}
	if !strings.Contains(out, string(aTextAfter)) || !strings.Contains(out, string(bTextBefore)) {
		t.Fatalf("Q7 since 0 must return both runs:\n%s", out)
	}
}

func TestExamplesAuditCookbookMatchesStore(t *testing.T) {
	root := moduleRoot(t)
	storeSQL, err := os.ReadFile(filepath.Join(root, "store", "sqlite", "cookbook.sql"))
	if err != nil {
		t.Fatal(err)
	}
	auditSQL, err := os.ReadFile(filepath.Join(root, "examples", "audit", "cookbook.sql"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(storeSQL, auditSQL) {
		t.Fatal("examples/audit/cookbook.sql must stay a byte-identical copy of store/sqlite/cookbook.sql")
	}
}

// cookbookQueries splits store/sqlite/cookbook.sql into its named queries.
func cookbookQueries(t *testing.T) map[string]string {
	t.Helper()
	src, err := os.ReadFile(filepath.Join(moduleRoot(t), "store", "sqlite", "cookbook.sql"))
	if err != nil {
		t.Fatal(err)
	}
	header := regexp.MustCompile(`(?m)^-- (Q\d+):`)
	out := map[string]string{}
	locs := header.FindAllStringSubmatchIndex(string(src), -1)
	for i, loc := range locs {
		end := len(src)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		out[string(src[loc[2]:loc[3]])] = strings.TrimSpace(string(src[loc[0]:end]))
	}
	if len(out) != 7 {
		t.Fatalf("cookbook.sql holds %d queries, want Q1..Q7", len(out))
	}
	return out
}

func TestDocsStoreQuotesCookbook(t *testing.T) {
	docs, err := os.ReadFile(filepath.Join(moduleRoot(t), "docs", "store.md"))
	if err != nil {
		t.Fatal(err)
	}
	for name, query := range cookbookQueries(t) {
		if !strings.Contains(string(docs), query) {
			t.Fatalf("docs/store.md does not quote %s verbatim", name)
		}
	}
	if !strings.Contains(string(docs), "outside the documented view surface") {
		t.Fatal("docs/store.md must say Q6 reads the events table, outside the documented view surface")
	}
}

func TestInstructionTailsIdentical(t *testing.T) {
	root := moduleRoot(t)
	// The split anchors at a line-start "## Build", as the files' own check
	// does; the sync note mentions "## Build" mid-line and must not match.
	tail := func(name string) string {
		b, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		i := regexp.MustCompile(`(?m)^## Build`).FindStringIndex(string(b))
		if i == nil {
			t.Fatalf("%s has no \"## Build\" section", name)
		}
		return string(b)[i[0]:]
	}

	agents, claude := tail("AGENTS.md"), tail("CLAUDE.md")
	if agents != claude {
		t.Fatal("AGENTS.md and CLAUDE.md tails from \"## Build\" differ")
	}
	if !strings.Contains(agents, "go doc github.com/dpoage/llmkit/store/sqlite") {
		t.Fatal("AGENTS.md and CLAUDE.md lack the store/sqlite package-table row")
	}
}

func TestDocsStoreGoSnippetsArePackageExamples(t *testing.T) {
	root := moduleRoot(t)
	docs, err := os.ReadFile(filepath.Join(root, "docs", "store.md"))
	if err != nil {
		t.Fatal(err)
	}
	src, err := os.ReadFile(filepath.Join(root, "store", "sqlite", "example_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	fence := regexp.MustCompile("(?s)```go\n(.*?)```")
	blocks := fence.FindAllStringSubmatch(string(docs), -1)
	if len(blocks) == 0 {
		t.Fatal("docs/store.md quotes no Go snippets")
	}
	for _, b := range blocks {
		if !strings.Contains(string(src), b[1]) {
			t.Fatalf("docs/store.md Go snippet is not verbatim package-example source:\n%s", b[1])
		}
	}
}

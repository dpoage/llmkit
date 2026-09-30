//go:build integration

package sandbox

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// containersForToken lists the names of the runtime's containers (running or
// not) whose command line carries token — the run's own container, found
// without knowing the name Exec generated.
func containersForToken(t *testing.T, runtime, token string) []string {
	t.Helper()
	out, err := exec.Command(runtime, "ps", "-a", "--no-trunc", "--format", "{{.Names}} {{.Command}}").Output()
	if err != nil {
		t.Fatalf("%s ps: %v", runtime, err)
	}
	var names []string
	for _, line := range strings.Split(string(out), "\n") {
		if name, cmdline, ok := strings.Cut(line, " "); ok && strings.Contains(cmdline, token) {
			names = append(names, name)
		}
	}
	return names
}

// TestIntegrationCLIKillReturnsPromptlyAndRemovesContainer: for every way a
// CLI run is killed — Spec.Timeout, idle watchdog, caller deadline, caller
// cancel — Exec reports the kill, returns within 3s of the kill instant
// (podman's own `rm -f` would take its 10s stop timeout), and the run's
// container is gone. Five runs per cause; every duration is logged.
func TestIntegrationCLIKillReturnsPromptlyAndRemovesContainer(t *testing.T) {
	const window = time.Second
	plain := newTestCLI(t)
	idle := newTestCLI(t, WithIdleTimeout(window))

	type outcome struct {
		res       Result
		err       error
		sinceKill time.Duration
	}
	causes := []struct {
		name    string
		run     func(spec Spec) outcome
		wantErr error // caller's end: the ctx error the cancelled error wraps
	}{
		{"spec timeout", func(spec Spec) outcome {
			spec.Timeout = window
			start := time.Now()
			res, err := plain.Exec(context.Background(), spec)
			return outcome{res, err, time.Since(start) - res.PrepDuration - window}
		}, nil},
		{"idle watchdog", func(spec Spec) outcome {
			spec.Timeout = 50 * time.Second
			start := time.Now()
			res, err := idle.Exec(context.Background(), spec)
			// res.Duration ends when the supervised runtime process
			// returned, i.e. at the kill; everything after is the reap.
			return outcome{res, err, time.Since(start) - res.PrepDuration - res.Duration}
		}, nil},
		{"caller deadline", func(spec Spec) outcome {
			spec.Timeout = 50 * time.Second
			ctx, cancel := context.WithTimeout(context.Background(), window)
			defer cancel()
			start := time.Now()
			res, err := plain.Exec(ctx, spec)
			return outcome{res, err, time.Since(start) - window}
		}, context.DeadlineExceeded},
		{"caller cancel", func(spec Spec) outcome {
			spec.Timeout = 50 * time.Second
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var killedAt time.Time
			timer := time.AfterFunc(window, func() { killedAt = time.Now(); cancel() })
			defer timer.Stop()
			res, err := plain.Exec(ctx, spec)
			return outcome{res, err, time.Since(killedAt)}
		}, context.Canceled},
	}
	for _, c := range causes {
		t.Run(c.name, func(t *testing.T) {
			var ds []time.Duration
			for i := range 5 {
				token := "llmkit-reap-" + randToken()
				spec := Spec{RepoDir: t.TempDir(), Cmd: []string{"sh", "-c", "sleep 60", token}}
				o := c.run(spec)
				ds = append(ds, o.sinceKill)
				switch {
				case c.wantErr != nil && (!errors.Is(o.err, c.wantErr) || !strings.HasPrefix(o.err.Error(), "sandbox: execution cancelled")):
					t.Errorf("run %d: err = %v, want the cancelled error wrapping %v (res %+v)", i, o.err, c.wantErr, o.res)
				case c.wantErr == nil && (o.err != nil || !o.res.TimedOut || o.res.ExitCode != -1):
					t.Errorf("run %d: res = %+v err = %v, want TimedOut, ExitCode -1, nil error", i, o.res, o.err)
				}
				if o.sinceKill > 3*time.Second {
					t.Errorf("run %d: Exec returned %v after the kill, want <= 3s", i, o.sinceKill)
				}
				deadline := time.Now().Add(5 * time.Second)
				var left []string
				for {
					if left = containersForToken(t, plain.runtime, token); len(left) == 0 || time.Now().After(deadline) {
						break
					}
					time.Sleep(100 * time.Millisecond)
				}
				if len(left) != 0 {
					t.Errorf("run %d: containers still present 5s after return: %v", i, left)
					_ = exec.Command(plain.runtime, append([]string{"rm", "-f"}, left...)...).Run()
				}
			}
			t.Logf("%s: durations from the kill instant: %v", c.name, ds)
		})
	}
}

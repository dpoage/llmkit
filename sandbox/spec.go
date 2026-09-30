package sandbox

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Backend-name constants: the ONLY spellings of the four backend names
// in non-test code. Documented values of UnsupportedSpecError.Backend
// and llmkit.ExecEvent.Backend; no consumer reads them symbolically, so
// they stay unexported.
const (
	backendCLI   = "cli"
	backendBwrap = "bwrap"
	backendHost  = "host"
	backendMock  = "mock"
)

// validateSpec admits a Spec for the named backend. PURE — no filesystem
// access — and every Exec method calls it FIRST, so a refused Spec is
// rejected before any workspace preparation, write, or runtime contact.
//
// A Spec malformed for EVERY backend returns *InvalidSpecError (fix the
// Spec); a well-formed Spec the named backend cannot honor returns
// *UnsupportedSpecError (pick another backend or drop the field).
// Universal classes are checked first, then backend-specific ones, so
// the error for a Spec with several problems is deterministic.
//
// rendered is the complete list of extra mounts the calling backend renders
// beyond the Spec (CLI: its host-toolchain binds; Bwrap: its host-toolchain
// and POSIX-baseline binds; HostExec and Mock: none), so a Spec mount that
// would collide with one is refused before any run instead of failing
// inside the runtime (CLI) or silently hiding a toolchain (Bwrap). The
// destinations a backend or its runtime provide themselves
// (WorkspaceMount on every backend; cliProvidedDestinations on the CLI;
// bwrapProvidedDestinations and fixedROAllowlist on Bwrap) are refused to
// Spec mounts by name here.
//
// Workspace EXISTENCE is deliberately not checked here: it is a
// filesystem property, and the run supervisor checks it (still before any
// write) as InvalidSpecError{Field: "Workspace"}. The Mock never checks it.
func validateSpec(backend string, spec Spec, rendered []ROMount) error {
	// Universal-invalid classes (InvalidSpecError on every backend,
	// Mock included).
	if len(spec.Cmd) == 0 {
		return &InvalidSpecError{Field: "Cmd", Reason: "must be non-empty"}
	}
	if spec.RepoDir == "" && spec.Workspace == "" {
		return &InvalidSpecError{Field: "RepoDir", Reason: "one of RepoDir or Workspace must be set"}
	}
	if spec.Workspace != "" && !filepath.IsAbs(spec.Workspace) {
		return &InvalidSpecError{Field: "Workspace", Reason: fmt.Sprintf("%q must be an absolute path", spec.Workspace)}
	}
	for key := range spec.WriteFiles {
		if _, err := sanitizeRelPath(key); err != nil {
			return &InvalidSpecError{Field: "WriteFiles", Reason: fmt.Sprintf("key %q: %v", key, err)}
		}
	}
	for _, entry := range spec.CaptureFiles {
		if _, err := sanitizeRelPath(entry); err != nil {
			return &InvalidSpecError{Field: "CaptureFiles", Reason: fmt.Sprintf("entry %q: %v", entry, err)}
		}
	}
	if err := validateMounts(spec.ROMounts, spec.RWMounts); err != nil {
		return err
	}
	for _, e := range spec.Env {
		if key, _, found := strings.Cut(e, "="); !found || key == "" {
			// A KEY without "=" is a host-env leak on the CLI backend
			// (`podman --env NAME` inherits NAME from the HOST) and was
			// silently dropped by the other renderers; an empty KEY
			// ("=VALUE") fails inside the run on the container backends
			// (podman exits 125, bwrap's setenv fails with exit 1). Refuse
			// both everywhere instead.
			return &InvalidSpecError{Field: "Env", Reason: fmt.Sprintf("entry %q must be KEY=VALUE with a non-empty KEY", e)}
		}
	}

	// Backend-specific classes (UnsupportedSpecError; the Mock records
	// every well-formed field, so it has none).
	switch backend {
	case backendCLI:
		if _, err := resolveNetworkMode(backend, NetworkNone, spec.Network, cliNetworks...); err != nil {
			return err
		}
		if spec.Workspace != "" && !cliPathExpressible(spec.Workspace) {
			return &UnsupportedSpecError{Backend: backend, Field: "Workspace", Value: spec.Workspace}
		}
		if err := validateCLIMounts(spec.ROMounts, spec.RWMounts, rendered); err != nil {
			return err
		}
	case backendBwrap:
		if spec.Image != "" {
			return &UnsupportedSpecError{Backend: backend, Field: "Image", Value: spec.Image}
		}
		if _, err := resolveNetworkMode(backend, NetworkNone, spec.Network, bwrapNetworks...); err != nil {
			return err
		}
		if err := validateBwrapMounts(spec.ROMounts, spec.RWMounts); err != nil {
			return err
		}
		if err := validateBwrapReserved(spec.ROMounts, spec.RWMounts, rendered); err != nil {
			return err
		}
	case backendHost:
		if spec.Image != "" {
			return &UnsupportedSpecError{Backend: backend, Field: "Image", Value: spec.Image}
		}
		if len(spec.ROMounts) > 0 {
			return &UnsupportedSpecError{Backend: backend, Field: "ROMounts", Value: fmt.Sprintf("%d mount(s)", len(spec.ROMounts))}
		}
		if len(spec.RWMounts) > 0 {
			return &UnsupportedSpecError{Backend: backend, Field: "RWMounts", Value: fmt.Sprintf("%d mount(s)", len(spec.RWMounts))}
		}
		if len(spec.SetupCmds) > 0 {
			return &UnsupportedSpecError{Backend: backend, Field: "SetupCmds", Value: fmt.Sprintf("%d command(s)", len(spec.SetupCmds))}
		}
		if _, err := resolveNetworkMode(backend, NetworkHost, spec.Network, NetworkHost); err != nil {
			return err
		}
	case backendMock:
		// The Mock records the Spec verbatim; only the universal classes
		// apply.
	}
	return nil
}

// cliProvidedDestinations are the container paths the CLI backend or its
// runtime mount themselves besides WorkspaceMount — the --tmpfs /tmp
// scratch plus the runtime's own mounts, measured on podman/crun as the
// union over network modes. A Spec mount whose cleaned ContainerPath
// equals one is refused.
var cliProvidedDestinations = []string{
	"/tmp", "/run", "/var/tmp", "/run/.containerenv",
	"/etc/hostname", "/etc/hosts", "/etc/resolv.conf",
	"/dev", "/dev/shm", "/dev/pts", "/dev/mqueue",
	"/dev/null", "/dev/zero", "/dev/full", "/dev/tty", "/dev/random", "/dev/urandom",
	"/proc", "/sys", "/sys/fs/cgroup",
	"/proc/acpi", "/proc/scsi", "/proc/kcore", "/proc/keys", "/proc/timer_list",
	"/proc/interrupts", "/proc/asound", "/proc/bus", "/proc/fs", "/proc/irq",
	"/proc/sys", "/proc/sysrq-trigger",
	"/sys/devices/virtual/powercap", "/sys/firmware",
}

// bwrapProvidedDestinations are the container paths the Bwrap backend
// provides itself besides WorkspaceMount and fixedROAllowlist: --proc,
// --dev and the device nodes it creates, the --tmpfs /tmp scratch, and
// /etc/resolv.conf under host networking. A Spec mount whose cleaned
// ContainerPath equals one is refused.
var bwrapProvidedDestinations = []string{
	"/proc", "/dev",
	"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty", "/dev/pts",
	"/tmp", "/etc/resolv.conf",
}

// cliPathExpressible reports whether p can be written in the CLI runtime's
// `-v host:ctr:opts` syntax, where ':' separates the fields and has no
// escape.
func cliPathExpressible(p string) bool {
	return !strings.Contains(p, ":")
}

// validateCLIMounts is the CLI-only mount class of validateSpec: both paths
// of a Spec mount must be expressible in the runtime's -v syntax
// (cliPathExpressible), and its cleaned ContainerPath may not equal a
// destination the CLI or its runtime provide (cliProvidedDestinations) or a
// rendered mount's ContainerPath. Each is a limit of THIS backend, so it is
// an *UnsupportedSpecError naming the list holding the mount.
func validateCLIMounts(ro, rw, rendered []ROMount) error {
	taken := reservedDestinations(cliProvidedDestinations, rendered)
	return firstUnsupportedMount(backendCLI, ro, rw, func(m ROMount) (string, bool) {
		for _, p := range []string{m.HostPath, m.ContainerPath} {
			if !cliPathExpressible(p) {
				return p, true
			}
		}
		return m.ContainerPath, taken[filepath.Clean(m.ContainerPath)]
	})
}

// validateBwrapReserved refuses a Spec mount whose cleaned ContainerPath
// equals a destination Bwrap provides itself (bwrapProvidedDestinations) or
// a rendered mount's ContainerPath (a later bind would silently hide the
// toolchain or the provided mount). The collision exists only for this
// backend's configuration, so it is an *UnsupportedSpecError naming the
// list holding the Spec mount.
func validateBwrapReserved(ro, rw, rendered []ROMount) error {
	taken := reservedDestinations(bwrapProvidedDestinations, rendered)
	return firstUnsupportedMount(backendBwrap, ro, rw, func(m ROMount) (string, bool) {
		return m.ContainerPath, taken[filepath.Clean(m.ContainerPath)]
	})
}

// reservedDestinations is the set of cleaned container paths a Spec mount
// may not target: provided plus every rendered mount's ContainerPath.
func reservedDestinations(provided []string, rendered []ROMount) map[string]bool {
	taken := make(map[string]bool, len(provided)+len(rendered))
	for _, p := range provided {
		taken[p] = true
	}
	for _, m := range rendered {
		taken[filepath.Clean(m.ContainerPath)] = true
	}
	return taken
}

// firstUnsupportedMount returns an *UnsupportedSpecError for backend naming
// the first Spec mount (ROMounts before RWMounts) refuse rejects, with the
// offending value refuse returns.
func firstUnsupportedMount(backend string, ro, rw []ROMount, refuse func(ROMount) (string, bool)) error {
	for _, list := range []struct {
		field  string
		mounts []ROMount
	}{{"ROMounts", ro}, {"RWMounts", rw}} {
		for _, m := range list.mounts {
			if value, bad := refuse(m); bad {
				return &UnsupportedSpecError{Backend: backend, Field: list.field, Value: value}
			}
		}
	}
	return nil
}

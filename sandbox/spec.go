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
// Workspace EXISTENCE is deliberately not checked here: it is a
// filesystem property, and the run supervisor checks it (still before any
// write) as InvalidSpecError{Field: "Workspace"}. The Mock never checks it.
func validateSpec(backend string, spec Spec) error {
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

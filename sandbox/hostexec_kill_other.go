//go:build !linux

package sandbox

import "os/exec"

// hostKiller is the non-Linux HostExec kill: exec's own context kill of the
// direct child. There is no portable, reuse-safe way to walk descendants.
type hostKiller struct{}

func newHostKiller() *hostKiller { return &hostKiller{} }

func (k *hostKiller) install(*exec.Cmd)          {}
func (k *hostKiller) afterStart(*exec.Cmd) error { return nil }
func (k *hostKiller) close()                     {}

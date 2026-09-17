package adapter

import (
	"fmt"

	"github.com/dpoage/llmkit"
)

// ValidateMediaBlock enforces the source rule for image and document
// blocks: exactly one of Data and URL must be set, and inline Data requires
// a MediaType (every provider wire format carries the MIME type next to the
// bytes). Both sources set — or neither — would otherwise force every
// adapter to silently pick a source, so the rule is checked here — once,
// for all three adapters — and each adapter calls it BEFORE any wire call,
// returning the error verbatim.
//
// provider names the adapter for the error message only. The returned
// error wraps llmkit.ErrInvalidRequest.
func ValidateMediaBlock(provider string, b llmkit.Block) error {
	switch {
	case len(b.Data) > 0 && b.URL != "":
		return llmkit.NewAPIError(provider, 0, 0, llmkit.ErrInvalidRequest,
			fmt.Sprintf("%s block: Data and URL are mutually exclusive; set exactly one", b.Kind), nil)
	case len(b.Data) == 0 && b.URL == "":
		return llmkit.NewAPIError(provider, 0, 0, llmkit.ErrInvalidRequest,
			fmt.Sprintf("%s block: exactly one of Data and URL must be set", b.Kind), nil)
	case len(b.Data) > 0 && b.MediaType == "":
		return llmkit.NewAPIError(provider, 0, 0, llmkit.ErrInvalidRequest,
			fmt.Sprintf("%s block: MediaType must be set when Data is inline", b.Kind), nil)
	}
	return nil
}

package adapter

import (
	"fmt"

	"github.com/dpoage/llmkit"
)

// roleBlockKinds pins which content block kinds each message role may
// carry. The rule is uniform across all three adapters and enforced before
// any wire call: RoleUser carries text/image/document; RoleAssistant carries
// text/thinking (a thinking block is re-emitted only when its Provider
// matches the adapter); RoleSystem and RoleToolResult carry text only.
var roleBlockKinds = map[llmkit.Role]map[llmkit.BlockKind]bool{
	llmkit.RoleUser:       {llmkit.BlockText: true, llmkit.BlockImage: true, llmkit.BlockDocument: true},
	llmkit.RoleAssistant:  {llmkit.BlockText: true, llmkit.BlockThinking: true},
	llmkit.RoleSystem:     {llmkit.BlockText: true},
	llmkit.RoleToolResult: {llmkit.BlockText: true},
}

// ValidateMessageBlocks enforces the per-role block-kind rule and the media
// source rule for every message BEFORE the adapter touches its wire format:
// a block kind outside the role's set, or an image/document failing
// ValidateMediaBlock, is an error wrapping llmkit.ErrInvalidRequest. Without
// this gate, invalid blocks would be silently dropped (m.Text()) or reach
// the provider and fail there — so every adapter calls it in its
// message-conversion entry point and returns the error verbatim.
//
// provider names the adapter for the error message only.
func ValidateMessageBlocks(provider string, m llmkit.Message) error {
	allowed := roleBlockKinds[m.Role]
	for _, b := range m.Content {
		if !allowed[b.Kind] {
			return &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 0, RetryAfter: 0, Provider: provider, Message: fmt.Sprintf("%s block not allowed in a %s message", b.Kind, m.Role), Err: nil}
		}
		if b.Kind == llmkit.BlockImage || b.Kind == llmkit.BlockDocument {
			if err := ValidateMediaBlock(provider, b); err != nil {
				return err
			}
		}
	}
	return nil
}

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
		return &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 0, RetryAfter: 0, Provider: provider, Message: fmt.Sprintf("%s block: Data and URL are mutually exclusive; set exactly one", b.Kind), Err: nil}
	case len(b.Data) == 0 && b.URL == "":
		return &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 0, RetryAfter: 0, Provider: provider, Message: fmt.Sprintf("%s block: exactly one of Data and URL must be set", b.Kind), Err: nil}
	case len(b.Data) > 0 && b.MediaType == "":
		return &llmkit.APIError{Kind: llmkit.ErrInvalidRequest, StatusCode: 0, RetryAfter: 0, Provider: provider, Message: fmt.Sprintf("%s block: MediaType must be set when Data is inline", b.Kind), Err: nil}
	}
	return nil
}

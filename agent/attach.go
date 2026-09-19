package agent

import (
	"github.com/dpoage/llmkit"
)

// Attach adds blocks to the task turn of a run. The user turn becomes
// [llmkit.UserMessage]([llmkit.Text](task), blocks...): the task text
// first, then the blocks in the order given. Use Attach to send an image or
// a document with the task. Attach applies to [Runner.Run],
// [Runner.RunJSON], and [Runner.RunJSONAs].
//
// Repeated Attach options accumulate in order. With an empty task the turn
// carries the blocks only; no empty text block is sent. RunJSON is the
// exception: its text block always exists because it carries the JSON
// instruction. With no blocks the turn keeps the plain
// [llmkit.TextMessage] shape, so runs without attachments are unchanged.
// A run with an empty task and attachments has no text to derive the
// transcript filename from; the autosave uses the name "run".
//
// Attachments appear on the task turn only. The empty-turn nudge, the
// max-tokens continuation, the forced finalization turn, and the repair
// turn stay text-only. With [Continue], the attached turn follows the seed.
//
// Attach does not validate block kinds. The adapters enforce which block
// kinds a user turn may carry and return an error wrapping
// [llmkit.ErrInvalidRequest] before any wire call.
// [llmkit.Capabilities].Images and Documents are advisory: no adapter reads
// them, so check them before you attach.
func Attach(blocks ...llmkit.Block) RunOption {
	return func(c *runConfig) {
		c.attach = append(c.attach, blocks...)
	}
}

// taskTurn builds the seeded user turn from task and attachments — the single
// place [Attach]'s shape rules live. With no attachments it returns
// [llmkit.TextMessage]([llmkit.RoleUser], task) byte-for-byte. With an empty
// task and non-empty attachments the Text block is omitted so
// [llmkit.UserMessage] still sees at least one block.
func taskTurn(task string, attach []llmkit.Block) llmkit.Message {
	if len(attach) == 0 {
		return llmkit.TextMessage(llmkit.RoleUser, task)
	}
	blocks := make([]llmkit.Block, 0, len(attach)+1)
	if task != "" {
		blocks = append(blocks, llmkit.Text(task))
	}
	return llmkit.UserMessage(append(blocks, attach...)...)
}

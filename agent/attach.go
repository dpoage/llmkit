package agent

import (
	"github.com/dpoage/llmkit"
)

// Attach appends blocks to the run's seeded task turn: instead of the plain
// [llmkit.TextMessage]([llmkit.RoleUser], task), the user turn becomes
// [llmkit.UserMessage]([llmkit.Text](task), blocks...) — the form for a
// prompt that carries an image or document alongside the task text (a
// diagram to read, a PDF to quote). Applies to [Runner.Run],
// [Runner.RunJSON], and [Runner.RunJSONAs].
//
// Multiple Attach options accumulate in the order given: Attach(a), Attach(b)
// yields a then b after the task text. The task text stays the FIRST block;
// an empty task with non-empty blocks omits the Text block entirely (the
// turn carries the blocks only — no empty text block, which adapters may
// refuse); an empty task AND no blocks keeps the exact plain-Run shape
// ([llmkit.TextMessage]), byte-for-byte, so no-attachment callers and their
// transcript fixtures are unaffected. On RunJSON the jsonInstruction suffix
// is appended to the TEXT block only; the attached blocks ride alongside
// untouched.
//
// Attachments ride on the task turn ONLY: they never appear on the
// empty-turn or max-tokens nudges, the forced finalization turn, or the
// repair completion — none of those re-asks the task, so they stay
// text-only user turns exactly as before. With [Continue] they compose: the
// attached turn is appended after the seed, like any continued task turn.
//
// Attach performs no block-kind validation. Which role may carry which kind
// is the adapters' per-role rule, enforced pre-wire: a block passed here
// that a user turn cannot carry simply fails there with an error wrapping
// [llmkit.ErrInvalidRequest] — do not expect the harness to duplicate that
// rule. [llmkit.Capabilities].Images/Documents are advisory (information for
// callers; no adapter reads them), so a caller consults them itself before
// attaching, exactly as when building [llmkit.Message] values by hand.
func Attach(blocks ...llmkit.Block) RunOption {
	return func(c *runConfig) {
		c.attach = append(c.attach, blocks...)
	}
}

// taskTurn builds the seeded user turn for a run from the task text and the
// caller's attachments — the single place [Attach]'s shape rules live. With
// no attachments it returns exactly
// [llmkit.TextMessage]([llmkit.RoleUser], task): the turn every pre-Attach
// run sent, byte-for-byte. With an empty task the Text block is omitted;
// [llmkit.UserMessage] still sees at least one block, so it cannot panic.
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

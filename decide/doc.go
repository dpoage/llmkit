// Package decide is the vocabulary for decisions: given a state and a set
// of typed questions, a [Decider] returns a [Response] of per-question
// answers. The package holds no HTTP code and imports no backend; package
// decide/systemone is the System One wire backend.
//
// A decision model is not a chat model. It has no messages, tools, or
// streaming, so a [Decider] is not an [llmkit.Client].
//
// # Questions
//
// [Question] is a sealed interface with three implementations:
// [Noul] (binary belief in [0, 1]), [Choice] (pick one option from a
// map of descriptions), and [Score] (rate against an ordered legend of
// at least two levels). [Questions] indexes one Ask's questions by id.
//
// State, instructions, and descriptions accept Go strings and anything
// that marshals to a JSON object or array. Descriptions and criteria
// also accept null. JSON numbers and booleans are refused.
//
// [Normalize] validates a state and questions and returns a [Request]: the
// state and questions in the root vocabulary ([llmkit.DecisionQuestion]),
// sorted by id, and the images found in them. A refusal is an
// [*llmkit.APIError] with Kind [llmkit.ErrInvalidRequest].
//
// # Images
//
// A state, an instructions value, or a description may hold an image: a
// JSON object that decodes as an [llmkit.Block] of kind "image" using only
// Block's member names, however built. [Normalize] replaces each image with
// a label and returns it once in [Request.Images].
//
// A backend declares what it carries as an [ImageSupport] through
// [ImageSupporter]. The zero value refuses every image, and an image the
// support does not carry is refused with [ErrImageUnsupported].
//
// # Answers
//
// [Response] splits answers by question kind. [Response.Nouls] maps a
// noul id to its 0..1 belief. [Response.Choices] maps a choice id to a
// [ChoiceAnswer]. [Response.Scores] maps a score id to a [ScoreAnswer]
// that carries the score, the reported Levels, and per-level Probabilities.
// Maps stay nil for kinds the Ask did not include.
//
// Read answers through the accessors [Response.Noul], [Response.Choice],
// and [Response.Score]. Each returns an error wrapping [ErrNotAsked] when
// the id was not asked as that kind, so a typo is an error, not a zero.
//
// A backend that reports no distribution leaves Probabilities nil and
// Confidence 0. A backend never fabricates either. [Check] reports whether
// a Response answers a [Questions] set exactly.
//
// # Deciders
//
// [Decider] is the interface a backend implements. Its errors wrap a root
// llmkit sentinel (llmkit.Err*) or chain the context's error. A successful
// Response satisfies [Check] against the questions asked. Besides the
// wrapped Decider's errors and a [Normalize] refusal, a Decider that decide
// provides returns only errors wrapping [ErrReplayDiverged] (distinct from
// agent.ErrReplayDiverged) or [ErrReplayedFailure].
//
// # Observability and replay
//
// [Observe] wraps any Decider and emits one [llmkit.KindDecision] event per
// Ask to an [llmkit.Observer], on the Ask's context, so inside an agent
// Runner turn the event carries the run's RunID and the turn's Step. A
// Decider that implements Identity() llmkit.Identity (the systemone Client
// does) tags the event's Backend and Model, and the wrapper forwards the
// method. An Ask that fails [Normalize] emits nothing. The event records
// each image as a reference, hash, media type and placements, not bytes.
//
// [Replay] is the other half: it builds a Decider from the Decision events
// an [llmkit.Source] holds for a run. Each Ask consumes the earliest
// recorded decision with the same Step, the same question ids and the same
// set of image hashes and returns the recorded answers, with no backend
// call. Paired with the agent package's replay client, a recorded run whose
// ToolPolicy asked a Decider replays offline with the same denials. Replay
// does not compare the state beyond its images and does not report recorded
// decisions the replay never asked for; see [Replay].
package decide
